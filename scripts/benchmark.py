"""Benchmark for DCNv4 / FlashDeformAttn CUDA vs Triton implementations.

Purpose
-------
This script measures performance (latency and VRAM) of CUDA vs Triton implementations
across various configurations. Timing is done with triton.testing.do_bench to reduce
host overhead and flush L2 cache between runs.

Key metrics
-----------
- Forward pass latency (ms)
- Backward pass latency (ms)
- Peak VRAM usage (MB)
- Speed ratio (Triton / CUDA)

Usage examples
--------------
  # Quick benchmark (small sizes, short timings)
  python benchmark.py --quick

  # Full benchmark with longer timing windows (ms)
  python benchmark.py --warmup-ms 25 --rep-ms 100

  # Legacy flag names still work (ms)
  python benchmark.py --warmup 25 --iterations 100

  # Benchmark only DCNv4
  python benchmark.py --suite dcnv4

  # Save JSON report
  python benchmark.py --report benchmark_results.json

  # Custom batch sizes
  python benchmark.py --batch-sizes 1,2,4,8,16

Notes
-----
- --warmup/--iterations are time windows in milliseconds (aliases: --warmup-ms/--rep-ms).

Exit codes
----------
Always 0 (benchmark does not fail on slow performance).

"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch
import triton.testing as triton_testing

from dcnv4 import triton_ops

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dcnv4 import _ops

    ops: _ops = torch.ops  # type: ignore[assignment]
else:
    ops = torch.ops

# ------------------------------
# Utilities
# ------------------------------


def _ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        msg = "CUDA is required for this benchmark."
        raise RuntimeError(msg)


def _clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _bench_ms(
    fn: Callable[[], Any],
    warmup_ms: int,
    rep_ms: int,
    stat: str,
) -> float:
    """Benchmark fn using triton.testing.do_bench and return ms."""
    return triton_testing.do_bench(
        fn,
        warmup=warmup_ms,
        rep=rep_ms,
        return_mode=stat,
    )


def _measure_peak_memory_mb(fn: Callable[[], Any]) -> float:
    """Run fn once and return peak GPU memory allocated in MB."""
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


@dataclass
class BenchmarkResult:
    name: str
    batch_size: int
    backend: str
    fwd_ms: float
    bwd_ms: float
    total_ms: float
    vram_mb: float


@dataclass
class ComparisonResult:
    name: str
    batch_size: int
    cuda_fwd_ms: float
    cuda_bwd_ms: float
    cuda_total_ms: float
    cuda_vram_mb: float
    triton_fwd_ms: float
    triton_bwd_ms: float
    triton_total_ms: float
    triton_vram_mb: float
    fwd_ratio: float
    bwd_ratio: float
    total_ratio: float
    vram_ratio: float


# ------------------------------
# DCNv4 Benchmark
# ------------------------------


@dataclass(frozen=True)
class DCNv4Config:
    H: int
    W: int
    C: int
    group: int
    kernel_h: int = 3
    kernel_w: int = 3
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 1
    pad_w: int = 1
    dil_h: int = 1
    dil_w: int = 1
    offset_scale: float = 1.0


def _dcnv4_offset_dim(group: int, kernel_h: int, kernel_w: int) -> int:
    k_total = kernel_h * kernel_w
    valid = group * k_total * 3
    return int(math.ceil(valid / 8) * 8)


def _make_dcnv4_ops(
    config: DCNv4Config,
    batch_size: int,
    backend: str,
) -> tuple[Callable[[], Any], Callable[[], Any]]:
    group_channels = config.C // config.group
    offset_dim = _dcnv4_offset_dim(config.group, config.kernel_h, config.kernel_w)

    value = torch.randn(
        batch_size,
        config.H,
        config.W,
        config.C,
        device="cuda",
        dtype=torch.float32,
    )
    offset = (
        torch.randn(
            batch_size,
            config.H,
            config.W,
            offset_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.1
    )
    grad_output = torch.randn(
        batch_size,
        config.H,
        config.W,
        config.C,
        device="cuda",
        dtype=torch.float32,
    )

    if backend == "CUDA":

        def fwd():
            return ops.dcnv4_C.dcnv4_forward(
                value,
                offset,
                config.kernel_h,
                config.kernel_w,
                config.stride_h,
                config.stride_w,
                config.pad_h,
                config.pad_w,
                config.dil_h,
                config.dil_w,
                config.group,
                group_channels,
                config.offset_scale,
                256,
                0,
                8,
                128,
                False,
            )

        def bwd():
            return ops.dcnv4_C.dcnv4_backward(
                value,
                offset,
                config.kernel_h,
                config.kernel_w,
                config.stride_h,
                config.stride_w,
                config.pad_h,
                config.pad_w,
                config.dil_h,
                config.dil_w,
                config.group,
                group_channels,
                config.offset_scale,
                256,
                grad_output,
                0,
                2,
                128,
                False,
            )
    else:

        def fwd():
            return triton_ops.dcnv4_forward(
                value,
                offset,
                config.kernel_h,
                config.kernel_w,
                config.stride_h,
                config.stride_w,
                config.pad_h,
                config.pad_w,
                config.dil_h,
                config.dil_w,
                config.group,
                group_channels,
                config.offset_scale,
                0,
                False,
            )

        def bwd():
            return triton_ops.dcnv4_backward(
                value,
                offset,
                grad_output,
                config.kernel_h,
                config.kernel_w,
                config.stride_h,
                config.stride_w,
                config.pad_h,
                config.pad_w,
                config.dil_h,
                config.dil_w,
                config.group,
                group_channels,
                config.offset_scale,
                0,
                False,
            )

    return fwd, bwd


def benchmark_dcnv4_single(
    config: DCNv4Config,
    batch_size: int,
    backend: str,
    *,
    warmup_ms: int,
    rep_ms: int,
    stat: str,
    profile: bool = False,
) -> BenchmarkResult:
    """Benchmark a single DCNv4 configuration."""
    _clear_cuda_cache()
    fwd, bwd = _make_dcnv4_ops(config, batch_size, backend)
    # One-shot warmup to trigger JIT compilation and cache setup.
    fwd()
    bwd()
    torch.cuda.synchronize()

    _clear_cuda_cache()
    fwd, bwd = _make_dcnv4_ops(config, batch_size, backend)

    name = f"DCNv4 H{config.H}xW{config.W} C{config.C} g{config.group}"

    if profile:
        import triton.profiler as proton

        safe_name = name.replace(" ", "_")
        hook = "triton" if backend == "Triton" else None
        proton.start(name=f"{safe_name}_{backend}", hook=hook)

    fwd_ms = _bench_ms(fwd, warmup_ms, rep_ms, stat)
    bwd_ms = _bench_ms(bwd, warmup_ms, rep_ms, stat)

    if profile:
        import triton.profiler as proton

        proton.finalize()
        print(f"  [Profiler] Saved profile for {name} ({backend})")

    _clear_cuda_cache()
    fwd, bwd = _make_dcnv4_ops(config, batch_size, backend)
    vram_mb = _measure_peak_memory_mb(lambda: (fwd(), bwd()))

    return BenchmarkResult(
        name=name,
        batch_size=batch_size,
        backend=backend,
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        total_ms=fwd_ms + bwd_ms,
        vram_mb=vram_mb,
    )


# ------------------------------
# FlashDeformAttn Benchmark
# ------------------------------


@dataclass(frozen=True)
class FlashConfig:
    n_heads: int
    d_per_head: int
    spatial_shapes: tuple[tuple[int, int], ...]
    n_points: int


def _flash_level_start_index(spatial_shapes: Sequence[tuple[int, int]]) -> torch.Tensor:
    start = [0]
    for h, w in spatial_shapes[:-1]:
        start.append(start[-1] + h * w)
    return torch.tensor(start, device="cuda", dtype=torch.long)


def _make_flash_ops(
    config: FlashConfig,
    batch_size: int,
    backend: str,
) -> tuple[Callable[[], Any], Callable[[], Any]]:
    total_len = sum(h * w for h, w in config.spatial_shapes)
    n_levels = len(config.spatial_shapes)
    k_total = n_levels * config.n_points

    spatial_shapes_t = torch.tensor(
        config.spatial_shapes,
        device="cuda",
        dtype=torch.long,
    )
    level_start_index = _flash_level_start_index(config.spatial_shapes)

    value = torch.randn(
        batch_size,
        total_len,
        config.n_heads,
        config.d_per_head,
        device="cuda",
        dtype=torch.float16,
    )
    sampling_loc_attn = torch.rand(
        batch_size,
        total_len,
        config.n_heads,
        k_total * 3,
        device="cuda",
        dtype=torch.float16,
    )
    grad_output_3d = torch.randn(
        batch_size,
        total_len,
        config.n_heads * config.d_per_head,
        device="cuda",
        dtype=torch.float16,
    )
    grad_output_4d = grad_output_3d.view(
        batch_size,
        total_len,
        config.n_heads,
        config.d_per_head,
    ).contiguous()

    if backend == "CUDA":

        def fwd():
            return ops.dcnv4_C.flash_deform_attn_forward(
                value,
                spatial_shapes_t,
                level_start_index,
                sampling_loc_attn,
                64,
                config.n_points,
                8,
                128,
            )

        def bwd():
            return ops.dcnv4_C.flash_deform_attn_backward(
                value,
                spatial_shapes_t,
                level_start_index,
                sampling_loc_attn,
                grad_output_3d.contiguous(),
                64,
                config.n_points,
                8,
                128,
            )
    else:

        def fwd():
            return triton_ops.flash_deform_attn_forward(
                value,
                spatial_shapes_t,
                level_start_index,
                sampling_loc_attn,
                config.n_points,
            )

        def bwd():
            return triton_ops.flash_deform_attn_backward(
                value,
                spatial_shapes_t,
                level_start_index,
                sampling_loc_attn,
                grad_output_4d,
                config.n_points,
            )

    return fwd, bwd


def benchmark_flash_single(
    config: FlashConfig,
    batch_size: int,
    backend: str,
    *,
    warmup_ms: int,
    rep_ms: int,
    stat: str,
    profile: bool = False,
) -> BenchmarkResult:
    """Benchmark a single FlashDeformAttn configuration."""
    _clear_cuda_cache()
    fwd, bwd = _make_flash_ops(config, batch_size, backend)
    # One-shot warmup to trigger JIT compilation and cache setup.
    fwd()
    bwd()
    torch.cuda.synchronize()

    _clear_cuda_cache()
    fwd, bwd = _make_flash_ops(config, batch_size, backend)

    shapes_str = "x".join(f"{h}x{w}" for h, w in config.spatial_shapes[:2]) + "..."
    name = f"Flash h{config.n_heads} d{config.d_per_head} [{shapes_str}]"

    if profile:
        import triton.profiler as proton

        safe_name = (
            name.replace(" ", "_").replace("[", "").replace("]", "").replace("...", "")
        )
        hook = "triton" if backend == "Triton" else None
        proton.start(name=f"{safe_name}_{backend}", hook=hook)

    fwd_ms = _bench_ms(fwd, warmup_ms, rep_ms, stat)
    bwd_ms = _bench_ms(bwd, warmup_ms, rep_ms, stat)

    if profile:
        import triton.profiler as proton

        proton.finalize()
        print(f"  [Profiler] Saved profile for {name} ({backend})")

    _clear_cuda_cache()
    fwd, bwd = _make_flash_ops(config, batch_size, backend)
    vram_mb = _measure_peak_memory_mb(lambda: (fwd(), bwd()))

    return BenchmarkResult(
        name=name,
        batch_size=batch_size,
        backend=backend,
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        total_ms=fwd_ms + bwd_ms,
        vram_mb=vram_mb,
    )


# ------------------------------
# Comparison Logic
# ------------------------------


def compare_results(cuda: BenchmarkResult, triton: BenchmarkResult) -> ComparisonResult:
    """Compare CUDA and Triton benchmark results."""
    return ComparisonResult(
        name=cuda.name,
        batch_size=cuda.batch_size,
        cuda_fwd_ms=cuda.fwd_ms,
        cuda_bwd_ms=cuda.bwd_ms,
        cuda_total_ms=cuda.total_ms,
        cuda_vram_mb=cuda.vram_mb,
        triton_fwd_ms=triton.fwd_ms,
        triton_bwd_ms=triton.bwd_ms,
        triton_total_ms=triton.total_ms,
        triton_vram_mb=triton.vram_mb,
        fwd_ratio=triton.fwd_ms / cuda.fwd_ms if cuda.fwd_ms > 0 else float("inf"),
        bwd_ratio=triton.bwd_ms / cuda.bwd_ms if cuda.bwd_ms > 0 else float("inf"),
        total_ratio=triton.total_ms / cuda.total_ms
        if cuda.total_ms > 0
        else float("inf"),
        vram_ratio=triton.vram_mb / cuda.vram_mb if cuda.vram_mb > 0 else float("inf"),
    )


# ------------------------------
# Suites
# ------------------------------


def dcnv4_configs(quick: bool) -> list[DCNv4Config]:
    """Get DCNv4 benchmark configurations."""
    if quick:
        return [
            DCNv4Config(H=32, W=32, C=64, group=4),
            DCNv4Config(H=64, W=64, C=128, group=8),
        ]
    return [
        DCNv4Config(H=32, W=32, C=64, group=4),
        DCNv4Config(H=64, W=64, C=128, group=8),
        DCNv4Config(H=64, W=64, C=256, group=8),
        DCNv4Config(H=128, W=128, C=256, group=8),
    ]


def flash_configs(quick: bool) -> list[FlashConfig]:
    """Get FlashDeformAttn benchmark configurations."""
    if quick:
        return [
            FlashConfig(
                n_heads=8,
                d_per_head=32,
                spatial_shapes=((32, 32), (16, 16), (8, 8), (4, 4)),
                n_points=4,
            ),
        ]
    return [
        FlashConfig(
            n_heads=8,
            d_per_head=32,
            spatial_shapes=((32, 32), (16, 16), (8, 8), (4, 4)),
            n_points=4,
        ),
        FlashConfig(
            n_heads=8,
            d_per_head=32,
            spatial_shapes=((64, 64), (32, 32), (16, 16), (8, 8)),
            n_points=4,
        ),
        FlashConfig(
            n_heads=8,
            d_per_head=64,
            spatial_shapes=((64, 64), (32, 32), (16, 16), (8, 8)),
            n_points=4,
        ),
    ]


# ------------------------------
# Printing
# ------------------------------


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_table_header() -> None:
    print(
        f"{'Batch':<6} {'Backend':<8} {'Fwd (ms)':<10} {'Bwd (ms)':<10} "
        f"{'Total (ms)':<12} {'VRAM (MB)':<10}",
    )
    print("-" * 90)


def print_result(r: BenchmarkResult) -> None:
    print(
        f"{r.batch_size:<6} {r.backend:<8} {r.fwd_ms:<10.3f} {r.bwd_ms:<10.3f} "
        f"{r.total_ms:<12.3f} {r.vram_mb:<10.1f}",
    )


def print_comparison_header() -> None:
    print(
        f"{'Batch':<6} {'Fwd Ratio':<12} {'Bwd Ratio':<12} "
        f"{'Total Ratio':<14} {'VRAM Ratio':<12}",
    )
    print("-" * 70)


def print_comparison(c: ComparisonResult) -> None:
    print(
        f"{c.batch_size:<6} {c.fwd_ratio:<12.2f}x {c.bwd_ratio:<12.2f}x "
        f"{c.total_ratio:<14.2f}x {c.vram_ratio:<12.2f}x",
    )


def print_summary(comparisons: list[ComparisonResult], name: str) -> None:
    if not comparisons:
        return

    print(f"\n{name} Summary (Triton / CUDA ratio):")
    print_comparison_header()
    for c in comparisons:
        print_comparison(c)

    # Averages
    avg_fwd = sum(c.fwd_ratio for c in comparisons) / len(comparisons)
    avg_bwd = sum(c.bwd_ratio for c in comparisons) / len(comparisons)
    avg_total = sum(c.total_ratio for c in comparisons) / len(comparisons)
    avg_vram = sum(c.vram_ratio for c in comparisons) / len(comparisons)
    print("-" * 70)
    print(
        f"{'AVG':<6} {avg_fwd:<12.2f}x {avg_bwd:<12.2f}x "
        f"{avg_total:<14.2f}x {avg_vram:<12.2f}x",
    )


# ------------------------------
# Main
# ------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DCNv4 / FlashDeformAttn CUDA vs Triton benchmark",
    )

    p.add_argument("--suite", choices=["all", "dcnv4", "flash"], default="all")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Run smaller configurations with shorter timing windows",
    )

    p.add_argument(
        "--warmup-ms",
        "--warmup",
        dest="warmup_ms",
        type=int,
        default=25,
        help="Warmup time in ms for triton.testing.do_bench (legacy: --warmup)",
    )
    p.add_argument(
        "--rep-ms",
        "--iterations",
        dest="rep_ms",
        type=int,
        default=100,
        help="Benchmark time in ms for triton.testing.do_bench (legacy: --iterations)",
    )
    p.add_argument(
        "--stat",
        choices=["mean", "median", "min", "max"],
        default="mean",
        help="Statistic reported by triton.testing.do_bench",
    )
    p.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated batch sizes",
    )

    p.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write JSON report to this path",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Enable Triton proton profiling",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    _ensure_cuda_available()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if args.quick:
        if args.warmup_ms == 25:
            args.warmup_ms = 10
        if args.rep_ms == 100:
            args.rep_ms = 30

    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "suite": args.suite,
        "quick": args.quick,
        "warmup": args.warmup_ms,
        "iterations": args.rep_ms,
        "stat": args.stat,
        "batch_sizes": batch_sizes,
        "results": [],
    }

    print("=" * 90)
    print("DCNv4 / FlashDeformAttn CUDA vs Triton BENCHMARK")
    print("=" * 90)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    print(
        f"Warmup: {args.warmup_ms} ms, Rep: {args.rep_ms} ms, Stat: {args.stat}",
    )
    print(f"Batch sizes: {batch_sizes}")

    all_comparisons: dict[str, list[ComparisonResult]] = {
        "DCNv4": [],
        "FlashDeformAttn": [],
    }

    # DCNv4 Benchmark
    if args.suite in ("all", "dcnv4"):
        configs = dcnv4_configs(args.quick)
        for config in configs:
            print_header(f"DCNv4: H{config.H}xW{config.W} C{config.C} g{config.group}")
            print_table_header()

            for B in batch_sizes:
                try:
                    cuda_res = benchmark_dcnv4_single(
                        config,
                        B,
                        "CUDA",
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                        stat=args.stat,
                        profile=args.profile,
                    )
                    print_result(cuda_res)

                    triton_res = benchmark_dcnv4_single(
                        config,
                        B,
                        "Triton",
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                        stat=args.stat,
                        profile=args.profile,
                    )
                    print_result(triton_res)

                    comp = compare_results(cuda_res, triton_res)
                    all_comparisons["DCNv4"].append(comp)

                    report["results"].append(
                        {
                            "type": "DCNv4",
                            "config": asdict(config),
                            "cuda": asdict(cuda_res),
                            "triton": asdict(triton_res),
                            "comparison": asdict(comp),
                        },
                    )
                except Exception as e:
                    print(f"  B={B}: ERROR - {e}")

    # FlashDeformAttn Benchmark
    if args.suite in ("all", "flash"):
        configs = flash_configs(args.quick)
        for config in configs:
            total_len = sum(h * w for h, w in config.spatial_shapes)
            print_header(
                f"FlashDeformAttn: h{config.n_heads} d{config.d_per_head} "
                f"L{total_len} pts{config.n_points}",
            )
            print_table_header()

            for B in batch_sizes:
                try:
                    cuda_res = benchmark_flash_single(
                        config,
                        B,
                        "CUDA",
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                        stat=args.stat,
                        profile=args.profile,
                    )
                    print_result(cuda_res)

                    triton_res = benchmark_flash_single(
                        config,
                        B,
                        "Triton",
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                        stat=args.stat,
                        profile=args.profile,
                    )
                    print_result(triton_res)

                    comp = compare_results(cuda_res, triton_res)
                    all_comparisons["FlashDeformAttn"].append(comp)

                    report["results"].append(
                        {
                            "type": "FlashDeformAttn",
                            "config": {
                                "n_heads": config.n_heads,
                                "d_per_head": config.d_per_head,
                                "spatial_shapes": config.spatial_shapes,
                                "n_points": config.n_points,
                            },
                            "cuda": asdict(cuda_res),
                            "triton": asdict(triton_res),
                            "comparison": asdict(comp),
                        },
                    )
                except Exception as e:
                    print(f"  B={B}: ERROR - {e}")

    # Summary
    print_header("SUMMARY")
    for name, comps in all_comparisons.items():
        if comps:
            print_summary(comps, name)

    # Tradeoff summary
    print("\n" + "=" * 90)
    print("TRADEOFF ANALYSIS")

    if args.report:
        try:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Wrote report: {args.report}")
        except Exception as e:
            print(f"WARNING: failed to write report: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
