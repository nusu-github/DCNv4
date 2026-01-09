#!/usr/bin/env python3
"""Performance benchmark for DCNv4 and FlashDeformAttn CUDA kernels.

This script measures execution time and memory usage for both forward
and backward passes across various input shapes and data types.

Usage:
    python scripts/benchmark.py [--warmup N] [--iterations N] [--output FILE]
    python scripts/benchmark.py --dcnv4-only
    python scripts/benchmark.py --flash-deform-only
    python scripts/benchmark.py --with-pytorch  # Compare with PyTorch reference

Output:
    - Console: Summary of timing results
    - JSON file: Full benchmark data for future reference
"""

import argparse
import gc
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from dcnv4.functions.table import TABLE

# =============================================================================
# Configuration
# =============================================================================

# DCNv4 benchmark shapes: (B, H, W, G, C)
# These shapes are selected from TABLE to ensure kernel compatibility
DCNV4_SHAPES: list[tuple[int, int, int, int, int]] = [
    # Inference scenarios (batch=1)
    (1, 64, 64, 8, 16),
    (1, 64, 64, 8, 32),
    (1, 64, 64, 8, 64),
    (1, 100, 160, 8, 16),
    (1, 100, 160, 8, 32),
    (1, 100, 160, 8, 64),
    (1, 200, 320, 8, 16),
    (1, 200, 320, 8, 32),
    (1, 200, 320, 8, 64),
    # Training scenarios (batch=64)
    (64, 7, 7, 8, 16),
    (64, 7, 7, 8, 32),
    (64, 7, 7, 8, 64),
    (64, 14, 14, 8, 16),
    (64, 14, 14, 8, 32),
    (64, 14, 14, 8, 64),
    (64, 28, 28, 8, 16),
    (64, 28, 28, 8, 32),
    (64, 28, 28, 8, 64),
    (64, 56, 56, 8, 16),
    (64, 56, 56, 8, 32),
    (64, 56, 56, 8, 64),
]

# FlashDeformAttn benchmark shapes: (B, Q, n_heads, head_dim, n_levels, n_points)
# Q must be divisible by block_multiplier (typically 8 or 16 depending on batch)
FLASH_DEFORM_SHAPES: list[tuple[int, int, int, int, int, int]] = [
    # Small batch (block_multiplier typically smaller)
    (1, 256, 8, 32, 4, 4),
    (1, 512, 8, 32, 4, 4),
    (1, 1024, 8, 32, 4, 4),
    (1, 256, 8, 64, 4, 4),
    (1, 512, 8, 64, 4, 4),
    (1, 1024, 8, 64, 4, 4),
    # Medium batch
    (2, 256, 8, 32, 4, 4),
    (2, 512, 8, 32, 4, 4),
    (2, 256, 8, 64, 4, 4),
    (2, 512, 8, 64, 4, 4),
    # Larger batch
    (4, 256, 8, 32, 4, 4),
    (4, 512, 8, 32, 4, 4),
    (8, 256, 8, 32, 4, 4),
    (8, 512, 8, 32, 4, 4),
    # Different head configurations
    (2, 256, 16, 16, 4, 4),
    (2, 256, 4, 64, 4, 4),
    # Different point counts
    (2, 256, 8, 32, 4, 8),
    (2, 256, 8, 32, 2, 4),
]

# DCNv4 kernel parameters
KERNEL = 3
STRIDE = 1
PAD = 1
DILATION = 1

# Spatial shapes for FlashDeformAttn
SPATIAL_SHAPES_CONFIG = {
    4: [(64, 64), (32, 32), (16, 16), (8, 8)],
    2: [(64, 64), (32, 32)],
    1: [(64, 64)],
}


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class TimingResult:
    """Timing result for a single benchmark run."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int


@dataclass
class MemoryResult:
    """Memory usage result."""

    peak_allocated_mb: float
    peak_reserved_mb: float


# =============================================================================
# Utility functions
# =============================================================================


def filter_valid_dcnv4_shapes(
    shapes: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    """Filter shapes to only those present in TABLE."""
    valid = []
    for b, h, w, g, c in shapes:
        key = f"{b}x{h}x{w}x{g}x{c}"
        if key in TABLE:
            valid.append((b, h, w, g, c))
    return valid


def reset_memory_stats() -> None:
    """Reset CUDA memory statistics."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_memory_usage() -> MemoryResult:
    """Get current peak memory usage."""
    return MemoryResult(
        peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024 / 1024,
        peak_reserved_mb=torch.cuda.max_memory_reserved() / 1024 / 1024,
    )


def benchmark_fn(
    fn: Callable,
    warmup: int = 10,
    iterations: int = 100,
) -> TimingResult:
    """Benchmark a function with proper GPU synchronization."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed iterations
    times: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times_tensor = torch.tensor(times)
    return TimingResult(
        mean_ms=times_tensor.mean().item(),
        std_ms=times_tensor.std().item(),
        min_ms=times_tensor.min().item(),
        max_ms=times_tensor.max().item(),
        iterations=iterations,
    )


# =============================================================================
# DCNv4 benchmark functions
# =============================================================================


def compute_offset_mask_channels(
    group: int,
    kernel: int,
    remove_center: int = 0,
) -> int:
    """Compute the number of offset_mask channels."""
    k = kernel * kernel - int(remove_center)
    return group * k * 3


def create_dcnv4_inputs(
    b: int,
    h: int,
    w: int,
    group: int,
    group_channels: int,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create input tensors for DCNv4."""
    c_total = group * group_channels
    offset_mask_channels = compute_offset_mask_channels(group, KERNEL)

    input_tensor = torch.randn(
        b,
        h,
        w,
        c_total,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    offset_mask = torch.randn(
        b,
        h,
        w,
        offset_mask_channels,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    return input_tensor, offset_mask


def benchmark_dcnv4_forward(
    shape: tuple[int, int, int, int, int],
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> tuple[TimingResult, MemoryResult]:
    """Benchmark DCNv4 forward pass."""
    from dcnv4.functions.dcnv4_func import dcnv4_forward

    b, h, w, group, group_channels = shape
    input_tensor, offset_mask = create_dcnv4_inputs(
        b,
        h,
        w,
        group,
        group_channels,
        dtype,
    )

    def forward_fn():
        return dcnv4_forward(
            input_tensor,
            offset_mask,
            KERNEL,
            KERNEL,
            STRIDE,
            STRIDE,
            PAD,
            PAD,
            DILATION,
            DILATION,
            group,
            group_channels,
            1.0,
            64,
            0,
            False,
        )

    reset_memory_stats()
    timing = benchmark_fn(forward_fn, warmup, iterations)
    memory = get_memory_usage()

    return timing, memory


def benchmark_dcnv4_backward(
    shape: tuple[int, int, int, int, int],
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> tuple[TimingResult, MemoryResult]:
    """Benchmark DCNv4 backward pass."""
    from dcnv4.functions.dcnv4_func import dcnv4_forward

    b, h, w, group, group_channels = shape

    def run_backward() -> None:
        # Fresh tensors for each backward to avoid accumulation issues
        input_tensor, offset_mask = create_dcnv4_inputs(
            b,
            h,
            w,
            group,
            group_channels,
            dtype,
            requires_grad=True,
        )
        output = dcnv4_forward(
            input_tensor,
            offset_mask,
            KERNEL,
            KERNEL,
            STRIDE,
            STRIDE,
            PAD,
            PAD,
            DILATION,
            DILATION,
            group,
            group_channels,
            1.0,
            64,
            0,
            False,
        )
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

    reset_memory_stats()
    timing = benchmark_fn(run_backward, warmup, iterations)
    memory = get_memory_usage()

    return timing, memory


def benchmark_dcnv4_pytorch_forward(
    shape: tuple[int, int, int, int, int],
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> tuple[TimingResult, MemoryResult]:
    """Benchmark DCNv4 PyTorch reference forward pass."""
    from dcnv4.functions.dcnv4_pytorch import dcnv4_forward_pytorch

    b, h, w, group, group_channels = shape
    input_tensor, offset_mask = create_dcnv4_inputs(
        b,
        h,
        w,
        group,
        group_channels,
        dtype,
    )

    def forward_fn():
        return dcnv4_forward_pytorch(
            input_tensor,
            offset_mask,
            KERNEL,
            KERNEL,
            STRIDE,
            STRIDE,
            PAD,
            PAD,
            DILATION,
            DILATION,
            group,
            group_channels,
            1.0,
            64,
            0,
            False,
        )

    reset_memory_stats()
    timing = benchmark_fn(forward_fn, warmup, iterations)
    memory = get_memory_usage()

    return timing, memory


# =============================================================================
# FlashDeformAttn benchmark functions
# =============================================================================


def create_flash_deform_inputs(
    b: int,
    q: int,
    n_heads: int,
    head_dim: int,
    n_levels: int,
    n_points: int,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Create input tensors for FlashDeformAttn."""
    spatial_sizes = SPATIAL_SHAPES_CONFIG.get(n_levels, [(32, 32)] * n_levels)
    total_len = sum(h * w for h, w in spatial_sizes)

    value = torch.randn(
        b,
        total_len,
        n_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )

    spatial_shapes = torch.tensor(spatial_sizes, dtype=torch.int64, device="cuda")

    level_start_index = torch.zeros(n_levels, dtype=torch.int64, device="cuda")
    for i in range(1, n_levels):
        level_start_index[i] = level_start_index[i - 1] + (
            spatial_shapes[i - 1, 0] * spatial_shapes[i - 1, 1]
        )

    total_dim = n_levels * n_points * 3
    sampling_loc_attn = torch.randn(
        b,
        q,
        n_heads,
        total_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    # Normalize coordinates (avoid in-place operation on leaf tensor)
    coords_dim = n_levels * n_points * 2
    coords = torch.sigmoid(sampling_loc_attn[..., :coords_dim].detach())
    attn_weights = sampling_loc_attn[..., coords_dim:].detach()
    sampling_loc_attn_normalized = torch.cat([coords, attn_weights], dim=-1)
    if requires_grad:
        sampling_loc_attn_normalized = sampling_loc_attn_normalized.requires_grad_(True)

    return (
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc_attn_normalized,
        64,
        n_points,
    )


def benchmark_flash_deform_forward(
    shape: tuple[int, int, int, int, int, int],
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> tuple[TimingResult, MemoryResult]:
    """Benchmark FlashDeformAttn forward pass."""
    from dcnv4.functions.flash_deform_attn_func import flash_deform_attn

    b, q, n_heads, head_dim, n_levels, n_points = shape
    inputs = create_flash_deform_inputs(
        b,
        q,
        n_heads,
        head_dim,
        n_levels,
        n_points,
        dtype,
    )
    value, spatial_shapes, level_start_index, sampling_loc_attn, im2col_step, K = inputs

    def forward_fn():
        return flash_deform_attn(
            value,
            spatial_shapes,
            level_start_index,
            sampling_loc_attn,
            im2col_step,
            K,
        )

    reset_memory_stats()
    timing = benchmark_fn(forward_fn, warmup, iterations)
    memory = get_memory_usage()

    return timing, memory


def benchmark_flash_deform_backward(
    shape: tuple[int, int, int, int, int, int],
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> tuple[TimingResult, MemoryResult]:
    """Benchmark FlashDeformAttn backward pass."""
    from dcnv4.functions.flash_deform_attn_func import flash_deform_attn

    b, q, n_heads, head_dim, n_levels, n_points = shape

    def run_backward() -> None:
        # Fresh tensors for each backward
        inputs = create_flash_deform_inputs(
            b,
            q,
            n_heads,
            head_dim,
            n_levels,
            n_points,
            dtype,
            requires_grad=True,
        )
        value, spatial_shapes, level_start_index, sampling_loc_attn, im2col_step, K = (
            inputs
        )

        output = flash_deform_attn(
            value,
            spatial_shapes,
            level_start_index,
            sampling_loc_attn,
            im2col_step,
            K,
        )
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

    reset_memory_stats()
    timing = benchmark_fn(run_backward, warmup, iterations)
    memory = get_memory_usage()

    return timing, memory


def benchmark_flash_deform_pytorch_forward(
    shape: tuple[int, int, int, int, int, int],
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> tuple[TimingResult, MemoryResult]:
    """Benchmark FlashDeformAttn PyTorch reference forward pass."""
    from dcnv4.modules.flash_deform_attn_torch import flash_deform_attn_torch

    b, q, n_heads, head_dim, n_levels, n_points = shape
    inputs = create_flash_deform_inputs(
        b,
        q,
        n_heads,
        head_dim,
        n_levels,
        n_points,
        dtype,
    )
    value, spatial_shapes, level_start_index, sampling_loc_attn, im2col_step, K = inputs

    def forward_fn():
        return flash_deform_attn_torch(
            value,
            spatial_shapes,
            level_start_index,
            sampling_loc_attn,
            im2col_step,
            K,
        )

    reset_memory_stats()
    timing = benchmark_fn(forward_fn, warmup, iterations)
    memory = get_memory_usage()

    return timing, memory


# =============================================================================
# Main benchmark routines
# =============================================================================


def run_dcnv4_benchmarks(
    warmup: int,
    iterations: int,
    include_pytorch: bool = False,
) -> dict:
    """Run all DCNv4 benchmarks."""
    results = {
        "cuda": {"float32": [], "float16": []},
    }
    if include_pytorch:
        results["pytorch"] = {"float32": [], "float16": []}

    # Filter to valid shapes
    valid_shapes = filter_valid_dcnv4_shapes(DCNV4_SHAPES)
    if not valid_shapes:
        print("  WARNING: No valid DCNv4 shapes found in TABLE")
        return results

    dtypes = [
        (torch.float32, "float32"),
        (torch.float16, "float16"),
    ]

    print("\n" + "=" * 70)
    print("DCNv4 CUDA Benchmark")
    print("=" * 70)
    print(f"Testing {len(valid_shapes)} shapes")

    for dtype, dtype_name in dtypes:
        print(f"\n--- {dtype_name.upper()} ---")

        for shape in valid_shapes:
            shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}"

            try:
                # CUDA forward
                fwd_timing, fwd_memory = benchmark_dcnv4_forward(
                    shape,
                    dtype,
                    warmup,
                    iterations,
                )
                # CUDA backward
                bwd_timing, bwd_memory = benchmark_dcnv4_backward(
                    shape,
                    dtype,
                    warmup,
                    iterations,
                )

                # Compute throughput (elements per second)
                b, h, w, g, c = shape
                num_elements = b * h * w * g * c
                throughput_fwd = num_elements / (fwd_timing.mean_ms / 1000)
                throughput_bwd = num_elements / (bwd_timing.mean_ms / 1000)

                results["cuda"][dtype_name].append(
                    {
                        "shape": list(shape),
                        "forward_ms": fwd_timing.mean_ms,
                        "forward_std_ms": fwd_timing.std_ms,
                        "backward_ms": bwd_timing.mean_ms,
                        "backward_std_ms": bwd_timing.std_ms,
                        "forward_memory_mb": fwd_memory.peak_allocated_mb,
                        "backward_memory_mb": bwd_memory.peak_allocated_mb,
                        "throughput_fwd": throughput_fwd,
                        "throughput_bwd": throughput_bwd,
                    },
                )

                print(
                    f"  {shape_key:20s} | "
                    f"fwd: {fwd_timing.mean_ms:7.3f} ms | "
                    f"bwd: {bwd_timing.mean_ms:7.3f} ms | "
                    f"mem: {fwd_memory.peak_allocated_mb:6.1f} MB",
                )

            except Exception as e:
                print(f"  {shape_key:20s} | SKIP: {e}")

    if include_pytorch:
        print("\n" + "=" * 70)
        print("DCNv4 PyTorch Reference Benchmark")
        print("=" * 70)

        for dtype, dtype_name in dtypes:
            print(f"\n--- {dtype_name.upper()} ---")

            for shape in valid_shapes:
                shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}"

                try:
                    fwd_timing, fwd_memory = benchmark_dcnv4_pytorch_forward(
                        shape,
                        dtype,
                        warmup,
                        iterations,
                    )

                    results["pytorch"][dtype_name].append(
                        {
                            "shape": list(shape),
                            "forward_ms": fwd_timing.mean_ms,
                            "forward_std_ms": fwd_timing.std_ms,
                            "forward_memory_mb": fwd_memory.peak_allocated_mb,
                        },
                    )

                    print(
                        f"  {shape_key:20s} | "
                        f"fwd: {fwd_timing.mean_ms:7.3f} ms | "
                        f"mem: {fwd_memory.peak_allocated_mb:6.1f} MB",
                    )

                except Exception as e:
                    print(f"  {shape_key:20s} | SKIP: {e}")

    return results


def run_flash_deform_benchmarks(
    warmup: int,
    iterations: int,
    include_pytorch: bool = False,
) -> dict:
    """Run all FlashDeformAttn benchmarks."""
    results = {
        "cuda": {"float32": [], "float16": []},
    }
    if include_pytorch:
        results["pytorch"] = {"float32": [], "float16": []}

    dtypes = [
        (torch.float32, "float32"),
        (torch.float16, "float16"),
    ]

    print("\n" + "=" * 70)
    print("FlashDeformAttn CUDA Benchmark")
    print("=" * 70)
    print(f"Testing {len(FLASH_DEFORM_SHAPES)} shapes")

    for dtype, dtype_name in dtypes:
        print(f"\n--- {dtype_name.upper()} ---")

        for shape in FLASH_DEFORM_SHAPES:
            shape_key = (
                f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}x{shape[5]}"
            )

            try:
                # CUDA forward
                fwd_timing, fwd_memory = benchmark_flash_deform_forward(
                    shape,
                    dtype,
                    warmup,
                    iterations,
                )
                # CUDA backward
                bwd_timing, bwd_memory = benchmark_flash_deform_backward(
                    shape,
                    dtype,
                    warmup,
                    iterations,
                )

                # Compute throughput
                b, q, n_heads, head_dim, _n_levels, _n_points = shape
                num_elements = b * q * n_heads * head_dim
                throughput_fwd = num_elements / (fwd_timing.mean_ms / 1000)
                throughput_bwd = num_elements / (bwd_timing.mean_ms / 1000)

                results["cuda"][dtype_name].append(
                    {
                        "shape": list(shape),
                        "forward_ms": fwd_timing.mean_ms,
                        "forward_std_ms": fwd_timing.std_ms,
                        "backward_ms": bwd_timing.mean_ms,
                        "backward_std_ms": bwd_timing.std_ms,
                        "forward_memory_mb": fwd_memory.peak_allocated_mb,
                        "backward_memory_mb": bwd_memory.peak_allocated_mb,
                        "throughput_fwd": throughput_fwd,
                        "throughput_bwd": throughput_bwd,
                    },
                )

                print(
                    f"  {shape_key:25s} | "
                    f"fwd: {fwd_timing.mean_ms:7.3f} ms | "
                    f"bwd: {bwd_timing.mean_ms:7.3f} ms | "
                    f"mem: {fwd_memory.peak_allocated_mb:6.1f} MB",
                )

            except Exception as e:
                print(f"  {shape_key:25s} | SKIP: {e}")

    if include_pytorch:
        print("\n" + "=" * 70)
        print("FlashDeformAttn PyTorch Reference Benchmark")
        print("=" * 70)

        for dtype, dtype_name in dtypes:
            print(f"\n--- {dtype_name.upper()} ---")

            for shape in FLASH_DEFORM_SHAPES:
                shape_key = (
                    f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}x{shape[5]}"
                )

                try:
                    fwd_timing, fwd_memory = benchmark_flash_deform_pytorch_forward(
                        shape,
                        dtype,
                        warmup,
                        iterations,
                    )

                    results["pytorch"][dtype_name].append(
                        {
                            "shape": list(shape),
                            "forward_ms": fwd_timing.mean_ms,
                            "forward_std_ms": fwd_timing.std_ms,
                            "forward_memory_mb": fwd_memory.peak_allocated_mb,
                        },
                    )

                    print(
                        f"  {shape_key:25s} | "
                        f"fwd: {fwd_timing.mean_ms:7.3f} ms | "
                        f"mem: {fwd_memory.peak_allocated_mb:6.1f} MB",
                    )

                except Exception as e:
                    print(f"  {shape_key:25s} | SKIP: {e}")

    return results


def print_speedup_summary(results: dict, name: str) -> None:
    """Print CUDA vs PyTorch speedup summary."""
    if "pytorch" not in results:
        return

    print("\n" + "=" * 70)
    print(f"{name} SPEEDUP SUMMARY (CUDA / PyTorch)")
    print("=" * 70)

    for dtype_name in ["float32", "float16"]:
        cuda_results = results["cuda"].get(dtype_name, [])
        pytorch_results = results["pytorch"].get(dtype_name, [])

        if not cuda_results or not pytorch_results:
            continue

        print(f"\n--- {dtype_name.upper()} ---")

        # Build lookup by shape
        pytorch_by_shape = {tuple(r["shape"]): r for r in pytorch_results}

        speedups = []
        for cuda_r in cuda_results:
            shape = tuple(cuda_r["shape"])
            if shape not in pytorch_by_shape:
                continue

            pytorch_r = pytorch_by_shape[shape]
            speedup = pytorch_r["forward_ms"] / cuda_r["forward_ms"]
            speedups.append(speedup)

            shape_str = "x".join(map(str, shape))
            print(f"  {shape_str:25s} | {speedup:6.2f}x")

        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"  {'AVERAGE':25s} | {avg_speedup:6.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DCNv4 and FlashDeformAttn performance",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of timed iterations (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--dcnv4-only",
        action="store_true",
        help="Only run DCNv4 benchmarks",
    )
    parser.add_argument(
        "--flash-deform-only",
        action="store_true",
        help="Only run FlashDeformAttn benchmarks",
    )
    parser.add_argument(
        "--with-pytorch",
        action="store_true",
        help="Include PyTorch reference benchmarks (slower, for comparison only)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)

    print("DCNv4 / FlashDeformAttn Performance Benchmark")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Timed iterations: {args.iterations}")

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "cuda_device": torch.cuda.get_device_name(0),
            "pytorch_version": torch.__version__,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
    }

    start_time = time.time()

    # Run benchmarks
    run_dcnv4 = not args.flash_deform_only
    run_flash = not args.dcnv4_only
    include_pytorch = args.with_pytorch

    if run_dcnv4:
        all_results["dcnv4"] = run_dcnv4_benchmarks(
            args.warmup,
            args.iterations,
            include_pytorch,
        )
        if include_pytorch:
            print_speedup_summary(all_results["dcnv4"], "DCNv4")

    if run_flash:
        all_results["flash_deform_attn"] = run_flash_deform_benchmarks(
            args.warmup,
            args.iterations,
            include_pytorch,
        )
        if include_pytorch:
            print_speedup_summary(all_results["flash_deform_attn"], "FlashDeformAttn")

    elapsed = time.time() - start_time
    all_results["metadata"]["elapsed_seconds"] = elapsed

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
