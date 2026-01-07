"""Regression test for DCNv4 / FlashDeformAttn CUDA vs Triton implementations.

Purpose
-------
This script is designed for *regression detection* (CI / local sanity), not statistical proof.
It compares CUDA-extension outputs against Triton outputs across a small suite of parameter
configurations and random seeds, checking numerical closeness with explicit atol/rtol
thresholds and reporting worst-case error metrics.

Key design choices
------------------
- No p-values / TOST. Those are not appropriate with per-element correlated samples.
- Trial is the unit of repetition; we track worst-case metrics across trials.
- Uses torch.testing.assert_close for full-tensor checks (robust), plus summarized metrics
  (max/mean/quantiles) to aid debugging.

Usage examples
--------------
  # Quick run (good for pre-commit)
  python dcnv4_flash_regression_test.py --quick

  # CI-style run (more trials)
  python dcnv4_flash_regression_test.py --suite all --trials-dcn 200 --trials-flash 100

  # Tune tolerances
  python dcnv4_flash_regression_test.py --dcn-atol 5e-4 --dcn-rtol 5e-4

  # Save a JSON report
  python dcnv4_flash_regression_test.py --report out_report.json

Exit codes
----------
0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

# Project imports
from dcnv4 import _C, triton_ops

# ------------------------------
# Utilities
# ------------------------------


def _ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        msg = "CUDA is required for this regression test."
        raise RuntimeError(msg)


def _set_global_numerics() -> None:
    # Reduce accidental drift from TF32. Custom kernels may not use this, but keep it consistent.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _isfinite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass
class DiffStats:
    max_abs: float
    mean_abs: float
    p99_abs: float
    p999_abs: float
    max_rel: float
    p99_rel: float
    p999_rel: float
    # Rate metrics: fraction of elements with nonzero or over-threshold diff
    rate_nonzero: float  # fraction of elements where diff != 0
    rate_over_1e8: float  # fraction of elements where diff > 1e-8 (for fp32)
    rate_over_1e4: float  # fraction of elements where diff > 1e-4 (for fp16)


@dataclass
class CaseResult:
    name: str
    passed: bool
    trials: int
    worst: dict[str, DiffStats]


def _tensor_diff_stats(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    rel_eps: float = 1e-12,
) -> DiffStats:
    """Compute summary stats of |a-b| and |a-b|/(|b|+eps) in float32."""
    # Compare in fp32 to avoid quantile instability in fp16.
    a32 = a.detach().float()
    b32 = b.detach().float()
    diff = (a32 - b32).abs()

    # Absolute stats
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    flat = diff.flatten()
    n_elements = flat.numel()

    # Quantiles: requires enough elements; for tiny tensors quantile still works.
    p99_abs = float(torch.quantile(flat, 0.99).item())
    p999_abs = float(torch.quantile(flat, 0.999).item())

    # Relative stats (reference=b)
    rel = diff / (b32.abs() + rel_eps)
    max_rel = float(rel.max().item())
    rflat = rel.flatten()
    p99_rel = float(torch.quantile(rflat, 0.99).item())
    p999_rel = float(torch.quantile(rflat, 0.999).item())

    # Rate metrics: fraction of elements with diff above thresholds
    rate_nonzero = float((flat > 0).sum().item()) / n_elements
    rate_over_1e8 = float((flat > 1e-8).sum().item()) / n_elements
    rate_over_1e4 = float((flat > 1e-4).sum().item()) / n_elements

    return DiffStats(
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        p999_abs=p999_abs,
        max_rel=max_rel,
        p99_rel=p99_rel,
        p999_rel=p999_rel,
        rate_nonzero=rate_nonzero,
        rate_over_1e8=rate_over_1e8,
        rate_over_1e4=rate_over_1e4,
    )


def _assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    tol: Tolerance,
    *,
    label: str,
) -> None:
    """Full-tensor closeness assertion with helpful error context."""
    if not _isfinite(a):
        msg = f"{label}: CUDA output contains NaN/Inf"
        raise AssertionError(msg)
    if not _isfinite(b):
        msg = f"{label}: Triton output contains NaN/Inf"
        raise AssertionError(msg)

    try:
        torch.testing.assert_close(a, b, rtol=tol.rtol, atol=tol.atol, equal_nan=False)
    except AssertionError as e:
        stats = _tensor_diff_stats(a, b)
        msg = (
            f"{label}: assert_close failed with (atol={tol.atol:.3e}, rtol={tol.rtol:.3e})\n"
            f"  max_abs={stats.max_abs:.3e}, p99_abs={stats.p99_abs:.3e}, p999_abs={stats.p999_abs:.3e}, mean_abs={stats.mean_abs:.3e}\n"
            f"  max_rel={stats.max_rel:.3e}, p99_rel={stats.p99_rel:.3e}, p999_rel={stats.p999_rel:.3e}\n"
        )
        raise AssertionError(msg) from e


def _update_worst(worst: dict[str, DiffStats], key: str, stats: DiffStats) -> None:
    prev = worst.get(key)
    if prev is None:
        worst[key] = stats
        return

    # Worst-case by max_abs (and keep corresponding other metrics, but merge conservatively).
    worst[key] = DiffStats(
        max_abs=max(prev.max_abs, stats.max_abs),
        mean_abs=max(prev.mean_abs, stats.mean_abs),
        p99_abs=max(prev.p99_abs, stats.p99_abs),
        p999_abs=max(prev.p999_abs, stats.p999_abs),
        max_rel=max(prev.max_rel, stats.max_rel),
        p99_rel=max(prev.p99_rel, stats.p99_rel),
        p999_rel=max(prev.p999_rel, stats.p999_rel),
        rate_nonzero=max(prev.rate_nonzero, stats.rate_nonzero),
        rate_over_1e8=max(prev.rate_over_1e8, stats.rate_over_1e8),
        rate_over_1e4=max(prev.rate_over_1e4, stats.rate_over_1e4),
    )


def _print_case_summary(res: CaseResult) -> None:
    print(f"\n[{res.name}] {'PASS' if res.passed else 'FAIL'}  (trials={res.trials})")
    for k, st in res.worst.items():
        # Line 1: absolute metrics
        print(
            f"  {k:<18} max_abs={st.max_abs:.3e}  p99_abs={st.p99_abs:.3e}  "
            f"p999_abs={st.p999_abs:.3e}  mean_abs={st.mean_abs:.3e}",
        )
        # Line 2: relative metrics
        print(
            f"  {'':<18} max_rel={st.max_rel:.3e}  p99_rel={st.p99_rel:.3e}  "
            f"p999_rel={st.p999_rel:.3e}",
        )
        # Line 3: rate metrics (fraction of elements with diff)
        print(
            f"  {'':<18} rate_nz={st.rate_nonzero:.2%}  "
            f"rate>1e-8={st.rate_over_1e8:.2%}  rate>1e-4={st.rate_over_1e4:.2%}",
        )


# ------------------------------
# DCNv4
# ------------------------------


@dataclass(frozen=True)
class DCNv4Case:
    B: int
    H: int
    W: int
    C: int
    group: int
    kernel_h: int
    kernel_w: int
    # fixed hyperparams used in your original call
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 1
    pad_w: int = 1
    dil_h: int = 1
    dil_w: int = 1
    offset_scale: float = 1.0


def _dcnv4_offset_dim(group: int, kernel_h: int, kernel_w: int) -> tuple[int, int]:
    k_total = kernel_h * kernel_w
    valid = group * k_total * 3
    offset_dim = int(math.ceil(valid / 8) * 8)
    return offset_dim, valid


def run_dcnv4_case(
    case: DCNv4Case,
    *,
    trials: int,
    seed_base: int,
    tol_fwd: Tolerance,
    tol_gi: Tolerance,
    tol_go: Tolerance,
    offset_input_scale: float = 0.1,
) -> CaseResult:
    name = f"DCNv4 B{case.B} H{case.H} W{case.W} C{case.C} g{case.group} k{case.kernel_h}x{case.kernel_w}"

    group_channels = case.C // case.group
    if case.C % case.group != 0:
        msg = f"{name}: C must be divisible by group"
        raise ValueError(msg)

    offset_dim, valid_range = _dcnv4_offset_dim(
        case.group,
        case.kernel_h,
        case.kernel_w,
    )

    worst: dict[str, DiffStats] = {}

    # Warmup once (helps on first JIT / autotune)
    torch.manual_seed(seed_base)
    value = torch.randn(
        case.B,
        case.H,
        case.W,
        case.C,
        device="cuda",
        dtype=torch.float32,
    )
    offset = (
        torch.randn(
            case.B,
            case.H,
            case.W,
            offset_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * offset_input_scale
    )
    grad_output = torch.randn(
        case.B,
        case.H,
        case.W,
        case.C,
        device="cuda",
        dtype=torch.float32,
    )

    _ = _C.dcnv4_forward(
        value,
        offset,
        case.kernel_h,
        case.kernel_w,
        case.stride_h,
        case.stride_w,
        case.pad_h,
        case.pad_w,
        case.dil_h,
        case.dil_w,
        case.group,
        group_channels,
        case.offset_scale,
        256,
        0,
        8,
        128,
        False,
    )
    _ = triton_ops.dcnv4_forward(
        value,
        offset,
        case.kernel_h,
        case.kernel_w,
        case.stride_h,
        case.stride_w,
        case.pad_h,
        case.pad_w,
        case.dil_h,
        case.dil_w,
        case.group,
        group_channels,
        case.offset_scale,
        0,
        False,
    )

    # Main trials
    for t in range(trials):
        torch.manual_seed(seed_base + t)
        value = torch.randn(
            case.B,
            case.H,
            case.W,
            case.C,
            device="cuda",
            dtype=torch.float32,
        )
        offset = (
            torch.randn(
                case.B,
                case.H,
                case.W,
                offset_dim,
                device="cuda",
                dtype=torch.float32,
            )
            * offset_input_scale
        )
        grad_output = torch.randn(
            case.B,
            case.H,
            case.W,
            case.C,
            device="cuda",
            dtype=torch.float32,
        )

        # Forward
        cuda_out = _C.dcnv4_forward(
            value,
            offset,
            case.kernel_h,
            case.kernel_w,
            case.stride_h,
            case.stride_w,
            case.pad_h,
            case.pad_w,
            case.dil_h,
            case.dil_w,
            case.group,
            group_channels,
            case.offset_scale,
            256,
            0,
            8,
            128,
            False,
        )
        triton_out = triton_ops.dcnv4_forward(
            value,
            offset,
            case.kernel_h,
            case.kernel_w,
            case.stride_h,
            case.stride_w,
            case.pad_h,
            case.pad_w,
            case.dil_h,
            case.dil_w,
            case.group,
            group_channels,
            case.offset_scale,
            0,
            False,
        )

        _assert_close(cuda_out, triton_out, tol_fwd, label=f"{name} / forward")
        _update_worst(worst, "forward", _tensor_diff_stats(cuda_out, triton_out))

        # Backward
        cuda_gi, cuda_go = _C.dcnv4_backward(
            value,
            offset,
            case.kernel_h,
            case.kernel_w,
            case.stride_h,
            case.stride_w,
            case.pad_h,
            case.pad_w,
            case.dil_h,
            case.dil_w,
            case.group,
            group_channels,
            case.offset_scale,
            256,
            grad_output,
            0,
            2,
            128,
            False,
        )
        triton_gi, triton_go = triton_ops.dcnv4_backward(
            value,
            offset,
            grad_output,
            case.kernel_h,
            case.kernel_w,
            case.stride_h,
            case.stride_w,
            case.pad_h,
            case.pad_w,
            case.dil_h,
            case.dil_w,
            case.group,
            group_channels,
            case.offset_scale,
            0,
            False,
        )

        _assert_close(cuda_gi, triton_gi, tol_gi, label=f"{name} / backward grad_input")
        _update_worst(worst, "grad_input", _tensor_diff_stats(cuda_gi, triton_gi))

        # Offset gradient: compare only valid range (spec), but also sanity-check padding is finite.
        cuda_go_v = cuda_go[..., :valid_range]
        triton_go_v = triton_go[..., :valid_range]
        _assert_close(
            cuda_go_v,
            triton_go_v,
            tol_go,
            label=f"{name} / backward grad_offset(valid)",
        )
        _update_worst(worst, "grad_offset", _tensor_diff_stats(cuda_go_v, triton_go_v))

        # Optional: if both produce padding, ensure finite to avoid silent explosions.
        if cuda_go.shape[-1] > valid_range:
            if not _isfinite(cuda_go[..., valid_range:]):
                msg = f"{name} / backward grad_offset(pad): CUDA contains NaN/Inf"
                raise AssertionError(msg)
        if triton_go.shape[-1] > valid_range:
            if not _isfinite(triton_go[..., valid_range:]):
                msg = f"{name} / backward grad_offset(pad): Triton contains NaN/Inf"
                raise AssertionError(msg)

    return CaseResult(name=name, passed=True, trials=trials, worst=worst)


# ------------------------------
# FlashDeformAttn
# ------------------------------


@dataclass(frozen=True)
class FlashCase:
    B: int
    n_heads: int
    d_per_head: int
    spatial_shapes: tuple[tuple[int, int], ...]
    n_points: int


def _flash_level_start_index(spatial_shapes: Sequence[tuple[int, int]]) -> torch.Tensor:
    start = [0]
    for h, w in spatial_shapes[:-1]:
        start.append(start[-1] + h * w)
    return torch.tensor(start, device="cuda", dtype=torch.long)


def run_flash_case(
    case: FlashCase,
    *,
    trials: int,
    seed_base: int,
    tol_fwd: Tolerance,
    tol_gv: Tolerance,
    tol_go: Tolerance,
) -> CaseResult:
    total_len = sum(h * w for h, w in case.spatial_shapes)
    n_levels = len(case.spatial_shapes)
    k_total = n_levels * case.n_points

    name = (
        f"FlashDeformAttn B{case.B} L{total_len} heads{case.n_heads} d{case.d_per_head} "
        f"levels{n_levels} points{case.n_points}"
    )

    spatial_shapes_t = torch.tensor(
        case.spatial_shapes,
        device="cuda",
        dtype=torch.long,
    )
    level_start_index = _flash_level_start_index(case.spatial_shapes)

    worst: dict[str, DiffStats] = {}

    # Warmup
    torch.manual_seed(seed_base)
    value = torch.randn(
        case.B,
        total_len,
        case.n_heads,
        case.d_per_head,
        device="cuda",
        dtype=torch.float16,
    )
    sampling_loc_attn = torch.rand(
        case.B,
        total_len,
        case.n_heads,
        k_total * 3,
        device="cuda",
        dtype=torch.float16,
    )
    grad_output_3d = torch.randn(
        case.B,
        total_len,
        case.n_heads * case.d_per_head,
        device="cuda",
        dtype=torch.float16,
    )

    _ = _C.flash_deform_attn_forward(
        value,
        spatial_shapes_t,
        level_start_index,
        sampling_loc_attn,
        64,
        case.n_points,
        8,
        128,
    )
    _ = triton_ops.flash_deform_attn_forward(
        value,
        spatial_shapes_t,
        level_start_index,
        sampling_loc_attn,
        case.n_points,
    )

    # Trials
    for t in range(trials):
        torch.manual_seed(seed_base + t)
        value = torch.randn(
            case.B,
            total_len,
            case.n_heads,
            case.d_per_head,
            device="cuda",
            dtype=torch.float16,
        )
        sampling_loc_attn = torch.rand(
            case.B,
            total_len,
            case.n_heads,
            k_total * 3,
            device="cuda",
            dtype=torch.float16,
        )
        grad_output_3d = torch.randn(
            case.B,
            total_len,
            case.n_heads * case.d_per_head,
            device="cuda",
            dtype=torch.float16,
        )

        # Forward
        cuda_out = _C.flash_deform_attn_forward(
            value,
            spatial_shapes_t,
            level_start_index,
            sampling_loc_attn,
            64,
            case.n_points,
            8,
            128,
        )
        triton_out = triton_ops.flash_deform_attn_forward(
            value,
            spatial_shapes_t,
            level_start_index,
            sampling_loc_attn,
            case.n_points,
        )

        # Compare in fp32 for stability
        _assert_close(
            cuda_out.float(),
            triton_out.float(),
            tol_fwd,
            label=f"{name} / forward",
        )
        _update_worst(worst, "forward", _tensor_diff_stats(cuda_out, triton_out))

        # Backward
        cuda_gv, cuda_go = _C.flash_deform_attn_backward(
            value,
            spatial_shapes_t,
            level_start_index,
            sampling_loc_attn,
            grad_output_3d.contiguous(),
            64,
            case.n_points,
            2,
            128,
        )
        grad_output_4d = grad_output_3d.view(
            case.B,
            total_len,
            case.n_heads,
            case.d_per_head,
        ).contiguous()
        triton_gv, triton_go = triton_ops.flash_deform_attn_backward(
            value,
            spatial_shapes_t,
            level_start_index,
            sampling_loc_attn,
            grad_output_4d,
            case.n_points,
        )

        _assert_close(
            cuda_gv.float(),
            triton_gv.float(),
            tol_gv,
            label=f"{name} / backward grad_value",
        )
        _update_worst(worst, "grad_value", _tensor_diff_stats(cuda_gv, triton_gv))

        _assert_close(
            cuda_go.float(),
            triton_go.float(),
            tol_go,
            label=f"{name} / backward grad_offset",
        )
        _update_worst(worst, "grad_offset", _tensor_diff_stats(cuda_go, triton_go))

    return CaseResult(name=name, passed=True, trials=trials, worst=worst)


# ------------------------------
# Suites
# ------------------------------


def dcnv4_suite(quick: bool) -> list[DCNv4Case]:
    # Keep sizes modest for CI. Add a few boundary-ish cases.
    #
    # CUDA kernel constraint (dcnv4_im2col_cuda.cuh:300-301):
    #   block_multiplier = block_thread / (D / d_stride) / G
    #   assert((B * Q) % block_multiplier == 0)
    #
    # With block_thread=128, d_stride=8:
    #   block_multiplier = 1024 / C
    # Also: D (= C / group) must be divisible by d_stride=8
    #
    # So test cases must satisfy:
    #   1. (C / group) % 8 == 0
    #   2. (B * H_out * W_out) % (1024 / C) == 0
    if quick:
        return [
            DCNv4Case(B=2, H=16, W=16, C=64, group=4, kernel_h=3, kernel_w=3),
            # Changed H=7,W=11 -> H=8,W=8 so Q=64, 64 % 32 == 0
            DCNv4Case(B=1, H=8, W=8, C=32, group=4, kernel_h=3, kernel_w=3),
        ]
    return [
        DCNv4Case(B=2, H=16, W=16, C=64, group=4, kernel_h=3, kernel_w=3),
        # Changed H=7,W=11 -> H=8,W=8 so Q=64, 64 % 32 == 0
        DCNv4Case(B=1, H=8, W=8, C=32, group=4, kernel_h=3, kernel_w=3),
        # Changed C=96,group=8 (D=12 not divisible by 8) -> C=64,group=8 (D=8)
        # Also B=2,H=9,W=9: Q=81, block_multiplier=16, B*Q=162, 162 % 16 != 0
        # Changed to H=8,W=8: Q=64, B*Q=128, 128 % 16 == 0
        DCNv4Case(B=2, H=8, W=8, C=64, group=8, kernel_h=3, kernel_w=3),
        # CUDA kernel only supports K=8 or K=9 (3x3 kernel with/without center)
        # Changed from 1x1 kernel to 3x3 with default padding
        # Keep pad=1 (default) so H_out == H_in (test creates offset with input dims)
        DCNv4Case(B=1, H=8, W=8, C=128, group=8, kernel_h=3, kernel_w=3),
        # Changed H=5,W=3 (Q=15) -> H=4,W=4 (Q=16) so 16 % 16 == 0
        DCNv4Case(B=1, H=4, W=4, C=64, group=4, kernel_h=3, kernel_w=3),
    ]


def flash_suite(quick: bool) -> list[FlashCase]:
    # CUDA kernel constraint (flash_deform_im2col_cuda.cuh:225-226):
    #   block_multiplier = block_thread / (D / d_stride) / G
    #   assert((B * Q) % block_multiplier == 0)
    #
    # With block_thread=128, d_stride=8:
    #   block_multiplier = 1024 / (D * G)
    # Also: D must be divisible by d_stride=8
    #
    # So test cases must satisfy:
    #   1. d_per_head % 8 == 0
    #   2. (B * total_len) % (1024 / (d_per_head * n_heads)) == 0
    if quick:
        return [
            FlashCase(
                B=2,
                n_heads=8,
                d_per_head=32,
                spatial_shapes=((16, 16), (8, 8), (4, 4), (2, 2)),
                n_points=4,
            ),
        ]
    return [
        FlashCase(
            B=2,
            n_heads=8,
            d_per_head=32,
            spatial_shapes=((16, 16), (8, 8), (4, 4), (2, 2)),
            n_points=4,
        ),
        # Changed: removed (1, 1) level since Q=85 not divisible by 4
        # New Q = 64+16+4 = 84, B*Q = 84, 84 % 4 == 0
        FlashCase(
            B=1,
            n_heads=4,
            d_per_head=64,
            spatial_shapes=((8, 8), (4, 4), (2, 2)),
            n_points=4,
        ),
        # Changed: odd spatial shapes -> even shapes for divisibility
        # block_multiplier = 1024 / (16 * 8) = 8
        # Q = 64+16+4+4 = 88, B*Q = 176, 176 % 8 == 0
        FlashCase(
            B=2,
            n_heads=8,
            d_per_head=16,
            spatial_shapes=((8, 8), (4, 4), (2, 2), (2, 2)),
            n_points=4,
        ),
    ]


# ------------------------------
# Main
# ------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DCNv4 / FlashDeformAttn CUDA vs Triton regression test",
    )

    p.add_argument("--suite", choices=["all", "dcnv4", "flash"], default="all")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller suite with fewer trials",
    )

    p.add_argument("--trials-dcn", type=int, default=50)
    p.add_argument("--trials-flash", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=0)

    # Tolerances: tightened based on observed worst-case values (4-8x margin).
    # DCNv4 observed: forward ~1.8e-7, grad_input ~4.8e-7, grad_offset ~3.8e-6
    p.add_argument("--dcn-atol", type=float, default=1e-6)
    p.add_argument("--dcn-rtol", type=float, default=1e-6)
    p.add_argument("--dcn-gi-atol", type=float, default=2e-6)
    p.add_argument("--dcn-gi-rtol", type=float, default=2e-6)
    p.add_argument("--dcn-go-atol", type=float, default=2e-5)
    p.add_argument("--dcn-go-rtol", type=float, default=2e-5)

    # Flash observed: forward ~4.9e-4, grad_value ~2.0e-3, grad_offset ~3.1e-2
    p.add_argument("--flash-atol", type=float, default=2e-3)
    p.add_argument("--flash-rtol", type=float, default=2e-3)
    p.add_argument("--flash-gv-atol", type=float, default=5e-3)
    p.add_argument("--flash-gv-rtol", type=float, default=5e-3)
    p.add_argument("--flash-go-atol", type=float, default=5e-2)
    p.add_argument(
        "--flash-go-rtol",
        type=float,
        default=1e-2,
    )  # rtol more important here

    p.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write a JSON report to this path",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    _ensure_cuda_available()
    _set_global_numerics()

    if args.quick:
        # Reduce default trials further for quick mode unless overridden explicitly.
        if "TRIALS_DCN" not in os.environ and args.trials_dcn == 50:
            args.trials_dcn = 15
        if "TRIALS_FLASH" not in os.environ and args.trials_flash == 30:
            args.trials_flash = 10

    report: dict[str, Any] = {
        "suite": args.suite,
        "quick": bool(args.quick),
        "trials_dcn": int(args.trials_dcn),
        "trials_flash": int(args.trials_flash),
        "seed_base": int(args.seed_base),
        "tolerances": {
            "dcn_fwd": {"atol": args.dcn_atol, "rtol": args.dcn_rtol},
            "dcn_gi": {"atol": args.dcn_gi_atol, "rtol": args.dcn_gi_rtol},
            "dcn_go": {"atol": args.dcn_go_atol, "rtol": args.dcn_go_rtol},
            "flash_fwd": {"atol": args.flash_atol, "rtol": args.flash_rtol},
            "flash_gv": {"atol": args.flash_gv_atol, "rtol": args.flash_gv_rtol},
            "flash_go": {"atol": args.flash_go_atol, "rtol": args.flash_go_rtol},
        },
        "results": [],
    }

    print("=" * 70)
    print("DCNv4 / FlashDeformAttn CUDA vs Triton REGRESSION TEST")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"quick={args.quick} suite={args.suite} seed_base={args.seed_base}")

    all_passed = True

    # DCNv4
    if args.suite in ("all", "dcnv4"):
        print("\n" + "=" * 70)
        print("1) DCNv4")
        print("=" * 70)
        cases = dcnv4_suite(args.quick)
        for c in cases:
            try:
                res = run_dcnv4_case(
                    c,
                    trials=args.trials_dcn,
                    seed_base=args.seed_base,
                    tol_fwd=Tolerance(args.dcn_atol, args.dcn_rtol),
                    tol_gi=Tolerance(args.dcn_gi_atol, args.dcn_gi_rtol),
                    tol_go=Tolerance(args.dcn_go_atol, args.dcn_go_rtol),
                )
                _print_case_summary(res)
                report["results"].append(
                    {
                        "name": res.name,
                        "passed": res.passed,
                        "trials": res.trials,
                        "worst": {k: asdict(v) for k, v in res.worst.items()},
                    },
                )
            except Exception as e:
                all_passed = False
                print(f"\n[{c}] FAIL")
                print(str(e))
                report["results"].append(
                    {
                        "name": f"DCNv4Case({asdict(c)})",
                        "passed": False,
                        "error": str(e),
                    },
                )

    # Flash
    if args.suite in ("all", "flash"):
        print("\n" + "=" * 70)
        print("2) FlashDeformAttn")
        print("=" * 70)
        cases = flash_suite(args.quick)
        for c in cases:
            try:
                res = run_flash_case(
                    c,
                    trials=args.trials_flash,
                    seed_base=args.seed_base + 10_000,
                    tol_fwd=Tolerance(args.flash_atol, args.flash_rtol),
                    tol_gv=Tolerance(args.flash_gv_atol, args.flash_gv_rtol),
                    tol_go=Tolerance(args.flash_go_atol, args.flash_go_rtol),
                )
                _print_case_summary(res)
                report["results"].append(
                    {
                        "name": res.name,
                        "passed": res.passed,
                        "trials": res.trials,
                        "worst": {k: asdict(v) for k, v in res.worst.items()},
                    },
                )
            except Exception as e:
                all_passed = False
                print(f"\n[{c}] FAIL")
                print(str(e))
                report["results"].append(
                    {
                        "name": f"FlashCase({asdict(c)})",
                        "passed": False,
                        "error": str(e),
                    },
                )

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print("PASS" if all_passed else "FAIL")
    print("=" * 70)

    if args.report:
        try:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Wrote report: {args.report}")
        except Exception as e:
            print(f"WARNING: failed to write report: {e}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
