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

  # Adversarial testing (boundary conditions, extreme offsets, scale sweeps)
  python dcnv4_flash_regression_test.py --adversarial

  # Fuzzing with random shapes (for nightly/paper experiments)
  python dcnv4_flash_regression_test.py --fuzz 100 --fuzz-seed 42

  # Full nightly suite (comprehensive coverage)
  python dcnv4_flash_regression_test.py --nightly

  # Tolerance analysis mode (extended trials, outputs distribution stats)
  python dcnv4_flash_regression_test.py --analyze-tolerance --report tolerance_report.json

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
import random
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
    # Symmetric relative error: |a-b| / max(|a|, |b|, eps) - neutral for paper reporting
    sym_max_rel: float
    sym_p99_rel: float
    sym_p999_rel: float
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

    # Relative stats (reference=b, asymmetric)
    rel = diff / (b32.abs() + rel_eps)
    max_rel = float(rel.max().item())
    rflat = rel.flatten()
    p99_rel = float(torch.quantile(rflat, 0.99).item())
    p999_rel = float(torch.quantile(rflat, 0.999).item())

    # Symmetric relative error: |a-b| / max(|a|, |b|, eps) - neutral metric for papers
    denom_sym = torch.maximum(a32.abs(), b32.abs()) + rel_eps
    sym_rel = diff / denom_sym
    sym_max_rel = float(sym_rel.max().item())
    sym_rflat = sym_rel.flatten()
    sym_p99_rel = float(torch.quantile(sym_rflat, 0.99).item())
    sym_p999_rel = float(torch.quantile(sym_rflat, 0.999).item())

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
        sym_max_rel=sym_max_rel,
        sym_p99_rel=sym_p99_rel,
        sym_p999_rel=sym_p999_rel,
        rate_nonzero=rate_nonzero,
        rate_over_1e8=rate_over_1e8,
        rate_over_1e4=rate_over_1e4,
    )


@dataclass
class ErrorDistribution:
    """Full error distribution for paper-quality reporting and tolerance analysis."""

    # Extended percentiles
    p50: float
    p90: float
    p95: float
    p99: float
    p999: float
    p9999: float
    max_val: float

    # CDF points: (threshold, fraction_below)
    cdf_points: list[tuple[float, float]]

    # Histogram data for plotting
    histogram_bins: list[float]
    histogram_counts: list[int]


def _compute_error_distribution(
    errors: torch.Tensor,
    *,
    cdf_thresholds: list[float] | None = None,
    n_bins: int = 50,
) -> ErrorDistribution:
    """Compute full error distribution statistics for tolerance analysis.

    Args:
        errors: 1D tensor of absolute errors (already flattened).
        cdf_thresholds: Thresholds for CDF computation. Defaults to log-spaced from 1e-8 to 1e-1.
        n_bins: Number of histogram bins.

    Returns:
        ErrorDistribution with percentiles, CDF, and histogram data.

    """
    if cdf_thresholds is None:
        cdf_thresholds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    flat = errors.flatten().float()
    n_elements = flat.numel()

    # Extended percentiles - use sampling for large tensors to avoid memory issues
    # torch.quantile() can fail with "input tensor is too large" for tensors > ~2^30 elements
    percentile_vals = [0.50, 0.90, 0.95, 0.99, 0.999, 0.9999]
    percentiles = []

    # Sample size for percentile estimation (10M elements is safe for most GPUs)
    MAX_SAMPLE_SIZE = 10_000_000

    if n_elements <= MAX_SAMPLE_SIZE:
        # Small enough - use exact computation with sorted values
        sorted_vals, _ = torch.sort(flat)
        for p in percentile_vals:
            idx = min(int(p * n_elements), n_elements - 1)
            percentiles.append(float(sorted_vals[idx].item()))
    else:
        # Large tensor - use random sampling for percentile estimation
        # Generate random indices without creating a full permutation (memory-efficient)
        sample_indices = torch.randint(
            0,
            n_elements,
            (MAX_SAMPLE_SIZE,),
            device=flat.device,
        )
        sample = flat[sample_indices]
        sorted_sample, _ = torch.sort(sample)
        n_sample = sorted_sample.numel()
        for p in percentile_vals:
            idx = min(int(p * n_sample), n_sample - 1)
            percentiles.append(float(sorted_sample[idx].item()))

    max_val = float(flat.max().item())

    # CDF points
    cdf_points = []
    for threshold in cdf_thresholds:
        fraction_below = float((flat <= threshold).sum().item()) / n_elements
        cdf_points.append((threshold, fraction_below))

    # Histogram (log-spaced bins for numerical errors)
    # For large tensors, use the sample for histogram to avoid memory issues
    if n_elements > MAX_SAMPLE_SIZE:
        # Reuse sample from percentile computation or create new one
        hist_sample_indices = torch.randint(
            0,
            n_elements,
            (MAX_SAMPLE_SIZE,),
            device=flat.device,
        )
        hist_data = flat[hist_sample_indices].cpu()
    else:
        hist_data = flat.cpu()

    min_nonzero = (
        hist_data[hist_data > 0].min().item() if (hist_data > 0).any() else 1e-12
    )
    log_min = max(math.log10(min_nonzero), -12)
    log_max = max(math.log10(max_val), log_min + 1) if max_val > 0 else log_min + 1
    bin_edges = torch.logspace(log_min, log_max, n_bins + 1)
    counts = torch.histogram(hist_data, bins=bin_edges)[0]

    histogram_bins = bin_edges.tolist()
    histogram_counts = counts.int().tolist()

    return ErrorDistribution(
        p50=percentiles[0],
        p90=percentiles[1],
        p95=percentiles[2],
        p99=percentiles[3],
        p999=percentiles[4],
        p9999=percentiles[5],
        max_val=max_val,
        cdf_points=cdf_points,
        histogram_bins=histogram_bins,
        histogram_counts=histogram_counts,
    )


def _estimate_theoretical_bound(
    op_type: str,
    n_samples: int,
    n_points: int,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Estimate theoretical error bound based on operation count and dtype.

    This provides a rough lower bound on expected numerical error based on:
    - Bilinear interpolation: O(n_samples * n_points * eps) due to weighted sums
    - Softmax: O(n_points * eps) due to exp/sum operations
    - Backward atomic adds: O(fan_in * eps) due to non-deterministic accumulation

    Args:
        op_type: One of "dcnv4_fwd", "dcnv4_bwd", "flash_fwd", "flash_bwd"
        n_samples: Number of output elements (B * H * W or B * Q)
        n_points: Number of sampling points (K for DCNv4, n_levels * n_points for Flash)
        dtype: Data type (affects machine epsilon)

    Returns:
        Estimated theoretical error bound.

    """
    # Machine epsilon
    if dtype == torch.float16:
        eps = 9.77e-4  # 2^-10
    elif dtype == torch.bfloat16:
        eps = 7.81e-3  # 2^-7
    else:  # float32
        eps = 1.19e-7  # 2^-23

    if op_type == "dcnv4_fwd":
        # Bilinear interp (4 weights) + weighted sum over K points
        # Error ~ 4 * n_points * eps per output element
        # But correlated, so sqrt scaling
        return 4.0 * math.sqrt(n_points) * eps

    if op_type == "dcnv4_bwd":
        # Backward has atomic adds with fan_in ~ n_points
        # Plus gradient of bilinear interp (more terms)
        # grad_offset has more operations than grad_input
        return 8.0 * math.sqrt(n_points) * eps

    if op_type == "flash_fwd":
        # Softmax over K points + bilinear interp per point
        # softmax: exp, sum, div ~ 3K ops
        # bilinear: 4K ops
        # Total ~ 7K ops, but highly correlated
        return 10.0 * math.sqrt(n_points) * eps

    if op_type == "flash_bwd":
        # Backward through softmax + bilinear
        # More operations, atomic adds
        return 20.0 * math.sqrt(n_points) * eps

    # Conservative fallback
    return 100.0 * math.sqrt(n_points) * eps


@dataclass
class ToleranceAnalysisResult:
    """Results from tolerance analysis mode."""

    op_name: str
    tensor_name: str
    n_trials: int
    n_elements_per_trial: int

    # Observed statistics
    observed_max: float
    observed_p9999: float
    observed_p999: float
    observed_p99: float

    # Theoretical bound
    theoretical_bound: float

    # Recommendation
    recommended_atol: float
    safety_margin: float

    # Full distribution (optional, for detailed analysis)
    distribution: ErrorDistribution | None = None


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    tol: Tolerance,
    *,
    label: str,
) -> None:
    """Full-tensor closeness assertion with helpful error context.

    Args:
        actual: The tensor being tested (Triton output).
        expected: The reference tensor (CUDA output, ground truth).
        tol: Tolerance thresholds.
        label: Description for error messages.

    """
    if not _isfinite(actual):
        msg = f"{label}: Triton output contains NaN/Inf"
        raise AssertionError(msg)
    if not _isfinite(expected):
        msg = f"{label}: CUDA output contains NaN/Inf"
        raise AssertionError(msg)

    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tol.rtol,
            atol=tol.atol,
            equal_nan=False,
        )
    except AssertionError as e:
        stats = _tensor_diff_stats(actual, expected)
        msg = (
            f"{label}: assert_close failed with (atol={tol.atol:.3e}, rtol={tol.rtol:.3e})\n"
            f"  max_abs={stats.max_abs:.3e}, p99_abs={stats.p99_abs:.3e}, p999_abs={stats.p999_abs:.3e}, mean_abs={stats.mean_abs:.3e}\n"
            f"  max_rel={stats.max_rel:.3e}, p99_rel={stats.p99_rel:.3e}, p999_rel={stats.p999_rel:.3e}\n"
            f"  sym_max={stats.sym_max_rel:.3e}, sym_p99={stats.sym_p99_rel:.3e}, sym_p999={stats.sym_p999_rel:.3e}\n"
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
        sym_max_rel=max(prev.sym_max_rel, stats.sym_max_rel),
        sym_p99_rel=max(prev.sym_p99_rel, stats.sym_p99_rel),
        sym_p999_rel=max(prev.sym_p999_rel, stats.sym_p999_rel),
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
        # Line 2: asymmetric relative metrics (reference=expected, i.e., CUDA)
        print(
            f"  {'':<18} max_rel={st.max_rel:.3e}  p99_rel={st.p99_rel:.3e}  "
            f"p999_rel={st.p999_rel:.3e}",
        )
        # Line 3: symmetric relative metrics (neutral for paper reporting)
        print(
            f"  {'':<18} sym_max={st.sym_max_rel:.3e}  sym_p99={st.sym_p99_rel:.3e}  "
            f"sym_p999={st.sym_p999_rel:.3e}",
        )
        # Line 4: rate metrics (fraction of elements with diff)
        print(
            f"  {'':<18} rate_nz={st.rate_nonzero:.2%}  "
            f"rate>1e-8={st.rate_over_1e8:.2%}  rate>1e-4={st.rate_over_1e4:.2%}",
        )


# ------------------------------
# Adversarial Input Generation
# ------------------------------

# Default test scales
DEFAULT_OFFSET_SCALES = [0.1]  # Standard random offset scale
ADVERSARIAL_OFFSET_SCALES = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0]  # Extreme offsets
DEFAULT_VALUE_SCALES = [1.0]  # Standard randn scale
ADVERSARIAL_VALUE_SCALES = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3]  # Stability sweep


@dataclass
class InputConfig:
    """Configuration for test input generation."""

    name: str
    value_scale: float = 1.0
    offset_scale: float = 0.1
    include_boundary_sampling: bool = False
    include_extreme_offsets: bool = False
    include_fixed_patterns: bool = False
    include_extreme_attention: bool = False


def _make_boundary_sampling_locations(
    B: int,
    Q: int,
    G: int,
    k_total: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate sampling locations that hit boundary conditions.

    Tests normalized coordinates at exact 0, 1, eps, 1-eps, and slightly out-of-bounds.
    """
    # k_total * 3 = (loc_x, loc_y, attn) per sampling point
    sampling_loc_attn = torch.zeros(B, Q, G, k_total * 3, device=device, dtype=dtype)

    # Split into location and attention parts
    n_locs = k_total * 2  # x, y pairs

    # Corner cases for normalized coordinates
    boundary_values = [
        0.0,  # Exact boundary
        1.0,  # Exact boundary
        1e-6,  # Near zero
        1.0 - 1e-6,  # Near one
        -0.05,  # Slightly out of bounds (negative)
        1.05,  # Slightly out of bounds (positive)
        0.5,  # Center (safe)
    ]

    # Fill locations with boundary values cyclically
    loc_part = sampling_loc_attn[..., :n_locs]
    for i, val in enumerate(boundary_values):
        idx = i % n_locs
        loc_part[..., idx :: len(boundary_values)] = val

    # Fill attention with uniform weights (will be softmaxed)
    attn_part = sampling_loc_attn[..., n_locs:]
    attn_part.fill_(0.0)  # Equal attention after softmax

    return sampling_loc_attn


def _make_extreme_attention_weights(
    B: int,
    Q: int,
    G: int,
    k_total: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate attention weights that stress softmax stability.

    Tests: very large logits, very negative logits, mixed extremes, uniform.
    """
    sampling_loc_attn = torch.rand(B, Q, G, k_total * 3, device=device, dtype=dtype)

    n_locs = k_total * 2
    attn_part = sampling_loc_attn[..., n_locs:]

    # Different extreme patterns across batch/query dimensions
    # Pattern 0: Very large positive logits (overflow risk)
    attn_part[0::4, ...] = 100.0  # Large but not inf after softmax

    # Pattern 1: Very negative logits (underflow risk)
    attn_part[1::4, ...] = -100.0

    # Pattern 2: Mixed extremes (one dominant)
    attn_part[2::4, ..., 0] = 100.0
    attn_part[2::4, ..., 1:] = -100.0

    # Pattern 3: All equal (uniform attention)
    attn_part[3::4, ...] = 0.0

    return sampling_loc_attn


def _make_fixed_pattern_value(
    shape: tuple[int, ...],
    pattern: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate fixed deterministic patterns for interpolation/aggregation testing.

    Patterns:
    - "impulse": Single non-zero element at center
    - "ramp": Linear gradient across spatial dimensions
    - "checkerboard": Alternating +1/-1 pattern
    - "edge": Sharp transitions at boundaries
    """
    if len(shape) == 4:
        B, H, W, C = shape
    else:
        msg = f"Unsupported shape for pattern generation: {shape}"
        raise ValueError(msg)

    value = torch.zeros(shape, device=device, dtype=dtype)

    if pattern == "impulse":
        # Single impulse at center
        h_mid, w_mid = H // 2, W // 2
        value[:, h_mid, w_mid, :] = 1.0

    elif pattern == "ramp":
        # Linear gradient in both spatial dimensions
        h_coords = torch.linspace(0, 1, H, device=device, dtype=dtype)
        w_coords = torch.linspace(0, 1, W, device=device, dtype=dtype)
        ramp = h_coords[:, None] + w_coords[None, :]
        value[...] = ramp[None, :, :, None].expand(B, H, W, C)

    elif pattern == "checkerboard":
        # Alternating +1/-1
        h_idx = torch.arange(H, device=device)
        w_idx = torch.arange(W, device=device)
        checker = ((h_idx[:, None] + w_idx[None, :]) % 2) * 2 - 1
        value[...] = checker.to(dtype)[None, :, :, None].expand(B, H, W, C)

    elif pattern == "edge":
        # Sharp transition at H/2 and W/2
        value[:, : H // 2, :, :] = -1.0
        value[:, H // 2 :, :, :] = 1.0
        value[:, :, : W // 2, :] -= 0.5
        value[:, :, W // 2 :, :] += 0.5

    else:
        msg = f"Unknown pattern: {pattern}"
        raise ValueError(msg)

    return value


def _make_flash_fixed_pattern_value(
    B: int,
    total_len: int,
    n_heads: int,
    d_per_head: int,
    pattern: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate fixed patterns for FlashDeformAttn (flattened spatial layout)."""
    value = torch.zeros(B, total_len, n_heads, d_per_head, device=device, dtype=dtype)

    if pattern == "impulse":
        # Impulse at center of flattened sequence
        mid = total_len // 2
        value[:, mid, :, :] = 1.0

    elif pattern == "ramp":
        # Linear ramp across sequence
        ramp = torch.linspace(0, 1, total_len, device=device, dtype=dtype)
        value[...] = ramp[None, :, None, None].expand(B, total_len, n_heads, d_per_head)

    elif pattern == "checkerboard":
        # Alternating pattern
        checker = (torch.arange(total_len, device=device) % 2) * 2 - 1
        value[...] = checker.to(dtype)[None, :, None, None].expand(
            B,
            total_len,
            n_heads,
            d_per_head,
        )

    elif pattern == "edge":
        # Sharp transition
        value[:, : total_len // 2, :, :] = -1.0
        value[:, total_len // 2 :, :, :] = 1.0

    else:
        msg = f"Unknown pattern: {pattern}"
        raise ValueError(msg)

    return value


FIXED_PATTERNS = ["impulse", "ramp", "checkerboard", "edge"]


def get_input_configs(adversarial: bool = False) -> list[InputConfig]:
    """Get list of input configurations to test.

    Args:
        adversarial: If True, include adversarial/corner-case configurations.

    Returns:
        List of InputConfig objects to test.

    """
    configs = [
        InputConfig(name="standard", value_scale=1.0, offset_scale=0.1),
    ]

    if adversarial:
        # Add offset scale sweep
        for offset_scale in ADVERSARIAL_OFFSET_SCALES:
            if offset_scale != 0.1:  # Skip duplicate of standard
                configs.append(
                    InputConfig(
                        name=f"offset_scale_{offset_scale}",
                        offset_scale=offset_scale,
                    ),
                )

        # Add value scale sweep
        for value_scale in ADVERSARIAL_VALUE_SCALES:
            if value_scale != 1.0:  # Skip duplicate
                configs.append(
                    InputConfig(
                        name=f"value_scale_{value_scale:.0e}",
                        value_scale=value_scale,
                    ),
                )

        # Add boundary and fixed pattern tests
        configs.extend(
            [
                InputConfig(
                    name="boundary_sampling",
                    include_boundary_sampling=True,
                ),
                InputConfig(
                    name="extreme_attention",
                    include_extreme_attention=True,
                ),
                InputConfig(
                    name="fixed_patterns",
                    include_fixed_patterns=True,
                ),
            ],
        )

    return configs


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
    input_config: InputConfig | None = None,
) -> CaseResult:
    if input_config is None:
        input_config = InputConfig(name="standard")

    base_name = f"DCNv4 B{case.B} H{case.H} W{case.W} C{case.C} g{case.group} k{case.kernel_h}x{case.kernel_w}"
    name = (
        f"{base_name} [{input_config.name}]"
        if input_config.name != "standard"
        else base_name
    )

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

    # Input generation helper
    def make_inputs(
        seed: int,
        pattern: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        shape = (case.B, case.H, case.W, case.C)

        if pattern is not None:
            # Use fixed pattern
            value = _make_fixed_pattern_value(shape, pattern, dtype=torch.float32)
        else:
            # Random inputs scaled by config
            value = (
                torch.randn(*shape, device="cuda", dtype=torch.float32)
                * input_config.value_scale
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
            * input_config.offset_scale
        )
        grad_output = (
            torch.randn(*shape, device="cuda", dtype=torch.float32)
            * input_config.value_scale
        )
        return value, offset, grad_output

    # Warmup once (helps on first JIT / autotune)
    value, offset, grad_output = make_inputs(seed_base)

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

    # Determine trials: use fixed patterns if configured, else random
    if input_config.include_fixed_patterns:
        # Run once per pattern instead of random trials
        trial_configs = [(i, pattern) for i, pattern in enumerate(FIXED_PATTERNS)]
    else:
        trial_configs = [(t, None) for t in range(trials)]

    # Main trials
    for t, pattern in trial_configs:
        value, offset, grad_output = make_inputs(seed_base + t, pattern=pattern)

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

        _assert_close(triton_out, cuda_out, tol_fwd, label=f"{name} / forward")
        _update_worst(worst, "forward", _tensor_diff_stats(triton_out, cuda_out))

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

        _assert_close(triton_gi, cuda_gi, tol_gi, label=f"{name} / backward grad_input")
        _update_worst(worst, "grad_input", _tensor_diff_stats(triton_gi, cuda_gi))

        # Offset gradient: compare only valid range (spec), but also sanity-check padding is finite.
        cuda_go_v = cuda_go[..., :valid_range]
        triton_go_v = triton_go[..., :valid_range]
        _assert_close(
            triton_go_v,
            cuda_go_v,
            tol_go,
            label=f"{name} / backward grad_offset(valid)",
        )
        _update_worst(worst, "grad_offset", _tensor_diff_stats(triton_go_v, cuda_go_v))

        # Optional: if both produce padding, ensure finite to avoid silent explosions.
        if cuda_go.shape[-1] > valid_range:
            if not _isfinite(cuda_go[..., valid_range:]):
                msg = f"{name} / backward grad_offset(pad): CUDA contains NaN/Inf"
                raise AssertionError(msg)
        if triton_go.shape[-1] > valid_range:
            if not _isfinite(triton_go[..., valid_range:]):
                msg = f"{name} / backward grad_offset(pad): Triton contains NaN/Inf"
                raise AssertionError(msg)

    return CaseResult(name=name, passed=True, trials=len(trial_configs), worst=worst)


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
    input_config: InputConfig | None = None,
) -> CaseResult:
    if input_config is None:
        input_config = InputConfig(name="standard")

    total_len = sum(h * w for h, w in case.spatial_shapes)
    n_levels = len(case.spatial_shapes)
    k_total = n_levels * case.n_points

    base_name = (
        f"FlashDeformAttn B{case.B} L{total_len} heads{case.n_heads} d{case.d_per_head} "
        f"levels{n_levels} points{case.n_points}"
    )
    name = (
        f"{base_name} [{input_config.name}]"
        if input_config.name != "standard"
        else base_name
    )

    spatial_shapes_t = torch.tensor(
        case.spatial_shapes,
        device="cuda",
        dtype=torch.long,
    )
    level_start_index = _flash_level_start_index(case.spatial_shapes)

    worst: dict[str, DiffStats] = {}

    # Input generation helper
    def make_inputs(
        seed: int,
        pattern: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)

        # Value tensor
        if pattern is not None:
            value = _make_flash_fixed_pattern_value(
                case.B,
                total_len,
                case.n_heads,
                case.d_per_head,
                pattern,
                dtype=torch.float16,
            )
        else:
            value = (
                torch.randn(
                    case.B,
                    total_len,
                    case.n_heads,
                    case.d_per_head,
                    device="cuda",
                    dtype=torch.float16,
                )
                * input_config.value_scale
            )

        # Sampling locations and attention
        if input_config.include_boundary_sampling:
            sampling_loc_attn = _make_boundary_sampling_locations(
                case.B,
                total_len,
                case.n_heads,
                k_total,
                dtype=torch.float16,
            )
        elif input_config.include_extreme_attention:
            sampling_loc_attn = _make_extreme_attention_weights(
                case.B,
                total_len,
                case.n_heads,
                k_total,
                dtype=torch.float16,
            )
        else:
            # Standard random in [0, 1)
            sampling_loc_attn = torch.rand(
                case.B,
                total_len,
                case.n_heads,
                k_total * 3,
                device="cuda",
                dtype=torch.float16,
            )

        # Gradient output (3D: B, total_len, heads*d)
        grad_output_3d = (
            torch.randn(
                case.B,
                total_len,
                case.n_heads * case.d_per_head,
                device="cuda",
                dtype=torch.float16,
            )
            * input_config.value_scale
        )

        return value, sampling_loc_attn, grad_output_3d

    # Warmup
    value, sampling_loc_attn, grad_output_3d = make_inputs(seed_base)

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

    # Determine trials: use fixed patterns if configured, else random
    if input_config.include_fixed_patterns:
        trial_configs = [(i, pattern) for i, pattern in enumerate(FIXED_PATTERNS)]
    else:
        trial_configs = [(t, None) for t in range(trials)]

    # Trials
    for t, pattern in trial_configs:
        value, sampling_loc_attn, grad_output_3d = make_inputs(
            seed_base + t,
            pattern=pattern,
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

        # Compare in fp32 for stability (Triton=actual, CUDA=expected)
        _assert_close(
            triton_out.float(),
            cuda_out.float(),
            tol_fwd,
            label=f"{name} / forward",
        )
        _update_worst(worst, "forward", _tensor_diff_stats(triton_out, cuda_out))

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
            triton_gv.float(),
            cuda_gv.float(),
            tol_gv,
            label=f"{name} / backward grad_value",
        )
        _update_worst(worst, "grad_value", _tensor_diff_stats(triton_gv, cuda_gv))

        _assert_close(
            triton_go.float(),
            cuda_go.float(),
            tol_go,
            label=f"{name} / backward grad_offset",
        )
        _update_worst(worst, "grad_offset", _tensor_diff_stats(triton_go, cuda_go))

    return CaseResult(name=name, passed=True, trials=len(trial_configs), worst=worst)


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
# Fuzzing Suite (Constraint-Aware Shape Generation)
# ------------------------------


def validate_dcnv4_constraints(case: DCNv4Case) -> tuple[bool, str]:
    """Check if DCNv4Case satisfies CUDA kernel constraints.

    Constraints:
    1. (C / group) % 8 == 0  (d_stride divisibility for vectorized loads)
    2. (B * H_out * W_out) % (1024 / C) == 0  (block_multiplier divisibility)
    3. kernel must be 3x3 (CUDA only supports K=8 or K=9)

    Returns:
        (valid, reason): True if valid, else False with explanation.

    """
    group_channels = case.C // case.group
    if case.C % case.group != 0:
        return False, f"C ({case.C}) not divisible by group ({case.group})"

    # Constraint 1: d_stride divisibility
    if group_channels % 8 != 0:
        return False, f"group_channels ({group_channels}) not divisible by 8"

    # Constraint 2: block_multiplier divisibility
    # H_out = (H + 2*pad - dil*(kernel-1) - 1) / stride + 1
    h_out = (
        case.H + 2 * case.pad_h - case.dil_h * (case.kernel_h - 1) - 1
    ) // case.stride_h + 1
    w_out = (
        case.W + 2 * case.pad_w - case.dil_w * (case.kernel_w - 1) - 1
    ) // case.stride_w + 1
    Q = h_out * w_out
    block_multiplier = 1024 // case.C
    if block_multiplier == 0:
        return False, f"C ({case.C}) too large, block_multiplier would be 0"
    if (case.B * Q) % block_multiplier != 0:
        return (
            False,
            f"(B*Q)={case.B * Q} not divisible by block_multiplier={block_multiplier}",
        )

    # Constraint 3: kernel size
    k_total = case.kernel_h * case.kernel_w
    if k_total not in (8, 9):
        return False, f"K={k_total} not in (8, 9), CUDA only supports 3x3 kernels"

    return True, "OK"


def validate_flash_constraints(case: FlashCase) -> tuple[bool, str]:
    """Check if FlashCase satisfies CUDA kernel constraints.

    Constraints:
    1. d_per_head % 8 == 0  (d_stride divisibility)
    2. (B * total_len) % (1024 / (d_per_head * n_heads)) == 0  (block_multiplier)

    Returns:
        (valid, reason): True if valid, else False with explanation.

    """
    # Constraint 1: d_stride divisibility
    if case.d_per_head % 8 != 0:
        return False, f"d_per_head ({case.d_per_head}) not divisible by 8"

    # Constraint 2: block_multiplier divisibility
    total_len = sum(h * w for h, w in case.spatial_shapes)
    embed_dim = case.d_per_head * case.n_heads
    block_multiplier = 1024 // embed_dim
    if block_multiplier == 0:
        return False, f"embed_dim ({embed_dim}) too large, block_multiplier would be 0"
    if (case.B * total_len) % block_multiplier != 0:
        return (
            False,
            f"(B*Q)={case.B * total_len} not divisible by block_multiplier={block_multiplier}",
        )

    return True, "OK"


def generate_valid_dcnv4_shape(
    rng: random.Random,
    max_attempts: int = 100,
) -> DCNv4Case | None:
    """Generate a random DCNv4Case satisfying CUDA kernel constraints.

    Uses rejection sampling with constraint-aware parameter selection.
    """
    for _ in range(max_attempts):
        # Choose C and group such that group_channels % 8 == 0
        # Valid group_channels: 8, 16, 24, 32, 64, 128
        group_channels = rng.choice([8, 16, 32, 64])
        group = rng.choice([1, 2, 4, 8])
        C = group * group_channels

        # Choose B and spatial dims such that (B * Q) % block_multiplier == 0
        block_multiplier = 1024 // C

        # For simplicity, choose H_out = W_out and make Q divisible
        # With stride=1, pad=1, kernel=3: H_out = H
        B = rng.choice([1, 2, 4])

        # Q must satisfy: (B * Q) % block_multiplier == 0
        # So Q must be a multiple of block_multiplier / gcd(B, block_multiplier)
        from math import gcd

        q_step = block_multiplier // gcd(B, block_multiplier)
        # Q = H * W, choose H = W = sqrt(Q) for simplicity
        # Pick Q as multiple of q_step, then find H, W
        q_mult = rng.randint(1, 16)
        Q_target = q_step * q_mult
        # Find H, W close to sqrt(Q_target)
        H = W = int(Q_target**0.5)
        while Q_target > H * W:
            H += 1
        # Adjust to make Q exactly divisible
        while (B * H * W) % block_multiplier != 0:
            W += 1
            if W > 64:
                break

        if W > 64:
            continue

        case = DCNv4Case(
            B=B,
            H=H,
            W=W,
            C=C,
            group=group,
            kernel_h=3,
            kernel_w=3,
        )

        valid, _ = validate_dcnv4_constraints(case)
        if valid:
            return case

    return None


def generate_valid_flash_shape(
    rng: random.Random,
    max_attempts: int = 100,
) -> FlashCase | None:
    """Generate a random FlashCase satisfying CUDA kernel constraints.

    Uses rejection sampling with constraint-aware parameter selection.
    """
    for _ in range(max_attempts):
        # Choose d_per_head divisible by 8
        d_per_head = rng.choice([8, 16, 32, 64])
        n_heads = rng.choice([1, 2, 4, 8])
        embed_dim = d_per_head * n_heads

        # block_multiplier = 1024 / embed_dim
        block_multiplier = 1024 // embed_dim
        if block_multiplier == 0:
            continue

        B = rng.choice([1, 2, 4])

        # Generate spatial shapes for 2-4 levels
        n_levels = rng.randint(2, 4)
        spatial_shapes: list[tuple[int, int]] = []

        # Start with largest level, each subsequent level is smaller
        base_h = rng.choice([8, 16, 32])
        base_w = rng.choice([8, 16, 32])
        for i in range(n_levels):
            h = max(2, base_h // (2**i))
            w = max(2, base_w // (2**i))
            spatial_shapes.append((h, w))

        total_len = sum(h * w for h, w in spatial_shapes)

        # Check if (B * total_len) % block_multiplier == 0
        if (B * total_len) % block_multiplier != 0:
            # Try to adjust by padding with small levels
            remainder = (B * total_len) % block_multiplier
            needed = block_multiplier - remainder
            if needed <= 4:
                # Add a small level to make it work
                spatial_shapes.append((2, needed // 2 + 1))
                total_len = sum(h * w for h, w in spatial_shapes)
            else:
                continue

        if (B * total_len) % block_multiplier != 0:
            continue

        n_points = rng.choice([4, 8])

        case = FlashCase(
            B=B,
            n_heads=n_heads,
            d_per_head=d_per_head,
            spatial_shapes=tuple(spatial_shapes),
            n_points=n_points,
        )

        valid, _ = validate_flash_constraints(case)
        if valid:
            return case

    return None


def fuzz_suite(
    n_cases: int,
    seed: int,
) -> tuple[list[DCNv4Case], list[FlashCase]]:
    """Generate n_cases random valid configurations for each operator.

    Args:
        n_cases: Number of cases to generate for each operator.
        seed: Random seed for reproducibility.

    Returns:
        (dcnv4_cases, flash_cases): Lists of valid test cases.

    """
    rng = random.Random(seed)

    dcnv4_cases: list[DCNv4Case] = []
    flash_cases: list[FlashCase] = []

    for _ in range(n_cases):
        dcn_case = generate_valid_dcnv4_shape(rng)
        if dcn_case is not None:
            dcnv4_cases.append(dcn_case)

        flash_case = generate_valid_flash_shape(rng)
        if flash_case is not None:
            flash_cases.append(flash_case)

    return dcnv4_cases, flash_cases


# ------------------------------
# Tolerance Analysis
# ------------------------------


def run_dcnv4_tolerance_analysis(
    case: DCNv4Case,
    *,
    trials: int,
    seed_base: int,
    include_adversarial: bool = True,
) -> list[ToleranceAnalysisResult]:
    """Run extended tolerance analysis for a DCNv4 case.

    Collects all errors across many trials to compute distribution statistics
    and recommend tolerances.
    """
    name = f"DCNv4 B{case.B} H{case.H} W{case.W} C{case.C} g{case.group}"
    group_channels = case.C // case.group
    offset_dim, valid_range = _dcnv4_offset_dim(
        case.group,
        case.kernel_h,
        case.kernel_w,
    )
    k_total = case.kernel_h * case.kernel_w

    # Collect all errors across trials
    fwd_errors: list[torch.Tensor] = []
    gi_errors: list[torch.Tensor] = []
    go_errors: list[torch.Tensor] = []

    input_configs = get_input_configs(adversarial=include_adversarial)

    for input_cfg in input_configs:
        cfg_trials = trials // len(input_configs)
        for t in range(cfg_trials):
            torch.manual_seed(seed_base + t)
            value = torch.randn(
                case.B,
                case.H,
                case.W,
                case.C,
                device="cuda",
                dtype=torch.float32,
            )
            value *= input_cfg.value_scale
            offset = torch.randn(
                case.B,
                case.H,
                case.W,
                offset_dim,
                device="cuda",
                dtype=torch.float32,
            )
            offset *= input_cfg.offset_scale
            grad_output = torch.randn(
                case.B,
                case.H,
                case.W,
                case.C,
                device="cuda",
                dtype=torch.float32,
            )
            grad_output *= input_cfg.value_scale

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
            fwd_errors.append((triton_out - cuda_out).abs().flatten())

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
            gi_errors.append((triton_gi - cuda_gi).abs().flatten())
            go_errors.append(
                (triton_go[..., :valid_range] - cuda_go[..., :valid_range])
                .abs()
                .flatten(),
            )

    # Compute distributions
    all_fwd = torch.cat(fwd_errors)
    all_gi = torch.cat(gi_errors)
    all_go = torch.cat(go_errors)

    n_samples = case.B * case.H * case.W
    dtype = torch.float32

    results = []

    for tensor_name, errors, op_type in [
        ("forward", all_fwd, "dcnv4_fwd"),
        ("grad_input", all_gi, "dcnv4_bwd"),
        ("grad_offset", all_go, "dcnv4_bwd"),
    ]:
        dist = _compute_error_distribution(errors)
        theoretical = _estimate_theoretical_bound(op_type, n_samples, k_total, dtype)

        # Recommendation: 2x max(observed_p9999, theoretical)
        observed_max = dist.max_val
        recommended = 2.0 * max(dist.p9999, theoretical)
        safety_margin = recommended / observed_max if observed_max > 0 else float("inf")

        results.append(
            ToleranceAnalysisResult(
                op_name=name,
                tensor_name=tensor_name,
                n_trials=trials,
                n_elements_per_trial=errors.numel() // trials,
                observed_max=observed_max,
                observed_p9999=dist.p9999,
                observed_p999=dist.p999,
                observed_p99=dist.p99,
                theoretical_bound=theoretical,
                recommended_atol=recommended,
                safety_margin=safety_margin,
                distribution=dist,
            ),
        )

    return results


def run_flash_tolerance_analysis(
    case: FlashCase,
    *,
    trials: int,
    seed_base: int,
    include_adversarial: bool = True,
) -> list[ToleranceAnalysisResult]:
    """Run extended tolerance analysis for a FlashDeformAttn case."""
    total_len = sum(h * w for h, w in case.spatial_shapes)
    n_levels = len(case.spatial_shapes)
    k_total = n_levels * case.n_points
    name = (
        f"FlashDeformAttn B{case.B} L{total_len} heads{case.n_heads} d{case.d_per_head}"
    )

    spatial_shapes_t = torch.tensor(
        case.spatial_shapes,
        device="cuda",
        dtype=torch.long,
    )
    level_start_index = _flash_level_start_index(case.spatial_shapes)

    # Collect errors
    fwd_errors: list[torch.Tensor] = []
    gv_errors: list[torch.Tensor] = []
    go_errors: list[torch.Tensor] = []

    input_configs = get_input_configs(adversarial=include_adversarial)

    for input_cfg in input_configs:
        cfg_trials = trials // len(input_configs)
        for t in range(cfg_trials):
            torch.manual_seed(seed_base + t)

            value = (
                torch.randn(
                    case.B,
                    total_len,
                    case.n_heads,
                    case.d_per_head,
                    device="cuda",
                    dtype=torch.float16,
                )
                * input_cfg.value_scale
            )

            if input_cfg.include_boundary_sampling:
                sampling_loc_attn = _make_boundary_sampling_locations(
                    case.B,
                    total_len,
                    case.n_heads,
                    k_total,
                    dtype=torch.float16,
                )
            elif input_cfg.include_extreme_attention:
                sampling_loc_attn = _make_extreme_attention_weights(
                    case.B,
                    total_len,
                    case.n_heads,
                    k_total,
                    dtype=torch.float16,
                )
            else:
                sampling_loc_attn = torch.rand(
                    case.B,
                    total_len,
                    case.n_heads,
                    k_total * 3,
                    device="cuda",
                    dtype=torch.float16,
                )

            grad_output_3d = (
                torch.randn(
                    case.B,
                    total_len,
                    case.n_heads * case.d_per_head,
                    device="cuda",
                    dtype=torch.float16,
                )
                * input_cfg.value_scale
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
            fwd_errors.append((triton_out.float() - cuda_out.float()).abs().flatten())

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
            gv_errors.append((triton_gv.float() - cuda_gv.float()).abs().flatten())
            go_errors.append((triton_go.float() - cuda_go.float()).abs().flatten())

    # Compute distributions
    all_fwd = torch.cat(fwd_errors)
    all_gv = torch.cat(gv_errors)
    all_go = torch.cat(go_errors)

    n_samples = case.B * total_len
    dtype = torch.float16

    results = []

    for tensor_name, errors, op_type in [
        ("forward", all_fwd, "flash_fwd"),
        ("grad_value", all_gv, "flash_bwd"),
        ("grad_offset", all_go, "flash_bwd"),
    ]:
        dist = _compute_error_distribution(errors)
        theoretical = _estimate_theoretical_bound(op_type, n_samples, k_total, dtype)

        observed_max = dist.max_val
        recommended = 2.0 * max(dist.p9999, theoretical)
        safety_margin = recommended / observed_max if observed_max > 0 else float("inf")

        results.append(
            ToleranceAnalysisResult(
                op_name=name,
                tensor_name=tensor_name,
                n_trials=trials,
                n_elements_per_trial=errors.numel() // trials,
                observed_max=observed_max,
                observed_p9999=dist.p9999,
                observed_p999=dist.p999,
                observed_p99=dist.p99,
                theoretical_bound=theoretical,
                recommended_atol=recommended,
                safety_margin=safety_margin,
                distribution=dist,
            ),
        )

    return results


def print_tolerance_analysis(results: list[ToleranceAnalysisResult]) -> None:
    """Print tolerance analysis results in a readable format."""
    print("\n" + "=" * 80)
    print("TOLERANCE ANALYSIS RESULTS")
    print("=" * 80)

    for r in results:
        print(f"\n{r.op_name} / {r.tensor_name}")
        print(f"  Trials: {r.n_trials}, Elements/trial: {r.n_elements_per_trial:,}")
        print(
            f"  Observed: max={r.observed_max:.3e}  p9999={r.observed_p9999:.3e}  p999={r.observed_p999:.3e}  p99={r.observed_p99:.3e}",
        )
        print(f"  Theoretical bound: {r.theoretical_bound:.3e}")
        print(
            f"  RECOMMENDED atol: {r.recommended_atol:.3e}  (safety margin: {r.safety_margin:.1f}x)",
        )

        if r.distribution:
            print("  CDF: ", end="")
            for threshold, frac in r.distribution.cdf_points[:5]:
                print(f"{threshold:.0e}:{frac:.1%} ", end="")
            print()


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
    p.add_argument(
        "--adversarial",
        action="store_true",
        help="Run adversarial input tests (boundary conditions, extreme offsets, scale sweeps)",
    )
    p.add_argument(
        "--fuzz",
        type=int,
        default=0,
        metavar="N",
        help="Generate and test N random valid shapes (fuzzing mode)",
    )
    p.add_argument(
        "--fuzz-seed",
        type=int,
        default=42,
        help="Seed for shape generation in fuzzing mode (default: 42)",
    )
    p.add_argument(
        "--nightly",
        action="store_true",
        help="Full nightly suite: 1000 fuzzed shapes + all adversarial tests",
    )
    p.add_argument(
        "--analyze-tolerance",
        action="store_true",
        help="Run extended tolerance analysis mode (1000 trials, outputs distribution stats)",
    )
    p.add_argument(
        "--tolerance-trials",
        type=int,
        default=1000,
        help="Number of trials for tolerance analysis mode (default: 1000)",
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

    # Handle nightly mode: enable fuzzing + adversarial
    if args.nightly:
        args.fuzz = 1000
        args.adversarial = True
        args.trials_dcn = 20
        args.trials_flash = 10

    # Handle tolerance analysis mode (separate path)
    if args.analyze_tolerance:
        print("=" * 80)
        print("DCNv4 / FlashDeformAttn TOLERANCE ANALYSIS MODE")
        print("=" * 80)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Trials per case: {args.tolerance_trials}")
        print("Including adversarial inputs: True")

        all_analysis_results: list[ToleranceAnalysisResult] = []

        # Run tolerance analysis on representative cases
        if args.suite in ("all", "dcnv4"):
            print("\n--- DCNv4 Tolerance Analysis ---")
            dcn_cases = dcnv4_suite(quick=False)[:2]  # Use first 2 cases
            for case in dcn_cases:
                print(f"Analyzing {case}...")
                results = run_dcnv4_tolerance_analysis(
                    case,
                    trials=args.tolerance_trials,
                    seed_base=args.seed_base,
                    include_adversarial=True,
                )
                all_analysis_results.extend(results)

        if args.suite in ("all", "flash"):
            print("\n--- FlashDeformAttn Tolerance Analysis ---")
            flash_cases = flash_suite(quick=False)[:2]  # Use first 2 cases
            for case in flash_cases:
                print(f"Analyzing {case}...")
                results = run_flash_tolerance_analysis(
                    case,
                    trials=args.tolerance_trials,
                    seed_base=args.seed_base + 10_000,
                    include_adversarial=True,
                )
                all_analysis_results.extend(results)

        # Print summary
        print_tolerance_analysis(all_analysis_results)

        # Save to report if requested
        if args.report:
            tolerance_report = {
                "mode": "tolerance_analysis",
                "tolerance_trials": args.tolerance_trials,
                "results": [],
            }
            for r in all_analysis_results:
                result_dict = {
                    "op_name": r.op_name,
                    "tensor_name": r.tensor_name,
                    "n_trials": r.n_trials,
                    "n_elements_per_trial": r.n_elements_per_trial,
                    "observed_max": r.observed_max,
                    "observed_p9999": r.observed_p9999,
                    "observed_p999": r.observed_p999,
                    "observed_p99": r.observed_p99,
                    "theoretical_bound": r.theoretical_bound,
                    "recommended_atol": r.recommended_atol,
                    "safety_margin": r.safety_margin,
                }
                if r.distribution:
                    result_dict["distribution"] = {
                        "p50": r.distribution.p50,
                        "p90": r.distribution.p90,
                        "p95": r.distribution.p95,
                        "p99": r.distribution.p99,
                        "p999": r.distribution.p999,
                        "p9999": r.distribution.p9999,
                        "max_val": r.distribution.max_val,
                        "cdf_points": r.distribution.cdf_points,
                        "histogram_bins": r.distribution.histogram_bins,
                        "histogram_counts": r.distribution.histogram_counts,
                    }
                tolerance_report["results"].append(result_dict)

            try:
                with open(args.report, "w", encoding="utf-8") as f:
                    json.dump(tolerance_report, f, indent=2)
                print(f"\nWrote tolerance analysis report: {args.report}")
            except Exception as e:
                print(f"WARNING: failed to write report: {e}")

        return 0  # Tolerance analysis always "passes" (informational)

    if args.quick:
        # Reduce default trials further for quick mode unless overridden explicitly.
        if "TRIALS_DCN" not in os.environ and args.trials_dcn == 50:
            args.trials_dcn = 15
        if "TRIALS_FLASH" not in os.environ and args.trials_flash == 30:
            args.trials_flash = 10

    # Get input configurations based on adversarial flag
    input_configs = get_input_configs(adversarial=args.adversarial)

    report: dict[str, Any] = {
        "suite": args.suite,
        "quick": bool(args.quick),
        "adversarial": bool(args.adversarial),
        "fuzz": int(args.fuzz),
        "trials_dcn": int(args.trials_dcn),
        "trials_flash": int(args.trials_flash),
        "seed_base": int(args.seed_base),
        "input_configs": [cfg.name for cfg in input_configs],
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
    mode_str = "nightly" if args.nightly else ("quick" if args.quick else "standard")
    adv_str = " +adversarial" if args.adversarial else ""
    fuzz_str = f" +fuzz({args.fuzz})" if args.fuzz > 0 else ""
    print(
        f"mode={mode_str}{adv_str}{fuzz_str} suite={args.suite} seed_base={args.seed_base}",
    )

    all_passed = True

    # DCNv4
    if args.suite in ("all", "dcnv4"):
        print("\n" + "=" * 70)
        print("1) DCNv4")
        print("=" * 70)
        cases = dcnv4_suite(args.quick)
        for c in cases:
            for input_cfg in input_configs:
                try:
                    res = run_dcnv4_case(
                        c,
                        trials=args.trials_dcn,
                        seed_base=args.seed_base,
                        tol_fwd=Tolerance(args.dcn_atol, args.dcn_rtol),
                        tol_gi=Tolerance(args.dcn_gi_atol, args.dcn_gi_rtol),
                        tol_go=Tolerance(args.dcn_go_atol, args.dcn_go_rtol),
                        input_config=input_cfg,
                    )
                    _print_case_summary(res)
                    report["results"].append(
                        {
                            "name": res.name,
                            "passed": res.passed,
                            "trials": res.trials,
                            "input_config": input_cfg.name,
                            "worst": {k: asdict(v) for k, v in res.worst.items()},
                        },
                    )
                except Exception as e:
                    all_passed = False
                    cfg_name = (
                        f" [{input_cfg.name}]" if input_cfg.name != "standard" else ""
                    )
                    print(f"\n[{c}{cfg_name}] FAIL")
                    print(str(e))
                    report["results"].append(
                        {
                            "name": f"DCNv4Case({asdict(c)})",
                            "input_config": input_cfg.name,
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
            for input_cfg in input_configs:
                try:
                    res = run_flash_case(
                        c,
                        trials=args.trials_flash,
                        seed_base=args.seed_base + 10_000,
                        tol_fwd=Tolerance(args.flash_atol, args.flash_rtol),
                        tol_gv=Tolerance(args.flash_gv_atol, args.flash_gv_rtol),
                        tol_go=Tolerance(args.flash_go_atol, args.flash_go_rtol),
                        input_config=input_cfg,
                    )
                    _print_case_summary(res)
                    report["results"].append(
                        {
                            "name": res.name,
                            "passed": res.passed,
                            "trials": res.trials,
                            "input_config": input_cfg.name,
                            "worst": {k: asdict(v) for k, v in res.worst.items()},
                        },
                    )
                except Exception as e:
                    all_passed = False
                    cfg_name = (
                        f" [{input_cfg.name}]" if input_cfg.name != "standard" else ""
                    )
                    print(f"\n[{c}{cfg_name}] FAIL")
                    print(str(e))
                    report["results"].append(
                        {
                            "name": f"FlashCase({asdict(c)})",
                            "input_config": input_cfg.name,
                            "passed": False,
                            "error": str(e),
                        },
                    )

    # Fuzzing tests (random shape generation)
    if args.fuzz > 0:
        print("\n" + "=" * 70)
        print(f"3) FUZZING ({args.fuzz} random shapes, seed={args.fuzz_seed})")
        print("=" * 70)

        fuzz_dcn_cases, fuzz_flash_cases = fuzz_suite(args.fuzz, args.fuzz_seed)
        print(
            f"Generated {len(fuzz_dcn_cases)} DCNv4 cases, {len(fuzz_flash_cases)} Flash cases",
        )

        # Use only standard input config for fuzzing (faster)
        fuzz_input_cfg = InputConfig(name="fuzz")
        fuzz_trials_dcn = min(args.trials_dcn, 10)  # Fewer trials for fuzzing
        fuzz_trials_flash = min(args.trials_flash, 5)

        if args.suite in ("all", "dcnv4"):
            print(f"\n--- DCNv4 Fuzzing ({len(fuzz_dcn_cases)} cases) ---")
            for i, c in enumerate(fuzz_dcn_cases):
                try:
                    res = run_dcnv4_case(
                        c,
                        trials=fuzz_trials_dcn,
                        seed_base=args.seed_base + 100_000 + i * 1000,
                        tol_fwd=Tolerance(args.dcn_atol, args.dcn_rtol),
                        tol_gi=Tolerance(args.dcn_gi_atol, args.dcn_gi_rtol),
                        tol_go=Tolerance(args.dcn_go_atol, args.dcn_go_rtol),
                        input_config=fuzz_input_cfg,
                    )
                    # Only print summary for failures or every 10th case
                    if not res.passed or (i + 1) % 10 == 0:
                        _print_case_summary(res)
                    else:
                        print(f"  [{i + 1}/{len(fuzz_dcn_cases)}] PASS", end="\r")
                    report["results"].append(
                        {
                            "name": res.name,
                            "passed": res.passed,
                            "trials": res.trials,
                            "input_config": "fuzz",
                            "fuzz_case_index": i,
                            "worst": {k: asdict(v) for k, v in res.worst.items()},
                        },
                    )
                except Exception as e:
                    all_passed = False
                    print(f"\n  [FUZZ {i + 1}] DCNv4Case({asdict(c)}) FAIL")
                    print(f"    {e}")
                    report["results"].append(
                        {
                            "name": f"DCNv4Case({asdict(c)})",
                            "input_config": "fuzz",
                            "fuzz_case_index": i,
                            "passed": False,
                            "error": str(e),
                        },
                    )
            print()  # Newline after progress indicator

        if args.suite in ("all", "flash"):
            print(f"\n--- FlashDeformAttn Fuzzing ({len(fuzz_flash_cases)} cases) ---")
            for i, c in enumerate(fuzz_flash_cases):
                try:
                    res = run_flash_case(
                        c,
                        trials=fuzz_trials_flash,
                        seed_base=args.seed_base + 200_000 + i * 1000,
                        tol_fwd=Tolerance(args.flash_atol, args.flash_rtol),
                        tol_gv=Tolerance(args.flash_gv_atol, args.flash_gv_rtol),
                        tol_go=Tolerance(args.flash_go_atol, args.flash_go_rtol),
                        input_config=fuzz_input_cfg,
                    )
                    if not res.passed or (i + 1) % 10 == 0:
                        _print_case_summary(res)
                    else:
                        print(f"  [{i + 1}/{len(fuzz_flash_cases)}] PASS", end="\r")
                    report["results"].append(
                        {
                            "name": res.name,
                            "passed": res.passed,
                            "trials": res.trials,
                            "input_config": "fuzz",
                            "fuzz_case_index": i,
                            "worst": {k: asdict(v) for k, v in res.worst.items()},
                        },
                    )
                except Exception as e:
                    all_passed = False
                    print(f"\n  [FUZZ {i + 1}] FlashCase({asdict(c)}) FAIL")
                    print(f"    {e}")
                    report["results"].append(
                        {
                            "name": f"FlashCase({asdict(c)})",
                            "input_config": "fuzz",
                            "fuzz_case_index": i,
                            "passed": False,
                            "error": str(e),
                        },
                    )
            print()  # Newline after progress indicator

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
