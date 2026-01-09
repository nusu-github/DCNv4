#!/usr/bin/env python3
"""Empirical tolerance measurement for FlashDeformAttn CUDA vs PyTorch reference.

This script measures the error distribution between the CUDA implementation
and the PyTorch reference implementation to establish data-driven tolerance
thresholds.

This is intended to be run ONCE locally (not in CI) to determine appropriate
rtol/atol values. The results should be committed as constants.

Usage:
    python scripts/measure_flash_deform_attn_tolerance.py [--seeds N] [--output FILE]

Output:
    - Console: Summary statistics and recommended tolerances
    - JSON file: Full measurement data for future reference
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch

# =============================================================================
# Configuration
# =============================================================================

# Representative shapes for multi-scale deformable attention
# Format: (B, num_queries, n_heads, head_dim, n_levels, n_points)
REPRESENTATIVE_SHAPES: list[tuple[int, int, int, int, int, int]] = [
    # Small configurations (inference-like)
    (1, 100, 8, 32, 4, 4),
    (1, 300, 8, 32, 4, 4),
    (1, 900, 8, 32, 4, 4),
    # Medium configurations
    (2, 100, 8, 32, 4, 4),
    (2, 300, 8, 64, 4, 4),
    (4, 100, 8, 32, 4, 4),
    # Large batch (training-like)
    (8, 100, 8, 32, 4, 4),
    (8, 300, 8, 32, 4, 4),
    # Different head configurations
    (2, 100, 4, 64, 4, 4),
    (2, 100, 16, 16, 4, 4),
    # Different level/point configurations
    (2, 100, 8, 32, 2, 4),
    (2, 100, 8, 32, 4, 8),
    (2, 100, 8, 32, 1, 4),
    # Larger queries
    (1, 1000, 8, 32, 4, 4),
    (2, 500, 8, 32, 4, 4),
]

# Standard spatial shapes for multi-scale features (FPN-style)
SPATIAL_SHAPES_CONFIG = {
    4: [(64, 64), (32, 32), (16, 16), (8, 8)],  # 4 levels
    2: [(64, 64), (32, 32)],  # 2 levels
    1: [(64, 64)],  # 1 level
}


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class ErrorStats:
    """Statistics for a single measurement."""

    max_abs: float
    max_rel: float
    mean_abs: float
    mean_rel: float
    p99_abs: float
    p99_rel: float
    p999_abs: float
    p999_rel: float
    num_elements: int
    num_nonzero_ref: int


@dataclass
class ShapeMeasurement:
    """Measurements for a single shape across all seeds."""

    shape: tuple[int, int, int, int, int, int]
    forward_errors: list[ErrorStats] = field(default_factory=list)
    grad_value_errors: list[ErrorStats] = field(default_factory=list)
    grad_sampling_loc_attn_errors: list[ErrorStats] = field(default_factory=list)
    backward_nondeterminism_value: list[float] = field(default_factory=list)
    backward_nondeterminism_sampling: list[float] = field(default_factory=list)


@dataclass
class HalfPrecisionMeasurement:
    """Measurements for half-precision types."""

    dtype_name: str
    shape: tuple[int, int, int, int, int, int]
    forward_errors: list[ErrorStats] = field(default_factory=list)


# =============================================================================
# Helper functions
# =============================================================================


def compute_error_stats(
    cuda_out: torch.Tensor,
    ref_out: torch.Tensor,
) -> ErrorStats:
    """Compute comprehensive error statistics."""
    diff = (cuda_out - ref_out).float()
    ref_float = ref_out.float()

    abs_err = diff.abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()

    # Relative error (filter small ref values to avoid misleading statistics)
    nonzero_mask = ref_float.abs() > 0.01
    num_nonzero = nonzero_mask.sum().item()

    if num_nonzero > 0:
        rel_err = (abs_err[nonzero_mask] / ref_float[nonzero_mask].abs()).float()
        max_rel = rel_err.max().item()
        mean_rel = rel_err.mean().item()

        # Percentiles
        rel_sorted = rel_err.flatten().sort().values
        p99_idx = int(len(rel_sorted) * 0.99)
        p999_idx = int(len(rel_sorted) * 0.999)
        p99_rel = rel_sorted[min(p99_idx, len(rel_sorted) - 1)].item()
        p999_rel = rel_sorted[min(p999_idx, len(rel_sorted) - 1)].item()
    else:
        max_rel = mean_rel = p99_rel = p999_rel = 0.0

    # Absolute error percentiles
    abs_sorted = abs_err.flatten().sort().values
    p99_idx = int(len(abs_sorted) * 0.99)
    p999_idx = int(len(abs_sorted) * 0.999)
    p99_abs = abs_sorted[min(p99_idx, len(abs_sorted) - 1)].item()
    p999_abs = abs_sorted[min(p999_idx, len(abs_sorted) - 1)].item()

    return ErrorStats(
        max_abs=max_abs,
        max_rel=max_rel,
        mean_abs=mean_abs,
        mean_rel=mean_rel,
        p99_abs=p99_abs,
        p99_rel=p99_rel,
        p999_abs=p999_abs,
        p999_rel=p999_rel,
        num_elements=cuda_out.numel(),
        num_nonzero_ref=int(num_nonzero),
    )


def create_test_inputs(
    B: int,
    Q: int,
    n_heads: int,
    head_dim: int,
    n_levels: int,
    n_points: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Create test inputs for FlashDeformAttn."""
    # Get spatial shapes for this level count
    spatial_sizes = SPATIAL_SHAPES_CONFIG.get(n_levels, [(32, 32)] * n_levels)

    # Compute total spatial tokens
    total_len = sum(h * w for h, w in spatial_sizes)

    # Value tensor: (B, total_len, n_heads, head_dim)
    value = torch.randn(B, total_len, n_heads, head_dim, device=device, dtype=dtype)

    # Spatial shapes: (n_levels, 2) - [H, W] for each level
    spatial_shapes = torch.tensor(spatial_sizes, dtype=torch.int64, device=device)

    # Level start indices
    level_start_index = torch.zeros(n_levels, dtype=torch.int64, device=device)
    for i in range(1, n_levels):
        level_start_index[i] = level_start_index[i - 1] + (
            spatial_shapes[i - 1, 0] * spatial_shapes[i - 1, 1]
        )

    # sampling_loc_attn: (B, Q, n_heads, L * K * 3)
    # Contains: L*K*2 coordinates + L*K attention logits
    total_dim = n_levels * n_points * 3
    sampling_loc_attn = torch.randn(
        B,
        Q,
        n_heads,
        total_dim,
        device=device,
        dtype=dtype,
    )

    # Normalize coordinates to [0, 1] range
    coords_dim = n_levels * n_points * 2
    sampling_loc_attn[..., :coords_dim] = torch.sigmoid(
        sampling_loc_attn[..., :coords_dim],
    )

    return (
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc_attn,
        64,  # im2col_step
        n_points,  # K
    )


def run_cuda_forward(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_loc_attn: torch.Tensor,
    im2col_step: int,
    K: int,
) -> torch.Tensor:
    """Run CUDA forward pass."""
    from dcnv4.functions.flash_deform_attn_func import flash_deform_attn

    return flash_deform_attn(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc_attn,
        im2col_step,
        K,
    )


def run_ref_forward(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_loc_attn: torch.Tensor,
    im2col_step: int,
    K: int,
) -> torch.Tensor:
    """Run PyTorch reference forward pass."""
    from dcnv4.modules.flash_deform_attn_torch import flash_deform_attn_torch

    return flash_deform_attn_torch(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc_attn,
        im2col_step,
        K,
    )


# =============================================================================
# Measurement functions
# =============================================================================


def measure_forward_error(
    shape: tuple[int, int, int, int, int, int],
    seed: int,
) -> ErrorStats:
    """Measure forward pass error for a single seed."""
    B, Q, n_heads, head_dim, n_levels, n_points = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = create_test_inputs(B, Q, n_heads, head_dim, n_levels, n_points)

    with torch.no_grad():
        out_cuda = run_cuda_forward(*inputs)
        out_ref = run_ref_forward(*inputs)

    return compute_error_stats(out_cuda, out_ref)


def measure_gradient_error(
    shape: tuple[int, int, int, int, int, int],
    seed: int,
) -> tuple[ErrorStats, ErrorStats]:
    """Measure gradient error for a single seed.

    Returns (grad_value_error, grad_sampling_loc_attn_error).
    """
    B, Q, n_heads, head_dim, n_levels, n_points = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs_cuda = create_test_inputs(B, Q, n_heads, head_dim, n_levels, n_points)
    (
        value_cuda,
        spatial_shapes,
        level_start_index,
        sampling_loc_attn_cuda,
        im2col_step,
        K,
    ) = inputs_cuda

    # Enable gradients
    value_cuda = value_cuda.clone().requires_grad_(True)
    sampling_loc_attn_cuda = sampling_loc_attn_cuda.clone().requires_grad_(True)

    # Reference path (same data)
    value_ref = value_cuda.detach().clone().requires_grad_(True)
    sampling_loc_attn_ref = sampling_loc_attn_cuda.detach().clone().requires_grad_(True)

    # Forward
    out_cuda = run_cuda_forward(
        value_cuda,
        spatial_shapes,
        level_start_index,
        sampling_loc_attn_cuda,
        im2col_step,
        K,
    )
    out_ref = run_ref_forward(
        value_ref,
        spatial_shapes,
        level_start_index,
        sampling_loc_attn_ref,
        im2col_step,
        K,
    )

    # Backward with same grad_output
    torch.manual_seed(seed + 1000000)
    grad_output = torch.randn_like(out_cuda)

    out_cuda.backward(grad_output)
    out_ref.backward(grad_output)

    assert value_cuda.grad is not None
    assert sampling_loc_attn_cuda.grad is not None
    assert value_ref.grad is not None
    assert sampling_loc_attn_ref.grad is not None

    grad_value_err = compute_error_stats(value_cuda.grad, value_ref.grad)
    grad_sampling_err = compute_error_stats(
        sampling_loc_attn_cuda.grad,
        sampling_loc_attn_ref.grad,
    )

    return grad_value_err, grad_sampling_err


def measure_backward_nondeterminism(
    shape: tuple[int, int, int, int, int, int],
    seed: int,
    num_runs: int = 5,
) -> tuple[float, float]:
    """Measure backward pass non-determinism.

    Runs the same backward pass multiple times and measures the max difference.
    Returns (max_diff_value, max_diff_sampling).
    """
    from dcnv4 import ops
    from dcnv4.functions.flash_deform_attn_func import findspec_bwd

    B, Q, n_heads, head_dim, n_levels, n_points = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = create_test_inputs(B, Q, n_heads, head_dim, n_levels, n_points)
    value, spatial_shapes, level_start_index, sampling_loc_attn, im2col_step, K = inputs

    # Generate output shape by running forward once
    with torch.no_grad():
        out = run_cuda_forward(*inputs)
    grad_output = torch.randn_like(out)

    # Get backward kernel config
    d_stride_bwd, blockthread_bwd = findspec_bwd(B, Q, n_heads, head_dim)

    # Run backward multiple times
    grad_values: list[torch.Tensor] = []
    grad_samplings: list[torch.Tensor] = []

    for _ in range(num_runs):
        grad_value, grad_sampling = ops.flash_deform_attn_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_loc_attn,
            grad_output.contiguous(),
            im2col_step,
            K,
            d_stride_bwd,
            blockthread_bwd,
        )
        grad_values.append(grad_value.clone())
        grad_samplings.append(grad_sampling.clone())

    # Compute max pairwise difference
    max_diff_value = 0.0
    max_diff_sampling = 0.0

    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            diff_value = (grad_values[i] - grad_values[j]).abs().max().item()
            diff_sampling = (grad_samplings[i] - grad_samplings[j]).abs().max().item()
            max_diff_value = max(max_diff_value, diff_value)
            max_diff_sampling = max(max_diff_sampling, diff_sampling)

    return max_diff_value, max_diff_sampling


def measure_half_precision_error(
    shape: tuple[int, int, int, int, int, int],
    seed: int,
    dtype: torch.dtype,
) -> ErrorStats:
    """Measure forward pass error for half-precision types."""
    B, Q, n_heads, head_dim, n_levels, n_points = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = create_test_inputs(
        B,
        Q,
        n_heads,
        head_dim,
        n_levels,
        n_points,
        dtype=dtype,
    )

    with torch.no_grad():
        out_cuda = run_cuda_forward(*inputs)
        out_ref = run_ref_forward(*inputs)

    return compute_error_stats(out_cuda, out_ref)


# =============================================================================
# Main measurement routine
# =============================================================================


def aggregate_stats(stats_list: list[ErrorStats]) -> dict:
    """Aggregate statistics across multiple measurements."""
    if not stats_list:
        return {}

    max_abs_values = [s.max_abs for s in stats_list]
    max_rel_values = [s.max_rel for s in stats_list]
    p99_abs_values = [s.p99_abs for s in stats_list]
    p99_rel_values = [s.p99_rel for s in stats_list]
    p999_abs_values = [s.p999_abs for s in stats_list]
    p999_rel_values = [s.p999_rel for s in stats_list]

    def percentile(values: list[float], p: float) -> float:
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    return {
        "max_abs": {
            "max": max(max_abs_values),
            "p99": percentile(max_abs_values, 99),
            "p999": percentile(max_abs_values, 99.9),
            "mean": sum(max_abs_values) / len(max_abs_values),
        },
        "max_rel": {
            "max": max(max_rel_values),
            "p99": percentile(max_rel_values, 99),
            "p999": percentile(max_rel_values, 99.9),
            "mean": sum(max_rel_values) / len(max_rel_values),
        },
        "p99_abs": {
            "max": max(p99_abs_values),
            "mean": sum(p99_abs_values) / len(p99_abs_values),
        },
        "p99_rel": {
            "max": max(p99_rel_values),
            "mean": sum(p99_rel_values) / len(p99_rel_values),
        },
        "p999_abs": {
            "max": max(p999_abs_values),
            "mean": sum(p999_abs_values) / len(p999_abs_values),
        },
        "p999_rel": {
            "max": max(p999_rel_values),
            "mean": sum(p999_rel_values) / len(p999_rel_values),
        },
        "num_samples": len(stats_list),
    }


def run_measurements(
    num_seeds: int = 200,
    include_half: bool = True,
) -> dict:
    """Run all measurements and return results."""
    results: dict = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_seeds": num_seeds,
            "shapes": [list(s) for s in REPRESENTATIVE_SHAPES],
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "N/A",
            "pytorch_version": torch.__version__,
        },
        "forward": [],
        "gradient": {"value": [], "sampling_loc_attn": []},
        "nondeterminism": {"value": [], "sampling_loc_attn": []},
        "half_precision": {"float16": [], "bfloat16": []},
    }

    total_shapes = len(REPRESENTATIVE_SHAPES)
    total_iterations = total_shapes * num_seeds

    print(f"Running measurements with {num_seeds} seeds across {total_shapes} shapes")
    print(f"Total iterations: {total_iterations}")
    print(f"CUDA device: {results['metadata']['cuda_device']}")
    print("-" * 60)

    # Forward pass measurements
    print("\n[1/4] Measuring forward pass errors...")
    forward_stats: list[ErrorStats] = []
    for shape_idx, shape in enumerate(REPRESENTATIVE_SHAPES):
        shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}x{shape[5]}"
        for seed in range(num_seeds):
            try:
                stats = measure_forward_error(shape, seed)
                forward_stats.append(stats)
            except RuntimeError as e:
                print(f"  Skip {shape_key} seed={seed}: {e}")
                continue

            if (seed + 1) % 50 == 0:
                print(
                    f"  Shape {shape_idx + 1}/{total_shapes} ({shape_key}): {seed + 1}/{num_seeds} seeds",
                )

    results["forward"] = aggregate_stats(forward_stats)

    # Gradient measurements
    print("\n[2/4] Measuring gradient errors...")
    grad_value_stats: list[ErrorStats] = []
    grad_sampling_stats: list[ErrorStats] = []
    for shape_idx, shape in enumerate(REPRESENTATIVE_SHAPES):
        shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}x{shape[5]}"
        for seed in range(num_seeds):
            try:
                value_err, sampling_err = measure_gradient_error(shape, seed)
                grad_value_stats.append(value_err)
                grad_sampling_stats.append(sampling_err)
            except RuntimeError as e:
                print(f"  Skip {shape_key} seed={seed}: {e}")
                continue

            if (seed + 1) % 50 == 0:
                print(
                    f"  Shape {shape_idx + 1}/{total_shapes} ({shape_key}): {seed + 1}/{num_seeds} seeds",
                )

    results["gradient"]["value"] = aggregate_stats(grad_value_stats)
    results["gradient"]["sampling_loc_attn"] = aggregate_stats(grad_sampling_stats)

    # Non-determinism measurements (fewer seeds)
    print("\n[3/4] Measuring backward non-determinism...")
    nondeterminism_value: list[float] = []
    nondeterminism_sampling: list[float] = []
    nondet_seeds = min(num_seeds, 50)
    for shape_idx, shape in enumerate(REPRESENTATIVE_SHAPES):
        shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}x{shape[5]}"
        for seed in range(nondet_seeds):
            try:
                diff_value, diff_sampling = measure_backward_nondeterminism(
                    shape,
                    seed,
                    num_runs=5,
                )
                nondeterminism_value.append(diff_value)
                nondeterminism_sampling.append(diff_sampling)
            except RuntimeError as e:
                print(f"  Skip {shape_key} seed={seed}: {e}")
                continue

            if (seed + 1) % 25 == 0:
                print(
                    f"  Shape {shape_idx + 1}/{total_shapes} ({shape_key}): {seed + 1}/{nondet_seeds} seeds",
                )

    results["nondeterminism"]["value"] = {
        "max": max(nondeterminism_value) if nondeterminism_value else 0,
        "mean": sum(nondeterminism_value) / len(nondeterminism_value)
        if nondeterminism_value
        else 0,
        "num_samples": len(nondeterminism_value),
    }
    results["nondeterminism"]["sampling_loc_attn"] = {
        "max": max(nondeterminism_sampling) if nondeterminism_sampling else 0,
        "mean": sum(nondeterminism_sampling) / len(nondeterminism_sampling)
        if nondeterminism_sampling
        else 0,
        "num_samples": len(nondeterminism_sampling),
    }

    # Half-precision measurements
    if include_half:
        half_seeds = min(num_seeds, 100)

        print("\n[4/4] Measuring half-precision errors...")
        for dtype, dtype_name in [
            (torch.float16, "float16"),
            (torch.bfloat16, "bfloat16"),
        ]:
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                print(f"  {dtype_name} not supported on this device, skipping")
                results["half_precision"][dtype_name] = {}
                continue

            print(f"  Measuring {dtype_name}...")
            hp_stats: list[ErrorStats] = []
            for shape_idx, shape in enumerate(REPRESENTATIVE_SHAPES):
                shape_key = (
                    f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}x{shape[5]}"
                )
                for seed in range(half_seeds):
                    try:
                        stats = measure_half_precision_error(shape, seed, dtype)
                        hp_stats.append(stats)
                    except RuntimeError as e:
                        print(f"    Skip {shape_key} seed={seed}: {e}")
                        continue

                    if (seed + 1) % 50 == 0:
                        print(
                            f"    Shape {shape_idx + 1}/{total_shapes} ({shape_key}): {seed + 1}/{half_seeds} seeds",
                        )

            results["half_precision"][dtype_name] = aggregate_stats(hp_stats)
    else:
        print("\n[4/4] Skipping half-precision measurements")

    return results


def print_summary(results: dict) -> None:
    """Print a human-readable summary of results."""
    print("\n" + "=" * 60)
    print("MEASUREMENT SUMMARY (FlashDeformAttn)")
    print("=" * 60)

    print(f"\nDevice: {results['metadata']['cuda_device']}")
    print(f"Seeds: {results['metadata']['num_seeds']}")
    print(f"Shapes tested: {len(results['metadata']['shapes'])}")

    print("\n" + "-" * 60)
    print("FORWARD PASS")
    print("-" * 60)
    fwd = results["forward"]
    if fwd:
        print(
            f"  Max absolute error:  max={fwd['max_abs']['max']:.2e}, p99.9={fwd['max_abs']['p999']:.2e}",
        )
        print(
            f"  Max relative error:  max={fwd['max_rel']['max']:.2e}, p99.9={fwd['max_rel']['p999']:.2e}",
        )

    print("\n" + "-" * 60)
    print("GRADIENT (value)")
    print("-" * 60)
    gv = results["gradient"]["value"]
    if gv:
        print(
            f"  Max absolute error:  max={gv['max_abs']['max']:.2e}, p99.9={gv['max_abs']['p999']:.2e}",
        )
        print(
            f"  Max relative error:  max={gv['max_rel']['max']:.2e}, p99.9={gv['max_rel']['p999']:.2e}",
        )

    print("\n" + "-" * 60)
    print("GRADIENT (sampling_loc_attn)")
    print("-" * 60)
    gs = results["gradient"]["sampling_loc_attn"]
    if gs:
        print(
            f"  Max absolute error:  max={gs['max_abs']['max']:.2e}, p99.9={gs['max_abs']['p999']:.2e}",
        )
        print(
            f"  Max relative error:  max={gs['max_rel']['max']:.2e}, p99.9={gs['max_rel']['p999']:.2e}",
        )

    print("\n" + "-" * 60)
    print("BACKWARD NON-DETERMINISM (atomicAdd)")
    print("-" * 60)
    nd_v = results["nondeterminism"]["value"]
    nd_s = results["nondeterminism"]["sampling_loc_attn"]
    if nd_v:
        print(
            f"  grad_value:             max={nd_v['max']:.2e}, mean={nd_v['mean']:.2e}",
        )
    if nd_s:
        print(
            f"  grad_sampling_loc_attn: max={nd_s['max']:.2e}, mean={nd_s['mean']:.2e}",
        )

    print("\n" + "-" * 60)
    print("HALF PRECISION (float16)")
    print("-" * 60)
    hp16 = results["half_precision"]["float16"]
    if hp16:
        print(
            f"  Max absolute error:  max={hp16['max_abs']['max']:.2e}, p99.9={hp16['max_abs']['p999']:.2e}",
        )
        print(
            f"  Max relative error:  max={hp16['max_rel']['max']:.2e}, p99.9={hp16['max_rel']['p999']:.2e}",
        )

    print("\n" + "-" * 60)
    print("HALF PRECISION (bfloat16)")
    print("-" * 60)
    hpbf = results["half_precision"]["bfloat16"]
    if hpbf:
        print(
            f"  Max absolute error:  max={hpbf['max_abs']['max']:.2e}, p99.9={hpbf['max_abs']['p999']:.2e}",
        )
        print(
            f"  Max relative error:  max={hpbf['max_rel']['max']:.2e}, p99.9={hpbf['max_rel']['p999']:.2e}",
        )


def recommend_tolerances(results: dict) -> dict:
    """Generate recommended tolerance values based on measurements."""
    recommendations: dict = {}

    # Safety factor for p99.9 -> threshold
    SAFETY_FACTOR = 2.0

    # Forward pass (float32)
    fwd = results["forward"]
    if fwd:
        recommendations["forward_float32"] = {
            "rtol": round(fwd["max_rel"]["p999"] * SAFETY_FACTOR, 6),
            "atol": round(fwd["max_abs"]["p999"] * SAFETY_FACTOR, 8),
            "rationale": f"p99.9 * {SAFETY_FACTOR} safety factor",
        }

    # Gradients (need higher tolerance due to accumulation + atomicAdd)
    gv = results["gradient"]["value"]
    gs = results["gradient"]["sampling_loc_attn"]
    nd_v = results["nondeterminism"]["value"]
    nd_s = results["nondeterminism"]["sampling_loc_attn"]

    if gv and nd_v:
        nondet_floor = nd_v["max"] * 2 if nd_v["max"] > 0 else 0
        recommendations["grad_value_float32"] = {
            "rtol": round(max(gv["max_rel"]["p999"] * SAFETY_FACTOR, 1e-4), 6),
            "atol": round(max(gv["max_abs"]["p999"] * SAFETY_FACTOR, nondet_floor), 8),
            "rationale": f"p99.9 * {SAFETY_FACTOR}, floor from non-determinism",
        }

    if gs and nd_s:
        nondet_floor = nd_s["max"] * 2 if nd_s["max"] > 0 else 0
        recommendations["grad_sampling_loc_attn_float32"] = {
            "rtol": round(max(gs["max_rel"]["p999"] * SAFETY_FACTOR, 1e-4), 6),
            "atol": round(max(gs["max_abs"]["p999"] * SAFETY_FACTOR, nondet_floor), 8),
            "rationale": f"p99.9 * {SAFETY_FACTOR}, floor from non-determinism",
        }

    # Half precision
    hp16 = results["half_precision"]["float16"]
    if hp16:
        recommendations["forward_float16"] = {
            "rtol": round(hp16["max_rel"]["p999"] * SAFETY_FACTOR, 4),
            "atol": round(hp16["max_abs"]["p999"] * SAFETY_FACTOR, 6),
            "rationale": f"p99.9 * {SAFETY_FACTOR} safety factor",
        }

    hpbf = results["half_precision"]["bfloat16"]
    if hpbf:
        recommendations["forward_bfloat16"] = {
            "rtol": round(hpbf["max_rel"]["p999"] * SAFETY_FACTOR, 4),
            "atol": round(hpbf["max_abs"]["p999"] * SAFETY_FACTOR, 6),
            "rationale": f"p99.9 * {SAFETY_FACTOR} safety factor",
        }

    return recommendations


def print_recommendations(recommendations: dict) -> None:
    """Print recommended tolerance values."""
    print("\n" + "=" * 60)
    print("RECOMMENDED TOLERANCES (FlashDeformAttn)")
    print("=" * 60)
    print("(Based on p99.9 with 2x safety factor)")
    print()

    for key, rec in recommendations.items():
        print(f"{key}:")
        print(f"  rtol = {rec['rtol']:.2e}")
        print(f"  atol = {rec['atol']:.2e}")
        print(f"  ({rec['rationale']})")
        print()

    # Generate copy-pasteable Python code
    print("\n" + "-" * 60)
    print("COPY-PASTE CODE:")
    print("-" * 60)
    print("# Auto-generated from measure_flash_deform_attn_tolerance.py")
    print("FLASH_DEFORM_ATTN_TOLERANCES = {")
    for key, rec in recommendations.items():
        print(f'    "{key}": {{"rtol": {rec["rtol"]:.2e}, "atol": {rec["atol"]:.2e}}},')
    print("}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure FlashDeformAttn numerical tolerances",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=200,
        help="Number of random seeds per shape (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="flash_deform_attn_tolerance_measurements.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Skip half-precision measurements",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)

    print("FlashDeformAttn Tolerance Measurement")
    print("=" * 60)

    start_time = time.time()

    results = run_measurements(
        num_seeds=args.seeds,
        include_half=not args.no_half,
    )

    elapsed = time.time() - start_time
    results["metadata"]["elapsed_seconds"] = elapsed

    # Save raw results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to: {output_path}")

    # Print summary
    print_summary(results)

    # Generate recommendations
    recommendations = recommend_tolerances(results)
    results["recommendations"] = recommendations
    print_recommendations(recommendations)

    # Update JSON with recommendations
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
