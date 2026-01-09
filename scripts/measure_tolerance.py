#!/usr/bin/env python3
"""Empirical tolerance measurement for DCNv4 CUDA vs PyTorch reference.

This script measures the error distribution between the CUDA implementation
and the PyTorch reference implementation to establish data-driven tolerance
thresholds.

This is intended to be run ONCE locally (not in CI) to determine appropriate
rtol/atol values. The results should be committed as constants.

Usage:
    python scripts/measure_tolerance.py [--seeds N] [--output FILE]

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

from dcnv4 import ops
from dcnv4.functions import compute_offset_mask_channels
from dcnv4.functions.dcnv4_func import dcnv4_forward, find_spec_bwd
from dcnv4.functions.dcnv4_pytorch import dcnv4_forward_pytorch
from dcnv4.functions.table import TABLE

# =============================================================================
# Configuration
# =============================================================================

# Kernel parameters (matching test suite)
KERNEL = 3
STRIDE = 1
PAD = 1
DILATION = 1

# Shape selection strategy:
# - Include both batch=1 (inference) and batch=64 (training) scenarios
# - Cover various spatial sizes (small 7x7 to large 200x320)
# - Cover different group counts (4-8) and channel widths (16-64)
REPRESENTATIVE_SHAPES: list[tuple[int, int, int, int, int]] = [
    # (B, H, W, G, C)
    # Small spatial, various groups
    (1, 7, 7, 4, 16),
    (1, 7, 7, 8, 32),
    (64, 7, 7, 4, 16),
    (64, 7, 7, 6, 32),
    # Medium spatial
    (1, 14, 14, 4, 16),
    (1, 14, 14, 8, 64),
    (64, 14, 14, 5, 32),
    (64, 28, 28, 4, 16),
    (64, 28, 28, 7, 64),
    # Large spatial (inference-like)
    (1, 50, 80, 4, 16),
    (1, 50, 80, 6, 32),
    (1, 100, 160, 4, 16),
    (1, 100, 160, 8, 64),
    (1, 200, 320, 4, 16),
    (1, 200, 320, 6, 32),
    # Square spatial
    (1, 64, 64, 4, 16),
    (1, 64, 64, 8, 64),
    (64, 56, 56, 4, 16),
    (64, 56, 56, 8, 64),
]


def filter_valid_shapes(
    shapes: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    """Filter shapes to only those present in TABLE."""
    valid = []
    for b, h, w, g, c in shapes:
        key = f"{b}x{h}x{w}x{g}x{c}"
        if key in TABLE:
            valid.append((b, h, w, g, c))
    return valid


VALID_SHAPES = filter_valid_shapes(REPRESENTATIVE_SHAPES)


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

    shape: tuple[int, int, int, int, int]
    forward_errors: list[ErrorStats] = field(default_factory=list)
    grad_input_errors: list[ErrorStats] = field(default_factory=list)
    grad_offset_errors: list[ErrorStats] = field(default_factory=list)
    backward_nondeterminism_input: list[float] = field(default_factory=list)
    backward_nondeterminism_offset: list[float] = field(default_factory=list)


@dataclass
class HalfPrecisionMeasurement:
    """Measurements for half-precision types."""

    dtype_name: str
    shape: tuple[int, int, int, int, int]
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


def run_cuda_forward(
    input_tensor: torch.Tensor,
    offset_mask: torch.Tensor,
    group: int,
    group_channels: int,
    remove_center: int,
    *,
    softmax: bool = False,
) -> torch.Tensor:
    """Run CUDA forward pass."""
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
        64,  # im2col_step
        remove_center,
        softmax,
    )


def run_ref_forward(
    input_tensor: torch.Tensor,
    offset_mask: torch.Tensor,
    group: int,
    group_channels: int,
    remove_center: int,
    *,
    softmax: bool = False,
) -> torch.Tensor:
    """Run PyTorch reference forward pass."""
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
        64,  # im2col_step (ignored in reference)
        remove_center,
        softmax,
    )


def run_cuda_backward(
    input_tensor: torch.Tensor,
    offset_mask: torch.Tensor,
    grad_output: torch.Tensor,
    group: int,
    group_channels: int,
    remove_center: int,
    *,
    softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run CUDA backward pass directly."""
    backward_d_stride, backward_block_thread = find_spec_bwd(
        input_tensor.shape[0],
        input_tensor.shape[1],
        input_tensor.shape[2],
        group,
        group_channels,
    )

    return ops.dcnv4_backward(
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
        64,  # im2col_step
        grad_output.contiguous(),
        remove_center,
        backward_d_stride,
        backward_block_thread,
        softmax,
    )


# =============================================================================
# Measurement functions
# =============================================================================


def measure_forward_error(
    shape: tuple[int, int, int, int, int],
    seed: int,
    remove_center: int = 0,
    softmax: bool = False,
) -> ErrorStats:
    """Measure forward pass error for a single seed."""
    b, h, w, group, group_channels = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    c_total = group * group_channels
    offset_mask_channels = compute_offset_mask_channels(group, KERNEL, remove_center)

    input_tensor = torch.randn(b, h, w, c_total, device="cuda", dtype=torch.float32)
    offset_mask = torch.randn(
        b,
        h,
        w,
        offset_mask_channels,
        device="cuda",
        dtype=torch.float32,
    )

    with torch.no_grad():
        out_cuda = run_cuda_forward(
            input_tensor,
            offset_mask,
            group,
            group_channels,
            remove_center,
            softmax=softmax,
        )
        out_ref = run_ref_forward(
            input_tensor,
            offset_mask,
            group,
            group_channels,
            remove_center,
            softmax=softmax,
        )

    return compute_error_stats(out_cuda, out_ref)


def measure_gradient_error(
    shape: tuple[int, int, int, int, int],
    seed: int,
    remove_center: int = 0,
    softmax: bool = False,
) -> tuple[ErrorStats, ErrorStats]:
    """Measure gradient error for a single seed.

    Returns (grad_input_error, grad_offset_error).
    """
    b, h, w, group, group_channels = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    c_total = group * group_channels
    offset_mask_channels = compute_offset_mask_channels(group, KERNEL, remove_center)

    # CUDA path
    input_cuda = torch.randn(
        b,
        h,
        w,
        c_total,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    offset_cuda = torch.randn(
        b,
        h,
        w,
        offset_mask_channels,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    # Reference path (same data)
    input_ref = input_cuda.detach().clone().requires_grad_(True)
    offset_ref = offset_cuda.detach().clone().requires_grad_(True)

    # Forward
    out_cuda = run_cuda_forward(
        input_cuda,
        offset_cuda,
        group,
        group_channels,
        remove_center,
        softmax=softmax,
    )
    out_ref = run_ref_forward(
        input_ref,
        offset_ref,
        group,
        group_channels,
        remove_center,
        softmax=softmax,
    )

    # Backward with same grad_output
    torch.manual_seed(seed + 1000000)  # Different seed for grad
    grad_output = torch.randn_like(out_cuda)

    out_cuda.backward(grad_output)
    out_ref.backward(grad_output)

    assert input_cuda.grad is not None
    assert offset_cuda.grad is not None
    assert input_ref.grad is not None
    assert offset_ref.grad is not None

    grad_input_err = compute_error_stats(input_cuda.grad, input_ref.grad)
    grad_offset_err = compute_error_stats(offset_cuda.grad, offset_ref.grad)

    return grad_input_err, grad_offset_err


def measure_backward_nondeterminism(
    shape: tuple[int, int, int, int, int],
    seed: int,
    remove_center: int = 0,
    softmax: bool = False,
    num_runs: int = 5,
) -> tuple[float, float]:
    """Measure backward pass non-determinism.

    Runs the same backward pass multiple times and measures the max difference.
    Returns (max_diff_input, max_diff_offset).
    """
    b, h, w, group, group_channels = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    c_total = group * group_channels
    offset_mask_channels = compute_offset_mask_channels(group, KERNEL, remove_center)

    input_tensor = torch.randn(b, h, w, c_total, device="cuda", dtype=torch.float32)
    offset_mask = torch.randn(
        b,
        h,
        w,
        offset_mask_channels,
        device="cuda",
        dtype=torch.float32,
    )

    # Generate output shape by running forward once
    with torch.no_grad():
        out = run_cuda_forward(
            input_tensor,
            offset_mask,
            group,
            group_channels,
            remove_center,
            softmax=softmax,
        )
    grad_output = torch.randn_like(out)

    # Run backward multiple times
    grad_inputs: list[torch.Tensor] = []
    grad_offsets: list[torch.Tensor] = []

    for _ in range(num_runs):
        grad_input, grad_offset = run_cuda_backward(
            input_tensor,
            offset_mask,
            grad_output,
            group,
            group_channels,
            remove_center,
            softmax=softmax,
        )
        grad_inputs.append(grad_input.clone())
        grad_offsets.append(grad_offset.clone())

    # Compute max pairwise difference
    max_diff_input = 0.0
    max_diff_offset = 0.0

    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            diff_input = (grad_inputs[i] - grad_inputs[j]).abs().max().item()
            diff_offset = (grad_offsets[i] - grad_offsets[j]).abs().max().item()
            max_diff_input = max(max_diff_input, diff_input)
            max_diff_offset = max(max_diff_offset, diff_offset)

    return max_diff_input, max_diff_offset


def measure_half_precision_error(
    shape: tuple[int, int, int, int, int],
    seed: int,
    dtype: torch.dtype,
    remove_center: int = 0,
) -> ErrorStats:
    """Measure forward pass error for half-precision types."""
    b, h, w, group, group_channels = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    c_total = group * group_channels
    offset_mask_channels = compute_offset_mask_channels(group, KERNEL, remove_center)

    input_tensor = torch.randn(b, h, w, c_total, device="cuda", dtype=dtype)
    offset_mask = torch.randn(b, h, w, offset_mask_channels, device="cuda", dtype=dtype)

    with torch.no_grad():
        out_cuda = run_cuda_forward(
            input_tensor,
            offset_mask,
            group,
            group_channels,
            remove_center,
        )
        out_ref = run_ref_forward(
            input_tensor,
            offset_mask,
            group,
            group_channels,
            remove_center,
        )

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
    num_seeds: int = 500,
    include_softmax: bool = True,
    include_half: bool = True,
) -> dict:
    """Run all measurements and return results."""
    results: dict = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_seeds": num_seeds,
            "shapes": [list(s) for s in VALID_SHAPES],
            "kernel": KERNEL,
            "stride": STRIDE,
            "pad": PAD,
            "dilation": DILATION,
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "N/A",
            "pytorch_version": torch.__version__,
        },
        "forward": {"no_softmax": [], "softmax": []},
        "gradient": {"input": [], "offset": []},
        "nondeterminism": {"input": [], "offset": []},
        "half_precision": {"float16": [], "bfloat16": []},
    }

    total_shapes = len(VALID_SHAPES)
    total_iterations = total_shapes * num_seeds

    print(f"Running measurements with {num_seeds} seeds across {total_shapes} shapes")
    print(f"Total iterations: {total_iterations}")
    print(f"CUDA device: {results['metadata']['cuda_device']}")
    print("-" * 60)

    # Forward pass measurements (no softmax)
    print("\n[1/6] Measuring forward pass errors (no softmax)...")
    forward_stats_no_softmax: list[ErrorStats] = []
    for shape_idx, shape in enumerate(VALID_SHAPES):
        shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}"
        for seed in range(num_seeds):
            try:
                stats = measure_forward_error(
                    shape,
                    seed,
                    remove_center=0,
                    softmax=False,
                )
                forward_stats_no_softmax.append(stats)
            except RuntimeError as e:
                print(f"  Skip {shape_key} seed={seed}: {e}")
                continue

            if (seed + 1) % 100 == 0:
                print(
                    f"  Shape {shape_idx + 1}/{total_shapes} ({shape_key}): {seed + 1}/{num_seeds} seeds",
                )

    results["forward"]["no_softmax"] = aggregate_stats(forward_stats_no_softmax)

    # Forward pass measurements (with softmax)
    if include_softmax:
        print("\n[2/6] Measuring forward pass errors (with softmax)...")
        forward_stats_softmax: list[ErrorStats] = []
        for shape_idx, shape in enumerate(VALID_SHAPES):
            shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}"
            for seed in range(num_seeds):
                try:
                    stats = measure_forward_error(
                        shape,
                        seed,
                        remove_center=0,
                        softmax=True,
                    )
                    forward_stats_softmax.append(stats)
                except RuntimeError as e:
                    print(f"  Skip {shape_key} seed={seed}: {e}")
                    continue

                if (seed + 1) % 100 == 0:
                    print(
                        f"  Shape {shape_idx + 1}/{total_shapes} ({shape_key}): {seed + 1}/{num_seeds} seeds",
                    )

        results["forward"]["softmax"] = aggregate_stats(forward_stats_softmax)
    else:
        print("\n[2/6] Skipping softmax measurements")

    # Gradient measurements
    print("\n[3/6] Measuring gradient errors...")
    grad_input_stats: list[ErrorStats] = []
    grad_offset_stats: list[ErrorStats] = []
    for shape_idx, shape in enumerate(VALID_SHAPES):
        shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}"
        for seed in range(num_seeds):
            try:
                input_err, offset_err = measure_gradient_error(
                    shape,
                    seed,
                    remove_center=0,
                )
                grad_input_stats.append(input_err)
                grad_offset_stats.append(offset_err)
            except RuntimeError as e:
                print(f"  Skip {shape_key} seed={seed}: {e}")
                continue

            if (seed + 1) % 100 == 0:
                print(
                    f"  Shape {shape_idx + 1}/{total_shapes} ({shape_key}): {seed + 1}/{num_seeds} seeds",
                )

    results["gradient"]["input"] = aggregate_stats(grad_input_stats)
    results["gradient"]["offset"] = aggregate_stats(grad_offset_stats)

    # Non-determinism measurements (fewer seeds, multiple runs per seed)
    print("\n[4/6] Measuring backward non-determinism...")
    nondeterminism_input: list[float] = []
    nondeterminism_offset: list[float] = []
    nondet_seeds = min(num_seeds, 100)  # Limit for non-determinism test
    for shape_idx, shape in enumerate(VALID_SHAPES):
        shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}"
        for seed in range(nondet_seeds):
            try:
                diff_input, diff_offset = measure_backward_nondeterminism(
                    shape,
                    seed,
                    remove_center=0,
                    num_runs=5,
                )
                nondeterminism_input.append(diff_input)
                nondeterminism_offset.append(diff_offset)
            except RuntimeError as e:
                print(f"  Skip {shape_key} seed={seed}: {e}")
                continue

            if (seed + 1) % 50 == 0:
                print(
                    f"  Shape {shape_idx + 1}/{total_shapes} ({shape_key}): {seed + 1}/{nondet_seeds} seeds",
                )

    results["nondeterminism"]["input"] = {
        "max": max(nondeterminism_input) if nondeterminism_input else 0,
        "mean": sum(nondeterminism_input) / len(nondeterminism_input)
        if nondeterminism_input
        else 0,
        "num_samples": len(nondeterminism_input),
    }
    results["nondeterminism"]["offset"] = {
        "max": max(nondeterminism_offset) if nondeterminism_offset else 0,
        "mean": sum(nondeterminism_offset) / len(nondeterminism_offset)
        if nondeterminism_offset
        else 0,
        "num_samples": len(nondeterminism_offset),
    }

    # Half-precision measurements
    if include_half:
        half_seeds = min(num_seeds, 200)  # Fewer seeds for half precision

        print("\n[5/6] Measuring float16 errors...")
        fp16_stats: list[ErrorStats] = []
        for shape_idx, shape in enumerate(VALID_SHAPES):
            shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}"
            for seed in range(half_seeds):
                try:
                    stats = measure_half_precision_error(shape, seed, torch.float16)
                    fp16_stats.append(stats)
                except RuntimeError as e:
                    print(f"  Skip {shape_key} seed={seed}: {e}")
                    continue

                if (seed + 1) % 100 == 0:
                    print(
                        f"  Shape {shape_idx + 1}/{total_shapes} ({shape_key}): {seed + 1}/{half_seeds} seeds",
                    )

        results["half_precision"]["float16"] = aggregate_stats(fp16_stats)

        print("\n[6/6] Measuring bfloat16 errors...")
        if torch.cuda.is_bf16_supported():
            bf16_stats: list[ErrorStats] = []
            for shape_idx, shape in enumerate(VALID_SHAPES):
                shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}"
                for seed in range(half_seeds):
                    try:
                        stats = measure_half_precision_error(
                            shape,
                            seed,
                            torch.bfloat16,
                        )
                        bf16_stats.append(stats)
                    except RuntimeError as e:
                        print(f"  Skip {shape_key} seed={seed}: {e}")
                        continue

                    if (seed + 1) % 100 == 0:
                        print(
                            f"  Shape {shape_idx + 1}/{total_shapes} ({shape_key}): {seed + 1}/{half_seeds} seeds",
                        )

            results["half_precision"]["bfloat16"] = aggregate_stats(bf16_stats)
        else:
            print("  bfloat16 not supported on this device, skipping")
            results["half_precision"]["bfloat16"] = {}
    else:
        print("\n[5/6] Skipping half-precision measurements")
        print("[6/6] Skipping half-precision measurements")

    return results


def print_summary(results: dict) -> None:
    """Print a human-readable summary of results."""
    print("\n" + "=" * 60)
    print("MEASUREMENT SUMMARY")
    print("=" * 60)

    print(f"\nDevice: {results['metadata']['cuda_device']}")
    print(f"Seeds: {results['metadata']['num_seeds']}")
    print(f"Shapes tested: {len(results['metadata']['shapes'])}")

    print("\n" + "-" * 60)
    print("FORWARD PASS (no softmax)")
    print("-" * 60)
    fwd = results["forward"]["no_softmax"]
    if fwd:
        print(
            f"  Max absolute error:  max={fwd['max_abs']['max']:.2e}, p99.9={fwd['max_abs']['p999']:.2e}",
        )
        print(
            f"  Max relative error:  max={fwd['max_rel']['max']:.2e}, p99.9={fwd['max_rel']['p999']:.2e}",
        )

    print("\n" + "-" * 60)
    print("FORWARD PASS (with softmax)")
    print("-" * 60)
    fwd_sm = results["forward"]["softmax"]
    if fwd_sm:
        print(
            f"  Max absolute error:  max={fwd_sm['max_abs']['max']:.2e}, p99.9={fwd_sm['max_abs']['p999']:.2e}",
        )
        print(
            f"  Max relative error:  max={fwd_sm['max_rel']['max']:.2e}, p99.9={fwd_sm['max_rel']['p999']:.2e}",
        )

    print("\n" + "-" * 60)
    print("GRADIENT (input)")
    print("-" * 60)
    gi = results["gradient"]["input"]
    if gi:
        print(
            f"  Max absolute error:  max={gi['max_abs']['max']:.2e}, p99.9={gi['max_abs']['p999']:.2e}",
        )
        print(
            f"  Max relative error:  max={gi['max_rel']['max']:.2e}, p99.9={gi['max_rel']['p999']:.2e}",
        )

    print("\n" + "-" * 60)
    print("GRADIENT (offset)")
    print("-" * 60)
    go = results["gradient"]["offset"]
    if go:
        print(
            f"  Max absolute error:  max={go['max_abs']['max']:.2e}, p99.9={go['max_abs']['p999']:.2e}",
        )
        print(
            f"  Max relative error:  max={go['max_rel']['max']:.2e}, p99.9={go['max_rel']['p999']:.2e}",
        )

    print("\n" + "-" * 60)
    print("BACKWARD NON-DETERMINISM (atomicAdd)")
    print("-" * 60)
    nd_i = results["nondeterminism"]["input"]
    nd_o = results["nondeterminism"]["offset"]
    if nd_i:
        print(f"  grad_input:  max={nd_i['max']:.2e}, mean={nd_i['mean']:.2e}")
    if nd_o:
        print(f"  grad_offset: max={nd_o['max']:.2e}, mean={nd_o['mean']:.2e}")

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
    fwd = results["forward"]["no_softmax"]
    if fwd:
        recommendations["forward_float32"] = {
            "rtol": round(fwd["max_rel"]["p999"] * SAFETY_FACTOR, 6),
            "atol": round(fwd["max_abs"]["p999"] * SAFETY_FACTOR, 8),
            "rationale": f"p99.9 * {SAFETY_FACTOR} safety factor",
        }

    # Forward pass with softmax
    fwd_sm = results["forward"]["softmax"]
    if fwd_sm:
        recommendations["forward_softmax_float32"] = {
            "rtol": round(fwd_sm["max_rel"]["p999"] * SAFETY_FACTOR, 6),
            "atol": round(fwd_sm["max_abs"]["p999"] * SAFETY_FACTOR, 8),
            "rationale": f"p99.9 * {SAFETY_FACTOR} safety factor",
        }

    # Gradients (need higher tolerance due to accumulation + atomicAdd)
    gi = results["gradient"]["input"]
    go = results["gradient"]["offset"]
    nd_i = results["nondeterminism"]["input"]
    nd_o = results["nondeterminism"]["offset"]

    if gi and nd_i:
        # Account for non-determinism as lower bound
        nondet_floor = nd_i["max"] * 2 if nd_i["max"] > 0 else 0
        recommendations["grad_input_float32"] = {
            "rtol": round(max(gi["max_rel"]["p999"] * SAFETY_FACTOR, 1e-4), 6),
            "atol": round(max(gi["max_abs"]["p999"] * SAFETY_FACTOR, nondet_floor), 8),
            "rationale": f"p99.9 * {SAFETY_FACTOR}, floor from non-determinism",
        }

    if go and nd_o:
        nondet_floor = nd_o["max"] * 2 if nd_o["max"] > 0 else 0
        recommendations["grad_offset_float32"] = {
            "rtol": round(max(go["max_rel"]["p999"] * SAFETY_FACTOR, 1e-4), 6),
            "atol": round(max(go["max_abs"]["p999"] * SAFETY_FACTOR, nondet_floor), 8),
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
    print("RECOMMENDED TOLERANCES")
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
    print("# Auto-generated from measure_tolerance.py")
    print("TOLERANCES = {")
    for key, rec in recommendations.items():
        print(f'    "{key}": {{"rtol": {rec["rtol"]:.2e}, "atol": {rec["atol"]:.2e}}},')
    print("}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure DCNv4 numerical tolerances")
    parser.add_argument(
        "--seeds",
        type=int,
        default=500,
        help="Number of random seeds per shape (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tolerance_measurements.json",
        help="Output JSON file path (default: tolerance_measurements.json)",
    )
    parser.add_argument(
        "--no-softmax",
        action="store_true",
        help="Skip softmax measurements",
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

    print("DCNv4 Tolerance Measurement")
    print("=" * 60)

    start_time = time.time()

    results = run_measurements(
        num_seeds=args.seeds,
        include_softmax=not args.no_softmax,
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
