"""Common utilities for Triton kernel implementations.

This module provides shared helper functions used by both the DCNv4 and
FlashDeformAttn Triton kernels, including:

    - Output size computation following PyTorch convolution conventions
    - Power-of-2 padding for efficient memory alignment
    - Autotuning configurations for Triton kernel optimization
"""

from __future__ import annotations

import triton

triton.testing


def _next_power_of_2(x: int) -> int:
    """Round up to the next power of 2.

    Used for padding block sizes to enable efficient memory access patterns
    and better utilization of GPU resources.

    Args:
        x: Input integer.

    Returns:
        Smallest power of 2 >= x. Returns 1 for x <= 1.

    Example:
        >>> _next_power_of_2(5)
        8
        >>> _next_power_of_2(16)
        16

    """
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _choose_block_d(channels: int) -> int:
    """Heuristic block size for channel tiling."""
    if channels <= 16:
        return 16
    if channels <= 32:
        return 32
    if channels <= 64:
        return 64
    if channels <= 128:
        return 128
    return 256


def _compute_output_hw(
    height_in: int,
    width_in: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
) -> tuple[int, int]:
    """Compute output spatial dimensions for a convolution operation.

    Follows the standard PyTorch convolution output size formula:
        out = floor((in + 2*pad - dilation*(kernel-1) - 1) / stride + 1)

    Args:
        height_in: Input height.
        width_in: Input width.
        kernel_h: Kernel height.
        kernel_w: Kernel width.
        stride_h: Stride in height dimension.
        stride_w: Stride in width dimension.
        pad_h: Padding in height dimension.
        pad_w: Padding in width dimension.
        dilation_h: Dilation in height dimension.
        dilation_w: Dilation in width dimension.

    Returns:
        Tuple of (height_out, width_out).

    """
    height_out = (
        height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)
    ) // stride_h + 1
    width_out = (
        width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)
    ) // stride_w + 1
    return height_out, width_out


def _get_autotune_config():
    """Generate Triton autotuning configurations.

    Creates a grid of configurations to search over during autotuning,
    varying block size, warp count, and pipeline stages.

    The autotuner will benchmark each configuration and select the fastest
    for the given input dimensions.

    Returns:
        List of triton.Config objects covering:
            - BLOCK_Q: 16, 32, 64, 128 (queries per thread block)
            - num_warps: 4, 8 (CUDA warps per thread block)
            - num_stages: 2, 3 (software pipelining stages)

    """
    configs = []
    for block_q in [16, 32, 64, 128]:
        for num_warps in [4, 8]:
            for num_stages in [2, 3]:
                configs.append(
                    triton.Config(
                        {"BLOCK_Q": block_q},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    ),
                )
    return configs


def _prune_block_configs(configs, q: int, d: int, block_d: int):
    """Prune autotune configs using lightweight size heuristics."""
    max_block_q = max(16, min(128, _next_power_of_2(q)))
    max_block_d = max(16, min(256, _next_power_of_2(d)))

    pruned = []
    for cfg in configs:
        block_q = cfg.kwargs.get("BLOCK_Q")
        if block_q is None:
            pruned.append(cfg)
            continue

        if block_q > max_block_q:
            continue
        if block_d > max_block_d:
            continue

        tile = block_q * block_d
        if tile <= 1024 and cfg.num_warps > 4:
            continue
        if tile >= 4096 and cfg.num_warps < 8:
            continue

        pruned.append(cfg)

    return pruned or configs


def _prune_configs_dcnv4(configs, _nargs, **kwargs):
    if "H_out" not in kwargs or "W_out" not in kwargs or "D" not in kwargs:
        return configs
    q = int(kwargs["H_out"]) * int(kwargs["W_out"])
    d = int(kwargs["D"])
    block_d = _choose_block_d(d)
    return _prune_block_configs(configs, q, d, block_d)


def _prune_configs_flash(configs, _nargs, **kwargs):
    if "Q" not in kwargs or "D" not in kwargs:
        return configs
    q = int(kwargs["Q"])
    d = int(kwargs["D"])
    block_d = _choose_block_d(d)
    return _prune_block_configs(configs, q, d, block_d)
