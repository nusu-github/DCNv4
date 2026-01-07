# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""Flash-style Multi-Scale Deformable Attention Autograd Function.

This module provides the PyTorch autograd Function for multi-scale deformable
attention with optimized memory access patterns inspired by FlashAttention.

The deformable attention operation differs from standard attention:
    1. Samples from multiple feature map levels (multi-scale)
    2. Uses a fixed number of sampling points per level (sparse)
    3. Sampling locations are dynamically predicted per query
    4. Uses softmax-normalized attention weights (unlike DCNv4)

Memory Optimization:
    The kernel leverages shared memory to cache frequently accessed data
    (sampling locations, attention weights) and uses vectorized loads for
    efficient memory bandwidth utilization. The shared memory requirements
    vary by GPU architecture (see shm_size_dict).

Supported GPU Architectures:
    - SM 7.0 (V100): 96KB shared memory
    - SM 7.5 (Turing): 64KB shared memory
    - SM 8.0 (A100): 163KB shared memory
    - SM 8.6 (RTX 30xx): 99KB shared memory
    - SM 8.7 (Jetson Orin): 163KB shared memory
    - SM 8.9 (RTX 40xx): 99KB shared memory
    - SM 9.0 (H100): 227KB shared memory
"""

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from dcnv4 import _C

try:
    from dcnv4 import triton_ops as _triton_ops
except Exception:  # pragma: no cover - optional dependency
    _triton_ops = None

# Shared memory capacity per GPU architecture (in bytes)
# Used to determine optimal kernel configurations
shm_size_dict = {
    "8.0": 163000,  # A100
    "8.6": 99000,  # RTX 30xx
    "8.7": 163000,  # Jetson Orin
    "8.9": 99000,  # RTX 40xx
    "9.0": 227000,  # H100
    "7.5": 64000,  # Turing
    "7.0": 96000,  # V100
}

cuda_capability = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"

if cuda_capability not in shm_size_dict:
    raise NotImplementedError

shm_size_cap = shm_size_dict[cuda_capability]


def _use_triton() -> bool:
    """Check if Triton backend should be used.

    Returns True if Triton is available and DCNV4_USE_TRITON=1 is set.
    """
    return (
        _triton_ops is not None
        and _triton_ops.is_available()
        and _triton_ops.use_triton()
    )


def factors(N):
    """Compute all factors of N for kernel configuration."""
    res = []
    for i in range(1, N + 1):
        if N % i == 0:
            res.append(i)
    return res


def findspec(B, Q, G, C):
    """Find optimal kernel configuration for forward pass.

    Args:
        B: Batch size.
        Q: Number of queries.
        G: Number of attention heads (groups).
        C: Channels per head.

    Returns:
        Tuple of (d_stride, n_thread) for kernel launch.

    """
    d_stride = 8
    ms = factors(B * Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


def findspec_bwd(B, Q, G, C):
    """Find optimal kernel configuration for backward pass.

    Args:
        B: Batch size.
        Q: Number of queries.
        G: Number of attention heads (groups).
        C: Channels per head.

    Returns:
        Tuple of (d_stride, n_thread) for backward kernel launch.

    """
    d_stride = 2 if C >= 64 else 1

    ms = factors(B * Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 256:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


class FlashDeformAttnFunction(Function):
    """PyTorch autograd Function for Flash-style Multi-Scale Deformable Attention.

    This function implements efficient multi-scale deformable attention with
    optimized CUDA kernels. It samples features from multiple feature levels
    at dynamically predicted locations and aggregates them with softmax-normalized
    attention weights.

    Key differences from DCNv4Function:
        1. Multi-scale: Samples from L feature levels instead of a single feature map
        2. Softmax: Uses softmax normalization on attention weights
        3. Reference-based: Sampling locations are offsets from reference points
    """

    @staticmethod
    @torch.autocast("cuda", enabled=True, dtype=torch.float16)
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_loc_attn,
        im2col_step,
        K=8,
    ):
        """Forward pass of Flash Deformable Attention.

        Args:
            ctx: Autograd context for saving tensors.
            value: Multi-scale feature values, shape (N, sum(H_l*W_l), n_heads, head_dim).
                All levels are concatenated along the spatial dimension.
            value_spatial_shapes: Spatial dimensions per level, shape (n_levels, 2).
                Each row is [H_l, W_l].
            value_level_start_index: Starting index per level, shape (n_levels,).
            sampling_loc_attn: Combined sampling locations and attention weights,
                shape (N, n_queries, n_heads, n_levels * K * 3).
                Layout: [loc_x, loc_y] * (L*K) + [attn_weight] * (L*K) per head.
            im2col_step: Batch processing step for memory efficiency.
            K: Number of sampling points per level. Default: 8.

        Returns:
            Output features of shape (N, n_queries, n_heads * head_dim).

        """
        ctx.im2col_step = im2col_step
        ctx.K = K
        d_stride, blockthread = findspec(
            value.shape[0],
            sampling_loc_attn.shape[1],
            value.shape[2],
            value.shape[3],
        )
        d_stride_backward, blockthread_backward = findspec_bwd(
            value.shape[0],
            sampling_loc_attn.shape[1],
            value.shape[2],
            value.shape[3],
        )

        ctx.d_stride_backward = d_stride_backward
        ctx.blockthread_backward = blockthread_backward

        if _use_triton():
            output = _triton_ops.flash_deform_attn_forward(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_loc_attn,
                K,
            )
        else:
            output = _C.flash_deform_attn_forward(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_loc_attn,
                ctx.im2col_step,
                K,
                d_stride,
                blockthread,
            )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Backward pass of Flash Deformable Attention.

        Computes gradients with respect to:
            1. Value features (grad_value)
            2. Sampling locations and attention weights (grad_sampling_loc_attn)

        The backward pass accounts for the softmax normalization when computing
        gradients for attention weights.

        Args:
            ctx: Autograd context with saved tensors from forward.
            grad_output: Gradient of loss with respect to output.

        Returns:
            Tuple of (grad_value, None, None, grad_sampling_loc_attn, None, None).

        """
        value, value_spatial_shapes, value_level_start_index, sampling_loc_attn = (
            ctx.saved_tensors
        )
        if _use_triton():
            grad_value, grad_sampling_loc_attn = _triton_ops.flash_deform_attn_backward(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_loc_attn,
                grad_output.contiguous(),
                ctx.K,
            )
        else:
            grad_value, grad_sampling_loc_attn = _C.flash_deform_attn_backward(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_loc_attn,
                grad_output.contiguous(),
                ctx.im2col_step,
                ctx.K,
                ctx.d_stride_backward,
                ctx.blockthread_backward,
            )

        return grad_value, None, None, grad_sampling_loc_attn, None, None
