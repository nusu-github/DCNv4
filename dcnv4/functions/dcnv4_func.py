# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
"""DCNv4 Autograd Function Implementation.

This module provides the PyTorch Custom Op for DCNv4, handling both
forward and backward passes through optimized CUDA or Triton kernels.

Key Optimizations (from DCNv4 paper):
    1. Thread Reuse: Channels within each group share sampling offsets and
       aggregation weights. Instead of using separate threads for each channel,
       DCNv4 processes multiple channels per thread, eliminating redundant
       memory reads for offset/weight values.

    2. Vectorized Memory Access: When processing D' channels per thread, memory
       loads/stores are vectorized (e.g., loading 4 floats with a single 128-bit
       instruction), reducing memory instruction count.

    3. Reduced Bilinear Interpolation: Since channels in a group share the same
       sampling locations, bilinear interpolation coefficients are computed once
       per group rather than per channel.

Kernel Configuration:
    The module includes lookup tables (TABLE, BWDTABLE) with pre-tuned kernel
    configurations for common input shapes. For unseen shapes, configurations
    are computed dynamically based on the input dimensions.

    Key parameters:
        - d_stride: Number of channels processed per thread (for vectorization)
        - n_thread: Total number of threads per CUDA block
"""

import os

import torch
import torch.library

from dcnv4 import ops

from .dcnv4_triton import (
    dcnv4_backward_triton,
    dcnv4_forward_triton,
    is_triton_available,
    triton_supports,
)
from .table import BWDTABLE, TABLE
from .utils import factors


def findspec(B, H, W, G, C):
    """Find optimal CUDA kernel configuration for forward pass."""
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in TABLE:
        return TABLE[key][0], TABLE[key][1]

    d_stride = 8
    ms = factors(B * H * W)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    TABLE[key] = (d_stride, n_thread)
    return d_stride, n_thread


def find_spec_bwd(B, H, W, G, C):
    """Find optimal CUDA kernel configuration for backward pass."""
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in BWDTABLE:
        return BWDTABLE[key][0], BWDTABLE[key][1]

    d_stride = 2 if C >= 64 else 1
    ms = factors(B * H * W)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 256:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


def _env_flag_set(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _use_triton(
    input: torch.Tensor,
    kernel_h: int,
    kernel_w: int,
    remove_center: int,
) -> bool:
    if not _env_flag_set("DCNV4_USE_TRITON"):
        return False
    if not input.is_cuda:
        return False
    if not is_triton_available():
        return False
    return triton_supports(kernel_h, kernel_w, remove_center)


@torch.library.custom_op("dcnv4::dcnv4_forward", mutates_args=())
def dcnv4_forward(
    input: torch.Tensor,
    offset_mask: torch.Tensor,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    group: int,
    group_channels: int,
    offset_scale: float,
    im2col_step: int,
    remove_center: int,
    softmax: bool = False,
) -> torch.Tensor:
    """Forward pass of DCNv4."""
    if _use_triton(input, kernel_h, kernel_w, remove_center):
        return dcnv4_forward_triton(
            input,
            offset_mask,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            group_channels,
            offset_scale,
            im2col_step,
            remove_center,
            softmax,
        )

    forward_d_stride, forward_block_thread = findspec(
        input.shape[0],
        input.shape[1],
        input.shape[2],
        group,
        group_channels,
    )

    return ops.dcnv4_forward(
        input,
        offset_mask,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        group,
        group_channels,
        offset_scale,
        im2col_step,
        remove_center,
        forward_d_stride,
        forward_block_thread,
        softmax,
    )


@dcnv4_forward.register_fake
def _(
    input,
    offset_mask,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    group,
    group_channels,
    offset_scale,
    im2col_step,
    remove_center,
    softmax=False,
):
    N, H_in, W_in, _ = input.shape
    H_out = (H_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    return input.new_empty(N, H_out, W_out, group * group_channels)


def dcnv4_backward(ctx, grad_output):
    input, offset_mask = ctx.saved_tensors
    kernel_h = ctx.kernel_h
    kernel_w = ctx.kernel_w
    stride_h = ctx.stride_h
    stride_w = ctx.stride_w
    pad_h = ctx.pad_h
    pad_w = ctx.pad_w
    dilation_h = ctx.dilation_h
    dilation_w = ctx.dilation_w
    group = ctx.group
    group_channels = ctx.group_channels
    offset_scale = ctx.offset_scale
    im2col_step = ctx.im2col_step
    remove_center = ctx.remove_center
    softmax = ctx.softmax

    if _use_triton(input, kernel_h, kernel_w, remove_center):
        grad_input, grad_offset_mask = dcnv4_backward_triton(
            input,
            offset_mask,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            group_channels,
            offset_scale,
            im2col_step,
            grad_output.contiguous(),
            remove_center,
            softmax,
        )
    else:
        backward_d_stride, backward_block_thread = find_spec_bwd(
            input.shape[0],
            input.shape[1],
            input.shape[2],
            group,
            group_channels,
        )

        grad_input, grad_offset_mask = ops.dcnv4_backward(
            input,
            offset_mask,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            group_channels,
            offset_scale,
            im2col_step,
            grad_output.contiguous(),
            remove_center,
            backward_d_stride,
            backward_block_thread,
            softmax,
        )

    return (
        grad_input,
        grad_offset_mask,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def setup_context(ctx, inputs, output) -> None:
    input, offset_mask = inputs[:2]
    ctx.save_for_backward(input, offset_mask)
    ctx.kernel_h = inputs[2]
    ctx.kernel_w = inputs[3]
    ctx.stride_h = inputs[4]
    ctx.stride_w = inputs[5]
    ctx.pad_h = inputs[6]
    ctx.pad_w = inputs[7]
    ctx.dilation_h = inputs[8]
    ctx.dilation_w = inputs[9]
    ctx.group = inputs[10]
    ctx.group_channels = inputs[11]
    ctx.offset_scale = inputs[12]
    ctx.im2col_step = inputs[13]
    ctx.remove_center = inputs[14]
    ctx.softmax = inputs[15]


dcnv4_forward.register_autograd(dcnv4_backward, setup_context=setup_context)
