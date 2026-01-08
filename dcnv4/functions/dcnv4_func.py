# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
"""DCNv4 Autograd Function Implementation.

This module provides the PyTorch autograd Function for DCNv4, handling both
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

import torch
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try:
    from dcnv4 import triton_ops as _triton_ops
except Exception:  # pragma: no cover - optional dependency
    _triton_ops = None

from .table import BWDTABLE, TABLE


def factors(N):
    """Compute all factors of N.

    Used to find valid multipliers for thread block configuration.

    Args:
        N: Integer to factorize.

    Returns:
        List of all factors of N in ascending order.

    """
    res = []
    for i in range(1, N + 1):
        if N % i == 0:
            res.append(i)
    return res


def _use_triton() -> bool:
    """Check if Triton backend should be used.

    Returns True if:
        1. Triton is available (import succeeded)
        2. Triton kernels are implemented
        3. DCNV4_USE_TRITON environment variable is set to "1"
    """
    return (
        _triton_ops is not None
        and _triton_ops.is_available()
        and _triton_ops.use_triton()
    )


def findspec(B, H, W, G, C):
    """Find optimal CUDA kernel configuration for forward pass.

    Determines the d_stride (channels per thread) and n_thread (threads per block)
    parameters for the DCNv4 forward kernel based on input dimensions.

    Args:
        B: Batch size.
        H: Feature map height.
        W: Feature map width.
        G: Number of groups.
        C: Channels per group (group_channels).

    Returns:
        Tuple of (d_stride, n_thread) for kernel launch configuration.
        Results are cached in TABLE for future lookups.

    """
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in TABLE:
        return TABLE[key][0], TABLE[key][1]

    # Default stride for vectorized loads (8 channels = 256 bits for fp32)
    d_stride = 8
    ms = factors(B * H * W)
    multiplier = 1
    # Find largest multiplier that keeps thread count reasonable
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    key = f"{B}x{H}x{W}x{G}x{C}"
    TABLE[key] = (d_stride, n_thread)
    return d_stride, n_thread


def find_spec_bwd(B, H, W, G, C):
    """Find optimal CUDA kernel configuration for backward pass.

    Similar to findspec but with different constraints for the backward kernel,
    which has different memory access patterns and computational requirements.

    Args:
        B: Batch size.
        H: Feature map height.
        W: Feature map width.
        G: Number of groups.
        C: Channels per group (group_channels).

    Returns:
        Tuple of (d_stride, n_thread) for backward kernel launch configuration.

    """
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in BWDTABLE:
        return BWDTABLE[key][0], BWDTABLE[key][1]

    # Smaller stride for backward (more atomic operations)
    d_stride = 2 if C >= 64 else 1

    ms = factors(B * H * W)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 256:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


class DCNv4Function(Function):
    """PyTorch autograd Function for Deformable Convolution v4.

    This function wraps the optimized CUDA/Triton kernels for DCNv4 and handles
    gradient computation for backpropagation.

    The DCNv4 operation can be summarized as:
        1. For each output location p_0, compute K sampling locations using
           base grid positions p_k plus learned offsets delta_p_k
        2. Sample input features at these locations using bilinear interpolation
        3. Aggregate sampled values using learned (unbounded) weights m_k

    Unlike DCNv3, DCNv4 does NOT apply softmax to the aggregation weights,
    allowing unbounded values similar to standard convolutions.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
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
    ):
        """Forward pass of DCNv4.

        Args:
            ctx: Autograd context for saving tensors needed in backward.
            input: Input feature map of shape (N, H, W, C).
            offset_mask: Combined offset and aggregation weights tensor of shape
                (N, H_out, W_out, G * K * 3) where K = kernel_h * kernel_w.
                Layout: [offset_x, offset_y] * K + [weight] * K for each group.
            kernel_h: Kernel height.
            kernel_w: Kernel width.
            stride_h: Stride in height dimension.
            stride_w: Stride in width dimension.
            pad_h: Padding in height dimension.
            pad_w: Padding in width dimension.
            dilation_h: Dilation in height dimension.
            dilation_w: Dilation in width dimension.
            group: Number of groups for grouped aggregation.
            group_channels: Channels per group (= total_channels // group).
            offset_scale: Scaling factor for sampling offsets.
            im2col_step: Batch processing step size (for memory efficiency).
            remove_center: If 1, excludes center point from sampling grid.

        Returns:
            Output feature map of shape (N, H_out, W_out, C).

        """
        forward_d_stride, forward_block_thread = findspec(
            input.shape[0],
            input.shape[1],
            input.shape[2],
            group,
            group_channels,
        )
        backward_d_stride, backward_block_thread = find_spec_bwd(
            input.shape[0],
            input.shape[1],
            input.shape[2],
            group,
            group_channels,
        )

        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center
        ctx.backward_d_stride = backward_d_stride
        ctx.backward_block_thread = backward_block_thread

        args = [
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
            ctx.im2col_step,
            remove_center,
            forward_d_stride,
            forward_block_thread,
            False,
        ]

        if _use_triton():
            output = _triton_ops.dcnv4_forward(
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
                remove_center,
                False,
            )
        else:
            output = torch.ops.dcnv4_C.dcnv4_forward(*args)
        ctx.save_for_backward(input, offset_mask)

        return output

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        """Backward pass of DCNv4.

        Computes gradients with respect to:
            1. Input features (grad_input)
            2. Sampling offsets and aggregation weights (grad_offset_mask)

        The backward kernel uses atomic operations for grad_input since multiple
        output locations may sample from the same input location.

        Args:
            ctx: Autograd context with saved tensors from forward.
            grad_output: Gradient of loss with respect to output, shape (N, H_out, W_out, C).

        Returns:
            Tuple of gradients: (grad_input, grad_offset_mask, None, None, ...)
            where None corresponds to non-tensor parameters.

        """
        input, offset_mask = ctx.saved_tensors

        args = [
            input,
            offset_mask,
            ctx.kernel_h,
            ctx.kernel_w,
            ctx.stride_h,
            ctx.stride_w,
            ctx.pad_h,
            ctx.pad_w,
            ctx.dilation_h,
            ctx.dilation_w,
            ctx.group,
            ctx.group_channels,
            ctx.offset_scale,
            ctx.im2col_step,
            grad_output.contiguous(),
            ctx.remove_center,
            ctx.backward_d_stride,
            ctx.backward_block_thread,
            False,
        ]

        if _use_triton():
            grad_input, grad_offset_mask = _triton_ops.dcnv4_backward(
                input,
                offset_mask,
                grad_output.contiguous(),
                ctx.kernel_h,
                ctx.kernel_w,
                ctx.stride_h,
                ctx.stride_w,
                ctx.pad_h,
                ctx.pad_w,
                ctx.dilation_h,
                ctx.dilation_w,
                ctx.group,
                ctx.group_channels,
                ctx.offset_scale,
                ctx.remove_center,
                False,
            )
        else:
            grad_input, grad_offset_mask = torch.ops.dcnv4_C.dcnv4_backward(*args)

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
        )
