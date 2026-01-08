# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""Flash-style Multi-Scale Deformable Attention Autograd Function."""

import torch
import torch.library

from dcnv4 import _ops

# Shared memory capacity per GPU architecture (in bytes)
shm_size_dict = {
    "8.0": 163000,  # A100
    "8.6": 99000,  # RTX 30xx
    "8.7": 163000,  # Jetson Orin
    "8.9": 99000,  # RTX 40xx
    "9.0": 227000,  # H100
    "7.5": 64000,  # Turing
    "7.0": 96000,  # V100
}

if torch.cuda.is_available():
    cuda_capability = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
    shm_size_cap = shm_size_dict.get(cuda_capability, 96000)
else:
    shm_size_cap = 96000


def factors(N):
    """Compute all factors of N."""
    res = []
    for i in range(1, N + 1):
        if N % i == 0:
            res.append(i)
    return res


def findspec(B, Q, G, C):
    """Find optimal kernel configuration for forward pass."""
    d_stride = 8
    ms = factors(B * Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


def findspec_bwd(B, Q, G, C):
    """Find optimal kernel configuration for backward pass."""
    d_stride = 2 if C >= 64 else 1
    ms = factors(B * Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 256:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


@torch.library.custom_op("dcnv4::flash_deform_attn", mutates_args=())
def flash_deform_attn(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_loc_attn: torch.Tensor,
    im2col_step: int,
    K: int,
) -> torch.Tensor:
    """Forward pass of Flash Deformable Attention."""
    d_stride, blockthread = findspec(
        value.shape[0],
        sampling_loc_attn.shape[1],
        value.shape[2],
        value.shape[3],
    )

    return _ops.dcnv4_C.flash_deform_attn_forward(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_loc_attn,
        im2col_step,
        K,
        d_stride,
        blockthread,
    )


@flash_deform_attn.register_fake
def _(
    value,
    value_spatial_shapes,
    value_level_start_index,
    sampling_loc_attn,
    im2col_step,
    K,
):
    B, _N, G, D = value.shape
    Q = sampling_loc_attn.shape[1]
    return value.new_empty(B, Q, G * D)


def flash_deform_attn_backward(ctx, grad_output):
    value, value_spatial_shapes, value_level_start_index, sampling_loc_attn = (
        ctx.saved_tensors
    )
    im2col_step = ctx.im2col_step
    K = ctx.K

    d_stride_backward, blockthread_backward = findspec_bwd(
        value.shape[0],
        sampling_loc_attn.shape[1],
        value.shape[2],
        value.shape[3],
    )

    grad_value, grad_sampling_loc_attn = _ops.dcnv4_C.flash_deform_attn_backward(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_loc_attn,
        grad_output.contiguous(),
        im2col_step,
        K,
        d_stride_backward,
        blockthread_backward,
    )

    return grad_value, None, None, grad_sampling_loc_attn, None, None


def setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2], inputs[3])
    ctx.im2col_step = inputs[4]
    ctx.K = inputs[5]


flash_deform_attn.register_autograd(
    flash_deform_attn_backward,
    setup_context=setup_context,
)
