# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""Low-level autograd functions for DCNv4 operations.

This subpackage provides PyTorch autograd Function implementations that wrap
the optimized CUDA/Triton kernels for DCNv4 and Flash Deformable Attention.

These functions handle:
    - Forward pass: Bilinear sampling at dynamically predicted locations
    - Backward pass: Gradient computation for input, offsets, and weights
    - Automatic kernel configuration based on input shapes for optimal performance

Functions:
    - DCNv4Function: Autograd function for Deformable Convolution v4
    - FlashDeformAttnFunction: Autograd function for multi-scale deformable attention

Note:
    These are low-level building blocks. For most use cases, prefer the high-level
    modules in dcnv4.modules (dcnv4 and FlashDeformAttn classes).

Backend Selection:
    The functions automatically select between CUDA and Triton backends based on
    availability and the DCNV4_USE_TRITON environment variable.

"""

from .dcnv4_func import dcnv4_forward
from .flash_deform_attn_func import flash_deform_attn

__all__ = ["dcnv4_forward", "flash_deform_attn"]
