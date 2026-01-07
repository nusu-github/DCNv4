# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""DCNv4 PyTorch modules.

This subpackage provides high-level PyTorch nn.Module implementations:

    - dcnv4: The main Deformable Convolution v4 module. A drop-in replacement
      for standard convolutions with dynamic, input-dependent spatial sampling.

    - FlashDeformAttn: Multi-scale deformable attention module with optimized
      CUDA kernels. Used in detection/segmentation models like DINO and Mask2Former.

Both modules leverage highly optimized CUDA kernels for efficient forward and
backward passes, with optional Triton-based implementations for portability.
"""

from .dcnv4 import dcnv4
from .flash_deform_attn import FlashDeformAttn

__all__ = ["FlashDeformAttn", "dcnv4"]
