# ------------------------------------------------------------------------------------------------
# DCNv4 Ops Registry
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
"""PyTorch custom ops registry for DCNv4.

This module provides access to the C++/CUDA implementations through torch.ops.
The ops are registered in the C++ extension (csrc/ops.cpp) using TORCH_LIBRARY.

Usage:
    from dcnv4.ops import dcnv4_forward, dcnv4_backward
    from dcnv4.ops import flash_deform_attn_forward, flash_deform_attn_backward
"""

import torch

from . import dcnv4_C  # noqa: F401

# Access ops through torch.ops namespace
# The ops are registered under the "dcnv4_C" namespace in csrc/ops.cpp
dcnv4_forward = torch.ops.dcnv4_C.dcnv4_forward
dcnv4_backward = torch.ops.dcnv4_C.dcnv4_backward
flash_deform_attn_forward = torch.ops.dcnv4_C.flash_deform_attn_forward
flash_deform_attn_backward = torch.ops.dcnv4_C.flash_deform_attn_backward

__all__ = [
    "dcnv4_backward",
    "dcnv4_forward",
    "flash_deform_attn_backward",
    "flash_deform_attn_forward",
]
