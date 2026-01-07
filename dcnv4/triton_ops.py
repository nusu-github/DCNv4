"""Triton backend for DCNv4 and FlashDeformAttn operations.

This module provides Triton-based implementations as an alternative to the
CUDA kernels. While the CUDA kernels are highly optimized for performance,
the Triton implementations offer several advantages:

    1. Portability: Triton kernels can run on a wider range of GPU architectures
       without recompilation.

    2. Readability: Triton code is more accessible than CUDA, making it easier
       to understand the algorithm and modify for research purposes.

    3. Development: Easier to prototype new features and optimizations.

Usage:
    To enable Triton backend, set the environment variable:
        export DCNV4_USE_TRITON=1

    By default (DCNV4_USE_TRITON=0), the optimized CUDA kernels are used.

Note:
    The Triton implementations prioritize correctness and portability over
    maximum performance. For production workloads, the CUDA backend is recommended.

Functions:
    - dcnv4_forward: Forward pass for DCNv4
    - dcnv4_backward: Backward pass for DCNv4
    - flash_deform_attn_forward: Forward pass for FlashDeformAttn
    - flash_deform_attn_backward: Backward pass for FlashDeformAttn
    - is_available: Check if Triton is available
    - use_triton: Check if Triton backend should be used (via env var)

"""

from __future__ import annotations

import os
from typing import NoReturn

try:
    from .triton.dcnv4_triton import dcnv4_backward, dcnv4_forward
    from .triton.flash_deform_triton import (
        flash_deform_attn_backward,
        flash_deform_attn_forward,
    )

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

    def dcnv4_forward(*args, **kwargs) -> NoReturn:
        msg = "Triton is not available."
        raise ImportError(msg)

    def dcnv4_backward(*args, **kwargs) -> NoReturn:
        msg = "Triton is not available."
        raise ImportError(msg)

    def flash_deform_attn_forward(*args, **kwargs) -> NoReturn:
        msg = "Triton is not available."
        raise ImportError(msg)

    def flash_deform_attn_backward(*args, **kwargs) -> NoReturn:
        msg = "Triton is not available."
        raise ImportError(msg)


def is_available() -> bool:
    """Check if Triton backend is available.

    Returns:
        True if Triton is installed and the kernels were successfully imported.

    """
    return _TRITON_AVAILABLE


def use_triton() -> bool:
    """Check if Triton backend should be used.

    The Triton backend is enabled by setting DCNV4_USE_TRITON=1 in the environment.

    Returns:
        True if DCNV4_USE_TRITON environment variable is set to "1".

    """
    flag = os.getenv("DCNV4_USE_TRITON", "0")
    return flag == "1"


__all__ = [
    "dcnv4_backward",
    "dcnv4_forward",
    "flash_deform_attn_backward",
    "flash_deform_attn_forward",
]
