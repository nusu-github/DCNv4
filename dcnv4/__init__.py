"""DCNv4: Deformable Convolution v4 - A highly efficient dynamic sparse operator.

This package provides the official implementation of DCNv4 as described in:
"Efficient Deformable ConvNets: Rethinking Dynamic and Sparse Operator for Vision Applications"
(https://arxiv.org/abs/2401.06197)

DCNv4 introduces two key improvements over DCNv3:
    1. Removal of softmax normalization on spatial aggregation weights, allowing
       unbounded dynamic weights similar to standard convolutions. This enhances
       expressive power and significantly accelerates convergence.
    2. Optimized memory access patterns that eliminate redundant operations by
       processing multiple channels sharing the same offset within a single thread,
       achieving 3x+ speedup over DCNv3.

Main Components:
    - dcnv4: The main DCNv4 module for use as a drop-in replacement for convolutions
    - FlashDeformAttn: Multi-scale deformable attention with optimized CUDA kernels
    - DCNv4Function: Low-level autograd function for DCNv4 forward/backward
    - FlashDeformAttnFunction: Low-level autograd function for deformable attention

Example:
    >>> import torch
    >>> from dcnv4 import dcnv4
    >>>
    >>> # Create DCNv4 layer (channels must be divisible by group)
    >>> layer = dcnv4(channels=64, kernel_size=3, group=4)
    >>> layer = layer.cuda()
    >>>
    >>> # Input shape: (batch, sequence_length, channels)
    >>> x = torch.randn(2, 64*64, 64).cuda()
    >>> output = layer(x, shape=(64, 64))  # Provide spatial shape

References:
    - Paper: https://arxiv.org/abs/2401.06197
    - Code: https://github.com/OpenGVLab/DCNv4

"""

from typing import TYPE_CHECKING

from dcnv4 import dcnv4_C  # ty:ignore[unresolved-import]  # noqa: F401

from .functions import DCNv4Function, FlashDeformAttnFunction
from .modules import FlashDeformAttn, dcnv4

if TYPE_CHECKING:
    from . import _ops

__all__ = ["DCNv4Function", "FlashDeformAttn", "FlashDeformAttnFunction", "dcnv4"]
