# --------------------------------------------------------
# Deformable Convolution v4
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
"""DCNv4 Module Implementation.

This module provides the main DCNv4 (Deformable Convolution v4) layer that can be
used as a drop-in replacement for standard convolutions in vision models.

Key Design Principles (from the paper):
    1. Dynamic Spatial Aggregation: Unlike standard convolutions with fixed weights,
       DCNv4 predicts input-dependent sampling offsets and aggregation weights,
       combining the benefits of convolution (local inductive bias) and attention
       (dynamic weighting).

    2. Unbounded Aggregation Weights: DCNv4 removes the softmax normalization used
       in DCNv3, allowing aggregation weights to have unbounded values like standard
       convolutions. This significantly improves convergence speed and expressive power.

    3. Grouped Spatial Aggregation: Features are divided into G groups, where each
       group has its own set of sampling offsets and weights. Channels within a group
       share the same spatial aggregation pattern.

    4. Separable Design: Following Xception, DCNv4 uses pointwise convolutions (1x1)
       before and after the deformable spatial aggregation to enhance expressiveness.

Mathematical Formulation:
    For each output location p_0 and group g:

        y_g = sum_{k=1}^{K} m_{gk} * x_g(p_0 + p_k + delta_p_{gk})

    where:
        - K is the number of sampling points (kernel_size^2)
        - m_{gk} are the dynamic aggregation weights (NOT softmax-normalized in DCNv4)
        - p_k are the base grid positions
        - delta_p_{gk} are the learned sampling offsets
        - x_g is the input feature for group g
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from dcnv4.functions.dcnv4_func import dcnv4_forward


class CenterFeatureScaleModule(nn.Module):
    """Computes scaling factors to blend center features with deformed features.

    This module is used when center_feature_scale=True in DCNv4. It learns to
    predict per-group scaling factors that control the interpolation between
    the original center features and the spatially aggregated features.

    This can help preserve fine-grained details while still benefiting from
    the adaptive spatial aggregation of DCNv4.
    """

    def forward(
        self,
        query,
        center_feature_scale_proj_weight,
        center_feature_scale_proj_bias,
    ):
        """Compute center feature scaling factors.

        Args:
            query: Input features of shape (N, H, W, C).
            center_feature_scale_proj_weight: Projection weight of shape (G, C).
            center_feature_scale_proj_bias: Projection bias of shape (G,).

        Returns:
            Scaling factors in range [0, 1] of shape (N, H, W, G).

        """
        return F.linear(
            query,
            weight=center_feature_scale_proj_weight,
            bias=center_feature_scale_proj_bias,
        ).sigmoid()


class dcnv4(nn.Module):
    """Deformable Convolution v4 (DCNv4) Module.

    DCNv4 is a highly efficient operator that combines the benefits of convolutions
    (local inductive bias, sliding window) with attention (dynamic, input-dependent
    weights). It achieves 3x+ speedup over DCNv3 through optimized memory access
    and improved convergence via unbounded aggregation weights.

    Architecture:
        Input -> [value_proj] -> DCNv4Core -> [center_scale] -> [output_proj] -> Output
                      |
                [offset_mask_dw] -> offset_mask (predicts offsets & weights)

    The core DCNv4 operation samples features at dynamically predicted locations
    using bilinear interpolation, weighted by learned aggregation weights.

    Attributes:
        channels: Number of input/output channels.
        kernel_size: Spatial size of the sampling grid (e.g., 3 for 3x3).
        group: Number of groups for spatial aggregation. Each group has independent
            offsets and weights. channels must be divisible by group.
        group_channels: Channels per group (= channels // group).
        offset_scale: Scaling factor for sampling offsets.

    Example:
        >>> layer = dcnv4(channels=64, kernel_size=3, group=4)
        >>> x = torch.randn(2, 4096, 64)  # (batch, seq_len, channels)
        >>> out = layer(x, shape=(64, 64))  # Provide spatial dimensions

    """

    def __init__(
        self,
        channels=64,
        kernel_size=3,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        dw_kernel_size=None,
        center_feature_scale=False,
        remove_center=False,
        output_bias=True,
        without_pointwise=False,
        **kwargs,
    ) -> None:
        """Initialize DCNv4 module.

        Args:
            channels: Number of input and output channels. Must be divisible by group,
                and (channels // group) must be divisible by 16 for CUDA efficiency.
            kernel_size: Size of the deformable convolution kernel. Default: 3.
            stride: Stride of the convolution. Default: 1.
            pad: Padding added to input. Default: 1.
            dilation: Spacing between kernel elements. Default: 1.
            group: Number of groups for grouped spatial aggregation. Each group learns
                independent sampling offsets and aggregation weights. Default: 4.
            offset_scale: Scaling factor applied to predicted offsets. Larger values
                allow sampling from a wider spatial region. Default: 1.0.
            dw_kernel_size: If specified, applies a depthwise convolution of this size
                before predicting offsets. Set to None to skip (faster). Default: None.
            center_feature_scale: If True, learns to blend the center (undeformed)
                features with the deformed output. Helps preserve fine details. Default: False.
            remove_center: If True, excludes the center point from the sampling grid,
                reducing computation for (kernel_size^2 - 1) points. Default: False.
            output_bias: Whether to include bias in the output projection. Default: True.
            without_pointwise: If True, skips the value_proj and output_proj linear
                layers, using only the core DCNv4 operation. Default: False.

        Raises:
            ValueError: If channels is not divisible by group.
            AssertionError: If (channels // group) is not divisible by 16.

        """
        super().__init__()
        if channels % group != 0:
            msg = f"channels must be divisible by group, but got {channels} and {group}"
            raise ValueError(msg)
        _d_per_group = channels // group

        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if _d_per_group % 16 != 0:
            msg = (
                f"channels per group must be divisible by 16 for CUDA efficiency, "
                f"but got {_d_per_group} (channels={channels}, group={group})"
            )
            raise ValueError(msg)

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        self.K = group * (kernel_size * kernel_size - self.remove_center)
        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(
                channels,
                channels,
                dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels,
            )
        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3) / 8) * 8))
        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float),
            )
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group),
            )
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self) -> None:
        """Initialize layer parameters.

        Offset and mask weights are initialized to zero so that initial sampling
        positions correspond to the regular grid (no deformation). This provides
        a stable starting point for training.

        Value and output projections use Xavier initialization for better gradient
        flow in deep networks.
        """
        # Zero-init offsets: start with regular grid sampling
        constant_(self.offset_mask.weight.data, 0.0)
        constant_(self.offset_mask.bias.data, 0.0)

        # Init modulation weights to 0.5 to avoid zero gradient at initialization
        points_per_group = self.kernel_size * self.kernel_size - self.remove_center
        for g in range(self.group):
            # The layout is [offsets_x, offsets_y, weights] per group
            # Offsets take 2 * points_per_group, weights take points_per_group
            weight_start = g * points_per_group * 3 + 2 * points_per_group
            weight_end = weight_start + points_per_group
            constant_(self.offset_mask.bias.data[weight_start:weight_end], 0.5)

        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.0)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.0)

    def forward(self, input, shape=None):
        """Apply DCNv4 operation.

        The forward pass consists of:
            1. Project input through value_proj (if enabled)
            2. Predict sampling offsets and aggregation weights from input
            3. Sample features at predicted locations using bilinear interpolation
            4. Aggregate sampled features with predicted weights
            5. Optionally blend with center features (if center_feature_scale=True)
            6. Project output through output_proj (if enabled)

        Args:
            input: Input tensor of shape (N, L, C) where L = H * W is the
                flattened spatial dimension.
            shape: Optional tuple (H, W) specifying spatial dimensions. If None,
                assumes square spatial dimensions (sqrt(L) x sqrt(L)).

        Returns:
            Output tensor of shape (N, L, C).

        Note:
            The input is expected in (N, L, C) format (sequence format), which is
            internally reshaped to (N, H, W, C) for the deformable convolution.

        """
        N, L, C = input.shape
        if shape is not None:
            H, W = shape
        else:
            H, W = int(L**0.5), int(L**0.5)

        x = input
        if not self.without_pointwise:
            x = self.value_proj(x)
        x = x.reshape(N, H, W, -1)
        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(
                input.view(N, H, W, C).permute(0, 3, 1, 2),
            )
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
        else:
            offset_mask_input = input
        offset_mask = self.offset_mask(offset_mask_input).reshape(N, H, W, -1)

        x_proj = x

        im2col_step = 256
        if N % im2col_step != 0:
            im2col_step = N

        x = dcnv4_forward(
            x,
            offset_mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.pad,
            self.pad,
            self.dilation,
            self.dilation,
            self.group,
            self.group_channels,
            self.offset_scale,
            im2col_step,
            self.remove_center,
        )

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x,
                self.center_feature_scale_proj_weight,
                self.center_feature_scale_proj_bias,
            )
            center_feature_scale = (
                center_feature_scale[..., None]
                .repeat(1, 1, 1, 1, self.channels // self.group)
                .flatten(-2)
            )
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = x.view(N, L, -1)

        if not self.without_pointwise:
            x = self.output_proj(x)
        return x
