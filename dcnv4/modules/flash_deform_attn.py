# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""Multi-Scale Deformable Attention with Flash-style memory optimization.

This module implements multi-scale deformable attention as introduced in
Deformable DETR, with DCNv4-style optimizations for improved efficiency.

Multi-scale deformable attention differs from standard attention in that:
    1. It samples features from multiple feature map levels (scales)
    2. Sampling locations are predicted dynamically per query
    3. Only a small fixed number of points (n_points) are sampled per level,
       making it O(1) with respect to spatial resolution

This is particularly useful for object detection and segmentation where
objects of varying scales need to be detected efficiently.

Architecture:
    Query -> sampling_offsets -> sampling locations per level
          -> attention_weights -> weights for each sampled point
    Value (multi-scale features) -> sampled at predicted locations
    Output = weighted sum of sampled values

Reference:
    - Deformable DETR: https://arxiv.org/abs/2010.04159
"""

import math
import warnings

import torch
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from dcnv4.functions.flash_deform_attn_func import flash_deform_attn
from dcnv4.functions.utils import is_power_of_2


class FlashDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention with optimized CUDA kernels.

    This module implements efficient multi-scale deformable attention for
    vision transformers and detection models. It dynamically samples features
    from multiple feature levels at learned locations.

    Key Features:
        - Multi-scale feature aggregation from n_levels feature maps
        - Dynamic sampling locations predicted from query features
        - Constant complexity O(n_levels * n_points) regardless of image size
        - Optimized CUDA kernels with memory access patterns from DCNv4

    Attributes:
        d_model: Hidden dimension of the model.
        n_levels: Number of feature map levels to sample from.
        n_heads: Number of attention heads.
        n_points: Number of sampling points per head per level.

    Example:
        >>> attn = FlashDeformAttn(d_model=256, n_levels=4, n_heads=8, n_points=4)
        >>> query = torch.randn(2, 100, 256)  # (batch, num_queries, d_model)
        >>> # reference_points: normalized coordinates [0,1] for each query
        >>> reference_points = torch.rand(2, 100, 4, 2)  # (batch, queries, levels, 2)
        >>> # input_flatten: flattened multi-scale features
        >>> input_flatten = torch.randn(2, 10000, 256)  # (batch, total_pixels, d_model)
        >>> spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20], [10, 10]])
        >>> level_start_index = torch.tensor([0, 6400, 8000, 8400])
        >>> out = attn(query, reference_points, input_flatten, spatial_shapes, level_start_index)

    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4) -> None:
        """Initialize Multi-Scale Deformable Attention module.

        Args:
            d_model: Hidden dimension of the model. Must be divisible by n_heads.
                For optimal CUDA performance, d_model // n_heads should be a power of 2.
            n_levels: Number of feature map levels (scales) to sample from. Typically
                4 for FPN-style feature pyramids.
            n_heads: Number of attention heads. Each head has independent sampling
                offsets and attention weights.
            n_points: Number of sampling points per attention head per feature level.
                Total points sampled = n_heads * n_levels * n_points.

        Raises:
            ValueError: If d_model is not divisible by n_heads.

        """
        super().__init__()
        if d_model % n_heads != 0:
            msg = (
                f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}"
            )
            raise ValueError(msg)
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.",
                stacklevel=2,
            )

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize sampling offsets to a radial pattern around the reference point.

        The bias initialization creates a grid of sampling points that:
            1. Spread outward from the center in a circular/radial pattern
            2. Each head samples in a different direction (evenly spaced angles)
            3. Points are distributed at increasing distances (1, 2, ..., n_points)

        This initialization ensures diverse spatial coverage from the start and
        helps the model quickly learn meaningful deformation patterns.
        """
        constant_(self.sampling_offsets.weight.data, 0.0)
        # Initialize sampling offsets in a radial pattern
        # Each head points in a different direction (angle = 2*pi*head_idx/n_heads)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        # Scale offsets: point i is at distance (i+1) from center
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        """Compute multi-scale deformable attention.

        For each query, this module:
            1. Predicts sampling offsets relative to reference points
            2. Predicts attention weights for all sampling points
            3. Samples values from multi-scale features at predicted locations
            4. Aggregates sampled values using attention weights

        Args:
            query: Query features of shape (N, Length_query, C).
            reference_points: Reference point coordinates, normalized to [0, 1].
                Shape (N, Length_query, n_levels, 2) for point references where
                (0, 0) is top-left and (1, 1) is bottom-right.
                Or shape (N, Length_query, n_levels, 4) for box references where
                the last two values are (width, height) of the reference box.
            input_flatten: Flattened multi-scale feature maps of shape
                (N, sum(H_l * W_l), C). Features from all levels are concatenated.
            input_spatial_shapes: Spatial dimensions of each level, shape (n_levels, 2).
                Each row is [H_l, W_l] for level l.
            input_level_start_index: Starting index of each level in input_flatten,
                shape (n_levels,). Used to locate features for each level.
            input_padding_mask: Optional mask indicating padded positions, shape
                (N, sum(H_l * W_l)). True for padded elements. Default: None.

        Returns:
            Output features of shape (N, Length_query, C).

        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N,
            Len_q,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            N,
            Len_q,
            self.n_heads,
            self.n_levels * self.n_points,
        )
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]],
                -1,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            msg = f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
            raise ValueError(
                msg,
            )

        # Cat sampling_offsets and attention_weights, generate sampling_loc_attn:
        sampling_locations = sampling_locations.flatten(-3).half()
        sampling_loc_attn = torch.cat([sampling_locations, attention_weights], dim=-1)

        im2col_step = self.im2col_step

        output = flash_deform_attn(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_loc_attn,
            im2col_step,
            self.n_points,
        )
        return self.output_proj(output)
