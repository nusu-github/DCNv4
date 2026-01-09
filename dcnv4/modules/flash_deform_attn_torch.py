# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""Pure PyTorch reference implementation of Flash Deformable Attention.

This module mirrors the CUDA kernel semantics and uses manual bilinear sampling.
"""

import math
import warnings

import torch
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def _is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2.

    Power-of-2 dimensions are more efficient for CUDA kernels due to
    better memory alignment and vectorized load/store operations.
    """
    if (not isinstance(n, int)) or (n < 0):
        msg = f"invalid input for _is_power_of_2: {n} (type: {type(n)})"
        raise ValueError(msg)
    return (n & (n - 1) == 0) and n != 0


def _bilinear_sample(
    value: torch.Tensor,
    h_im: torch.Tensor,
    w_im: torch.Tensor,
) -> torch.Tensor:
    """Manual bilinear sampling matching the CUDA kernel behavior.

    Args:
        value: Tensor of shape (B, H, W, G, D).
        h_im: Sampling y coordinates, shape (B, Q, G, K).
        w_im: Sampling x coordinates, shape (B, Q, G, K).

    Returns:
        Sampled values of shape (B, Q, G, K, D).

    """
    B, H, W, G, D = value.shape
    _, Q, _, K = h_im.shape

    h0 = torch.floor(h_im)
    w0 = torch.floor(w_im)
    h1 = h0 + 1
    w1 = w0 + 1
    lh = h_im - h0
    lw = w_im - w0
    hh = 1 - lh
    hw = 1 - lw

    h0i = h0.to(torch.int64)
    w0i = w0.to(torch.int64)
    h1i = h1.to(torch.int64)
    w1i = w1.to(torch.int64)

    value_flat = value.permute(0, 3, 1, 2, 4).reshape(B * G, H * W, D)

    def gather(h_idx: torch.Tensor, w_idx: torch.Tensor) -> torch.Tensor:
        mask = (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
        h_idx = h_idx.clamp(0, H - 1)
        w_idx = w_idx.clamp(0, W - 1)
        idx = h_idx * W + w_idx
        idx_flat = idx.permute(0, 2, 1, 3).reshape(B * G, Q * K)
        gathered = value_flat.gather(
            1,
            idx_flat.unsqueeze(-1).expand(-1, -1, D),
        )
        gathered = gathered.view(B, G, Q, K, D).permute(0, 2, 1, 3, 4)
        return gathered * mask.to(value.dtype).unsqueeze(-1)

    v1 = gather(h0i, w0i)
    v2 = gather(h0i, w1i)
    v3 = gather(h1i, w0i)
    v4 = gather(h1i, w1i)

    w1t = (hh * hw).unsqueeze(-1)
    w2t = (hh * lw).unsqueeze(-1)
    w3t = (lh * hw).unsqueeze(-1)
    w4t = (lh * lw).unsqueeze(-1)

    out = v1 * w1t + v2 * w2t + v3 * w3t + v4 * w4t
    valid = (h_im > -1) & (w_im > -1) & (h_im < H) & (w_im < W)
    return out * valid.to(value.dtype).unsqueeze(-1)


def flash_deform_attn_torch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_loc_attn: torch.Tensor,
    im2col_step: int,
    K: int,
) -> torch.Tensor:
    """Pure PyTorch reference for flash_deform_attn.

    Args:
        value: (B, N, G, D) value tensor.
        value_spatial_shapes: (L, 2) spatial shapes per level.
        value_level_start_index: (L,) level start indices.
        sampling_loc_attn: (B, Q, G, L * K * 3) concatenated coords and weights.
        im2col_step: Unused, kept for API compatibility.
        K: Number of sampling points per level.

    Returns:
        Tensor of shape (B, Q, G * D).

    """
    del im2col_step
    B, _, G, D = value.shape
    L = int(value_spatial_shapes.shape[0])
    Q = sampling_loc_attn.shape[1]

    expected = L * K * 3
    if sampling_loc_attn.shape[-1] != expected:
        msg = (
            "sampling_loc_attn last dim must be L * K * 3, "
            f"but got {sampling_loc_attn.shape[-1]} (expected {expected})."
        )
        raise ValueError(msg)

    orig_dtype = value.dtype
    compute_dtype = (
        torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
    )
    value = value.to(compute_dtype)
    sampling_loc_attn = sampling_loc_attn.to(compute_dtype)

    coords_dim = L * K * 2
    coords = sampling_loc_attn[..., :coords_dim].reshape(B, Q, G, L, K, 2)
    attn_logits = sampling_loc_attn[..., coords_dim:].reshape(B, Q, G, L * K)
    # Match CUDA kernel softmax: exp(logits - max) / sumexp.
    attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True).values
    attn = attn_logits.exp()
    attn = attn / attn.sum(dim=-1, keepdim=True)
    attn = attn.reshape(B, Q, G, L, K)

    output = value.new_zeros((B, Q, G, D))
    for level in range(L):
        spatial_h = int(value_spatial_shapes[level, 0].item())
        spatial_w = int(value_spatial_shapes[level, 1].item())
        level_start = int(value_level_start_index[level].item())
        level_end = level_start + spatial_h * spatial_w

        value_level = value[:, level_start:level_end].reshape(
            B,
            spatial_h,
            spatial_w,
            G,
            D,
        )
        coords_level = coords[:, :, :, level]
        loc_w = coords_level[..., 0]
        loc_h = coords_level[..., 1]

        h_im = loc_h * spatial_h - 0.5
        w_im = loc_w * spatial_w - 0.5

        sampled = _bilinear_sample(value_level, h_im, w_im)
        attn_level = attn[:, :, :, level].unsqueeze(-1)
        output = output + (sampled * attn_level).sum(dim=3)

    output = output.view(B, Q, G * D)
    if output.dtype != orig_dtype:
        output = output.to(orig_dtype)
    return output


class FlashDeformAttnTorch(nn.Module):
    """Multi-Scale Deformable Attention with pure PyTorch reference kernels."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4) -> None:
        """Initialize Multi-Scale Deformable Attention module.

        Args:
            d_model: Hidden dimension of the model. Must be divisible by n_heads.
                For optimal CUDA performance, d_model // n_heads should be a power of 2.
            n_levels: Number of feature map levels (scales) to sample from.
            n_heads: Number of attention heads. Each head has independent sampling
                offsets and attention weights.
            n_points: Number of sampling points per attention head per feature level.

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
        if not _is_power_of_2(_d_per_head):
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
        """Initialize sampling offsets to a radial pattern around the reference point."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
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
        """Compute multi-scale deformable attention (reference implementation)."""
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        if (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() != Len_in:
            msg = "input_spatial_shapes do not match input_flatten length"
            raise ValueError(msg)

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
            msg = (
                "Last dim of reference_points must be 2 or 4, "
                f"but got {reference_points.shape[-1]} instead."
            )
            raise ValueError(msg)

        sampling_locations = sampling_locations.flatten(-3)
        sampling_loc_attn = torch.cat([sampling_locations, attention_weights], dim=-1)

        output = flash_deform_attn_torch(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_loc_attn,
            self.im2col_step,
            self.n_points,
        )
        return self.output_proj(output)


__all__ = ["FlashDeformAttnTorch"]
