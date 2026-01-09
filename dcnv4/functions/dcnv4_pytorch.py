# --------------------------------------------------------
# Deformable Convolution v4 (PyTorch reference)
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
"""Pure PyTorch reference implementation of DCNv4.

This is a slow, numerically-oriented implementation intended for
correctness testing and debugging. It mirrors the CUDA kernel logic
in dcnv4/csrc/cuda/dcnv4_im2col_cuda.cuh.
"""

import torch


def _bilinear_sample(
    value: torch.Tensor,
    h_im: torch.Tensor,
    w_im: torch.Tensor,
) -> torch.Tensor:
    """Bilinear sampling that matches the CUDA boundary checks.

    Args:
        value: (B, H, W, D)
        h_im: (B, H_out, W_out)
        w_im: (B, H_out, W_out)

    Returns:
        Sampled values of shape (B, H_out, W_out, D).

    """
    B, H, W, D = value.shape
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

    def gather(h_idx: torch.Tensor, w_idx: torch.Tensor) -> torch.Tensor:
        mask = (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
        h_idx = h_idx.clamp(0, H - 1)
        w_idx = w_idx.clamp(0, W - 1)
        idx = h_idx * W + w_idx
        idx_flat = idx.view(B, -1)
        value_flat = value.view(B, H * W, D)
        gathered = value_flat.gather(
            1,
            idx_flat.unsqueeze(-1).expand(-1, -1, D),
        )
        gathered = gathered.view(B, h_im.shape[1], h_im.shape[2], D)
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


def dcnv4_forward_pytorch(
    input: torch.Tensor,
    offset_mask: torch.Tensor,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    group: int,
    group_channels: int,
    offset_scale: float,
    im2col_step: int,
    remove_center: int,
    softmax: bool = False,
) -> torch.Tensor:
    """Pure PyTorch DCNv4 forward (slow reference).
    This matches the CUDA kernel semantics, including offset layout and
    boundary checks. `im2col_step` is accepted for API compatibility.
    """
    del im2col_step
    if input.dim() != 4:
        msg = f"input must be (B, H, W, C), got {input.shape}"
        raise ValueError(msg)
    if offset_mask.dim() != 4:
        msg = f"offset_mask must be (B, H_out, W_out, C), got {offset_mask.shape}"
        raise ValueError(msg)
    B, H_in, W_in, C = input.shape
    if group * group_channels != C:
        msg = (
            "Input channels and group * group_channels mismatch: "
            f"{C} vs {group * group_channels}."
        )
        raise ValueError(msg)
    H_out = (H_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    if offset_mask.shape[1] != H_out or offset_mask.shape[2] != W_out:
        msg = (
            "offset_mask spatial dims must be (H_out, W_out) = "
            f"({H_out}, {W_out}), got ({offset_mask.shape[1]}, {offset_mask.shape[2]})."
        )
        raise ValueError(msg)
    K = kernel_h * kernel_w - int(remove_center)
    needed = group * K * 3
    if offset_mask.shape[3] < needed:
        msg = (
            "offset_mask last dim too small: needs at least "
            f"{needed}, got {offset_mask.shape[3]}."
        )
        raise ValueError(msg)
    dtype = input.dtype
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    value = input.to(compute_dtype)
    offset_mask = offset_mask.to(compute_dtype)
    out = value.new_zeros((B, H_out, W_out, C))
    base_h = (dilation_h * (kernel_h - 1)) // 2
    base_w = (dilation_w * (kernel_w - 1)) // 2
    out_h_idx = torch.arange(H_out, device=value.device, dtype=compute_dtype)
    out_w_idx = torch.arange(W_out, device=value.device, dtype=compute_dtype)
    p0_h = (base_h - pad_h + out_h_idx * stride_h).view(1, H_out, 1)
    p0_w = (base_w - pad_w + out_w_idx * stride_w).view(1, 1, W_out)
    offset_scale_t = torch.tensor(
        offset_scale,
        device=value.device,
        dtype=compute_dtype,
    )
    p0_h = p0_h - base_h * offset_scale_t
    p0_w = p0_w - base_w * offset_scale_t
    center_h = kernel_h // 2
    center_w = kernel_w // 2
    positions = []
    for i in range(kernel_w):
        for j in range(kernel_h):
            if int(remove_center) and i == center_w and j == center_h:
                continue
            positions.append((i, j))
    if len(positions) != K:
        msg = "Internal error: position count does not match K."
        raise RuntimeError(msg)
    for g in range(group):
        start = g * K * 3
        offsets = offset_mask[..., start : start + 2 * K].view(
            B,
            H_out,
            W_out,
            K,
            2,
        )
        weights = offset_mask[..., start + 2 * K : start + 3 * K].view(
            B,
            H_out,
            W_out,
            K,
        )
        if softmax:
            weights = torch.softmax(weights, dim=-1)
        value_g = value[..., g * group_channels : (g + 1) * group_channels]
        out_g = out[..., g * group_channels : (g + 1) * group_channels]
        for k, (i, j) in enumerate(positions):
            offset_x = offsets[..., k, 0]
            offset_y = offsets[..., k, 1]
            attn = weights[..., k]
            h_im = p0_h + (j * dilation_h + offset_y) * offset_scale_t
            w_im = p0_w + (i * dilation_w + offset_x) * offset_scale_t
            sampled = _bilinear_sample(value_g, h_im, w_im)
            out_g += sampled * attn.unsqueeze(-1)
    if out.dtype != dtype:
        out = out.to(dtype)
    return out


__all__ = ["dcnv4_forward_pytorch"]
