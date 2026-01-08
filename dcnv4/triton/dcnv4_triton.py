from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import (
    _choose_block_d,
    _compute_output_hw,
    _get_autotune_config,
    _get_autotune_config_bwd,
    _next_power_of_2,
    _prune_configs_dcnv4,
)


@triton.autotune(
    configs=_get_autotune_config(),
    key=["G", "D"],
    prune_configs_by={"early_config_prune": _prune_configs_dcnv4},
)
@triton.jit
def _dcnv4_fwd_kernel(
    value_ptr,
    offset_ptr,
    output_ptr,
    H_in,
    W_in,
    H_out,
    W_out,
    G,
    D,
    stride_vb,
    stride_vh,
    stride_vw,
    stride_vc,
    stride_ob,
    stride_oh,
    stride_ow,
    stride_oc,
    stride_outb,
    stride_outh,
    stride_outw,
    stride_outc,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    offset_scale,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REMOVE_CENTER: tl.constexpr,
    SOFTMAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
) -> None:
    pid_q = tl.program_id(0)
    pid_gd = tl.program_id(1)
    b = tl.program_id(2)

    g = pid_gd // NUM_D_BLOCKS
    pid_d = pid_gd - g * NUM_D_BLOCKS

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    tl.max_contiguous(offs_q, BLOCK_Q)
    mask_q = offs_q < (H_out * W_out)

    h_out = offs_q // W_out
    w_out = offs_q % W_out

    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    tl.max_contiguous(d, BLOCK_D)
    mask_c = d < D
    c = g * D + d

    # offset_ptr assumed contiguous in H, W -> stride_ow is stride for Q
    offset_base = offset_ptr + b * stride_ob + offs_q * stride_ow + g * K_TOTAL * 3

    mask_idx = tl.arange(0, BLOCK_K)
    tl.max_contiguous(mask_idx, BLOCK_K)

    # [BLOCK_Q, BLOCK_K]
    mask_vals = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)

    if SOFTMAX:
        # Load all masks for Softmax (avoid -inf in fp16 paths)
        mask_k = mask_idx < K_TOTAL
        mask_qk = mask_q[:, None] & mask_k[None, :]
        ptr_mask = offset_base[:, None] + (2 * K_TOTAL + mask_idx[None, :])
        mask_vals = tl.load(ptr_mask, mask=mask_qk, other=0.0).to(tl.float32)
        mask_vals = tl.where(mask_qk, mask_vals, -1.0e10)

        maxv = tl.max(mask_vals, axis=1)
        expv = tl.exp(mask_vals - maxv[:, None])
        denom = tl.sum(expv, axis=1)
        mask_vals = expv / denom[:, None]

    half_w = (dilation_w * (KERNEL_W - 1)) // 2
    half_h = (dilation_h * (KERNEL_H - 1)) // 2

    # p0 calculation vector [BLOCK_Q]
    p0_w = half_w - pad_w + w_out * stride_w
    p0_h = half_h - pad_h + h_out * stride_h
    p0_w = p0_w.to(tl.float32)
    p0_h = p0_h.to(tl.float32)
    p0_w_ = p0_w - half_w * offset_scale
    p0_h_ = p0_h - half_h * offset_scale

    acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    center_h = KERNEL_H // 2
    center_w = KERNEL_W // 2

    point_idx = 0
    for i in range(KERNEL_W):
        for j in range(KERNEL_H):
            if (REMOVE_CENTER == 0) or (i != center_w or j != center_h):
                # Load offsets just in time
                # [BLOCK_Q]
                ptr_x = offset_base + 2 * point_idx
                ptr_y = offset_base + 2 * point_idx + 1
                off_x = tl.load(ptr_x, mask=mask_q, other=0.0).to(tl.float32)
                off_y = tl.load(ptr_y, mask=mask_q, other=0.0).to(tl.float32)

                attn = tl.zeros([BLOCK_Q], dtype=tl.float32)
                if SOFTMAX:
                    # Triton does not support dynamic tensor indexing in all versions.
                    mask_now = (mask_idx == point_idx)[None, :]
                    attn = tl.sum(mask_vals * mask_now, axis=1)
                else:
                    ptr_mask_val = offset_base + 2 * K_TOTAL + point_idx
                    attn = tl.load(ptr_mask_val, mask=mask_q, other=0.0).to(tl.float32)

                w_im = p0_w_ + (i * dilation_w + off_x) * offset_scale
                h_im = p0_h_ + (j * dilation_h + off_y) * offset_scale

                valid = (h_im > -1) & (w_im > -1) & (h_im < H_in) & (w_im < W_in)
                valid = valid & mask_q  # Apply mask_q to validity

                h_low = tl.floor(h_im)
                w_low = tl.floor(w_im)
                h_low_i = h_low.to(tl.int32)
                w_low_i = w_low.to(tl.int32)
                h_high_i = h_low_i + 1
                w_high_i = w_low_i + 1

                lh = h_im - h_low
                lw = w_im - w_low
                hh = 1.0 - lh
                hw = 1.0 - lw

                w1 = hh * hw
                w2 = hh * lw
                w3 = lh * hw
                w4 = lh * lw

                # value loading
                # base [1, BLOCK_D]
                base = value_ptr + b * stride_vb + c * stride_vc
                base = base[None, :]

                # offset_base_val [BLOCK_Q, 1]
                offset_val_h_low = h_low_i * stride_vh
                offset_val_w_low = w_low_i * stride_vw
                offset_base_val = offset_val_h_low + offset_val_w_low
                offset_base_val = offset_base_val[:, None]

                ptr1 = base + offset_base_val  # [BLOCK_Q, BLOCK_D]

                check_h_low = (h_low_i >= 0) & (h_low_i < H_in)
                check_w_low = (w_low_i >= 0) & (w_low_i < W_in)
                check_h_high = (h_high_i >= 0) & (h_high_i < H_in)
                check_w_high = (w_high_i >= 0) & (w_high_i < W_in)

                # Broadcast checks to [BLOCK_Q, 1]
                valid_bc = valid[:, None]
                mask_c_bc = mask_c[None, :]  # [1, BLOCK_D]

                mask1 = (
                    mask_c_bc & valid_bc & (check_h_low[:, None] & check_w_low[:, None])
                )
                v1 = tl.load(ptr1, mask=mask1, other=0.0).to(tl.float32)

                mask2 = (
                    mask_c_bc
                    & valid_bc
                    & (check_h_low[:, None] & check_w_high[:, None])
                )
                v2 = tl.load(ptr1 + stride_vw, mask=mask2, other=0.0).to(tl.float32)

                mask3 = (
                    mask_c_bc
                    & valid_bc
                    & (check_h_high[:, None] & check_w_low[:, None])
                )
                v3 = tl.load(ptr1 + stride_vh, mask=mask3, other=0.0).to(tl.float32)

                mask4 = (
                    mask_c_bc
                    & valid_bc
                    & (check_h_high[:, None] & check_w_high[:, None])
                )
                v4 = tl.load(ptr1 + stride_vh + stride_vw, mask=mask4, other=0.0).to(
                    tl.float32,
                )

                interp = tl.fma(
                    v1,
                    w1[:, None],
                    tl.fma(v2, w2[:, None], tl.fma(v3, w3[:, None], v4 * w4[:, None])),
                )
                acc += attn[:, None] * interp

                point_idx += 1

    # Store
    # out_ptrs [BLOCK_Q, BLOCK_D]
    # h_out, w_out [BLOCK_Q]
    out_ptrs = (
        output_ptr
        + b * stride_outb
        + h_out[:, None] * stride_outh
        + w_out[:, None] * stride_outw
        + c[None, :] * stride_outc
    )
    tl.store(out_ptrs, acc.to(tl.float32), mask=(mask_q[:, None] & mask_c[None, :]))


@triton.autotune(
    configs=_get_autotune_config_bwd(),
    key=["G", "D"],
    reset_to_zero=["grad_input_ptr"],
    prune_configs_by={"early_config_prune": _prune_configs_dcnv4},
)
@triton.jit
def _dcnv4_bwd_input_kernel(
    offset_ptr,
    grad_out_ptr,
    grad_input_ptr,
    H_in,
    W_in,
    H_out,
    W_out,
    G,
    D,
    stride_ob,
    stride_oh,
    stride_ow,
    stride_oc,
    stride_gob,
    stride_goh,
    stride_gow,
    stride_goc,
    stride_gib,
    stride_gih,
    stride_giw,
    stride_gic,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    offset_scale,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REMOVE_CENTER: tl.constexpr,
    SOFTMAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
) -> None:
    pid_q = tl.program_id(0)
    pid_gd = tl.program_id(1)
    b = tl.program_id(2)

    g = pid_gd // NUM_D_BLOCKS
    pid_d = pid_gd - g * NUM_D_BLOCKS

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    tl.max_contiguous(offs_q, BLOCK_Q)
    mask_q = offs_q < (H_out * W_out)

    h_out = offs_q // W_out
    w_out = offs_q % W_out

    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    tl.max_contiguous(d, BLOCK_D)
    mask_c = d < D
    c = g * D + d

    mask_idx = tl.arange(0, BLOCK_K)
    tl.max_contiguous(mask_idx, BLOCK_K)
    offset_base = offset_ptr + b * stride_ob + offs_q * stride_ow + g * K_TOTAL * 3

    # Load masks for softmax
    mask_vals = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)
    if SOFTMAX:
        mask_k = mask_idx < K_TOTAL
        mask_qk = mask_q[:, None] & mask_k[None, :]
        ptr_mask = offset_base[:, None] + (2 * K_TOTAL + mask_idx[None, :])
        mask_vals = tl.load(ptr_mask, mask=mask_qk, other=0.0).to(tl.float32)
        mask_vals = tl.where(mask_qk, mask_vals, -1.0e10)
        maxv = tl.max(mask_vals, axis=1)
        expv = tl.exp(mask_vals - maxv[:, None])
        denom = tl.sum(expv, axis=1)
        mask_vals = expv / denom[:, None]

    half_w = (dilation_w * (KERNEL_W - 1)) // 2
    half_h = (dilation_h * (KERNEL_H - 1)) // 2
    p0_w = half_w - pad_w + w_out * stride_w
    p0_h = half_h - pad_h + h_out * stride_h
    p0_w = p0_w.to(tl.float32)
    p0_h = p0_h.to(tl.float32)
    p0_w_ = p0_w - half_w * offset_scale
    p0_h_ = p0_h - half_h * offset_scale

    center_h = KERNEL_H // 2
    center_w = KERNEL_W // 2

    go_ptrs = (
        grad_out_ptr
        + b * stride_gob
        + h_out[:, None] * stride_goh
        + w_out[:, None] * stride_gow
        + c[None, :] * stride_goc
    )
    # [BLOCK_Q, BLOCK_D]
    go = tl.load(go_ptrs, mask=(mask_q[:, None] & mask_c[None, :]), other=0.0).to(
        tl.float32,
    )

    point_idx = 0
    for i in range(KERNEL_W):
        for j in range(KERNEL_H):
            if (REMOVE_CENTER == 0) or (i != center_w or j != center_h):
                # Load offsets just in time
                ptr_x = offset_base + 2 * point_idx
                ptr_y = offset_base + 2 * point_idx + 1
                off_x = tl.load(ptr_x, mask=mask_q, other=0.0).to(tl.float32)
                off_y = tl.load(ptr_y, mask=mask_q, other=0.0).to(tl.float32)

                attn = tl.zeros([BLOCK_Q], dtype=tl.float32)
                if SOFTMAX:
                    mask_now = (mask_idx == point_idx)[None, :]
                    attn = tl.sum(mask_vals * mask_now, axis=1)
                else:
                    ptr_mask_val = offset_base + 2 * K_TOTAL + point_idx
                    attn = tl.load(ptr_mask_val, mask=mask_q, other=0.0).to(tl.float32)

                w_im = p0_w_ + (i * dilation_w + off_x) * offset_scale
                h_im = p0_h_ + (j * dilation_h + off_y) * offset_scale

                valid = (h_im > -1) & (w_im > -1) & (h_im < H_in) & (w_im < W_in)
                valid = valid & mask_q

                h_low = tl.floor(h_im)
                w_low = tl.floor(w_im)
                h_low_i = h_low.to(tl.int32)
                w_low_i = w_low.to(tl.int32)
                h_high_i = h_low_i + 1
                w_high_i = w_low_i + 1

                lh = h_im - h_low
                lw = w_im - w_low
                hh = 1.0 - lh
                hw = 1.0 - lw

                w1 = hh * hw
                w2 = hh * lw
                w3 = lh * hw
                w4 = lh * lw

                check_h_low = (h_low_i >= 0) & (h_low_i < H_in)
                check_w_low = (w_low_i >= 0) & (w_low_i < W_in)
                check_h_high = (h_high_i >= 0) & (h_high_i < H_in)
                check_w_high = (w_high_i >= 0) & (w_high_i < W_in)

                valid_bc = valid[:, None]
                mask_c_bc = mask_c[None, :]

                mask1 = (
                    mask_c_bc & valid_bc & (check_h_low[:, None] & check_w_low[:, None])
                )
                mask2 = (
                    mask_c_bc
                    & valid_bc
                    & (check_h_low[:, None] & check_w_high[:, None])
                )
                mask3 = (
                    mask_c_bc
                    & valid_bc
                    & (check_h_high[:, None] & check_w_low[:, None])
                )
                mask4 = (
                    mask_c_bc
                    & valid_bc
                    & (check_h_high[:, None] & check_w_high[:, None])
                )

                # Atomic add to grad_input
                base_grad_in = grad_input_ptr + b * stride_gib + c * stride_gic
                offset_base_grad = h_low_i * stride_gih + w_low_i * stride_giw
                g_ptr1 = base_grad_in[None, :] + offset_base_grad[:, None]

                tl.atomic_add(g_ptr1, go * attn[:, None] * w1[:, None], mask=mask1)
                tl.atomic_add(
                    g_ptr1 + stride_giw,
                    go * attn[:, None] * w2[:, None],
                    mask=mask2,
                )
                tl.atomic_add(
                    g_ptr1 + stride_gih,
                    go * attn[:, None] * w3[:, None],
                    mask=mask3,
                )
                tl.atomic_add(
                    g_ptr1 + stride_gih + stride_giw,
                    go * attn[:, None] * w4[:, None],
                    mask=mask4,
                )

                point_idx += 1


@triton.autotune(
    configs=_get_autotune_config_bwd(),
    key=["G", "D"],
    prune_configs_by={"early_config_prune": _prune_configs_dcnv4},
)
@triton.jit
def _dcnv4_bwd_offset_kernel(
    value_ptr,
    offset_ptr,
    grad_out_ptr,
    grad_offset_ptr,
    grad_attn_ptr,
    H_in,
    W_in,
    H_out,
    W_out,
    G,
    D,
    stride_vb,
    stride_vh,
    stride_vw,
    stride_vc,
    stride_ob,
    stride_oh,
    stride_ow,
    stride_oc,
    stride_gob,
    stride_goh,
    stride_gow,
    stride_goc,
    stride_gobf,
    stride_gohf,
    stride_gowf,
    stride_gocf,
    stride_gab,
    stride_gaq,
    stride_gag,
    stride_gak,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    offset_scale,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REMOVE_CENTER: tl.constexpr,
    SOFTMAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
    WRITE_GRAD_ATTN: tl.constexpr,
) -> None:
    pid_q = tl.program_id(0)
    pid_gd = tl.program_id(1)
    b = tl.program_id(2)

    g = pid_gd // NUM_D_BLOCKS
    pid_d = pid_gd - g * NUM_D_BLOCKS

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    tl.max_contiguous(offs_q, BLOCK_Q)
    mask_q = offs_q < (H_out * W_out)

    h_out = offs_q // W_out
    w_out = offs_q % W_out

    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    tl.max_contiguous(d, BLOCK_D)
    mask_c = d < D
    c = g * D + d

    mask_idx = tl.arange(0, BLOCK_K)
    tl.max_contiguous(mask_idx, BLOCK_K)
    offset_base = offset_ptr + b * stride_ob + offs_q * stride_ow + g * K_TOTAL * 3

    # Load masks for softmax
    mask_vals = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)
    if SOFTMAX:
        mask_k = mask_idx < K_TOTAL
        mask_qk = mask_q[:, None] & mask_k[None, :]
        ptr_mask = offset_base[:, None] + (2 * K_TOTAL + mask_idx[None, :])
        mask_vals = tl.load(ptr_mask, mask=mask_qk, other=0.0).to(tl.float32)
        mask_vals = tl.where(mask_qk, mask_vals, -1.0e10)
        maxv = tl.max(mask_vals, axis=1)
        expv = tl.exp(mask_vals - maxv[:, None])
        denom = tl.sum(expv, axis=1)
        mask_vals = expv / denom[:, None]

    grad_offset_base = (
        grad_offset_ptr + b * stride_gobf + offs_q * stride_gowf + g * K_TOTAL * 3
    )

    half_w = (dilation_w * (KERNEL_W - 1)) // 2
    half_h = (dilation_h * (KERNEL_H - 1)) // 2
    p0_w = half_w - pad_w + w_out * stride_w
    p0_h = half_h - pad_h + h_out * stride_h
    p0_w = p0_w.to(tl.float32)
    p0_h = p0_h.to(tl.float32)
    p0_w_ = p0_w - half_w * offset_scale
    p0_h_ = p0_h - half_h * offset_scale

    center_h = KERNEL_H // 2
    center_w = KERNEL_W // 2

    if SOFTMAX:
        grad_attn = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)

    go_ptrs = (
        grad_out_ptr
        + b * stride_gob
        + h_out[:, None] * stride_goh
        + w_out[:, None] * stride_gow
        + c[None, :] * stride_goc
    )
    go = tl.load(go_ptrs, mask=(mask_q[:, None] & mask_c[None, :]), other=0.0).to(
        tl.float32,
    )

    point_idx = 0
    for i in range(KERNEL_W):
        for j in range(KERNEL_H):
            if (REMOVE_CENTER == 0) or (i != center_w or j != center_h):
                ptr_x = offset_base + 2 * point_idx
                ptr_y = offset_base + 2 * point_idx + 1
                off_x = tl.load(ptr_x, mask=mask_q, other=0.0).to(tl.float32)
                off_y = tl.load(ptr_y, mask=mask_q, other=0.0).to(tl.float32)

                attn = tl.zeros([BLOCK_Q], dtype=tl.float32)
                if SOFTMAX:
                    mask_now = (mask_idx == point_idx)[None, :]
                    attn = tl.sum(mask_vals * mask_now, axis=1)
                else:
                    ptr_mask_val = offset_base + 2 * K_TOTAL + point_idx
                    attn = tl.load(ptr_mask_val, mask=mask_q, other=0.0).to(tl.float32)

                w_im = p0_w_ + (i * dilation_w + off_x) * offset_scale
                h_im = p0_h_ + (j * dilation_h + off_y) * offset_scale

                valid = (h_im > -1) & (w_im > -1) & (h_im < H_in) & (w_im < W_in)
                valid = valid & mask_q

                h_low = tl.floor(h_im)
                w_low = tl.floor(w_im)
                h_low_i = h_low.to(tl.int32)
                w_low_i = w_low.to(tl.int32)
                h_high_i = h_low_i + 1
                w_high_i = w_low_i + 1

                lh = h_im - h_low
                lw = w_im - w_low
                hh = 1.0 - lh
                hw = 1.0 - lw

                w1 = hh * hw
                w2 = hh * lw
                w3 = lh * hw
                w4 = lh * lw

                base = value_ptr + b * stride_vb + c * stride_vc
                base = base[None, :]

                offset_val_h_low = h_low_i * stride_vh
                offset_val_w_low = w_low_i * stride_vw
                offset_base_val = offset_val_h_low + offset_val_w_low
                offset_base_val = offset_base_val[:, None]

                ptr1 = base + offset_base_val

                check_h_low = (h_low_i >= 0) & (h_low_i < H_in)
                check_w_low = (w_low_i >= 0) & (w_low_i < W_in)
                check_h_high = (h_high_i >= 0) & (h_high_i < H_in)
                check_w_high = (w_high_i >= 0) & (w_high_i < W_in)

                valid_bc = valid[:, None]
                mask_c_bc = mask_c[None, :]

                mask1 = (
                    mask_c_bc & valid_bc & (check_h_low[:, None] & check_w_low[:, None])
                )
                v1 = tl.load(ptr1, mask=mask1, other=0.0).to(tl.float32)

                mask2 = (
                    mask_c_bc
                    & valid_bc
                    & (check_h_low[:, None] & check_w_high[:, None])
                )
                v2 = tl.load(ptr1 + stride_vw, mask=mask2, other=0.0).to(tl.float32)

                mask3 = (
                    mask_c_bc
                    & valid_bc
                    & (check_h_high[:, None] & check_w_low[:, None])
                )
                v3 = tl.load(ptr1 + stride_vh, mask=mask3, other=0.0).to(tl.float32)

                mask4 = (
                    mask_c_bc
                    & valid_bc
                    & (check_h_high[:, None] & check_w_high[:, None])
                )
                v4 = tl.load(ptr1 + stride_vh + stride_vw, mask=mask4, other=0.0).to(
                    tl.float32,
                )

                interp = tl.fma(
                    v1,
                    w1[:, None],
                    tl.fma(v2, w2[:, None], tl.fma(v3, w3[:, None], v4 * w4[:, None])),
                )

                grad_attn_val = tl.sum(go * interp, axis=1)

                if SOFTMAX:
                    if WRITE_GRAD_ATTN:
                        ptr_grad_attn = (
                            grad_attn_ptr
                            + b * stride_gab
                            + offs_q * stride_gaq
                            + g * stride_gag
                            + point_idx * stride_gak
                        )
                        tl.atomic_add(ptr_grad_attn, grad_attn_val, mask=mask_q)
                    else:
                        col_mask = (mask_idx == point_idx)[None, :]
                        grad_attn = tl.where(
                            col_mask,
                            grad_attn_val[:, None],
                            grad_attn,
                        )
                else:
                    ptr_grad_mask = grad_offset_base + (2 * K_TOTAL + point_idx)
                    if NUM_D_BLOCKS == 1:
                        tl.store(ptr_grad_mask, grad_attn_val, mask=mask_q)
                    else:
                        tl.atomic_add(ptr_grad_mask, grad_attn_val, mask=mask_q)

                grad_w = (
                    (-hh[:, None]) * v1
                    + hh[:, None] * v2
                    + (-lh[:, None]) * v3
                    + lh[:, None] * v4
                )
                grad_h = (
                    (-hw[:, None]) * v1
                    + (-lw[:, None]) * v2
                    + hw[:, None] * v3
                    + lw[:, None] * v4
                )

                grad_off_w_val = offset_scale * attn * tl.sum(go * grad_w, axis=1)
                grad_off_h_val = offset_scale * attn * tl.sum(go * grad_h, axis=1)

                ptr_grad_w = grad_offset_base + point_idx * 2
                ptr_grad_h = grad_offset_base + point_idx * 2 + 1
                if NUM_D_BLOCKS == 1:
                    tl.store(ptr_grad_w, grad_off_w_val, mask=mask_q)
                    tl.store(ptr_grad_h, grad_off_h_val, mask=mask_q)
                else:
                    tl.atomic_add(ptr_grad_w, grad_off_w_val, mask=mask_q)
                    tl.atomic_add(ptr_grad_h, grad_off_h_val, mask=mask_q)

                point_idx += 1

    if SOFTMAX and (not WRITE_GRAD_ATTN):
        grad_attn_scaled = grad_attn * mask_vals
        sum_scaled = tl.sum(grad_attn_scaled, axis=1)
        grad_mask = grad_attn_scaled - mask_vals * sum_scaled[:, None]

        ptr_grad_mask = grad_offset_base[:, None] + (2 * K_TOTAL + mask_idx[None, :])
        tl.store(
            ptr_grad_mask,
            grad_mask,
            mask=(mask_q[:, None] & (mask_idx[None, :] < K_TOTAL)),
        )


@triton.autotune(
    configs=_get_autotune_config(),
    key=["H_out", "W_out", "G"],
)
@triton.jit
def _dcnv4_softmax_bwd_kernel(
    offset_ptr,
    grad_attn_ptr,
    grad_offset_ptr,
    H_out,
    W_out,
    G,
    stride_ob,
    stride_oh,
    stride_ow,
    stride_oc,
    stride_gab,
    stride_gaq,
    stride_gag,
    stride_gak,
    stride_gobf,
    stride_gohf,
    stride_gowf,
    stride_gocf,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_Q: tl.constexpr,
) -> None:
    pid_q = tl.program_id(0)
    g = tl.program_id(1)
    b = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    tl.max_contiguous(offs_q, BLOCK_Q)
    mask_q = offs_q < (H_out * W_out)

    mask_idx = tl.arange(0, BLOCK_K)
    tl.max_contiguous(mask_idx, BLOCK_K)
    mask_k = mask_idx < K_TOTAL
    mask_qk = mask_q[:, None] & mask_k[None, :]

    offset_base = offset_ptr + b * stride_ob + offs_q * stride_ow + g * K_TOTAL * 3
    ptr_mask = offset_base[:, None] + (2 * K_TOTAL + mask_idx[None, :])
    mask_vals = tl.load(ptr_mask, mask=mask_qk, other=0.0).to(tl.float32)
    mask_vals = tl.where(mask_qk, mask_vals, -1.0e10)
    maxv = tl.max(mask_vals, axis=1)
    expv = tl.exp(mask_vals - maxv[:, None])
    denom = tl.sum(expv, axis=1)
    mask_vals = expv / denom[:, None]

    grad_attn_ptrs = (
        grad_attn_ptr
        + b * stride_gab
        + offs_q[:, None] * stride_gaq
        + g * stride_gag
        + mask_idx[None, :] * stride_gak
    )
    grad_attn = tl.load(grad_attn_ptrs, mask=mask_qk, other=0.0).to(tl.float32)

    grad_attn_scaled = grad_attn * mask_vals
    sum_scaled = tl.sum(grad_attn_scaled, axis=1)
    grad_mask = grad_attn_scaled - mask_vals * sum_scaled[:, None]

    grad_offset_base = (
        grad_offset_ptr + b * stride_gobf + offs_q * stride_gowf + g * K_TOTAL * 3
    )
    ptr_grad_mask = grad_offset_base[:, None] + (2 * K_TOTAL + mask_idx[None, :])
    tl.store(ptr_grad_mask, grad_mask, mask=mask_qk)


def dcnv4_forward(
    value: torch.Tensor,
    offset: torch.Tensor,
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
    remove_center: int,
    softmax: bool,
) -> torch.Tensor:
    value = value.contiguous()
    offset = offset.contiguous()

    B, H_in, W_in, C = value.shape
    if group * group_channels != C:
        msg = "Input channels must equal group * group_channels."
        raise ValueError(msg)

    H_out, W_out = _compute_output_hw(
        H_in,
        W_in,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
    )

    if offset.shape[1] != H_out or offset.shape[2] != W_out:
        msg = "Offset spatial shape must match output spatial shape."
        raise ValueError(msg)

    k_total = kernel_h * kernel_w - int(remove_center)
    needed = group * k_total * 3
    if offset.shape[3] < needed:
        msg = "Offset last dimension is too small for group * K * 3."
        raise ValueError(msg)

    output = torch.zeros((B, H_out, W_out, C), device=value.device, dtype=torch.float32)

    BLOCK_D = _choose_block_d(group_channels)
    num_d_blocks = triton.cdiv(group_channels, BLOCK_D)
    BLOCK_K = _next_power_of_2(k_total)

    # Grid now depends on BLOCK_Q
    def grid(META):
        return (
            triton.cdiv(H_out * W_out, META["BLOCK_Q"]),
            group * num_d_blocks,
            B,
        )

    _dcnv4_fwd_kernel[grid](
        value,
        offset,
        output,
        H_in,
        W_in,
        H_out,
        W_out,
        group,
        group_channels,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        offset.stride(0),
        offset.stride(1),
        offset.stride(2),
        offset.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        offset_scale,
        KERNEL_H=kernel_h,
        KERNEL_W=kernel_w,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        REMOVE_CENTER=int(remove_center),
        SOFTMAX=bool(softmax),
        BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=num_d_blocks,
    )

    return output.to(dtype=value.dtype)


def dcnv4_backward(
    value: torch.Tensor,
    offset: torch.Tensor,
    grad_output: torch.Tensor,
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
    remove_center: int,
    softmax: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.contiguous()
    offset = offset.contiguous()
    grad_output = grad_output.contiguous()

    B, H_in, W_in, _C = value.shape
    H_out, W_out = grad_output.shape[1], grad_output.shape[2]

    k_total = kernel_h * kernel_w - int(remove_center)
    grad_input = torch.zeros_like(value, dtype=torch.float32)
    grad_offset = torch.zeros_like(offset, dtype=torch.float32)

    BLOCK_D = _choose_block_d(group_channels)
    num_d_blocks = triton.cdiv(group_channels, BLOCK_D)
    BLOCK_K = _next_power_of_2(k_total)
    use_grad_attn = bool(softmax) and num_d_blocks > 1
    grad_attn = None
    if use_grad_attn:
        grad_attn = torch.zeros(
            (B, H_out * W_out, group, k_total),
            device=value.device,
            dtype=torch.float32,
        )

    def grid(META):
        return (
            triton.cdiv(H_out * W_out, META["BLOCK_Q"]),
            group * num_d_blocks,
            B,
        )

    _dcnv4_bwd_input_kernel[grid](
        offset,
        grad_output,
        grad_input,
        H_in,
        W_in,
        H_out,
        W_out,
        group,
        group_channels,
        offset.stride(0),
        offset.stride(1),
        offset.stride(2),
        offset.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        offset_scale,
        KERNEL_H=kernel_h,
        KERNEL_W=kernel_w,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        REMOVE_CENTER=int(remove_center),
        SOFTMAX=bool(softmax),
        BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=num_d_blocks,
    )

    _dcnv4_bwd_offset_kernel[grid](
        value,
        offset,
        grad_output,
        grad_offset,
        grad_attn if grad_attn is not None else grad_offset,
        H_in,
        W_in,
        H_out,
        W_out,
        group,
        group_channels,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        offset.stride(0),
        offset.stride(1),
        offset.stride(2),
        offset.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_offset.stride(0),
        grad_offset.stride(1),
        grad_offset.stride(2),
        grad_offset.stride(3),
        (grad_attn.stride(0) if grad_attn is not None else 0),
        (grad_attn.stride(1) if grad_attn is not None else 0),
        (grad_attn.stride(2) if grad_attn is not None else 0),
        (grad_attn.stride(3) if grad_attn is not None else 0),
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        offset_scale,
        KERNEL_H=kernel_h,
        KERNEL_W=kernel_w,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        REMOVE_CENTER=int(remove_center),
        SOFTMAX=bool(softmax),
        BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=num_d_blocks,
        WRITE_GRAD_ATTN=use_grad_attn,
    )

    if use_grad_attn:

        def grid_softmax(META):
            return (
                triton.cdiv(H_out * W_out, META["BLOCK_Q"]),
                group,
                B,
            )

        _dcnv4_softmax_bwd_kernel[grid_softmax](
            offset,
            grad_attn,
            grad_offset,
            H_out,
            W_out,
            group,
            offset.stride(0),
            offset.stride(1),
            offset.stride(2),
            offset.stride(3),
            grad_attn.stride(0),
            grad_attn.stride(1),
            grad_attn.stride(2),
            grad_attn.stride(3),
            grad_offset.stride(0),
            grad_offset.stride(1),
            grad_offset.stride(2),
            grad_offset.stride(3),
            K_TOTAL=k_total,
            BLOCK_K=BLOCK_K,
        )

    if value.dtype in (torch.float16, torch.bfloat16):
        return grad_input.to(value.dtype), grad_offset.to(offset.dtype)
    return grad_input, grad_offset
