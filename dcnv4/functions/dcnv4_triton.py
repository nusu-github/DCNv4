# ------------------------------------------------------------------------------------------------
# DCNv4 Triton backend
# Copyright (c) 2024
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
"""Triton implementation for DCNv4 forward/backward.

This backend mirrors the CUDA kernels for 3x3 DCNv4 with optional center removal.
It is designed for portability across GPU architectures without recompilation.
"""

from __future__ import annotations

import torch

try:  # Optional dependency
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - handled by availability checks
    triton = None
    tl = None


def is_triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def triton_supports(kernel_h: int, kernel_w: int, remove_center: int) -> bool:
    return kernel_h == 3 and kernel_w == 3 and remove_center in (0, 1)


if triton is not None:
    _DCNV4_AUTOTUNE_CONFIGS = [
        triton.Config(
            {"BLOCK_D": 16, "BLOCK_K": 16, "BLOCK_Q": 2},
            num_warps=2,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_D": 16, "BLOCK_K": 16, "BLOCK_Q": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_D": 32, "BLOCK_K": 16, "BLOCK_Q": 2},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_D": 32, "BLOCK_K": 16, "BLOCK_Q": 4},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_D": 32, "BLOCK_K": 16, "BLOCK_Q": 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_D": 64, "BLOCK_K": 16, "BLOCK_Q": 2},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_D": 64, "BLOCK_K": 16, "BLOCK_Q": 4},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_D": 64, "BLOCK_K": 16, "BLOCK_Q": 8},
            num_warps=8,
            num_stages=4,
        ),
    ]

    @triton.autotune(configs=_DCNV4_AUTOTUNE_CONFIGS, key=["D", "BLOCK_Q"])
    @triton.jit
    def _dcnv4_forward_kernel(
        value_ptr,
        offset_ptr,
        output_ptr,
        stride_v_b,
        stride_v_h,
        stride_v_w,
        stride_v_c,
        stride_o_b,
        stride_o_h,
        stride_o_w,
        stride_o_c,
        stride_out_b,
        stride_out_h,
        stride_out_w,
        stride_out_c,
        B,
        H_in,
        W_in,
        H_out,
        W_out,
        G,
        D,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        offset_scale,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        SOFTMAX: tl.constexpr,
    ) -> None:
        pid_q = tl.program_id(0)
        pid_bg = tl.program_id(1)
        pid_db = tl.program_id(2)

        q = pid_q
        bg = pid_bg
        d_block = pid_db

        b = bg // G
        g = bg - b * G

        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        c_offsets = g * D + offs_d

        offs_q = q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        mask_q = offs_q < H_out * W_out
        out_h = offs_q // W_out
        out_w = offs_q - out_h * W_out

        # offsets/masks
        offset_base = (
            offset_ptr
            + b * stride_o_b
            + out_h * stride_o_h
            + out_w * stride_o_w
            + (g * K * 3) * stride_o_c
        )
        k_idx = tl.arange(0, BLOCK_K)
        mask_k = k_idx < K

        safe_k = tl.where(mask_k, k_idx, 0)
        offset_x = tl.load(
            offset_base[:, None] + (safe_k[None, :] * 2) * stride_o_c,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        offset_y = tl.load(
            offset_base[:, None] + (safe_k[None, :] * 2 + 1) * stride_o_c,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        mask_logits = tl.load(
            offset_base[:, None] + (2 * K + safe_k[None, :]) * stride_o_c,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        if SOFTMAX:
            mask_qk = mask_q[:, None] & mask_k[None, :]
            masked_logits = tl.where(mask_qk, mask_logits, -1.0e20)
            m = tl.max(masked_logits, axis=1)
            ex = tl.exp(masked_logits - m[:, None]) * mask_qk
            denom = tl.sum(ex, axis=1)
            denom = tl.where(mask_q, denom, 1.0)
            attn = ex / denom[:, None]
        else:
            attn = mask_logits

        if K == 9:
            dx = k_idx // 3
            dy = k_idx % 3
        else:
            dx = tl.where(k_idx < 3, 0, tl.where(k_idx < 5, 1, 2))
            dy = tl.where(
                k_idx < 3,
                k_idx,
                tl.where(k_idx < 5, k_idx * 2 - 6, k_idx - 5),
            )
        dx = dx.to(tl.float32)
        dy = dy.to(tl.float32)

        base_h = (dilation_h * (3 - 1)) // 2
        base_w = (dilation_w * (3 - 1)) // 2
        p0_h = base_h - pad_h + out_h * stride_h
        p0_w = base_w - pad_w + out_w * stride_w
        p0_h = p0_h - base_h * offset_scale
        p0_w = p0_w - base_w * offset_scale

        w_im = p0_w[:, None] + (dx[None, :] * dilation_w + offset_x) * offset_scale
        h_im = p0_h[:, None] + (dy[None, :] * dilation_h + offset_y) * offset_scale
        valid = (
            mask_q[:, None]
            & mask_k[None, :]
            & (h_im > -1)
            & (w_im > -1)
            & (h_im < H_in)
            & (w_im < W_in)
        )

        h0 = tl.floor(h_im).to(tl.int32)
        w0 = tl.floor(w_im).to(tl.int32)
        h1 = h0 + 1
        w1 = w0 + 1

        lh = h_im - h0.to(tl.float32)
        lw = w_im - w0.to(tl.float32)
        hh = 1.0 - lh
        hw = 1.0 - lw

        w1v = hh * hw
        w2v = hh * lw
        w3v = lh * hw
        w4v = lh * lw

        h0c = tl.maximum(tl.minimum(h0, H_in - 1), 0)
        w0c = tl.maximum(tl.minimum(w0, W_in - 1), 0)
        h1c = tl.maximum(tl.minimum(h1, H_in - 1), 0)
        w1c = tl.maximum(tl.minimum(w1, W_in - 1), 0)

        base_ptr = value_ptr + b * stride_v_b
        ptr_base = base_ptr + c_offsets[None, None, :] * stride_v_c

        mask_v1 = valid & (h0 >= 0) & (w0 >= 0)
        mask_v2 = valid & (h0 >= 0) & (w1 < W_in)
        mask_v3 = valid & (h1 < H_in) & (w0 >= 0)
        mask_v4 = valid & (h1 < H_in) & (w1 < W_in)

        v1 = tl.load(
            ptr_base + h0c[:, :, None] * stride_v_h + w0c[:, :, None] * stride_v_w,
            mask=mask_v1[:, :, None] & mask_d[None, None, :],
            other=0.0,
        )
        v2 = tl.load(
            ptr_base + h0c[:, :, None] * stride_v_h + w1c[:, :, None] * stride_v_w,
            mask=mask_v2[:, :, None] & mask_d[None, None, :],
            other=0.0,
        )
        v3 = tl.load(
            ptr_base + h1c[:, :, None] * stride_v_h + w0c[:, :, None] * stride_v_w,
            mask=mask_v3[:, :, None] & mask_d[None, None, :],
            other=0.0,
        )
        v4 = tl.load(
            ptr_base + h1c[:, :, None] * stride_v_h + w1c[:, :, None] * stride_v_w,
            mask=mask_v4[:, :, None] & mask_d[None, None, :],
            other=0.0,
        )

        v1 = v1.to(tl.float32)
        v2 = v2.to(tl.float32)
        v3 = v3.to(tl.float32)
        v4 = v4.to(tl.float32)

        sample = (
            w1v[:, :, None] * v1
            + w2v[:, :, None] * v2
            + w3v[:, :, None] * v3
            + w4v[:, :, None] * v4
        )
        acc = tl.sum(attn[:, :, None] * sample, axis=1)

        out_ptr = (
            output_ptr
            + b * stride_out_b
            + out_h[:, None] * stride_out_h
            + out_w[:, None] * stride_out_w
            + c_offsets[None, :] * stride_out_c
        )
        tl.store(out_ptr, acc, mask=mask_q[:, None] & mask_d[None, :])

    @triton.autotune(
        configs=_DCNV4_AUTOTUNE_CONFIGS,
        key=["D", "BLOCK_Q"],
        reset_to_zero=["grad_in_ptr"],
    )
    @triton.jit
    def _dcnv4_backward_input_kernel(
        offset_ptr,
        grad_out_ptr,
        grad_in_ptr,
        stride_o_b,
        stride_o_h,
        stride_o_w,
        stride_o_c,
        stride_go_b,
        stride_go_h,
        stride_go_w,
        stride_go_c,
        stride_gi_b,
        stride_gi_h,
        stride_gi_w,
        stride_gi_c,
        B,
        H_in,
        W_in,
        H_out,
        W_out,
        G,
        D,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        offset_scale,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        SOFTMAX: tl.constexpr,
    ) -> None:
        pid_q = tl.program_id(0)
        pid_bg = tl.program_id(1)
        pid_db = tl.program_id(2)

        q = pid_q
        bg = pid_bg
        d_block = pid_db

        b = bg // G
        g = bg - b * G

        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        c_offsets = g * D + offs_d

        offs_q = q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        mask_q = offs_q < H_out * W_out
        out_h = offs_q // W_out
        out_w = offs_q - out_h * W_out

        offset_base = (
            offset_ptr
            + b * stride_o_b
            + out_h * stride_o_h
            + out_w * stride_o_w
            + (g * K * 3) * stride_o_c
        )
        k_idx = tl.arange(0, BLOCK_K)
        mask_k = k_idx < K

        safe_k = tl.where(mask_k, k_idx, 0)
        offset_x = tl.load(
            offset_base[:, None] + (safe_k[None, :] * 2) * stride_o_c,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        offset_y = tl.load(
            offset_base[:, None] + (safe_k[None, :] * 2 + 1) * stride_o_c,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        mask_logits = tl.load(
            offset_base[:, None] + (2 * K + safe_k[None, :]) * stride_o_c,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        if SOFTMAX:
            mask_qk = mask_q[:, None] & mask_k[None, :]
            masked_logits = tl.where(mask_qk, mask_logits, -1.0e20)
            m = tl.max(masked_logits, axis=1)
            ex = tl.exp(masked_logits - m[:, None]) * mask_qk
            denom = tl.sum(ex, axis=1)
            denom = tl.where(mask_q, denom, 1.0)
            attn = ex / denom[:, None]
        else:
            attn = mask_logits

        if K == 9:
            dx = k_idx // 3
            dy = k_idx % 3
        else:
            dx = tl.where(k_idx < 3, 0, tl.where(k_idx < 5, 1, 2))
            dy = tl.where(
                k_idx < 3,
                k_idx,
                tl.where(k_idx < 5, k_idx * 2 - 6, k_idx - 5),
            )
        dx = dx.to(tl.float32)
        dy = dy.to(tl.float32)

        base_h = (dilation_h * (3 - 1)) // 2
        base_w = (dilation_w * (3 - 1)) // 2
        p0_h = base_h - pad_h + out_h * stride_h
        p0_w = base_w - pad_w + out_w * stride_w
        p0_h = p0_h - base_h * offset_scale
        p0_w = p0_w - base_w * offset_scale

        grad_out_ptr_block = (
            grad_out_ptr
            + b * stride_go_b
            + out_h[:, None] * stride_go_h
            + out_w[:, None] * stride_go_w
            + c_offsets[None, :] * stride_go_c
        )
        top_grad = tl.load(
            grad_out_ptr_block,
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        w_im = p0_w[:, None] + (dx[None, :] * dilation_w + offset_x) * offset_scale
        h_im = p0_h[:, None] + (dy[None, :] * dilation_h + offset_y) * offset_scale
        valid = (
            mask_q[:, None]
            & mask_k[None, :]
            & (h_im > -1)
            & (w_im > -1)
            & (h_im < H_in)
            & (w_im < W_in)
        )

        h0 = tl.floor(h_im).to(tl.int32)
        w0 = tl.floor(w_im).to(tl.int32)
        h1 = h0 + 1
        w1 = w0 + 1

        lh = h_im - h0.to(tl.float32)
        lw = w_im - w0.to(tl.float32)
        hh = 1.0 - lh
        hw = 1.0 - lw

        w1v = hh * hw
        w2v = hh * lw
        w3v = lh * hw
        w4v = lh * lw

        h0c = tl.maximum(tl.minimum(h0, H_in - 1), 0)
        w0c = tl.maximum(tl.minimum(w0, W_in - 1), 0)
        h1c = tl.maximum(tl.minimum(h1, H_in - 1), 0)
        w1c = tl.maximum(tl.minimum(w1, W_in - 1), 0)

        base_ptr = grad_in_ptr + b * stride_gi_b
        ptr_base = base_ptr + c_offsets[None, None, :] * stride_gi_c

        top_grad_b = top_grad[:, None, :]
        grad1 = top_grad_b * attn[:, :, None] * w1v[:, :, None]
        grad2 = top_grad_b * attn[:, :, None] * w2v[:, :, None]
        grad3 = top_grad_b * attn[:, :, None] * w3v[:, :, None]
        grad4 = top_grad_b * attn[:, :, None] * w4v[:, :, None]

        mask_v1 = valid & (h0 >= 0) & (w0 >= 0)
        mask_v2 = valid & (h0 >= 0) & (w1 < W_in)
        mask_v3 = valid & (h1 < H_in) & (w0 >= 0)
        mask_v4 = valid & (h1 < H_in) & (w1 < W_in)

        tl.atomic_add(
            ptr_base + h0c[:, :, None] * stride_gi_h + w0c[:, :, None] * stride_gi_w,
            grad1,
            mask=mask_v1[:, :, None] & mask_d[None, None, :],
        )
        tl.atomic_add(
            ptr_base + h0c[:, :, None] * stride_gi_h + w1c[:, :, None] * stride_gi_w,
            grad2,
            mask=mask_v2[:, :, None] & mask_d[None, None, :],
        )
        tl.atomic_add(
            ptr_base + h1c[:, :, None] * stride_gi_h + w0c[:, :, None] * stride_gi_w,
            grad3,
            mask=mask_v3[:, :, None] & mask_d[None, None, :],
        )
        tl.atomic_add(
            ptr_base + h1c[:, :, None] * stride_gi_h + w1c[:, :, None] * stride_gi_w,
            grad4,
            mask=mask_v4[:, :, None] & mask_d[None, None, :],
        )

    @triton.autotune(
        configs=_DCNV4_AUTOTUNE_CONFIGS,
        key=["D", "BLOCK_Q"],
        reset_to_zero=["grad_off_ptr"],
    )
    @triton.jit
    def _dcnv4_backward_offset_kernel(
        value_ptr,
        offset_ptr,
        grad_out_ptr,
        grad_off_ptr,
        stride_v_b,
        stride_v_h,
        stride_v_w,
        stride_v_c,
        stride_o_b,
        stride_o_h,
        stride_o_w,
        stride_o_c,
        stride_go_b,
        stride_go_h,
        stride_go_w,
        stride_go_c,
        stride_gm_b,
        stride_gm_h,
        stride_gm_w,
        stride_gm_c,
        B,
        H_in,
        W_in,
        H_out,
        W_out,
        G,
        D,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        offset_scale,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        SOFTMAX: tl.constexpr,
    ) -> None:
        pid_q = tl.program_id(0)
        pid_bg = tl.program_id(1)
        pid_db = tl.program_id(2)

        q = pid_q
        bg = pid_bg
        d_block = pid_db

        b = bg // G
        g = bg - b * G

        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        c_offsets = g * D + offs_d

        offs_q = q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        mask_q = offs_q < H_out * W_out
        out_h = offs_q // W_out
        out_w = offs_q - out_h * W_out

        offset_base = (
            offset_ptr
            + b * stride_o_b
            + out_h * stride_o_h
            + out_w * stride_o_w
            + (g * K * 3) * stride_o_c
        )
        k_idx = tl.arange(0, BLOCK_K)
        mask_k = k_idx < K

        safe_k = tl.where(mask_k, k_idx, 0)
        offset_x = tl.load(
            offset_base[:, None] + (safe_k[None, :] * 2) * stride_o_c,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        offset_y = tl.load(
            offset_base[:, None] + (safe_k[None, :] * 2 + 1) * stride_o_c,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        mask_logits = tl.load(
            offset_base[:, None] + (2 * K + safe_k[None, :]) * stride_o_c,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        if SOFTMAX:
            mask_qk = mask_q[:, None] & mask_k[None, :]
            masked_logits = tl.where(mask_qk, mask_logits, -1.0e20)
            m = tl.max(masked_logits, axis=1)
            ex = tl.exp(masked_logits - m[:, None]) * mask_qk
            denom = tl.sum(ex, axis=1)
            denom = tl.where(mask_q, denom, 1.0)
            attn = ex / denom[:, None]
        else:
            attn = mask_logits

        if K == 9:
            dx = k_idx // 3
            dy = k_idx % 3
        else:
            dx = tl.where(k_idx < 3, 0, tl.where(k_idx < 5, 1, 2))
            dy = tl.where(
                k_idx < 3,
                k_idx,
                tl.where(k_idx < 5, k_idx * 2 - 6, k_idx - 5),
            )
        dx = dx.to(tl.float32)
        dy = dy.to(tl.float32)

        base_h = (dilation_h * (3 - 1)) // 2
        base_w = (dilation_w * (3 - 1)) // 2
        p0_h = base_h - pad_h + out_h * stride_h
        p0_w = base_w - pad_w + out_w * stride_w
        p0_h = p0_h - base_h * offset_scale
        p0_w = p0_w - base_w * offset_scale

        grad_out_ptr_block = (
            grad_out_ptr
            + b * stride_go_b
            + out_h[:, None] * stride_go_h
            + out_w[:, None] * stride_go_w
            + c_offsets[None, :] * stride_go_c
        )
        top_grad = tl.load(
            grad_out_ptr_block,
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        grad_off_base = (
            grad_off_ptr
            + b * stride_gm_b
            + out_h * stride_gm_h
            + out_w * stride_gm_w
            + (g * K * 3) * stride_gm_c
        )

        w_im = p0_w[:, None] + (dx[None, :] * dilation_w + offset_x) * offset_scale
        h_im = p0_h[:, None] + (dy[None, :] * dilation_h + offset_y) * offset_scale
        valid = (
            mask_q[:, None]
            & mask_k[None, :]
            & (h_im > -1)
            & (w_im > -1)
            & (h_im < H_in)
            & (w_im < W_in)
        )

        h0 = tl.floor(h_im).to(tl.int32)
        w0 = tl.floor(w_im).to(tl.int32)
        h1 = h0 + 1
        w1 = w0 + 1

        lh = h_im - h0.to(tl.float32)
        lw = w_im - w0.to(tl.float32)
        hh = 1.0 - lh
        hw = 1.0 - lw

        w1v = hh * hw
        w2v = hh * lw
        w3v = lh * hw
        w4v = lh * lw

        h0c = tl.maximum(tl.minimum(h0, H_in - 1), 0)
        w0c = tl.maximum(tl.minimum(w0, W_in - 1), 0)
        h1c = tl.maximum(tl.minimum(h1, H_in - 1), 0)
        w1c = tl.maximum(tl.minimum(w1, W_in - 1), 0)

        base_ptr = value_ptr + b * stride_v_b
        ptr_base = base_ptr + c_offsets[None, None, :] * stride_v_c

        mask_v1 = valid & (h0 >= 0) & (w0 >= 0)
        mask_v2 = valid & (h0 >= 0) & (w1 < W_in)
        mask_v3 = valid & (h1 < H_in) & (w0 >= 0)
        mask_v4 = valid & (h1 < H_in) & (w1 < W_in)

        v1 = tl.load(
            ptr_base + h0c[:, :, None] * stride_v_h + w0c[:, :, None] * stride_v_w,
            mask=mask_v1[:, :, None] & mask_d[None, None, :],
            other=0.0,
        )
        v2 = tl.load(
            ptr_base + h0c[:, :, None] * stride_v_h + w1c[:, :, None] * stride_v_w,
            mask=mask_v2[:, :, None] & mask_d[None, None, :],
            other=0.0,
        )
        v3 = tl.load(
            ptr_base + h1c[:, :, None] * stride_v_h + w0c[:, :, None] * stride_v_w,
            mask=mask_v3[:, :, None] & mask_d[None, None, :],
            other=0.0,
        )
        v4 = tl.load(
            ptr_base + h1c[:, :, None] * stride_v_h + w1c[:, :, None] * stride_v_w,
            mask=mask_v4[:, :, None] & mask_d[None, None, :],
            other=0.0,
        )

        v1 = v1.to(tl.float32)
        v2 = v2.to(tl.float32)
        v3 = v3.to(tl.float32)
        v4 = v4.to(tl.float32)

        top_grad_b = top_grad[:, None, :]
        sample = (
            w1v[:, :, None] * v1
            + w2v[:, :, None] * v2
            + w3v[:, :, None] * v3
            + w4v[:, :, None] * v4
        )
        grad_z = tl.sum(sample * top_grad_b, axis=2)

        grad_w_weight = (
            -hh[:, :, None] * v1
            + hh[:, :, None] * v2
            - lh[:, :, None] * v3
            + lh[:, :, None] * v4
        )
        grad_h_weight = (
            -hw[:, :, None] * v1
            - lw[:, :, None] * v2
            + hw[:, :, None] * v3
            + lw[:, :, None] * v4
        )

        grad_x = tl.sum(grad_w_weight * top_grad_b, axis=2) * attn * offset_scale
        grad_y = tl.sum(grad_h_weight * top_grad_b, axis=2) * attn * offset_scale

        grad_x = tl.where(valid, grad_x, 0.0)
        grad_y = tl.where(valid, grad_y, 0.0)
        grad_z = tl.where(valid, grad_z, 0.0)

        tl.atomic_add(
            grad_off_base[:, None] + (k_idx[None, :] * 2) * stride_gm_c,
            grad_x,
            mask=mask_q[:, None] & mask_k[None, :],
        )
        tl.atomic_add(
            grad_off_base[:, None] + (k_idx[None, :] * 2 + 1) * stride_gm_c,
            grad_y,
            mask=mask_q[:, None] & mask_k[None, :],
        )
        tl.atomic_add(
            grad_off_base[:, None] + (2 * K + k_idx[None, :]) * stride_gm_c,
            grad_z,
            mask=mask_q[:, None] & mask_k[None, :],
        )

    @triton.jit
    def _dcnv4_backward_softmax_kernel(
        offset_ptr,
        grad_off_ptr,
        stride_o_b,
        stride_o_h,
        stride_o_w,
        stride_o_c,
        stride_gm_b,
        stride_gm_h,
        stride_gm_w,
        stride_gm_c,
        B,
        H_out,
        W_out,
        G,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid_q = tl.program_id(0)
        pid_bg = tl.program_id(1)

        q = pid_q
        bg = pid_bg

        b = bg // G
        g = bg - b * G

        out_h = q // W_out
        out_w = q - out_h * W_out

        offset_base = (
            offset_ptr
            + b * stride_o_b
            + out_h * stride_o_h
            + out_w * stride_o_w
            + (g * K * 3) * stride_o_c
        )
        k_idx = tl.arange(0, BLOCK_K)
        mask_k = k_idx < K
        safe_k = tl.where(mask_k, k_idx, 0)
        mask_logits = tl.load(
            offset_base + (2 * K + safe_k) * stride_o_c,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        masked_logits = tl.where(mask_k, mask_logits, -1.0e20)
        m = tl.max(masked_logits, axis=0)
        ex = tl.exp(mask_logits - m) * mask_k
        denom = tl.sum(ex, axis=0)
        attn = ex / denom

        grad_base = (
            grad_off_ptr
            + b * stride_gm_b
            + out_h * stride_gm_h
            + out_w * stride_gm_w
            + (g * K * 3) * stride_gm_c
        )
        grad_weights = tl.load(
            grad_base + (2 * K + k_idx) * stride_gm_c,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        sum_val = tl.sum(attn * grad_weights, axis=0)
        grad_logits = attn * (grad_weights - sum_val)

        tl.store(
            grad_base + (2 * K + k_idx) * stride_gm_c,
            grad_logits,
            mask=mask_k,
        )


def _check_contiguous(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_contiguous():
        msg = f"{name} tensor has to be contiguous"
        raise RuntimeError(msg)


def dcnv4_forward_triton(
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
    del im2col_step
    if triton is None:
        msg = "Triton is not available"
        raise RuntimeError(msg)
    _check_contiguous(input, "input")
    _check_contiguous(offset_mask, "offset_mask")
    if not input.is_cuda or not offset_mask.is_cuda:
        msg = "input and offset_mask must be CUDA tensors"
        raise RuntimeError(msg)
    if input.dtype != offset_mask.dtype:
        msg = "value and p_offset must have the same dtype"
        raise RuntimeError(msg)
    if input.device != offset_mask.device:
        msg = "value and p_offset must be on the same device"
        raise RuntimeError(msg)
    if not triton_supports(kernel_h, kernel_w, remove_center):
        msg = "Triton backend only supports 3x3 kernels"
        raise RuntimeError(msg)

    B, H_in, W_in, C = input.shape
    padded_offset_dim = offset_mask.shape[3]
    if padded_offset_dim % 8 != 0:
        msg = f"padded_offset_dim ({padded_offset_dim}) must be divisible by 8"
        raise RuntimeError(
            msg,
        )
    H_out = (H_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    if group * group_channels != C:
        msg = (
            "Input channels and group times group channels wont match: "
            f"({C} vs {group * group_channels})."
        )
        raise RuntimeError(
            msg,
        )

    output = torch.empty((B, H_out, W_out, C), device=input.device, dtype=input.dtype)

    D = group_channels
    Q = H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(Q, meta["BLOCK_Q"]),
            B * group,
            triton.cdiv(D, meta["BLOCK_D"]),
        )

    K = kernel_h * kernel_w - int(remove_center)

    _dcnv4_forward_kernel[grid](
        input,
        offset_mask,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        offset_mask.stride(0),
        offset_mask.stride(1),
        offset_mask.stride(2),
        offset_mask.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        B,
        H_in,
        W_in,
        H_out,
        W_out,
        group,
        D,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        offset_scale,
        K=K,
        SOFTMAX=softmax,
    )
    return output


def dcnv4_backward_triton(
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
    grad_output: torch.Tensor,
    remove_center: int,
    softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del im2col_step
    if triton is None:
        msg = "Triton is not available"
        raise RuntimeError(msg)
    _check_contiguous(input, "input")
    _check_contiguous(offset_mask, "offset_mask")
    _check_contiguous(grad_output, "grad_output")
    if not input.is_cuda or not offset_mask.is_cuda or not grad_output.is_cuda:
        msg = "input, offset_mask, grad_output must be CUDA tensors"
        raise RuntimeError(msg)
    if input.dtype != offset_mask.dtype or input.dtype != grad_output.dtype:
        msg = "value, p_offset, and grad_output must have the same dtype"
        raise RuntimeError(msg)
    if input.device != offset_mask.device or input.device != grad_output.device:
        msg = "All tensors must be on the same device"
        raise RuntimeError(msg)
    if not triton_supports(kernel_h, kernel_w, remove_center):
        msg = "Triton backend only supports 3x3 kernels"
        raise RuntimeError(msg)

    B, H_in, W_in, C = input.shape
    padded_offset_dim = offset_mask.shape[3]
    if padded_offset_dim % 8 != 0:
        msg = f"padded_offset_dim ({padded_offset_dim}) must be divisible by 8"
        raise RuntimeError(
            msg,
        )
    H_out = (H_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    if group * group_channels != C:
        msg = (
            "Input channels and group times group channels wont match: "
            f"({C} vs {group * group_channels})."
        )
        raise RuntimeError(
            msg,
        )

    compute_dtype = (
        torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype
    )

    grad_input = torch.zeros_like(input, dtype=compute_dtype)
    grad_offset = torch.zeros_like(offset_mask, dtype=compute_dtype)

    D = group_channels
    Q = H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(Q, meta["BLOCK_Q"]),
            B * group,
            triton.cdiv(D, meta["BLOCK_D"]),
        )

    K = kernel_h * kernel_w - int(remove_center)
    BLOCK_K = 16

    _dcnv4_backward_input_kernel[grid](
        offset_mask,
        grad_output,
        grad_input,
        offset_mask.stride(0),
        offset_mask.stride(1),
        offset_mask.stride(2),
        offset_mask.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        B,
        H_in,
        W_in,
        H_out,
        W_out,
        group,
        D,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        offset_scale,
        K=K,
        SOFTMAX=softmax,
    )

    _dcnv4_backward_offset_kernel[grid](
        input,
        offset_mask,
        grad_output,
        grad_offset,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        offset_mask.stride(0),
        offset_mask.stride(1),
        offset_mask.stride(2),
        offset_mask.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_offset.stride(0),
        grad_offset.stride(1),
        grad_offset.stride(2),
        grad_offset.stride(3),
        B,
        H_in,
        W_in,
        H_out,
        W_out,
        group,
        D,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        offset_scale,
        K=K,
        SOFTMAX=softmax,
    )

    if softmax:
        softmax_grid = (Q, B * group)
        _dcnv4_backward_softmax_kernel[softmax_grid](
            offset_mask,
            grad_offset,
            offset_mask.stride(0),
            offset_mask.stride(1),
            offset_mask.stride(2),
            offset_mask.stride(3),
            grad_offset.stride(0),
            grad_offset.stride(1),
            grad_offset.stride(2),
            grad_offset.stride(3),
            B,
            H_out,
            W_out,
            group,
            K=K,
            BLOCK_K=BLOCK_K,
            num_warps=2,
        )

    if compute_dtype != input.dtype:
        grad_input = grad_input.to(input.dtype)
        grad_offset = grad_offset.to(input.dtype)

    return grad_input, grad_offset


__all__ = [
    "dcnv4_backward_triton",
    "dcnv4_forward_triton",
    "is_triton_available",
    "triton_supports",
]
