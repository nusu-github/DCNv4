from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import (
    _choose_block_d,
    _get_autotune_config,
    _next_power_of_2,
    _prune_configs_flash,
)


@triton.autotune(
    configs=_get_autotune_config(),
    key=["G", "D"],
    prune_configs_by={"early_config_prune": _prune_configs_flash},
)
@triton.jit
def _flash_deform_fwd_kernel(
    value_ptr,
    spatial_shapes_ptr,
    level_start_ptr,
    offset_ptr,
    output_ptr,
    N,
    Q,
    G,
    D,
    stride_vb,
    stride_vn,
    stride_vg,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_og,
    stride_od,
    stride_outb,
    stride_outq,
    stride_outg,
    stride_outd,
    L: tl.constexpr,
    K: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    mask_q = offs_q < Q

    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    tl.max_contiguous(d, BLOCK_D)
    mask_c = d < D
    c = g * D + d

    mask_idx = tl.arange(0, BLOCK_K)
    tl.max_contiguous(mask_idx, BLOCK_K)
    offset_base = offset_ptr + b * stride_ob + offs_q * stride_oq + g * stride_og

    # Load all masks (Softmax is always true for flash deform?)
    # FlashDeformAttn usually has softmax
    ptr_mask = offset_base[:, None] + (2 * K_TOTAL + mask_idx[None, :])
    mask_vals = tl.load(
        ptr_mask,
        mask=(mask_q[:, None] & (mask_idx[None, :] < K_TOTAL)),
        other=-1e10,  # Use large negative instead of -inf to avoid NaN in backward
    ).to(tl.float32)
    maxv = tl.max(mask_vals, axis=1)
    expv = tl.exp(mask_vals - maxv[:, None])
    denom = tl.sum(expv, axis=1)
    mask_vals = expv / denom[:, None]

    acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    point_idx = 0
    for li in range(L):
        spatial_h = tl.load(spatial_shapes_ptr + li * 2 + 0)
        spatial_w = tl.load(spatial_shapes_ptr + li * 2 + 1)
        level_start = tl.load(level_start_ptr + li)

        for _ki in range(K):
            # Load offsets just in time
            ptr_x = offset_base + 2 * point_idx
            ptr_y = offset_base + 2 * point_idx + 1
            off_x = tl.load(ptr_x, mask=mask_q, other=0.0).to(tl.float32)
            off_y = tl.load(ptr_y, mask=mask_q, other=0.0).to(tl.float32)

            mask_now = (mask_idx == point_idx)[None, :]
            attn = tl.sum(mask_vals * mask_now, axis=1)

            h_im = off_y * spatial_h - 0.5
            w_im = off_x * spatial_w - 0.5
            valid = (h_im > -1) & (w_im > -1) & (h_im < spatial_h) & (w_im < spatial_w)
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

            base = (
                value_ptr
                + b * stride_vb
                + (level_start + 0) * stride_vn
                + g * stride_vg
                + d * stride_vd
            )
            base = base[None, :]

            stride_vh = spatial_w * stride_vn
            stride_vw = stride_vn

            offset_base_val = h_low_i * stride_vh + w_low_i * stride_vw
            offset_base_val = offset_base_val[:, None]
            ptr1 = base + offset_base_val

            check_h_low = (h_low_i >= 0) & (h_low_i < spatial_h)
            check_w_low = (w_low_i >= 0) & (w_low_i < spatial_w)
            check_h_high = (h_high_i >= 0) & (h_high_i < spatial_h)
            check_w_high = (w_high_i >= 0) & (w_high_i < spatial_w)

            valid_bc = valid[:, None]
            mask_c_bc = mask_c[None, :]

            mask1 = mask_c_bc & valid_bc & (check_h_low[:, None] & check_w_low[:, None])
            v1 = tl.load(ptr1, mask=mask1, other=0.0).to(tl.float32)

            mask2 = (
                mask_c_bc & valid_bc & (check_h_low[:, None] & check_w_high[:, None])
            )
            v2 = tl.load(ptr1 + stride_vw, mask=mask2, other=0.0).to(tl.float32)

            mask3 = (
                mask_c_bc & valid_bc & (check_h_high[:, None] & check_w_low[:, None])
            )
            v3 = tl.load(ptr1 + stride_vh, mask=mask3, other=0.0).to(tl.float32)

            mask4 = (
                mask_c_bc & valid_bc & (check_h_high[:, None] & check_w_high[:, None])
            )
            v4 = tl.load(ptr1 + stride_vh + stride_vw, mask=mask4, other=0.0).to(
                tl.float32,
            )

            interp = (
                v1 * w1[:, None]
                + v2 * w2[:, None]
                + v3 * w3[:, None]
                + v4 * w4[:, None]
            )
            acc += attn[:, None] * interp

            point_idx += 1

    out_ptrs = (
        output_ptr
        + b * stride_outb
        + offs_q[:, None] * stride_outq
        + c[None, :] * stride_outd
    )
    tl.store(out_ptrs, acc.to(tl.float32), mask=(mask_q[:, None] & mask_c[None, :]))


@triton.autotune(
    configs=_get_autotune_config(),
    key=["G", "D"],
    reset_to_zero=["grad_input_ptr"],
    prune_configs_by={"early_config_prune": _prune_configs_flash},
)
@triton.jit
def _flash_deform_bwd_kernel(
    value_ptr,
    spatial_shapes_ptr,
    level_start_ptr,
    offset_ptr,
    grad_out_ptr,
    grad_input_ptr,
    grad_offset_ptr,
    N,
    Q,
    G,
    D,
    stride_vb,
    stride_vn,
    stride_vg,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_og,
    stride_od,
    stride_gob,
    stride_goq,
    stride_gog,
    stride_god,
    stride_gib,
    stride_gin,
    stride_gig,
    stride_gid,
    stride_gobf,
    stride_goqf,
    stride_gogf,
    stride_godf,
    L: tl.constexpr,
    K: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    mask_q = offs_q < Q

    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    tl.max_contiguous(d, BLOCK_D)
    mask_c = d < D
    c = g * D + d

    mask_idx = tl.arange(0, BLOCK_K)
    tl.max_contiguous(mask_idx, BLOCK_K)
    offset_base = offset_ptr + b * stride_ob + offs_q * stride_oq + g * stride_og

    ptr_mask = offset_base[:, None] + (2 * K_TOTAL + mask_idx[None, :])
    mask_vals = tl.load(
        ptr_mask,
        mask=(mask_q[:, None] & (mask_idx[None, :] < K_TOTAL)),
        other=-1e10,  # Use large negative instead of -inf to avoid NaN in backward
    ).to(tl.float32)
    maxv = tl.max(mask_vals, axis=1)
    expv = tl.exp(mask_vals - maxv[:, None])
    denom = tl.sum(expv, axis=1)
    mask_vals = expv / denom[:, None]

    go_ptrs = (
        grad_out_ptr
        + b * stride_gob
        + offs_q[:, None] * stride_goq
        + c[None, :] * stride_god
    )
    go = tl.load(go_ptrs, mask=(mask_q[:, None] & mask_c[None, :]), other=0.0).to(
        tl.float32,
    )

    grad_attn = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)
    grad_offset_base = (
        grad_offset_ptr + b * stride_gobf + offs_q * stride_goqf + g * stride_gogf
    )

    point_idx = 0
    for li in range(L):
        spatial_h = tl.load(spatial_shapes_ptr + li * 2 + 0)
        spatial_w = tl.load(spatial_shapes_ptr + li * 2 + 1)
        level_start = tl.load(level_start_ptr + li)

        for _ki in range(K):
            ptr_x = offset_base + 2 * point_idx
            ptr_y = offset_base + 2 * point_idx + 1
            off_x = tl.load(ptr_x, mask=mask_q, other=0.0).to(tl.float32)
            off_y = tl.load(ptr_y, mask=mask_q, other=0.0).to(tl.float32)

            mask_now = (mask_idx == point_idx)[None, :]
            attn = tl.sum(mask_vals * mask_now, axis=1)

            h_im = off_y * spatial_h - 0.5
            w_im = off_x * spatial_w - 0.5
            valid = (h_im > -1) & (w_im > -1) & (h_im < spatial_h) & (w_im < spatial_w)
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

            base = (
                value_ptr
                + b * stride_vb
                + (level_start + 0) * stride_vn
                + g * stride_vg
                + d * stride_vd
            )
            base = base[None, :]
            stride_vh = spatial_w * stride_vn
            stride_vw = stride_vn
            offset_base_val = h_low_i * stride_vh + w_low_i * stride_vw
            offset_base_val = offset_base_val[:, None]
            ptr1 = base + offset_base_val

            check_h_low = (h_low_i >= 0) & (h_low_i < spatial_h)
            check_w_low = (w_low_i >= 0) & (w_low_i < spatial_w)
            check_h_high = (h_high_i >= 0) & (h_high_i < spatial_h)
            check_w_high = (w_high_i >= 0) & (w_high_i < spatial_w)

            valid_bc = valid[:, None]
            mask_c_bc = mask_c[None, :]

            mask1 = mask_c_bc & valid_bc & (check_h_low[:, None] & check_w_low[:, None])
            v1 = tl.load(ptr1, mask=mask1, other=0.0).to(tl.float32)

            mask2 = (
                mask_c_bc & valid_bc & (check_h_low[:, None] & check_w_high[:, None])
            )
            v2 = tl.load(ptr1 + stride_vw, mask=mask2, other=0.0).to(tl.float32)

            mask3 = (
                mask_c_bc & valid_bc & (check_h_high[:, None] & check_w_low[:, None])
            )
            v3 = tl.load(ptr1 + stride_vh, mask=mask3, other=0.0).to(tl.float32)

            mask4 = (
                mask_c_bc & valid_bc & (check_h_high[:, None] & check_w_high[:, None])
            )
            v4 = tl.load(ptr1 + stride_vh + stride_vw, mask=mask4, other=0.0).to(
                tl.float32,
            )

            interp = (
                v1 * w1[:, None]
                + v2 * w2[:, None]
                + v3 * w3[:, None]
                + v4 * w4[:, None]
            )
            grad_attn_val = tl.sum(go * interp, axis=1)

            col_mask = (mask_idx == point_idx)[None, :]
            grad_attn = tl.where(col_mask, grad_attn_val[:, None], grad_attn)

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

            grad_off_w_val = spatial_w * attn * tl.sum(go * grad_w, axis=1)
            grad_off_h_val = spatial_h * attn * tl.sum(go * grad_h, axis=1)

            ptr_grad_w = grad_offset_base + point_idx * 2
            ptr_grad_h = grad_offset_base + point_idx * 2 + 1
            tl.store(ptr_grad_w, grad_off_w_val, mask=mask_q)
            tl.store(ptr_grad_h, grad_off_h_val, mask=mask_q)

            base_grad_in = (
                grad_input_ptr
                + b * stride_gib
                + (level_start + 0) * stride_gin
                + g * stride_gig
                + d * stride_gid
            )
            stride_gih = spatial_w * stride_gin
            stride_giw = stride_gin
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

    grad_attn_scaled = grad_attn * mask_vals
    sum_scaled = tl.sum(grad_attn_scaled, axis=1)
    grad_mask = grad_attn_scaled - mask_vals * sum_scaled[:, None]

    ptr_grad_mask = grad_offset_base[:, None] + (2 * K_TOTAL + mask_idx[None, :])
    tl.store(
        ptr_grad_mask,
        grad_mask,
        mask=(mask_q[:, None] & (mask_idx[None, :] < K_TOTAL)),
    )


def flash_deform_attn_forward(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_loc_attn: torch.Tensor,
    K: int,
) -> torch.Tensor:
    value = value.contiguous()
    spatial_shapes = spatial_shapes.contiguous()
    level_start_index = level_start_index.contiguous()
    sampling_loc_attn = sampling_loc_attn.contiguous()

    B, N, G, D = value.shape
    Q = sampling_loc_attn.shape[1]
    L = spatial_shapes.shape[0]

    k_total = L * K
    if sampling_loc_attn.shape[-1] < k_total * 3:
        msg = "sampling_loc_attn last dimension is too small."
        raise ValueError(msg)

    output = torch.zeros((B, Q, G * D), device=value.device, dtype=torch.float32)

    BLOCK_D = _choose_block_d(D)
    num_d_blocks = triton.cdiv(D, BLOCK_D)
    BLOCK_K = _next_power_of_2(k_total)

    def grid(META):
        return (
            triton.cdiv(Q, META["BLOCK_Q"]),
            G * num_d_blocks,
            B,
        )

    _flash_deform_fwd_kernel[grid](
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc_attn,
        output,
        N,
        Q,
        G,
        D,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        sampling_loc_attn.stride(0),
        sampling_loc_attn.stride(1),
        sampling_loc_attn.stride(2),
        sampling_loc_attn.stride(3),
        output.stride(0),
        output.stride(1),
        G * D,
        output.stride(2),
        L=L,
        K=K,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=num_d_blocks,
    )

    return output.to(dtype=value.dtype)


def flash_deform_attn_backward(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_loc_attn: torch.Tensor,
    grad_output: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.contiguous()
    spatial_shapes = spatial_shapes.contiguous()
    level_start_index = level_start_index.contiguous()
    sampling_loc_attn = sampling_loc_attn.contiguous()
    grad_output = grad_output.contiguous()

    B, N, G, D = value.shape
    Q = sampling_loc_attn.shape[1]
    L = spatial_shapes.shape[0]

    k_total = L * K
    grad_input = torch.zeros_like(value, dtype=torch.float32)
    grad_offset = torch.zeros_like(sampling_loc_attn, dtype=torch.float32)

    BLOCK_D = _choose_block_d(D)
    num_d_blocks = triton.cdiv(D, BLOCK_D)
    BLOCK_K = _next_power_of_2(k_total)

    def grid(META):
        return (
            triton.cdiv(Q, META["BLOCK_Q"]),
            G * num_d_blocks,
            B,
        )

    _flash_deform_bwd_kernel[grid](
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc_attn,
        grad_output,
        grad_input,
        grad_offset,
        N,
        Q,
        G,
        D,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        sampling_loc_attn.stride(0),
        sampling_loc_attn.stride(1),
        sampling_loc_attn.stride(2),
        sampling_loc_attn.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        grad_offset.stride(0),
        grad_offset.stride(1),
        grad_offset.stride(2),
        grad_offset.stride(3),
        L=L,
        K=K,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=num_d_blocks,
    )

    if value.dtype in (torch.float16, torch.bfloat16):
        return grad_input.to(value.dtype), grad_offset.to(sampling_loc_attn.dtype)
    return grad_input, grad_offset
