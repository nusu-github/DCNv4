#!/usr/bin/env python3
"""List CUDA kernel names for profiling with ncu.

This script launches each kernel once so that ncu can capture their names.

Usage:
    # List all kernel names
    ncu --launch-count 10 --section "" python scripts/list_kernels.py

    # With demangled names
    ncu --launch-count 10 --section "" --print-kernel-base demangled python scripts/list_kernels.py

    # Filter by kernel name pattern
    ncu -k "dcnv4" --section "" python scripts/list_kernels.py
"""

import argparse

import torch


def run_dcnv4_kernels() -> None:
    """Run DCNv4 forward and backward kernels."""
    from dcnv4.functions.dcnv4_func import dcnv4_forward

    print("Running DCNv4 kernels...")

    # Shape compatible with TABLE constraints
    # offset_mask_channels = G * k * 3 must be divisible by 8
    # Using G=8, k=9 -> 8 * 9 * 3 = 216 (divisible by 8)
    B, H, W, G, C = 1, 64, 64, 8, 16
    kernel_h, kernel_w = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation_h, dilation_w = 1, 1
    k = kernel_h * kernel_w
    offset_mask_channels = G * k * 3

    for dtype in [torch.float32, torch.float16]:
        dtype_name = "fp32" if dtype == torch.float32 else "fp16"
        print(f"  DCNv4 forward ({dtype_name})")

        input_tensor = torch.randn(B, H, W, G * C, device="cuda", dtype=dtype)
        offset_mask = torch.randn(
            B,
            H,
            W,
            offset_mask_channels,
            device="cuda",
            dtype=dtype,
        )

        output = dcnv4_forward(
            input_tensor,
            offset_mask,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            G,
            C,
            1.0,
            64,
            0,
            False,
        )
        torch.cuda.synchronize()

        # Backward pass
        print(f"  DCNv4 backward ({dtype_name})")
        input_tensor = torch.randn(
            B,
            H,
            W,
            G * C,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        offset_mask = torch.randn(
            B,
            H,
            W,
            offset_mask_channels,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )

        output = dcnv4_forward(
            input_tensor,
            offset_mask,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            G,
            C,
            1.0,
            64,
            0,
            False,
        )
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        torch.cuda.synchronize()


def run_flash_deform_attn_kernels() -> None:
    """Run FlashDeformAttn forward and backward kernels."""
    from dcnv4.functions.flash_deform_attn_func import flash_deform_attn

    print("Running FlashDeformAttn kernels...")

    # Small shape for quick kernel launch
    B, Q, n_heads, head_dim = 1, 64, 4, 32
    n_levels, n_points = 2, 4
    spatial_shapes_list = [(16, 16), (8, 8)]
    total_len = sum(h * w for h, w in spatial_shapes_list)

    for dtype in [torch.float32, torch.float16]:
        dtype_name = "fp32" if dtype == torch.float32 else "fp16"
        print(f"  FlashDeformAttn forward ({dtype_name})")

        value = torch.randn(B, total_len, n_heads, head_dim, device="cuda", dtype=dtype)
        spatial_shapes = torch.tensor(
            spatial_shapes_list,
            dtype=torch.int64,
            device="cuda",
        )
        level_start_index = torch.tensor([0, 256], dtype=torch.int64, device="cuda")

        coords_dim = n_levels * n_points * 2
        attn_dim = n_levels * n_points
        total_dim = coords_dim + attn_dim
        sampling_loc_attn = torch.randn(
            B,
            Q,
            n_heads,
            total_dim,
            device="cuda",
            dtype=dtype,
        )
        sampling_loc_attn[..., :coords_dim] = torch.sigmoid(
            sampling_loc_attn[..., :coords_dim],
        )

        output = flash_deform_attn(
            value,
            spatial_shapes,
            level_start_index,
            sampling_loc_attn,
            64,
            n_points,
        )
        torch.cuda.synchronize()

        # Backward pass
        print(f"  FlashDeformAttn backward ({dtype_name})")
        value = torch.randn(
            B,
            total_len,
            n_heads,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        sampling_loc_attn = torch.randn(
            B,
            Q,
            n_heads,
            total_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )

        output = flash_deform_attn(
            value,
            spatial_shapes,
            level_start_index,
            sampling_loc_attn,
            64,
            n_points,
        )
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        torch.cuda.synchronize()


def run_triton_kernels() -> None:
    """Run Triton-based DCNv4 kernels if available."""
    try:
        from dcnv4.functions.dcnv4_triton import dcnv4_forward_triton
    except ImportError:
        print("Triton kernels not available, skipping...")
        return

    print("Running Triton DCNv4 kernels...")

    B, H, W, G, C = 1, 16, 16, 4, 16
    kernel_h, kernel_w = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation_h, dilation_w = 1, 1
    from dcnv4.functions.utils import compute_offset_mask_channels

    offset_mask_channels = compute_offset_mask_channels(G, kernel_h, 0)

    for dtype in [torch.float32, torch.float16]:
        dtype_name = "fp32" if dtype == torch.float32 else "fp16"
        print(f"  DCNv4 Triton forward ({dtype_name})")

        input_tensor = torch.randn(B, H, W, G * C, device="cuda", dtype=dtype)
        offset_mask = torch.randn(
            B,
            H,
            W,
            offset_mask_channels,
            device="cuda",
            dtype=dtype,
        )

        try:
            dcnv4_forward_triton(
                input_tensor,
                offset_mask,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                G,
                C,
                1.0,
                0,
                False,
            )
            torch.cuda.synchronize()
        except Exception as e:
            print(f"    Triton forward failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch CUDA kernels for ncu profiling",
    )
    parser.add_argument(
        "--dcnv4-only",
        action="store_true",
        help="Only run DCNv4 kernels",
    )
    parser.add_argument(
        "--flash-deform-only",
        action="store_true",
        help="Only run FlashDeformAttn kernels",
    )
    parser.add_argument(
        "--triton-only",
        action="store_true",
        help="Only run Triton kernels",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    run_all = not (args.dcnv4_only or args.flash_deform_only or args.triton_only)

    if run_all or args.dcnv4_only:
        run_dcnv4_kernels()
        print()

    if run_all or args.flash_deform_only:
        run_flash_deform_attn_kernels()
        print()

    if run_all or args.triton_only:
        run_triton_kernels()
        print()

    print("Done. Use with ncu to see kernel names:")
    print('  ncu --launch-count 20 --section "" python scripts/list_kernels.py')


if __name__ == "__main__":
    main()
