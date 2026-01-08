"""Test dynamic vs constexpr indexing in Triton."""

import torch
import triton
import triton.language as tl


# Test 1: constexpr index (should work)
@triton.jit
def kernel_constexpr(
    in_ptr,
    out_ptr,
    stride_row,
    stride_col,
    BLOCK: tl.constexpr,
    COL: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    # 2D tensor loaded as block
    data = tl.load(
        in_ptr + rows[:, None] * stride_row + tl.arange(0, 8)[None, :] * stride_col,
    )
    # Direct index with constexpr
    col_data = data[:, COL]
    tl.store(out_ptr + rows, col_data)


# Test 2: dynamic index in loop (like the real kernel)
@triton.jit
def kernel_dynamic(
    in_ptr,
    out_ptr,
    stride_row,
    stride_col,
    BLOCK: tl.constexpr,
    NUM_COLS: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    # Load 2D block
    data = tl.load(
        in_ptr
        + rows[:, None] * stride_row
        + tl.arange(0, NUM_COLS)[None, :] * stride_col,
    )

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for point_idx in range(NUM_COLS):
        # This is what fails - dynamic index
        col_data = data[:, point_idx]
        acc += col_data

    tl.store(out_ptr + rows, acc)


# Test 3: tl.static_range instead of range
@triton.jit
def kernel_static_range(
    in_ptr,
    out_ptr,
    stride_row,
    stride_col,
    BLOCK: tl.constexpr,
    NUM_COLS: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    data = tl.load(
        in_ptr
        + rows[:, None] * stride_row
        + tl.arange(0, NUM_COLS)[None, :] * stride_col,
    )

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for point_idx in tl.static_range(NUM_COLS):
        col_data = data[:, point_idx]
        acc += col_data

    tl.store(out_ptr + rows, acc)


# Test 4: Original approach with mask and sum
@triton.jit
def kernel_mask_sum(
    in_ptr,
    out_ptr,
    stride_row,
    stride_col,
    BLOCK: tl.constexpr,
    NUM_COLS: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    data = tl.load(
        in_ptr
        + rows[:, None] * stride_row
        + tl.arange(0, NUM_COLS)[None, :] * stride_col,
    )

    mask_idx = tl.arange(0, NUM_COLS)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for point_idx in range(NUM_COLS):
        mask_now = (mask_idx == point_idx)[None, :]
        col_data = tl.sum(data * mask_now, axis=1)
        acc += col_data

    tl.store(out_ptr + rows, acc)


if __name__ == "__main__":
    print("Test 1: constexpr index")
    try:
        inp = torch.randn(64, 8, device="cuda")
        out = torch.zeros(64, device="cuda")
        kernel_constexpr[(4,)](inp, out, inp.stride(0), inp.stride(1), BLOCK=16, COL=3)
        print(f"  ✓ constexpr works: {torch.allclose(out, inp[:, 3])}")
    except Exception as e:
        print(f"  ✗ constexpr FAILED: {e}")

    print("\nTest 2: dynamic index in loop (range)")
    try:
        inp = torch.randn(64, 8, device="cuda")
        out = torch.zeros(64, device="cuda")
        kernel_dynamic[(4,)](
            inp,
            out,
            inp.stride(0),
            inp.stride(1),
            BLOCK=16,
            NUM_COLS=8,
        )
        expected = inp.sum(dim=1)
        print(f"  ✓ dynamic index works: {torch.allclose(out, expected)}")
    except Exception as e:
        print(f"  ✗ dynamic index FAILED: {e}")

    print("\nTest 3: tl.static_range (constexpr loop)")
    try:
        inp = torch.randn(64, 8, device="cuda")
        out = torch.zeros(64, device="cuda")
        kernel_static_range[(4,)](
            inp,
            out,
            inp.stride(0),
            inp.stride(1),
            BLOCK=16,
            NUM_COLS=8,
        )
        expected = inp.sum(dim=1)
        print(f"  ✓ tl.static_range works: {torch.allclose(out, expected)}")
    except Exception as e:
        print(f"  ✗ tl.static_range FAILED: {e}")

    print("\nTest 4: Original mask+sum approach")
    try:
        inp = torch.randn(64, 8, device="cuda")
        out = torch.zeros(64, device="cuda")
        kernel_mask_sum[(4,)](
            inp,
            out,
            inp.stride(0),
            inp.stride(1),
            BLOCK=16,
            NUM_COLS=8,
        )
        expected = inp.sum(dim=1)
        print(f"  ✓ mask+sum works: {torch.allclose(out, expected)}")
    except Exception as e:
        print(f"  ✗ mask+sum FAILED: {e}")
