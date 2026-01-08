"""Minimal tests to verify if proposed optimization features are available.
This is NOT integration testing - just checking API availability.
"""

from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
print("-" * 60)


def test_swizzle2d() -> bool | None:
    """Test 1: tl.swizzle2d availability."""
    print("\n[TEST 1] tl.swizzle2d")
    try:
        # Check if swizzle2d exists in triton.language
        if hasattr(tl, "swizzle2d"):
            print("  ✓ tl.swizzle2d is available")

            # Try to compile a minimal kernel using it
            # swizzle2d(i, j, size_i, size_j, size_g) - transforms row-major to column-major per group
            @triton.jit
            def _test_swizzle_kernel(
                out_ptr,
                SIZE_I: tl.constexpr,
                SIZE_J: tl.constexpr,
                SIZE_G: tl.constexpr,
            ) -> None:
                pid = tl.program_id(0)
                # Convert linear pid to 2D indices
                pid_i = pid // SIZE_J
                pid_j = pid % SIZE_J
                # Apply swizzle transformation
                new_i, new_j = tl.swizzle2d(pid_i, pid_j, SIZE_I, SIZE_J, SIZE_G)
                tl.store(out_ptr + pid, new_i * 10 + new_j)

            out = torch.zeros(16, device="cuda", dtype=torch.int32)
            _test_swizzle_kernel[(16,)](out, SIZE_I=4, SIZE_J=4, SIZE_G=2)
            print("  ✓ Kernel compiled and ran successfully")
            print(f"    Output sample: {out[:8].tolist()}")
            return True
        print("  ✗ tl.swizzle2d not found in triton.language")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_block_ptr() -> bool | None:
    """Test 2: tl.make_block_ptr / tl.advance (Block Pointer API)."""
    print("\n[TEST 2] Block Pointer API (tl.make_block_ptr)")
    try:
        has_make_block_ptr = hasattr(tl, "make_block_ptr")
        has_advance = hasattr(tl, "advance")
        print(f"  tl.make_block_ptr exists: {has_make_block_ptr}")
        print(f"  tl.advance exists: {has_advance}")

        if has_make_block_ptr:

            @triton.jit
            def _test_block_ptr_kernel(
                in_ptr,
                out_ptr,
                stride_m,
                stride_n,
                BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr,
            ) -> None:
                pid_m = tl.program_id(0)
                pid_n = tl.program_id(1)

                block_ptr = tl.make_block_ptr(
                    base=in_ptr,
                    shape=(64, 64),
                    strides=(stride_m, stride_n),
                    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                data = tl.load(block_ptr, boundary_check=(0, 1))

                out_block_ptr = tl.make_block_ptr(
                    base=out_ptr,
                    shape=(64, 64),
                    strides=(stride_m, stride_n),
                    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                tl.store(out_block_ptr, data, boundary_check=(0, 1))

            inp = torch.randn(64, 64, device="cuda", dtype=torch.float32)
            out = torch.zeros_like(inp)
            grid = (4, 4)
            _test_block_ptr_kernel[grid](
                inp,
                out,
                inp.stride(0),
                inp.stride(1),
                BLOCK_M=16,
                BLOCK_N=16,
            )

            if torch.allclose(inp, out):
                print("  ✓ Block Pointer kernel works correctly")
                return True
            print("  ✗ Block Pointer kernel output mismatch")
            return False
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_eviction_policy() -> bool | None:
    """Test 3: eviction_policy parameter in tl.load/tl.store."""
    print("\n[TEST 3] eviction_policy parameter")
    try:

        @triton.jit
        def _test_eviction_kernel(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK: tl.constexpr,
        ) -> None:
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n_elements

            # Test eviction policies
            data = tl.load(in_ptr + offsets, mask=mask, eviction_policy="evict_last")
            tl.store(out_ptr + offsets, data, mask=mask, eviction_policy="evict_first")

        inp = torch.randn(1024, device="cuda", dtype=torch.float32)
        out = torch.zeros_like(inp)
        _test_eviction_kernel[(32,)](inp, out, 1024, BLOCK=32)

        if torch.allclose(inp, out):
            print("  ✓ eviction_policy parameter works")
            print("    Supported: evict_first, evict_last")
            return True
        print("  ✗ Output mismatch")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_dot_scaled() -> bool | None:
    """Test 4: tl.dot_scaled (FP8 support)."""
    print("\n[TEST 4] tl.dot_scaled (FP8)")
    try:
        has_dot_scaled = hasattr(tl, "dot_scaled")
        print(f"  tl.dot_scaled exists: {has_dot_scaled}")

        if has_dot_scaled:
            # Check FP8 dtype support
            has_float8_e4m3fn = hasattr(tl, "float8e4nv") or hasattr(
                tl,
                "float8_e4m3fn",
            )
            has_float8_e5m2 = hasattr(tl, "float8e5") or hasattr(tl, "float8_e5m2")
            print(f"  FP8 E4M3 dtype: {has_float8_e4m3fn}")
            print(f"  FP8 E5M2 dtype: {has_float8_e5m2}")

            # List available fp8 types
            fp8_types = [attr for attr in dir(tl) if "float8" in attr.lower()]
            print(f"  Available FP8 types: {fp8_types}")

            print("  ✓ tl.dot_scaled is available (FP8 GEMM)")
            return True
        print("  ✗ tl.dot_scaled not found")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_libdevice() -> bool | None:
    """Test 5: tl.extra.cuda.libdevice (fast math intrinsics)."""
    print("\n[TEST 5] tl.extra.cuda.libdevice")
    try:
        from triton.language.extra.cuda import libdevice

        # Check available functions
        funcs = [f for f in dir(libdevice) if not f.startswith("_")]
        print("  ✓ libdevice is available")
        print(f"    Available functions ({len(funcs)}): {funcs[:10]}...")

        # Test a simple kernel using libdevice
        # Note: must import outside kernel and call directly
        @triton.jit
        def _test_libdevice_kernel(
            in_ptr,
            out_ptr,
            n_elements,
            BLOCK: tl.constexpr,
        ) -> None:
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            # Use libdevice fast_expf (imported at module level)
            y = tl.extra.cuda.libdevice.fast_expf(x)
            tl.store(out_ptr + offsets, y, mask=mask)

        inp = torch.randn(1024, device="cuda", dtype=torch.float32)
        out = torch.zeros_like(inp)
        _test_libdevice_kernel[(32,)](inp, out, 1024, BLOCK=32)

        expected = torch.exp(inp)
        if torch.allclose(out, expected, rtol=1e-3, atol=1e-3):
            print("  ✓ libdevice.fast_expf works correctly")
            return True
        print(
            "  ~ libdevice.fast_expf has precision differences (expected for fast math)",
        )
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_tma_experimental() -> bool | None:
    """Test 6: TMA Experimental API."""
    print("\n[TEST 6] TMA Experimental API")
    try:
        # Check for experimental TMA APIs
        has_tensor_desc = hasattr(tl, "_experimental_make_tensor_descriptor")
        has_desc_load = hasattr(tl, "_experimental_descriptor_load")

        print(f"  tl._experimental_make_tensor_descriptor: {has_tensor_desc}")
        print(f"  tl._experimental_descriptor_load: {has_desc_load}")

        # Also check in triton.language.extra if exists
        experimental_attrs = [
            attr
            for attr in dir(tl)
            if "experimental" in attr.lower() or "tma" in attr.lower()
        ]
        print(f"  Experimental/TMA attrs: {experimental_attrs}")

        if has_tensor_desc or has_desc_load:
            print("  ✓ TMA experimental API is available")
            return True
        print(
            "  ✗ TMA experimental API not found (may require Triton 3.0+ and H100)",
        )
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_torch_compile() -> bool | None:
    """Test 7: torch.compile compatibility with Triton custom ops."""
    print("\n[TEST 7] torch.compile compatibility")
    try:
        # Simple test - check if torch.compile works at all
        @torch.compile
        def simple_fn(x):
            return x * 2 + 1

        x = torch.randn(100, device="cuda")
        y = simple_fn(x)
        expected = x * 2 + 1

        if torch.allclose(y, expected):
            print("  ✓ torch.compile works")

        # Check compile modes
        print("  Available compile modes: default, reduce-overhead, max-autotune")

        # Test with reduce-overhead mode
        @torch.compile(mode="reduce-overhead")
        def compiled_fn(x):
            return x.sin() + x.cos()

        y2 = compiled_fn(x)
        if y2 is not None:
            print("  ✓ mode='reduce-overhead' works")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_direct_indexing() -> bool | None:
    """Test 8: Direct 2D tensor indexing (data[:, point_idx]).

    This tests if we can replace mask+sum with direct slicing.
    IMPORTANT: This tests actual 2D tensor slicing, not pointer arithmetic.
    """
    print("\n[TEST 8] Direct 2D tensor indexing in Triton")
    try:

        @triton.jit
        def _test_2d_slice_kernel(
            in_ptr,
            out_ptr,
            stride_row,
            stride_col,
            BLOCK: tl.constexpr,
            NUM_COLS: tl.constexpr,
        ) -> None:
            pid = tl.program_id(0)
            rows = pid * BLOCK + tl.arange(0, BLOCK)
            # Load as 2D block
            data = tl.load(
                in_ptr
                + rows[:, None] * stride_row
                + tl.arange(0, NUM_COLS)[None, :] * stride_col,
            )
            # Try direct 2D slice (this is what the proposal suggests)
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for point_idx in tl.static_range(NUM_COLS):
                col_data = data[:, point_idx]  # Direct slice
                acc += col_data

            tl.store(out_ptr + rows, acc)

        inp = torch.randn(64, 8, device="cuda", dtype=torch.float32)
        out = torch.zeros(64, device="cuda", dtype=torch.float32)

        _test_2d_slice_kernel[(4,)](
            inp,
            out,
            inp.stride(0),
            inp.stride(1),
            BLOCK=16,
            NUM_COLS=8,
        )

        expected = inp.sum(dim=1)
        if torch.allclose(out, expected):
            print("  ✓ Direct 2D slicing works")
            return True
        print("  ✗ Output mismatch")
        return False
    except Exception as e:
        print(f"  ✗ 2D tensor slicing NOT supported: {type(e).__name__}")
        print("    Triton does not support data[:, idx] syntax")
        print("    Must use mask+sum approach instead")
        return False


if __name__ == "__main__":
    results = {}

    results["swizzle2d"] = test_swizzle2d()
    results["block_ptr"] = test_block_ptr()
    results["eviction_policy"] = test_eviction_policy()
    results["dot_scaled"] = test_dot_scaled()
    results["libdevice"] = test_libdevice()
    results["tma_experimental"] = test_tma_experimental()
    results["torch_compile"] = test_torch_compile()
    results["direct_indexing"] = test_direct_indexing()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s}: {status}")

    total_pass = sum(results.values())
    total = len(results)
    print(f"\nTotal: {total_pass}/{total} features available")
