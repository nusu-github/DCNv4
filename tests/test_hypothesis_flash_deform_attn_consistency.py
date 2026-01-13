"""CUDA vs PyTorch reference consistency tests for FlashDeformAttn.

Tolerance values are derived from empirical measurements
(see scripts/measure_flash_deform_attn_tolerance.py).

Key findings from measurement (NVIDIA L40S, 200 seeds, 15 shapes):

    Forward (float32):
        max_abs = 1.19e-6, p99.9 = 9.54e-7
        Recommended: rtol = 3.4e-5, atol = 1.9e-6 (with 2x safety margin)

    Gradient value (float32):
        max_abs = 4.53e-6, p99.9 = 4.05e-6
        Non-determinism (atomicAdd): max = 3.1e-6
        Recommended: rtol = 2.24e-4, atol = 8.1e-6

    Gradient sampling_loc_attn (float32):
        max_abs = 6.1e-4, p99.9 = 4.88e-4
        Non-determinism: max = 0 (deterministic)
        Recommended: rtol = 5.57e-3, atol = 9.77e-4

    Half precision:
        float16:  max_abs = 9.77e-4
        bfloat16: max_abs = 7.81e-3
        Recommended: rtol/atol = 2e-3 (fp16), 1.56e-2 (bf16)
"""

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dcnv4.functions.flash_deform_attn_func import flash_deform_attn
from dcnv4.modules.flash_deform_attn_torch import flash_deform_attn_torch

# =============================================================================
# Empirically-derived tolerance constants
# Source: scripts/measure_flash_deform_attn_tolerance.py
# (200 seeds x 15 shapes on NVIDIA L40S)
# =============================================================================

# Forward pass (float32): p99.9 * 2x safety margin
FORWARD_RTOL = 3.40e-5
FORWARD_ATOL = 1.91e-6

# Gradient value (float32): p99.9 * 2x, with non-determinism floor
GRAD_VALUE_RTOL = 2.24e-4
GRAD_VALUE_ATOL = 8.11e-6

# Gradient sampling_loc_attn (float32): p99.9 * 2x
GRAD_SAMPLING_RTOL = 5.57e-3
GRAD_SAMPLING_ATOL = 9.77e-4

# Half precision tolerances
HALF_FLOAT16_RTOL = 2.00e-3
HALF_FLOAT16_ATOL = 1.95e-3
HALF_BFLOAT16_RTOL = 1.56e-2
HALF_BFLOAT16_ATOL = 1.56e-2

# Self-consistency tests (same implementation, different params)
SELF_CONSISTENCY_ATOL = 1e-5

# Out-of-bounds tests (output should be exactly zero)
OOB_ATOL = 1e-6


def _valid_shapes() -> list[tuple[int, int, int, int, int, int, int]]:
    """Generate valid test shapes for FlashDeformAttn.

    Returns shapes as (B, L, K, G, D, Q, spatial_config_idx) tuples.
    L = number of levels, K = points per level.

    Note: CUDA kernel only supports K=4 or K=8.
    """
    shapes = []
    for B in [1, 2]:
        for L in [1, 2, 4]:
            for K in [4, 8]:  # CUDA kernel constraint: K must be 4 or 8
                for G in [2, 4, 8]:
                    for D in [8, 16, 32]:
                        for Q in [10, 50]:
                            for spatial_idx in [0, 1]:
                                shapes.append((B, L, K, G, D, Q, spatial_idx))
    # Limit to a representative subset
    return shapes[::8][:15]


VALID_SHAPES = _valid_shapes()

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for FlashDeformAttn CUDA/PyTorch consistency tests",
)


def _make_spatial_shapes(
    L: int,
    spatial_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Create spatial shapes and level start indices.

    Returns:
        spatial_shapes: (L, 2) tensor of (H, W) per level.
        level_start_index: (L,) tensor of start indices.
        N: Total spatial tokens.

    """
    if spatial_idx == 0:
        configs = [(8, 8), (4, 4), (2, 2), (1, 1)][:L]
    else:
        configs = [(6, 8), (3, 4), (2, 2), (1, 1)][:L]

    spatial_shapes = torch.tensor(configs, dtype=torch.int64, device=device)
    level_start_index = torch.zeros(L, dtype=torch.int64, device=device)
    for i in range(1, L):
        level_start_index[i] = level_start_index[i - 1] + (
            spatial_shapes[i - 1, 0] * spatial_shapes[i - 1, 1]
        )
    N = sum(h * w for h, w in configs)
    return spatial_shapes, level_start_index, N


def _make_inputs(
    B: int,
    L: int,
    K: int,
    G: int,
    D: int,
    Q: int,
    spatial_idx: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> dict:
    """Create inputs for FlashDeformAttn testing."""
    spatial_shapes, level_start_index, N = _make_spatial_shapes(
        L,
        spatial_idx,
        torch.device(device),
    )

    value = torch.randn(B, N, G, D, dtype=dtype, device=device)

    # sampling_loc_attn: (B, Q, G, L * K * 3)
    # First L*K*2 elements: normalized coordinates [0, 1]
    # Last L*K elements: attention logits
    coords_dim = L * K * 2
    total_dim = L * K * 3

    sampling_loc_attn = torch.randn(B, Q, G, total_dim, dtype=dtype, device=device)
    # Normalize coordinates to [0, 1]
    sampling_loc_attn[..., :coords_dim] = torch.sigmoid(
        sampling_loc_attn[..., :coords_dim],
    )

    return {
        "value": value,
        "spatial_shapes": spatial_shapes,
        "level_start_index": level_start_index,
        "sampling_loc_attn": sampling_loc_attn,
        "im2col_step": 64,
        "K": K,
    }


def _run_cuda(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_loc_attn: torch.Tensor,
    im2col_step: int,
    K: int,
) -> torch.Tensor:
    """Run CUDA implementation."""
    return flash_deform_attn(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc_attn,
        im2col_step,
        K,
    )


def _run_ref(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_loc_attn: torch.Tensor,
    im2col_step: int,
    K: int,
) -> torch.Tensor:
    """Run PyTorch reference implementation."""
    return flash_deform_attn_torch(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc_attn,
        im2col_step,
        K,
    )


@settings(max_examples=6, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_flash_deform_attn_cuda_matches_pytorch_reference(shape, seed) -> None:
    """Test that CUDA forward and backward match PyTorch reference."""
    B, L, K, G, D, Q, spatial_idx = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = _make_inputs(B, L, K, G, D, Q, spatial_idx)

    # Prepare tensors for gradient computation
    value_cuda = inputs["value"].clone().requires_grad_(True)
    sampling_cuda = inputs["sampling_loc_attn"].clone().requires_grad_(True)

    value_ref = inputs["value"].clone().requires_grad_(True)
    sampling_ref = inputs["sampling_loc_attn"].clone().requires_grad_(True)

    # Forward pass
    out_cuda = _run_cuda(
        value_cuda,
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        sampling_cuda,
        inputs["im2col_step"],
        inputs["K"],
    )
    out_ref = _run_ref(
        value_ref,
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        sampling_ref,
        inputs["im2col_step"],
        inputs["K"],
    )

    assert out_cuda.shape == out_ref.shape
    torch.testing.assert_close(out_cuda, out_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)

    # Backward pass
    grad = torch.randn_like(out_cuda)
    out_cuda.backward(grad)
    out_ref.backward(grad)

    assert value_cuda.grad is not None
    assert sampling_cuda.grad is not None
    assert value_ref.grad is not None
    assert sampling_ref.grad is not None

    torch.testing.assert_close(
        value_cuda.grad,
        value_ref.grad,
        rtol=GRAD_VALUE_RTOL,
        atol=GRAD_VALUE_ATOL,
    )
    torch.testing.assert_close(
        sampling_cuda.grad,
        sampling_ref.grad,
        rtol=GRAD_SAMPLING_RTOL,
        atol=GRAD_SAMPLING_ATOL,
    )


@settings(max_examples=4, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    im2col_step=st.sampled_from([1, 2, 8, 32, 128]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_im2col_step_invariance(shape, im2col_step, seed) -> None:
    """Test that different im2col_step values produce identical results."""
    B, L, K, G, D, Q, spatial_idx = shape
    assume(im2col_step != 64)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = _make_inputs(B, L, K, G, D, Q, spatial_idx)

    out_base = _run_cuda(
        inputs["value"],
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        64,
        inputs["K"],
    )
    out_alt = _run_cuda(
        inputs["value"],
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        im2col_step,
        inputs["K"],
    )

    # Same CUDA kernel with different batch grouping - should be identical
    torch.testing.assert_close(out_base, out_alt, rtol=0, atol=SELF_CONSISTENCY_ATOL)


@settings(max_examples=4, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
)
def test_oob_coordinates_yield_zero_output(shape) -> None:
    """Test that out-of-bounds coordinates produce zero output."""
    B, L, K, G, D, Q, spatial_idx = shape

    inputs = _make_inputs(B, L, K, G, D, Q, spatial_idx)

    # Set all coordinates far out of bounds (-10, which is well below 0)
    coords_dim = L * K * 2
    inputs["sampling_loc_attn"][..., :coords_dim] = -10.0

    value_cuda = inputs["value"].clone().requires_grad_(True)
    sampling_cuda = inputs["sampling_loc_attn"].clone().requires_grad_(True)

    out_cuda = _run_cuda(
        value_cuda,
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        sampling_cuda,
        inputs["im2col_step"],
        inputs["K"],
    )

    # OOB sampling should produce zero output
    torch.testing.assert_close(
        out_cuda,
        torch.zeros_like(out_cuda),
        rtol=0,
        atol=OOB_ATOL,
    )

    out_cuda.sum().backward()
    assert value_cuda.grad is not None
    torch.testing.assert_close(
        value_cuda.grad,
        torch.zeros_like(value_cuda.grad),
        rtol=0,
        atol=OOB_ATOL,
    )

    # Verify reference also produces zero
    out_ref = _run_ref(
        inputs["value"],
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        inputs["im2col_step"],
        inputs["K"],
    )
    torch.testing.assert_close(
        out_ref,
        torch.zeros_like(out_ref),
        rtol=0,
        atol=OOB_ATOL,
    )


@settings(max_examples=4, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    shift=st.floats(
        min_value=-2.0,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_softmax_shift_invariance(shape, shift, seed) -> None:
    """Test that shifting attention logits doesn't change output (softmax property)."""
    B, L, K, G, D, Q, spatial_idx = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = _make_inputs(B, L, K, G, D, Q, spatial_idx)

    coords_dim = L * K * 2
    sampling_shifted = inputs["sampling_loc_attn"].clone()
    sampling_shifted[..., coords_dim:] += shift

    out_base_cuda = _run_cuda(
        inputs["value"],
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        inputs["im2col_step"],
        inputs["K"],
    )
    out_shifted_cuda = _run_cuda(
        inputs["value"],
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        sampling_shifted,
        inputs["im2col_step"],
        inputs["K"],
    )

    # Softmax is shift-invariant: softmax(x + c) = softmax(x)
    torch.testing.assert_close(
        out_base_cuda,
        out_shifted_cuda,
        rtol=0,
        atol=SELF_CONSISTENCY_ATOL,
    )

    # Verify same property for reference
    out_base_ref = _run_ref(
        inputs["value"],
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        inputs["im2col_step"],
        inputs["K"],
    )
    out_shifted_ref = _run_ref(
        inputs["value"],
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        sampling_shifted,
        inputs["im2col_step"],
        inputs["K"],
    )
    torch.testing.assert_close(
        out_base_ref,
        out_shifted_ref,
        rtol=0,
        atol=SELF_CONSISTENCY_ATOL,
    )


@settings(max_examples=3, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_linearity_in_value(shape, seed) -> None:
    """Test that f(v1 + v2) = f(v1) + f(v2) (linearity in value)."""
    B, L, K, G, D, Q, spatial_idx = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs1 = _make_inputs(B, L, K, G, D, Q, spatial_idx)
    inputs2 = _make_inputs(B, L, K, G, D, Q, spatial_idx)

    # Use same sampling locations
    sampling = inputs1["sampling_loc_attn"]

    out_cuda_sum = _run_cuda(
        inputs1["value"] + inputs2["value"],
        inputs1["spatial_shapes"],
        inputs1["level_start_index"],
        sampling,
        inputs1["im2col_step"],
        inputs1["K"],
    )
    out_cuda_parts = _run_cuda(
        inputs1["value"],
        inputs1["spatial_shapes"],
        inputs1["level_start_index"],
        sampling,
        inputs1["im2col_step"],
        inputs1["K"],
    ) + _run_cuda(
        inputs2["value"],
        inputs2["spatial_shapes"],
        inputs2["level_start_index"],
        sampling,
        inputs2["im2col_step"],
        inputs2["K"],
    )

    # Bilinear interpolation is linear in value: f(a+b) = f(a) + f(b)
    torch.testing.assert_close(
        out_cuda_sum,
        out_cuda_parts,
        rtol=FORWARD_RTOL,
        atol=FORWARD_ATOL,
    )

    # Verify same for reference
    out_ref_sum = _run_ref(
        inputs1["value"] + inputs2["value"],
        inputs1["spatial_shapes"],
        inputs1["level_start_index"],
        sampling,
        inputs1["im2col_step"],
        inputs1["K"],
    )
    out_ref_parts = _run_ref(
        inputs1["value"],
        inputs1["spatial_shapes"],
        inputs1["level_start_index"],
        sampling,
        inputs1["im2col_step"],
        inputs1["K"],
    ) + _run_ref(
        inputs2["value"],
        inputs2["spatial_shapes"],
        inputs2["level_start_index"],
        sampling,
        inputs2["im2col_step"],
        inputs2["K"],
    )
    torch.testing.assert_close(
        out_ref_sum,
        out_ref_parts,
        rtol=FORWARD_RTOL,
        atol=FORWARD_ATOL,
    )


@settings(max_examples=3, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_scaling_in_value(shape, scale, seed) -> None:
    """Test that f(a * v) = a * f(v) (homogeneity in value)."""
    B, L, K, G, D, Q, spatial_idx = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = _make_inputs(B, L, K, G, D, Q, spatial_idx)

    out_scaled = _run_cuda(
        inputs["value"] * scale,
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        inputs["im2col_step"],
        inputs["K"],
    )
    out_base = _run_cuda(
        inputs["value"],
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        inputs["im2col_step"],
        inputs["K"],
    )

    torch.testing.assert_close(
        out_scaled,
        out_base * scale,
        rtol=FORWARD_RTOL,
        atol=FORWARD_ATOL * scale,
    )


@settings(max_examples=3, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_half_precision_parity(shape, seed) -> None:
    """Test CUDA vs reference parity for half precision dtypes."""
    B, L, K, G, D, Q, spatial_idx = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for dtype in (torch.float16, torch.bfloat16):
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue

        inputs = _make_inputs(B, L, K, G, D, Q, spatial_idx, dtype=dtype)

        out_cuda = _run_cuda(
            inputs["value"],
            inputs["spatial_shapes"],
            inputs["level_start_index"],
            inputs["sampling_loc_attn"],
            inputs["im2col_step"],
            inputs["K"],
        )
        out_ref = _run_ref(
            inputs["value"],
            inputs["spatial_shapes"],
            inputs["level_start_index"],
            inputs["sampling_loc_attn"],
            inputs["im2col_step"],
            inputs["K"],
        )

        # Note: PyTorch reference upcasts to float32 internally, so output is float32
        # CUDA keeps the original dtype
        assert out_cuda.dtype == dtype

        # Use dtype-specific tolerances
        if dtype == torch.bfloat16:
            rtol, atol = HALF_BFLOAT16_RTOL, HALF_BFLOAT16_ATOL
        else:
            rtol, atol = HALF_FLOAT16_RTOL, HALF_FLOAT16_ATOL

        torch.testing.assert_close(
            out_cuda.float(),
            out_ref.float(),
            rtol=rtol,
            atol=atol,
        )


def _batch_shapes() -> list[tuple[int, int, int, int, int, int, int]]:
    """Return shapes with B > 1 for batch independence test."""
    return [s for s in VALID_SHAPES if s[0] > 1]


def _fallback_batch_shapes() -> list[tuple[int, int, int, int, int, int, int]]:
    """Create a valid batch shape when VALID_SHAPES contains only B=1.

    We reuse an existing VALID_SHAPES entry (which already satisfies the CUDA
    kernel constraints) and only bump B to 2.
    """
    if len(VALID_SHAPES) == 0:
        # Extremely defensive fallback: must satisfy the CUDA constraint
        # "Q divisible by block_multiplier" (common case: 20).
        return [(2, 2, 4, 4, 16, 20, 0)]

    _b, l, k, g, d, q, spatial_idx = VALID_SHAPES[0]
    return [(2, l, k, g, d, q, spatial_idx)]


BATCH_SHAPES = _batch_shapes() or _fallback_batch_shapes()


@settings(max_examples=3, deadline=None)
@given(
    shape=st.sampled_from(BATCH_SHAPES),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_batch_independence(shape, seed) -> None:
    """Test that batch elements are processed independently."""
    B, L, K, G, D, Q, spatial_idx = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = _make_inputs(B, L, K, G, D, Q, spatial_idx)

    full_output = _run_cuda(
        inputs["value"],
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        inputs["im2col_step"],
        inputs["K"],
    )

    # Process each batch element separately
    for b in range(B):
        single_output = _run_cuda(
            inputs["value"][b : b + 1],
            inputs["spatial_shapes"],
            inputs["level_start_index"],
            inputs["sampling_loc_attn"][b : b + 1],
            inputs["im2col_step"],
            inputs["K"],
        )

        torch.testing.assert_close(
            full_output[b : b + 1],
            single_output,
            rtol=0,
            atol=0,
        )


@settings(max_examples=3, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_uniform_value_with_center_coords(shape, seed) -> None:
    """Test that uniform values with center coords produce uniform output."""
    B, L, K, G, D, Q, spatial_idx = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = _make_inputs(B, L, K, G, D, Q, spatial_idx)
    uniform_val = 2.71828

    # Set all values to uniform constant
    value_uniform = torch.full_like(inputs["value"], uniform_val)

    # Set coordinates to center (0.5, 0.5)
    coords_dim = L * K * 2
    inputs["sampling_loc_attn"][..., :coords_dim] = 0.5

    out_cuda = _run_cuda(
        value_uniform,
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        inputs["im2col_step"],
        inputs["K"],
    )

    # Output should be uniform_val (weighted sum of same values)
    expected = torch.full((B, Q, G * D), uniform_val, device="cuda")
    torch.testing.assert_close(out_cuda, expected, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


@settings(max_examples=2, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_deterministic_output(shape, seed) -> None:
    """Test that same inputs produce identical outputs."""
    B, L, K, G, D, Q, spatial_idx = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = _make_inputs(B, L, K, G, D, Q, spatial_idx)

    out1 = _run_cuda(
        inputs["value"],
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        inputs["im2col_step"],
        inputs["K"],
    )
    out2 = _run_cuda(
        inputs["value"],
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        inputs["im2col_step"],
        inputs["K"],
    )

    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


@settings(max_examples=3, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_zero_value_produces_zero_output(shape, seed) -> None:
    """Test that zero values produce zero output."""
    B, L, K, G, D, Q, spatial_idx = shape

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = _make_inputs(B, L, K, G, D, Q, spatial_idx)
    zero_value = torch.zeros_like(inputs["value"])

    out_cuda = _run_cuda(
        zero_value,
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        inputs["im2col_step"],
        inputs["K"],
    )

    torch.testing.assert_close(out_cuda, torch.zeros_like(out_cuda), rtol=0, atol=0)

    out_ref = _run_ref(
        zero_value,
        inputs["spatial_shapes"],
        inputs["level_start_index"],
        inputs["sampling_loc_attn"],
        inputs["im2col_step"],
        inputs["K"],
    )
    torch.testing.assert_close(out_ref, torch.zeros_like(out_ref), rtol=0, atol=0)
