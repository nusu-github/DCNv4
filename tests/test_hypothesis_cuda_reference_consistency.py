"""CUDA vs PyTorch reference consistency tests for DCNv4.

Tolerance values are derived from empirical measurements (see scripts/measure_tolerance.py).
Key findings from measurement (NVIDIA L40S, 500 seeds, 15 shapes):

    Forward (float32):
        max_abs = 5.7e-6, p99.9 = 3.8e-6
        Recommended: atol = 1e-5 (with 2x safety margin)

    Gradient (float32):
        grad_input:  max_abs = 1.1e-5, p99.9 = 9.5e-6
        grad_offset: max_abs = 3.1e-5, p99.9 = 3.1e-5
        Non-determinism (atomicAdd): max = 9.5e-6
        Recommended: atol = 5e-5 (accounts for non-determinism)

    Half precision:
        float16:  max_abs = 7.8e-3
        bfloat16: max_abs = 6.3e-2
        Recommended: atol = 2e-2 (fp16), 1.5e-1 (bf16)

Note: rtol is not used because relative error explodes for near-zero values
(observed max_rel > 30 due to division by small reference values).
"""

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dcnv4.functions import compute_offset_mask_channels
from dcnv4.functions.dcnv4_func import dcnv4_forward
from dcnv4.functions.dcnv4_pytorch import dcnv4_forward_pytorch
from dcnv4.functions.table import TABLE

# =============================================================================
# Empirically-derived tolerance constants
# Source: scripts/measure_tolerance.py (500 seeds x 15 shapes on NVIDIA L40S)
# =============================================================================

# Forward pass (float32): max_abs = 5.7e-6, using ~2x safety margin
FORWARD_ATOL = 1e-5

# Gradient computation (float32): max_abs = 3.1e-5, plus non-determinism floor
# Non-determinism from atomicAdd: max = 9.5e-6
GRAD_ATOL = 5e-5

# Half precision tolerances
# float16: max_abs = 7.8e-3 -> 2e-2 with margin
# bfloat16: max_abs = 6.3e-2 -> 1.5e-1 with margin
HALF_FLOAT16_ATOL = 2e-2
HALF_BFLOAT16_ATOL = 1.5e-1

# Self-consistency tests (same implementation, different params)
# Should be exact or near-exact
SELF_CONSISTENCY_ATOL = 1e-5

# Out-of-bounds tests (output should be exactly zero)
OOB_ATOL = 1e-6


def _valid_shapes() -> list[tuple[int, int, int, int, int]]:
    shapes: list[tuple[int, int, int, int, int]] = []
    for key in TABLE:
        b_str, h_str, w_str, g_str, c_str = key.split("x")
        b, h, w, g, c = int(b_str), int(h_str), int(w_str), int(g_str), int(c_str)
        if b == 1 and h <= 64 and w <= 64 and c == 16:
            shapes.append((b, h, w, g, c))
    return shapes


VALID_SHAPES = _valid_shapes()

if not VALID_SHAPES:
    pytest.skip(
        "No valid DCNv4 shapes found in TABLE for hypothesis test.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for DCNv4 CUDA/PyTorch consistency tests",
)

KERNEL = 3
STRIDE = 1
PAD = 1
DILATION = 1


def _maybe_skip_cuda(exc: RuntimeError) -> None:
    msg = str(exc)
    if "invalid kernel shape" in msg or "block_multiplier" in msg:
        assume(False)
    raise


def _run_cuda(
    input_tensor: torch.Tensor,
    offset_mask: torch.Tensor,
    group: int,
    group_channels: int,
    remove_center: int,
    *,
    im2col_step: int = 64,
    softmax: bool = False,
) -> torch.Tensor:
    try:
        return dcnv4_forward(
            input_tensor,
            offset_mask,
            KERNEL,
            KERNEL,
            STRIDE,
            STRIDE,
            PAD,
            PAD,
            DILATION,
            DILATION,
            group,
            group_channels,
            1.0,
            im2col_step,
            remove_center,
            softmax,
        )
    except RuntimeError as exc:
        _maybe_skip_cuda(exc)
        raise


def _run_ref(
    input_tensor: torch.Tensor,
    offset_mask: torch.Tensor,
    group: int,
    group_channels: int,
    remove_center: int,
    *,
    im2col_step: int = 64,
    softmax: bool = False,
) -> torch.Tensor:
    return dcnv4_forward_pytorch(
        input_tensor,
        offset_mask,
        KERNEL,
        KERNEL,
        STRIDE,
        STRIDE,
        PAD,
        PAD,
        DILATION,
        DILATION,
        group,
        group_channels,
        1.0,
        im2col_step,
        remove_center,
        softmax,
    )


@settings(max_examples=6, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    remove_center=st.sampled_from([0, 1]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_dcnv4_cuda_matches_pytorch_reference_hypothesis(
    shape,
    remove_center,
    seed,
) -> None:
    b, h, w, group, group_channels = shape
    if KERNEL == 1 and remove_center:
        assume(False)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    c_total = group * group_channels
    h_out = h
    w_out = w
    offset_mask_channels = compute_offset_mask_channels(group, KERNEL, remove_center)

    base_input = torch.randn(b, h, w, c_total, device="cuda", dtype=torch.float32)
    base_offset_mask = torch.randn(
        b,
        h_out,
        w_out,
        offset_mask_channels,
        device="cuda",
        dtype=torch.float32,
    )

    input_cuda = base_input.clone().requires_grad_(True)
    offset_cuda = base_offset_mask.clone().requires_grad_(True)

    input_ref = base_input.clone().requires_grad_(True)
    offset_ref = base_offset_mask.clone().requires_grad_(True)

    out_cuda = _run_cuda(input_cuda, offset_cuda, group, group_channels, remove_center)
    out_ref = _run_ref(input_ref, offset_ref, group, group_channels, remove_center)

    assert out_cuda.shape == out_ref.shape
    torch.testing.assert_close(out_cuda, out_ref, rtol=0, atol=FORWARD_ATOL)

    grad = torch.randn_like(out_cuda)
    out_cuda.backward(grad)
    out_ref.backward(grad)

    assert input_cuda.grad is not None
    assert offset_cuda.grad is not None
    assert input_ref.grad is not None
    assert offset_ref.grad is not None

    torch.testing.assert_close(input_cuda.grad, input_ref.grad, rtol=0, atol=GRAD_ATOL)
    torch.testing.assert_close(
        offset_cuda.grad,
        offset_ref.grad,
        rtol=0,
        atol=GRAD_ATOL,
    )


@settings(max_examples=4, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    remove_center=st.sampled_from([0, 1]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_offset_mask_padding_ignored(shape, remove_center, seed) -> None:
    b, h, w, group, group_channels = shape
    k_points = KERNEL * KERNEL - remove_center
    needed = group * k_points * 3
    padded = compute_offset_mask_channels(group, KERNEL, remove_center)
    assume(padded > needed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    c_total = group * group_channels
    h_out = h
    w_out = w

    input_tensor = torch.randn(b, h, w, c_total, device="cuda", dtype=torch.float32)
    base_offset = torch.randn(
        b,
        h_out,
        w_out,
        padded,
        device="cuda",
        dtype=torch.float32,
    )

    offset_zero = base_offset.clone()
    offset_zero[..., needed:] = 0

    offset_rand = base_offset.clone()
    offset_rand[..., needed:] = torch.randn_like(offset_rand[..., needed:])

    out_cuda_zero = _run_cuda(
        input_tensor,
        offset_zero,
        group,
        group_channels,
        remove_center,
    )
    out_cuda_rand = _run_cuda(
        input_tensor,
        offset_rand,
        group,
        group_channels,
        remove_center,
    )
    # Padding region should be completely ignored - expect exact equality
    torch.testing.assert_close(
        out_cuda_zero,
        out_cuda_rand,
        rtol=0,
        atol=SELF_CONSISTENCY_ATOL,
    )

    out_ref_zero = _run_ref(
        input_tensor,
        offset_zero,
        group,
        group_channels,
        remove_center,
    )
    out_ref_rand = _run_ref(
        input_tensor,
        offset_rand,
        group,
        group_channels,
        remove_center,
    )
    torch.testing.assert_close(
        out_ref_zero,
        out_ref_rand,
        rtol=0,
        atol=SELF_CONSISTENCY_ATOL,
    )


@settings(max_examples=4, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    remove_center=st.sampled_from([0, 1]),
    im2col_step=st.sampled_from([1, 2, 8, 64, 256]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_im2col_step_invariance(shape, remove_center, im2col_step, seed) -> None:
    b, h, w, group, group_channels = shape
    assume(im2col_step != 64)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    c_total = group * group_channels
    h_out = h
    w_out = w
    offset_mask_channels = compute_offset_mask_channels(group, KERNEL, remove_center)

    input_tensor = torch.randn(b, h, w, c_total, device="cuda", dtype=torch.float32)
    offset_mask = torch.randn(
        b,
        h_out,
        w_out,
        offset_mask_channels,
        device="cuda",
        dtype=torch.float32,
    )

    out_base = _run_cuda(
        input_tensor,
        offset_mask,
        group,
        group_channels,
        remove_center,
        im2col_step=64,
    )
    out_alt = _run_cuda(
        input_tensor,
        offset_mask,
        group,
        group_channels,
        remove_center,
        im2col_step=im2col_step,
    )

    # Same CUDA kernel with different batch grouping - should be identical
    torch.testing.assert_close(out_base, out_alt, rtol=0, atol=SELF_CONSISTENCY_ATOL)


@settings(max_examples=4, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    remove_center=st.sampled_from([0, 1]),
)
def test_oob_offsets_yield_zero_output(shape, remove_center) -> None:
    b, h, w, group, group_channels = shape
    c_total = group * group_channels
    h_out = h
    w_out = w

    k_points = KERNEL * KERNEL - remove_center
    needed = group * k_points * 3
    padded = compute_offset_mask_channels(group, KERNEL, remove_center)

    input_tensor = torch.randn(b, h, w, c_total, device="cuda", dtype=torch.float32)

    offset_mask = torch.zeros(
        b,
        h_out,
        w_out,
        padded,
        device="cuda",
        dtype=torch.float32,
    )
    for g in range(group):
        start = g * k_points * 3
        offset_mask[..., start : start + 2 * k_points] = 1000.0
        offset_mask[..., start + 2 * k_points : start + 3 * k_points] = torch.randn(
            b,
            h_out,
            w_out,
            k_points,
            device="cuda",
            dtype=torch.float32,
        ).reshape(b, h_out, w_out, k_points)

    if padded > needed:
        offset_mask[..., needed:] = torch.randn_like(offset_mask[..., needed:])

    input_cuda = input_tensor.clone().requires_grad_(True)
    offset_cuda = offset_mask.clone().requires_grad_(True)

    out_cuda = _run_cuda(input_cuda, offset_cuda, group, group_channels, remove_center)
    # OOB sampling should produce exactly zero output
    torch.testing.assert_close(
        out_cuda,
        torch.zeros_like(out_cuda),
        rtol=0,
        atol=OOB_ATOL,
    )

    out_cuda.sum().backward()
    assert input_cuda.grad is not None
    torch.testing.assert_close(
        input_cuda.grad,
        torch.zeros_like(input_cuda.grad),
        rtol=0,
        atol=OOB_ATOL,
    )

    out_ref = _run_ref(input_tensor, offset_mask, group, group_channels, remove_center)
    torch.testing.assert_close(
        out_ref,
        torch.zeros_like(out_ref),
        rtol=0,
        atol=OOB_ATOL,
    )


@settings(max_examples=4, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    remove_center=st.sampled_from([0, 1]),
    shift=st.floats(
        min_value=-2.0,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_softmax_shift_invariance(shape, remove_center, shift, seed) -> None:
    b, h, w, group, group_channels = shape
    k_points = KERNEL * KERNEL - remove_center
    padded = compute_offset_mask_channels(group, KERNEL, remove_center)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    c_total = group * group_channels
    h_out = h
    w_out = w

    input_tensor = torch.randn(b, h, w, c_total, device="cuda", dtype=torch.float32)
    offset_mask = torch.randn(
        b,
        h_out,
        w_out,
        padded,
        device="cuda",
        dtype=torch.float32,
    )

    offset_shift = offset_mask.clone()
    for g in range(group):
        start = g * k_points * 3
        offset_shift[..., start + 2 * k_points : start + 3 * k_points] += shift

    out_cuda_base = _run_cuda(
        input_tensor,
        offset_mask,
        group,
        group_channels,
        remove_center,
        softmax=True,
    )
    out_cuda_shift = _run_cuda(
        input_tensor,
        offset_shift,
        group,
        group_channels,
        remove_center,
        softmax=True,
    )
    # Softmax is shift-invariant: softmax(x + c) = softmax(x)
    torch.testing.assert_close(
        out_cuda_base,
        out_cuda_shift,
        rtol=0,
        atol=SELF_CONSISTENCY_ATOL,
    )

    out_ref_base = _run_ref(
        input_tensor,
        offset_mask,
        group,
        group_channels,
        remove_center,
        softmax=True,
    )
    out_ref_shift = _run_ref(
        input_tensor,
        offset_shift,
        group,
        group_channels,
        remove_center,
        softmax=True,
    )
    torch.testing.assert_close(
        out_ref_base,
        out_ref_shift,
        rtol=0,
        atol=SELF_CONSISTENCY_ATOL,
    )


@settings(max_examples=3, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    remove_center=st.sampled_from([0, 1]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_linearity_fixed_offset_mask(shape, remove_center, seed) -> None:
    b, h, w, group, group_channels = shape
    c_total = group * group_channels
    h_out = h
    w_out = w
    offset_mask_channels = compute_offset_mask_channels(group, KERNEL, remove_center)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    input_a = torch.randn(b, h, w, c_total, device="cuda", dtype=torch.float32)
    input_b = torch.randn(b, h, w, c_total, device="cuda", dtype=torch.float32)
    offset_mask = torch.randn(
        b,
        h_out,
        w_out,
        offset_mask_channels,
        device="cuda",
        dtype=torch.float32,
    )

    out_cuda_sum = _run_cuda(
        input_a + input_b,
        offset_mask,
        group,
        group_channels,
        remove_center,
    )
    out_cuda_parts = _run_cuda(
        input_a,
        offset_mask,
        group,
        group_channels,
        remove_center,
    ) + _run_cuda(input_b, offset_mask, group, group_channels, remove_center)
    # Bilinear interpolation is linear: f(a+b) = f(a) + f(b)
    torch.testing.assert_close(out_cuda_sum, out_cuda_parts, rtol=0, atol=FORWARD_ATOL)

    out_ref_sum = _run_ref(
        input_a + input_b,
        offset_mask,
        group,
        group_channels,
        remove_center,
    )
    out_ref_parts = _run_ref(
        input_a,
        offset_mask,
        group,
        group_channels,
        remove_center,
    ) + _run_ref(input_b, offset_mask, group, group_channels, remove_center)
    torch.testing.assert_close(out_ref_sum, out_ref_parts, rtol=0, atol=FORWARD_ATOL)


@settings(max_examples=3, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    remove_center=st.sampled_from([0, 1]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_half_precision_parity(shape, remove_center, seed) -> None:
    b, h, w, group, group_channels = shape
    c_total = group * group_channels
    h_out = h
    w_out = w
    offset_mask_channels = compute_offset_mask_channels(group, KERNEL, remove_center)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for dtype in (torch.float16, torch.bfloat16):
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue

        input_tensor = torch.randn(b, h, w, c_total, device="cuda", dtype=dtype)
        offset_mask = torch.randn(
            b,
            h_out,
            w_out,
            offset_mask_channels,
            device="cuda",
            dtype=dtype,
        )

        out_cuda = _run_cuda(
            input_tensor,
            offset_mask,
            group,
            group_channels,
            remove_center,
        )
        out_ref = _run_ref(
            input_tensor,
            offset_mask,
            group,
            group_channels,
            remove_center,
        )

        assert out_cuda.dtype == dtype
        assert out_ref.dtype == dtype

        # Use dtype-specific tolerances
        atol = HALF_BFLOAT16_ATOL if dtype == torch.bfloat16 else HALF_FLOAT16_ATOL
        torch.testing.assert_close(out_cuda, out_ref, rtol=0, atol=atol)
