"""Triton vs PyTorch reference consistency tests for DCNv4.

These tests validate the Triton backend against the pure PyTorch reference.
They mirror the CUDA consistency tests but are gated on Triton availability.
"""

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dcnv4.functions import compute_offset_mask_channels
from dcnv4.functions.dcnv4_func import dcnv4_forward
from dcnv4.functions.dcnv4_pytorch import dcnv4_forward_pytorch
from dcnv4.functions.dcnv4_triton import is_triton_available
from dcnv4.functions.table import TABLE

# =============================================================================
# Tolerance constants (slightly relaxed vs CUDA due to atomic ordering)
# =============================================================================

FORWARD_ATOL = 1e-5
GRAD_ATOL = 1e-4
HALF_FLOAT16_ATOL = 2e-2
HALF_BFLOAT16_ATOL = 1.5e-1
SELF_CONSISTENCY_ATOL = 1e-5
OOB_ATOL = 1e-6


@pytest.fixture(autouse=True)
def _force_triton_backend(monkeypatch) -> None:
    monkeypatch.setenv("DCNV4_USE_TRITON", "1")


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
        "No valid DCNv4 shapes found in TABLE for Triton tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.skipif(
    not is_triton_available(),
    reason="Triton + CUDA are required for DCNv4 Triton tests",
)

KERNEL = 3
STRIDE = 1
PAD = 1
DILATION = 1


def _run_triton(
    input_tensor: torch.Tensor,
    offset_mask: torch.Tensor,
    group: int,
    group_channels: int,
    remove_center: int,
    *,
    im2col_step: int = 64,
    softmax: bool = False,
) -> torch.Tensor:
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


@settings(max_examples=5, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    remove_center=st.sampled_from([0, 1]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_dcnv4_triton_matches_pytorch_reference(shape, remove_center, seed) -> None:
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

    input_triton = base_input.clone().requires_grad_(True)
    offset_triton = base_offset_mask.clone().requires_grad_(True)

    input_ref = base_input.clone().requires_grad_(True)
    offset_ref = base_offset_mask.clone().requires_grad_(True)

    out_triton = _run_triton(
        input_triton,
        offset_triton,
        group,
        group_channels,
        remove_center,
    )
    out_ref = _run_ref(input_ref, offset_ref, group, group_channels, remove_center)

    assert out_triton.shape == out_ref.shape
    torch.testing.assert_close(out_triton, out_ref, rtol=0, atol=FORWARD_ATOL)

    grad = torch.randn_like(out_triton)
    out_triton.backward(grad)
    out_ref.backward(grad)

    assert input_triton.grad is not None
    assert offset_triton.grad is not None
    assert input_ref.grad is not None
    assert offset_ref.grad is not None

    torch.testing.assert_close(
        input_triton.grad,
        input_ref.grad,
        rtol=0,
        atol=GRAD_ATOL,
    )
    torch.testing.assert_close(
        offset_triton.grad,
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
def test_offset_mask_padding_ignored_triton(shape, remove_center, seed) -> None:
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

    out_triton_zero = _run_triton(
        input_tensor,
        offset_zero,
        group,
        group_channels,
        remove_center,
    )
    out_triton_rand = _run_triton(
        input_tensor,
        offset_rand,
        group,
        group_channels,
        remove_center,
    )
    torch.testing.assert_close(
        out_triton_zero,
        out_triton_rand,
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
def test_im2col_step_invariance_triton(shape, remove_center, im2col_step, seed) -> None:
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

    out_base = _run_triton(
        input_tensor,
        offset_mask,
        group,
        group_channels,
        remove_center,
        im2col_step=64,
    )
    out_alt = _run_triton(
        input_tensor,
        offset_mask,
        group,
        group_channels,
        remove_center,
        im2col_step=im2col_step,
    )

    torch.testing.assert_close(out_base, out_alt, rtol=0, atol=SELF_CONSISTENCY_ATOL)


@settings(max_examples=4, deadline=None)
@given(
    shape=st.sampled_from(VALID_SHAPES),
    remove_center=st.sampled_from([0, 1]),
)
def test_oob_offsets_yield_zero_output_triton(shape, remove_center) -> None:
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

    input_triton = input_tensor.clone().requires_grad_(True)
    offset_triton = offset_mask.clone().requires_grad_(True)

    out_triton = _run_triton(
        input_triton,
        offset_triton,
        group,
        group_channels,
        remove_center,
    )
    torch.testing.assert_close(
        out_triton,
        torch.zeros_like(out_triton),
        rtol=0,
        atol=OOB_ATOL,
    )

    out_triton.sum().backward()
    assert input_triton.grad is not None
    torch.testing.assert_close(
        input_triton.grad,
        torch.zeros_like(input_triton.grad),
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
def test_softmax_shift_invariance_triton(shape, remove_center, shift, seed) -> None:
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

    out_triton_base = _run_triton(
        input_tensor,
        offset_mask,
        group,
        group_channels,
        remove_center,
        softmax=True,
    )
    out_triton_shift = _run_triton(
        input_tensor,
        offset_shift,
        group,
        group_channels,
        remove_center,
        softmax=True,
    )
    torch.testing.assert_close(
        out_triton_base,
        out_triton_shift,
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
def test_linearity_fixed_offset_mask_triton(shape, remove_center, seed) -> None:
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

    out_triton_sum = _run_triton(
        input_a + input_b,
        offset_mask,
        group,
        group_channels,
        remove_center,
    )
    out_triton_parts = _run_triton(
        input_a,
        offset_mask,
        group,
        group_channels,
        remove_center,
    ) + _run_triton(input_b, offset_mask, group, group_channels, remove_center)
    torch.testing.assert_close(
        out_triton_sum,
        out_triton_parts,
        rtol=0,
        atol=FORWARD_ATOL,
    )

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
def test_half_precision_parity_triton(shape, remove_center, seed) -> None:
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

        out_triton = _run_triton(
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

        assert out_triton.dtype == dtype
        assert out_ref.dtype == dtype

        atol = HALF_BFLOAT16_ATOL if dtype == torch.bfloat16 else HALF_FLOAT16_ATOL
        torch.testing.assert_close(out_triton, out_ref, rtol=0, atol=atol)
