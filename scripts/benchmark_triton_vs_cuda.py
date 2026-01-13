#!/usr/bin/env python
"""Benchmark DCNv4 CUDA vs Triton using triton.testing."""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

import torch
import triton.testing as tt

from dcnv4 import ops
from dcnv4.functions.dcnv4_func import find_spec_bwd, findspec
from dcnv4.functions.dcnv4_triton import (
    dcnv4_backward_triton,
    dcnv4_forward_triton,
    is_triton_available,
)
from dcnv4.functions.table import TABLE
from dcnv4.functions.utils import compute_offset_mask_channels

if TYPE_CHECKING:
    from collections.abc import Iterable

KERNEL = 3
STRIDE = 1
PAD = 1
DILATION = 1
OFFSET_SCALE = 1.0


def _parse_shape(shape: str) -> tuple[int, int, int, int, int]:
    parts = shape.split("x")
    if len(parts) != 5:
        msg = f"Invalid shape '{shape}', expected BxHxWxGxC."
        raise ValueError(msg)
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def _default_shapes() -> list[tuple[int, int, int, int, int]]:
    shapes: list[tuple[int, int, int, int, int]] = []
    for key in TABLE:
        shapes.append(_parse_shape(key))
    return shapes


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    msg = f"Unsupported dtype '{name}'."
    raise ValueError(msg)


def _make_inputs(
    *,
    B: int,
    H: int,
    W: int,
    G: int,
    C: int,
    dtype: torch.dtype,
    remove_center: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    total_c = G * C
    offset_channels = compute_offset_mask_channels(G, KERNEL, remove_center)
    input_tensor = torch.randn(B, H, W, total_c, device=device, dtype=dtype)
    offset_mask = torch.randn(B, H, W, offset_channels, device=device, dtype=dtype)
    return input_tensor.contiguous(), offset_mask.contiguous(), total_c, H, W


def _make_grad_output(
    *,
    B: int,
    H_in: int,
    W_in: int,
    total_c: int,
    dtype: torch.dtype,
    pad_h: int,
    pad_w: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int,
    dilation_w: int,
) -> torch.Tensor:
    h_out = (H_in + 2 * pad_h - (dilation_h * (KERNEL - 1) + 1)) // stride_h + 1
    w_out = (W_in + 2 * pad_w - (dilation_w * (KERNEL - 1) + 1)) // stride_w + 1
    return torch.randn(
        B,
        h_out,
        w_out,
        total_c,
        device="cuda",
        dtype=dtype,
    ).contiguous()


def _bench_dcnv4(
    *,
    B: int,
    H: int,
    W: int,
    G: int,
    C: int,
    provider: str,
    mode: str,
    dtype: torch.dtype,
    remove_center: int,
    softmax: bool,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[float, float, float]:
    input_tensor, offset_mask, total_c, H_in, W_in = _make_inputs(
        B=B,
        H=H,
        W=W,
        G=G,
        C=C,
        dtype=dtype,
        remove_center=remove_center,
    )

    if provider == "cuda":
        d_stride, block_thread = findspec(B, H, W, G, C)
        if mode == "forward":

            def fn():
                return ops.dcnv4_forward(
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
                    G,
                    C,
                    OFFSET_SCALE,
                    64,
                    remove_center,
                    d_stride,
                    block_thread,
                    softmax,
                )
        else:
            grad_output = _make_grad_output(
                B=B,
                H_in=H_in,
                W_in=W_in,
                total_c=total_c,
                dtype=dtype,
                pad_h=PAD,
                pad_w=PAD,
                stride_h=STRIDE,
                stride_w=STRIDE,
                dilation_h=DILATION,
                dilation_w=DILATION,
            )
            bwd_d_stride, bwd_block_thread = find_spec_bwd(B, H, W, G, C)

            def fn():
                return ops.dcnv4_backward(
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
                    G,
                    C,
                    OFFSET_SCALE,
                    64,
                    grad_output,
                    remove_center,
                    bwd_d_stride,
                    bwd_block_thread,
                    softmax,
                )
    elif provider == "triton":
        if mode == "forward":

            def fn():
                return dcnv4_forward_triton(
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
                    G,
                    C,
                    OFFSET_SCALE,
                    64,
                    remove_center,
                    softmax,
                )
        else:
            grad_output = _make_grad_output(
                B=B,
                H_in=H_in,
                W_in=W_in,
                total_c=total_c,
                dtype=dtype,
                pad_h=PAD,
                pad_w=PAD,
                stride_h=STRIDE,
                stride_w=STRIDE,
                dilation_h=DILATION,
                dilation_w=DILATION,
            )

            def fn():
                return dcnv4_backward_triton(
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
                    G,
                    C,
                    OFFSET_SCALE,
                    64,
                    grad_output,
                    remove_center,
                    softmax,
                )
    else:
        msg = f"Unknown provider '{provider}'."
        raise ValueError(msg)

    ms, ms_min, ms_max = tt.do_bench(
        fn,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    return ms, ms_min, ms_max


def _make_report(
    *,
    shapes: Iterable[tuple[int, int, int, int, int]],
    mode: str,
    dtype: torch.dtype,
    remove_center: int,
    softmax: bool,
    warmup_ms: int,
    rep_ms: int,
):
    benchmarks = tt.Benchmark(
        x_names=["B", "H", "W", "G", "C"],
        x_vals=list(shapes),
        line_arg="provider",
        line_vals=["cuda", "triton"],
        line_names=["CUDA", "Triton"],
        plot_name=f"dcnv4_{mode}",
        xlabel="shape (B,H,W,G,C)",
        ylabel="ms",
        args={
            "mode": mode,
            "dtype": dtype,
            "remove_center": remove_center,
            "softmax": softmax,
            "warmup_ms": warmup_ms,
            "rep_ms": rep_ms,
        },
    )

    @tt.perf_report(benchmarks)
    def _bench(
        B,
        H,
        W,
        G,
        C,
        provider,
        mode,
        dtype,
        remove_center,
        softmax,
        warmup_ms,
        rep_ms,
    ):
        return _bench_dcnv4(
            B=B,
            H=H,
            W=W,
            G=G,
            C=C,
            provider=provider,
            mode=mode,
            dtype=dtype,
            remove_center=remove_center,
            softmax=softmax,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )

    return _bench


def _write_fallback_html(
    *,
    save_path: str,
    mode: str,
    header: list[str],
    rows: list[list[str]],
    append: bool,
) -> None:
    if not save_path:
        return
    os.makedirs(save_path, exist_ok=True)
    html_path = os.path.join(save_path, "results.html")
    body = []
    body.append(f"<h2>{mode}</h2>")
    body.append("<table border='1' cellspacing='0' cellpadding='4'>")
    body.append("<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>")
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    body.append("</table>")
    content = "\n".join(body) + "\n"
    if append and os.path.exists(html_path):
        with open(html_path, "a", encoding="utf-8") as f:
            f.write(content)
        return
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><body>\n")
        f.write(content)
        f.write("</body></html>\n")


def _run_fallback_table(
    *,
    shapes: Iterable[tuple[int, int, int, int, int]],
    mode: str,
    dtype: torch.dtype,
    remove_center: int,
    softmax: bool,
    warmup_ms: int,
    rep_ms: int,
    save_path: str,
    append_html: bool,
) -> None:
    header = ["B", "H", "W", "G", "C", "CUDA (ms)", "Triton (ms)"]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header))))
    rows: list[list[str]] = []
    for B, H, W, G, C in shapes:
        results = {}
        for provider in ("cuda", "triton"):
            ms, ms_min, ms_max = _bench_dcnv4(
                B=B,
                H=H,
                W=W,
                G=G,
                C=C,
                provider=provider,
                mode=mode,
                dtype=dtype,
                remove_center=remove_center,
                softmax=softmax,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
            )
            results[provider] = f"{ms:.3f} ({ms_min:.3f}-{ms_max:.3f})"
        row = [
            str(B),
            str(H),
            str(W),
            str(G),
            str(C),
            results["cuda"],
            results["triton"],
        ]
        print(" | ".join(row))
        rows.append(row)
    _write_fallback_html(
        save_path=save_path,
        mode=mode,
        header=header,
        rows=rows,
        append=append_html,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DCNv4 CUDA vs Triton with triton.testing",
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "backward", "both"],
        default="forward",
    )
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--remove-center", action="store_true")
    parser.add_argument("--softmax", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument(
        "--shapes",
        nargs="*",
        default=None,
        help="Override shapes (BxHxWxGxC)",
    )
    parser.add_argument("--print-data", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--save-path", default="", help="Directory to save CSV/plots")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        msg = "CUDA is required for this benchmark."
        raise RuntimeError(msg)

    if not is_triton_available():
        msg = "Triton is not available on this system."
        raise RuntimeError(msg)

    dtype = _dtype_from_name(args.dtype)
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        msg = "bfloat16 is not supported on this GPU."
        raise RuntimeError(msg)

    shapes = (
        [_parse_shape(s) for s in args.shapes] if args.shapes else _default_shapes()
    )

    fallback_written = False

    if args.mode in {"forward", "both"}:
        report = _make_report(
            shapes=shapes,
            mode="forward",
            dtype=dtype,
            remove_center=int(args.remove_center),
            softmax=args.softmax,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        try:
            report.run(
                show_plots=args.show_plots,
                print_data=args.print_data,
                save_path=args.save_path,
            )
        except ModuleNotFoundError:
            _run_fallback_table(
                shapes=shapes,
                mode="forward",
                dtype=dtype,
                remove_center=int(args.remove_center),
                softmax=args.softmax,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
                save_path=args.save_path,
                append_html=False,
            )
            fallback_written = True

    if args.mode in {"backward", "both"}:
        report = _make_report(
            shapes=shapes,
            mode="backward",
            dtype=dtype,
            remove_center=int(args.remove_center),
            softmax=args.softmax,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        try:
            report.run(
                show_plots=args.show_plots,
                print_data=args.print_data,
                save_path=args.save_path,
            )
        except ModuleNotFoundError:
            _run_fallback_table(
                shapes=shapes,
                mode="backward",
                dtype=dtype,
                remove_center=int(args.remove_center),
                softmax=args.softmax,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
                save_path=args.save_path,
                append_html=fallback_written,
            )


if __name__ == "__main__":
    main()
