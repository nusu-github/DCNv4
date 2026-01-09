#!/usr/bin/env python3
"""Run scripts/measure_tolerance.py on Modal (GPU) and save results locally.

This is intended for one-off empirical tolerance measurement when you don't have
a suitable local CUDA environment.

Prereqs (local machine):
  - `pip install modal`
  - `modal setup`

Usage:
  modal run scripts/modal_measure_tolerance.py --seeds 200 --output tolerance.json

Notes:
  - This builds DCNv4 inside a CUDA *devel* image (nvcc available).
  - Results are returned to the local entrypoint and written to `--output`.

"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "dcnv4-measure-tolerance"

# Persist results inside Modal as well (optional convenience).
RESULTS_VOLUME_NAME = "dcnv4-tolerance-results"
RESULTS_DIR = "/root/results"

results_volume = modal.Volume.from_name(
    RESULTS_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)

# CUDA *devel* image is required to compile DCNv4 (nvcc).
# If you need to match a specific CUDA version, change this tag.
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("build-essential", "cmake", "ninja-build", "git")
    .env(
        {
            "CC": "gcc",
            "CXX": "g++",
            "CUDACXX": "/usr/local/cuda/bin/nvcc",
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "TORCH_CUDA_ARCH_LIST": "7.5+PTX",
        },
    )
    .uv_pip_install("numpy", "scikit-build-core>=0.11", "torch", "torchvision")
    .add_local_dir(
        ".",
        "/root/DCNv4",
        ignore=[
            "**/.git/**",
            "**/.venv/**",
            "**/__pycache__/**",
            "**/build/**",
            "**/.ruff_cache/**",
        ],
        copy=True,
    )
    .run_commands(
        "cd /root/DCNv4 && rm -rf build && CC=gcc CXX=g++ CUDACXX=/usr/local/cuda/bin/nvcc pip install --no-build-isolation -v -e .",
    )
)

app = modal.App(APP_NAME)


@app.function(
    gpu=["L40S"],
    image=base_image,
    memory=16 * 1024,
    cpu=4,
    timeout=2 * 60 * 60,
    volumes={RESULTS_DIR: results_volume},
)
def run_measurement(
    *,
    seeds: int,
    output_name: str,
    no_softmax: bool,
    no_half: bool,
) -> dict:
    import subprocess

    out_path = Path(RESULTS_DIR) / output_name

    cmd = [
        "python",
        "/root/DCNv4/scripts/measure_tolerance.py",
        "--seeds",
        str(seeds),
        "--output",
        str(out_path),
    ]
    if no_softmax:
        cmd.append("--no-softmax")
    if no_half:
        cmd.append("--no-half")

    subprocess.run(cmd, check=True)

    # Persist to the Modal Volume.
    results_volume.commit()

    return json.loads(out_path.read_text())


@app.local_entrypoint()
def main(
    seeds: int = 500,
    output: str = "tolerance_measurements.json",
    no_softmax: bool = False,
    no_half: bool = False,
) -> None:
    output_path = Path(output)

    # Store on Modal volume under a stable name; write full JSON locally.
    res = run_measurement.remote(
        seeds=seeds,
        output_name=output_path.name,
        no_softmax=no_softmax,
        no_half=no_half,
    )

    output_path.write_text(json.dumps(res, indent=2) + "\n")
    print(f"Wrote results to: {output_path.resolve()}")
    print(
        f"Modal volume: {RESULTS_VOLUME_NAME} (path {RESULTS_DIR}/{output_path.name})",
    )
