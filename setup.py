# setup.py
import glob
import os
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

THIS_DIR = Path(__file__).resolve().parent


def _rel_from_setup(path: Path) -> str:
    """Return a POSIX relative path from setup.py directory.

    Newer setuptools editable builds reject absolute paths in setup() args.
    """
    return os.path.relpath(path, THIS_DIR).replace(os.sep, "/")


# --- Sources (matches CMakeLists.txt) ---
# IMPORTANT: setuptools.build_meta editable builds require *relative* paths.
CPP_SOURCES = [_rel_from_setup(THIS_DIR / "dcnv4" / "csrc" / "ops.cpp")]
CUDA_SOURCES = sorted(
    _rel_from_setup(Path(p))
    for p in glob.glob(str(THIS_DIR / "dcnv4" / "csrc" / "cuda" / "*.cu"))
)

if not CUDA_SOURCES:
    msg = "No CUDA sources found: dcnv4/csrc/cuda/*.cu"
    raise RuntimeError(msg)

# --- Build toggles ---
# You can append flags from env without editing files:
#   DCNV4_CXX_FLAGS="-g" DCNV4_NVCC_FLAGS="--use_fast_math" pip install -v .
CXX_FLAGS_EXTRA = os.environ.get("DCNV4_CXX_FLAGS", "").split()
NVCC_FLAGS_EXTRA = os.environ.get("DCNV4_NVCC_FLAGS", "").split()

# Optional debug: DEBUG=1 -> add -g (keeps -O3 unless you want to remove it)
DEBUG = os.environ.get("DEBUG", "0") == "1"

if CUDA_HOME is None and os.environ.get("FORCE_CUDA", "0") != "1":
    msg = (
        "CUDA_HOME is None (nvcc not found). "
        "Install CUDA toolkit / use a PyTorch devel image, or set FORCE_CUDA=1 if you know what you're doing."
    )
    raise RuntimeError(
        msg,
    )

# --- Compile definitions (matches CMakeLists.txt) ---
# CMake: WITH_CUDA, TORCH_EXTENSION_NAME=dcnv4_C
define_macros = [
    ("WITH_CUDA", None),
    ("TORCH_EXTENSION_NAME", "dcnv4_C"),
]

# CMake had these CUDA-only compile definitions:
cuda_only_defines = [
    "__CUDA_NO_FP4_CONVERSION_OPERATORS__",
    "__CUDA_NO_FP4_CONVERSIONS__",
    "__CUDA_NO_FP6_CONVERSION_OPERATORS__",
    "__CUDA_NO_FP6_CONVERSIONS__",
    "__CUDA_NO_FP8_CONVERSION_OPERATORS__",
    "__CUDA_NO_FP8_CONVERSIONS__",
    "__CUDA_NO_HALF_CONVERSIONS__",
    "__CUDA_NO_HALF_OPERATORS__",
    "__CUDA_NO_HALF2_OPERATORS__",
    "__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "__CUDA_NO_BFLOAT16_OPERATORS__",
    "__CUDA_NO_BFLOAT162_OPERATORS__",
]

# --- Compile flags (matches CMakeLists.txt) ---
CSRC_ABS = str(THIS_DIR / "dcnv4" / "csrc")

cxx_flags = ["-O3", "-std=c++17", f"-I{CSRC_ABS}"]
if DEBUG:
    cxx_flags += ["-g"]

nvcc_flags = (
    [
        "-O3",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-std=c++17",
    ]
    + [f"-I{CSRC_ABS}"]
    + [f"-D{d}" for d in cuda_only_defines]
)

# Windows: match the common torch extension pattern
if sys.platform == "win32":
    # If you build with MSVC, these typically help
    cxx_flags = ["/O2", "/std:c++17"] + (["/Zi"] if DEBUG else [])
    nvcc_flags += ["-Xcompiler", "/Zc:__cplusplus"]

# Append user-provided extra flags
cxx_flags += CXX_FLAGS_EXTRA
nvcc_flags += NVCC_FLAGS_EXTRA

ext_modules = [
    CUDAExtension(
        name="dcnv4.dcnv4_C",
        sources=CPP_SOURCES + CUDA_SOURCES,
        include_dirs=[
            _rel_from_setup(THIS_DIR / "dcnv4" / "csrc"),
        ],
        define_macros=define_macros,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    ),
]

setup(
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
