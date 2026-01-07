# ------------------------------------------------------------------------------------------------
# Deformable Convolution v4
# Copyright (c) 2024 OpenGVLab
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def get_extensions():
    this_dir = Path(__file__).parent.resolve()
    extensions_dir = this_dir / "src"

    main_sources = list(extensions_dir.glob("*.cpp"))
    source_cpu = list((extensions_dir / "cpu").glob("*.cpp"))
    source_cuda = list((extensions_dir / "cuda").glob("*.cu"))

    sources = main_sources + source_cpu

    extension_type = CppExtension
    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    if (torch.cuda.is_available() or CUDA_HOME is not None or force_cuda) and len(
        source_cuda,
    ) > 0:
        extension_type = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    sources = [str(s.relative_to(this_dir)) for s in sources]

    include_dirs = [str(extensions_dir)]

    return [
        extension_type(
            name="dcnv4._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
    ]


setup(
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
