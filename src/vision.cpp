/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <Python.h>
#include <torch/library.h>

#include "dcnv4.h"

// Macro helpers for token expansion
#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// Expands NAME macro before passing to TORCH_LIBRARY
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// Register Python module so the .so can be imported
#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

// Expands NAME macro before passing to TORCH_LIBRARY_IMPL
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE)                        \
  TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)

// Schema definitions (device-agnostic)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("flash_deform_attn_forward(Tensor value, Tensor spatial_shapes, "
        "Tensor level_start_index, Tensor sampling_loc_attn, "
        "int im2col_step, int K, int d_stride, int block_thread) -> Tensor");
  m.def("flash_deform_attn_backward(Tensor value, Tensor spatial_shapes, "
        "Tensor level_start_index, Tensor sampling_loc_attn, "
        "Tensor grad_output, int im2col_step, int K, int d_stride, "
        "int block_thread) -> Tensor[]");
  m.def("dcnv4_forward(Tensor value, Tensor p_offset, "
        "int kernel_h, int kernel_w, int stride_h, int stride_w, "
        "int pad_h, int pad_w, int dilation_h, int dilation_w, "
        "int group, int group_channels, float offset_scale, "
        "int im2col_step, int remove_center, int d_stride, "
        "int block_thread, bool softmax) -> Tensor");
  m.def("dcnv4_backward(Tensor value, Tensor p_offset, "
        "int kernel_h, int kernel_w, int stride_h, int stride_w, "
        "int pad_h, int pad_w, int dilation_h, int dilation_w, "
        "int group, int group_channels, float offset_scale, "
        "int im2col_step, Tensor grad_output, int remove_center, "
        "int d_stride, int block_thread, bool softmax) -> Tensor[]");
}

// CUDA implementations
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("flash_deform_attn_forward", &flash_deform_attn_forward);
  m.impl("flash_deform_attn_backward", &flash_deform_attn_backward);
  m.impl("dcnv4_forward", &dcnv4_forward);
  m.impl("dcnv4_backward", &dcnv4_backward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
