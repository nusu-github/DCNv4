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

// Meta implementations for torch.compile support
at::Tensor dcnv4_forward_meta(const at::Tensor &value,
                              const at::Tensor &p_offset, int64_t kernel_h,
                              int64_t kernel_w, int64_t stride_h,
                              int64_t stride_w, int64_t pad_h, int64_t pad_w,
                              int64_t dilation_h, int64_t dilation_w,
                              int64_t group, int64_t group_channels,
                              double offset_scale, int64_t im2col_step,
                              int64_t remove_center, int64_t d_stride,
                              int64_t block_thread, bool softmax) {

  const int64_t batch = value.size(0);
  const int64_t height_in = value.size(1);
  const int64_t width_in = value.size(2);

  const int64_t height_out =
      (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int64_t width_out =
      (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  return value.new_empty(
      {batch, height_out, width_out, group * group_channels});
}

std::vector<at::Tensor> dcnv4_backward_meta(
    const at::Tensor &value, const at::Tensor &p_offset, int64_t kernel_h,
    int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_h,
    int64_t pad_w, int64_t dilation_h, int64_t dilation_w, int64_t group,
    int64_t group_channels, double offset_scale, int64_t im2col_step,
    const at::Tensor &grad_output, int64_t remove_center, int64_t d_stride,
    int64_t block_thread, bool softmax) {

  return {torch::empty_like(value), torch::empty_like(p_offset)};
}

at::Tensor flash_deform_attn_forward_meta(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc_attn,
    int64_t im2col_step, int64_t K, int64_t d_stride, int64_t block_thread) {

  const int64_t batch = value.size(0);
  const int64_t num_heads = value.size(2);
  const int64_t num_channels = value.size(3);
  const int64_t num_query = sampling_loc_attn.size(1);

  return value.new_empty({batch, num_query, num_heads * num_channels});
}

std::vector<at::Tensor> flash_deform_attn_backward_meta(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc_attn,
    const at::Tensor &grad_output, int64_t im2col_step, int64_t K,
    int64_t d_stride, int64_t block_thread) {

  return {torch::empty_like(value), torch::empty_like(sampling_loc_attn)};
}

// Register Meta implementations
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Meta, m) {
  m.impl("flash_deform_attn_forward", &flash_deform_attn_forward_meta);
  m.impl("flash_deform_attn_backward", &flash_deform_attn_backward_meta);
  m.impl("dcnv4_forward", &dcnv4_forward_meta);
  m.impl("dcnv4_backward", &dcnv4_backward_meta);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)