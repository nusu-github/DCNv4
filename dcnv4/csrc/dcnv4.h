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

#pragma once

#include "cuda/dcnv4_cuda.h"
#include "cuda/flash_deform_attn_cuda.h"

at::Tensor flash_deform_attn_forward(const at::Tensor &value,
                                     const at::Tensor &spatial_shapes,
                                     const at::Tensor &level_start_index,
                                     const at::Tensor &sampling_loc_attn,
                                     int64_t im2col_step, int64_t K,
                                     int64_t d_stride, int64_t block_thread) {
  return flash_deform_attn_cuda_forward(
      value, spatial_shapes, level_start_index, sampling_loc_attn,
      static_cast<int>(im2col_step), static_cast<int>(K),
      static_cast<int>(d_stride), static_cast<int>(block_thread));
}

std::vector<at::Tensor> flash_deform_attn_backward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc_attn,
    const at::Tensor &grad_output, int64_t im2col_step, int64_t K,
    int64_t d_stride, int64_t block_thread) {
  return flash_deform_attn_cuda_backward(
      value, spatial_shapes, level_start_index, sampling_loc_attn, grad_output,
      static_cast<int>(im2col_step), static_cast<int>(K),
      static_cast<int>(d_stride), static_cast<int>(block_thread));
}

at::Tensor dcnv4_forward(const at::Tensor &value, const at::Tensor &p_offset,
                         int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                         int64_t stride_w, int64_t pad_h, int64_t pad_w,
                         int64_t dilation_h, int64_t dilation_w, int64_t group,
                         int64_t group_channels, double offset_scale,
                         int64_t im2col_step, int64_t remove_center,
                         int64_t d_stride, int64_t block_thread, bool softmax) {
  return dcnv4_cuda_forward(
      value, p_offset, static_cast<int>(kernel_h), static_cast<int>(kernel_w),
      static_cast<int>(stride_h), static_cast<int>(stride_w),
      static_cast<int>(pad_h), static_cast<int>(pad_w),
      static_cast<int>(dilation_h), static_cast<int>(dilation_w),
      static_cast<int>(group), static_cast<int>(group_channels),
      static_cast<float>(offset_scale), static_cast<int>(im2col_step),
      static_cast<int>(remove_center), static_cast<int>(d_stride),
      static_cast<int>(block_thread), softmax);
}

std::vector<at::Tensor>
dcnv4_backward(const at::Tensor &value, const at::Tensor &p_offset,
               int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
               int64_t stride_w, int64_t pad_h, int64_t pad_w,
               int64_t dilation_h, int64_t dilation_w, int64_t group,
               int64_t group_channels, double offset_scale, int64_t im2col_step,
               const at::Tensor &grad_output, int64_t remove_center,
               int64_t d_stride, int64_t block_thread, bool softmax) {
  return dcnv4_cuda_backward(
      value, p_offset, static_cast<int>(kernel_h), static_cast<int>(kernel_w),
      static_cast<int>(stride_h), static_cast<int>(stride_w),
      static_cast<int>(pad_h), static_cast<int>(pad_w),
      static_cast<int>(dilation_h), static_cast<int>(dilation_w),
      static_cast<int>(group), static_cast<int>(group_channels),
      static_cast<float>(offset_scale), static_cast<int>(im2col_step),
      grad_output, static_cast<int>(remove_center), static_cast<int>(d_stride),
      static_cast<int>(block_thread), softmax);
}