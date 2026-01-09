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

#include "cuda/dcnv4_col2im_cuda.cuh"
#include "cuda/dcnv4_im2col_cuda.cuh"
#include <vector>

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

at::Tensor dcnv4_cuda_forward(
    const at::Tensor &value, const at::Tensor &p_offset, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int group, const int group_channels, const float offset_scale,
    const int im2col_step, const int remove_center, const int d_stride,
    const int block_thread, const bool softmax) {
  TORCH_CHECK(value.is_contiguous(), "input tensor has to be contiguous");
  TORCH_CHECK(value.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(p_offset.is_contiguous(), "input tensor has to be contiguous");
  TORCH_CHECK(p_offset.is_cuda(), "input must be a CUDA tensor");

  // Type and shape validation
  TORCH_CHECK(value.scalar_type() == p_offset.scalar_type(),
              "value and p_offset must have the same dtype");
  TORCH_CHECK(value.device() == p_offset.device(),
              "value and p_offset must be on the same device");
  TORCH_CHECK(kernel_h > 0 && kernel_w > 0,
              "kernel dimensions must be positive");

  at::cuda::CUDAGuard device_guard(value.device());

  const int batch = value.size(0);
  const int height_in = value.size(1);
  const int width_in = value.size(2);
  const int channels = value.size(3);
  const int padded_offset_dim = p_offset.size(3);

  // tensor core requirement
  TORCH_CHECK(padded_offset_dim % 8 == 0, "padded_offset_dim (",
              padded_offset_dim,
              ") must be divisible by 8 for tensor core requirements");

  const int height_out =
      (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int width_out =
      (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  TORCH_CHECK(channels == (group * group_channels),
              "Input channels and group times group channels wont match: (",
              channels, " vs ", group * group_channels, ").");

  auto output = torch::zeros(
      {batch, height_out, width_out, group * group_channels}, value.options());

  auto per_value_size = height_in * width_in * channels;
  auto per_offset_size = height_out * width_out * padded_offset_dim;
  auto per_out_size = height_out * width_out * group * group_channels;

  for (int n = 0; n < batch; n += im2col_step) {
    int current_batch = std::min(batch - n, im2col_step);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(),
        "dcnv4_forward_cuda", ([&] {
          dcnv4_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                            value.data_ptr<scalar_t>() + n * per_value_size,
                            p_offset.data_ptr<scalar_t>() + n * per_offset_size,
                            output.data_ptr<scalar_t>() + n * per_out_size,
                            kernel_h, kernel_w, stride_h, stride_w, pad_h,
                            pad_w, dilation_h, dilation_w, group,
                            group_channels, current_batch, height_in, width_in,
                            height_out, width_out, offset_scale, remove_center,
                            d_stride, block_thread, softmax, padded_offset_dim);
        }));
  }

  return output;
}

std::vector<at::Tensor>
dcnv4_cuda_backward(const at::Tensor &value, const at::Tensor &p_offset,
                    const int kernel_h, const int kernel_w, const int stride_h,
                    const int stride_w, const int pad_h, const int pad_w,
                    const int dilation_h, const int dilation_w, const int group,
                    const int group_channels, const float offset_scale,
                    const int im2col_step, const at::Tensor &grad_output,
                    const int remove_center, const int d_stride,
                    const int block_thread, const bool softmax) {
  TORCH_CHECK(value.is_contiguous(), "input tensor has to be contiguous");
  TORCH_CHECK(p_offset.is_contiguous(), "offset tensor has to be contiguous");
  TORCH_CHECK(grad_output.is_contiguous(),
              "grad_output tensor has to be contiguous");

  TORCH_CHECK(value.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(p_offset.is_cuda(), "offset must be a CUDA tensor");
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

  // Type and shape validation
  TORCH_CHECK(value.scalar_type() == p_offset.scalar_type(),
              "value and p_offset must have the same dtype");
  TORCH_CHECK(value.scalar_type() == grad_output.scalar_type(),
              "value and grad_output must have the same dtype");
  TORCH_CHECK(value.device() == p_offset.device() &&
                  value.device() == grad_output.device(),
              "All tensors must be on the same device");
  TORCH_CHECK(kernel_h > 0 && kernel_w > 0,
              "kernel dimensions must be positive");

  at::cuda::CUDAGuard device_guard(value.device());

  const int batch = value.size(0);
  const int height_in = value.size(1);
  const int width_in = value.size(2);
  const int channels = value.size(3);
  const int padded_offset_dim = p_offset.size(3);
  TORCH_CHECK(padded_offset_dim % 8 == 0, "padded_offset_dim (",
              padded_offset_dim,
              ") must be divisible by 8 for tensor core requirements");

  const int height_out =
      (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int width_out =
      (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  TORCH_CHECK(channels == (group * group_channels),
              "Input channels and group times group channels wont match: (",
              channels, " vs ", group * group_channels, ").");

  auto dtype = value.dtype();
  if (dtype == at::kHalf || dtype == at::kBFloat16) {
    dtype = at::kFloat;
  }

  auto grad_input = torch::zeros_like(value, dtype);
  auto grad_offset = torch::zeros_like(p_offset, dtype);

  auto per_value_size = height_in * width_in * channels;
  auto per_offset_size = height_out * width_out * padded_offset_dim;
  auto per_grad_output_size = height_out * width_out * group * group_channels;

  for (int n = 0; n < batch; n += im2col_step) {
    int current_batch = std::min(batch - n, im2col_step);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(),
        "dcnv4_backward_cuda", ([&] {
          dcnv4_col2im_cuda(
              at::cuda::getCurrentCUDAStream(),
              value.data_ptr<scalar_t>() + n * per_value_size,
              p_offset.data_ptr<scalar_t>() + n * per_offset_size,
              grad_output.data_ptr<scalar_t>() + n * per_grad_output_size,
              kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
              dilation_w, group, group_channels, current_batch, height_in,
              width_in, height_out, width_out, offset_scale, remove_center,
              grad_input.data_ptr<opmath_t>() + n * per_value_size,
              grad_offset.data_ptr<opmath_t>() + n * per_offset_size, d_stride,
              block_thread, softmax, padded_offset_dim);
        }));
  }

  if (value.dtype() == at::kHalf) {
    return {grad_input.to(at::kHalf), grad_offset.to(at::kHalf)};
  } else if (value.dtype() == at::kBFloat16) {
    return {grad_input.to(at::kBFloat16), grad_offset.to(at::kBFloat16)};
  } else {
    return {grad_input, grad_offset};
  }
}