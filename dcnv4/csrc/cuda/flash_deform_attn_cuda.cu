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

#include "cuda/flash_deform_col2im_cuda.cuh"
#include "cuda/flash_deform_im2col_cuda.cuh"
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

at::Tensor flash_deform_attn_cuda_forward(const at::Tensor &value,
                                          const at::Tensor &spatial_shapes,
                                          const at::Tensor &level_start_index,
                                          const at::Tensor &sampling_loc_attn,
                                          const int im2col_step, const int K,
                                          const int d_stride,
                                          const int block_thread) {
  TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
  TORCH_CHECK(spatial_shapes.is_contiguous(),
              "spatial_shapes tensor has to be contiguous");
  TORCH_CHECK(level_start_index.is_contiguous(),
              "level_start_index tensor has to be contiguous");
  TORCH_CHECK(sampling_loc_attn.is_contiguous(),
              "sampling_loc_attn tensor has to be contiguous");

  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
  TORCH_CHECK(level_start_index.is_cuda(),
              "level_start_index must be a CUDA tensor");
  TORCH_CHECK(sampling_loc_attn.is_cuda(),
              "sampling_loc_attn must be a CUDA tensor");

  // Type and shape validation
  TORCH_CHECK(value.scalar_type() == sampling_loc_attn.scalar_type(),
              "value and sampling_loc_attn must have the same dtype");
  TORCH_CHECK(value.device() == spatial_shapes.device() &&
                  value.device() == level_start_index.device() &&
                  value.device() == sampling_loc_attn.device(),
              "All tensors must be on the same device");

  at::cuda::CUDAGuard device_guard(value.device());

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int num_channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);
  const int num_query = sampling_loc_attn.size(1);
  const int num_point = K;

  auto output = torch::zeros({batch, num_query, num_heads, num_channels},
                             value.options());

  auto per_value_size = spatial_size * num_heads * num_channels;
  auto per_offset_size = num_query * num_heads * num_levels * num_point * 3;
  auto per_out_size = num_query * num_heads * num_channels;

  for (int n = 0; n < batch; n += im2col_step) {
    int current_batch = std::min(batch - n, im2col_step);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(),
        "flash_deform_attn_forward_cuda", ([&] {
          flash_deformable_im2col_cuda(
              at::cuda::getCurrentCUDAStream(),
              value.data_ptr<scalar_t>() + n * per_value_size,
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(),
              sampling_loc_attn.data_ptr<scalar_t>() + n * per_offset_size,
              output.data_ptr<scalar_t>() + n * per_out_size, current_batch,
              spatial_size, num_heads, num_channels, num_levels, num_query,
              num_point, d_stride, block_thread, true);
        }));
  }
  output = output.view({batch, num_query, num_heads * num_channels});
  return output;
}

std::vector<at::Tensor> flash_deform_attn_cuda_backward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc_attn,
    const at::Tensor &grad_output, const int im2col_step, const int K,
    const int d_stride, const int block_thread) {
  TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
  TORCH_CHECK(spatial_shapes.is_contiguous(),
              "spatial_shapes tensor has to be contiguous");
  TORCH_CHECK(level_start_index.is_contiguous(),
              "level_start_index tensor has to be contiguous");
  TORCH_CHECK(sampling_loc_attn.is_contiguous(),
              "sampling_loc_attn tensor has to be contiguous");
  TORCH_CHECK(grad_output.is_contiguous(),
              "grad_output tensor has to be contiguous");

  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
  TORCH_CHECK(level_start_index.is_cuda(),
              "level_start_index must be a CUDA tensor");
  TORCH_CHECK(sampling_loc_attn.is_cuda(),
              "sampling_loc_attn must be a CUDA tensor");
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

  // Type and shape validation
  TORCH_CHECK(value.scalar_type() == sampling_loc_attn.scalar_type(),
              "value and sampling_loc_attn must have the same dtype");
  TORCH_CHECK(value.scalar_type() == grad_output.scalar_type(),
              "value and grad_output must have the same dtype");
  TORCH_CHECK(value.device() == spatial_shapes.device() &&
                  value.device() == level_start_index.device() &&
                  value.device() == sampling_loc_attn.device() &&
                  value.device() == grad_output.device(),
              "All tensors must be on the same device");

  at::cuda::CUDAGuard device_guard(value.device());

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int num_channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);
  const int num_query = sampling_loc_attn.size(1);
  const int num_point = K;

  auto dtype = value.dtype();
  if (dtype == at::kHalf || dtype == at::kBFloat16) {
    dtype = at::kFloat;
  }

  auto grad_input = torch::zeros_like(value, dtype);
  auto grad_offset = torch::zeros_like(sampling_loc_attn, dtype);

  auto per_value_size = spatial_size * num_heads * num_channels;
  auto per_offset_size = num_query * num_heads * num_levels * num_point * 3;
  auto per_out_size = num_query * num_heads * num_channels;

  for (int n = 0; n < batch; n += im2col_step) {
    int current_batch = std::min(batch - n, im2col_step);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(),
        "flash_deform_attn_backward_cuda", ([&] {
          flash_deformable_col2im_cuda(
              at::cuda::getCurrentCUDAStream(),
              value.data_ptr<scalar_t>() + n * per_value_size,
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(),
              sampling_loc_attn.data_ptr<scalar_t>() + n * per_offset_size,
              grad_output.data_ptr<scalar_t>() + n * per_out_size,
              current_batch, spatial_size, num_heads, num_channels, num_levels,
              num_query, num_point,
              grad_input.data_ptr<opmath_t>() + n * per_value_size,
              grad_offset.data_ptr<opmath_t>() + n * per_offset_size, d_stride,
              block_thread);
        }));
  }

  if (value.dtype() == at::kHalf) {
    grad_offset = grad_offset.clamp(-65504.0, 65504.0);
    return {grad_input.to(at::kHalf), grad_offset.to(at::kHalf)};
  } else if (value.dtype() == at::kBFloat16) {
    return {grad_input.to(at::kBFloat16), grad_offset.to(at::kBFloat16)};
  } else {
    return {grad_input, grad_offset};
  }
}