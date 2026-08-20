/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

constexpr int64_t kHeadDim = 128;

void check_npu_tensor(const torch::Tensor& tensor, const char* name) {
  CHECK(tensor.defined()) << name << " must be defined.";
  CHECK(tensor.device().is_privateuseone())
      << name << " must be an NPU tensor.";
  CHECK(tensor.is_contiguous()) << name << " must be contiguous.";
}

}  // namespace

torch::Tensor npu_mega_gdn_decode(const torch::Tensor& qkv,
                                  const torch::Tensor& z,
                                  const torch::Tensor& b,
                                  const torch::Tensor& a,
                                  const torch::Tensor& conv_weight,
                                  torch::Tensor& conv_state,
                                  const torch::Tensor& a_log,
                                  const torch::Tensor& dt_bias,
                                  torch::Tensor& ssm_state,
                                  const torch::Tensor& read_state_indices,
                                  const torch::Tensor& write_state_indices,
                                  const torch::Tensor& norm_weight) {
  check_npu_tensor(qkv, "qkv");
  check_npu_tensor(z, "z");
  check_npu_tensor(b, "b");
  check_npu_tensor(a, "a");
  check_npu_tensor(conv_weight, "conv_weight");
  check_npu_tensor(conv_state, "conv_state");
  check_npu_tensor(a_log, "a_log");
  check_npu_tensor(dt_bias, "dt_bias");
  check_npu_tensor(ssm_state, "ssm_state");
  check_npu_tensor(read_state_indices, "read_state_indices");
  check_npu_tensor(write_state_indices, "write_state_indices");
  check_npu_tensor(norm_weight, "norm_weight");

  CHECK_EQ(qkv.dim(), 2) << "qkv must be [batch, channels].";
  CHECK_EQ(z.dim(), 3) << "z must be [batch, value_heads, 128].";
  CHECK_EQ(b.dim(), 2) << "b must be [batch, value_heads].";
  CHECK_EQ(a.sizes(), b.sizes()) << "a shape must match b.";
  const int64_t batch_size = qkv.size(0);
  const int64_t conv_dim = qkv.size(1);
  const int64_t num_value_heads = z.size(1);
  CHECK_GE(batch_size, 1);
  CHECK_LE(batch_size, 32);
  CHECK_EQ(z.size(0), batch_size);
  CHECK_EQ(z.size(2), kHeadDim);
  CHECK_EQ(b.sizes(), torch::IntArrayRef({batch_size, num_value_heads}));

  CHECK_EQ(qkv.scalar_type(), torch::kBFloat16);
  CHECK_EQ(z.scalar_type(), torch::kBFloat16);
  CHECK_EQ(b.scalar_type(), torch::kBFloat16);
  CHECK_EQ(a.scalar_type(), torch::kBFloat16);
  CHECK_EQ(conv_weight.scalar_type(), torch::kBFloat16);
  CHECK_EQ(conv_state.scalar_type(), torch::kBFloat16);
  CHECK_EQ(norm_weight.scalar_type(), torch::kBFloat16);
  CHECK_EQ(a_log.scalar_type(), torch::kFloat32);
  CHECK_EQ(dt_bias.scalar_type(), torch::kFloat32);
  CHECK_EQ(ssm_state.scalar_type(), torch::kFloat32);
  CHECK_EQ(read_state_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(write_state_indices.scalar_type(), torch::kInt32);

  CHECK_EQ(conv_weight.sizes(), torch::IntArrayRef({4, conv_dim}));
  CHECK_EQ(conv_state.dim(), 3);
  CHECK_EQ(conv_state.size(1), 3);
  CHECK_EQ(conv_state.size(2), conv_dim);
  const int64_t num_state_slots = conv_state.size(0);
  CHECK_GE(num_state_slots, 1);
  CHECK_LE(num_state_slots, 1024);
  CHECK_EQ(ssm_state.sizes(),
           torch::IntArrayRef(
               {num_state_slots, num_value_heads, kHeadDim, kHeadDim}));
  CHECK_EQ(a_log.sizes(), torch::IntArrayRef({num_value_heads}));
  CHECK_EQ(dt_bias.sizes(), torch::IntArrayRef({num_value_heads}));
  CHECK_EQ(norm_weight.sizes(), torch::IntArrayRef({kHeadDim}));
  for (const auto& indices : {read_state_indices, write_state_indices}) {
    CHECK_EQ(indices.sizes(), torch::IntArrayRef({batch_size}));
  }

  const auto device = qkv.device();
  for (const auto& tensor : {z,
                             b,
                             a,
                             conv_weight,
                             conv_state,
                             a_log,
                             dt_bias,
                             ssm_state,
                             read_state_indices,
                             write_state_indices,
                             norm_weight}) {
    CHECK_EQ(tensor.device(), device)
        << "all inputs must be on one NPU device.";
  }

  torch::Tensor conv_out = torch::empty_like(qkv);
  torch::Tensor out = torch::empty_like(z);
  EXEC_NPU_CMD(aclnnMegaGdnDecode,
               qkv,
               z,
               b,
               a,
               conv_weight,
               conv_state,
               a_log,
               dt_bias,
               ssm_state,
               read_state_indices,
               write_state_indices,
               norm_weight,
               conv_out,
               conv_state,
               ssm_state,
               out);
  return out;
}

}  // namespace xllm::kernel::npu
