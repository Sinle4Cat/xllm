/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

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

void check_draft_tensor(const torch::Tensor& tensor,
                        torch::ScalarType dtype,
                        const torch::Device& device,
                        const char* name) {
  CHECK(tensor.defined()) << name << " must be defined.";
  CHECK_EQ(tensor.scalar_type(), dtype) << name << " has an invalid dtype.";
  CHECK(tensor.device().is_privateuseone())
      << name << " must be an NPU tensor.";
  CHECK_EQ(tensor.device(), device) << "all inputs must be on one NPU device.";
  CHECK(tensor.is_contiguous()) << name << " must be contiguous.";
}

}  // namespace

torch::Tensor npu_mega_gdn_draft_decode(
    const torch::Tensor& qkv,
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
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& state_validity_mask,
    const torch::Tensor& norm_weight) {
  CHECK(qkv.defined()) << "qkv must be defined.";
  const torch::Device device = qkv.device();
  check_draft_tensor(qkv, torch::kBFloat16, device, "qkv");
  check_draft_tensor(z, torch::kBFloat16, device, "z");
  check_draft_tensor(b, torch::kBFloat16, device, "b");
  check_draft_tensor(a, torch::kBFloat16, device, "a");
  check_draft_tensor(conv_weight, torch::kBFloat16, device, "conv_weight");
  check_draft_tensor(conv_state, torch::kBFloat16, device, "conv_state");
  check_draft_tensor(a_log, torch::kFloat32, device, "a_log");
  check_draft_tensor(dt_bias, torch::kFloat32, device, "dt_bias");
  check_draft_tensor(ssm_state, torch::kFloat32, device, "ssm_state");
  check_draft_tensor(
      read_state_indices, torch::kInt32, device, "read_state_indices");
  check_draft_tensor(
      write_state_indices, torch::kInt32, device, "write_state_indices");
  check_draft_tensor(q_cu_seq_lens, torch::kInt32, device, "q_cu_seq_lens");
  check_draft_tensor(
      state_validity_mask, torch::kBool, device, "state_validity_mask");
  check_draft_tensor(norm_weight, torch::kBFloat16, device, "norm_weight");

  CHECK_EQ(qkv.dim(), 2) << "qkv must be [tokens, channels].";
  CHECK_EQ(z.dim(), 3) << "z must be [tokens, value_heads, 128].";
  CHECK_EQ(a.dim(), 2) << "a must be [tokens, value_heads].";
  CHECK_EQ(b.dim(), 2) << "b must be [tokens, value_heads].";
  const int64_t total_tokens = qkv.size(0);
  const int64_t conv_dim = qkv.size(1);
  const int64_t batch_size = read_state_indices.numel();
  const int64_t num_value_heads = z.size(1);
  CHECK_GE(total_tokens, 1);
  CHECK_GE(batch_size, 1);
  CHECK_LE(batch_size, 32);
  CHECK_LE(total_tokens, 2 * batch_size);
  CHECK_EQ(z.sizes(),
           torch::IntArrayRef({total_tokens, num_value_heads, kHeadDim}));
  CHECK_EQ(a.sizes(), torch::IntArrayRef({total_tokens, num_value_heads}));
  CHECK_EQ(b.sizes(), a.sizes());
  const int64_t qk_width = conv_dim - num_value_heads * kHeadDim;
  CHECK_GT(qk_width, 0);
  CHECK_EQ(qk_width % (2 * kHeadDim), 0);
  const int64_t num_key_heads = qk_width / (2 * kHeadDim);
  CHECK_GE(num_key_heads, 1);
  CHECK_LE(num_key_heads, 16);
  CHECK_EQ(num_key_heads & (num_key_heads - 1), 0);
  CHECK_EQ(num_value_heads % num_key_heads, 0);
  CHECK_GE(num_value_heads / num_key_heads, 1);
  CHECK_LE(num_value_heads / num_key_heads, 4);

  CHECK_EQ(conv_weight.sizes(), torch::IntArrayRef({4, conv_dim}));
  CHECK_EQ(conv_state.dim(), 3);
  CHECK_EQ(conv_state.size(1), 3);
  CHECK_EQ(conv_state.size(2), conv_dim);
  const int64_t num_state_slots = conv_state.size(0);
  CHECK_GE(num_state_slots, 1);
  CHECK_EQ(ssm_state.sizes(),
           torch::IntArrayRef(
               {num_state_slots, num_value_heads, kHeadDim, kHeadDim}));
  CHECK_EQ(a_log.sizes(), torch::IntArrayRef({num_value_heads}));
  CHECK_EQ(dt_bias.sizes(), torch::IntArrayRef({num_value_heads}));
  CHECK_EQ(read_state_indices.sizes(), torch::IntArrayRef({batch_size}));
  CHECK_EQ(write_state_indices.sizes(), torch::IntArrayRef({batch_size}));
  CHECK_EQ(q_cu_seq_lens.sizes(), torch::IntArrayRef({batch_size + 1}));
  CHECK_EQ(state_validity_mask.sizes(), torch::IntArrayRef({batch_size}));
  CHECK_EQ(norm_weight.sizes(), torch::IntArrayRef({kHeadDim}));

  torch::Tensor conv_out = torch::empty_like(qkv);
  torch::Tensor out = torch::empty_like(z);
  const bool fla_ssm_state_layout = true;
  EXEC_NPU_CMD(aclnnMegaGdnDraftDecode,
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
               q_cu_seq_lens,
               state_validity_mask,
               norm_weight,
               fla_ssm_state_layout,
               conv_out,
               conv_state,
               ssm_state,
               out);
  return out;
}

}  // namespace xllm::kernel::npu
