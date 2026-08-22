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

#include <glog/logging.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/utils.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kHeadDim = 128;

void check_draft_tensor(const torch::Tensor& tensor,
                        torch::ScalarType dtype,
                        const torch::Device& device,
                        const char* name) {
  check_tensor(tensor, name, "mega_gdn_draft_decode");
  CHECK_EQ(tensor.scalar_type(), dtype)
      << "mega_gdn_draft_decode: " << name << " has an invalid dtype";
  CHECK_EQ(tensor.device().type(), c10::DeviceType::PrivateUse1)
      << "mega_gdn_draft_decode: " << name << " must be on NPU";
  CHECK_EQ(tensor.device(), device)
      << "mega_gdn_draft_decode: all tensors must share an NPU device";
  CHECK(tensor.is_contiguous())
      << "mega_gdn_draft_decode: " << name << " must be contiguous";
}

}  // namespace

torch::Tensor mega_gdn_draft_decode(const torch::Tensor& qkv,
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
                                    const torch::Tensor& norm_weight,
                                    bool fla_ssm_state_layout) {
  CHECK(qkv.defined()) << "mega_gdn_draft_decode: qkv is not defined";
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

  CHECK_EQ(qkv.dim(), 2);
  CHECK_EQ(z.dim(), 3);
  CHECK_EQ(a.dim(), 2);
  CHECK_EQ(b.dim(), 2);
  const int64_t total_tokens = qkv.size(0);
  const int64_t conv_dim = qkv.size(1);
  const int64_t batch_size = read_state_indices.numel();
  CHECK_GE(total_tokens, 1);
  CHECK_GE(batch_size, 1);
  CHECK_LE(batch_size, 32);
  CHECK_LE(total_tokens, 2 * batch_size);
  CHECK_EQ(z.size(0), total_tokens);
  CHECK_EQ(z.size(2), kHeadDim);
  const int64_t num_v_heads = z.size(1);
  const int64_t qk_width = conv_dim - num_v_heads * kHeadDim;
  CHECK_GT(qk_width, 0);
  CHECK_EQ(qk_width % (2 * kHeadDim), 0);
  const int64_t num_k_heads = qk_width / (2 * kHeadDim);
  CHECK_GE(num_k_heads, 1);
  CHECK_LE(num_k_heads, 16);
  CHECK_EQ(num_k_heads & (num_k_heads - 1), 0);
  CHECK_EQ(num_v_heads % num_k_heads, 0);
  CHECK_GE(num_v_heads / num_k_heads, 1);
  CHECK_LE(num_v_heads / num_k_heads, 4);

  CHECK(a.sizes() == torch::IntArrayRef({total_tokens, num_v_heads}));
  CHECK_EQ(a.sizes(), b.sizes());
  CHECK(conv_weight.sizes() == torch::IntArrayRef({4, conv_dim}));
  CHECK_EQ(conv_state.dim(), 3);
  const int64_t num_state_slots = conv_state.size(0);
  CHECK(conv_state.sizes() ==
        torch::IntArrayRef({num_state_slots, 3, conv_dim}));
  CHECK(a_log.sizes() == torch::IntArrayRef({num_v_heads}));
  CHECK(dt_bias.sizes() == torch::IntArrayRef({num_v_heads}));
  CHECK(ssm_state.sizes() ==
        torch::IntArrayRef({num_state_slots, num_v_heads, kHeadDim, kHeadDim}));
  CHECK(read_state_indices.sizes() == torch::IntArrayRef({batch_size}));
  CHECK(write_state_indices.sizes() == torch::IntArrayRef({batch_size}));
  CHECK(q_cu_seq_lens.sizes() == torch::IntArrayRef({batch_size + 1}));
  CHECK(state_validity_mask.sizes() == torch::IntArrayRef({batch_size}));
  CHECK(norm_weight.sizes() == torch::IntArrayRef({kHeadDim}));

  auto conv_out = torch::empty_like(qkv);
  auto out = torch::empty_like(z);
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
