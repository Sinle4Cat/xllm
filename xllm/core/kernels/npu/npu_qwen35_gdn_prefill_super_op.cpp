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

#include <mutex>
#include <unordered_map>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/utils.h"

namespace xllm::kernel::npu {

namespace {
constexpr int64_t kChunkSize = 128;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kValueHeads = 24;

struct PrefillMaskCache {
  torch::Tensor mask_lower;
  torch::Tensor mask_full;
  torch::Tensor minus_identity;
};

std::unordered_map<int32_t, PrefillMaskCache> g_prefill_mask_cache;
std::mutex g_prefill_mask_cache_mutex;

PrefillMaskCache get_or_create_prefill_masks(const torch::Device& device) {
  const int32_t device_index = static_cast<int32_t>(device.index());
  std::lock_guard<std::mutex> lock(g_prefill_mask_cache_mutex);
  auto it = g_prefill_mask_cache.find(device_index);
  if (it != g_prefill_mask_cache.end()) {
    return it->second;
  }

  PrefillMaskCache cache;
  cache.mask_lower = torch::tril(
      torch::ones({kChunkSize, kChunkSize},
                  torch::TensorOptions(device).dtype(torch::kFloat32)),
      /*diagonal=*/-1);
  cache.mask_full = torch::tril(
      torch::ones({kChunkSize, kChunkSize},
                  torch::TensorOptions(device).dtype(torch::kFloat32)),
      /*diagonal=*/0);
  cache.minus_identity =
      torch::zeros({kChunkSize, kChunkSize},
                   torch::TensorOptions(device).dtype(torch::kFloat16));
  cache.minus_identity.diagonal().fill_(-1);
  g_prefill_mask_cache.emplace(device_index, cache);
  return cache;
}
}  // namespace

torch::Tensor qwen35_gdn_prefill_super_op(const torch::Tensor& mixed_qkv,
                                          const torch::Tensor& z,
                                          const torch::Tensor& b,
                                          const torch::Tensor& a,
                                          const torch::Tensor& conv_weight,
                                          torch::Tensor& conv_state,
                                          const torch::Tensor& a_log,
                                          const torch::Tensor& dt_bias,
                                          torch::Tensor& ssm_state,
                                          const torch::Tensor& norm_weight,
                                          const torch::Tensor& cu_seqlens,
                                          int64_t conv_state_index,
                                          int64_t ssm_state_index) {
  check_tensor(mixed_qkv, "mixed_qkv", "qwen35_gdn_prefill_super_op");
  check_tensor(z, "z", "qwen35_gdn_prefill_super_op");
  check_tensor(conv_state, "conv_state", "qwen35_gdn_prefill_super_op");
  check_tensor(ssm_state, "ssm_state", "qwen35_gdn_prefill_super_op");

  const int64_t total_tokens = mixed_qkv.size(0);
  const int64_t num_chunks = (total_tokens + kChunkSize - 1) / kChunkSize;
  const int64_t num_matrices = num_chunks * kValueHeads;
  auto masks = get_or_create_prefill_masks(mixed_qkv.device());
  auto cu_seqlens_int32 =
      cu_seqlens.scalar_type() == torch::kInt32 && cu_seqlens.is_contiguous()
          ? cu_seqlens
          : cu_seqlens.to(torch::kInt32).contiguous();

  auto opts_fp16 =
      torch::TensorOptions(mixed_qkv.device()).dtype(torch::kFloat16);
  auto opts_fp32 =
      torch::TensorOptions(mixed_qkv.device()).dtype(torch::kFloat32);
  auto opts_bf16 =
      torch::TensorOptions(mixed_qkv.device()).dtype(torch::kBFloat16);

  auto packed_qkv = torch::empty({total_tokens, 5120}, opts_fp16);
  auto g = torch::empty({1, total_tokens, kValueHeads}, opts_fp32);
  auto beta = torch::empty({1, total_tokens, kValueHeads}, opts_fp16);
  auto initial_state =
      torch::empty({1, kValueHeads, kHeadDim, kHeadDim}, opts_fp16);
  auto mega_out =
      torch::empty({1, total_tokens, kValueHeads, kHeadDim}, opts_fp16);
  auto g_sum = torch::empty({1, total_tokens, kValueHeads}, opts_fp32);
  auto g_t = torch::empty({kValueHeads, total_tokens}, opts_fp32);
  auto beta_t = torch::empty({kValueHeads, total_tokens}, opts_fp16);
  auto mega_a =
      torch::empty({1, total_tokens, kValueHeads, kChunkSize}, opts_fp16);
  auto a_inv_f32 =
      torch::empty({1, total_tokens, kValueHeads, kChunkSize}, opts_fp32);
  auto a_inv =
      torch::empty({1, total_tokens, kValueHeads, kChunkSize}, opts_fp16);
  auto w = torch::empty({1, total_tokens, kValueHeads, kHeadDim}, opts_fp16);
  auto u = torch::empty_like(w);
  auto h = torch::empty({num_matrices, kHeadDim, kHeadDim}, opts_fp16);
  auto v_new = torch::empty_like(w);
  auto final_state = torch::empty({kValueHeads, kHeadDim, kHeadDim}, opts_fp16);
  auto out = torch::empty({total_tokens, kValueHeads, kHeadDim}, opts_bf16);

  EXEC_NPU_CMD(aclnnQwen35GdnPrefillSuperOp,
               mixed_qkv,
               z,
               b,
               a,
               conv_weight,
               conv_state,
               a_log,
               dt_bias,
               ssm_state,
               norm_weight,
               masks.mask_lower,
               masks.mask_full,
               masks.minus_identity,
               cu_seqlens_int32,
               num_matrices,
               conv_state_index,
               ssm_state_index,
               packed_qkv,
               g,
               beta,
               initial_state,
               mega_out,
               g_sum,
               g_t,
               beta_t,
               mega_a,
               a_inv_f32,
               a_inv,
               w,
               u,
               h,
               v_new,
               final_state,
               conv_state,
               ssm_state,
               out);
  return out;
}

}  // namespace xllm::kernel::npu
