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

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/utils.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace xllm::kernel::npu {

namespace {
constexpr int64_t kChunkSize = 128;

struct MaskCache {
  torch::Tensor mask_lower;
  torch::Tensor mask_full;
  torch::Tensor minus_identity;
};

std::unordered_map<int32_t, MaskCache> g_mask_cache;
std::mutex g_mask_cache_mutex;

MaskCache get_or_create_masks(const torch::Device& device) {
  const int32_t device_index = static_cast<int32_t>(device.index());
  std::lock_guard<std::mutex> lock(g_mask_cache_mutex);
  auto it = g_mask_cache.find(device_index);
  if (it != g_mask_cache.end()) {
    return it->second;
  }

  MaskCache cache;
  cache.mask_lower = torch::tril(
      torch::ones({kChunkSize, kChunkSize},
                  torch::TensorOptions(device).dtype(torch::kFloat32)),
      /*diagonal=*/-1);
  cache.mask_full = torch::tril(
      torch::ones({kChunkSize, kChunkSize},
                  torch::TensorOptions(device).dtype(torch::kFloat32)));
  cache.minus_identity =
      torch::zeros({kChunkSize, kChunkSize},
                   torch::TensorOptions(device).dtype(torch::kBFloat16));
  cache.minus_identity.diagonal().fill_(-1);
  g_mask_cache.emplace(device_index, cache);
  return cache;
}
}  // namespace

torch::Tensor mega_gdn_prefill_op(const torch::Tensor& mixed_qkv,
                                  const torch::Tensor& b,
                                  const torch::Tensor& a,
                                  const torch::Tensor& z,
                                  const torch::Tensor& conv_weight,
                                  torch::Tensor& conv_state,
                                  const torch::Tensor& a_log,
                                  const torch::Tensor& dt_bias,
                                  const torch::Tensor& conv_state_read_indices,
                                  const torch::Tensor& conv_state_write_indices,
                                  const torch::Tensor& ssm_state_read_indices,
                                  const torch::Tensor& ssm_state_write_indices,
                                  torch::Tensor& ssm_cache,
                                  const torch::Tensor& cu_seqlens,
                                  const torch::Tensor& norm_weight,
                                  int64_t num_matrices) {
  check_tensor(mixed_qkv, "mixed_qkv", "mega_gdn_prefill_op");
  check_tensor(b, "b", "mega_gdn_prefill_op");
  check_tensor(a, "a", "mega_gdn_prefill_op");
  check_tensor(z, "z", "mega_gdn_prefill_op");
  check_tensor(conv_weight, "conv_weight", "mega_gdn_prefill_op");
  check_tensor(conv_state, "conv_state", "mega_gdn_prefill_op");
  check_tensor(a_log, "a_log", "mega_gdn_prefill_op");
  check_tensor(dt_bias, "dt_bias", "mega_gdn_prefill_op");
  check_tensor(conv_state_read_indices,
               "conv_state_read_indices",
               "mega_gdn_prefill_op");
  check_tensor(conv_state_write_indices,
               "conv_state_write_indices",
               "mega_gdn_prefill_op");
  check_tensor(
      ssm_state_read_indices, "ssm_state_read_indices", "mega_gdn_prefill_op");
  check_tensor(ssm_state_write_indices,
               "ssm_state_write_indices",
               "mega_gdn_prefill_op");
  check_tensor(ssm_cache, "ssm_cache", "mega_gdn_prefill_op");
  check_tensor(cu_seqlens, "cu_seqlens", "mega_gdn_prefill_op");
  check_tensor(norm_weight, "norm_weight", "mega_gdn_prefill_op");

  auto masks = get_or_create_masks(mixed_qkv.device());
  uint64_t ffts_addr = 0;
  if (!is_ascend950()) {
    uint32_t ffts_len = 0;
    const int32_t status = rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    CHECK_EQ(status, 0) << "rtGetC2cCtrlAddr failed for mega_gdn_prefill_op";
    CHECK_GT(ffts_len, 0)
        << "rtGetC2cCtrlAddr returned an empty FFTS control region";
  }
  CHECK_GT(num_matrices, 0) << "num_matrices must be positive";
  int64_t ffts_addr_arg = static_cast<int64_t>(ffts_addr);

  auto norm_output = torch::empty_like(z);
  if (std::getenv("XLLM_DEBUG_ZERO_MEGA_GDN_PREFILL_OUTPUT") != nullptr) {
    norm_output.zero_();
  }
  EXEC_NPU_CMD(aclnnMegaGdnPrefillOp,
               mixed_qkv,
               b,
               a,
               z,
               conv_weight,
               conv_state,
               a_log,
               dt_bias,
               conv_state_read_indices,
               conv_state_write_indices,
               ssm_state_read_indices,
               ssm_state_write_indices,
               ssm_cache,
               masks.mask_lower,
               masks.mask_full,
               masks.minus_identity,
               cu_seqlens,
               norm_weight,
               ffts_addr_arg,
               num_matrices,
               norm_output);
  return norm_output;
}

}  // namespace xllm::kernel::npu
