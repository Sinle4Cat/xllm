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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/utils.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kMaxKvStackLen = 512;
constexpr int64_t kMaxQRowsPerTask = 128;
constexpr int64_t kMaxXfaInferCores = 25;
constexpr int64_t kDefaultCubeCoreCount = 20;

struct XfaInferCoreNode {
  uint32_t start_b_idx = 0;
  uint32_t start_n1_idx = 0;
  uint32_t start_s2_idx = 0;
  uint32_t end_b_idx = 0;
  uint32_t end_n1_idx = 0;
  uint32_t end_s2_idx = 0;
  uint64_t first_split_kv_task_lse_offset = 0;
  uint64_t first_split_kv_task_o_offset = 0;
};

struct XfaInferSplitNode {
  uint32_t batch_idx = 0;
  uint32_t head_start_idx = 0;
  uint32_t head_end_idx = 0;
  uint32_t q_start_idx = 0;
  uint32_t q_end_idx = 0;
  uint32_t split_num = 0;
  uint64_t lse_task_offset = 0;
  uint64_t o_task_offset = 0;
};

struct XfaInferExtraInfo {
  XfaInferCoreNode core_info[kMaxXfaInferCores];
  XfaInferSplitNode split_info[kMaxXfaInferCores];
  uint32_t total_split_node_num = 0;
};

int64_t ceil_div(int64_t value, int64_t divisor) {
  CHECK_GT(divisor, 0);
  return (value + divisor - 1) / divisor;
}

int64_t get_cube_core_count() {
  const char* value = std::getenv("XLLM_X_FLASH_ATTENTION_INFER_CUBE_CORE_NUM");
  if (value == nullptr) {
    return kDefaultCubeCoreCount;
  }

  char* end = nullptr;
  const int64_t parsed = std::strtoll(value, &end, 10);
  if (end == value || parsed <= 0 || parsed > kMaxXfaInferCores) {
    LOG(WARNING) << "Ignore invalid XLLM_X_FLASH_ATTENTION_INFER_CUBE_CORE_NUM="
                 << value << ", use default " << kDefaultCubeCoreCount;
    return kDefaultCubeCoreCount;
  }
  return parsed;
}

void validate_extra_tiling_inputs(const std::vector<int64_t>& actual_q_lens,
                                  const std::vector<int64_t>& actual_kv_lens,
                                  int64_t num_heads,
                                  int64_t num_key_value_heads,
                                  int64_t block_size,
                                  bool use_fd) {
  CHECK(!actual_q_lens.empty()) << "actual_q_lens must not be empty";
  CHECK_EQ(actual_q_lens.size(), actual_kv_lens.size())
      << "actual_q_lens and actual_kv_lens size mismatch";
  CHECK_GT(num_heads, 0) << "num_heads must be positive";
  CHECK_GT(num_key_value_heads, 0) << "num_key_value_heads must be positive";
  CHECK_EQ(num_heads % num_key_value_heads, 0)
      << "num_heads must be divisible by num_key_value_heads";
  CHECK_GT(block_size, 0) << "block_size must be positive";
  CHECK_EQ(kMaxKvStackLen % block_size, 0)
      << "block_size must divide " << kMaxKvStackLen;

  const int64_t group_size = num_heads / num_key_value_heads;
  int64_t uniform_q_len = -1;
  int64_t previous_q_len = 0;
  for (int64_t batch_idx = 0;
       batch_idx < static_cast<int64_t>(actual_q_lens.size());
       ++batch_idx) {
    const int64_t cumulative_q_len = actual_q_lens[batch_idx];
    const int64_t q_len = cumulative_q_len - previous_q_len;
    previous_q_len = cumulative_q_len;
    CHECK_GT(q_len, 0) << "q length must be positive";
    if (use_fd) {
      CHECK_LE(q_len * group_size, kMaxQRowsPerTask)
          << "x_flash_attention_infer FD path requires "
          << "q_len * group_size <= " << kMaxQRowsPerTask;
    } else if (uniform_q_len < 0) {
      uniform_q_len = q_len;
    } else {
      CHECK_EQ(q_len, uniform_q_len)
          << "x_flash_attention_infer noFD path currently requires "
          << "uniform q lengths across the chunked-prefill batch";
    }
    CHECK_GT(actual_kv_lens[batch_idx], 0) << "kv length must be positive";
  }
}

torch::Tensor build_placeholder_extra_tiling(const torch::Tensor& reference) {
  return torch::zeros({1}, reference.options().dtype(torch::kInt32));
}

void fill_inactive_core_info(XfaInferExtraInfo* extra_info) {
  const uint32_t inactive = std::numeric_limits<uint32_t>::max();
  for (int64_t core_idx = 0; core_idx < kMaxXfaInferCores; ++core_idx) {
    extra_info->core_info[core_idx].start_b_idx = inactive;
  }
}

c10::optional<torch::Tensor> to_c10_optional_tensor(
    const std::optional<torch::Tensor>& tensor_opt) {
  if (tensor_opt.has_value() && tensor_opt.value().defined()) {
    return tensor_opt.value();
  }
  return c10::nullopt;
}

}  // namespace

torch::Tensor build_x_flash_attention_infer_extra_tiling(
    const std::vector<int64_t>& actual_q_lens,
    const std::vector<int64_t>& actual_kv_lens,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t block_size,
    bool use_fd,
    const torch::Tensor& reference) {
  validate_extra_tiling_inputs(actual_q_lens,
                               actual_kv_lens,
                               num_heads,
                               num_key_value_heads,
                               block_size,
                               use_fd);

  if (!use_fd) {
    return build_placeholder_extra_tiling(reference);
  }

  const int64_t batch_size = static_cast<int64_t>(actual_q_lens.size());
  const int64_t task_count = batch_size * num_key_value_heads;
  const int64_t cube_core_count = get_cube_core_count();
  const int64_t active_core_count = std::min(task_count, cube_core_count);

  XfaInferExtraInfo extra_info{};
  fill_inactive_core_info(&extra_info);
  extra_info.total_split_node_num = 0;

  for (int64_t core_idx = 0; core_idx < active_core_count; ++core_idx) {
    const int64_t start_task = core_idx * task_count / active_core_count;
    const int64_t end_task =
        ((core_idx + 1) * task_count / active_core_count) - 1;
    const int64_t start_batch_idx = start_task / num_key_value_heads;
    const int64_t end_batch_idx = end_task / num_key_value_heads;
    const int64_t end_kv_len = actual_kv_lens[end_batch_idx];
    const int64_t end_s2_idx = ceil_div(end_kv_len, kMaxKvStackLen);

    XfaInferCoreNode& core_info = extra_info.core_info[core_idx];
    core_info.start_b_idx = static_cast<uint32_t>(start_batch_idx);
    core_info.start_n1_idx =
        static_cast<uint32_t>(start_task % num_key_value_heads);
    core_info.start_s2_idx = 0;
    core_info.end_b_idx = static_cast<uint32_t>(end_batch_idx);
    core_info.end_n1_idx =
        static_cast<uint32_t>(end_task % num_key_value_heads);
    core_info.end_s2_idx = static_cast<uint32_t>(end_s2_idx);
    core_info.first_split_kv_task_lse_offset = 0;
    core_info.first_split_kv_task_o_offset = 0;
  }

  std::vector<int32_t> extra_words(
      (static_cast<int64_t>(sizeof(XfaInferExtraInfo)) +
       static_cast<int64_t>(sizeof(int32_t)) - 1) /
          static_cast<int64_t>(sizeof(int32_t)),
      0);
  std::memcpy(extra_words.data(), &extra_info, sizeof(extra_info));

  torch::Tensor cpu_extra_tiling =
      torch::from_blob(extra_words.data(),
                       {static_cast<int64_t>(extra_words.size())},
                       torch::TensorOptions().dtype(torch::kInt32))
          .clone();
  return cpu_extra_tiling.to(reference.device(), torch::kInt32, true, true);
}

torch::Tensor x_flash_attention_infer(const torch::Tensor& query,
                                      const torch::Tensor& key_cache,
                                      const torch::Tensor& value_cache,
                                      const std::optional<torch::Tensor>& mask,
                                      const torch::Tensor& block_table,
                                      const torch::Tensor& actual_q_lens,
                                      const torch::Tensor& actual_kv_lens,
                                      const torch::Tensor& extra_tiling,
                                      int64_t num_heads,
                                      int64_t num_key_value_heads,
                                      double scale,
                                      const std::string& layout) {
  check_tensor(query, "query", "x_flash_attention_infer");
  check_tensor(key_cache, "key_cache", "x_flash_attention_infer");
  check_tensor(value_cache, "value_cache", "x_flash_attention_infer");
  check_tensor(block_table, "block_table", "x_flash_attention_infer");
  check_tensor(actual_q_lens, "actual_q_lens", "x_flash_attention_infer");
  check_tensor(actual_kv_lens, "actual_kv_lens", "x_flash_attention_infer");
  check_tensor(extra_tiling, "extra_tiling", "x_flash_attention_infer");
  CHECK(mask.has_value() && mask.value().defined())
      << "x_flash_attention_infer requires causal mask because the current "
         "kernel only dispatches causal-mask tiling keys";
  CHECK(query.dtype() == torch::kFloat16 || query.dtype() == torch::kBFloat16)
      << "query must be FLOAT16 or BFLOAT16";
  CHECK_EQ(key_cache.dtype(), query.dtype())
      << "key_cache dtype must match query dtype";
  CHECK_EQ(value_cache.dtype(), query.dtype())
      << "value_cache dtype must match query dtype";
  CHECK_EQ(block_table.dtype(), torch::kInt32) << "block_table must be INT32";
  CHECK_EQ(actual_q_lens.dtype(), torch::kInt32)
      << "actual_q_lens must be INT32";
  CHECK_EQ(actual_kv_lens.dtype(), torch::kInt32)
      << "actual_kv_lens must be INT32";
  CHECK_EQ(extra_tiling.dtype(), torch::kInt32) << "extra_tiling must be INT32";

  torch::Tensor output = torch::empty_like(query);
  std::string layout_arg = layout;
  char* layout_ptr = const_cast<char*>(layout_arg.c_str());
  c10::optional<torch::Tensor> mask_tensor = to_c10_optional_tensor(mask);

  EXEC_NPU_CMD(aclnnXFlashAttentionInfer,
               query,
               key_cache,
               value_cache,
               mask_tensor,
               block_table,
               actual_q_lens,
               actual_kv_lens,
               extra_tiling,
               layout_ptr,
               num_heads,
               num_key_value_heads,
               scale,
               output);

  return output;
}

}  // namespace xllm::kernel::npu
