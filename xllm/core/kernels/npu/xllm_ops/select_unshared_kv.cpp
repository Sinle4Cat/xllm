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

#include <torch/torch.h>

#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "third_party/torch_npu_ops/ascendc_npu/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

// Reorder per-layer unshared KV caches after beam selection in REC
// multi-round decoding.
// Inputs:
//   beam_index: source beam chosen for each output beam.
//   x_key_block/x_value_block: per-layer unshared K/V caches to update.
//   block_table/group_offset: request mapping and per-request beam offsets
//   expected by the NPU kernel.
//   decode_step/beam_size/layer_num: cache slot and layout metadata.
// Output:
//   x_key_block/x_value_block are updated in place to match the selected
//   beams for the next round.
void select_unshared_kv(const torch::Tensor& beam_index,
                        const std::vector<torch::Tensor>& x_key_block,
                        const std::vector<torch::Tensor>& x_value_block,
                        const torch::Tensor& block_table,
                        const torch::Tensor& group_offset,
                        int64_t decode_step,
                        int64_t beam_size,
                        int64_t layer_num) {
  check_tensor(beam_index, "beam_index", "select_unshared_kv");
  check_tensor(block_table, "block_table", "select_unshared_kv");
  check_tensor(group_offset, "group_offset", "select_unshared_kv");
  for (const auto& t : x_key_block) {
    check_tensor(t, "x_key_block[i]", "select_unshared_kv");
  }
  for (const auto& t : x_value_block) {
    check_tensor(t, "x_value_block[i]", "select_unshared_kv");
  }
  torch::TensorList x_key_block_list(x_key_block);
  torch::TensorList x_value_block_list(x_value_block);
  EXEC_NPU_CMD(aclnnSelectUnsharedKV,
               beam_index,
               block_table,
               x_key_block_list,
               x_value_block_list,
               group_offset,
               decode_step,
               beam_size,
               layer_num,
               x_key_block_list,
               x_value_block_list);
}
}  // namespace xllm::kernel::npu
