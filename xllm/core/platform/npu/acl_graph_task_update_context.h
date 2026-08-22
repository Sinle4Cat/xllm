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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace xllm::npu {

constexpr int64_t kCausalConv1dGraphPadSlotId = -1;
constexpr int64_t kCausalConv1dActivationSilu = 1;
constexpr int64_t kCausalConv1dRunModeForward = 0;
constexpr int64_t kCausalConv1dRunModeUpdate = 1;

enum class CausalConv1dGraphBranch {
  kDecode,
  kSpecVerify,
};

enum class PagedAttentionGraphBranch {
  kDecode,
  kExpandedDecode,
};

struct CausalConv1dGraphTask {
  torch::Tensor output;
  torch::Tensor x;
  torch::Tensor weight;
  torch::Tensor conv_state;
  std::optional<torch::Tensor> bias;
  int64_t activation_mode = kCausalConv1dActivationSilu;
  int64_t pad_slot_id = kCausalConv1dGraphPadSlotId;
  int64_t run_mode = kCausalConv1dRunModeUpdate;
  CausalConv1dGraphBranch branch = CausalConv1dGraphBranch::kDecode;
  c10_npu::NPUTaskGroupHandle handle{};
  std::shared_ptr<c10_npu::NPUEvent> event;
};

struct PagedAttentionGraphTask {
  torch::Tensor output;
  torch::Tensor softmax_lse;
  // A5 graph-task updates must keep their FIA workspace alive for the
  // lifetime of the captured task. Allocating a new workspace on every
  // update makes the graph retain one allocator block per decode step.
  torch::Tensor workspace;
  torch::Tensor query;
  torch::Tensor key_cache;
  torch::Tensor value_cache;
  torch::Tensor block_table;
  int64_t num_heads = 0;
  int64_t num_key_value_heads = 0;
  double scale = 1.0;
  int64_t block_size = 0;
  PagedAttentionGraphBranch branch = PagedAttentionGraphBranch::kDecode;
  c10_npu::NPUTaskGroupHandle handle{};
  std::shared_ptr<c10_npu::NPUEvent> event;
};

class AclGraphTaskUpdateContext final {
 public:
  void begin_capture() {
    capturing = true;
    causal_conv1d_tasks.clear();
    paged_attention_tasks.clear();
  }

  void end_capture() { capturing = false; }

  bool capturing = false;
  std::vector<CausalConv1dGraphTask> causal_conv1d_tasks;
  std::vector<PagedAttentionGraphTask> paged_attention_tasks;
};

}  // namespace xllm::npu
