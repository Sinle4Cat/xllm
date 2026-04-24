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

// Run one beam-search update in REC multi-round decoding on NPU.
// Inputs:
//   logprobs: accumulated beam scores from the previous round.
//   top_tokens/top_logprobs: current candidate tokens and their scores.
//   sequence_group: beam-to-sequence layout before this round.
//   current_step: decode round index passed to the NPU kernel.
// Outputs:
//   out_token_ids/out_token_index/out_log_probs: selected next tokens, source
//   beam indices, and updated accumulated scores.
//   out_beam_count_prefix_sums/out_sequence: per-request beam offsets and the
//   next sequence layout after this step.
void beam_search_rec(const torch::Tensor& logprobs,
                     const torch::Tensor& top_tokens,
                     const torch::Tensor& top_logprobs,
                     torch::Tensor& sequence_group,
                     int64_t current_step,
                     torch::Tensor& out_token_ids,
                     torch::Tensor& out_token_index,
                     torch::Tensor& out_log_probs,
                     torch::Tensor& out_beam_count_prefix_sums,
                     torch::Tensor& out_sequence) {
  check_tensor(logprobs, "logprobs", "beam_search_rec");
  check_tensor(top_tokens, "top_tokens", "beam_search_rec");
  check_tensor(top_logprobs, "top_logprobs", "beam_search_rec");
  check_tensor(sequence_group, "sequence_group", "beam_search_rec");
  EXEC_NPU_CMD(aclnnBeamSearchGroup,
               logprobs,
               top_tokens,
               top_logprobs,
               sequence_group,
               current_step,
               out_token_ids,
               out_token_index,
               out_log_probs,
               out_beam_count_prefix_sums,
               out_sequence);
}
}  // namespace xllm::kernel::npu
