/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

// Used by the standard BeamSearcher sampling path on NPU.
// This wrapper runs one regular beam-search update from the current beam
// scores and candidate token scores.
// Inputs:
//   logprobs: accumulated scores for the current live beams.
//   top_tokens: candidate token ids for this step.
//   top_logprobs: candidate token scores for this step.
// Outputs:
//   src_seq_idxes: source beam selected for each new beam.
//   out_logprobs: updated accumulated beam scores.
//   out_tokens: chosen next token ids for the next sampling step.
void beam_search(const torch::Tensor& logprobs,
                 const torch::Tensor& top_tokens,
                 const torch::Tensor& top_logprobs,
                 torch::Tensor& src_seq_idxes,
                 torch::Tensor& out_logprobs,
                 torch::Tensor& out_tokens) {
  check_tensor(logprobs, "logprobs", "beam_search");
  check_tensor(top_tokens, "top_tokens", "beam_search");
  check_tensor(top_logprobs, "top_logprobs", "beam_search");
  EXEC_NPU_CMD(aclnnBeamSearch,
               logprobs,
               top_tokens,
               top_logprobs,
               out_tokens,
               src_seq_idxes,
               out_logprobs);
}
}  // namespace xllm::kernel::npu
