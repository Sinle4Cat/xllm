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

// Used by the sampling logits preprocessing path on NPU.
// This wrapper applies top-k and top-p filtering before token sampling so the
// downstream sampler only sees the kept candidates.
// Inputs:
//   topK: top-k threshold tensor for this sampling step.
//   topP: top-p threshold tensor for this sampling step.
// Outputs:
//   logits: logits tensor filtered in place and consumed by the sampler.
void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP) {
  check_tensor(logits, "logits", "top_k_top_p");
  check_tensor(topK, "topK", "top_k_top_p");
  check_tensor(topP, "topP", "top_k_top_p");
  EXEC_NPU_CMD(aclnnApplyTopKTopP, logits, topP, topK, logits);
}
}  // namespace xllm::kernel::npu
