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

// Used by schedule overlap on NPU.
// This wrapper prepares the next decode-step input under the ACLNN update
// rule.
// Inputs:
//   src: sampled tokens from the previous step.
// Outputs:
//   dst: current-step input token tensor, updated in place after replacement.
void replace_token(torch::Tensor& dst, torch::Tensor& src) {
  check_tensor(dst, "dst", "replace_token");
  check_tensor(src, "src", "replace_token");
  EXEC_NPU_CMD(aclnnReplaceToken, dst, src, dst);
}
}  // namespace xllm::kernel::npu
