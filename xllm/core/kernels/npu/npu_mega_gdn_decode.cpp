/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/utils.h"

namespace xllm::kernel::npu {

torch::Tensor mega_gdn_decode(const torch::Tensor& qkv,
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
                              const torch::Tensor& norm_weight,
                              bool fla_ssm_state_layout) {
  check_tensor(qkv, "qkv", "mega_gdn_decode");
  check_tensor(z, "z", "mega_gdn_decode");
  check_tensor(conv_state, "conv_state", "mega_gdn_decode");
  check_tensor(ssm_state, "ssm_state", "mega_gdn_decode");
  check_tensor(read_state_indices, "read_state_indices", "mega_gdn_decode");
  check_tensor(write_state_indices, "write_state_indices", "mega_gdn_decode");

  auto conv_out = torch::empty_like(qkv);
  auto out = torch::empty_like(z);
  EXEC_NPU_CMD(aclnnMegaGdnDecode,
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
               norm_weight,
               fla_ssm_state_layout,
               conv_out,
               conv_state,
               ssm_state,
               out);
  return out;
}

}  // namespace xllm::kernel::npu
