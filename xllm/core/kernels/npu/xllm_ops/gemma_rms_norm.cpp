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

#include <vector>

#include "third_party/torch_npu_ops/ascendc_npu/ascendc_ops_api.h"
#include "third_party/torch_npu_ops/ascendc_npu/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace {

std::vector<int64_t> build_rstd_shape(const torch::Tensor& x,
                                      const torch::Tensor& gamma) {
  int64_t dim_x = x.dim();
  int64_t dim_gamma = gamma.dim();
  int64_t diff = dim_x - dim_gamma;
  std::vector<int64_t> new_shape;
  if (diff > 0) {
    new_shape.reserve(dim_x);
    torch::IntArrayRef x_sizes = x.sizes();
    for (int64_t i = 0; i < diff; ++i) {
      new_shape.push_back(x_sizes[i]);
    }
    for (int64_t i = 0; i < dim_gamma; ++i) {
      new_shape.push_back(1);
    }
    return new_shape;
  }
  new_shape.assign(dim_x, 1);
  return new_shape;
}

}  // namespace

namespace xllm::kernel::npu {

void gemma_rms_norm(const torch::Tensor& x,
                    const torch::Tensor& gamma,
                    double epsilon,
                    torch::Tensor& rstd_out,
                    torch::Tensor& y_out) {
  std::vector<int64_t> rstd_shape = build_rstd_shape(x, gamma);
  torch::Tensor rstd =
      torch::empty(rstd_shape, x.options().dtype(torch::kFloat));
  torch::Tensor y = torch::empty(x.sizes(), x.options());
  EXEC_NPU_CMD(aclnnGemmaRmsNorm, x, gamma, epsilon, y, rstd);
  y_out = y;
  rstd_out = rstd;
}

}  // namespace xllm::kernel::npu
