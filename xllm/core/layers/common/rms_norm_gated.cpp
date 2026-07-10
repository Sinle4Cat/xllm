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

#include "rms_norm_gated.h"

#include <glog/logging.h>

#include <cstdlib>

#include "framework/state_dict/utils.h"
#include "xllm/core/kernels/ops_api.h"
#if defined(USE_NPU)
#include "xllm/core/kernels/npu/tilelang/tilelang_ops_api.h"
#endif

namespace xllm {
namespace layer {

RmsNormGatedImpl::RmsNormGatedImpl(int64_t dim,
                                   double eps,
                                   const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_parameter(
      "weight", torch::empty({dim}, options), /*requires_grad=*/false);
}

torch::Tensor RmsNormGatedImpl::forward(torch::Tensor& input,
                                        std::optional<torch::Tensor> gate) {
  xllm::kernel::GatedLayerNormParams params;
  params.x = input;
  params.weight = weight_;
  torch::Tensor bias;
  params.bias = bias;
  params.eps = eps_;
  if (gate.has_value()) {
    params.z = gate;
  }
  params.group_size = input.size(-1);
  params.is_rms_norm = true;
  auto ret = xllm::kernel::gated_layer_norm(params);
  return ret;
}

bool RmsNormGatedImpl::supports_fused_scale_gated_rmsnorm(
    const torch::Tensor& gate,
    int64_t head_size) const {
#if defined(USE_NPU)
  static const bool disabled =
      std::getenv("XLLM_DISABLE_FUSED_SCALE_GATED_RMSNORM") != nullptr;
  return !disabled && gate.defined() && weight_.defined() && gate.dim() > 0 &&
         gate.size(-1) == head_size && weight_.numel() == head_size &&
         gate.device() == weight_.device() && gate.is_contiguous() &&
         weight_.is_contiguous() &&
         xllm::kernel::npu::tilelang::
             has_fused_scale_gated_rmsnorm_specialization(
                 head_size,
                 torch::kFloat16,
                 gate.scalar_type(),
                 weight_.scalar_type());
#else
  return false;
#endif
}

torch::Tensor RmsNormGatedImpl::forward_scaled(torch::Tensor& input,
                                               torch::Tensor& gate,
                                               float scale) {
#if defined(USE_NPU)
  if (input.dim() == 2 && input.is_contiguous() &&
      input.size(-1) == norm_dim_ && input.sizes() == gate.sizes() &&
      supports_fused_scale_gated_rmsnorm(gate, norm_dim_)) {
    return xllm::kernel::npu::tilelang::fused_scale_gated_rmsnorm(
        input, gate, weight_, static_cast<float>(eps_), scale);
  }
#endif
  auto scaled = (input * scale).to(weight_.scalar_type());
  return forward(scaled, gate);
}

void RmsNormGatedImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
}

}  // namespace layer
}  // namespace xllm
