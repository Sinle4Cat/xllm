/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_FUSED_SCALE_GATED_RMSNORM_REGISTRY_INC
#error "XLLM_TL_FUSED_SCALE_GATED_RMSNORM_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kCompileMaxRows = 262144;
constexpr int64_t kSupportedHeadSize = 128;

#include XLLM_TL_FUSED_SCALE_GATED_RMSNORM_REGISTRY_INC

FusedScaleGatedRmsnormSpecialization build_runtime_specialization(
    int64_t head_size,
    torch::ScalarType output_dtype) {
  CHECK_LE(head_size, std::numeric_limits<int32_t>::max())
      << "TileLang fused_scale_gated_rmsnorm: head_size exceeds int32";
  return make_fused_scale_gated_rmsnorm_specialization(
      FusedScaleGatedRmsnormHeadSize{static_cast<int32_t>(head_size)},
      FusedScaleGatedRmsnormDType{to_tilelang_dtype(output_dtype)});
}

void check_supported(const torch::Tensor& x,
                     const torch::Tensor& gate,
                     const torch::Tensor& weight,
                     float eps,
                     float scale) {
  CHECK(x.defined()) << "TileLang fused_scale_gated_rmsnorm: x must be defined";
  CHECK(gate.defined())
      << "TileLang fused_scale_gated_rmsnorm: gate must be defined";
  CHECK(weight.defined())
      << "TileLang fused_scale_gated_rmsnorm: weight must be defined";
  CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
        gate.device().type() == c10::DeviceType::PrivateUse1 &&
        weight.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang fused_scale_gated_rmsnorm: all tensors must be on NPU";
  CHECK_EQ(x.device(), gate.device())
      << "TileLang fused_scale_gated_rmsnorm: x/gate device mismatch";
  CHECK_EQ(x.device(), weight.device())
      << "TileLang fused_scale_gated_rmsnorm: x/weight device mismatch";

  CHECK_EQ(x.dim(), 2)
      << "TileLang fused_scale_gated_rmsnorm: x must be 2D [M, 128]";
  CHECK_EQ(gate.dim(), 2)
      << "TileLang fused_scale_gated_rmsnorm: gate must be 2D [M, 128]";
  CHECK_EQ(weight.dim(), 1)
      << "TileLang fused_scale_gated_rmsnorm: weight must be 1D [128]";
  CHECK_EQ(x.sizes(), gate.sizes())
      << "TileLang fused_scale_gated_rmsnorm: x/gate shape mismatch";
  CHECK_EQ(x.size(1), kSupportedHeadSize)
      << "TileLang fused_scale_gated_rmsnorm: only head_size=128 is supported";
  CHECK_EQ(weight.size(0), x.size(1))
      << "TileLang fused_scale_gated_rmsnorm: weight size mismatch";

  CHECK_EQ(x.dtype(), torch::kFloat16)
      << "TileLang fused_scale_gated_rmsnorm: x must be float16";
  CHECK_EQ(gate.dtype(), torch::kBFloat16)
      << "TileLang fused_scale_gated_rmsnorm: gate must be bfloat16";
  CHECK_EQ(weight.dtype(), torch::kBFloat16)
      << "TileLang fused_scale_gated_rmsnorm: weight must be bfloat16";
  CHECK(x.is_contiguous())
      << "TileLang fused_scale_gated_rmsnorm: x must be contiguous";
  CHECK(gate.is_contiguous())
      << "TileLang fused_scale_gated_rmsnorm: gate must be contiguous";
  CHECK(weight.is_contiguous())
      << "TileLang fused_scale_gated_rmsnorm: weight must be contiguous";
  CHECK_GT(x.size(0), 0)
      << "TileLang fused_scale_gated_rmsnorm: num_rows must be > 0";
  CHECK(std::isfinite(eps) && eps >= 0.0F)
      << "TileLang fused_scale_gated_rmsnorm: eps must be finite and >= 0";
  CHECK(std::isfinite(scale))
      << "TileLang fused_scale_gated_rmsnorm: scale must be finite";
}

void run_chunk(const torch::Tensor& x,
               const torch::Tensor& gate,
               const torch::Tensor& weight,
               torch::Tensor& output,
               float eps,
               float scale) {
  auto specialization =
      build_runtime_specialization(x.size(1), gate.scalar_type());
  const auto* entry =
      find_fused_scale_gated_rmsnorm_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang fused_scale_gated_rmsnorm: no compiled variant. Available "
         "variants: "
      << available_fused_scale_gated_rmsnorm_variant_keys();
  CHECK_LE(x.size(0), kCompileMaxRows)
      << "TileLang fused_scale_gated_rmsnorm: chunk exceeds compile limit "
      << kCompileMaxRows;

  const int32_t device_id = x.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(static_cast<uint8_t*>(x.data_ptr()),
            static_cast<uint8_t*>(gate.data_ptr()),
            static_cast<uint8_t*>(weight.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()),
            static_cast<int32_t>(x.size(0)),
            eps,
            scale,
            stream);
}

}  // namespace

bool has_fused_scale_gated_rmsnorm_specialization(
    int64_t head_size,
    torch::ScalarType x_dtype,
    torch::ScalarType gate_dtype,
    torch::ScalarType weight_dtype) {
  if (head_size != kSupportedHeadSize || x_dtype != torch::kFloat16 ||
      gate_dtype != torch::kBFloat16 || weight_dtype != torch::kBFloat16) {
    return false;
  }
  const auto specialization =
      build_runtime_specialization(head_size, gate_dtype);
  return find_fused_scale_gated_rmsnorm_kernel_entry(specialization) != nullptr;
}

torch::Tensor fused_scale_gated_rmsnorm(const torch::Tensor& x,
                                        const torch::Tensor& gate,
                                        const torch::Tensor& weight,
                                        float eps,
                                        float scale) {
  check_supported(x, gate, weight, eps, scale);
  auto output = torch::empty_like(gate);
  for (int64_t start = 0; start < x.size(0); start += kCompileMaxRows) {
    const int64_t chunk_rows = std::min(kCompileMaxRows, x.size(0) - start);
    auto x_chunk = x.narrow(0, start, chunk_rows);
    auto gate_chunk = gate.narrow(0, start, chunk_rows);
    auto output_chunk = output.narrow(0, start, chunk_rows);
    run_chunk(x_chunk, gate_chunk, weight, output_chunk, eps, scale);
  }
  return output;
}

}  // namespace xllm::kernel::npu::tilelang
