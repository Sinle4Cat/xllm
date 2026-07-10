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
#include <cstdint>
#include <limits>
#include <tuple>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_QWEN35_PROJECTION_LAYOUT_REGISTRY_INC
#error "XLLM_TL_QWEN35_PROJECTION_LAYOUT_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kCompileMaxRows = 64;

#include XLLM_TL_QWEN35_PROJECTION_LAYOUT_REGISTRY_INC

Qwen35ProjectionLayoutSpecialization build_runtime_specialization(
    int64_t qkv_size,
    int64_t z_size,
    int64_t num_heads,
    torch::ScalarType dtype) {
  CHECK_LE(qkv_size, std::numeric_limits<int32_t>::max())
      << "TileLang qwen35_projection_layout: qkv_size exceeds int32";
  CHECK_LE(z_size, std::numeric_limits<int32_t>::max())
      << "TileLang qwen35_projection_layout: z_size exceeds int32";
  CHECK_LE(num_heads, std::numeric_limits<int32_t>::max())
      << "TileLang qwen35_projection_layout: num_heads exceeds int32";
  return make_qwen35_projection_layout_specialization(
      Qwen35ProjectionLayoutQkvSize{static_cast<int32_t>(qkv_size)},
      Qwen35ProjectionLayoutZSize{static_cast<int32_t>(z_size)},
      Qwen35ProjectionLayoutNumHeads{static_cast<int32_t>(num_heads)},
      Qwen35ProjectionLayoutDType{to_tilelang_dtype(dtype)});
}

void check_supported(const torch::Tensor& projection,
                     int64_t qkv_size,
                     int64_t z_size,
                     int64_t num_heads) {
  CHECK(projection.defined())
      << "TileLang qwen35_projection_layout: projection must be defined";
  CHECK(projection.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang qwen35_projection_layout: projection must be on NPU";
  CHECK_EQ(projection.dim(), 2)
      << "TileLang qwen35_projection_layout: projection must be 2D";
  CHECK_EQ(projection.dtype(), torch::kBFloat16)
      << "TileLang qwen35_projection_layout: projection must be bfloat16";
  CHECK(projection.is_contiguous())
      << "TileLang qwen35_projection_layout: projection must be contiguous";
  CHECK_GT(projection.size(0), 0)
      << "TileLang qwen35_projection_layout: num_rows must be positive";
  CHECK_GT(qkv_size, 0)
      << "TileLang qwen35_projection_layout: qkv_size must be positive";
  CHECK_GT(z_size, 0)
      << "TileLang qwen35_projection_layout: z_size must be positive";
  CHECK_GT(num_heads, 0)
      << "TileLang qwen35_projection_layout: num_heads must be positive";
  CHECK_EQ(projection.size(1), qkv_size + z_size + 2 * num_heads)
      << "TileLang qwen35_projection_layout: projection width mismatch";

  const auto specialization = build_runtime_specialization(
      qkv_size, z_size, num_heads, projection.scalar_type());
  CHECK(find_qwen35_projection_layout_kernel_entry(specialization) != nullptr)
      << "TileLang qwen35_projection_layout: no compiled variant. Available "
         "variants: "
      << available_qwen35_projection_layout_variant_keys();
}

void run_chunk(const torch::Tensor& projection,
               torch::Tensor& qkv,
               torch::Tensor& z,
               torch::Tensor& b,
               torch::Tensor& a) {
  CHECK_LE(projection.size(0), kCompileMaxRows)
      << "TileLang qwen35_projection_layout: chunk exceeds compile limit "
      << kCompileMaxRows;
  const auto specialization = build_runtime_specialization(
      qkv.size(1), z.size(1), b.size(1), projection.scalar_type());
  const auto* entry =
      find_qwen35_projection_layout_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang qwen35_projection_layout: no compiled variant";

  const int32_t device_id = projection.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(static_cast<uint8_t*>(projection.data_ptr()),
            static_cast<uint8_t*>(qkv.data_ptr()),
            static_cast<uint8_t*>(z.data_ptr()),
            static_cast<uint8_t*>(b.data_ptr()),
            static_cast<uint8_t*>(a.data_ptr()),
            static_cast<int>(projection.size(0)),
            stream);
}

}  // namespace

bool has_qwen35_projection_layout_specialization(int64_t qkv_size,
                                                 int64_t z_size,
                                                 int64_t num_heads,
                                                 torch::ScalarType dtype) {
  if (qkv_size <= 0 || z_size <= 0 || num_heads <= 0 ||
      qkv_size > std::numeric_limits<int32_t>::max() ||
      z_size > std::numeric_limits<int32_t>::max() ||
      num_heads > std::numeric_limits<int32_t>::max() ||
      dtype != torch::kBFloat16) {
    return false;
  }
  const auto specialization =
      build_runtime_specialization(qkv_size, z_size, num_heads, dtype);
  return find_qwen35_projection_layout_kernel_entry(specialization) != nullptr;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
qwen35_projection_layout(const torch::Tensor& projection,
                         int64_t qkv_size,
                         int64_t z_size,
                         int64_t num_heads) {
  check_supported(projection, qkv_size, z_size, num_heads);
  auto qkv = torch::empty({projection.size(0), qkv_size}, projection.options());
  auto z = torch::empty({projection.size(0), z_size}, projection.options());
  auto b = torch::empty({projection.size(0), num_heads}, projection.options());
  auto a = torch::empty_like(b);

  for (int64_t start = 0; start < projection.size(0);
       start += kCompileMaxRows) {
    const int64_t chunk_rows =
        std::min(kCompileMaxRows, projection.size(0) - start);
    auto projection_chunk = projection.narrow(0, start, chunk_rows);
    auto qkv_chunk = qkv.narrow(0, start, chunk_rows);
    auto z_chunk = z.narrow(0, start, chunk_rows);
    auto b_chunk = b.narrow(0, start, chunk_rows);
    auto a_chunk = a.narrow(0, start, chunk_rows);
    run_chunk(projection_chunk, qkv_chunk, z_chunk, b_chunk, a_chunk);
  }
  return {qkv, z, b, a};
}

}  // namespace xllm::kernel::npu::tilelang
