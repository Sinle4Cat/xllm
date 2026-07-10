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

#ifndef XLLM_TL_CAUSAL_CONV1D_QKV_PREPARE_REGISTRY_INC
#error "XLLM_TL_CAUSAL_CONV1D_QKV_PREPARE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kCompileMaxTokens = 8192;

#include XLLM_TL_CAUSAL_CONV1D_QKV_PREPARE_REGISTRY_INC

CausalConv1dQkvPrepareSpecialization build_runtime_specialization(
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_dim) {
  CHECK_LE(num_k_heads, std::numeric_limits<int32_t>::max());
  CHECK_LE(num_v_heads, std::numeric_limits<int32_t>::max());
  CHECK_LE(head_dim, std::numeric_limits<int32_t>::max());
  return make_causal_conv1d_qkv_prepare_specialization(
      CausalConv1dQkvPrepareNumKHeads{static_cast<int32_t>(num_k_heads)},
      CausalConv1dQkvPrepareNumVHeads{static_cast<int32_t>(num_v_heads)},
      CausalConv1dQkvPrepareHeadDim{static_cast<int32_t>(head_dim)});
}

void check_supported(const torch::Tensor& mixed_qkv,
                     int64_t num_k_heads,
                     int64_t num_v_heads,
                     int64_t head_dim) {
  CHECK(mixed_qkv.defined())
      << "TileLang causal_conv1d_qkv_prepare: input must be defined";
  CHECK(mixed_qkv.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang causal_conv1d_qkv_prepare: input must be on NPU";
  CHECK_EQ(mixed_qkv.dim(), 2)
      << "TileLang causal_conv1d_qkv_prepare: input must be [T,width]";
  CHECK_EQ(mixed_qkv.scalar_type(), torch::kBFloat16)
      << "TileLang causal_conv1d_qkv_prepare: input must be bfloat16";
  CHECK(mixed_qkv.is_contiguous())
      << "TileLang causal_conv1d_qkv_prepare: input must be contiguous";
  CHECK_GT(mixed_qkv.size(0), 0);
  CHECK_EQ(mixed_qkv.size(1), (2 * num_k_heads + num_v_heads) * head_dim)
      << "TileLang causal_conv1d_qkv_prepare: input width mismatch";
  const auto specialization =
      build_runtime_specialization(num_k_heads, num_v_heads, head_dim);
  CHECK(find_causal_conv1d_qkv_prepare_kernel_entry(specialization) != nullptr)
      << "TileLang causal_conv1d_qkv_prepare: no compiled variant. Available "
         "variants: "
      << available_causal_conv1d_qkv_prepare_variant_keys();
}

void run_chunk(const torch::Tensor& mixed_qkv,
               torch::Tensor& q,
               torch::Tensor& k,
               torch::Tensor& v,
               float eps) {
  CHECK_LE(mixed_qkv.size(0), kCompileMaxTokens);
  const auto specialization =
      build_runtime_specialization(q.size(2), v.size(2), q.size(3));
  const auto* entry =
      find_causal_conv1d_qkv_prepare_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang causal_conv1d_qkv_prepare: no compiled variant";
  const int32_t device_id = mixed_qkv.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(static_cast<uint8_t*>(mixed_qkv.data_ptr()),
            static_cast<uint8_t*>(q.data_ptr()),
            static_cast<uint8_t*>(k.data_ptr()),
            static_cast<uint8_t*>(v.data_ptr()),
            static_cast<int32_t>(mixed_qkv.size(0)),
            eps,
            stream);
}

}  // namespace

bool has_causal_conv1d_qkv_prepare_specialization(int64_t num_k_heads,
                                                  int64_t num_v_heads,
                                                  int64_t head_dim) {
  if (num_k_heads <= 0 || num_v_heads <= 0 || head_dim <= 0 ||
      num_k_heads > std::numeric_limits<int32_t>::max() ||
      num_v_heads > std::numeric_limits<int32_t>::max() ||
      head_dim > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  const auto specialization =
      build_runtime_specialization(num_k_heads, num_v_heads, head_dim);
  return find_causal_conv1d_qkv_prepare_kernel_entry(specialization) != nullptr;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
causal_conv1d_qkv_prepare(const torch::Tensor& mixed_qkv,
                          int64_t num_k_heads,
                          int64_t num_v_heads,
                          int64_t head_dim,
                          float eps) {
  check_supported(mixed_qkv, num_k_heads, num_v_heads, head_dim);
  auto options = mixed_qkv.options().dtype(torch::kFloat16);
  auto q = torch::empty({1, mixed_qkv.size(0), num_k_heads, head_dim}, options);
  auto k = torch::empty_like(q);
  auto v = torch::empty({1, mixed_qkv.size(0), num_v_heads, head_dim}, options);

  for (int64_t start = 0; start < mixed_qkv.size(0);
       start += kCompileMaxTokens) {
    const int64_t chunk_tokens =
        std::min(kCompileMaxTokens, mixed_qkv.size(0) - start);
    auto input_chunk = mixed_qkv.narrow(0, start, chunk_tokens);
    auto q_chunk = q.narrow(1, start, chunk_tokens);
    auto k_chunk = k.narrow(1, start, chunk_tokens);
    auto v_chunk = v.narrow(1, start, chunk_tokens);
    run_chunk(input_chunk, q_chunk, k_chunk, v_chunk, eps);
  }
  return {q, k, v};
}

}  // namespace xllm::kernel::npu::tilelang
