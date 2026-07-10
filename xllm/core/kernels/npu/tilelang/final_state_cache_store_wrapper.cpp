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

#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_FINAL_STATE_CACHE_STORE_REGISTRY_INC
#error "XLLM_TL_FINAL_STATE_CACHE_STORE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_FINAL_STATE_CACHE_STORE_REGISTRY_INC

FinalStateCacheStoreSpecialization build_runtime_specialization(
    int64_t num_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  CHECK_LE(num_heads, std::numeric_limits<int32_t>::max());
  CHECK_LE(head_k_dim, std::numeric_limits<int32_t>::max());
  CHECK_LE(head_v_dim, std::numeric_limits<int32_t>::max());
  return make_final_state_cache_store_specialization(
      FinalStateCacheStoreNumHeads{static_cast<int32_t>(num_heads)},
      FinalStateCacheStoreHeadKDim{static_cast<int32_t>(head_k_dim)},
      FinalStateCacheStoreHeadVDim{static_cast<int32_t>(head_v_dim)});
}

void check_supported(const torch::Tensor& final_state,
                     const torch::Tensor& cache_slot) {
  CHECK(final_state.defined() && cache_slot.defined())
      << "TileLang final_state_cache_store: tensors must be defined";
  CHECK(final_state.device().type() == c10::DeviceType::PrivateUse1 &&
        cache_slot.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang final_state_cache_store: tensors must be on NPU";
  CHECK_EQ(final_state.device(), cache_slot.device())
      << "TileLang final_state_cache_store: tensors must share a device";
  CHECK_EQ(final_state.dim(), 4)
      << "TileLang final_state_cache_store: final_state must be [1,H,K,V]";
  CHECK_EQ(final_state.size(0), 1)
      << "TileLang final_state_cache_store: only one sequence is supported";
  CHECK_EQ(final_state.scalar_type(), torch::kFloat16)
      << "TileLang final_state_cache_store: final_state must be float16";
  CHECK_EQ(cache_slot.scalar_type(), torch::kFloat32)
      << "TileLang final_state_cache_store: cache_slot must be float32";
  CHECK(final_state.is_contiguous() && cache_slot.is_contiguous())
      << "TileLang final_state_cache_store: tensors must be contiguous";
  CHECK_EQ(final_state.sizes(), cache_slot.sizes())
      << "TileLang final_state_cache_store: shape mismatch";

  const auto specialization = build_runtime_specialization(
      final_state.size(1), final_state.size(2), final_state.size(3));
  CHECK(find_final_state_cache_store_kernel_entry(specialization) != nullptr)
      << "TileLang final_state_cache_store: no compiled variant. Available "
         "variants: "
      << available_final_state_cache_store_variant_keys();
}

}  // namespace

bool has_final_state_cache_store_specialization(int64_t num_heads,
                                                int64_t head_k_dim,
                                                int64_t head_v_dim) {
  if (num_heads <= 0 || head_k_dim <= 0 || head_v_dim <= 0 ||
      num_heads > std::numeric_limits<int32_t>::max() ||
      head_k_dim > std::numeric_limits<int32_t>::max() ||
      head_v_dim > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  const auto specialization =
      build_runtime_specialization(num_heads, head_k_dim, head_v_dim);
  return find_final_state_cache_store_kernel_entry(specialization) != nullptr;
}

void final_state_cache_store(const torch::Tensor& final_state,
                             torch::Tensor& cache_slot) {
  check_supported(final_state, cache_slot);
  const auto specialization = build_runtime_specialization(
      final_state.size(1), final_state.size(2), final_state.size(3));
  const auto* entry = find_final_state_cache_store_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang final_state_cache_store: no compiled variant";

  const int32_t device_id = final_state.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(static_cast<uint8_t*>(final_state.data_ptr()),
            static_cast<uint8_t*>(cache_slot.data_ptr()),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
