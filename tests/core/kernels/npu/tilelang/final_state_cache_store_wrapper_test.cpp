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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <functional>
#include <iostream>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kNumHeads = 24;
constexpr int64_t kHeadKDim = 128;
constexpr int64_t kHeadVDim = 128;

class TileLangFinalStateCacheStoreWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

double measure_npu_event_us(const std::function<void()>& fn,
                            int32_t device_id,
                            int warmup_iters = 10,
                            int measure_iters = 100) {
  const aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  for (int i = 0; i < warmup_iters; ++i) {
    fn();
  }
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

  aclrtEvent start_event = nullptr;
  aclrtEvent end_event = nullptr;
  CHECK_EQ(aclrtCreateEvent(&start_event), ACL_SUCCESS);
  CHECK_EQ(aclrtCreateEvent(&end_event), ACL_SUCCESS);
  CHECK_EQ(aclrtRecordEvent(start_event, stream), ACL_SUCCESS);
  for (int i = 0; i < measure_iters; ++i) {
    fn();
  }
  CHECK_EQ(aclrtRecordEvent(end_event, stream), ACL_SUCCESS);
  CHECK_EQ(aclrtSynchronizeEvent(end_event), ACL_SUCCESS);

  float elapsed_ms = 0.0F;
  CHECK_EQ(aclrtEventElapsedTime(&elapsed_ms, start_event, end_event),
           ACL_SUCCESS);
  CHECK_EQ(aclrtDestroyEvent(start_event), ACL_SUCCESS);
  CHECK_EQ(aclrtDestroyEvent(end_event), ACL_SUCCESS);
  return static_cast<double>(elapsed_ms) * 1000.0 / measure_iters;
}

TEST_F(TileLangFinalStateCacheStoreWrapperTest, ExactCastAndSlotStore) {
  ASSERT_TRUE(has_final_state_cache_store_specialization(
      kNumHeads, kHeadKDim, kHeadVDim));
  auto half_options = torch::TensorOptions()
                          .device(torch::Device("npu:0"))
                          .dtype(torch::kFloat16);
  auto float_options = half_options.dtype(torch::kFloat32);
  torch::manual_seed(53);
  auto final_state =
      torch::randn({1, kNumHeads, kHeadKDim, kHeadVDim}, half_options);
  auto cache =
      torch::zeros({3, kNumHeads, kHeadKDim, kHeadVDim}, float_options);
  auto cache_slot = cache.narrow(0, 1, 1);

  final_state_cache_store(final_state, cache_slot);
  EXPECT_TRUE(torch::equal(cache_slot, final_state.to(torch::kFloat32)));
  EXPECT_EQ(torch::count_nonzero(cache.select(0, 0)).item<int64_t>(), 0);
  EXPECT_EQ(torch::count_nonzero(cache.select(0, 2)).item<int64_t>(), 0);

  const double latency_us = measure_npu_event_us(
      [&]() { final_state_cache_store(final_state, cache_slot); },
      final_state.device().index());
  std::cout << "final_state_cache_store aot_event_us=" << latency_us
            << std::endl;
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
