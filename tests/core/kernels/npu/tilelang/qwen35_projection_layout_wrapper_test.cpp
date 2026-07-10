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
#include <tuple>
#include <vector>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kQkvSize = 5120;
constexpr int64_t kZSize = 3072;
constexpr int64_t kNumHeads = 24;
constexpr int64_t kProjectionSize = kQkvSize + kZSize + 2 * kNumHeads;

class TileLangQwen35ProjectionLayoutWrapperTest : public ::testing::Test {
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

void run_case(int64_t num_rows) {
  auto options = torch::TensorOptions()
                     .device(torch::Device("npu:0"))
                     .dtype(torch::kBFloat16);
  torch::manual_seed(41 + num_rows);
  auto projection = torch::randn({num_rows, kProjectionSize}, options);

  torch::Tensor qkv, z, b, a;
  std::tie(qkv, z, b, a) =
      qwen35_projection_layout(projection, kQkvSize, kZSize, kNumHeads);
  const auto reference =
      torch::split(projection, {kQkvSize, kZSize, kNumHeads, kNumHeads}, -1);

  const std::vector<torch::Tensor> outputs = {qkv, z, b, a};
  ASSERT_EQ(outputs.size(), reference.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    EXPECT_TRUE(outputs[i].is_contiguous());
    EXPECT_TRUE(torch::equal(outputs[i], reference[i]));
  }

  const double latency_us = measure_npu_event_us(
      [&]() {
        std::tie(qkv, z, b, a) =
            qwen35_projection_layout(projection, kQkvSize, kZSize, kNumHeads);
      },
      projection.device().index());
  std::cout << "qwen35_projection_layout rows=" << num_rows
            << ", aot_event_us=" << latency_us << std::endl;
}

TEST_F(TileLangQwen35ProjectionLayoutWrapperTest, ExactSplitAndAotLatency) {
  EXPECT_TRUE(has_qwen35_projection_layout_specialization(
      kQkvSize, kZSize, kNumHeads, torch::kBFloat16));
  for (const int64_t num_rows : {1, 2, 4}) {
    SCOPED_TRACE(::testing::Message() << "num_rows=" << num_rows);
    run_case(num_rows);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
