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

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kNumTokens = 2048;
constexpr int64_t kNumKHeads = 8;
constexpr int64_t kNumVHeads = 24;
constexpr int64_t kHeadDim = 128;
constexpr float kEps = 1e-6F;

class TileLangCausalConv1dQkvPrepareWrapperTest : public ::testing::Test {
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

torch::Tensor normalize_reference(const torch::Tensor& input) {
  auto fp32 = input.to(torch::kFloat32);
  auto norm = torch::rsqrt(
      torch::sum(fp32 * fp32, /*dim=*/-1, /*keepdim=*/true) + kEps);
  return (fp32 * norm).to(torch::kBFloat16).to(torch::kFloat16);
}

TEST_F(TileLangCausalConv1dQkvPrepareWrapperTest,
       MatchesReferenceAndMeasuresLatency) {
  ASSERT_TRUE(has_causal_conv1d_qkv_prepare_specialization(
      kNumKHeads, kNumVHeads, kHeadDim));
  auto options = torch::TensorOptions()
                     .device(torch::Device("npu:0"))
                     .dtype(torch::kBFloat16);
  torch::manual_seed(61);
  auto input = torch::randn(
      {kNumTokens, (2 * kNumKHeads + kNumVHeads) * kHeadDim}, options);
  torch::Tensor q, k, v;
  std::tie(q, k, v) =
      causal_conv1d_qkv_prepare(input, kNumKHeads, kNumVHeads, kHeadDim, kEps);
  auto input_heads =
      input.view({kNumTokens, 2 * kNumKHeads + kNumVHeads, kHeadDim});
  auto q_ref = normalize_reference(input_heads.slice(1, 0, kNumKHeads));
  auto k_ref =
      normalize_reference(input_heads.slice(1, kNumKHeads, 2 * kNumKHeads));
  auto v_ref = input_heads.slice(1, 2 * kNumKHeads).to(torch::kFloat16);

  EXPECT_TRUE(torch::allclose(q.squeeze(0), q_ref, 2e-3, 2e-3));
  EXPECT_TRUE(torch::allclose(k.squeeze(0), k_ref, 2e-3, 2e-3));
  EXPECT_TRUE(torch::equal(v.squeeze(0), v_ref));
  EXPECT_TRUE(q.is_contiguous() && k.is_contiguous() && v.is_contiguous());

  const double latency_us = measure_npu_event_us(
      [&]() {
        std::tie(q, k, v) = causal_conv1d_qkv_prepare(
            input, kNumKHeads, kNumVHeads, kHeadDim, kEps);
      },
      input.device().index());
  std::cout << "causal_conv1d_qkv_prepare aot_event_us=" << latency_us
            << std::endl;
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
