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
#include <torch_npu/torch_npu.h>

#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "triton_npu/torch_api/triton_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangFusedScaleGatedRmsnormWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

struct TestCase {
  std::string name;
  int64_t num_rows;
  int64_t seed;
  std::optional<float> x_fill;
  std::optional<float> gate_fill;
  bool unit_weight = false;
};

torch::Tensor torch_reference(const torch::Tensor& x,
                              const torch::Tensor& gate,
                              const torch::Tensor& weight,
                              float eps,
                              float scale) {
  auto scaled = (x.to(torch::kFloat32) * scale).to(torch::kBFloat16);
  auto scaled_fp32 = scaled.to(torch::kFloat32);
  auto inv_rms = torch::rsqrt(scaled_fp32.square().mean(-1, true) + eps);
  return (scaled_fp32 * inv_rms * weight.to(torch::kFloat32) *
          torch::silu(gate.to(torch::kFloat32)))
      .to(torch::kBFloat16);
}

void run_case(const TestCase& test_case) {
  constexpr int64_t kHeadSize = 128;
  constexpr float kEps = 1e-6F;
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadSize));
  const auto device = torch::Device("npu:0");
  torch::manual_seed(test_case.seed);

  auto fp16_options =
      torch::TensorOptions().dtype(torch::kFloat16).device(device);
  auto bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  auto x = test_case.x_fill.has_value()
               ? torch::full({test_case.num_rows, kHeadSize},
                             test_case.x_fill.value(),
                             fp16_options)
               : torch::randn({test_case.num_rows, kHeadSize}, fp16_options);
  auto gate = test_case.gate_fill.has_value()
                  ? torch::full({test_case.num_rows, kHeadSize},
                                test_case.gate_fill.value(),
                                bf16_options)
                  : torch::randn({test_case.num_rows, kHeadSize}, bf16_options);
  auto weight = test_case.unit_weight ? torch::ones({kHeadSize}, bf16_options)
                                      : torch::randn({kHeadSize}, bf16_options);

  auto output = fused_scale_gated_rmsnorm(x, gate, weight, kEps, scale);
  auto reference = torch_reference(x, gate, weight, kEps, scale);
  auto scaled = (x.to(torch::kFloat32) * scale).to(torch::kBFloat16);
  torch::Tensor bias;
  std::optional<torch::Tensor> gate_optional = gate;
  auto production_reference = xllm::kernel::npu::layer_norm_fwd(
      scaled, weight, bias, kEps, gate_optional, kHeadSize, true, true);
  auto max_diff = (output.to(torch::kFloat32) - reference.to(torch::kFloat32))
                      .abs()
                      .max()
                      .item<float>();
  auto production_diff =
      (output.to(torch::kFloat32) - production_reference.to(torch::kFloat32))
          .abs();
  auto production_max_diff = production_diff.max().item<float>();
  auto production_mismatch_count =
      output.ne(production_reference).sum().item<int64_t>();
  const auto production_mismatch_ratio =
      static_cast<double>(production_mismatch_count) / output.numel();
  auto production_torch_diff =
      (production_reference.to(torch::kFloat32) - reference.to(torch::kFloat32))
          .abs();
  auto production_torch_max_diff = production_torch_diff.max().item<float>();
  auto production_torch_mismatch_count =
      production_reference.ne(reference).sum().item<int64_t>();
  const auto production_torch_mismatch_ratio =
      static_cast<double>(production_torch_mismatch_count) / output.numel();

  std::cout << test_case.name << ": torch_max_diff=" << max_diff
            << ", production_max_diff=" << production_max_diff
            << ", production_mismatch_ratio=" << production_mismatch_ratio
            << ", production_torch_max_diff=" << production_torch_max_diff
            << ", production_torch_mismatch_ratio="
            << production_torch_mismatch_ratio << std::endl;

  EXPECT_TRUE(torch::allclose(output, reference, 1e-2, 1e-2))
      << "output mismatch, max_diff=" << max_diff;
  EXPECT_TRUE(torch::allclose(output, production_reference, 1e-2, 1e-2))
      << "production output mismatch, max_diff=" << production_max_diff
      << ", mismatch_ratio=" << production_mismatch_ratio;
}

TEST_F(TileLangFusedScaleGatedRmsnormWrapperTest, MatchesTorchReference) {
  const std::vector<TestCase> cases = {
      {
          .name = "tail_rows_17",
          .num_rows = 17,
          .seed = 23,
          .x_fill = std::nullopt,
          .gate_fill = std::nullopt,
      },
      {
          .name = "qwen35_27b_tp2_prefill_2k",
          .num_rows = 2048 * 24,
          .seed = 24,
          .x_fill = std::nullopt,
          .gate_fill = std::nullopt,
      },
      {
          .name = "diagnose_rmsnorm_with_constant_gate",
          .num_rows = 1024,
          .seed = 25,
          .x_fill = std::nullopt,
          .gate_fill = 10.0F,
          .unit_weight = true,
      },
      {
          .name = "diagnose_silu_with_unit_scaled_input",
          .num_rows = 1024,
          .seed = 26,
          .x_fill = std::sqrt(128.0F),
          .gate_fill = std::nullopt,
          .unit_weight = true,
      },
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    run_case(test_case);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
