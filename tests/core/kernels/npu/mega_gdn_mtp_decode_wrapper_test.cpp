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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <array>
#include <cstdlib>
#include <optional>
#include <string>

#include "core/kernels/npu/npu_ops_api.h"

namespace xllm::kernel::npu {
namespace {

class ScopedEnvironmentVariable {
 public:
  explicit ScopedEnvironmentVariable(const char* name) : name_(name) {
    if (const char* value = std::getenv(name)) {
      original_value_ = value;
    }
  }

  ~ScopedEnvironmentVariable() {
    if (original_value_.has_value()) {
      setenv(name_.c_str(), original_value_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  void set() { setenv(name_.c_str(), "1", 1); }
  void unset() { unsetenv(name_.c_str()); }

 private:
  std::string name_;
  std::optional<std::string> original_value_;
};

TEST(MegaGdnMtpDecodeWrapperTest, RejectsCpuInputsBeforeAclnnDispatch) {
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kSequenceLength = 2;
  constexpr int64_t kNumKHeads = 1;
  constexpr int64_t kNumVHeads = 1;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kConvDim = (2 * kNumKHeads + kNumVHeads) * kHeadDim;
  constexpr int64_t kNumStateSlots = 1;

  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32);
  const torch::TensorOptions int32_options =
      torch::TensorOptions().dtype(torch::kInt32);

  torch::Tensor qkv =
      torch::zeros({kBatchSize, kSequenceLength, kConvDim}, bf16_options);
  torch::Tensor z = torch::zeros(
      {kBatchSize, kSequenceLength, kNumVHeads, kHeadDim}, bf16_options);
  torch::Tensor b =
      torch::zeros({kBatchSize, kSequenceLength, kNumVHeads}, bf16_options);
  torch::Tensor a = torch::zeros_like(b);
  torch::Tensor conv_weight = torch::zeros({4, kConvDim}, bf16_options);
  torch::Tensor conv_state = torch::zeros(
      {kNumStateSlots, kSequenceLength + 2, kConvDim}, bf16_options);
  torch::Tensor a_log = torch::zeros({kNumVHeads}, fp32_options);
  torch::Tensor dt_bias = torch::zeros_like(a_log);
  torch::Tensor ssm_state = torch::zeros(
      {kNumStateSlots * kSequenceLength, kNumVHeads, kHeadDim, kHeadDim},
      fp32_options);
  torch::Tensor read_state_indices = torch::zeros({kBatchSize}, int32_options);
  torch::Tensor write_state_indices = torch::zeros({kBatchSize}, int32_options);
  torch::Tensor num_accepted_tokens = torch::ones({kBatchSize}, int32_options);
  torch::Tensor norm_weight = torch::ones({kHeadDim}, bf16_options);

  EXPECT_DEATH(mega_gdn_mtp_decode(qkv,
                                   z,
                                   b,
                                   a,
                                   conv_weight,
                                   conv_state,
                                   a_log,
                                   dt_bias,
                                   ssm_state,
                                   read_state_indices,
                                   write_state_indices,
                                   num_accepted_tokens,
                                   norm_weight),
               "qkv must be on NPU");
}

class MegaGdnMtpDecodeWrapperNpuTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

TEST_F(MegaGdnMtpDecodeWrapperNpuTest,
       DispatchesZeroGoldenAndPreservesStateStorage) {
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kSequenceLength = 2;
  constexpr int64_t kNumKHeads = 1;
  constexpr int64_t kNumVHeads = 1;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kConvDim = (2 * kNumKHeads + kNumVHeads) * kHeadDim;
  constexpr int64_t kNumStateSlots = 1;

  const torch::Device device("npu:0");
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::TensorOptions int32_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  torch::Tensor qkv =
      torch::zeros({kBatchSize, kSequenceLength, kConvDim}, bf16_options);
  torch::Tensor z = torch::zeros(
      {kBatchSize, kSequenceLength, kNumVHeads, kHeadDim}, bf16_options);
  torch::Tensor b =
      torch::zeros({kBatchSize, kSequenceLength, kNumVHeads}, bf16_options);
  torch::Tensor a = torch::zeros_like(b);
  torch::Tensor conv_weight = torch::zeros({4, kConvDim}, bf16_options);
  torch::Tensor conv_state = torch::zeros(
      {kNumStateSlots, kSequenceLength + 2, kConvDim}, bf16_options);
  torch::Tensor a_log = torch::zeros({kNumVHeads}, fp32_options);
  torch::Tensor dt_bias = torch::zeros_like(a_log);
  torch::Tensor ssm_state = torch::zeros(
      {kNumStateSlots * kSequenceLength, kNumVHeads, kHeadDim, kHeadDim},
      fp32_options);
  torch::Tensor read_state_indices = torch::zeros({kBatchSize}, int32_options);
  torch::Tensor write_state_indices = torch::zeros({kBatchSize}, int32_options);
  torch::Tensor num_accepted_tokens = torch::ones({kBatchSize}, int32_options);
  torch::Tensor norm_weight = torch::ones({kHeadDim}, bf16_options);
  const void* conv_state_storage = conv_state.data_ptr();
  const void* ssm_state_storage = ssm_state.data_ptr();
  ScopedEnvironmentVariable segmented_path("XLLM_USE_TILELANG_MTP_SEGMENTED");
  segmented_path.set();

  torch::Tensor out = mega_gdn_mtp_decode(qkv,
                                          z,
                                          b,
                                          a,
                                          conv_weight,
                                          conv_state,
                                          a_log,
                                          dt_bias,
                                          ssm_state,
                                          read_state_indices,
                                          write_state_indices,
                                          num_accepted_tokens,
                                          norm_weight);

  EXPECT_EQ(out.sizes(), z.sizes());
  EXPECT_EQ(out.device(), device);
  EXPECT_EQ(conv_state.data_ptr(), conv_state_storage);
  EXPECT_EQ(ssm_state.data_ptr(), ssm_state_storage);
  EXPECT_TRUE(torch::isfinite(out.cpu()).all().item<bool>());
  EXPECT_EQ(out.cpu().count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(conv_state.cpu().count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(ssm_state.cpu().count_nonzero().item<int64_t>(), 0);
}

TEST_F(MegaGdnMtpDecodeWrapperNpuTest,
       SegmentedTilelangMatchesAclnnAcrossSupportedKAndStateModes) {
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kNumKHeads = 8;
  constexpr int64_t kNumVHeads = 24;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kConvDim = (2 * kNumKHeads + kNumVHeads) * kHeadDim;
  constexpr std::array<int64_t, 6> kSpeculativeTokens = {1, 2, 3, 4, 5, 8};

  const torch::Device device("npu:0");
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::TensorOptions int32_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  ScopedEnvironmentVariable segmented_path("XLLM_USE_TILELANG_MTP_SEGMENTED");

  for (const int64_t speculative_tokens : kSpeculativeTokens) {
    const int64_t sequence_length = speculative_tokens + 1;
    const std::array<int64_t, 3> accepted_counts = {
        1, (sequence_length + 1) / 2, sequence_length};
    for (const int64_t accepted_count : accepted_counts) {
      for (const bool prefix_fork : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "K=" << speculative_tokens
                                          << ", accepted=" << accepted_count
                                          << ", prefix_fork=" << prefix_fork);
        const int64_t num_state_slots = prefix_fork ? 2 : 1;
        torch::manual_seed(20260730 + speculative_tokens * 100 +
                           accepted_count * 10 + prefix_fork);
        torch::Tensor qkv =
            torch::randn({kBatchSize, sequence_length, kConvDim},
                         bf16_options) *
            0.1;
        torch::Tensor z =
            torch::randn({kBatchSize, sequence_length, kNumVHeads, kHeadDim},
                         bf16_options) *
            0.1;
        torch::Tensor b =
            torch::randn({kBatchSize, sequence_length, kNumVHeads},
                         bf16_options) *
            0.1;
        torch::Tensor a = torch::randn_like(b) * 0.1;
        torch::Tensor conv_weight =
            torch::randn({4, kConvDim}, bf16_options) * 0.1;
        torch::Tensor initial_conv_state =
            torch::randn({num_state_slots, sequence_length + 2, kConvDim},
                         bf16_options) *
            0.1;
        torch::Tensor a_log = torch::full({kNumVHeads}, -1.0, fp32_options);
        torch::Tensor dt_bias = torch::zeros({kNumVHeads}, fp32_options);
        torch::Tensor initial_ssm_state =
            torch::randn({num_state_slots * sequence_length,
                          kNumVHeads,
                          kHeadDim,
                          kHeadDim},
                         fp32_options) *
            0.1;
        torch::Tensor read_state_indices =
            torch::zeros({kBatchSize}, int32_options);
        torch::Tensor write_state_indices =
            torch::full({kBatchSize}, prefix_fork ? 1 : 0, int32_options);
        torch::Tensor num_accepted_tokens =
            torch::full({kBatchSize}, accepted_count, int32_options);
        torch::Tensor norm_weight =
            torch::randn({kHeadDim}, bf16_options) * 0.1;

        torch::Tensor aclnn_conv_state = initial_conv_state.clone();
        torch::Tensor aclnn_ssm_state = initial_ssm_state.clone();
        torch::Tensor segmented_conv_state = initial_conv_state.clone();
        torch::Tensor segmented_ssm_state = initial_ssm_state.clone();

        segmented_path.unset();
        torch::Tensor aclnn_out = mega_gdn_mtp_decode(qkv,
                                                      z,
                                                      b,
                                                      a,
                                                      conv_weight,
                                                      aclnn_conv_state,
                                                      a_log,
                                                      dt_bias,
                                                      aclnn_ssm_state,
                                                      read_state_indices,
                                                      write_state_indices,
                                                      num_accepted_tokens,
                                                      norm_weight);
        segmented_path.set();
        torch::Tensor segmented_out = mega_gdn_mtp_decode(qkv,
                                                          z,
                                                          b,
                                                          a,
                                                          conv_weight,
                                                          segmented_conv_state,
                                                          a_log,
                                                          dt_bias,
                                                          segmented_ssm_state,
                                                          read_state_indices,
                                                          write_state_indices,
                                                          num_accepted_tokens,
                                                          norm_weight);

        torch::Tensor aclnn_conv_cpu = aclnn_conv_state.cpu();
        torch::Tensor segmented_conv_cpu = segmented_conv_state.cpu();
        torch::Tensor aclnn_ssm_cpu = aclnn_ssm_state.cpu();
        torch::Tensor segmented_ssm_cpu = segmented_ssm_state.cpu();
        torch::Tensor aclnn_out_cpu = aclnn_out.cpu();
        torch::Tensor segmented_out_cpu = segmented_out.cpu();
        EXPECT_TRUE(
            torch::allclose(segmented_conv_cpu, aclnn_conv_cpu, 0.0, 0.0));
        EXPECT_TRUE(
            torch::allclose(segmented_ssm_cpu, aclnn_ssm_cpu, 5e-3, 2e-6))
            << "max SSM abs error: "
            << (segmented_ssm_cpu - aclnn_ssm_cpu).abs().max().item<float>();
        EXPECT_TRUE(
            torch::allclose(segmented_out_cpu, aclnn_out_cpu, 5e-3, 2e-2))
            << "max output abs error: "
            << (segmented_out_cpu - aclnn_out_cpu).abs().max().item<float>();
      }
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::npu
