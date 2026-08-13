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

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_MEGA_GDN_MTP_DECODE_CONV_REGISTRY_INC
#error "XLLM_TL_MEGA_GDN_MTP_DECODE_CONV_REGISTRY_INC is not defined"
#endif

#ifndef XLLM_TL_MEGA_GDN_MTP_DECODE_RECURRENT_REGISTRY_INC
#error "XLLM_TL_MEGA_GDN_MTP_DECODE_RECURRENT_REGISTRY_INC is not defined"
#endif

#ifndef XLLM_TL_MEGA_GDN_MTP_DECODE_NORM_REGISTRY_INC
#error "XLLM_TL_MEGA_GDN_MTP_DECODE_NORM_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int32_t kCompiledMaxBatchSize = 4;
constexpr int32_t kCompiledMaxStateSlots = 256;
constexpr int32_t kCompiledNumKHeads = 8;
constexpr int32_t kCompiledNumVHeads = 24;
constexpr int64_t kHeadDim = 128;

#include XLLM_TL_MEGA_GDN_MTP_DECODE_CONV_REGISTRY_INC
#include XLLM_TL_MEGA_GDN_MTP_DECODE_RECURRENT_REGISTRY_INC
#include XLLM_TL_MEGA_GDN_MTP_DECODE_NORM_REGISTRY_INC

uint8_t* mutable_data_ptr(const torch::Tensor& tensor) {
  return reinterpret_cast<uint8_t*>(const_cast<void*>(tensor.data_ptr()));
}

void check_segmented_specialization(const torch::Tensor& qkv,
                                    const torch::Tensor& z,
                                    const torch::Tensor& conv_state) {
  CHECK_GE(qkv.size(0), 1)
      << "TileLang MegaGdnMtpDecode: batch_size must be positive";
  CHECK_LE(qkv.size(0), kCompiledMaxBatchSize)
      << "TileLang MegaGdnMtpDecode: batch_size exceeds compiled maximum "
      << kCompiledMaxBatchSize;
  CHECK_GE(conv_state.size(0), 1)
      << "TileLang MegaGdnMtpDecode: num_state_slots must be positive";
  CHECK_LE(conv_state.size(0), kCompiledMaxStateSlots)
      << "TileLang MegaGdnMtpDecode: num_state_slots exceeds compiled maximum "
      << kCompiledMaxStateSlots;
  CHECK_EQ(z.size(2), kCompiledNumVHeads)
      << "TileLang MegaGdnMtpDecode: num_v_heads must be "
      << kCompiledNumVHeads;
  CHECK_EQ(z.size(3), kHeadDim)
      << "TileLang MegaGdnMtpDecode: head_dim must be " << kHeadDim;
  const int64_t qk_width = qkv.size(2) - kCompiledNumVHeads * kHeadDim;
  CHECK_EQ(qk_width, 2 * kCompiledNumKHeads * kHeadDim)
      << "TileLang MegaGdnMtpDecode: num_k_heads must be "
      << kCompiledNumKHeads;
}

MegaGdnMtpDecodeConvSpecialization make_conv_specialization(
    int32_t speculative_tokens) {
  return make_mega_gdn_mtp_decode_conv_specialization(
      MegaGdnMtpDecodeConvSpeculativeTokens{speculative_tokens},
      MegaGdnMtpDecodeConvMaxBatchSize{kCompiledMaxBatchSize},
      MegaGdnMtpDecodeConvNumStateSlots{kCompiledMaxStateSlots},
      MegaGdnMtpDecodeConvNumKHeads{kCompiledNumKHeads},
      MegaGdnMtpDecodeConvNumVHeads{kCompiledNumVHeads},
      MegaGdnMtpDecodeConvDType{TilelangDType::kBF16});
}

MegaGdnMtpDecodeRecurrentSpecialization make_recurrent_specialization(
    int32_t speculative_tokens) {
  return make_mega_gdn_mtp_decode_recurrent_specialization(
      MegaGdnMtpDecodeRecurrentSpeculativeTokens{speculative_tokens},
      MegaGdnMtpDecodeRecurrentMaxBatchSize{kCompiledMaxBatchSize},
      MegaGdnMtpDecodeRecurrentNumStateSlots{kCompiledMaxStateSlots},
      MegaGdnMtpDecodeRecurrentNumKHeads{kCompiledNumKHeads},
      MegaGdnMtpDecodeRecurrentNumVHeads{kCompiledNumVHeads},
      MegaGdnMtpDecodeRecurrentDType{TilelangDType::kBF16});
}

MegaGdnMtpDecodeNormSpecialization make_norm_specialization(
    int32_t speculative_tokens) {
  return make_mega_gdn_mtp_decode_norm_specialization(
      MegaGdnMtpDecodeNormSpeculativeTokens{speculative_tokens},
      MegaGdnMtpDecodeNormMaxBatchSize{kCompiledMaxBatchSize},
      MegaGdnMtpDecodeNormNumStateSlots{kCompiledMaxStateSlots},
      MegaGdnMtpDecodeNormNumKHeads{kCompiledNumKHeads},
      MegaGdnMtpDecodeNormNumVHeads{kCompiledNumVHeads},
      MegaGdnMtpDecodeNormDType{TilelangDType::kBF16});
}

}  // namespace

bool has_mega_gdn_mtp_decode_segmented_specialization(
    int64_t batch_size,
    int64_t num_state_slots,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t speculative_tokens) {
  if (batch_size < 1 || batch_size > kCompiledMaxBatchSize ||
      num_state_slots < 1 || num_state_slots > kCompiledMaxStateSlots ||
      num_k_heads != kCompiledNumKHeads || num_v_heads != kCompiledNumVHeads ||
      speculative_tokens < 1) {
    return false;
  }
  const int32_t compiled_speculative_tokens =
      static_cast<int32_t>(speculative_tokens);
  return find_mega_gdn_mtp_decode_conv_kernel_entry(make_conv_specialization(
             compiled_speculative_tokens)) != nullptr &&
         find_mega_gdn_mtp_decode_recurrent_kernel_entry(
             make_recurrent_specialization(compiled_speculative_tokens)) !=
             nullptr &&
         find_mega_gdn_mtp_decode_norm_kernel_entry(
             make_norm_specialization(compiled_speculative_tokens)) != nullptr;
}

torch::Tensor mega_gdn_mtp_decode_segmented(
    const torch::Tensor& qkv,
    const torch::Tensor& z,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& conv_weight,
    torch::Tensor& conv_state,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& ssm_state,
    const torch::Tensor& read_state_indices,
    const torch::Tensor& write_state_indices,
    const torch::Tensor& num_accepted_tokens,
    const torch::Tensor& norm_weight) {
  check_segmented_specialization(qkv, z, conv_state);
  const int32_t batch_size = static_cast<int32_t>(qkv.size(0));
  const int32_t speculative_tokens = static_cast<int32_t>(qkv.size(1) - 1);

  const auto* conv_entry = find_mega_gdn_mtp_decode_conv_kernel_entry(
      make_conv_specialization(speculative_tokens));
  const auto* recurrent_entry = find_mega_gdn_mtp_decode_recurrent_kernel_entry(
      make_recurrent_specialization(speculative_tokens));
  const auto* norm_entry = find_mega_gdn_mtp_decode_norm_kernel_entry(
      make_norm_specialization(speculative_tokens));
  CHECK(conv_entry != nullptr)
      << "TileLang MegaGdnMtpDecode: missing Conv specialization for K="
      << speculative_tokens;
  CHECK(recurrent_entry != nullptr)
      << "TileLang MegaGdnMtpDecode: missing recurrent specialization for K="
      << speculative_tokens;
  CHECK(norm_entry != nullptr)
      << "TileLang MegaGdnMtpDecode: missing Norm specialization for K="
      << speculative_tokens;

  auto conv_out = torch::empty_like(qkv);
  const auto options_fp32 = qkv.options().dtype(torch::kFloat32);
  auto qk_prepared = torch::empty(
      {batch_size, qkv.size(1), 2, kCompiledNumKHeads, kHeadDim}, options_fp32);
  auto gate_prepared = torch::empty(
      {batch_size, qkv.size(1), 2, kCompiledNumVHeads}, options_fp32);
  auto readout = torch::empty_like(z);
  auto out = torch::empty_like(z);
  const int32_t device_id = qkv.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  conv_entry->fn(mutable_data_ptr(qkv),
                 mutable_data_ptr(b),
                 mutable_data_ptr(a),
                 mutable_data_ptr(conv_weight),
                 mutable_data_ptr(conv_state),
                 mutable_data_ptr(a_log),
                 mutable_data_ptr(dt_bias),
                 mutable_data_ptr(read_state_indices),
                 mutable_data_ptr(write_state_indices),
                 mutable_data_ptr(num_accepted_tokens),
                 mutable_data_ptr(conv_out),
                 mutable_data_ptr(conv_state),
                 mutable_data_ptr(qk_prepared),
                 mutable_data_ptr(gate_prepared),
                 batch_size,
                 stream);
  recurrent_entry->fn(mutable_data_ptr(conv_out),
                      mutable_data_ptr(qk_prepared),
                      mutable_data_ptr(gate_prepared),
                      mutable_data_ptr(ssm_state),
                      mutable_data_ptr(read_state_indices),
                      mutable_data_ptr(write_state_indices),
                      mutable_data_ptr(num_accepted_tokens),
                      mutable_data_ptr(ssm_state),
                      mutable_data_ptr(readout),
                      batch_size,
                      stream);
  norm_entry->fn(mutable_data_ptr(readout),
                 mutable_data_ptr(z),
                 mutable_data_ptr(norm_weight),
                 mutable_data_ptr(out),
                 batch_size,
                 stream);
  return out;
}

}  // namespace xllm::kernel::npu::tilelang
