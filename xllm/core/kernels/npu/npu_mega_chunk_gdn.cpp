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

#include <cmath>
#include <sstream>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/utils.h"

namespace xllm::kernel::npu {
namespace {
constexpr int64_t kMegaChunkSize = 128;

torch::Tensor make_mask_lower(const torch::Tensor& q) {
  return torch::ones({kMegaChunkSize, kMegaChunkSize}, q.options())
      .to(torch::kFloat32)
      .tril(-1);
}

torch::Tensor make_mask_full(const torch::Tensor& q) {
  return torch::ones({kMegaChunkSize, kMegaChunkSize}, q.options())
      .to(torch::kFloat32)
      .tril(0);
}

torch::Tensor make_minus_identity(const torch::Tensor& q) {
  torch::Tensor minus_identity = torch::zeros(
      {kMegaChunkSize, kMegaChunkSize}, q.options().dtype(torch::kFloat16));
  return minus_identity.fill_diagonal_(-1);
}

torch::Tensor make_empty_state(const torch::Tensor& q,
                               int64_t num_sequences,
                               int64_t num_heads) {
  const int64_t head_dim = q.size(3);
  return torch::zeros({num_sequences, num_heads, head_dim, head_dim},
                      q.options().dtype(torch::kFloat16));
}

std::string tensor_shape(const torch::Tensor& tensor) {
  std::ostringstream oss;
  oss << "[";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << tensor.size(i);
  }
  oss << "]";
  return oss.str();
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> npu_mega_chunk_gdn(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const std::optional<float>& scale,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    const torch::Tensor& cu_seqlens,
    int64_t num_matrices) {
  check_tensor(q, "q", "mega_chunk_gdn");
  check_tensor(k, "k", "mega_chunk_gdn");
  check_tensor(v, "v", "mega_chunk_gdn");
  check_tensor(g, "g", "mega_chunk_gdn");
  check_tensor(beta, "beta", "mega_chunk_gdn");
  check_tensor(cu_seqlens, "cu_seqlens", "mega_chunk_gdn");
  CHECK_EQ(q.dim(), 4) << "mega_chunk_gdn expects q shape [1, T, Hg, D].";
  CHECK_EQ(k.dim(), 4) << "mega_chunk_gdn expects k shape [1, T, Hg, D].";
  CHECK_EQ(v.dim(), 4) << "mega_chunk_gdn expects v shape [1, T, H, D].";
  CHECK_EQ(g.dim(), 3) << "mega_chunk_gdn expects g shape [1, T, H].";
  CHECK_EQ(beta.dim(), 3) << "mega_chunk_gdn expects beta shape [1, T, H].";
  CHECK_EQ(q.size(0), 1) << "mega_chunk_gdn only supports packed B=1 input.";
  CHECK(q.sizes() == k.sizes()) << "mega_chunk_gdn q/k shape mismatch.";
  CHECK_EQ(v.size(0), q.size(0)) << "mega_chunk_gdn v batch mismatch.";
  CHECK_EQ(v.size(1), q.size(1)) << "mega_chunk_gdn v token mismatch.";
  CHECK_EQ(g.size(0), q.size(0)) << "mega_chunk_gdn g batch mismatch.";
  CHECK_EQ(g.size(1), q.size(1)) << "mega_chunk_gdn g token mismatch.";
  CHECK_EQ(beta.sizes(), g.sizes()) << "mega_chunk_gdn beta/g shape mismatch.";
  CHECK_EQ(g.size(2), v.size(2)) << "mega_chunk_gdn g/v head mismatch.";
  CHECK_EQ(q.size(3), kMegaChunkSize)
      << "mega_chunk_gdn expects head dimension 128.";
  CHECK_EQ(v.size(3), kMegaChunkSize)
      << "mega_chunk_gdn expects value head dimension 128.";
  CHECK_GT(num_matrices, 0) << "mega_chunk_gdn num_matrices must be positive.";
  CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2)
      << "mega_chunk_gdn expects 1-D cu_seqlens.";

  const torch::ScalarType input_dtype = q.scalar_type();
  const int64_t total_tokens = q.size(1);
  const int64_t num_heads = v.size(2);
  const int64_t head_dim = q.size(3);
  const int64_t num_sequences = cu_seqlens.numel() - 1;
  const float scale_value = scale.has_value()
                                ? scale.value()
                                : std::pow(static_cast<float>(head_dim), -0.5f);

  torch::Tensor q_half = q.to(torch::kFloat16).contiguous();
  torch::Tensor k_half = k.to(torch::kFloat16).contiguous();
  torch::Tensor v_half = v.to(torch::kFloat16).contiguous();
  torch::Tensor beta_half = beta.to(torch::kFloat16).contiguous();
  torch::Tensor g_float = g.to(torch::kFloat32).contiguous();
  torch::Tensor cu_prepared = cu_seqlens.to(torch::kInt32).contiguous();

  const bool has_initial_state = initial_state.has_value();
  torch::Tensor initial_state_arg =
      has_initial_state ? initial_state.value().to(torch::kFloat16).contiguous()
                        : make_empty_state(q_half, num_sequences, num_heads);

  torch::Tensor mask_lower = make_mask_lower(q_half);
  torch::Tensor mask_full = make_mask_full(q_half);
  torch::Tensor minus_identity = make_minus_identity(q_half);
  torch::Tensor out = torch::empty_like(v_half);
  torch::Tensor g_sum =
      torch::empty(g_float.sizes(), g_float.options().dtype(torch::kFloat32));
  torch::Tensor g_t = torch::empty({num_heads, total_tokens},
                                   g_float.options().dtype(torch::kFloat32));
  torch::Tensor beta_t =
      torch::empty({num_heads, total_tokens}, beta_half.options());
  torch::Tensor a = torch::zeros({1, total_tokens, num_heads, kMegaChunkSize},
                                 beta_half.options());
  torch::Tensor a_inv_f32 =
      torch::zeros({1, total_tokens, num_heads, kMegaChunkSize},
                   g_float.options().dtype(torch::kFloat32));
  torch::Tensor a_inv = torch::zeros_like(a);
  torch::Tensor w = torch::empty_like(v_half);
  torch::Tensor u = torch::empty_like(v_half);
  torch::Tensor h =
      torch::zeros({num_matrices, head_dim, head_dim}, v_half.options());
  torch::Tensor v_new = torch::empty_like(v_half);
  torch::Tensor final_state = torch::zeros(
      {num_sequences * num_heads, head_dim, head_dim}, v_half.options());

  LOG(INFO) << "mega_chunk_gdn aclnn inputs: q=" << tensor_shape(q_half)
            << " k=" << tensor_shape(k_half) << " v=" << tensor_shape(v_half)
            << " g=" << tensor_shape(g_float)
            << " beta=" << tensor_shape(beta_half)
            << " cu_seqlens=" << tensor_shape(cu_prepared)
            << " initial_state=" << tensor_shape(initial_state_arg)
            << " out=" << tensor_shape(out) << " h=" << tensor_shape(h)
            << " final_state=" << tensor_shape(final_state)
            << " num_matrices=" << num_matrices
            << " has_initial_state=" << has_initial_state
            << " output_final_state=" << output_final_state;

  EXEC_NPU_CMD(aclnnMegaChunkGdn,
               q_half,
               k_half,
               v_half,
               g_float,
               beta_half,
               mask_lower,
               mask_full,
               minus_identity,
               cu_prepared,
               initial_state_arg,
               num_matrices,
               has_initial_state,
               out,
               g_sum,
               g_t,
               beta_t,
               a,
               a_inv_f32,
               a_inv,
               w,
               u,
               h,
               v_new,
               final_state);

  torch::Tensor final_state_out;
  if (output_final_state) {
    final_state_out =
        final_state.view({num_sequences, num_heads, head_dim, head_dim})
            .to(torch::kFloat32);
  }
  return {(out * scale_value).to(input_dtype), final_state_out};
}

}  // namespace xllm::kernel::npu
