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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/utils.h"

namespace {

c10::optional<torch::Tensor> to_c10_optional_tensor(
    const std::optional<torch::Tensor>& tensor_opt) {
  if (tensor_opt.has_value() && tensor_opt.value().defined()) {
    return tensor_opt.value();
  }
  return c10::nullopt;
}

c10::optional<torch::IntArrayRef> to_c10_optional_int_array_ref(
    const std::optional<torch::IntArrayRef>& array_opt) {
  if (array_opt.has_value()) {
    return array_opt.value();
  }
  return c10::nullopt;
}

}  // namespace

namespace xllm::kernel::npu {

std::pair<torch::Tensor, torch::Tensor> npu_recompute_w_u_fwd_aclnn(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& beta,
    const torch::Tensor& g,
    const torch::Tensor& A,
    int64_t chunk_size,
    const std::optional<torch::IntArrayRef>& cu_seqlens,
    const std::optional<torch::IntArrayRef>& chunk_indices) {
  check_tensor(k, "k", "npu_recompute_w_u_fwd_aclnn");
  check_tensor(v, "v", "npu_recompute_w_u_fwd_aclnn");
  check_tensor(beta, "beta", "npu_recompute_w_u_fwd_aclnn");
  check_tensor(g, "g", "npu_recompute_w_u_fwd_aclnn");
  check_tensor(A, "A", "npu_recompute_w_u_fwd_aclnn");

  torch::Tensor w_out = torch::empty_like(k);
  torch::Tensor u_out = torch::empty_like(v);
  c10::optional<torch::Tensor> none_tensor = c10::nullopt;
  auto cu_seqlens_ref = to_c10_optional_int_array_ref(cu_seqlens);
  auto chunk_indices_ref = to_c10_optional_int_array_ref(chunk_indices);

  EXEC_NPU_CMD(aclnnRecomputeWUFwd,
               k,
               v,
               beta,
               A,
               g,
               none_tensor,
               cu_seqlens_ref,
               chunk_indices_ref,
               chunk_size,
               w_out,
               u_out);
  return {w_out, u_out};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
npu_chunk_gated_delta_rule_fwd_h_aclnn(
    const torch::Tensor& k,
    const torch::Tensor& w,
    const torch::Tensor& u,
    const torch::Tensor& g,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    int64_t chunk_size,
    const std::optional<torch::IntArrayRef>& cu_seqlens,
    const std::optional<torch::IntArrayRef>& chunk_indices) {
  check_tensor(k, "k", "npu_chunk_gated_delta_rule_fwd_h_aclnn");
  check_tensor(w, "w", "npu_chunk_gated_delta_rule_fwd_h_aclnn");
  check_tensor(u, "u", "npu_chunk_gated_delta_rule_fwd_h_aclnn");
  check_tensor(g, "g", "npu_chunk_gated_delta_rule_fwd_h_aclnn");

  const int64_t batch = k.size(0);
  const int64_t k_num_heads = k.size(1);
  const int64_t seq_len = k.size(2);
  const int64_t k_head_dim = k.size(3);
  const int64_t v_num_heads = u.size(1);
  const int64_t v_head_dim = u.size(3);
  const int64_t num_chunks =
      chunk_indices.has_value()
          ? static_cast<int64_t>(chunk_indices.value().size() / 2)
          : (seq_len + chunk_size - 1) / chunk_size;
  const int64_t state_batch =
      cu_seqlens.has_value()
          ? static_cast<int64_t>(cu_seqlens.value().size() - 1)
          : batch;

  torch::Tensor h_out = torch::zeros(
      {batch, v_num_heads, num_chunks, k_head_dim, v_head_dim}, k.options());
  torch::Tensor v_new_out = torch::empty_like(u);
  torch::Tensor final_state_out =
      output_final_state
          ? torch::empty({state_batch, v_num_heads, k_head_dim, v_head_dim},
                         initial_state.has_value()
                             ? initial_state.value().options()
                             : k.options().dtype(torch::kFloat32))
          : torch::empty({1}, k.options());

  c10::optional<torch::Tensor> initial_state_tensor =
      to_c10_optional_tensor(initial_state);
  c10::optional<torch::Tensor> none_tensor = c10::nullopt;
  auto cu_seqlens_ref = to_c10_optional_int_array_ref(cu_seqlens);
  auto chunk_indices_ref = to_c10_optional_int_array_ref(chunk_indices);
  bool save_new_value = true;
  bool use_exp2 = false;
  bool transpose_state_layout = false;

  EXEC_NPU_CMD(aclnnChunkGatedDeltaRuleFwdH,
               k,
               w,
               u,
               g,
               none_tensor,
               initial_state_tensor,
               output_final_state,
               chunk_size,
               save_new_value,
               cu_seqlens_ref,
               chunk_indices_ref,
               use_exp2,
               transpose_state_layout,
               h_out,
               v_new_out,
               final_state_out);
  if (!output_final_state) {
    final_state_out = torch::Tensor();
  }
  return std::make_tuple(h_out, v_new_out, final_state_out);
}

torch::Tensor npu_chunk_fwd_o_aclnn(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& h,
    const torch::Tensor& g,
    double scale,
    int64_t chunk_size,
    const std::optional<torch::IntArrayRef>& cu_seqlens,
    const std::optional<torch::IntArrayRef>& chunk_indices) {
  check_tensor(q, "q", "npu_chunk_fwd_o_aclnn");
  check_tensor(k, "k", "npu_chunk_fwd_o_aclnn");
  check_tensor(v, "v", "npu_chunk_fwd_o_aclnn");
  check_tensor(h, "h", "npu_chunk_fwd_o_aclnn");
  check_tensor(g, "g", "npu_chunk_fwd_o_aclnn");

  torch::Tensor out = torch::empty_like(v);
  auto cu_seqlens_ref = to_c10_optional_int_array_ref(cu_seqlens);
  auto chunk_indices_ref = to_c10_optional_int_array_ref(chunk_indices);

  EXEC_NPU_CMD(aclnnChunkFwdO,
               q,
               k,
               v,
               h,
               g,
               cu_seqlens_ref,
               chunk_indices_ref,
               scale,
               chunk_size,
               out);
  return out;
}

}  // namespace xllm::kernel::npu
