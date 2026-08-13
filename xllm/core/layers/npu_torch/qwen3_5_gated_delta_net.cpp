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

#include "qwen3_5_gated_delta_net.h"

#include <glog/logging.h>

#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "core/kernels/ops_api.h"

namespace xllm {
namespace layer {

Qwen3_5GatedDeltaNetImpl::Qwen3_5GatedDeltaNetImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : Qwen3NextGatedDeltaNetImpl(args,
                                 quant_args,
                                 parallel_args,
                                 options,
                                 /*init_projections=*/false) {
  const int64_t qkv_size = (2 * k_size_ + v_size_) / tp_size_;
  const int64_t z_size = v_size_ / tp_size_;
  const int64_t num_heads = num_v_heads_ / tp_size_;
  use_fused_projection_ =
      std::getenv("XLLM_DISABLE_FUSED_QWEN35_PROJECTION") == nullptr &&
      quant_args.quant_method().empty() && quant_args.quant_descs().empty() &&
      !quant_args.is_compressed_tensors_w8a8_dynamic() &&
      xllm::kernel::npu::tilelang::has_qwen35_projection_layout_specialization(
          qkv_size, z_size, num_heads, options.dtype().toScalarType());
  if (use_fused_projection_) {
    in_proj_fused_ = register_module(
        "in_proj_fused",
        ColumnParallelLinear(args.hidden_size(),
                             2 * k_size_ + 2 * v_size_ + 2 * num_v_heads_,
                             /*bias=*/false,
                             /*gather_output=*/false,
                             quant_args,
                             parallel_args.tp_group_,
                             options));
    return;
  }

  in_proj_qkv_ = register_module("in_proj_qkv",
                                 ColumnParallelLinear(args.hidden_size(),
                                                      k_size_ * 2 + v_size_,
                                                      /*bias=*/false,
                                                      /*gather_output=*/false,
                                                      quant_args,
                                                      parallel_args.tp_group_,
                                                      options));
  in_proj_z_ = register_module("in_proj_z",
                               ColumnParallelLinear(args.hidden_size(),
                                                    v_size_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  in_proj_b_ = register_module("in_proj_b",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  in_proj_a_ = register_module("in_proj_a",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Qwen3_5GatedDeltaNetImpl::project_split_activations(
    const torch::Tensor& hidden_states) {
  if (!use_fused_projection_) {
    return {in_proj_qkv_->forward(hidden_states),
            in_proj_z_->forward(hidden_states),
            in_proj_b_->forward(hidden_states),
            in_proj_a_->forward(hidden_states)};
  }

  const std::vector<int64_t> local_sizes = {(2 * k_size_ + v_size_) / tp_size_,
                                            v_size_ / tp_size_,
                                            num_v_heads_ / tp_size_,
                                            num_v_heads_ / tp_size_};
  const auto weight_slices =
      torch::split(in_proj_fused_->weight(), local_sizes, /*dim=*/0);
  CHECK_EQ(weight_slices.size(), local_sizes.size());

  std::vector<torch::Tensor> projections;
  projections.reserve(weight_slices.size());
  for (const auto& weight_slice : weight_slices) {
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = hidden_states;
    matmul_params.b = weight_slice;
    matmul_params.bias = std::nullopt;
    projections.emplace_back(xllm::kernel::matmul(matmul_params));
  }
  return {projections[0], projections[1], projections[2], projections[3]};
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::merge_qkvz_from_split_activations(
    const torch::Tensor& qkv,
    const torch::Tensor& z) const {
  CHECK_EQ(qkv.dim(), 3) << "Expected qkv activation to be 3D, got "
                         << qkv.sizes();
  CHECK_EQ(z.dim(), 3) << "Expected z activation to be 3D, got " << z.sizes();
  CHECK_EQ(qkv.size(0), z.size(0)) << "qkv/z batch size mismatch.";
  CHECK_EQ(qkv.size(1), z.size(1)) << "qkv/z sequence size mismatch.";
  CHECK_EQ(qkv.size(2), (2 * k_size_ + v_size_) / tp_size_)
      << "Unexpected qkv hidden size for Qwen3.5.";
  CHECK_EQ(z.size(2), v_size_ / tp_size_)
      << "Unexpected z hidden size for Qwen3.5.";
  CHECK_GT(num_k_heads_, 0) << "linear_num_key_heads must be positive.";
  CHECK_EQ(num_v_heads_ % num_k_heads_, 0)
      << "linear_num_value_heads must be divisible by linear_num_key_heads.";

  const int64_t bs = qkv.size(0);
  const int64_t seqlen = qkv.size(1);
  const int64_t local_k_heads = num_k_heads_ / tp_size_;
  const int64_t local_v_heads = num_v_heads_ / tp_size_;
  const int64_t num_v_heads_per_k = num_v_heads_ / num_k_heads_;

  auto qkv_split = torch::split(
      qkv, {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_}, 2);
  auto q = qkv_split[0].view({bs, seqlen, local_k_heads, head_k_dim_});
  auto k = qkv_split[1].view({bs, seqlen, local_k_heads, head_k_dim_});
  auto v = qkv_split[2].view({bs, seqlen, local_v_heads, head_v_dim_});
  auto z_view = z.view({bs, seqlen, local_v_heads, head_v_dim_});

  v = v.view({bs, seqlen, local_k_heads, num_v_heads_per_k * head_v_dim_});
  z_view =
      z_view.view({bs, seqlen, local_k_heads, num_v_heads_per_k * head_v_dim_});

  return torch::cat({q, k, v, z_view}, -1).view({bs, seqlen, -1}).contiguous();
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::merge_ba_from_split_activations(
    const torch::Tensor& b,
    const torch::Tensor& a) const {
  CHECK_EQ(b.dim(), 3) << "Expected b activation to be 3D, got " << b.sizes();
  CHECK_EQ(a.dim(), 3) << "Expected a activation to be 3D, got " << a.sizes();
  CHECK_EQ(b.size(0), a.size(0)) << "b/a batch size mismatch.";
  CHECK_EQ(b.size(1), a.size(1)) << "b/a sequence size mismatch.";
  CHECK_EQ(b.size(2), num_v_heads_ / tp_size_)
      << "Unexpected b hidden size for Qwen3.5.";
  CHECK_EQ(a.size(2), num_v_heads_ / tp_size_)
      << "Unexpected a hidden size for Qwen3.5.";
  CHECK_GT(num_k_heads_, 0) << "linear_num_key_heads must be positive.";
  CHECK_EQ(num_v_heads_ % num_k_heads_, 0)
      << "linear_num_value_heads must be divisible by linear_num_key_heads.";

  const int64_t bs = b.size(0);
  const int64_t seqlen = b.size(1);
  const int64_t local_k_heads = num_k_heads_ / tp_size_;
  const int64_t num_v_heads_per_k = num_v_heads_ / num_k_heads_;

  auto b_view = b.view({bs, seqlen, local_k_heads, num_v_heads_per_k});
  auto a_view = a.view({bs, seqlen, local_k_heads, num_v_heads_per_k});
  return torch::cat({b_view, a_view}, -1).view({bs, seqlen, -1}).contiguous();
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3_5GatedDeltaNetImpl::project_decode_inputs(
    const torch::Tensor& hidden_states) {
  const auto reshape_projection = [](const torch::Tensor& projection) {
    return projection.view({projection.size(0), -1, projection.size(-1)});
  };
  auto [qkv_flat, z_flat, b_flat, a_flat] =
      project_split_activations(hidden_states);
  auto qkv = reshape_projection(qkv_flat);
  auto z_proj = reshape_projection(z_flat);
  auto b_proj = reshape_projection(b_flat);
  auto a_proj = reshape_projection(a_flat);
  return {merge_qkvz_from_split_activations(qkv, z_proj),
          merge_ba_from_split_activations(b_proj, a_proj)};
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3_5GatedDeltaNetImpl::project_flat_inputs(
    const torch::Tensor& hidden_states) {
  auto [qkv_flat, z_flat, b_flat, a_flat] =
      project_split_activations(hidden_states);
  auto qkv = qkv_flat.unsqueeze(0);
  auto z_proj = z_flat.unsqueeze(0);
  auto b_proj = b_flat.unsqueeze(0);
  auto a_proj = a_flat.unsqueeze(0);
  auto qkvz = merge_qkvz_from_split_activations(qkv, z_proj);
  auto ba = merge_ba_from_split_activations(b_proj, a_proj);
  return {qkvz.view({hidden_states.size(0), qkvz.size(-1)}).contiguous(),
          ba.view({hidden_states.size(0), ba.size(-1)}).contiguous()};
}

std::optional<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
Qwen3_5GatedDeltaNetImpl::project_prefill_split_inputs(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata) {
  auto [qkv_flat, z_flat, b_flat, a_flat] =
      project_split_activations(hidden_states);
  auto qkv = reshape_projected_tokens_with_pad(attn_metadata, qkv_flat);
  auto z_proj = reshape_projected_tokens_with_pad(attn_metadata, z_flat);
  auto b_proj = reshape_projected_tokens_with_pad(attn_metadata, b_flat);
  auto a_proj = reshape_projected_tokens_with_pad(attn_metadata, a_flat);

  const int64_t batch_size = qkv.size(0);
  const int64_t seq_len = qkv.size(1);
  auto z =
      z_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  auto b = b_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_});
  auto a = a_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_});
  return std::make_tuple(qkv, z, b, a);
}

std::optional<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
Qwen3_5GatedDeltaNetImpl::project_decode_split_inputs(
    const torch::Tensor& hidden_states) {
  if (!use_fused_projection_) {
    return std::nullopt;
  }

  auto projection = in_proj_fused_->forward(hidden_states);
  const int64_t batch_size = projection.size(0);
  const int64_t qkv_size = (2 * k_size_ + v_size_) / tp_size_;
  const int64_t z_size = v_size_ / tp_size_;
  const int64_t num_heads = num_v_heads_ / tp_size_;
  auto projection_flat =
      projection.view({-1, projection.size(-1)}).contiguous();

  torch::Tensor qkv, z, b, a;
  std::tie(qkv, z, b, a) =
      xllm::kernel::npu::tilelang::qwen35_projection_layout(
          projection_flat, qkv_size, z_size, num_heads);
  const int64_t seq_len = qkv.size(0) / batch_size;
  qkv = qkv.view({batch_size, seq_len, qkv_size});
  z = z.view({batch_size, seq_len, num_heads, head_v_dim_});
  b = b.view({batch_size, seq_len, num_heads});
  a = a.view({batch_size, seq_len, num_heads});
  return std::make_tuple(qkv, z, b, a);
}

void Qwen3_5GatedDeltaNetImpl::load_projection_state_dict(
    const StateDict& state_dict) {
  if (use_fused_projection_) {
    std::unordered_map<std::string, torch::Tensor> fused_state_dict;
    auto qkv_weight = state_dict.get_tensor("in_proj_qkv.weight");
    if (qkv_weight.defined()) {
      CHECK_EQ(qkv_weight.dim(), 2)
          << state_dict.prefix() << "in_proj_qkv.weight must be 2D";
      CHECK_EQ(qkv_weight.size(0), 2 * k_size_ + v_size_)
          << state_dict.prefix() << "in_proj_qkv.weight size mismatch";
      auto qkv_weights =
          torch::split(qkv_weight, {k_size_, k_size_, v_size_}, /*dim=*/0);
      fused_state_dict.emplace("q.weight", qkv_weights[0]);
      fused_state_dict.emplace("k.weight", qkv_weights[1]);
      fused_state_dict.emplace("v.weight", qkv_weights[2]);
    }

    const auto add_projection_weight = [&](const std::string& checkpoint_name,
                                           const std::string& fused_name) {
      auto weight = state_dict.get_tensor(checkpoint_name);
      if (weight.defined()) {
        fused_state_dict.emplace(fused_name, weight);
      }
    };
    add_projection_weight("in_proj_z.weight", "z.weight");
    add_projection_weight("in_proj_b.weight", "b.weight");
    add_projection_weight("in_proj_a.weight", "a.weight");

    if (!fused_state_dict.empty()) {
      in_proj_fused_->load_state_dict(
          StateDict(std::move(fused_state_dict),
                    std::string(state_dict.prefix())),
          {"q.", "k.", "v.", "z.", "b.", "a."});
    }
    return;
  }

  auto in_proj_qkv_state_dict = state_dict.get_dict_with_prefix("in_proj_qkv.");
  if (in_proj_qkv_state_dict.size() > 0 && !in_proj_qkv_->is_weight_loaded()) {
    in_proj_qkv_->load_state_dict(
        in_proj_qkv_state_dict,
        /*shard_tensor_count=*/3,
        /*shard_sizes=*/
        {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_});
  }

  auto in_proj_z_state_dict = state_dict.get_dict_with_prefix("in_proj_z.");
  if (in_proj_z_state_dict.size() > 0 && !in_proj_z_->is_weight_loaded()) {
    in_proj_z_->load_state_dict(in_proj_z_state_dict);
  }

  auto in_proj_b_state_dict = state_dict.get_dict_with_prefix("in_proj_b.");
  if (in_proj_b_state_dict.size() > 0 && !in_proj_b_->is_weight_loaded()) {
    in_proj_b_->load_state_dict(in_proj_b_state_dict);
  }

  auto in_proj_a_state_dict = state_dict.get_dict_with_prefix("in_proj_a.");
  if (in_proj_a_state_dict.size() > 0 && !in_proj_a_->is_weight_loaded()) {
    in_proj_a_->load_state_dict(in_proj_a_state_dict);
  }
}

void Qwen3_5GatedDeltaNetImpl::verify_projection_weights(
    const std::string& prefix) const {
  if (use_fused_projection_) {
    CHECK(in_proj_fused_ && in_proj_fused_->is_weight_loaded())
        << "Missing required Qwen3.5 fused projection weights after all shards "
           "loaded: "
        << prefix;
    return;
  }

  CHECK(in_proj_qkv_ && in_proj_qkv_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_qkv.weight";
  CHECK(in_proj_z_ && in_proj_z_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_z.weight";
  CHECK(in_proj_b_ && in_proj_b_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_b.weight";
  CHECK(in_proj_a_ && in_proj_a_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_a.weight";
}

}  // namespace layer
}  // namespace xllm
