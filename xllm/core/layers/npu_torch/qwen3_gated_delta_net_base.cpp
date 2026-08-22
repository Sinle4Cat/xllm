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

#include "qwen3_gated_delta_net_base.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <tuple>

#include "framework/kv_cache/kv_cache_utils.h"
#include "layers/npu_torch/qwen3_gated_delta_net_route.h"
#include "xllm/core/kernels/npu/npu_ops_api.h"
#include "xllm/core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "xllm/core/kernels/ops_api.h"
#include "xllm/core/platform/npu/acl_graph_task_update_context.h"

namespace xllm {
namespace layer {

namespace {
constexpr int32_t kMaxMtpDebugLayers = 128;
std::array<std::atomic<int64_t>, kMaxMtpDebugLayers>
    g_mtp_verify_call_counters{};

struct MtpGdnDebugCapture {
  bool enabled = false;
  int64_t call = -1;
  int64_t batch_size = 0;
  int64_t seq_len = 0;
  std::string dump_dir;
  std::string active_positions;
  std::vector<int64_t> selected_rows;
  torch::Tensor row_indices;
  std::vector<std::pair<std::string, torch::Tensor>> tensors;
};

bool debug_index_selected(const char* values,
                          int64_t index,
                          const char* environment_name) {
  CHECK(values != nullptr);
  const char* cursor = values;
  while (*cursor != '\0') {
    char* end = nullptr;
    const int64_t candidate = std::strtoll(cursor, &end, /*base=*/10);
    CHECK(end != cursor && candidate >= 0)
        << environment_name
        << " must be a comma-separated list of non-negative integers";
    if (candidate == index) {
      return true;
    }
    if (*end == '\0') {
      return false;
    }
    CHECK_EQ(*end, ',')
        << environment_name
        << " must be a comma-separated list of non-negative integers";
    cursor = end + 1;
  }
  return false;
}

std::vector<int64_t> parse_debug_rows(const char* values,
                                      int64_t batch_size,
                                      const char* environment_name) {
  CHECK(values != nullptr);
  std::vector<int64_t> rows;
  const char* cursor = values;
  while (*cursor != '\0') {
    char* end = nullptr;
    const int64_t row = std::strtoll(cursor, &end, /*base=*/10);
    CHECK(end != cursor && row >= 0 && row < batch_size)
        << environment_name << " contains an invalid batch row " << row
        << " for batch_size=" << batch_size;
    rows.push_back(row);
    if (*end == '\0') {
      break;
    }
    CHECK_EQ(*end, ',')
        << environment_name
        << " must be a comma-separated list of non-negative integers";
    cursor = end + 1;
  }
  CHECK(!rows.empty()) << environment_name << " must not be empty";
  return rows;
}

torch::Tensor select_mtp_debug_token_rows(const torch::Tensor& tensor,
                                          const torch::Tensor& row_indices,
                                          int64_t batch_size,
                                          int64_t seq_len) {
  CHECK(tensor.defined() && tensor.dim() >= 1);
  CHECK_EQ(tensor.size(0), batch_size * seq_len)
      << "unexpected flattened token dimension for MTP layer capture";
  torch::Tensor step_offsets = torch::arange(seq_len, row_indices.options());
  torch::Tensor token_indices =
      (row_indices.unsqueeze(1) * seq_len + step_offsets.unsqueeze(0))
          .flatten();
  return torch::index_select(tensor, 0, token_indices);
}

MtpGdnDebugCapture prepare_mtp_gdn_debug_capture(
    bool use_spec_verify,
    bool enable_graph,
    int32_t layer_id,
    int64_t rank,
    int64_t seq_len,
    const ModelInputParams& input_params,
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& z,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& conv_cache,
    const torch::Tensor& ssm_cache,
    const torch::Tensor& logical_state_read_indices) {
  MtpGdnDebugCapture capture;
  const char* dump_dir = std::getenv("XLLM_DEBUG_MEGA_GDN_MTP_LAYER_DUMP_DIR");
  const char* dump_call =
      std::getenv("XLLM_DEBUG_MEGA_GDN_MTP_LAYER_DUMP_CALL");
  const char* active_rows =
      std::getenv("XLLM_DEBUG_MEGA_GDN_MTP_LAYER_DUMP_ACTIVE_ROWS");
  if (!use_spec_verify || enable_graph || dump_dir == nullptr ||
      input_params.embedding.request_ids.empty()) {
    return capture;
  }
  CHECK_GE(layer_id, 0);
  CHECK_LT(layer_id, kMaxMtpDebugLayers);

  capture.call = g_mtp_verify_call_counters[layer_id].fetch_add(1);
  const bool selected_by_call =
      dump_call != nullptr &&
      debug_index_selected(
          dump_call, capture.call, "XLLM_DEBUG_MEGA_GDN_MTP_LAYER_DUMP_CALL");
  if (active_rows == nullptr && !selected_by_call) {
    return capture;
  }

  const int64_t batch_size = mixed_qkv.size(0);
  if (active_rows != nullptr) {
    capture.selected_rows =
        parse_debug_rows(active_rows,
                         batch_size,
                         "XLLM_DEBUG_MEGA_GDN_MTP_LAYER_DUMP_ACTIVE_ROWS");
  } else {
    capture.selected_rows.resize(batch_size);
    std::iota(capture.selected_rows.begin(), capture.selected_rows.end(), 0);
  }
  capture.row_indices = torch::tensor(capture.selected_rows,
                                      mixed_qkv.options().dtype(torch::kLong));
  CHECK(input_params.num_accepted_tokens.defined());
  const torch::Tensor selected_read_indices =
      torch::index_select(logical_state_read_indices, 0, capture.row_indices);
  const torch::Tensor selected_accepted_tokens = torch::index_select(
      input_params.num_accepted_tokens, 0, capture.row_indices);
  torch::Tensor read_checkpoint_indices =
      selected_read_indices * seq_len + selected_accepted_tokens - 1;
  capture.enabled = true;
  capture.batch_size = batch_size;
  capture.seq_len = seq_len;
  capture.dump_dir = dump_dir;
  const char* active_positions =
      std::getenv("XLLM_DEBUG_MEGA_GDN_MTP_LAYER_DUMP_ACTIVE_POSITIONS");
  if (active_positions != nullptr) {
    capture.active_positions = active_positions;
  }
  capture.tensors = {
      {"hidden_states", torch::Tensor()},
      {"qkv", torch::index_select(mixed_qkv, 0, capture.row_indices)},
      {"z", torch::index_select(z, 0, capture.row_indices)},
      {"b", torch::index_select(b, 0, capture.row_indices)},
      {"a", torch::index_select(a, 0, capture.row_indices)},
      {"conv_state_read",
       torch::index_select(conv_cache, 0, selected_read_indices)},
      {"ssm_state_read",
       torch::index_select(ssm_cache, 0, read_checkpoint_indices)},
      {"read_state_indices", selected_read_indices},
      {"num_accepted_tokens", selected_accepted_tokens},
  };
  (void)rank;
  return capture;
}

void save_mtp_gdn_debug_capture(MtpGdnDebugCapture& capture,
                                int64_t rank,
                                int32_t layer_id,
                                bool used_mtp_super_op,
                                const std::vector<std::string>& request_ids) {
  if (!capture.enabled) {
    return;
  }
  std::filesystem::path output_dir = std::filesystem::path(capture.dump_dir) /
                                     ("rank_" + std::to_string(rank)) /
                                     ("call_" + std::to_string(capture.call)) /
                                     ("layer_" + std::to_string(layer_id));
  std::error_code error;
  std::filesystem::create_directories(output_dir, error);
  CHECK(!error) << "failed to create GDN layer dump directory " << output_dir
                << ": " << error.message();
  for (const auto& [name, tensor] : capture.tensors) {
    if (tensor.defined()) {
      torch::save(tensor.detach().cpu(),
                  (output_dir / (name + ".pt")).string());
    }
  }
  std::ofstream metadata(output_dir / "metadata.tsv");
  CHECK(metadata) << "failed to write GDN layer dump metadata under "
                  << output_dir;
  metadata << "path\t" << (used_mtp_super_op ? "fused" : "small") << '\n';
  metadata << "call\t" << capture.call << '\n';
  metadata << "layer\t" << layer_id << '\n';
  metadata << "rank\t" << rank << '\n';
  if (!capture.active_positions.empty()) {
    metadata << "capture_base_positions\t" << capture.active_positions << '\n';
  }
  for (int64_t row : capture.selected_rows) {
    CHECK_LT(row, static_cast<int64_t>(request_ids.size()));
    metadata << "request_id\t" << row << '\t' << request_ids[row] << '\n';
  }
}

torch::Tensor l2norm(const torch::Tensor& x, int64_t dim, double eps = 1e-6) {
  auto norm = torch::sqrt(torch::sum(torch::square(x), dim, true) + eps);
  return x / norm;
}

torch::Tensor repeat_tensor_heads(const torch::Tensor& tensor,
                                  int64_t target_heads,
                                  int64_t head_dim) {
  const int64_t current_heads = tensor.size(head_dim);
  if (current_heads == target_heads) {
    return tensor;
  }
  CHECK_GT(current_heads, 0) << "current heads must be positive";
  CHECK_EQ(target_heads % current_heads, 0)
      << "target heads must be divisible by current heads, target_heads="
      << target_heads << ", current_heads=" << current_heads;

  const int64_t repeats = target_heads / current_heads;
  std::vector<int64_t> view_shape = tensor.sizes().vec();
  view_shape.insert(view_shape.begin() + head_dim + 1, 1);
  std::vector<int64_t> expand_shape = view_shape;
  expand_shape[head_dim + 1] = repeats;
  std::vector<int64_t> output_shape = tensor.sizes().vec();
  output_shape[head_dim] = target_heads;
  return tensor.unsqueeze(head_dim + 1)
      .expand(expand_shape)
      .reshape(output_shape)
      .contiguous();
}

std::tuple<torch::Tensor, torch::Tensor> torch_recurrent_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    std::optional<torch::Tensor> initial_state,
    bool output_final_state = true,
    bool use_qk_l2norm_in_kernel = true) {
  auto initial_dtype = query.dtype();

  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, -1, 1e-6);
    key = l2norm(key, -1, 1e-6);
  }

  auto to_float32_and_transpose = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_float32_and_transpose(query);
  key = to_float32_and_transpose(key);
  value = to_float32_and_transpose(value);
  beta = to_float32_and_transpose(beta);
  g = to_float32_and_transpose(g);
  const int64_t value_num_heads = value.size(1);
  query = repeat_tensor_heads(query, value_num_heads, 1);
  key = repeat_tensor_heads(key, value_num_heads, 1);

  int64_t batch_size = key.size(0);
  int64_t num_heads = key.size(1);
  int64_t sequence_length = key.size(2);
  int64_t k_head_dim = key.size(3);
  int64_t v_head_dim = value.size(3);

  float scale_val = 1.0 / std::sqrt(static_cast<float>(query.size(-1)));
  torch::Tensor scale = torch::tensor(scale_val, query.options());
  query = query * scale;
  torch::Tensor core_attn_out = torch::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  torch::Tensor last_recurrent_state;
  if (!initial_state.has_value()) {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  } else {
    last_recurrent_state =
        initial_state.value().to(value.device(), torch::kFloat32);
  }

  for (int64_t i = 0; i < sequence_length; ++i) {
    torch::Tensor q_t = query.select(2, i);
    torch::Tensor k_t = key.select(2, i);
    torch::Tensor v_t = value.select(2, i);
    torch::Tensor g_t = g.select(2, i).exp().unsqueeze(-1).unsqueeze(-1);
    torch::Tensor beta_t = beta.select(2, i).unsqueeze(-1);
    last_recurrent_state = last_recurrent_state * g_t;
    torch::Tensor kv_mem =
        torch::sum(last_recurrent_state * k_t.unsqueeze(-1), -2);
    torch::Tensor delta = (v_t - kv_mem) * beta_t;
    last_recurrent_state =
        last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
    core_attn_out.select(2, i) =
        torch::sum(last_recurrent_state * q_t.unsqueeze(-1), -2);
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core_attn_out, last_recurrent_state);
}

std::tuple<torch::Tensor, torch::Tensor> torch_chunk_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    int64_t chunk_size = 64,
    c10::optional<torch::Tensor> initial_state = c10::nullopt,
    bool output_final_state = true,
    bool use_qk_l2norm_in_kernel = true) {
  auto initial_dtype = query.dtype();
  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, -1, 1e-6);
    key = l2norm(key, -1, 1e-6);
  }
  auto to_float32 = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };

  query = to_float32(query);
  key = to_float32(key);
  value = to_float32(value);
  beta = to_float32(beta);
  g = to_float32(g);
  const int64_t value_num_heads = value.size(1);
  query = repeat_tensor_heads(query, value_num_heads, 1);
  key = repeat_tensor_heads(key, value_num_heads, 1);

  int64_t batch_size = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t sequence_length = query.size(2);
  int64_t k_head_dim = key.size(-1);
  int64_t v_head_dim = value.size(-1);

  int64_t pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
  query = torch::nn::functional::pad(
      query, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  key = torch::nn::functional::pad(
      key, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  value = torch::nn::functional::pad(
      value, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  beta = torch::nn::functional::pad(
      beta, torch::nn::functional::PadFuncOptions({0, pad_size}));
  g = torch::nn::functional::pad(
      g, torch::nn::functional::PadFuncOptions({0, pad_size}));

  int64_t total_sequence_length = sequence_length + pad_size;
  float scale = 1.0 / std::sqrt(static_cast<float>(query.size(-1)));
  query = query * scale;
  auto v_beta = value * beta.unsqueeze(-1);
  auto k_beta = key * beta.unsqueeze(-1);
  auto reshape_to_chunks = [chunk_size](torch::Tensor x) {
    auto shape = x.sizes();
    std::vector<int64_t> new_shape = {
        shape[0], shape[1], shape[2] / chunk_size, chunk_size, shape[3]};
    return x.reshape(new_shape);
  };

  query = reshape_to_chunks(query);
  key = reshape_to_chunks(key);
  value = reshape_to_chunks(value);
  k_beta = reshape_to_chunks(k_beta);
  v_beta = reshape_to_chunks(v_beta);

  auto g_shape = g.sizes();
  std::vector<int64_t> g_new_shape = {
      g_shape[0], g_shape[1], g_shape[2] / chunk_size, chunk_size};
  g = g.reshape(g_new_shape);
  auto mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      0);

  g = g.cumsum(-1);
  auto g_diff = g.unsqueeze(-1) - g.unsqueeze(-2);
  auto decay_mask = g_diff.tril().exp().to(torch::kFloat32);
  decay_mask = decay_mask.tril();
  auto attn = -(torch::matmul(k_beta, key.transpose(-1, -2)) * decay_mask)
                   .masked_fill(mask, 0.0);
  for (int64_t i = 1; i < chunk_size; ++i) {
    if (!attn.is_contiguous()) {
      attn = attn.contiguous();
    }
    auto row = attn.slice(-2, i, i + 1)
                   .slice(-1, 0, i)
                   .squeeze(-2)
                   .clone()
                   .contiguous();
    auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone().contiguous();
    auto row_unsq = row.unsqueeze(-1).contiguous();
    auto row_sub_mul = (row_unsq * sub).contiguous();
    auto row_sub_sum = row_sub_mul.sum(-2).contiguous();
    auto row_final = (row + row_sub_sum).contiguous();
    attn.index_put_({torch::indexing::Ellipsis,
                     torch::indexing::Slice(i, i + 1),
                     torch::indexing::Slice(0, i)},
                    row_final.unsqueeze(-2));
  }

  attn = attn +
         torch::eye(
             chunk_size,
             torch::TensorOptions().dtype(attn.dtype()).device(attn.device()));
  value = torch::matmul(attn, v_beta);
  auto k_cumdecay = torch::matmul(attn, (k_beta * g.exp().unsqueeze(-1)));
  torch::Tensor last_recurrent_state;
  if (!initial_state.has_value()) {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(value.dtype()).device(value.device()));
  } else {
    last_recurrent_state = initial_state.value().to(value);
  }
  auto core_attn_out = torch::zeros_like(value);
  mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      1);
  int64_t num_chunks = total_sequence_length / chunk_size;
  for (int64_t i = 0; i < num_chunks; ++i) {
    auto q_i = query.select(2, i);
    auto k_i = key.select(2, i);
    auto v_i = value.select(2, i);
    auto attn_i =
        (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_mask.select(2, i))
            .masked_fill_(mask, 0.0);
    auto v_prime = torch::matmul(k_cumdecay.select(2, i), last_recurrent_state);
    auto v_new = v_i - v_prime;
    auto attn_inter = torch::matmul(q_i * g.select(2, i).unsqueeze(-1).exp(),
                                    last_recurrent_state);
    core_attn_out.select(2, i) = attn_inter + torch::matmul(attn_i, v_new);
    auto g_i_last = g.select(2, i).select(-1, -1).unsqueeze(-1);
    auto g_exp_term = (g_i_last - g.select(2, i)).exp().unsqueeze(-1);
    auto k_g_exp = (k_i * g_exp_term).transpose(-1, -2).contiguous();
    last_recurrent_state = last_recurrent_state * g_i_last.unsqueeze(-1).exp() +
                           torch::matmul(k_g_exp, v_new);
  }
  auto core_attn_out_shape = core_attn_out.sizes();
  std::vector<int64_t> reshape_shape = {
      core_attn_out_shape[0],
      core_attn_out_shape[1],
      core_attn_out_shape[2] * core_attn_out_shape[3],
      core_attn_out_shape[4]};
  core_attn_out = core_attn_out.reshape(reshape_shape);
  core_attn_out = core_attn_out.slice(2, 0, sequence_length);
  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core_attn_out, last_recurrent_state);
}

int64_t get_checkpoint_stride(const torch::Tensor& conv_cache,
                              const torch::Tensor& ssm_cache) {
  if (!conv_cache.defined() || !ssm_cache.defined() ||
      conv_cache.numel() == 0 || ssm_cache.numel() == 0) {
    return 1;
  }
  CHECK_GT(conv_cache.size(0), 0) << "conv cache must have positive batch dim";
  CHECK_EQ(ssm_cache.size(0) % conv_cache.size(0), 0)
      << "ssm cache checkpoint layout mismatch, ssm_rows=" << ssm_cache.size(0)
      << ", conv_rows=" << conv_cache.size(0);
  return ssm_cache.size(0) / conv_cache.size(0);
}

torch::Tensor build_linear_state_base_indices(
    const torch::Tensor& logical_state_indices,
    int64_t checkpoint_stride) {
  if (checkpoint_stride == 1) {
    return logical_state_indices;
  }
  return logical_state_indices * checkpoint_stride;
}

torch::Tensor expand_sequence_tensor_to_batch(const torch::Tensor& tensor,
                                              int64_t target_batch,
                                              const char* tensor_name) {
  CHECK(tensor.defined()) << tensor_name << " must be defined";
  CHECK_EQ(tensor.dim(), 1) << tensor_name << " must be a 1D tensor.";
  const int64_t source_batch = tensor.size(0);
  if (source_batch == target_batch) {
    return tensor.contiguous();
  }
  CHECK_GT(source_batch, 0) << tensor_name << " must not be empty.";
  CHECK_EQ(target_batch % source_batch, 0)
      << tensor_name << " cannot be expanded from " << source_batch << " to "
      << target_batch;
  const int64_t repeat_count = target_batch / source_batch;
  return tensor.unsqueeze(1)
      .expand({source_batch, repeat_count})
      .reshape({target_batch})
      .contiguous();
}

torch::Tensor resolve_linear_state_indices(
    const std::vector<int32_t>& state_ids,
    const torch::Tensor& state_indices,
    const torch::Tensor& fallback_indices,
    const torch::Device& device) {
  if (!state_indices.defined()) {
    if (state_ids.empty()) {
      return fallback_indices;
    }
    return torch::tensor(
        state_ids, torch::TensorOptions().dtype(torch::kInt).device(device));
  }
  torch::Tensor indices = state_indices;
  if (indices.device() != device || indices.scalar_type() != torch::kInt) {
    indices =
        indices.to(torch::TensorOptions().dtype(torch::kInt).device(device),
                   /*non_blocking=*/true,
                   /*copy=*/true);
  }
  return indices.contiguous();
}

torch::Tensor run_causal_conv1d_graph_update(
    const std::shared_ptr<xllm::npu::AclGraphTaskUpdateContext>& graph_context,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_state,
    const std::optional<torch::Tensor>& bias,
    const std::vector<int64_t>& query_start_loc,
    const std::vector<int64_t>& cache_indices,
    const std::vector<int64_t>& num_accepted_tokens,
    xllm::npu::CausalConv1dGraphBranch branch) {
  CHECK(graph_context != nullptr && graph_context->capturing)
      << "causal_conv1d graph update can only be registered during capture";

  c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
  auto event = std::make_shared<c10_npu::NPUEvent>(ACL_EVENT_EXTERNAL);
  event->block(stream);
  event->reset(stream);

  torch::Tensor output;
  c10_npu::graph_task_group_begin(stream);
  const std::vector<int64_t> empty_host_args;
  CHECK(!query_start_loc.empty())
      << "query_start_loc must be populated for causal_conv1d graph update";
  CHECK_EQ(query_start_loc.back(), x.size(0))
      << "query_start_loc must be padded to x.shape[0] during graph capture";
  CHECK_EQ(cache_indices.size() + 1, query_start_loc.size())
      << "cache_indices must be sequence-scoped";
  if (branch == xllm::npu::CausalConv1dGraphBranch::kSpecVerify) {
    CHECK_EQ(num_accepted_tokens.size(), cache_indices.size())
        << "num_accepted_tokens must be sequence-scoped for spec verify";
  }

  output = torch::empty_like(x);
  xllm::kernel::causal_conv1d_out(output,
                                  x,
                                  weight,
                                  conv_state,
                                  bias,
                                  torch::IntArrayRef(query_start_loc),
                                  torch::IntArrayRef(cache_indices),
                                  torch::IntArrayRef(empty_host_args),
                                  torch::IntArrayRef(num_accepted_tokens),
                                  xllm::npu::kCausalConv1dActivationSilu,
                                  xllm::npu::kCausalConv1dGraphPadSlotId,
                                  xllm::npu::kCausalConv1dRunModeUpdate);
  c10_npu::NPUTaskGroupHandle handle = c10_npu::graph_task_group_end(stream);

  xllm::npu::CausalConv1dGraphTask task;
  task.output = output;
  task.x = x;
  task.weight = weight;
  task.conv_state = conv_state;
  task.bias = bias;
  task.activation_mode = xllm::npu::kCausalConv1dActivationSilu;
  task.pad_slot_id = xllm::npu::kCausalConv1dGraphPadSlotId;
  task.run_mode = xllm::npu::kCausalConv1dRunModeUpdate;
  task.branch = branch;
  task.handle = handle;
  task.event = std::move(event);
  graph_context->causal_conv1d_tasks.emplace_back(std::move(task));
  return output;
}

torch::Tensor run_spec_verify_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor& ssm_cache,
    const torch::Tensor& checkpoint_indices,
    const torch::Tensor& num_accepted_tokens,
    const torch::Tensor& cu_seq_lens,
    const std::vector<int32_t>& q_seq_lens_vec,
    double scale) {
  const auto device = value.device();
  const int64_t batch_size = value.size(0);
  const int64_t seq_len = value.size(1);
  const int64_t total_seq_len = batch_size * seq_len;
  CHECK_EQ(cu_seq_lens.numel(), batch_size + 1)
      << "GDN spec verify cu_seq_lens must be cumulative.";
  CHECK_EQ(q_seq_lens_vec.size(), static_cast<size_t>(batch_size))
      << "GDN spec verify q_seq_lens_vec must be per sequence.";
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    CHECK_EQ(q_seq_lens_vec[batch_idx], seq_len)
        << "Qwen3.5 spec verify fused recurrent path expects dense "
           "same-length validate tokens.";
  }

  xllm::kernel::FusedRecurrentGatedDeltaRuleParams params;
  params.q = query.reshape({1, total_seq_len, query.size(-2), query.size(-1)})
                 .contiguous();
  params.k =
      key.reshape({1, total_seq_len, key.size(-2), key.size(-1)}).contiguous();
  params.v = value.reshape({1, total_seq_len, value.size(-2), value.size(-1)})
                 .contiguous();
  params.g = g.to(torch::kFloat32)
                 .reshape({1, total_seq_len, g.size(-1)})
                 .contiguous();
  params.beta = beta.reshape({1, total_seq_len, beta.size(-1)}).contiguous();
  params.scale = static_cast<float>(scale);
  params.initial_state = ssm_cache;
  params.inplace_final_state = true;
  params.cu_seqlens = cu_seq_lens.to(torch::kLong).contiguous();
  params.ssm_state_indices = checkpoint_indices.contiguous();
  params.num_accepted_tokens =
      num_accepted_tokens.to(device, torch::kInt32).contiguous();
  params.use_qk_l2norm_in_kernel = true;

  auto output_and_state =
      xllm::kernel::fused_recurrent_gated_delta_rule(params);
  return output_and_state.first.view(
      {batch_size, seq_len, value.size(-2), value.size(-1)});
}

}  // namespace

Qwen3GatedDeltaNetBaseImpl::Qwen3GatedDeltaNetBaseImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  tp_size_ = parallel_args.tp_group_->world_size();
  rank_ = parallel_args.tp_group_->rank();
  num_k_heads_ = args.linear_num_key_heads();
  num_v_heads_ = args.linear_num_value_heads();
  head_k_dim_ = args.linear_key_head_dim();
  head_v_dim_ = args.linear_value_head_dim();
  k_size_ = num_k_heads_ * head_k_dim_;
  v_size_ = num_v_heads_ * head_v_dim_;
  conv_kernel_size_ = args.linear_conv_kernel_dim();
  const int64_t local_conv_dim = (k_size_ * 2 + v_size_) / tp_size_;
  conv1d_zero_bias_ = torch::zeros({local_conv_dim}, options);

  // Shared causal conv projection over mixed QKV states.
  conv1d_ = register_module("conv1d",
                            ColumnParallelLinear(args.linear_conv_kernel_dim(),
                                                 k_size_ * 2 + v_size_,
                                                 /*bias=*/false,
                                                 /*gather_output=*/false,
                                                 quant_args,
                                                 parallel_args.tp_group_,
                                                 options));

  auto opts = options.dtype(torch::kFloat32);
  dt_bias_ = register_parameter("dt_bias",
                                torch::ones({num_v_heads_ / tp_size_}, opts),
                                /*requires_grad=*/false);

  A_log_ = register_parameter("A_log",
                              torch::empty({num_v_heads_ / tp_size_}, opts),
                              /*requires_grad=*/false);

  // Output projection and gated RMSNorm shared by hybrid variants.
  o_proj_ = register_module("out_proj",
                            RowParallelLinear(v_size_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));

  norm_ = register_module(
      "norm", RmsNormGated(head_v_dim_, args.rms_norm_eps(), options));
}

void Qwen3GatedDeltaNetBaseImpl::load_common_state_dict(
    const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = tp_size_;
  const int32_t shard_tensor_count = 3;
  const std::vector<int64_t> shard_sizes = {
      k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};

  if (auto w = state_dict.get_tensor("conv1d.weight"); w.defined()) {
    conv1d_->load_state_dict(
        StateDict({{"weight", w.squeeze(1)}},
                  static_cast<std::string>(state_dict.prefix()) + "conv1d."),
        shard_tensor_count,
        shard_sizes);
    conv1d_->weight().set_(conv1d_->weight().transpose(0, 1).contiguous());
  }
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
  if (auto w = state_dict.get_tensor("norm.weight"); w.defined()) {
    norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  LOAD_SHARDED_WEIGHT(dt_bias, 0);
  LOAD_SHARDED_WEIGHT(A_log, 0);
}

void Qwen3GatedDeltaNetBaseImpl::verify_common_loaded_weights(
    const std::string& prefix) const {
  CHECK(dt_bias_is_loaded_)
      << "Missing required weight after all shards loaded: " << prefix
      << "dt_bias";
  CHECK(A_log_is_loaded_) << "Missing required weight after all shards loaded: "
                          << prefix << "A_log";
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3GatedDeltaNetBaseImpl::project_padded_inputs(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata) {
  if (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill) {
    auto [qkvz_flat, ba_flat] = project_flat_inputs(hidden_states);
    return {reshape_projected_tokens_with_pad(attn_metadata, qkvz_flat),
            reshape_projected_tokens_with_pad(attn_metadata, ba_flat)};
  }
  return project_decode_inputs(hidden_states);
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // Dummy shards do not own valid GDN state and must not enter collectives.
  if (attn_metadata.is_dummy) {
    return torch::zeros_like(hidden_states);
  }
  const FlashComm1Context* fc1_ctx = get_current_flash_comm1_context();
  torch::Tensor h = hidden_states;
  if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    h = gather_sequence(hidden_states, *fc1_ctx);
  }

  const int64_t original_num_tokens = h.size(0);
  const bool use_spec_verify = input_params.is_spec_verify;
  const bool is_any_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  if (use_spec_verify && attn_metadata.is_chunked_prefill) {
    CHECK(detail::is_dense_mega_gdn_mtp_verify_batch(
        original_num_tokens,
        attn_metadata.max_query_len,
        attn_metadata.q_seq_lens_vec))
        << "MegaGdnMtpDecode requires a dense mixed verify batch: "
        << "total_tokens=" << original_num_tokens
        << ", max_query_len=" << attn_metadata.max_query_len
        << ", sequence_count=" << attn_metadata.q_seq_lens_vec.size();
  }
  torch::Tensor mixed_qkv, z, b, a;
  torch::Tensor processed_q, processed_k, processed_v;
  int64_t batch_size = 0;
  int64_t seq_len = 0;

  std::optional<
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
      projected_split_inputs;
  if (!use_spec_verify) {
    projected_split_inputs =
        is_any_prefill ? project_prefill_split_inputs(h, attn_metadata)
                       : project_decode_split_inputs(h);
  }
  if (projected_split_inputs.has_value()) {
    std::tie(mixed_qkv, z, b, a) = projected_split_inputs.value();
    batch_size = mixed_qkv.size(0);
    seq_len = mixed_qkv.size(1);
  } else {
    auto [qkvz_padded, ba_padded] = project_padded_inputs(h, attn_metadata);
    batch_size = qkvz_padded.size(0);
    seq_len = qkvz_padded.size(1);

    torch::Tensor qkvz_flat =
        qkvz_padded.view({batch_size * seq_len, qkvz_padded.size(-1)});
    torch::Tensor ba_flat =
        ba_padded.view({batch_size * seq_len, ba_padded.size(-1)});
    xllm::kernel::FusedQkvzbaSplitReshapeParams fused_params;
    fused_params.mixed_qkvz = qkvz_flat;
    fused_params.mixed_ba = ba_flat;
    fused_params.num_heads_qk = static_cast<int32_t>(num_k_heads_ / tp_size_);
    fused_params.num_heads_v = static_cast<int32_t>(num_v_heads_ / tp_size_);
    fused_params.head_qk = static_cast<int32_t>(head_k_dim_);
    fused_params.head_v = static_cast<int32_t>(head_v_dim_);

    std::tie(mixed_qkv, z, b, a) =
        xllm::kernel::fused_qkvzba_split_reshape_cat(fused_params);

    mixed_qkv = mixed_qkv.view({batch_size, seq_len, mixed_qkv.size(-1)});
    z = z.view({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
    b = b.view({batch_size, seq_len, num_v_heads_ / tp_size_});
    a = a.view({batch_size, seq_len, num_v_heads_ / tp_size_});
  }
  const bool fla_ssm_state_layout = use_fla_ssm_state_layout();
  const bool use_direct_prefill_qkv =
      std::getenv("XLLM_DISABLE_CAUSAL_CONV1D_DIRECT_QKV") == nullptr &&
      !use_spec_verify && attn_metadata.is_prefill &&
      !attn_metadata.is_chunked_prefill && batch_size == 1 &&
      attn_metadata.q_seq_lens_vec.size() == 1 &&
      attn_metadata.q_seq_lens_vec[0] == seq_len && fla_ssm_state_layout &&
      num_k_heads_ / tp_size_ == 8 && num_v_heads_ / tp_size_ == 24 &&
      head_k_dim_ == 128 && head_v_dim_ == 128;
  bool used_direct_prefill_qkv = false;

  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  const bool is_fresh_prefill =
      is_any_prefill && !input_params.linear_state_validity_mask.empty() &&
      std::all_of(
          input_params.linear_state_validity_mask.begin(),
          input_params.linear_state_validity_mask.end(),
          [](int64_t has_initial_state) { return has_initial_state == 0; });
  if (is_fresh_prefill) {
    check_linear_attention_kv_cache_guards(
        conv_cache, ssm_cache, "fresh_prefill_entry");
  }
  torch::Device device = mixed_qkv.device();
  torch::Tensor conv_weight = conv1d_->weight();
  torch::Tensor logical_state_indices =
      get_linear_state_indices(input_params, device);
  torch::Tensor logical_state_read_indices = resolve_linear_state_indices(
      input_params.embedding.linear_state_read_ids,
      input_params.embedding.linear_state_read_indices,
      logical_state_indices,
      device);
  torch::Tensor logical_state_write_indices = resolve_linear_state_indices(
      input_params.embedding.linear_state_write_ids,
      input_params.embedding.linear_state_write_indices,
      logical_state_indices,
      device);
  const int64_t checkpoint_stride =
      get_checkpoint_stride(conv_cache, ssm_cache);
  const int64_t num_state_slots =
      conv_cache.defined() && conv_cache.dim() > 0 ? conv_cache.size(0) : 0;
  const bool is_supported_mtp_verify =
      use_spec_verify && seq_len > 1 &&
      detail::is_supported_mega_gdn_mtp_k(seq_len - 1);
  if (is_supported_mtp_verify && !input_params.enable_graph) {
    CHECK(detail::has_valid_mega_gdn_mtp_state_metadata(
        batch_size,
        seq_len,
        num_state_slots,
        attn_metadata.q_seq_lens_vec,
        input_params.embedding.linear_state_ids,
        input_params.embedding.linear_state_read_ids,
        input_params.embedding.linear_state_write_ids,
        input_params.num_accepted_tokens_host))
        << "Invalid MegaGdnMtpDecode state metadata";
  }
  const bool has_forked_mtp_state =
      use_spec_verify && seq_len > 1 &&
      detail::has_forked_mega_gdn_mtp_state(
          input_params.embedding.linear_state_ids,
          input_params.embedding.linear_state_read_ids,
          input_params.embedding.linear_state_write_ids);
  MtpGdnDebugCapture mtp_gdn_debug =
      prepare_mtp_gdn_debug_capture(use_spec_verify,
                                    input_params.enable_graph,
                                    layer_id_,
                                    rank_,
                                    seq_len,
                                    input_params,
                                    mixed_qkv,
                                    z,
                                    b,
                                    a,
                                    conv_cache,
                                    ssm_cache,
                                    logical_state_read_indices);
  if (mtp_gdn_debug.enabled) {
    mtp_gdn_debug.tensors.front().second =
        select_mtp_debug_token_rows(hidden_states,
                                    mtp_gdn_debug.row_indices,
                                    mtp_gdn_debug.batch_size,
                                    mtp_gdn_debug.seq_len);
    mtp_gdn_debug.tensors.emplace_back(
        "write_state_indices",
        torch::index_select(
            logical_state_write_indices, 0, mtp_gdn_debug.row_indices));
  }
  torch::Tensor linear_state_base_indices =
      build_linear_state_base_indices(logical_state_indices, checkpoint_stride);
  auto graph_context = input_params.graph.acl_graph_task_update_context;
  const bool register_conv1d_graph_update =
      graph_context != nullptr && graph_context->capturing;

  torch::Tensor fused_decode_norm_out;
  bool used_decode_super_op = false;
  const auto& norm_weight = norm_->weight();
  const int64_t local_num_k_heads = num_k_heads_ / tp_size_;
  const int64_t local_num_v_heads = num_v_heads_ / tp_size_;
  const int64_t mega_gdn_conv_dim =
      2 * local_num_k_heads * head_k_dim_ + local_num_v_heads * head_v_dim_;
  if (std::getenv("XLLM_DISABLE_QWEN35_GDN_DECODE_SUPER_OP") == nullptr &&
      !use_spec_verify && !is_any_prefill && checkpoint_stride == 1 &&
      batch_size >= 1 && batch_size <= 4 && seq_len == 1 &&
      detail::is_supported_mega_gdn_head_geometry(
          local_num_k_heads, local_num_v_heads, head_k_dim_, head_v_dim_) &&
      std::abs(norm_->eps() - 1e-6) < 1e-12 &&
      input_params.embedding.linear_state_ids.size() == batch_size &&
      std::all_of(input_params.embedding.linear_state_ids.begin(),
                  input_params.embedding.linear_state_ids.end(),
                  [](int64_t state_id) { return state_id >= 0; }) &&
      mixed_qkv.dim() == 3 && mixed_qkv.size(0) == batch_size &&
      mixed_qkv.size(1) == 1 && mixed_qkv.size(2) == mega_gdn_conv_dim &&
      z.dim() == 4 && z.size(0) == batch_size && z.size(1) == 1 &&
      z.size(2) == local_num_v_heads && z.size(3) == head_v_dim_ &&
      a.dim() == 3 && b.dim() == 3 && a.size(0) == batch_size &&
      b.size(0) == batch_size && a.size(1) == 1 && b.size(1) == 1 &&
      a.size(2) == local_num_v_heads && b.size(2) == local_num_v_heads &&
      mixed_qkv.scalar_type() == torch::kBFloat16 &&
      z.scalar_type() == torch::kBFloat16 &&
      a.scalar_type() == torch::kBFloat16 &&
      b.scalar_type() == torch::kBFloat16 && conv_weight.dim() == 2 &&
      conv_weight.size(0) == 4 && conv_weight.size(1) == mega_gdn_conv_dim &&
      conv_weight.scalar_type() == torch::kBFloat16 && conv_cache.dim() == 3 &&
      conv_cache.size(1) == 3 && conv_cache.size(2) == mega_gdn_conv_dim &&
      conv_cache.scalar_type() == torch::kBFloat16 && ssm_cache.dim() == 4 &&
      ssm_cache.size(1) == local_num_v_heads &&
      ssm_cache.size(2) == head_v_dim_ && ssm_cache.size(3) == head_k_dim_ &&
      ssm_cache.scalar_type() == torch::kFloat32 &&
      A_log_.numel() == local_num_v_heads &&
      A_log_.scalar_type() == torch::kFloat32 &&
      dt_bias_.numel() == local_num_v_heads &&
      dt_bias_.scalar_type() == torch::kFloat32 &&
      logical_state_read_indices.dim() == 1 &&
      logical_state_read_indices.numel() == batch_size &&
      logical_state_read_indices.scalar_type() == torch::kInt32 &&
      logical_state_write_indices.dim() == 1 &&
      logical_state_write_indices.numel() == batch_size &&
      logical_state_write_indices.scalar_type() == torch::kInt32 &&
      norm_weight.numel() == head_v_dim_ &&
      norm_weight.scalar_type() == torch::kBFloat16 &&
      mixed_qkv.is_contiguous() && z.is_contiguous() && a.is_contiguous() &&
      b.is_contiguous() && conv_weight.is_contiguous() &&
      conv_cache.is_contiguous() && ssm_cache.is_contiguous() &&
      A_log_.is_contiguous() && dt_bias_.is_contiguous() &&
      logical_state_read_indices.is_contiguous() &&
      logical_state_write_indices.is_contiguous() &&
      norm_weight.is_contiguous()) {
    fused_decode_norm_out = xllm::kernel::npu::mega_gdn_decode(
        mixed_qkv.view({batch_size, mega_gdn_conv_dim}),
        z.view({batch_size, local_num_v_heads, head_v_dim_}),
        b.view({batch_size, local_num_v_heads}),
        a.view({batch_size, local_num_v_heads}),
        conv_weight,
        conv_cache,
        A_log_,
        dt_bias_,
        ssm_cache,
        logical_state_read_indices,
        logical_state_write_indices,
        norm_weight,
        fla_ssm_state_layout);
    fused_decode_norm_out = fused_decode_norm_out.view(
        {batch_size, 1, local_num_v_heads, head_v_dim_});
    used_decode_super_op = true;
  }

  torch::Tensor fused_draft_norm_out;
  bool used_draft_super_op = false;
  const int64_t draft_batch_size =
      static_cast<int64_t>(input_params.mtp_draft_q_seq_lens_host.size());
  const torch::Tensor& draft_q_cu_seq_lens =
      input_params.mtp_draft_q_cu_seq_lens;
  torch::Tensor draft_state_validity_mask =
      input_params.embedding.linear_state_validity_mask;
  if (!draft_state_validity_mask.defined() &&
      attn_metadata.has_initial_states.defined()) {
    draft_state_validity_mask = attn_metadata.has_initial_states;
  }
  if (!used_decode_super_op &&
      std::getenv("XLLM_DISABLE_MEGA_GDN_DRAFT_DECODE") == nullptr &&
      detail::can_use_mega_gdn_draft_decode(
          input_params.is_mtp_draft,
          use_spec_verify,
          original_num_tokens,
          num_state_slots,
          input_params.mtp_draft_q_seq_lens_host,
          input_params.embedding.linear_state_ids,
          input_params.embedding.linear_state_read_ids,
          input_params.embedding.linear_state_write_ids,
          input_params.linear_state_validity_mask) &&
      checkpoint_stride == 1 && draft_batch_size >= 1 &&
      draft_batch_size <= 32 &&
      detail::is_supported_mega_gdn_head_geometry(
          local_num_k_heads, local_num_v_heads, head_k_dim_, head_v_dim_) &&
      std::abs(norm_->eps() - 1e-6) < 1e-12 && mixed_qkv.dim() == 3 &&
      mixed_qkv.size(0) == batch_size && mixed_qkv.size(1) == seq_len &&
      mixed_qkv.size(2) == mega_gdn_conv_dim && z.dim() == 4 &&
      z.size(0) == batch_size && z.size(1) == seq_len &&
      z.size(2) == local_num_v_heads && z.size(3) == head_v_dim_ &&
      a.dim() == 3 && b.dim() == 3 && a.sizes() == b.sizes() &&
      a.size(0) == batch_size && a.size(1) == seq_len &&
      a.size(2) == local_num_v_heads &&
      mixed_qkv.scalar_type() == torch::kBFloat16 &&
      z.scalar_type() == torch::kBFloat16 &&
      a.scalar_type() == torch::kBFloat16 &&
      b.scalar_type() == torch::kBFloat16 && conv_weight.dim() == 2 &&
      conv_weight.size(0) == 4 && conv_weight.size(1) == mega_gdn_conv_dim &&
      conv_weight.scalar_type() == torch::kBFloat16 && conv_cache.dim() == 3 &&
      conv_cache.size(1) == 3 && conv_cache.size(2) == mega_gdn_conv_dim &&
      conv_cache.scalar_type() == torch::kBFloat16 && ssm_cache.dim() == 4 &&
      ssm_cache.size(0) == num_state_slots &&
      ssm_cache.size(1) == local_num_v_heads &&
      ssm_cache.size(2) == head_v_dim_ && ssm_cache.size(3) == head_k_dim_ &&
      ssm_cache.scalar_type() == torch::kFloat32 &&
      A_log_.numel() == local_num_v_heads &&
      A_log_.scalar_type() == torch::kFloat32 &&
      dt_bias_.numel() == local_num_v_heads &&
      dt_bias_.scalar_type() == torch::kFloat32 &&
      logical_state_read_indices.dim() == 1 &&
      logical_state_read_indices.numel() == draft_batch_size &&
      logical_state_read_indices.scalar_type() == torch::kInt32 &&
      logical_state_write_indices.dim() == 1 &&
      logical_state_write_indices.numel() == draft_batch_size &&
      logical_state_write_indices.scalar_type() == torch::kInt32 &&
      draft_q_cu_seq_lens.defined() && draft_q_cu_seq_lens.dim() == 1 &&
      draft_q_cu_seq_lens.numel() == draft_batch_size + 1 &&
      draft_q_cu_seq_lens.scalar_type() == torch::kInt32 &&
      draft_state_validity_mask.defined() &&
      draft_state_validity_mask.dim() == 1 &&
      draft_state_validity_mask.numel() == draft_batch_size &&
      draft_state_validity_mask.scalar_type() == torch::kBool &&
      norm_weight.numel() == head_v_dim_ &&
      norm_weight.scalar_type() == torch::kBFloat16 &&
      mixed_qkv.is_contiguous() && z.is_contiguous() && a.is_contiguous() &&
      b.is_contiguous() && conv_weight.is_contiguous() &&
      conv_cache.is_contiguous() && ssm_cache.is_contiguous() &&
      A_log_.is_contiguous() && dt_bias_.is_contiguous() &&
      logical_state_read_indices.is_contiguous() &&
      logical_state_write_indices.is_contiguous() &&
      draft_q_cu_seq_lens.is_contiguous() &&
      draft_state_validity_mask.is_contiguous() &&
      norm_weight.is_contiguous()) {
    torch::Tensor packed_qkv =
        reshape_qkvz_unpad(attn_metadata, mixed_qkv)
            .view({original_num_tokens, mega_gdn_conv_dim});
    torch::Tensor packed_z =
        reshape_qkvz_unpad(attn_metadata, z)
            .view({original_num_tokens, local_num_v_heads, head_v_dim_});
    torch::Tensor packed_b =
        reshape_qkvz_unpad(attn_metadata, b)
            .view({original_num_tokens, local_num_v_heads});
    torch::Tensor packed_a =
        reshape_qkvz_unpad(attn_metadata, a)
            .view({original_num_tokens, local_num_v_heads});
    torch::Tensor packed_norm_out =
        xllm::kernel::npu::mega_gdn_draft_decode(packed_qkv,
                                                 packed_z,
                                                 packed_b,
                                                 packed_a,
                                                 conv_weight,
                                                 conv_cache,
                                                 A_log_,
                                                 dt_bias_,
                                                 ssm_cache,
                                                 logical_state_read_indices,
                                                 logical_state_write_indices,
                                                 draft_q_cu_seq_lens,
                                                 draft_state_validity_mask,
                                                 norm_weight,
                                                 fla_ssm_state_layout);
    if (is_any_prefill) {
      fused_draft_norm_out =
          reshape_projected_tokens_with_pad(
              attn_metadata, packed_norm_out.view({original_num_tokens, -1}))
              .view({batch_size, seq_len, local_num_v_heads, head_v_dim_});
    } else {
      fused_draft_norm_out = packed_norm_out.view(
          {batch_size, seq_len, local_num_v_heads, head_v_dim_});
    }
    used_draft_super_op = true;
  }

  torch::Tensor fused_mtp_norm_out;
  bool used_mtp_super_op = false;
  if (std::getenv("XLLM_DISABLE_MEGA_GDN_MTP_DECODE") == nullptr &&
      detail::can_use_mega_gdn_mtp_decode(
          use_spec_verify,
          attn_metadata.is_prefill,
          attn_metadata.is_chunked_prefill,
          input_params.enable_graph,
          batch_size,
          seq_len,
          num_state_slots,
          attn_metadata.q_seq_lens_vec,
          input_params.embedding.linear_state_ids,
          input_params.embedding.linear_state_read_ids,
          input_params.embedding.linear_state_write_ids,
          input_params.num_accepted_tokens_host) &&
      checkpoint_stride == seq_len && batch_size >= 1 && batch_size <= 32 &&
      detail::is_supported_mega_gdn_head_geometry(
          local_num_k_heads, local_num_v_heads, head_k_dim_, head_v_dim_) &&
      std::abs(norm_->eps() - 1e-6) < 1e-12 &&
      input_params.num_accepted_tokens.defined() && mixed_qkv.dim() == 3 &&
      mixed_qkv.size(0) == batch_size && mixed_qkv.size(1) == seq_len &&
      mixed_qkv.size(2) == mega_gdn_conv_dim && z.dim() == 4 &&
      z.size(0) == batch_size && z.size(1) == seq_len &&
      z.size(2) == local_num_v_heads && z.size(3) == head_v_dim_ &&
      a.dim() == 3 && b.dim() == 3 &&
      a.sizes() ==
          torch::IntArrayRef({batch_size, seq_len, local_num_v_heads}) &&
      b.sizes() == a.sizes() && mixed_qkv.scalar_type() == torch::kBFloat16 &&
      z.scalar_type() == torch::kBFloat16 &&
      a.scalar_type() == torch::kBFloat16 &&
      b.scalar_type() == torch::kBFloat16 && conv_weight.dim() == 2 &&
      conv_weight.size(0) == 4 && conv_weight.size(1) == mixed_qkv.size(2) &&
      conv_weight.scalar_type() == torch::kBFloat16 && conv_cache.dim() == 3 &&
      conv_cache.size(1) == seq_len + 2 &&
      conv_cache.size(2) == mixed_qkv.size(2) &&
      conv_cache.scalar_type() == torch::kBFloat16 && ssm_cache.dim() == 4 &&
      ssm_cache.size(0) == conv_cache.size(0) * seq_len &&
      ssm_cache.size(1) == local_num_v_heads &&
      ssm_cache.size(2) == head_v_dim_ && ssm_cache.size(3) == head_k_dim_ &&
      ssm_cache.scalar_type() == torch::kFloat32 &&
      A_log_.numel() == local_num_v_heads &&
      A_log_.scalar_type() == torch::kFloat32 &&
      dt_bias_.numel() == local_num_v_heads &&
      dt_bias_.scalar_type() == torch::kFloat32 &&
      logical_state_read_indices.dim() == 1 &&
      logical_state_read_indices.numel() == batch_size &&
      logical_state_read_indices.scalar_type() == torch::kInt32 &&
      logical_state_write_indices.dim() == 1 &&
      logical_state_write_indices.numel() == batch_size &&
      logical_state_write_indices.scalar_type() == torch::kInt32 &&
      input_params.num_accepted_tokens.dim() == 1 &&
      input_params.num_accepted_tokens.numel() == batch_size &&
      input_params.num_accepted_tokens.scalar_type() == torch::kInt32 &&
      norm_weight.numel() == head_v_dim_ &&
      norm_weight.scalar_type() == torch::kBFloat16 &&
      mixed_qkv.is_contiguous() && z.is_contiguous() && a.is_contiguous() &&
      b.is_contiguous() && conv_weight.is_contiguous() &&
      conv_cache.is_contiguous() && ssm_cache.is_contiguous() &&
      A_log_.is_contiguous() && dt_bias_.is_contiguous() &&
      logical_state_read_indices.is_contiguous() &&
      logical_state_write_indices.is_contiguous() &&
      input_params.num_accepted_tokens.is_contiguous() &&
      norm_weight.is_contiguous()) {
    const char* dump_dir = std::getenv("XLLM_DEBUG_MEGA_GDN_MTP_DUMP_DIR");
    const char* dump_layer = std::getenv("XLLM_DEBUG_MEGA_GDN_MTP_DUMP_LAYER");
    const char* dump_rank = std::getenv("XLLM_DEBUG_MEGA_GDN_MTP_DUMP_RANK");
    const bool capture_debug_inputs =
        dump_dir != nullptr &&
        (dump_layer == nullptr || std::atoi(dump_layer) == layer_id_) &&
        (dump_rank == nullptr || std::atoi(dump_rank) == rank_);
    std::vector<std::pair<std::string, torch::Tensor>> debug_tensors;
    if (capture_debug_inputs) {
      torch::Tensor read_checkpoint_indices =
          logical_state_read_indices * seq_len +
          input_params.num_accepted_tokens - 1;
      debug_tensors = {
          {"qkv", mixed_qkv.clone()},
          {"z", z.clone()},
          {"b", b.clone()},
          {"a", a.clone()},
          {"conv_weight", conv_weight.clone()},
          {"conv_state_read",
           torch::index_select(
               conv_cache, /*dim=*/0, logical_state_read_indices)},
          {"a_log", A_log_.clone()},
          {"dt_bias", dt_bias_.clone()},
          {"ssm_state_read",
           torch::index_select(ssm_cache, /*dim=*/0, read_checkpoint_indices)},
          {"read_state_indices", logical_state_read_indices.clone()},
          {"write_state_indices", logical_state_write_indices.clone()},
          {"num_accepted_tokens", input_params.num_accepted_tokens.clone()},
          {"norm_weight", norm_weight.clone()},
      };
    }
    if (std::getenv("XLLM_LOG_MEGA_GDN_MTP_ROUTE") != nullptr) {
      LOG_FIRST_N(INFO, 1) << "[MEGA_GDN_MTP_ROUTE] graph="
                           << input_params.enable_graph << ", k=" << seq_len - 1
                           << ", batch_size=" << batch_size
                           << ", layer=" << layer_id_ << ", rank=" << rank_;
    }
    fused_mtp_norm_out =
        xllm::kernel::npu::mega_gdn_mtp_decode(mixed_qkv,
                                               z,
                                               b,
                                               a,
                                               conv_weight,
                                               conv_cache,
                                               A_log_,
                                               dt_bias_,
                                               ssm_cache,
                                               logical_state_read_indices,
                                               logical_state_write_indices,
                                               input_params.num_accepted_tokens,
                                               norm_weight,
                                               fla_ssm_state_layout);
    if (std::getenv("XLLM_DEBUG_MEGA_GDN_MTP_FINITE") != nullptr) {
      static std::atomic<bool> reported_nonfinite{false};
      const bool output_is_finite =
          torch::isfinite(fused_mtp_norm_out).all().item<bool>();
      if (!output_is_finite && !reported_nonfinite.exchange(true)) {
        if (capture_debug_inputs) {
          torch::Tensor step_offsets =
              torch::arange(seq_len, logical_state_write_indices.options());
          torch::Tensor write_checkpoint_indices =
              (logical_state_write_indices.unsqueeze(/*dim=*/1) * seq_len +
               step_offsets.unsqueeze(/*dim=*/0))
                  .flatten();
          debug_tensors.emplace_back(
              "conv_state_write",
              torch::index_select(
                  conv_cache, /*dim=*/0, logical_state_write_indices));
          debug_tensors.emplace_back(
              "ssm_state_write",
              torch::index_select(
                  ssm_cache, /*dim=*/0, write_checkpoint_indices));
          debug_tensors.emplace_back("out", fused_mtp_norm_out.clone());
          for (const auto& [name, tensor] : debug_tensors) {
            torch::save(tensor.cpu(),
                        std::string(dump_dir) + "/rank" +
                            std::to_string(rank_) + "_" + name + ".pt");
          }
        }
        const int32_t read_state_id =
            input_params.embedding.linear_state_read_ids.empty()
                ? input_params.embedding.linear_state_ids.front()
                : input_params.embedding.linear_state_read_ids.front();
        const int32_t write_state_id =
            input_params.embedding.linear_state_write_ids.empty()
                ? input_params.embedding.linear_state_ids.front()
                : input_params.embedding.linear_state_write_ids.front();
        LOG(ERROR) << "[MTP_GDN_NONFINITE] layer=" << layer_id_
                   << ", rank=" << rank_ << ", read_state_id=" << read_state_id
                   << ", write_state_id=" << write_state_id << ", accepted="
                   << input_params.num_accepted_tokens_host.front();
      }
    }
    used_mtp_super_op = true;
  }
  CHECK(!has_forked_mtp_state || used_mtp_super_op)
      << "Forked MTP linear state requires MegaGdnMtpDecode; the legacy "
         "Conv/SSM fallback only supports same-slot state";

  torch::Tensor fused_prefill_norm_out;
  bool used_prefill_super_op = false;
  if (!used_draft_super_op &&
      std::getenv("XLLM_ENABLE_QWEN35_GDN_PREFILL_SUPER_OP") != nullptr &&
      std::getenv("XLLM_DISABLE_QWEN35_GDN_PREFILL_SUPER_OP") == nullptr &&
      !use_spec_verify && attn_metadata.is_prefill &&
      !attn_metadata.is_chunked_prefill && checkpoint_stride == 1 &&
      fla_ssm_state_layout && batch_size == 1 && seq_len > 1 &&
      num_k_heads_ / tp_size_ == 8 && num_v_heads_ / tp_size_ == 24 &&
      head_k_dim_ == 128 && head_v_dim_ == 128 &&
      std::abs(norm_->eps() - 1e-6) < 1e-12 &&
      attn_metadata.q_seq_lens_vec.size() == 1 &&
      attn_metadata.q_seq_lens_vec[0] == seq_len &&
      input_params.embedding.linear_state_ids.size() == 1 &&
      input_params.embedding.linear_state_ids[0] >= 0 &&
      input_params.linear_state_validity_mask.size() == 1 &&
      input_params.linear_state_validity_mask[0] == 0 && mixed_qkv.dim() == 3 &&
      mixed_qkv.size(0) == 1 && mixed_qkv.size(1) == seq_len &&
      mixed_qkv.size(2) == 5120 && z.dim() == 4 && z.size(0) == 1 &&
      z.size(1) == seq_len && z.size(2) == 24 && z.size(3) == 128 &&
      a.dim() == 3 && b.dim() == 3 && a.size(0) == 1 && b.size(0) == 1 &&
      a.size(1) == seq_len && b.size(1) == seq_len && a.size(2) == 24 &&
      b.size(2) == 24 && mixed_qkv.scalar_type() == torch::kBFloat16 &&
      z.scalar_type() == torch::kBFloat16 &&
      a.scalar_type() == torch::kBFloat16 &&
      b.scalar_type() == torch::kBFloat16 && conv_weight.dim() == 2 &&
      conv_weight.size(0) == 4 && conv_weight.size(1) == 5120 &&
      conv_weight.scalar_type() == torch::kBFloat16 && conv_cache.dim() == 3 &&
      conv_cache.size(1) == 3 && conv_cache.size(2) == 5120 &&
      conv_cache.scalar_type() == torch::kBFloat16 && ssm_cache.dim() == 4 &&
      ssm_cache.size(1) == 24 && ssm_cache.size(2) == 128 &&
      ssm_cache.size(3) == 128 && ssm_cache.scalar_type() == torch::kFloat32 &&
      A_log_.numel() == 24 && A_log_.scalar_type() == torch::kFloat32 &&
      dt_bias_.numel() == 24 && dt_bias_.scalar_type() == torch::kFloat32 &&
      norm_weight.numel() == 128 &&
      norm_weight.scalar_type() == torch::kBFloat16 &&
      attn_metadata.q_cu_seq_lens.dim() == 1 &&
      attn_metadata.q_cu_seq_lens.numel() == 2 &&
      attn_metadata.q_cu_seq_lens.scalar_type() == torch::kInt32 &&
      mixed_qkv.is_contiguous() && z.is_contiguous() && a.is_contiguous() &&
      b.is_contiguous() && conv_weight.is_contiguous() &&
      conv_cache.is_contiguous() && ssm_cache.is_contiguous() &&
      A_log_.is_contiguous() && dt_bias_.is_contiguous() &&
      norm_weight.is_contiguous() &&
      attn_metadata.q_cu_seq_lens.is_contiguous()) {
    const int64_t state_index =
        static_cast<int64_t>(input_params.embedding.linear_state_ids[0]);
    fused_prefill_norm_out = xllm::kernel::npu::qwen35_gdn_prefill_super_op(
        mixed_qkv.view({seq_len, 5120}),
        z.view({seq_len, 24, 128}),
        b.view({seq_len, 24}),
        a.view({seq_len, 24}),
        conv_weight,
        conv_cache,
        A_log_,
        dt_bias_,
        ssm_cache,
        norm_weight,
        attn_metadata.q_cu_seq_lens,
        state_index,
        state_index);
    fused_prefill_norm_out = fused_prefill_norm_out.view({1, seq_len, 24, 128});
    used_prefill_super_op = true;
  }

  if (!used_decode_super_op && !used_draft_super_op && !used_mtp_super_op &&
      !used_prefill_super_op) {
    if (!use_spec_verify && is_any_prefill) {
      torch::IntArrayRef num_accepted_tokens_opt;
      std::vector<int64_t> linear_state_indices_vec(
          input_params.embedding.linear_state_ids.begin(),
          input_params.embedding.linear_state_ids.end());
      torch::Tensor conv_input = reshape_qkvz_unpad(attn_metadata, mixed_qkv);
      if (use_direct_prefill_qkv && conv_input.dim() == 2 &&
          conv_input.scalar_type() == torch::kBFloat16 &&
          conv_input.is_contiguous()) {
        std::tie(processed_q, processed_k, processed_v) =
            xllm::kernel::npu::causal_conv1d_qkv(
                conv_input,
                conv_weight,
                conv_cache,
                torch::IntArrayRef(input_params.parallel.query_start_loc),
                torch::IntArrayRef(linear_state_indices_vec),
                torch::IntArrayRef(input_params.linear_state_validity_mask),
                num_k_heads_ / tp_size_,
                num_v_heads_ / tp_size_,
                head_k_dim_,
                head_v_dim_);
        used_direct_prefill_qkv = true;
      } else {
        mixed_qkv = xllm::kernel::causal_conv1d(
            conv_input,
            conv_weight,
            conv_cache,
            std::optional<torch::Tensor>(),  // bias (no bias for qwen3)
            torch::IntArrayRef(input_params.parallel.query_start_loc),
            torch::IntArrayRef(linear_state_indices_vec),
            torch::IntArrayRef(input_params.linear_state_validity_mask),
            num_accepted_tokens_opt,
            xllm::npu::kCausalConv1dActivationSilu,
            xllm::npu::kCausalConv1dGraphPadSlotId,
            xllm::npu::kCausalConv1dRunModeForward);

        mixed_qkv = reshape_projected_tokens_with_pad(attn_metadata, mixed_qkv);
        mixed_qkv = mixed_qkv.transpose(1, 2);
      }
    } else {
      if (use_spec_verify) {
        CHECK(input_params.num_accepted_tokens.defined())
            << "num_accepted_tokens must be populated for Qwen3.5 spec verify";
      }
      torch::Tensor conv_input = reshape_qkvz_unpad(attn_metadata, mixed_qkv);
      const auto& num_accepted = use_spec_verify
                                     ? input_params.num_accepted_tokens_host
                                     : std::vector<int64_t>();
      const std::vector<int64_t> linear_state_indices_host(
          input_params.embedding.linear_state_ids.begin(),
          input_params.embedding.linear_state_ids.end());
      if (register_conv1d_graph_update) {
        if (use_spec_verify) {
          const auto conv1d_branch =
              xllm::npu::CausalConv1dGraphBranch::kSpecVerify;
          mixed_qkv = run_causal_conv1d_graph_update(
              graph_context,
              conv_input,
              conv_weight,
              conv_cache,
              std::optional<torch::Tensor>(),
              input_params.parallel.query_start_loc,
              linear_state_indices_host,
              num_accepted,
              conv1d_branch);
        } else {
          auto conv_input_2d =
              conv_input.dim() == 3
                  ? conv_input.reshape({-1, conv_input.size(-1)})
                  : conv_input;
          xllm::kernel::CausalConv1dUpdateParams conv1d_params;
          conv1d_params.x = conv_input_2d;
          conv1d_params.conv_state = conv_cache;
          conv1d_params.weight = conv_weight;
          if (std::getenv("XLLM_DISABLE_CACHED_CAUSAL_CONV1D_BIAS") ==
              nullptr) {
            conv1d_params.bias = conv1d_zero_bias_;
          }
          conv1d_params.conv_state_indices = logical_state_indices;
          conv1d_params.query_start_loc = attn_metadata.q_cu_seq_lens;
          conv1d_params.max_query_len = attn_metadata.max_query_len;
          if (std::getenv(
                  "XLLM_DISABLE_REUSED_CAUSAL_CONV1D_INITIAL_STATE_MODE") ==
              nullptr) {
            conv1d_params.initial_state_mode = attn_metadata.q_seq_lens;
          }
          mixed_qkv = xllm::kernel::causal_conv1d_update(conv1d_params);
          if (conv_input.dim() == 3) {
            mixed_qkv =
                mixed_qkv.view({conv_input.size(0), -1, mixed_qkv.size(-1)});
          }
        }
      } else {
        if (use_spec_verify) {
          torch::Tensor output = torch::empty_like(conv_input);
          xllm::kernel::causal_conv1d_out(
              output,
              conv_input,
              conv_weight,
              conv_cache,
              std::optional<torch::Tensor>(),
              torch::IntArrayRef(input_params.parallel.query_start_loc),
              torch::IntArrayRef(linear_state_indices_host),
              torch::IntArrayRef(std::vector<int64_t>()),
              torch::IntArrayRef(num_accepted),
              xllm::npu::kCausalConv1dActivationSilu,
              xllm::npu::kCausalConv1dGraphPadSlotId,
              xllm::npu::kCausalConv1dRunModeUpdate);
          mixed_qkv = output;
        } else {
          auto conv_input_2d =
              conv_input.dim() == 3
                  ? conv_input.reshape({-1, conv_input.size(-1)})
                  : conv_input;
          xllm::kernel::CausalConv1dUpdateParams conv1d_params;
          conv1d_params.x = conv_input_2d;
          conv1d_params.conv_state = conv_cache;
          conv1d_params.weight = conv_weight;
          if (std::getenv("XLLM_DISABLE_CACHED_CAUSAL_CONV1D_BIAS") ==
              nullptr) {
            conv1d_params.bias = conv1d_zero_bias_;
          }
          conv1d_params.conv_state_indices = logical_state_indices;
          conv1d_params.query_start_loc = attn_metadata.q_cu_seq_lens;
          conv1d_params.max_query_len = attn_metadata.max_query_len;
          if (std::getenv(
                  "XLLM_DISABLE_REUSED_CAUSAL_CONV1D_INITIAL_STATE_MODE") ==
              nullptr) {
            conv1d_params.initial_state_mode = attn_metadata.q_seq_lens;
          }
          mixed_qkv = xllm::kernel::causal_conv1d_update(conv1d_params);
          if (conv_input.dim() == 3) {
            mixed_qkv =
                mixed_qkv.view({conv_input.size(0), -1, mixed_qkv.size(-1)});
          }
        }
      }
      mixed_qkv = reshape_projected_tokens_with_pad(attn_metadata, mixed_qkv);
      mixed_qkv = mixed_qkv.transpose(1, 2);
    }
  }
  const bool use_fused_sigmoid_gdn_decode =
      fla_ssm_state_layout && !use_spec_verify && !is_any_prefill &&
      checkpoint_stride == 1;
  torch::Tensor g;
  torch::Tensor beta;
  // Compute gated delta net decay and beta terms.
  if (!used_draft_super_op && !used_mtp_super_op && !used_prefill_super_op) {
    if (use_spec_verify || attn_metadata.is_chunked_prefill ||
        checkpoint_stride > 1) {
      beta = torch::sigmoid(b);
      torch::Tensor A_log_exp = A_log_.exp();
      torch::Tensor a_float = a.to(torch::kFloat32);
      torch::Tensor a_plus_dt = a_float + dt_bias_;
      torch::Tensor softplus_out = torch::nn::functional::softplus(
          a_plus_dt,
          torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(
              20.0));
      g = -A_log_exp * softplus_out;
      g = g.to(a.dtype()).contiguous();
    } else if (attn_metadata.is_prefill) {
      xllm::kernel::FusedGdnGatingParams gdn_params;
      gdn_params.A_log = A_log_;
      gdn_params.a = a.contiguous().view({-1, a.size(-1)});
      gdn_params.b = b.contiguous().view({-1, b.size(-1)});
      gdn_params.dt_bias = dt_bias_;
      gdn_params.beta = 1.0f;
      gdn_params.threshold = 20.0f;
      std::tie(g, beta) = xllm::kernel::fused_gdn_gating(gdn_params);
      g = g.squeeze(0).contiguous().view({batch_size, seq_len, a.size(-1)});
      beta =
          beta.squeeze(0).contiguous().view({batch_size, seq_len, b.size(-1)});
    } else if (!use_fused_sigmoid_gdn_decode) {
      xllm::kernel::FusedGdnGatingParams gdn_params;
      gdn_params.A_log = A_log_;
      gdn_params.a = a.view({-1, a.size(-1)});
      gdn_params.b = b.view({-1, b.size(-1)});
      gdn_params.dt_bias = dt_bias_;
      gdn_params.beta = 1.0f;
      gdn_params.threshold = 20.0f;
      std::tie(g, beta) = xllm::kernel::fused_gdn_gating(gdn_params);
    }
  }
  bool use_fused_prefill_qkv_prepare = used_direct_prefill_qkv;
  if (!used_decode_super_op && !used_draft_super_op && !used_mtp_super_op &&
      !used_prefill_super_op) {
    if (!used_direct_prefill_qkv && !use_spec_verify &&
        attn_metadata.is_prefill && !attn_metadata.is_chunked_prefill &&
        batch_size == 1 && attn_metadata.q_seq_lens_vec.size() == 1 &&
        attn_metadata.q_seq_lens_vec[0] == seq_len && fla_ssm_state_layout &&
        std::getenv("XLLM_DISABLE_CAUSAL_CONV1D_QKV_PREPARE") == nullptr &&
        xllm::kernel::npu::tilelang::
            has_causal_conv1d_qkv_prepare_specialization(
                num_k_heads_ / tp_size_,
                num_v_heads_ / tp_size_,
                head_k_dim_)) {
      auto conv_qkv = mixed_qkv.transpose(1, 2).reshape({seq_len, -1});
      if (conv_qkv.scalar_type() == torch::kBFloat16 &&
          conv_qkv.is_contiguous()) {
        std::tie(processed_q, processed_k, processed_v) =
            xllm::kernel::npu::tilelang::causal_conv1d_qkv_prepare(
                conv_qkv,
                num_k_heads_ / tp_size_,
                num_v_heads_ / tp_size_,
                head_k_dim_);
        use_fused_prefill_qkv_prepare = true;
      }
    }
    if (!use_fused_prefill_qkv_prepare) {
      std::tie(processed_q, processed_k, processed_v) =
          process_mixed_qkv(mixed_qkv);
    }
  }
  torch::Tensor core_attn_out;
  torch::Tensor last_recurrent_state;
  bool use_fused_prefill_output_norm = false;
  float deferred_mega_output_scale = 1.0F;
  // Apply chunked or recurrent gated-delta attention and update caches.
  if (used_mtp_super_op) {
    core_attn_out = fused_mtp_norm_out;
  } else if (used_draft_super_op) {
    core_attn_out = fused_draft_norm_out;
  } else if (used_decode_super_op) {
    core_attn_out = fused_decode_norm_out;
  } else if (used_prefill_super_op) {
    core_attn_out = fused_prefill_norm_out;
  } else if (use_spec_verify) {
    torch::Tensor spec_num_accepted_tokens = expand_sequence_tensor_to_batch(
        input_params.num_accepted_tokens.to(device, torch::kInt32),
        batch_size,
        "num_accepted_tokens");
    torch::Tensor spec_linear_state_base_indices =
        expand_sequence_tensor_to_batch(
            linear_state_base_indices, batch_size, "linear_state_base_indices");
    torch::Tensor step_offsets =
        torch::arange(seq_len,
                      torch::TensorOptions()
                          .dtype(spec_linear_state_base_indices.dtype())
                          .device(device));
    torch::Tensor checkpoint_indices =
        spec_linear_state_base_indices.unsqueeze(1) + step_offsets;
    double scale = 1.0 / std::sqrt(static_cast<float>(processed_q.size(-1)));
    core_attn_out =
        run_spec_verify_gated_delta_rule(processed_q,
                                         processed_k,
                                         processed_v,
                                         g,
                                         beta,
                                         ssm_cache,
                                         checkpoint_indices,
                                         spec_num_accepted_tokens,
                                         attn_metadata.q_cu_seq_lens,
                                         attn_metadata.q_seq_lens_vec,
                                         scale);
  } else if (is_any_prefill) {
    CHECK_GE(attn_metadata.q_seq_lens_vec.size(),
             static_cast<size_t>(batch_size))
        << "q_seq_lens_vec must be populated for Qwen3.5 prefill.";
    const bool use_single_prefill_pack =
        batch_size == 1 && attn_metadata.q_seq_lens_vec.size() == 1 &&
        attn_metadata.q_seq_lens_vec[0] == seq_len;
    torch::Tensor packed_processed_q;
    torch::Tensor packed_processed_k;
    torch::Tensor packed_processed_v;
    torch::Tensor packed_g_tensor;
    torch::Tensor packed_beta_tensor;
    if (use_single_prefill_pack) {
      packed_processed_q = processed_q;
      packed_processed_k = processed_k;
      packed_processed_v = processed_v;
      packed_g_tensor = g;
      packed_beta_tensor = beta;
    } else {
      std::vector<torch::Tensor> packed_q;
      std::vector<torch::Tensor> packed_k;
      std::vector<torch::Tensor> packed_v;
      std::vector<torch::Tensor> packed_g;
      std::vector<torch::Tensor> packed_beta;
      packed_q.reserve(batch_size);
      packed_k.reserve(batch_size);
      packed_v.reserve(batch_size);
      packed_g.reserve(batch_size);
      packed_beta.reserve(batch_size);
      for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int64_t valid_len = attn_metadata.q_seq_lens_vec[batch_idx];
        packed_q.emplace_back(
            processed_q[batch_idx].narrow(/*dim=*/0, /*start=*/0, valid_len));
        packed_k.emplace_back(
            processed_k[batch_idx].narrow(/*dim=*/0, /*start=*/0, valid_len));
        packed_v.emplace_back(
            processed_v[batch_idx].narrow(/*dim=*/0, /*start=*/0, valid_len));
        packed_g.emplace_back(
            g[batch_idx].narrow(/*dim=*/0, /*start=*/0, valid_len));
        packed_beta.emplace_back(
            beta[batch_idx].narrow(/*dim=*/0, /*start=*/0, valid_len));
      }
      packed_processed_q = torch::cat(packed_q, 0).unsqueeze(0);
      packed_processed_k = torch::cat(packed_k, 0).unsqueeze(0);
      packed_processed_v = torch::cat(packed_v, 0).unsqueeze(0);
      packed_g_tensor = torch::cat(packed_g, 0).unsqueeze(0);
      packed_beta_tensor = torch::cat(packed_beta, 0).unsqueeze(0);
    }

    xllm::kernel::MegaChunkGdnParams mega_chunk_gdn_params;
    mega_chunk_gdn_params.q = packed_processed_q;
    mega_chunk_gdn_params.k = packed_processed_k;
    mega_chunk_gdn_params.v = packed_processed_v;
    mega_chunk_gdn_params.g = packed_g_tensor;
    mega_chunk_gdn_params.beta = packed_beta_tensor;
    // Get initial state from ssm_cache for sequences with previous state
    // Shape: [batch_size, num_heads, head_k_dim, head_v_dim]
    torch::Tensor initial_state_tensor =
        torch::index_select(ssm_cache, 0, linear_state_base_indices);
    CHECK_EQ(input_params.linear_state_validity_mask.size(),
             input_params.embedding.linear_state_ids.size())
        << "linear state validity mask must be sequence-scoped.";
    for (size_t i = 0; i < input_params.linear_state_validity_mask.size();
         ++i) {
      if (input_params.linear_state_validity_mask[i] == 0) {
        initial_state_tensor.select(0, static_cast<int64_t>(i)).fill_(0.0);
      }
    }
    if (!fla_ssm_state_layout && attn_metadata.is_chunked_prefill) {
      initial_state_tensor =
          initial_state_tensor.transpose(-1, -2).contiguous();
    }
    mega_chunk_gdn_params.initial_state = initial_state_tensor;
    mega_chunk_gdn_params.output_final_state = true;
    mega_chunk_gdn_params.cu_seqlens = attn_metadata.q_cu_seq_lens;
    mega_chunk_gdn_params.q_seq_lens = c10::ArrayRef<int32_t>(
        attn_metadata.q_seq_lens_vec.data(), static_cast<size_t>(batch_size));
    mega_chunk_gdn_params.use_qk_l2norm_in_kernel =
        !use_fused_prefill_qkv_prepare;
    int64_t direct_state_cache_index = -1;
    if (use_single_prefill_pack && fla_ssm_state_layout &&
        ssm_cache.scalar_type() == torch::kFloat32 &&
        ssm_cache.is_contiguous() && checkpoint_stride > 0 &&
        input_params.embedding.linear_state_ids.size() == 1) {
      direct_state_cache_index =
          static_cast<int64_t>(input_params.embedding.linear_state_ids[0]) *
          checkpoint_stride;
    }
    const bool use_direct_prefill_state_store =
        std::getenv("XLLM_DISABLE_DIRECT_PREFILL_STATE_STORE") == nullptr &&
        direct_state_cache_index >= 0 &&
        direct_state_cache_index < ssm_cache.size(0) &&
        xllm::kernel::npu::tilelang::has_final_state_cache_store_specialization(
            processed_v.size(2), processed_q.size(3), processed_v.size(3));
    mega_chunk_gdn_params.defer_final_state_cast =
        use_direct_prefill_state_store;
    use_fused_prefill_output_norm =
        use_single_prefill_pack &&
        norm_->supports_fused_scale_gated_rmsnorm(z, processed_v.size(-1));
    if (use_fused_prefill_output_norm) {
      deferred_mega_output_scale =
          1.0F / std::sqrt(static_cast<float>(processed_q.size(-1)));
      mega_chunk_gdn_params.scale = deferred_mega_output_scale;
      mega_chunk_gdn_params.defer_output_scale = true;
    }
    torch::Tensor packed_core_attn_out;
    std::tie(packed_core_attn_out, last_recurrent_state) =
        xllm::kernel::mega_chunk_gdn(mega_chunk_gdn_params);
    if (use_single_prefill_pack) {
      core_attn_out = packed_core_attn_out;
    } else {
      core_attn_out = torch::zeros_like(processed_v);
      int64_t packed_offset = 0;
      for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int64_t valid_len = attn_metadata.q_seq_lens_vec[batch_idx];
        core_attn_out[batch_idx]
            .narrow(/*dim=*/0, /*start=*/0, valid_len)
            .copy_(packed_core_attn_out[0].narrow(
                /*dim=*/0, packed_offset, valid_len));
        packed_offset += valid_len;
      }
    }
    if (use_direct_prefill_state_store) {
      auto cache_slot =
          ssm_cache.narrow(/*dim=*/0, direct_state_cache_index, /*length=*/1);
      xllm::kernel::npu::tilelang::final_state_cache_store(last_recurrent_state,
                                                           cache_slot);
    } else {
      torch::Tensor state_to_store =
          fla_ssm_state_layout ? last_recurrent_state
                               : last_recurrent_state.transpose(-1, -2);
      ssm_cache.index_put_({linear_state_base_indices},
                           state_to_store.to(ssm_cache.dtype()));
    }
  } else if (checkpoint_stride > 1) {
    auto ssm_state =
        torch::index_select(ssm_cache, 0, linear_state_base_indices);
    if (!fla_ssm_state_layout) {
      ssm_state = ssm_state.transpose(-1, -2);
    }
    ssm_state = ssm_state.contiguous();
    std::tie(core_attn_out, last_recurrent_state) =
        torch_recurrent_gated_delta_rule(
            processed_q, processed_k, processed_v, g, beta, ssm_state);
    torch::Tensor state_to_store = fla_ssm_state_layout
                                       ? last_recurrent_state
                                       : last_recurrent_state.transpose(-1, -2);
    ssm_cache.index_put_({linear_state_base_indices},
                         state_to_store.to(ssm_cache.dtype()));
  } else {
    double scale = 1.0 / std::sqrt(static_cast<float>(processed_q.size(-1)));
    if (fla_ssm_state_layout) {
      xllm::kernel::FusedSigmoidGatingDeltaRuleUpdateParams params;
      params.A_log = A_log_.contiguous();
      params.a = a.contiguous();
      params.dt_bias = dt_bias_.contiguous();
      params.q = processed_q.contiguous();
      params.k = processed_k.contiguous();
      params.v = processed_v.contiguous();
      params.b = b.contiguous();
      params.initial_state_source = ssm_cache;
      params.initial_state_indices = linear_state_base_indices.contiguous();
      params.cu_seqlens = attn_metadata.q_cu_seq_lens.contiguous();
      params.scale = static_cast<float>(scale);
      params.use_qk_l2norm_in_kernel = true;
      params.softplus_beta = 1.0f;
      params.softplus_threshold = 20.0f;
      core_attn_out =
          xllm::kernel::fused_sigmoid_gating_delta_rule_update(params);
    } else {
      processed_q = xllm::kernel::l2_norm(processed_q, /*eps=*/1e-6);
      processed_k = xllm::kernel::l2_norm(processed_k, /*eps=*/1e-6);
      auto zero = torch::zeros({1}, attn_metadata.q_seq_lens.options());
      torch::Tensor actual_seq_lengths =
          torch::cat({zero, attn_metadata.q_seq_lens}, 0);
      core_attn_out = xllm::kernel::recurrent_gated_delta_rule(
                          processed_q.reshape(
                              {-1, processed_q.size(-2), processed_q.size(-1)}),
                          processed_k.reshape(
                              {-1, processed_k.size(-2), processed_k.size(-1)}),
                          processed_v.reshape(
                              {-1, processed_v.size(-2), processed_v.size(-1)}),
                          ssm_cache,
                          beta.squeeze(0).contiguous(),
                          scale,
                          actual_seq_lengths,
                          logical_state_indices,
                          c10::nullopt,
                          g.squeeze(0).contiguous(),
                          c10::nullopt)
                          .unsqueeze(0)
                          .contiguous();
    }
  }
  torch::Tensor norm_out;
  if (used_decode_super_op || used_draft_super_op || used_mtp_super_op ||
      used_prefill_super_op) {
    norm_out = core_attn_out;
  } else {
    if (!use_fused_prefill_output_norm &&
        core_attn_out.scalar_type() != z.scalar_type()) {
      core_attn_out = core_attn_out.to(z.scalar_type());
    }
    auto z_reshaped = z.view({-1, z.size(-1)});
    auto core_attn_out_reshaped =
        core_attn_out.view({-1, core_attn_out.size(-1)});
    norm_out = use_fused_prefill_output_norm
                   ? norm_->forward_scaled(core_attn_out_reshaped,
                                           z_reshaped,
                                           deferred_mega_output_scale)
                   : norm_->forward(core_attn_out_reshaped, z_reshaped);
  }
  auto z_shape_og = z.sizes().vec();
  norm_out = norm_out.view(z_shape_og);
  norm_out = norm_out.view({-1, norm_out.size(2), norm_out.size(3)});
  // Project the normalized attention output back to hidden size.
  auto rearranged_norm =
      norm_out.reshape({norm_out.size(0), norm_out.size(1) * norm_out.size(2)});
  rearranged_norm = reshape_qkvz_unpad(attn_metadata, rearranged_norm);
  // For chunked prefill or spec verify, reshape_projected_tokens_with_pad may
  // pad each batch to max_len, causing output tokens > original_num_tokens. We
  // need to slice back to original_num_tokens to match the residual shape.
  if (rearranged_norm.size(0) > original_num_tokens) {
    // Slice excess padding tokens
    rearranged_norm =
        rearranged_norm.slice(0, 0, original_num_tokens).contiguous();
  }
  torch::Tensor projected_output;
  if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    projected_output = o_proj_->forward(
        rearranged_norm, row_parallel_reduce_mode_for_fc1(*fc1_ctx));
  } else {
    projected_output = o_proj_->forward(rearranged_norm);
  }
  if (mtp_gdn_debug.enabled) {
    torch::Tensor selected_write_indices = torch::index_select(
        logical_state_write_indices, 0, mtp_gdn_debug.row_indices);
    torch::Tensor step_offsets =
        torch::arange(seq_len, selected_write_indices.options());
    torch::Tensor write_checkpoint_indices =
        (selected_write_indices.unsqueeze(1) * seq_len +
         step_offsets.unsqueeze(0))
            .flatten();
    mtp_gdn_debug.tensors.emplace_back(
        "conv_state_write",
        torch::index_select(conv_cache, 0, selected_write_indices));
    mtp_gdn_debug.tensors.emplace_back(
        "ssm_state_write",
        torch::index_select(ssm_cache, 0, write_checkpoint_indices));
    mtp_gdn_debug.tensors.emplace_back(
        "norm_out",
        select_mtp_debug_token_rows(norm_out,
                                    mtp_gdn_debug.row_indices,
                                    mtp_gdn_debug.batch_size,
                                    mtp_gdn_debug.seq_len));
    mtp_gdn_debug.tensors.emplace_back(
        "rearranged_norm",
        select_mtp_debug_token_rows(rearranged_norm,
                                    mtp_gdn_debug.row_indices,
                                    mtp_gdn_debug.batch_size,
                                    mtp_gdn_debug.seq_len));
    mtp_gdn_debug.tensors.emplace_back(
        "projected_output",
        select_mtp_debug_token_rows(projected_output,
                                    mtp_gdn_debug.row_indices,
                                    mtp_gdn_debug.batch_size,
                                    mtp_gdn_debug.seq_len));
    save_mtp_gdn_debug_capture(mtp_gdn_debug,
                               rank_,
                               layer_id_,
                               used_mtp_super_op,
                               input_params.embedding.request_ids);
  }
  return projected_output;
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::reshape_qkvz_unpad(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& padded_qkvz) const {
  const bool has_padded_queries =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  if (!has_padded_queries) {
    return padded_qkvz;
  }
  std::vector<torch::Tensor> valid_batches;
  const bool has_host_lens = !attn_metadata.q_seq_lens_vec.empty();
  int64_t bs = has_host_lens
                   ? static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size())
                   : attn_metadata.q_seq_lens.size(0);
  valid_batches.reserve(bs);
  int64_t max_len = attn_metadata.max_query_len;
  const auto& ori_seq_lens = attn_metadata.q_seq_lens;
  auto reshaped_qkvz = padded_qkvz.view({bs, max_len, -1});
  for (int64_t b = 0; b < bs; ++b) {
    int64_t ori_len = has_host_lens ? attn_metadata.q_seq_lens_vec[b]
                                    : ori_seq_lens[b].template item<int64_t>();
    torch::Tensor valid_batch =
        reshaped_qkvz[b].slice(/*dim=*/0, /*start=*/0, ori_len);
    valid_batches.emplace_back(valid_batch);
  }
  if (valid_batches.size() == 1) {
    return valid_batches[0].contiguous();
  }
  return torch::cat(valid_batches, 0).contiguous();
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::get_linear_state_indices(
    const ModelInputParams& input_params,
    const torch::Device& device) const {
  CHECK(!input_params.embedding.linear_state_ids.empty())
      << "linear_state_ids must be populated for gated delta net";
  if (input_params.embedding.linear_state_indices.defined()) {
    auto indices = input_params.embedding.linear_state_indices;
    if (indices.device() != device || indices.scalar_type() != torch::kInt) {
      indices =
          indices.to(torch::TensorOptions().dtype(torch::kInt).device(device),
                     /*non_blocking=*/true,
                     /*copy=*/true);
    }
    return indices.contiguous();
  }
  return torch::tensor(
      input_params.embedding.linear_state_ids,
      torch::TensorOptions().dtype(torch::kInt).device(device));
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::reshape_projected_tokens_with_pad(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& projected_tokens) const {
  const bool has_host_lens = !attn_metadata.q_seq_lens_vec.empty();
  int64_t bs = has_host_lens
                   ? static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size())
                   : attn_metadata.q_seq_lens.size(0);
  int64_t max_len = attn_metadata.max_query_len;
  const auto& start_loc = attn_metadata.q_seq_lens;
  const bool need_padding =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  if (!need_padding) {
    return projected_tokens.view({bs, -1, projected_tokens.size(-1)});
  }
  if (has_host_lens && bs == 1 && attn_metadata.q_seq_lens_vec[0] == max_len &&
      projected_tokens.dim() == 2 && projected_tokens.size(0) == max_len) {
    return projected_tokens.view({1, max_len, projected_tokens.size(-1)});
  }
  std::vector<torch::Tensor> batches;
  batches.reserve(bs);
  int64_t idx = 0;
  for (int64_t b = 0; b < bs; ++b) {
    int64_t cur_len = has_host_lens ? attn_metadata.q_seq_lens_vec[b]
                                    : start_loc[b].template item<int64_t>();
    torch::Tensor batch =
        projected_tokens.slice(/*dim=*/0, idx, idx + cur_len).contiguous();
    idx = idx + cur_len;
    if (batch.size(0) != max_len) {
      batch = batch.size(0) > max_len
                  ? batch.slice(/*dim=*/0, /*start=*/0, max_len).contiguous()
                  : torch::nn::functional::pad(
                        batch,
                        torch::nn::functional::PadFuncOptions(
                            {0, 0, 0, max_len - batch.size(0)}))
                        .contiguous();
    }
    batches.emplace_back(batch);
  }
  auto ret = torch::stack(batches, 0).contiguous();
  return ret;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Qwen3GatedDeltaNetBaseImpl::process_mixed_qkv(torch::Tensor& mixed_qkv) const {
  mixed_qkv = mixed_qkv.transpose(1, 2);
  int64_t batch_size = mixed_qkv.size(0);
  int64_t seq_len = mixed_qkv.size(1);
  std::vector<int64_t> split_sizes = {
      k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};
  auto processed_qkv = torch::split(mixed_qkv, split_sizes, 2);
  auto processed_q = processed_qkv[0];
  auto processed_k = processed_qkv[1];
  auto processed_v = processed_qkv[2];
  processed_q = processed_q.view(
      {batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_k = processed_k.view(
      {batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_v = processed_v.view(
      {batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  return std::make_tuple(processed_q, processed_k, processed_v);
}

}  // namespace layer
}  // namespace xllm
