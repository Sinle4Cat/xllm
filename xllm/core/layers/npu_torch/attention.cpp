/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "attention.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "kernels/npu/npu_ops_api.h"
#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#include "kernels/ops_api.h"

namespace {

enum class XFlashAttentionInferMode {
  kDisabled = 0,
  kFd = 1,
  kNoFd = 2,
};

// The current FD kernel family is tuned for chunked-prefill tasks that fill a
// meaningful portion of the 128-row q tile. In warmed TP2 mixed-load testing,
// very small chunked-prefill tasks regressed end-to-end latency, so keep those
// shapes on the existing fused-infer-attention path until a dedicated kernel or
// shape-specific evidence shows they win.
constexpr int64_t kMaxXFlashAttentionInferFdQRowsPerTask = 128;

std::pair<int64_t, int64_t> get_value_range(
    const std::vector<int64_t>& values) {
  if (values.empty()) {
    return {0, 0};
  }
  const auto minmax = std::minmax_element(values.begin(), values.end());
  return {*minmax.first, *minmax.second};
}

std::pair<int64_t, int64_t> get_q_len_range(
    const xllm::layer::AttentionMetadata& attn_metadata) {
  if (attn_metadata.q_cu_seq_lens_host_vec.empty()) {
    return {0, 0};
  }

  int64_t min_q_len = std::numeric_limits<int64_t>::max();
  int64_t max_q_len = 0;
  int64_t previous_q_len = 0;
  for (int64_t cumulative_q_len : attn_metadata.q_cu_seq_lens_host_vec) {
    const int64_t q_len = cumulative_q_len - previous_q_len;
    previous_q_len = cumulative_q_len;
    min_q_len = std::min(min_q_len, q_len);
    max_q_len = std::max(max_q_len, q_len);
  }
  return {min_q_len, max_q_len};
}

std::pair<int64_t, int64_t> get_fd_q_rows_range(
    const xllm::layer::AttentionMetadata& attn_metadata,
    int64_t group_size) {
  const std::pair<int64_t, int64_t> q_len_range =
      get_q_len_range(attn_metadata);
  return {q_len_range.first * group_size, q_len_range.second * group_size};
}

const char* x_flash_attention_infer_mode_name(XFlashAttentionInferMode mode) {
  switch (mode) {
    case XFlashAttentionInferMode::kFd:
      return "fd";
    case XFlashAttentionInferMode::kNoFd:
      return "no_fd";
    case XFlashAttentionInferMode::kDisabled:
    default:
      return "disabled";
  }
}

bool is_x_flash_attention_infer_enabled() {
  const char* value = std::getenv("XLLM_ENABLE_X_FLASH_ATTENTION_INFER");
  if (value == nullptr) {
    return false;
  }
  const std::string flag_value(value);
  return flag_value == "1" || flag_value == "true" || flag_value == "TRUE" ||
         flag_value == "on" || flag_value == "ON";
}

XFlashAttentionInferMode get_x_flash_attention_infer_mode(
    const xllm::layer::AttentionMetadata& attn_metadata,
    const torch::Tensor& query,
    const torch::Tensor& k_cache,
    const std::optional<torch::Tensor>& v_cache,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t block_size) {
  if (!is_x_flash_attention_infer_enabled()) {
    return XFlashAttentionInferMode::kDisabled;
  }
  if (attn_metadata.enable_cuda_graph || !attn_metadata.is_chunked_prefill) {
    return XFlashAttentionInferMode::kDisabled;
  }
  if (!attn_metadata.fia_attn_mask.defined() ||
      !attn_metadata.block_table.defined() || !v_cache.has_value() ||
      !v_cache.value().defined()) {
    return XFlashAttentionInferMode::kDisabled;
  }
  if (query.dim() != 3 || k_cache.dim() != 4 || v_cache.value().dim() != 4) {
    return XFlashAttentionInferMode::kDisabled;
  }
  if (query.dtype() != torch::kFloat16 && query.dtype() != torch::kBFloat16) {
    return XFlashAttentionInferMode::kDisabled;
  }
  if (k_cache.dtype() != query.dtype() ||
      v_cache.value().dtype() != query.dtype()) {
    return XFlashAttentionInferMode::kDisabled;
  }
  if (block_size <= 0 || 512 % block_size != 0) {
    return XFlashAttentionInferMode::kDisabled;
  }
  if (num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    return XFlashAttentionInferMode::kDisabled;
  }
  if (attn_metadata.q_cu_seq_lens_host_vec.empty() ||
      !attn_metadata.q_cu_seq_lens_no_zero.defined() ||
      attn_metadata.x_flash_attention_infer_cache == nullptr ||
      !attn_metadata.x_flash_attention_infer_cache->actual_q_lens.defined() ||
      attn_metadata.q_cu_seq_lens_host_vec.size() !=
          attn_metadata.kv_seq_lens_host_vec.size()) {
    return XFlashAttentionInferMode::kDisabled;
  }
  const int64_t group_size = num_heads / num_kv_heads;
  int64_t uniform_q_len = -1;
  bool is_fd_eligible = true;
  int64_t previous_q_len = 0;
  for (int64_t cumulative_q_len : attn_metadata.q_cu_seq_lens_host_vec) {
    const int64_t q_len = cumulative_q_len - previous_q_len;
    previous_q_len = cumulative_q_len;
    if (q_len <= 0) {
      return XFlashAttentionInferMode::kDisabled;
    }
    if (uniform_q_len < 0) {
      uniform_q_len = q_len;
    } else if (q_len != uniform_q_len) {
      uniform_q_len = 0;
    }
    const int64_t fd_q_rows = q_len * group_size;
    if (fd_q_rows > kMaxXFlashAttentionInferFdQRowsPerTask) {
      is_fd_eligible = false;
    }
  }
  if (is_fd_eligible) {
    return XFlashAttentionInferMode::kFd;
  }
  if (uniform_q_len > 0) {
    return XFlashAttentionInferMode::kNoFd;
  }
  return XFlashAttentionInferMode::kDisabled;
}

torch::Tensor get_x_flash_attention_infer_extra_tiling(
    const xllm::layer::AttentionMetadata& attn_metadata,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t block_size,
    bool use_fd,
    const torch::Tensor& reference) {
  CHECK(attn_metadata.x_flash_attention_infer_cache != nullptr)
      << "x_flash_attention_infer cache must be initialized";
  xllm::layer::XFlashAttentionInferCache& cache =
      *attn_metadata.x_flash_attention_infer_cache;
  if (cache.extra_tiling.defined() && cache.cached_num_heads == num_heads &&
      cache.cached_num_kv_heads == num_kv_heads &&
      cache.cached_block_size == block_size && cache.cached_use_fd == use_fd &&
      cache.extra_tiling.device() == reference.device()) {
    return cache.extra_tiling;
  }

  cache.extra_tiling =
      xllm::kernel::npu::build_x_flash_attention_infer_extra_tiling(
          attn_metadata.q_cu_seq_lens_host_vec,
          attn_metadata.kv_seq_lens_host_vec,
          num_heads,
          num_kv_heads,
          block_size,
          use_fd,
          reference);
  cache.cached_num_heads = num_heads;
  cache.cached_num_kv_heads = num_kv_heads;
  cache.cached_block_size = block_size;
  cache.cached_use_fd = use_fd;
  return cache.extra_tiling;
}

torch::Tensor get_x_flash_attention_infer_actual_q_lens(
    const xllm::layer::AttentionMetadata& attn_metadata,
    const torch::Tensor& reference) {
  CHECK(attn_metadata.x_flash_attention_infer_cache != nullptr)
      << "x_flash_attention_infer cache must be initialized";
  xllm::layer::XFlashAttentionInferCache& cache =
      *attn_metadata.x_flash_attention_infer_cache;
  CHECK(cache.actual_q_lens.defined())
      << "x_flash_attention_infer actual_q_lens must be initialized";
  if (cache.actual_q_lens.device() == reference.device() &&
      cache.actual_q_lens.dtype() == torch::kInt32) {
    return cache.actual_q_lens;
  }
  cache.actual_q_lens =
      cache.actual_q_lens.to(reference.device(), torch::kInt32, true, true);
  return cache.actual_q_lens;
}

}  // namespace

namespace xllm {
namespace layer {

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window),
      scale_(scale) {
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  std::optional<torch::Tensor> output_lse = std::nullopt;
  torch::Tensor output = torch::empty_like(query);

  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v = value.view({-1, num_kv_heads_, head_size_});
  std::optional<torch::Tensor> v_cache = kv_cache.get_v_cache();

  // Reshape and cache key/value
  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key.view({-1, num_kv_heads_, head_size_});
  reshape_paged_cache_params.value = v;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  if (attn_metadata.use_expanded_decode_for_spec_verify_attention) {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  } else if (only_prefill) {
    prefill_forward(query, key, value, output, k_cache, v_cache, attn_metadata);
  } else {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void AttentionImpl::prefill_forward(torch::Tensor& query,
                                    torch::Tensor& key,
                                    torch::Tensor& value,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  if (attn_metadata.is_prefill) {
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_});

    auto fia_result = xllm::kernel::npu::npu_fused_infer_attention(
        query,
        key,
        value,
        attn_metadata.fia_attn_mask.defined()
            ? std::make_optional(attn_metadata.fia_attn_mask)
            : std::nullopt,
        std::nullopt,
        attn_metadata.q_cu_seq_lens_host_vec,
        attn_metadata.kv_cu_seq_lens_host_vec,
        num_heads_,
        num_kv_heads_,
        scale_,
        /*block_size=*/0,
        /*sparse_mode=*/3,
        "TND");
    output.copy_(std::get<0>(fia_result).view_as(output));
  } else if (attn_metadata.is_chunked_prefill) {
    const XFlashAttentionInferMode xfa_mode =
        get_x_flash_attention_infer_mode(attn_metadata,
                                         query,
                                         k_cache,
                                         v_cache,
                                         num_heads_,
                                         num_kv_heads_,
                                         k_cache.size(1));
    const std::pair<int64_t, int64_t> q_len_range =
        get_q_len_range(attn_metadata);
    const std::pair<int64_t, int64_t> kv_len_range =
        get_value_range(attn_metadata.kv_seq_lens_host_vec);
    const std::pair<int64_t, int64_t> fd_q_rows_range =
        get_fd_q_rows_range(attn_metadata, num_heads_ / num_kv_heads_);
    VLOG(1) << "Evaluated x_flash_attention_infer for chunked prefill: mode="
            << x_flash_attention_infer_mode_name(xfa_mode) << ", "
            << "num_heads=" << num_heads_ << ", num_kv_heads=" << num_kv_heads_
            << ", batch_size=" << attn_metadata.q_cu_seq_lens_host_vec.size()
            << ", q_len_range=[" << q_len_range.first << ", "
            << q_len_range.second << "]"
            << ", kv_len_range=[" << kv_len_range.first << ", "
            << kv_len_range.second << "]"
            << ", fd_q_rows_range=[" << fd_q_rows_range.first << ", "
            << fd_q_rows_range.second << "]"
            << ", total_q_tokens="
            << (attn_metadata.q_cu_seq_lens_host_vec.empty()
                    ? 0
                    : attn_metadata.q_cu_seq_lens_host_vec.back());
    if (xfa_mode != XFlashAttentionInferMode::kDisabled) {
      VLOG(1) << "Using x_flash_attention_infer for chunked prefill: mode="
              << x_flash_attention_infer_mode_name(xfa_mode) << ", "
              << "num_heads=" << num_heads_
              << ", num_kv_heads=" << num_kv_heads_
              << ", batch_size=" << attn_metadata.q_cu_seq_lens_host_vec.size()
              << ", q_len_range=[" << q_len_range.first << ", "
              << q_len_range.second << "]"
              << ", kv_len_range=[" << kv_len_range.first << ", "
              << kv_len_range.second << "]"
              << ", max_tokens_per_chunk="
              << (attn_metadata.q_cu_seq_lens_host_vec.empty()
                      ? 0
                      : attn_metadata.q_cu_seq_lens_host_vec.back());
      torch::Tensor extra_tiling = get_x_flash_attention_infer_extra_tiling(
          attn_metadata,
          num_heads_,
          num_kv_heads_,
          k_cache.size(1),
          xfa_mode == XFlashAttentionInferMode::kFd,
          query);
      torch::Tensor xfa_output = xllm::kernel::npu::x_flash_attention_infer(
          query,
          k_cache,
          v_cache.value(),
          std::make_optional(attn_metadata.fia_attn_mask),
          attn_metadata.block_table,
          get_x_flash_attention_infer_actual_q_lens(attn_metadata, query),
          attn_metadata.kv_seq_lens,
          extra_tiling,
          num_heads_,
          num_kv_heads_,
          scale_,
          "TND");
      output.copy_(xfa_output.view_as(output));
      return;
    }
    torch::Tensor k = k_cache.view({k_cache.size(0), k_cache.size(1), -1});
    torch::Tensor v = v_cache.value().view(
        {v_cache.value().size(0), v_cache.value().size(1), -1});
    auto fia_result = xllm::kernel::npu::npu_fused_infer_attention(
        query,
        k,
        v,
        attn_metadata.fia_attn_mask.defined()
            ? std::make_optional(attn_metadata.fia_attn_mask)
            : std::nullopt,
        attn_metadata.block_table.defined()
            ? std::make_optional(attn_metadata.block_table)
            : std::nullopt,
        attn_metadata.q_cu_seq_lens_host_vec,
        attn_metadata.kv_seq_lens_host_vec,
        num_heads_,
        num_kv_heads_,
        scale_,
        /*block_size=*/k_cache.size(1),
        /*sparse_mode=*/3,
        "TND");
    output.copy_(std::get<0>(fia_result).view_as(output));
  }
}

void AttentionImpl::decoder_forward(torch::Tensor& query,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, 1, num_heads_, head_size_});
  output = output.view({-1, 1, num_heads_, head_size_});

  torch::Tensor kv_seq_lens;
  torch::Tensor block_table = attn_metadata.block_table;
  torch::Tensor tiling_data = attn_metadata.paged_attention_tiling_data;
  if (attn_metadata.use_expanded_decode_for_spec_verify_attention) {
    block_table = attn_metadata.expanded_block_table;
    tiling_data = attn_metadata.expanded_paged_attention_tiling_data;
    if (attn_metadata.expanded_kv_seq_lens_host.defined()) {
      kv_seq_lens = attn_metadata.expanded_kv_seq_lens_host;
    } else {
      kv_seq_lens = attn_metadata.expanded_kv_seq_lens;
    }
  } else if (attn_metadata.kv_seq_lens_host.defined()) {
    kv_seq_lens = attn_metadata.kv_seq_lens_host;
  } else {
    // Fallback if host tensor isn't prepared.
    kv_seq_lens = attn_metadata.kv_seq_lens;
  }

  if (tiling_data.defined()) {
    // Use CustomPagedAttention for ACL graph mode to avoid .to(kCPU) operations

    xllm::kernel::npu::batch_decode_acl_graph(query,
                                              k_cache,
                                              v_cache.value_or(torch::Tensor()),
                                              scale_,
                                              block_table,
                                              kv_seq_lens,
                                              tiling_data,
                                              output);
  } else {
    // Standard PagedAttention path
    xllm::kernel::npu::batch_decode(query,
                                    k_cache,
                                    v_cache.value_or(torch::Tensor()),
                                    scale_,
                                    block_table,
                                    kv_seq_lens,
                                    output);
  }
}

}  // namespace layer
}  // namespace xllm
