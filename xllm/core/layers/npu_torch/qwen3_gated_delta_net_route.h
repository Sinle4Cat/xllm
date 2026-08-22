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

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllm {
namespace layer {
namespace detail {

inline bool is_supported_mega_gdn_mtp_k(int64_t speculative_tokens) {
  return speculative_tokens >= 1 && speculative_tokens <= 16;
}

inline bool is_supported_mega_gdn_head_geometry(int64_t num_k_heads,
                                                int64_t num_v_heads,
                                                int64_t head_k_dim,
                                                int64_t head_v_dim) {
  if (head_k_dim != 128 || head_v_dim != 128 || num_k_heads < 1 ||
      num_k_heads > 16 || (num_k_heads & (num_k_heads - 1)) != 0 ||
      num_v_heads % num_k_heads != 0) {
    return false;
  }
  const int64_t num_v_heads_per_k_head = num_v_heads / num_k_heads;
  return num_v_heads_per_k_head >= 1 && num_v_heads_per_k_head <= 4;
}

inline bool is_dense_mega_gdn_mtp_verify_batch(
    int64_t total_tokens,
    int64_t sequence_length,
    const std::vector<int32_t>& q_seq_lens) {
  if (total_tokens <= 0 || sequence_length <= 1 || q_seq_lens.empty()) {
    return false;
  }
  const int64_t batch_size = static_cast<int64_t>(q_seq_lens.size());
  if (total_tokens != batch_size * sequence_length) {
    return false;
  }
  for (const int32_t q_seq_len : q_seq_lens) {
    if (q_seq_len != sequence_length) {
      return false;
    }
  }
  return true;
}

inline const std::vector<int32_t>& resolve_mega_gdn_mtp_state_ids(
    const std::vector<int32_t>& legacy_state_ids,
    const std::vector<int32_t>& explicit_state_ids) {
  return explicit_state_ids.empty() ? legacy_state_ids : explicit_state_ids;
}

inline bool has_valid_mega_gdn_mtp_state_metadata(
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_state_slots,
    const std::vector<int32_t>& q_seq_lens,
    const std::vector<int32_t>& legacy_state_ids,
    const std::vector<int32_t>& explicit_read_state_ids,
    const std::vector<int32_t>& explicit_write_state_ids,
    const std::vector<int64_t>& num_accepted_tokens) {
  if (batch_size <= 0 || seq_len <= 1 || num_state_slots <= 0) {
    return false;
  }

  const size_t expected_size = static_cast<size_t>(batch_size);
  const std::vector<int32_t>& read_state_ids =
      resolve_mega_gdn_mtp_state_ids(legacy_state_ids, explicit_read_state_ids);
  const std::vector<int32_t>& write_state_ids = resolve_mega_gdn_mtp_state_ids(
      legacy_state_ids, explicit_write_state_ids);
  if (q_seq_lens.size() != expected_size ||
      read_state_ids.size() != expected_size ||
      write_state_ids.size() != expected_size ||
      num_accepted_tokens.size() != expected_size) {
    return false;
  }

  for (size_t batch_idx = 0; batch_idx < expected_size; ++batch_idx) {
    const int32_t q_seq_len = q_seq_lens[batch_idx];
    const int32_t read_state_id = read_state_ids[batch_idx];
    const int32_t write_state_id = write_state_ids[batch_idx];
    const int64_t accepted_tokens = num_accepted_tokens[batch_idx];
    if (q_seq_len <= 0 || q_seq_len > seq_len || accepted_tokens < 1 ||
        accepted_tokens > q_seq_len || read_state_id < 0 ||
        read_state_id >= num_state_slots || write_state_id < 0 ||
        write_state_id >= num_state_slots) {
      return false;
    }
  }

  for (size_t write_idx = 0; write_idx < expected_size; ++write_idx) {
    for (size_t other_idx = 0; other_idx < expected_size; ++other_idx) {
      if (write_idx == other_idx) {
        continue;
      }
      if (write_state_ids[write_idx] == write_state_ids[other_idx] ||
          write_state_ids[write_idx] == read_state_ids[other_idx]) {
        return false;
      }
    }
  }
  return true;
}

inline bool has_forked_mega_gdn_mtp_state(
    const std::vector<int32_t>& legacy_state_ids,
    const std::vector<int32_t>& explicit_read_state_ids,
    const std::vector<int32_t>& explicit_write_state_ids) {
  const std::vector<int32_t>& read_state_ids =
      resolve_mega_gdn_mtp_state_ids(legacy_state_ids, explicit_read_state_ids);
  const std::vector<int32_t>& write_state_ids = resolve_mega_gdn_mtp_state_ids(
      legacy_state_ids, explicit_write_state_ids);
  return read_state_ids != write_state_ids;
}

inline bool can_use_mega_gdn_mtp_decode(
    bool use_spec_verify,
    bool is_prefill,
    bool is_chunked_prefill,
    bool /*enable_graph*/,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_state_slots,
    const std::vector<int32_t>& q_seq_lens,
    const std::vector<int32_t>& legacy_state_ids,
    const std::vector<int32_t>& explicit_read_state_ids,
    const std::vector<int32_t>& explicit_write_state_ids,
    const std::vector<int64_t>& num_accepted_tokens) {
  const bool is_supported_verify_phase = !is_prefill || is_chunked_prefill;
  // ACL graph capture supplies persistent device buffers for the dynamic
  // state indices and accepted-token counts, so it shares eager eligibility.
  if (!use_spec_verify || !is_supported_verify_phase || batch_size <= 0 ||
      seq_len <= 1 || num_state_slots <= 0 ||
      !is_supported_mega_gdn_mtp_k(seq_len - 1)) {
    return false;
  }

  if (!has_valid_mega_gdn_mtp_state_metadata(batch_size,
                                             seq_len,
                                             num_state_slots,
                                             q_seq_lens,
                                             legacy_state_ids,
                                             explicit_read_state_ids,
                                             explicit_write_state_ids,
                                             num_accepted_tokens)) {
    return false;
  }

  return is_dense_mega_gdn_mtp_verify_batch(
      batch_size * seq_len, seq_len, q_seq_lens);
}

inline bool can_use_mega_gdn_draft_decode(
    bool is_mtp_draft,
    bool use_spec_verify,
    int64_t total_tokens,
    int64_t num_state_slots,
    const std::vector<int32_t>& draft_q_seq_lens,
    const std::vector<int32_t>& legacy_state_ids,
    const std::vector<int32_t>& explicit_read_state_ids,
    const std::vector<int32_t>& explicit_write_state_ids,
    const std::vector<int64_t>& state_validity_mask) {
  if (!is_mtp_draft || use_spec_verify || total_tokens <= 0 ||
      num_state_slots <= 0 || draft_q_seq_lens.empty()) {
    return false;
  }

  const size_t batch_size = draft_q_seq_lens.size();
  const std::vector<int32_t>& read_state_ids =
      resolve_mega_gdn_mtp_state_ids(legacy_state_ids, explicit_read_state_ids);
  const std::vector<int32_t>& write_state_ids = resolve_mega_gdn_mtp_state_ids(
      legacy_state_ids, explicit_write_state_ids);
  if (batch_size > 32 || read_state_ids.size() != batch_size ||
      write_state_ids.size() != batch_size ||
      state_validity_mask.size() != batch_size) {
    return false;
  }

  int64_t grouped_tokens = 0;
  for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    const int32_t q_len = draft_q_seq_lens[batch_idx];
    const int32_t read_state_id = read_state_ids[batch_idx];
    const int32_t write_state_id = write_state_ids[batch_idx];
    if (q_len < 1 || q_len > 2 || read_state_id < 0 ||
        read_state_id >= num_state_slots || write_state_id < 0 ||
        write_state_id >= num_state_slots ||
        (state_validity_mask[batch_idx] != 0 &&
         state_validity_mask[batch_idx] != 1)) {
      return false;
    }
    grouped_tokens += q_len;
  }
  if (grouped_tokens != total_tokens) {
    return false;
  }

  // Different logical sequences execute recurrent updates concurrently. A
  // write slot therefore cannot alias another sequence's read or write slot.
  for (size_t write_idx = 0; write_idx < batch_size; ++write_idx) {
    for (size_t other_idx = 0; other_idx < batch_size; ++other_idx) {
      if (write_idx == other_idx) {
        continue;
      }
      if (write_state_ids[write_idx] == write_state_ids[other_idx] ||
          write_state_ids[write_idx] == read_state_ids[other_idx]) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace detail
}  // namespace layer
}  // namespace xllm
