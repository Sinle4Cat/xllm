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

  const size_t expected_size = static_cast<size_t>(batch_size);
  for (size_t batch_idx = 0; batch_idx < expected_size; ++batch_idx) {
    if (q_seq_lens[batch_idx] != seq_len) {
      return false;
    }
  }
  return true;
}

}  // namespace detail
}  // namespace layer
}  // namespace xllm
