/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/utils.h"

namespace {

constexpr int64_t kSwaIntMax = 2147483647;

torch::Tensor ascend950_packed_causal_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale) {
  CHECK_EQ(actual_seq_lengths.size(), actual_seq_lengths_kv.size())
      << "query and key/value sequence counts must match";

  torch::Tensor output = torch::empty_like(query);
  int64_t query_start = 0;
  int64_t key_value_start = 0;
  for (size_t index = 0; index < actual_seq_lengths.size(); ++index) {
    const int64_t query_end = actual_seq_lengths[index];
    const int64_t key_value_end = actual_seq_lengths_kv[index];
    const int64_t query_length = query_end - query_start;
    const int64_t key_value_length = key_value_end - key_value_start;
    CHECK_EQ(query_length, key_value_length)
        << "Ascend950 torch attention fallback only supports non-chunked "
           "prefill";

    const torch::Tensor query_slice =
        query.narrow(0, query_start, query_length);
    torch::Tensor key_slice = key.narrow(0, key_value_start, key_value_length);
    torch::Tensor value_slice =
        value.narrow(0, key_value_start, key_value_length);
    key_slice = xllm::kernel::npu::expand_kv_heads(
        key_slice, num_heads, num_key_value_heads);
    value_slice = xllm::kernel::npu::expand_kv_heads(
        value_slice, num_heads, num_key_value_heads);

    const torch::Tensor query_4d = query_slice.permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor key_4d = key_slice.permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor value_4d = value_slice.permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor sequence_output =
        torch::scaled_dot_product_attention(query_4d,
                                            key_4d,
                                            value_4d,
                                            /*attn_mask=*/std::nullopt,
                                            /*dropout_p=*/0.0,
                                            /*is_causal=*/true,
                                            /*scale=*/scale)
            .squeeze(0)
            .permute({1, 0, 2});
    output.narrow(0, query_start, query_length).copy_(sequence_output);
    query_start = query_end;
    key_value_start = key_value_end;
  }

  CHECK_EQ(query_start, query.size(0));
  CHECK_EQ(key_value_start, key.size(0));
  return output;
}

torch::Tensor infer_attention_output(
    const torch::Tensor& query,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& block_table,
    int64_t num_heads,
    const std::string& input_layout) {
  if (input_layout == "TND" || input_layout == "NTD") {
    int64_t value_dim = query.size(-1);
    if (!block_table.has_value() && value.dim() >= 3) {
      value_dim = value.size(-1);
    }
    return torch::empty({query.size(0), num_heads, value_dim}, query.options());
  }

  if (input_layout == "BSH") {
    return torch::empty_like(query);
  }

  if (input_layout == "BNSD") {
    int64_t value_dim = query.size(-1);
    if (!block_table.has_value() && value.dim() >= 4) {
      value_dim = value.size(-1);
    }
    return torch::empty(
        {query.size(0), query.size(1), query.size(2), value_dim},
        query.options());
  }

  LOG(FATAL) << "Unsupported FIA input_layout: " << input_layout;
  return torch::Tensor();
}

torch::Tensor infer_softmax_lse(const torch::Tensor& query,
                                int64_t num_heads,
                                const std::string& input_layout,
                                bool softmax_lse_flag) {
  auto options = query.options().dtype(torch::kFloat32);
  if (!softmax_lse_flag) {
    return torch::empty({0}, options);
  }

  if (input_layout == "TND" || input_layout == "NTD") {
    return torch::empty({query.size(0), num_heads, 1}, options);
  }

  if (input_layout == "BSH") {
    return torch::empty({query.size(0), num_heads, query.size(1), 1}, options);
  }

  if (input_layout == "BNSD") {
    return torch::empty({query.size(0), query.size(1), query.size(2), 1},
                        options);
  }

  LOG(FATAL) << "Unsupported FIA input_layout: " << input_layout;
  return torch::Tensor();
}

std::optional<torch::Tensor> to_optional_tensor(
    const std::optional<torch::Tensor>& tensor_opt) {
  if (tensor_opt.has_value() && tensor_opt.value().defined()) {
    return tensor_opt.value();
  }
  return std::nullopt;
}

std::vector<c10::SymInt> to_symints(const std::vector<int64_t>& values) {
  std::vector<c10::SymInt> result;
  result.reserve(values.size());
  for (const int64_t value : values) {
    result.emplace_back(value);
  }
  return result;
}

void dispatch_fia_out_with_workspace(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& atten_mask,
    const std::optional<torch::Tensor>& block_table,
    const std::vector<c10::SymInt>& actual_seq_lengths,
    const std::vector<c10::SymInt>& actual_seq_lengths_kv,
    int64_t num_heads,
    double scale,
    int64_t num_key_value_heads,
    int64_t sparse_mode,
    int64_t block_size,
    bool softmax_lse_flag,
    const torch::Tensor& workspace,
    torch::Tensor& output,
    torch::Tensor& softmax_lse) {
  // The candidate torch_npu exports a generated C++ out wrapper whose
  // reference-return signature differs from its registered schema. Use the
  // boxed schema (the same path as Python's `.out`) so both torch_npu layouts
  // remain ABI-compatible.
  static const c10::OperatorHandle op =
      c10::Dispatcher::singleton().findSchemaOrThrow(
          "npu::npu_fused_infer_attention_score", "out");
  const c10::IValue none;
  std::vector<c10::IValue> stack;
  stack.reserve(42);
  stack.emplace_back(query);
  stack.emplace_back(key);
  stack.emplace_back(value);
  stack.emplace_back(none);  // pse_shift
  stack.emplace_back(atten_mask.has_value() ? c10::IValue(*atten_mask) : none);
  stack.emplace_back(actual_seq_lengths);
  stack.emplace_back(actual_seq_lengths_kv);
  for (int i = 0; i < 11; ++i) {
    stack.emplace_back(none);  // quantization inputs
  }
  stack.emplace_back(block_table.has_value() ? c10::IValue(*block_table)
                                             : none);
  for (int i = 0; i < 8; ++i) {
    stack.emplace_back(none);  // padding, shared-prefix, and rope inputs
  }
  stack.emplace_back(num_heads);
  stack.emplace_back(scale);
  stack.emplace_back(kSwaIntMax);
  stack.emplace_back(/*next_tokens=*/0);
  stack.emplace_back(std::string("BSND"));
  stack.emplace_back(num_key_value_heads);
  stack.emplace_back(sparse_mode);
  stack.emplace_back(/*inner_precise=*/0);
  stack.emplace_back(block_size);
  stack.emplace_back(/*antiquant_mode=*/0);
  stack.emplace_back(/*key_antiquant_mode=*/0);
  stack.emplace_back(/*value_antiquant_mode=*/0);
  stack.emplace_back(softmax_lse_flag);
  stack.emplace_back(workspace);
  stack.emplace_back(std::vector<at::Tensor>{output, softmax_lse});
  CHECK_EQ(stack.size(), 42);
  op.callBoxed(stack);
  CHECK_EQ(stack.size(), 2);
}

}  // namespace

namespace xllm::kernel::npu {

void npu_flash_attention_score_v4_out(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& attention_mask,
    int64_t num_heads,
    double scale,
    torch::Tensor& output) {
  check_tensor(query, "query", "npu_flash_attention_score_v4_out");
  check_tensor(key, "key", "npu_flash_attention_score_v4_out");
  check_tensor(value, "value", "npu_flash_attention_score_v4_out");
  CHECK_EQ(query.dim(), 4);
  CHECK_EQ(key.dim(), 4);
  CHECK_EQ(value.dim(), 4);
  CHECK_EQ(query.size(0), key.size(0));
  CHECK_EQ(key.sizes(), value.sizes());
  CHECK_EQ(query.size(1), num_heads);
  CHECK_EQ(output.sizes(), query.sizes());

  const std::optional<torch::Tensor> none_tensor = std::nullopt;
  const std::optional<torch::IntArrayRef> none_int_array = std::nullopt;
  const std::optional<torch::Tensor> mask = to_optional_tensor(attention_mask);
  torch::Tensor softmax_max =
      torch::empty({query.size(0), num_heads, query.size(2), 8},
                   query.options().dtype(torch::kFloat32));
  torch::Tensor softmax_sum = torch::empty_like(softmax_max);
  torch::Tensor softmax_out = torch::empty({0}, query.options());

  std::string input_layout = "BNSD";
  char* input_layout_ptr = const_cast<char*>(input_layout.c_str());
  std::string softmax_layout;
  char* softmax_layout_ptr = const_cast<char*>(softmax_layout.c_str());
  double keep_prob = 1.0;
  int64_t pre_tokens = kSwaIntMax;
  int64_t next_tokens = kSwaIntMax;
  int64_t inner_precise = 0;
  int64_t sparse_mode = 0;
  CHECK(query.scalar_type() == torch::kBFloat16 ||
        query.scalar_type() == torch::kFloat16)
      << "FlashAttentionScoreV4 graph path supports BF16/FP16 only";
  // FlashAttentionScoreV4 uses its own output dtype attribute encoding:
  // 0 = FLOAT16, 1 = BFLOAT16 (not aclDataType values).
  int64_t out_dtype = query.scalar_type() == torch::kBFloat16 ? 1 : 0;
  int64_t pse_type = 1;
  int64_t seed = 0;
  int64_t offset = 0;

  EXEC_NPU_CMD(aclnnFlashAttentionScoreV4,
               query,
               key,
               value,
               none_tensor,  // real_shift
               none_tensor,  // drop_mask
               none_tensor,  // padding_mask
               mask,
               none_tensor,  // query_rope
               none_tensor,  // key_rope
               none_tensor,  // d_scale_q
               none_tensor,  // d_scale_k
               none_tensor,  // d_scale_v
               none_tensor,  // sink
               none_int_array,
               none_int_array,
               none_int_array,
               none_int_array,
               none_int_array,
               scale,
               keep_prob,
               pre_tokens,
               next_tokens,
               num_heads,
               input_layout_ptr,
               inner_precise,
               sparse_mode,
               out_dtype,
               pse_type,
               softmax_layout_ptr,
               seed,
               offset,
               softmax_max,
               softmax_sum,
               softmax_out,
               output);
}

std::tuple<torch::Tensor, torch::Tensor> npu_fused_infer_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& atten_mask,
    const std::optional<torch::Tensor>& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size,
    int64_t sparse_mode,
    const std::string& input_layout,
    bool softmax_lse_flag,
    bool is_causal) {
  check_tensor(query, "query", "npu_fused_infer_attention");
  check_tensor(key, "key", "npu_fused_infer_attention");
  check_tensor(value, "value", "npu_fused_infer_attention");
  CHECK_GT(num_heads, 0) << "num_heads must be positive";
  CHECK(!actual_seq_lengths.empty()) << "actual_seq_lengths must not be empty";
  CHECK(!actual_seq_lengths_kv.empty())
      << "actual_seq_lengths_kv must not be empty";

  torch::Tensor output = infer_attention_output(
      query, value, block_table, num_heads, input_layout);
  torch::Tensor softmax_lse =
      infer_softmax_lse(query, num_heads, input_layout, softmax_lse_flag);

  npu_fused_infer_attention_out(query,
                                key,
                                value,
                                atten_mask,
                                block_table,
                                actual_seq_lengths,
                                actual_seq_lengths_kv,
                                num_heads,
                                num_key_value_heads,
                                scale,
                                block_size,
                                sparse_mode,
                                input_layout,
                                softmax_lse_flag,
                                is_causal,
                                output,
                                softmax_lse);

  return {output, softmax_lse};
}

void npu_fused_infer_attention_out(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& atten_mask,
    const std::optional<torch::Tensor>& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size,
    int64_t sparse_mode,
    const std::string& input_layout,
    bool softmax_lse_flag,
    bool is_causal,
    torch::Tensor& output,
    torch::Tensor& softmax_lse) {
  check_tensor(query, "query", "npu_fused_infer_attention_out");
  check_tensor(key, "key", "npu_fused_infer_attention_out");
  check_tensor(value, "value", "npu_fused_infer_attention_out");
  CHECK_GT(num_heads, 0) << "num_heads must be positive";
  CHECK(!actual_seq_lengths.empty()) << "actual_seq_lengths must not be empty";
  CHECK(!actual_seq_lengths_kv.empty())
      << "actual_seq_lengths_kv must not be empty";
  CHECK(output.defined()) << "output must be preallocated";
  CHECK(softmax_lse.defined()) << "softmax_lse must be preallocated";

  if (is_ascend950() && input_layout == "TND" && !block_table.has_value()) {
    CHECK(!softmax_lse_flag)
        << "Ascend950 torch attention fallback does not return softmax_lse";
    output.copy_(ascend950_packed_causal_attention(query,
                                                   key,
                                                   value,
                                                   actual_seq_lengths,
                                                   actual_seq_lengths_kv,
                                                   num_heads,
                                                   num_key_value_heads,
                                                   scale));
    return;
  }

  std::vector<torch::Tensor> key_tensors_vec{key};
  std::vector<torch::Tensor> value_tensors_vec{value};
  torch::TensorList key_tensors(key_tensors_vec);
  torch::TensorList value_tensors(value_tensors_vec);

  std::optional<torch::Tensor> none_tensor = std::nullopt;
  std::optional<torch::Tensor> atten_mask_tensor =
      to_optional_tensor(atten_mask);
  std::optional<torch::Tensor> block_table_tensor =
      to_optional_tensor(block_table);

  torch::IntArrayRef actual_seq_lengths_ref(actual_seq_lengths);
  torch::IntArrayRef actual_seq_lengths_kv_ref(actual_seq_lengths_kv);
  std::optional<torch::IntArrayRef> actual_seq_lengths_opt =
      actual_seq_lengths_ref;
  std::optional<torch::IntArrayRef> actual_seq_lengths_kv_opt =
      actual_seq_lengths_kv_ref;
  std::optional<torch::IntArrayRef> none_int_array = std::nullopt;

  std::string layout = input_layout;
  char* input_layout_ptr = const_cast<char*>(layout.c_str());
  int64_t pre_tokens = kSwaIntMax;
  // Paged decode queries represent the newest token. Even though the
  // one-token attention call does not need a causal mask tensor, FIA must not
  // expose unwritten positions from the tail of the current KV block.
  int64_t next_tokens = 0;
  int64_t inner_precise = 0;
  int64_t antiquant_mode = 0;
  int64_t key_antiquant_mode = 0;
  int64_t value_antiquant_mode = 0;
  int64_t query_quant_mode = 0;
  int64_t pse_type = 0;

  if (is_ascend950()) {
    EXEC_NPU_CMD(aclnnFusedInferAttentionScoreV5,
                 query,
                 key_tensors,
                 value_tensors,
                 none_tensor,  // pse_shift
                 atten_mask_tensor,
                 actual_seq_lengths_opt,
                 actual_seq_lengths_kv_opt,
                 none_tensor,  // dequant_scale1
                 none_tensor,  // quant_scale1
                 none_tensor,  // dequant_scale2
                 none_tensor,  // quant_scale2
                 none_tensor,  // quant_offset2
                 none_tensor,  // antiquant_scale
                 none_tensor,  // antiquant_offset
                 block_table_tensor,
                 none_tensor,     // query_padding_size
                 none_tensor,     // kv_padding_size
                 none_tensor,     // key_antiquant_scale
                 none_tensor,     // key_antiquant_offset
                 none_tensor,     // value_antiquant_scale
                 none_tensor,     // value_antiquant_offset
                 none_tensor,     // key_shared_prefix
                 none_tensor,     // value_shared_prefix
                 none_int_array,  // actual_shared_prefix_len
                 none_tensor,     // query_rope
                 none_tensor,     // key_rope
                 none_tensor,     // key_rope_antiquant_scale
                 none_tensor,     // dequant_scale_query
                 none_tensor,     // learnable_sink
                 none_int_array,  // q_start_idx
                 none_int_array,  // kv_start_idx
                 num_heads,
                 scale,
                 pre_tokens,
                 next_tokens,
                 input_layout_ptr,
                 num_key_value_heads,
                 sparse_mode,
                 inner_precise,
                 block_size,
                 antiquant_mode,
                 softmax_lse_flag,
                 key_antiquant_mode,
                 value_antiquant_mode,
                 query_quant_mode,
                 pse_type,
                 output,
                 softmax_lse);
    return;
  }

  EXEC_NPU_CMD(aclnnFusedInferAttentionScoreV3,
               query,
               key_tensors,
               value_tensors,
               none_tensor,  // pse_shift
               atten_mask_tensor,
               actual_seq_lengths_opt,
               actual_seq_lengths_kv_opt,
               none_tensor,  // dequant_scale1
               none_tensor,  // quant_scale1
               none_tensor,  // dequant_scale2
               none_tensor,  // quant_scale2
               none_tensor,  // quant_offset2
               none_tensor,  // antiquant_scale
               none_tensor,  // antiquant_offset
               block_table_tensor,
               none_tensor,     // query_padding_size
               none_tensor,     // kv_padding_size
               none_tensor,     // key_antiquant_scale
               none_tensor,     // key_antiquant_offset
               none_tensor,     // value_antiquant_scale
               none_tensor,     // value_antiquant_offset
               none_tensor,     // key_shared_prefix
               none_tensor,     // value_shared_prefix
               none_int_array,  // actual_shared_prefix_len
               none_tensor,     // query_rope
               none_tensor,     // key_rope
               none_tensor,     // key_rope_antiquant_scale
               num_heads,
               scale,
               pre_tokens,
               next_tokens,
               input_layout_ptr,
               num_key_value_heads,
               sparse_mode,
               inner_precise,
               block_size,
               antiquant_mode,
               softmax_lse_flag,
               key_antiquant_mode,
               value_antiquant_mode,
               output,
               softmax_lse);
}

torch::Tensor npu_fused_infer_attention_graph_workspace(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& atten_mask,
    const std::optional<torch::Tensor>& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size,
    int64_t sparse_mode,
    const std::string& input_layout,
    bool softmax_lse_flag) {
  CHECK(is_ascend950())
      << "persistent FIA graph workspace is only enabled on Ascend950";
  check_tensor(query, "query", "npu_fused_infer_attention_graph_workspace");
  check_tensor(key, "key", "npu_fused_infer_attention_graph_workspace");
  check_tensor(value, "value", "npu_fused_infer_attention_graph_workspace");
  CHECK(!actual_seq_lengths.empty()) << "actual_seq_lengths must not be empty";
  CHECK(!actual_seq_lengths_kv.empty())
      << "actual_seq_lengths_kv must not be empty";

  const std::optional<torch::Tensor> none_tensor = std::nullopt;
  const std::optional<torch::Tensor> atten_mask_tensor =
      to_optional_tensor(atten_mask);
  const std::optional<torch::Tensor> block_table_tensor =
      to_optional_tensor(block_table);
  const std::vector<c10::SymInt> query_lengths = to_symints(actual_seq_lengths);
  const std::vector<c10::SymInt> key_value_lengths =
      to_symints(actual_seq_lengths_kv);
  const at::OptionalSymIntArrayRef no_lengths = std::nullopt;

  return at_npu::native::custom_ops::
      _npu_fused_infer_attention_score_get_max_workspace(
          query,
          key,
          value,
          none_tensor,  // pse_shift
          atten_mask_tensor,
          c10::SymIntArrayRef(query_lengths),
          c10::SymIntArrayRef(key_value_lengths),
          none_tensor,  // dequant_scale1
          none_tensor,  // quant_scale1
          none_tensor,  // dequant_scale2
          none_tensor,  // quant_scale2
          none_tensor,  // quant_offset2
          none_tensor,  // antiquant_scale
          none_tensor,  // antiquant_offset
          none_tensor,  // key_antiquant_scale
          none_tensor,  // key_antiquant_offset
          none_tensor,  // value_antiquant_scale
          none_tensor,  // value_antiquant_offset
          block_table_tensor,
          none_tensor,  // query_padding_size
          none_tensor,  // kv_padding_size
          none_tensor,  // key_shared_prefix
          none_tensor,  // value_shared_prefix
          no_lengths,   // actual_shared_prefix_len
          none_tensor,  // query_rope
          none_tensor,  // key_rope
          none_tensor,  // key_rope_antiquant_scale
          num_heads,
          scale,
          kSwaIntMax,
          /*next_tokens=*/0,
          input_layout,
          num_key_value_heads,
          sparse_mode,
          /*inner_precise=*/0,
          block_size,
          /*antiquant_mode=*/0,
          /*key_antiquant_mode=*/0,
          /*value_antiquant_mode=*/0,
          softmax_lse_flag);
}

void npu_fused_infer_attention_graph_out(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& atten_mask,
    const std::optional<torch::Tensor>& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size,
    int64_t sparse_mode,
    const std::string& input_layout,
    bool softmax_lse_flag,
    const torch::Tensor& workspace,
    torch::Tensor& output,
    torch::Tensor& softmax_lse) {
  CHECK(is_ascend950())
      << "persistent FIA graph workspace is only enabled on Ascend950";
  check_tensor(query, "query", "npu_fused_infer_attention_graph_out");
  check_tensor(key, "key", "npu_fused_infer_attention_graph_out");
  check_tensor(value, "value", "npu_fused_infer_attention_graph_out");
  check_tensor(workspace, "workspace", "npu_fused_infer_attention_graph_out");
  CHECK(output.defined()) << "output must be preallocated";
  CHECK(softmax_lse.defined()) << "softmax_lse must be preallocated";

  const std::optional<torch::Tensor> atten_mask_tensor =
      to_optional_tensor(atten_mask);
  const std::optional<torch::Tensor> block_table_tensor =
      to_optional_tensor(block_table);
  const std::vector<c10::SymInt> query_lengths = to_symints(actual_seq_lengths);
  const std::vector<c10::SymInt> key_value_lengths =
      to_symints(actual_seq_lengths_kv);
  CHECK_EQ(input_layout, "BSND")
      << "persistent FIA graph update currently supports BSND only";
  dispatch_fia_out_with_workspace(query,
                                  key,
                                  value,
                                  atten_mask_tensor,
                                  block_table_tensor,
                                  query_lengths,
                                  key_value_lengths,
                                  num_heads,
                                  scale,
                                  num_key_value_heads,
                                  sparse_mode,
                                  block_size,
                                  softmax_lse_flag,
                                  workspace,
                                  output,
                                  softmax_lse);
}

}  // namespace xllm::kernel::npu
