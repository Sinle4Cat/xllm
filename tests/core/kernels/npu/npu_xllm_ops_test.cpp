/* Copyright 2026 The xLLM Authors.

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

// NPU acceptance test for the xllm_ops torch-op library.
//
// Mirrors the CUDA xllm_ops_test: verifies TORCH_LIBRARY registrations survive
// linking on NPU (PrivateUse1), ops are callable via the dispatcher, and the
// embedded Python interpreter sees torch.ops.xllm_ops.*.

#include <acl/acl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/xllm_torch_ops.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace py = pybind11;

namespace xllm::kernel::npu {
std::pair<torch::Tensor, torch::Tensor> apply_npu_partial_rotary_embedding(
    const torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int64_t head_size,
    int64_t rotary_dim,
    const torch::Tensor& cos_sin_cache,
    bool is_neox_style);
}  // namespace xllm::kernel::npu

namespace xllm {
namespace {

torch::Tensor rms_norm_reference(const torch::Tensor& input,
                                 const torch::Tensor& weight,
                                 double eps) {
  auto x = input.to(torch::kFloat32);
  auto var = x.pow(2).mean(-1, /*keepdim=*/true);
  auto normed = x * torch::rsqrt(var + eps);
  return (normed * weight.to(torch::kFloat32)).to(input.scalar_type());
}

torch::Tensor silu_and_mul_reference(const torch::Tensor& input) {
  const int64_t d = input.size(-1) / 2;
  auto a = input.slice(-1, 0, d);
  auto b = input.slice(-1, d, 2 * d);
  return (a * torch::sigmoid(a)) * b;
}

void prepend_python_model_path() {
  std::filesystem::path repo_root(__FILE__);
  for (int i = 0; i < 5; ++i) {
    repo_root = repo_root.parent_path();
  }
  const std::string python_model_path = repo_root.string();
  py::list sys_path = py::module_::import("sys").attr("path");
  sys_path.attr("insert")(0, python_model_path);
}

bool is_npu_available() {
  return c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
             ->deviceCount() > 0;
}

bool is_ascend950_device() {
  const char* soc_name = aclrtGetSocName();
  return soc_name != nullptr &&
         std::string(soc_name).find("Ascend950") != std::string::npos;
}

torch::Tensor expand_kv_heads_reference(const torch::Tensor& tensor,
                                        int64_t num_heads) {
  const int64_t num_kv_heads = tensor.size(1);
  EXPECT_EQ(num_heads % num_kv_heads, 0);
  const int64_t expansion_factor = num_heads / num_kv_heads;
  return tensor.unsqueeze(2)
      .expand({tensor.size(0), num_kv_heads, expansion_factor, tensor.size(2)})
      .reshape({tensor.size(0), num_heads, tensor.size(2)});
}

torch::Tensor packed_causal_attention_reference(const torch::Tensor& query,
                                                const torch::Tensor& key,
                                                const torch::Tensor& value,
                                                double scale) {
  const auto query_float = query.to(torch::kFloat32).permute({1, 0, 2});
  const auto key_float =
      expand_kv_heads_reference(key.to(torch::kFloat32), query.size(1));
  const auto value_float =
      expand_kv_heads_reference(value.to(torch::kFloat32), query.size(1));
  auto scores =
      torch::matmul(query_float, key_float.permute({1, 2, 0})) * scale;
  const auto causal_mask =
      torch::ones({query.size(0), key.size(0)}, torch::kBool).triu(1);
  scores.masked_fill_(causal_mask, -std::numeric_limits<float>::infinity());
  return torch::matmul(torch::softmax(scores, -1),
                       value_float.permute({1, 0, 2}))
      .permute({1, 0, 2});
}

torch::Tensor decode_attention_reference(const torch::Tensor& query,
                                         const torch::Tensor& key,
                                         const torch::Tensor& value,
                                         double scale) {
  const auto query_float = query.to(torch::kFloat32).squeeze(0);
  const auto key_float =
      expand_kv_heads_reference(key.to(torch::kFloat32), query.size(1));
  const auto value_float =
      expand_kv_heads_reference(value.to(torch::kFloat32), query.size(1));
  const auto scores =
      torch::matmul(query_float.unsqueeze(1), key_float.permute({1, 2, 0})) *
      scale;
  return torch::matmul(torch::softmax(scores, -1),
                       value_float.permute({1, 0, 2}))
      .squeeze(1)
      .unsqueeze(0);
}

class NpuXllmOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    xllm::ensure_xllm_torch_ops_registered();
    if (!is_npu_available()) {
      GTEST_SKIP() << "NPU not available; skipping xllm_ops NPU test.";
    }
    if (!Py_IsInitialized()) {
      setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
      Py_InitializeEx(0);
    }
    py::gil_scoped_acquire gil;
    prepend_python_model_path();
    py::module_::import("xllm.python._npu_bootstrap");
    py::module_::import("xllm.python").attr("initialize_runtime")();
  }
};

TEST_F(NpuXllmOpsTest, DispatcherRmsNormMatchesReference) {
  py::gil_scoped_acquire gil;
  auto opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kPrivateUse1);
  auto input = torch::randn({8, 128}, opts);
  auto weight = torch::randn({128}, opts);
  const double eps = 1e-6;

  auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("xllm_ops::rms_norm", "");
  auto out = op.typed<torch::Tensor(
      const torch::Tensor&, const torch::Tensor&, double)>()
                 .call(input, weight, eps);

  auto ref = rms_norm_reference(input, weight, eps);
  EXPECT_TRUE(
      torch::allclose(out.cpu(), ref.cpu(), /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.cpu().to(torch::kFloat32) - ref.cpu().to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, DispatcherSiluAndMulMatchesReference) {
  py::gil_scoped_acquire gil;
  auto opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kPrivateUse1);
  auto gate_up = torch::randn({8, 256}, opts);

  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "xllm_ops::silu_and_mul", "");
  auto out = op.typed<torch::Tensor(const torch::Tensor&)>().call(gate_up);

  auto ref = silu_and_mul_reference(gate_up);
  ASSERT_EQ(out.size(-1), 128);
  EXPECT_TRUE(
      torch::allclose(out.cpu(), ref.cpu(), /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.cpu().to(torch::kFloat32) - ref.cpu().to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, DispatcherQuantizeMatchesStaticW8A8Reference) {
  py::gil_scoped_acquire gil;
  const auto input_opts = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto scale_opts = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto zero_point_opts = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto input_cpu = torch::tensor(
      std::vector<float>{-40.0F, -1.0F, -0.25F, 0.0F, 0.25F, 1.0F, 40.0F},
      input_opts);
  const auto scale_cpu = torch::full({1}, 0.25, scale_opts);
  const auto zero_point_cpu = torch::full({1}, 2, zero_point_opts);
  const auto input = input_cpu.to(torch::kPrivateUse1);
  const auto scale = scale_cpu.to(torch::kPrivateUse1);
  const auto zero_point = zero_point_cpu.to(torch::kPrivateUse1);

  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "xllm_ops::quantize_per_tensor", "");
  auto actual = op.typed<torch::Tensor(const torch::Tensor&,
                                       const torch::Tensor&,
                                       const torch::Tensor&,
                                       at::ScalarType,
                                       int64_t)>()
                    .call(input, scale, zero_point, at::ScalarType::QInt8, -1);

  const auto expected =
      torch::clamp(torch::round(input_cpu.to(torch::kFloat32) /
                                    scale_cpu.to(torch::kFloat32) +
                                zero_point_cpu.to(torch::kFloat32)),
                   -128,
                   127)
          .to(torch::kInt8);
  EXPECT_EQ(actual.scalar_type(), torch::kInt8);
  EXPECT_TRUE(torch::equal(actual.cpu(), expected));
}

TEST_F(NpuXllmOpsTest, EmbeddedInterpreterSeesOps) {
  py::gil_scoped_acquire gil;

  auto opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kPrivateUse1);
  auto gate_up = torch::randn({8, 256}, opts);

  py::module_ torch_mod = py::module_::import("torch");
  py::object xllm_ops = torch_mod.attr("ops").attr("xllm_ops");
  py::object out_obj = xllm_ops.attr("silu_and_mul")(gate_up);
  auto out = out_obj.cast<torch::Tensor>();

  auto ref = silu_and_mul_reference(gate_up);
  ASSERT_EQ(out.size(-1), 128);
  EXPECT_TRUE(
      torch::allclose(out.cpu(), ref.cpu(), /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max abs diff = "
      << (out.cpu().to(torch::kFloat32) - ref.cpu().to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, Qwen35_27B_TP4_FullAttentionMatchesReference) {
  py::gil_scoped_acquire gil;
  if (!is_ascend950_device()) {
    GTEST_SKIP() << "Ascend950 is required for the A5 attention path.";
  }

  constexpr int64_t kSequenceLength = 129;
  constexpr int64_t kQueryHeads = 6;
  constexpr int64_t kKvHeads = 1;
  constexpr int64_t kHeadDim = 256;
  constexpr double kScale = 1.0 / 16.0;
  torch::manual_seed(20260729);

  const auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  const auto query_cpu =
      (0.25 * torch::randn({kSequenceLength, kQueryHeads, kHeadDim}, cpu_float))
          .to(torch::kBFloat16);
  const auto key_cpu =
      (0.25 * torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float))
          .to(torch::kBFloat16);
  const auto value_cpu =
      torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float)
          .to(torch::kBFloat16);
  const auto query = query_cpu.to(torch::kPrivateUse1);
  const auto key = key_cpu.to(torch::kPrivateUse1);
  const auto value = value_cpu.to(torch::kPrivateUse1);

  const auto [actual, softmax_lse] =
      xllm::kernel::npu::npu_fused_infer_attention(query,
                                                   key,
                                                   value,
                                                   std::nullopt,
                                                   std::nullopt,
                                                   {kSequenceLength},
                                                   {kSequenceLength},
                                                   kQueryHeads,
                                                   kKvHeads,
                                                   kScale,
                                                   /*block_size=*/128,
                                                   /*sparse_mode=*/0,
                                                   /*input_layout=*/"TND",
                                                   /*softmax_lse_flag=*/false);
  const auto expected =
      packed_causal_attention_reference(query_cpu, key_cpu, value_cpu, kScale);

  EXPECT_EQ(actual.sizes(), query.sizes());
  EXPECT_EQ(softmax_lse.numel(), 0);
  EXPECT_TRUE(torch::allclose(actual.cpu().to(torch::kFloat32),
                              expected,
                              /*rtol=*/5e-2,
                              /*atol=*/5e-2))
      << "max abs diff = "
      << (actual.cpu().to(torch::kFloat32) - expected)
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, Qwen35_27B_TP4_KvCacheCrosses128TokenBoundary) {
  py::gil_scoped_acquire gil;
  if (!is_ascend950_device()) {
    GTEST_SKIP() << "Ascend950 is required for the A5 paged-cache path.";
  }

  constexpr int64_t kSequenceLength = 130;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kNumPhysicalBlocks = 3;
  constexpr int64_t kQueryHeads = 6;
  constexpr int64_t kKvHeads = 1;
  constexpr int64_t kHeadDim = 256;
  constexpr double kScale = 1.0 / 16.0;
  torch::manual_seed(20260730);

  const auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  const auto key_cpu =
      torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float)
          .to(torch::kBFloat16);
  const auto value_cpu =
      torch::randn({kSequenceLength, kKvHeads, kHeadDim}, cpu_float)
          .to(torch::kBFloat16);
  const auto query_cpu =
      (0.25 * torch::randn({1, kQueryHeads, kHeadDim}, cpu_float))
          .to(torch::kBFloat16);
  auto key = key_cpu.to(torch::kPrivateUse1);
  auto value_tensor = value_cpu.to(torch::kPrivateUse1);
  std::optional<torch::Tensor> value = value_tensor;

  const auto npu_bfloat = torch::TensorOptions()
                              .dtype(torch::kBFloat16)
                              .device(torch::kPrivateUse1);
  auto key_cache = torch::zeros(
      {kNumPhysicalBlocks, kBlockSize, kKvHeads, kHeadDim}, npu_bfloat);
  auto value_cache_tensor = torch::zeros_like(key_cache);
  std::optional<torch::Tensor> value_cache = value_cache_tensor;

  const auto first_block_slots =
      torch::arange(2 * kBlockSize,
                    3 * kBlockSize,
                    torch::TensorOptions().dtype(torch::kInt32));
  const auto second_block_slots =
      torch::arange(0, 2, torch::TensorOptions().dtype(torch::kInt32));
  const auto slot_mapping = torch::cat({first_block_slots, second_block_slots})
                                .to(torch::kPrivateUse1);
  xllm::kernel::npu::reshape_paged_cache(
      key, value, key_cache, value_cache, slot_mapping);

  auto expected_key_cache =
      torch::zeros({kNumPhysicalBlocks, kBlockSize, kKvHeads, kHeadDim},
                   torch::TensorOptions().dtype(torch::kBFloat16));
  auto expected_value_cache = torch::zeros_like(expected_key_cache);
  expected_key_cache[2].copy_(key_cpu.narrow(0, 0, kBlockSize));
  expected_value_cache[2].copy_(value_cpu.narrow(0, 0, kBlockSize));
  expected_key_cache[0].narrow(0, 0, 2).copy_(key_cpu.narrow(0, kBlockSize, 2));
  expected_value_cache[0].narrow(0, 0, 2).copy_(
      value_cpu.narrow(0, kBlockSize, 2));

  EXPECT_TRUE(torch::equal(key_cache.cpu(), expected_key_cache));
  EXPECT_TRUE(torch::equal(value_cache.value().cpu(), expected_value_cache));

  const auto query = query_cpu.to(torch::kPrivateUse1);
  const auto block_table =
      torch::tensor({{2, 0}}, torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kPrivateUse1);
  const auto seq_lens =
      torch::tensor({kSequenceLength},
                    torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::kPrivateUse1);
  auto actual = torch::empty_like(query);
  xllm::kernel::npu::batch_decode(query,
                                  key_cache,
                                  value_cache.value(),
                                  kScale,
                                  block_table,
                                  seq_lens,
                                  actual);
  const auto expected =
      decode_attention_reference(query_cpu, key_cpu, value_cpu, kScale);

  EXPECT_TRUE(torch::allclose(actual.cpu().to(torch::kFloat32),
                              expected,
                              /*rtol=*/5e-2,
                              /*atol=*/5e-2))
      << "max abs diff = "
      << (actual.cpu().to(torch::kFloat32) - expected)
             .abs()
             .max()
             .item<float>();
}

TEST_F(NpuXllmOpsTest, ModelExecutorUsesExplicitRuntimeBatchLimit) {
  py::gil_scoped_acquire gil;
  prepend_python_model_path();

  py::exec(R"PY(
import torch
from unittest.mock import patch

from xllm.python.layers.attention import Attention
from xllm.python.model_executor import executor as executor_module


class FakeBackend:
    def __init__(self, **kwargs):
        pass

    def bind_kv_caches(self, kv_caches):
        pass

    def prepare(self, metadata, *, graph_mode=False):
        pass

    def execute(self, q, k, v, layer):
        return q

    @property
    def num_kv_blocks(self):
        return 0

    @property
    def page_size(self):
        return 1


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(1, device="privateuseone:0")
        )
        self.attention = Attention(1, 1, 8, 1.0, 0, 0)
        self.model = torch.nn.Identity()


with patch.object(
    executor_module, "_create_attention_backend", return_value=FakeBackend()
):
    model_executor = executor_module.ModelExecutor(
        FakeModel(),
        {"python_graph_backend": "off"},
        max_seqs_per_batch=3,
    )
    assert model_executor._num_attention_layers == 1
    assert model_executor.decode_graph_runner is None
    assert model_executor.inductor_runner is None
)PY");
}

TEST_F(NpuXllmOpsTest, LightningIndexerOutKeepsBuffersAcrossGraphReplay) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch

query = torch.randn(
    (1, 64, 128), dtype=torch.bfloat16, device="privateuseone:0"
)
key = torch.randn(
    (1, 16, 1, 128), dtype=torch.bfloat16, device="privateuseone:0"
)
weights = torch.randn(
    (1, 64), dtype=torch.bfloat16, device="privateuseone:0"
)
query_seq_lengths = torch.tensor(
    [1], dtype=torch.int32, device="privateuseone:0"
)
key_seq_lengths = torch.tensor(
    [16], dtype=torch.int32, device="privateuseone:0"
)
block_table = torch.tensor(
    [[0]], dtype=torch.int32, device="privateuseone:0"
)
sparse_indices = torch.empty(
    (1, 1, 4), dtype=torch.int32, device="privateuseone:0"
)
sparse_values = torch.empty(
    (1, 1, 4), dtype=torch.bfloat16, device="privateuseone:0"
)
indices_address = sparse_indices.data_ptr()
values_address = sparse_values.data_ptr()


def run_indexer():
    return torch.ops.xllm_ops.lightning_indexer_out(
        query,
        key,
        weights,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        "TND",
        "PA_BSND",
        4,
        3,
        2**63 - 1,
        2**63 - 1,
        False,
        sparse_indices,
        sparse_values,
    )


eager_result = run_indexer()
assert eager_result.shape == (1, 1, 4)
assert eager_result.dtype == torch.int32
assert eager_result.data_ptr() == indices_address
assert sparse_values.shape == (1, 1, 4)
assert sparse_values.dtype == torch.bfloat16
assert sparse_values.data_ptr() == values_address

stream = torch.npu.Stream()
graph = torch.npu.NPUGraph()
with torch.npu.stream(stream):
    run_indexer()
torch.npu.synchronize()
with torch.npu.stream(stream):
    with torch.npu.graph(graph, stream=stream):
        graph_result = run_indexer()
torch.npu.synchronize()

with torch.npu.stream(stream):
    query.add_(0.25)
    graph.replay()
    query.sub_(0.5)
    graph.replay()
torch.npu.synchronize()

assert graph_result.data_ptr() == indices_address
assert sparse_indices.data_ptr() == indices_address
assert sparse_values.data_ptr() == values_address
)PY");
}

TEST_F(NpuXllmOpsTest, SparseFlashAttentionOutKeepsBufferAcrossGraphReplay) {
  py::gil_scoped_acquire gil;

  py::exec(R"PY(
import torch

query = torch.randn(
    (1, 8, 512), dtype=torch.bfloat16, device="privateuseone:0"
)
key = torch.randn(
    (1, 16, 1, 512), dtype=torch.bfloat16, device="privateuseone:0"
)
value = torch.randn_like(key)
sparse_indices = torch.tensor(
    [[[0, 1, 2, 3]]], dtype=torch.int32, device="privateuseone:0"
)
block_table = torch.tensor(
    [[0]], dtype=torch.int32, device="privateuseone:0"
)
actual_seq_lengths_query = torch.tensor(
    [1], dtype=torch.int32, device="privateuseone:0"
)
actual_seq_lengths_kv = torch.tensor(
    [16], dtype=torch.int32, device="privateuseone:0"
)
query_rope = torch.randn(
    (1, 8, 64), dtype=torch.bfloat16, device="privateuseone:0"
)
key_rope = torch.randn(
    (1, 16, 1, 64), dtype=torch.bfloat16, device="privateuseone:0"
)
output = torch.empty_like(query)
output_address = output.data_ptr()


def run_attention():
    return torch.ops.xllm_ops.sparse_flash_attention_out(
        query,
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        1.0 / 16.0,
        1,
        "TND",
        "PA_BSND",
        3,
        output,
    )


eager_result = run_attention()
assert eager_result.shape == query.shape
assert eager_result.dtype == query.dtype
assert eager_result.data_ptr() == output_address

stream = torch.npu.Stream()
graph = torch.npu.NPUGraph()
with torch.npu.stream(stream):
    run_attention()
torch.npu.synchronize()
with torch.npu.stream(stream):
    with torch.npu.graph(graph, stream=stream):
        graph_result = run_attention()
torch.npu.synchronize()

with torch.npu.stream(stream):
    query.add_(0.25)
    graph.replay()
    query.sub_(0.5)
    graph.replay()
torch.npu.synchronize()

assert graph_result.data_ptr() == output_address
assert output.data_ptr() == output_address
)PY");
}

TEST_F(NpuXllmOpsTest, FlashAttentionScoreV4MaskedDecodeMatchesSdpa) {
  if (!is_npu_available() || !is_ascend950_device()) {
    GTEST_SKIP() << "requires an Ascend950 NPU";
  }

  constexpr int64_t kBatch = 1;
  constexpr int64_t kHeads = 6;
  constexpr int64_t kSeqLen = 384;
  constexpr int64_t kValidLen = 133;
  constexpr int64_t kHeadDim = 256;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  auto options = torch::TensorOptions()
                     .device(torch::kPrivateUse1)
                     .dtype(torch::kBFloat16);
  torch::manual_seed(20260817);
  auto query = torch::randn({kBatch, kHeads, 1, kHeadDim}, options);
  auto key = torch::randn({kBatch, kHeads, kSeqLen, kHeadDim}, options);
  auto value = torch::randn_like(key);
  auto positions = torch::arange(
      kSeqLen,
      torch::TensorOptions().device(torch::kPrivateUse1).dtype(torch::kInt32));
  auto acl_mask = positions.ge(kValidLen).view({kBatch, 1, 1, kSeqLen});

  auto output = torch::empty_like(query);
  kernel::npu::npu_flash_attention_score_v4_out(
      query, key, value, acl_mask, kHeads, scale, output);
  auto reference =
      torch::scaled_dot_product_attention(query,
                                          key.narrow(2, 0, kValidLen),
                                          value.narrow(2, 0, kValidLen),
                                          /*attn_mask=*/std::nullopt,
                                          /*dropout_p=*/0.0,
                                          /*is_causal=*/false,
                                          /*scale=*/scale);
  c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
      ->synchronizeDevice(0);
  auto error =
      (output.to(torch::kFloat32) - reference.to(torch::kFloat32)).abs().cpu();
  LOG(INFO) << "FlashAttentionScoreV4 vs SDPA max_abs="
            << error.max().item<float>()
            << ", mean_abs=" << error.mean().item<float>();
  EXPECT_TRUE(torch::allclose(output.to(torch::kFloat32),
                              reference.to(torch::kFloat32),
                              /*rtol=*/1e-3,
                              /*atol=*/1e-3));
}

TEST_F(NpuXllmOpsTest, Ascend950PagedAttentionDeviceLengthsMatchCpuLengths) {
  if (!is_npu_available() || !is_ascend950_device()) {
    GTEST_SKIP() << "requires an Ascend950 NPU";
  }

  constexpr int64_t kBatch = 2;
  constexpr int64_t kHeads = 6;
  constexpr int64_t kKvHeads = 2;
  constexpr int64_t kHeadDim = 256;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kNumBlocks = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  auto options = torch::TensorOptions()
                     .device(torch::kPrivateUse1)
                     .dtype(torch::kBFloat16);
  torch::manual_seed(20260817);
  auto query = torch::randn({kBatch, kHeads, kHeadDim}, options);
  auto key_cache =
      torch::randn({kNumBlocks, kBlockSize, kKvHeads, kHeadDim}, options);
  auto value_cache = torch::randn_like(key_cache);
  auto block_table_cpu = torch::tensor(
      {{2, 5, 1}, {7, 0, 4}}, torch::TensorOptions().dtype(torch::kInt32));
  auto block_table = block_table_cpu.to(torch::kPrivateUse1);
  auto seq_lens_cpu =
      torch::tensor({133, 259}, torch::TensorOptions().dtype(torch::kInt32));
  auto seq_lens_device = seq_lens_cpu.to(torch::kPrivateUse1);

  auto eager_output = torch::empty_like(query);
  auto device_output = torch::empty_like(query);
  kernel::npu::batch_decode(query,
                            key_cache,
                            value_cache,
                            scale,
                            block_table,
                            seq_lens_cpu,
                            eager_output);
  kernel::npu::batch_decode(query,
                            key_cache,
                            value_cache,
                            scale,
                            block_table,
                            seq_lens_device,
                            device_output);
  c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
      ->synchronizeDevice(0);
  auto error =
      (device_output.to(torch::kFloat32) - eager_output.to(torch::kFloat32))
          .abs()
          .cpu();
  LOG(INFO) << "Ascend950 device-length paged attention vs CPU-length path "
            << "max_abs=" << error.max().item<float>()
            << ", mean_abs=" << error.mean().item<float>()
            << ", mismatches=" << error.ne(0).sum().item<int64_t>();
  EXPECT_TRUE(torch::equal(device_output, eager_output));
}

TEST_F(NpuXllmOpsTest, Ascend950PagedAttentionGraphReplayTracksInputs) {
  if (!is_npu_available() || !is_ascend950_device()) {
    GTEST_SKIP() << "requires an Ascend950 NPU";
  }

  constexpr int64_t kBatch = 2;
  constexpr int64_t kHeads = 6;
  constexpr int64_t kKvHeads = 2;
  constexpr int64_t kHeadDim = 256;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kNumBlocks = 10;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  auto npu_options = torch::TensorOptions()
                         .device(torch::kPrivateUse1)
                         .dtype(torch::kBFloat16);
  auto cpu_int_options = torch::TensorOptions().dtype(torch::kInt32);
  torch::manual_seed(20260817);

  const auto query_base = torch::randn({kBatch, kHeads, kHeadDim}, npu_options);
  const auto query_changed =
      torch::randn({kBatch, kHeads, kHeadDim}, npu_options);
  const auto key_cache_base =
      torch::randn({kNumBlocks, kBlockSize, kKvHeads, kHeadDim}, npu_options);
  const auto value_cache_base = torch::randn_like(key_cache_base);
  const auto key_cache_changed = torch::randn_like(key_cache_base);
  const auto value_cache_changed = torch::randn_like(value_cache_base);
  const auto block_table_base_cpu =
      torch::tensor({{2, 5, 1}, {7, 0, 4}}, cpu_int_options);
  const auto block_table_changed_cpu =
      torch::tensor({{6, 3, 8}, {1, 9, 2}}, cpu_int_options);
  const auto seq_lens_base_cpu = torch::tensor({133, 259}, cpu_int_options);
  const auto seq_lens_changed_cpu = torch::tensor({257, 129}, cpu_int_options);

  // These tensors retain their addresses for the full lifetime of the graph.
  auto graph_query = query_base.clone();
  auto graph_key_cache = key_cache_base.clone();
  auto graph_value_cache = value_cache_base.clone();
  auto graph_block_table = block_table_base_cpu.to(torch::kPrivateUse1);
  auto graph_seq_lens = seq_lens_base_cpu.to(torch::kPrivateUse1);
  auto graph_output = torch::empty_like(graph_query);

  c10_npu::NPUStream capture_stream = c10_npu::getStreamFromPool(true, 0);
  c10_npu::NPUGraph graph;
  {
    c10_npu::NPUStreamGuard stream_guard(capture_stream);
    kernel::npu::batch_decode(graph_query,
                              graph_key_cache,
                              graph_value_cache,
                              scale,
                              graph_block_table,
                              graph_seq_lens,
                              graph_output);
  }
  capture_stream.synchronize();
  {
    c10_npu::NPUStreamGuard stream_guard(capture_stream);
    graph.capture_begin(
        {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    kernel::npu::batch_decode(graph_query,
                              graph_key_cache,
                              graph_value_cache,
                              scale,
                              graph_block_table,
                              graph_seq_lens,
                              graph_output);
    graph.capture_end();
  }
  capture_stream.synchronize();

  auto replay_and_compare = [&](const char* name,
                                const torch::Tensor& query,
                                const torch::Tensor& key_cache,
                                const torch::Tensor& value_cache,
                                const torch::Tensor& block_table_cpu,
                                const torch::Tensor& seq_lens_cpu) {
    auto block_table_device = block_table_cpu.to(torch::kPrivateUse1);
    auto seq_lens_device = seq_lens_cpu.to(torch::kPrivateUse1);
    {
      c10_npu::NPUStreamGuard stream_guard(capture_stream);
      graph_query.copy_(query);
      graph_key_cache.copy_(key_cache);
      graph_value_cache.copy_(value_cache);
      graph_block_table.copy_(block_table_device);
      graph_seq_lens.copy_(seq_lens_device);
      graph.replay();
    }
    capture_stream.synchronize();

    auto eager_output = torch::empty_like(graph_output);
    kernel::npu::batch_decode(graph_query,
                              graph_key_cache,
                              graph_value_cache,
                              scale,
                              graph_block_table,
                              seq_lens_cpu,
                              eager_output);
    c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
        ->synchronizeDevice(0);
    const auto error =
        (graph_output.to(torch::kFloat32) - eager_output.to(torch::kFloat32))
            .abs()
            .cpu();
    const int64_t mismatches = error.ne(0).sum().item<int64_t>();
    LOG(INFO) << "Paged attention graph replay case=" << name
              << ", max_abs=" << error.max().item<float>()
              << ", mean_abs=" << error.mean().item<float>()
              << ", mismatches=" << mismatches;
    EXPECT_EQ(mismatches, 0) << "graph replay froze or lost " << name;
  };

  replay_and_compare("unchanged",
                     query_base,
                     key_cache_base,
                     value_cache_base,
                     block_table_base_cpu,
                     seq_lens_base_cpu);
  replay_and_compare("query",
                     query_changed,
                     key_cache_base,
                     value_cache_base,
                     block_table_base_cpu,
                     seq_lens_base_cpu);
  replay_and_compare("seq_lens",
                     query_base,
                     key_cache_base,
                     value_cache_base,
                     block_table_base_cpu,
                     seq_lens_changed_cpu);
  replay_and_compare("block_table",
                     query_base,
                     key_cache_base,
                     value_cache_base,
                     block_table_changed_cpu,
                     seq_lens_base_cpu);
  replay_and_compare("kv_cache",
                     query_base,
                     key_cache_changed,
                     value_cache_changed,
                     block_table_base_cpu,
                     seq_lens_base_cpu);
}

TEST_F(NpuXllmOpsTest, Ascend950CacheWriteThenAttentionGraphReplay) {
  if (!is_npu_available() || !is_ascend950_device()) {
    GTEST_SKIP() << "requires an Ascend950 NPU";
  }

  constexpr int64_t kBatch = 2;
  constexpr int64_t kHeads = 6;
  constexpr int64_t kKvHeads = 2;
  constexpr int64_t kHeadDim = 256;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kNumBlocks = 10;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  auto npu_options = torch::TensorOptions()
                         .device(torch::kPrivateUse1)
                         .dtype(torch::kBFloat16);
  auto cpu_int_options = torch::TensorOptions().dtype(torch::kInt32);
  torch::manual_seed(20260818);

  const auto query_base = torch::randn({kBatch, kHeads, kHeadDim}, npu_options);
  const auto query_changed =
      torch::randn({kBatch, kHeads, kHeadDim}, npu_options);
  const auto new_key_base =
      torch::randn({kBatch, kKvHeads, kHeadDim}, npu_options);
  const auto new_value_base = torch::randn_like(new_key_base);
  const auto new_key_changed = torch::randn_like(new_key_base);
  const auto new_value_changed = torch::randn_like(new_value_base);
  const auto cache_base =
      torch::randn({kNumBlocks, kBlockSize, kKvHeads, kHeadDim}, npu_options);
  const auto value_cache_base = torch::randn_like(cache_base);
  const auto block_table_cpu =
      torch::tensor({{2, 5, 1}, {7, 0, 4}}, cpu_int_options);
  const auto block_table = block_table_cpu.to(torch::kPrivateUse1);
  const auto seq_lens_base_cpu = torch::tensor({133, 259}, cpu_int_options);
  const auto seq_lens_changed_cpu = torch::tensor({257, 129}, cpu_int_options);
  // Each slot is the final token selected by its corresponding sequence.
  const auto slots_base_cpu =
      torch::tensor({5 * kBlockSize + 4, 4 * kBlockSize + 2}, cpu_int_options);
  const auto slots_changed_cpu =
      torch::tensor({1 * kBlockSize + 0, 0 * kBlockSize + 0}, cpu_int_options);

  auto graph_query = query_base.clone();
  auto graph_new_key = new_key_base.clone();
  auto graph_new_value = new_value_base.clone();
  auto graph_key_cache = cache_base.clone();
  auto graph_value_cache = value_cache_base.clone();
  auto graph_block_table = block_table.clone();
  auto graph_seq_lens = seq_lens_base_cpu.to(torch::kPrivateUse1);
  auto graph_slots = slots_base_cpu.to(torch::kPrivateUse1);
  auto graph_output = torch::empty_like(graph_query);

  auto run_graph_body = [&]() {
    std::optional<torch::Tensor> value = graph_new_value;
    std::optional<torch::Tensor> value_cache = graph_value_cache;
    kernel::npu::reshape_paged_cache(
        graph_new_key, value, graph_key_cache, value_cache, graph_slots);
    kernel::npu::batch_decode(graph_query,
                              graph_key_cache,
                              graph_value_cache,
                              scale,
                              graph_block_table,
                              graph_seq_lens,
                              graph_output);
  };

  c10_npu::NPUStream capture_stream = c10_npu::getStreamFromPool(true, 0);
  c10_npu::NPUGraph graph;
  {
    c10_npu::NPUStreamGuard stream_guard(capture_stream);
    run_graph_body();
  }
  capture_stream.synchronize();
  graph_key_cache.copy_(cache_base);
  graph_value_cache.copy_(value_cache_base);
  c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
      ->synchronizeDevice(0);
  {
    c10_npu::NPUStreamGuard stream_guard(capture_stream);
    graph.capture_begin(
        {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    run_graph_body();
    graph.capture_end();
  }
  capture_stream.synchronize();

  auto replay_and_compare = [&](const char* name,
                                const torch::Tensor& query,
                                const torch::Tensor& new_key,
                                const torch::Tensor& new_value,
                                const torch::Tensor& seq_lens_cpu,
                                const torch::Tensor& slots_cpu) {
    const auto seq_lens_device = seq_lens_cpu.to(torch::kPrivateUse1);
    const auto slots_device = slots_cpu.to(torch::kPrivateUse1);
    {
      c10_npu::NPUStreamGuard stream_guard(capture_stream);
      graph_query.copy_(query);
      graph_new_key.copy_(new_key);
      graph_new_value.copy_(new_value);
      graph_key_cache.copy_(cache_base);
      graph_value_cache.copy_(value_cache_base);
      graph_seq_lens.copy_(seq_lens_device);
      graph_slots.copy_(slots_device);
      graph.replay();
    }
    capture_stream.synchronize();

    auto eager_key_cache = cache_base.clone();
    auto eager_value_cache_tensor = value_cache_base.clone();
    auto eager_key = new_key;
    std::optional<torch::Tensor> eager_value = new_value;
    std::optional<torch::Tensor> eager_value_cache = eager_value_cache_tensor;
    kernel::npu::reshape_paged_cache(eager_key,
                                     eager_value,
                                     eager_key_cache,
                                     eager_value_cache,
                                     slots_device);
    auto eager_output = torch::empty_like(graph_output);
    kernel::npu::batch_decode(query,
                              eager_key_cache,
                              eager_value_cache_tensor,
                              scale,
                              block_table,
                              seq_lens_cpu,
                              eager_output);
    c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
        ->synchronizeDevice(0);

    const auto output_error =
        (graph_output.to(torch::kFloat32) - eager_output.to(torch::kFloat32))
            .abs()
            .cpu();
    const int64_t output_mismatches = output_error.ne(0).sum().item<int64_t>();
    const int64_t key_cache_mismatches =
        graph_key_cache.ne(eager_key_cache).sum().cpu().item<int64_t>();
    const int64_t value_cache_mismatches =
        graph_value_cache.ne(eager_value_cache_tensor)
            .sum()
            .cpu()
            .item<int64_t>();
    LOG(INFO) << "Cache-write graph replay case=" << name
              << ", output_max_abs=" << output_error.max().item<float>()
              << ", output_mismatches=" << output_mismatches
              << ", key_cache_mismatches=" << key_cache_mismatches
              << ", value_cache_mismatches=" << value_cache_mismatches;
    EXPECT_EQ(key_cache_mismatches, 0) << name;
    EXPECT_EQ(value_cache_mismatches, 0) << name;
    EXPECT_EQ(output_mismatches, 0) << name;
  };

  replay_and_compare("unchanged",
                     query_base,
                     new_key_base,
                     new_value_base,
                     seq_lens_base_cpu,
                     slots_base_cpu);
  replay_and_compare("new_kv",
                     query_base,
                     new_key_changed,
                     new_value_changed,
                     seq_lens_base_cpu,
                     slots_base_cpu);
  replay_and_compare("query_seq_and_slot",
                     query_changed,
                     new_key_changed,
                     new_value_changed,
                     seq_lens_changed_cpu,
                     slots_changed_cpu);
}

TEST_F(NpuXllmOpsTest, Ascend950RecurrentStateGraphReplayTracksInputs) {
  if (!is_npu_available() || !is_ascend950_device()) {
    GTEST_SKIP() << "requires an Ascend950 NPU";
  }

  constexpr int64_t kBatch = 2;
  constexpr int64_t kKeyHeads = 2;
  constexpr int64_t kValueHeads = 4;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kStateSlots = 6;
  auto bf16_options = torch::TensorOptions()
                          .device(torch::kPrivateUse1)
                          .dtype(torch::kBFloat16);
  auto fp32_options = bf16_options.dtype(torch::kFloat32);
  auto int_options = bf16_options.dtype(torch::kInt32);
  torch::manual_seed(20260817);

  const auto q_base =
      torch::randn({kBatch, 1, kKeyHeads, kHeadDim}, bf16_options);
  const auto k_base = torch::randn_like(q_base);
  const auto v_base =
      torch::randn({kBatch, 1, kValueHeads, kHeadDim}, bf16_options);
  const auto g_base =
      torch::randn({kBatch, 1, kValueHeads}, fp32_options) * 0.01;
  const auto beta_base =
      torch::sigmoid(torch::randn({kBatch, 1, kValueHeads}, bf16_options));
  const auto cache_base = torch::randn(
      {kStateSlots, kValueHeads, kHeadDim, kHeadDim}, fp32_options);
  const auto indices_base = torch::tensor({0, 3}, int_options);

  const auto q_changed = torch::randn_like(q_base);
  const auto k_changed = torch::randn_like(k_base);
  const auto v_changed = torch::randn_like(v_base);
  const auto g_changed =
      torch::randn({kBatch, 1, kValueHeads}, fp32_options) * 0.01;
  const auto beta_changed =
      torch::sigmoid(torch::randn({kBatch, 1, kValueHeads}, bf16_options));
  const auto indices_changed = torch::tensor({2, 5}, int_options);

  auto graph_q = q_base.clone();
  auto graph_k = k_base.clone();
  auto graph_v = v_base.clone();
  auto graph_g = g_base.clone();
  auto graph_beta = beta_base.clone();
  auto graph_cache = cache_base.clone();
  auto graph_indices = indices_base.clone();
  auto graph_output = torch::empty_like(v_base);

  auto recurrent_step = [](const torch::Tensor& query,
                           const torch::Tensor& key,
                           const torch::Tensor& value,
                           const torch::Tensor& g,
                           const torch::Tensor& beta,
                           torch::Tensor& state_cache,
                           const torch::Tensor& state_indices,
                           torch::Tensor& output) {
    auto q_norm =
        query / torch::sqrt(torch::sum(torch::square(query), -1, true) + 1e-6);
    auto k_norm =
        key / torch::sqrt(torch::sum(torch::square(key), -1, true) + 1e-6);
    q_norm = q_norm.transpose(1, 2).contiguous().to(torch::kFloat32);
    k_norm = k_norm.transpose(1, 2).contiguous().to(torch::kFloat32);
    auto value_fp32 = value.transpose(1, 2).contiguous().to(torch::kFloat32);
    auto beta_fp32 = beta.transpose(1, 2).contiguous().to(torch::kFloat32);
    auto g_fp32 = g.transpose(1, 2).contiguous().to(torch::kFloat32);
    q_norm =
        q_norm.unsqueeze(2)
            .expand({kBatch, kKeyHeads, kValueHeads / kKeyHeads, 1, kHeadDim})
            .reshape({kBatch, kValueHeads, 1, kHeadDim})
            .contiguous();
    k_norm =
        k_norm.unsqueeze(2)
            .expand({kBatch, kKeyHeads, kValueHeads / kKeyHeads, 1, kHeadDim})
            .reshape({kBatch, kValueHeads, 1, kHeadDim})
            .contiguous();
    q_norm = q_norm * (1.0 / std::sqrt(static_cast<double>(kHeadDim)));

    auto state = torch::index_select(state_cache, 0, state_indices);
    auto q_t = q_norm.select(2, 0);
    auto k_t = k_norm.select(2, 0);
    auto v_t = value_fp32.select(2, 0);
    auto g_t = g_fp32.select(2, 0).exp().unsqueeze(-1).unsqueeze(-1);
    auto beta_t = beta_fp32.select(2, 0).unsqueeze(-1);
    state = state * g_t;
    auto kv_mem = torch::sum(state * k_t.unsqueeze(-1), -2);
    auto delta = (v_t - kv_mem) * beta_t;
    state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
    auto step_output = torch::sum(state * q_t.unsqueeze(-1), -2)
                           .unsqueeze(2)
                           .transpose(1, 2)
                           .contiguous()
                           .to(value.scalar_type());
    output.copy_(step_output);
    state_cache.index_put_({state_indices}, state);
  };

  auto run_graph_body = [&]() {
    recurrent_step(graph_q,
                   graph_k,
                   graph_v,
                   graph_g,
                   graph_beta,
                   graph_cache,
                   graph_indices,
                   graph_output);
  };

  c10_npu::NPUStream capture_stream = c10_npu::getStreamFromPool(true, 0);
  c10_npu::NPUGraph graph;
  {
    c10_npu::NPUStreamGuard stream_guard(capture_stream);
    run_graph_body();
  }
  capture_stream.synchronize();
  graph_cache.copy_(cache_base);
  c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
      ->synchronizeDevice(0);
  {
    c10_npu::NPUStreamGuard stream_guard(capture_stream);
    graph.capture_begin(
        {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    run_graph_body();
    graph.capture_end();
  }
  capture_stream.synchronize();

  auto replay_and_compare = [&](const char* name,
                                const torch::Tensor& query,
                                const torch::Tensor& key,
                                const torch::Tensor& value,
                                const torch::Tensor& g,
                                const torch::Tensor& beta,
                                const torch::Tensor& indices) {
    {
      c10_npu::NPUStreamGuard stream_guard(capture_stream);
      graph_q.copy_(query);
      graph_k.copy_(key);
      graph_v.copy_(value);
      graph_g.copy_(g);
      graph_beta.copy_(beta);
      graph_cache.copy_(cache_base);
      graph_indices.copy_(indices);
      graph.replay();
    }
    capture_stream.synchronize();

    auto eager_cache = cache_base.clone();
    auto eager_output = torch::empty_like(graph_output);
    recurrent_step(
        query, key, value, g, beta, eager_cache, indices, eager_output);
    c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
        ->synchronizeDevice(0);

    const auto output_error =
        (graph_output.to(torch::kFloat32) - eager_output.to(torch::kFloat32))
            .abs()
            .cpu();
    const auto cache_error = (graph_cache - eager_cache).abs().cpu();
    LOG(INFO) << "Recurrent graph replay case=" << name
              << ", output_max_abs=" << output_error.max().item<float>()
              << ", output_mismatches="
              << output_error.ne(0).sum().item<int64_t>()
              << ", cache_max_abs=" << cache_error.max().item<float>()
              << ", cache_mismatches="
              << cache_error.ne(0).sum().item<int64_t>();
    EXPECT_EQ(output_error.ne(0).sum().item<int64_t>(), 0) << name;
    EXPECT_EQ(cache_error.ne(0).sum().item<int64_t>(), 0) << name;
  };

  replay_and_compare(
      "unchanged", q_base, k_base, v_base, g_base, beta_base, indices_base);
  replay_and_compare("changed_inputs_and_slots",
                     q_changed,
                     k_changed,
                     v_changed,
                     g_changed,
                     beta_changed,
                     indices_changed);
}

TEST_F(NpuXllmOpsTest, Ascend950GraphMropeMatchesPartialRotaryEmbedding) {
  if (!is_npu_available() || !is_ascend950_device()) {
    GTEST_SKIP() << "requires an Ascend950 NPU";
  }

  constexpr int64_t kTokens = 8;
  constexpr int64_t kQueryHeads = 6;
  constexpr int64_t kKeyHeads = 1;
  constexpr int64_t kHeadDim = 256;
  constexpr int64_t kRotaryDim = 64;
  constexpr int64_t kMaxPosition = 512;
  const std::vector<int64_t> mrope_section = {11, 11, 10};
  auto bf16_options = torch::TensorOptions()
                          .device(torch::kPrivateUse1)
                          .dtype(torch::kBFloat16);
  auto int_options = bf16_options.dtype(torch::kInt32);
  torch::manual_seed(20260817);

  auto inv_freq =
      1.0 /
      torch::pow(
          10000000.0,
          torch::arange(0, kRotaryDim, 2, bf16_options.dtype(torch::kFloat32)) /
              static_cast<double>(kRotaryDim));
  auto frequencies = torch::einsum(
      "i,j->ij",
      {torch::arange(0, kMaxPosition, 1, bf16_options.dtype(torch::kFloat32)),
       inv_freq});
  auto cos_sin_cache = torch::cat({frequencies.cos(), frequencies.sin()}, -1)
                           .to(torch::kBFloat16)
                           .contiguous();
  auto temporal_positions =
      torch::tensor({17, 18, 19, 20, 21, 22, 23, 24}, int_options);
  auto positions = temporal_positions.unsqueeze(0).expand({3, kTokens}).clone();
  auto query = torch::randn({kTokens, kQueryHeads * kHeadDim}, bf16_options);
  auto key = torch::randn({kTokens, kKeyHeads * kHeadDim}, bf16_options);

  auto eager_query_input = query;
  auto eager_key_input = key;
  auto [eager_query, eager_key] =
      kernel::npu::apply_npu_partial_rotary_embedding(temporal_positions,
                                                      eager_query_input,
                                                      eager_key_input,
                                                      kHeadDim,
                                                      kRotaryDim,
                                                      cos_sin_cache,
                                                      /*is_neox_style=*/true);

  auto selected =
      cos_sin_cache.index_select(0, positions.permute({1, 0}).reshape({-1}))
          .view({kTokens, 3 * kRotaryDim});
  std::vector<int64_t> gather_indices(kRotaryDim);
  const int64_t half = kRotaryDim / 2;
  for (int64_t i = 0; i < half; ++i) {
    int64_t axis = 0;
    if ((i % 3) == 1 && i <= 3 * mrope_section[1]) {
      axis = 1;
    } else if ((i % 3) == 2 && i <= 3 * mrope_section[2]) {
      axis = 2;
    }
    gather_indices[i] = axis * kRotaryDim + i;
    gather_indices[half + i] = axis * kRotaryDim + half + i;
  }
  auto gather = torch::tensor(
      gather_indices,
      torch::TensorOptions().device(torch::kPrivateUse1).dtype(torch::kLong));
  auto merged = selected.index_select(-1, gather);
  auto cos_half = merged.slice(-1, 0, half);
  auto sin_half = merged.slice(-1, half, kRotaryDim);
  auto cos = torch::cat({cos_half, cos_half}, -1).unsqueeze(1);
  auto sin = torch::cat({sin_half, sin_half}, -1).unsqueeze(1);
  auto rotate = [&](const torch::Tensor& flat, int64_t heads) {
    auto input = flat.view({kTokens, heads, kHeadDim});
    auto input_rot = input.slice(-1, 0, kRotaryDim).contiguous();
    auto unused_key = input_rot.clone();
    kernel::npu::apply_rotary(
        input_rot, unused_key, cos, sin, /*input_layout=*/"BSND");
    return torch::cat({input_rot, input.slice(-1, kRotaryDim, kHeadDim)}, -1)
        .view_as(flat);
  };
  auto graph_query = rotate(query, kQueryHeads);
  auto graph_key = rotate(key, kKeyHeads);
  c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
      ->synchronizeDevice(0);

  auto query_error =
      (graph_query.to(torch::kFloat32) - eager_query.to(torch::kFloat32))
          .abs()
          .cpu();
  auto key_error =
      (graph_key.to(torch::kFloat32) - eager_key.to(torch::kFloat32))
          .abs()
          .cpu();
  LOG(INFO) << "Graph mRoPE vs eager partial rotary: q_max_abs="
            << query_error.max().item<float>()
            << ", q_mismatches=" << query_error.ne(0).sum().item<int64_t>()
            << ", k_max_abs=" << key_error.max().item<float>()
            << ", k_mismatches=" << key_error.ne(0).sum().item<int64_t>();
  EXPECT_TRUE(torch::allclose(graph_query.to(torch::kFloat32),
                              eager_query.to(torch::kFloat32),
                              /*rtol=*/1e-3,
                              /*atol=*/1e-3));
  EXPECT_TRUE(torch::allclose(graph_key.to(torch::kFloat32),
                              eager_key.to(torch::kFloat32),
                              /*rtol=*/1e-3,
                              /*atol=*/1e-3));
}

TEST_F(NpuXllmOpsTest, Ascend950GraphMropeRotaryCaptureReplayTracksInputs) {
  if (!is_npu_available() || !is_ascend950_device()) {
    GTEST_SKIP() << "requires an Ascend950 NPU";
  }

  constexpr int64_t kTokens = 8;
  constexpr int64_t kQueryHeads = 6;
  constexpr int64_t kKeyHeads = 1;
  constexpr int64_t kRotaryDim = 64;
  auto options = torch::TensorOptions()
                     .device(torch::kPrivateUse1)
                     .dtype(torch::kBFloat16);
  torch::manual_seed(20260818);

  auto query_base = torch::randn({kTokens, kQueryHeads, kRotaryDim}, options);
  auto key_base = torch::randn({kTokens, kKeyHeads, kRotaryDim}, options);
  auto query_changed = torch::randn_like(query_base);
  auto key_changed = torch::randn_like(key_base);
  auto cos_base = torch::randn({kTokens, 1, kRotaryDim}, options);
  auto sin_base = torch::randn_like(cos_base);
  auto cos_changed = torch::randn_like(cos_base);
  auto sin_changed = torch::randn_like(sin_base);

  auto graph_query = query_base.clone();
  auto graph_key = key_base.clone();
  auto graph_cos = cos_base.clone();
  auto graph_sin = sin_base.clone();
  auto graph_query_output = torch::empty_like(graph_query);
  auto graph_key_output = torch::empty_like(graph_key);
  auto run_graph_body = [&]() {
    auto query = graph_query.contiguous();
    auto key = graph_key.contiguous();
    kernel::npu::apply_rotary(
        query, key, graph_cos, graph_sin, /*input_layout=*/"BSND");
    graph_query_output.copy_(query);
    graph_key_output.copy_(key);
  };

  c10_npu::NPUStream capture_stream = c10_npu::getStreamFromPool(true, 0);
  c10_npu::NPUGraph graph;
  {
    c10_npu::NPUStreamGuard stream_guard(capture_stream);
    run_graph_body();
  }
  capture_stream.synchronize();
  {
    c10_npu::NPUStreamGuard stream_guard(capture_stream);
    graph.capture_begin(
        {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    run_graph_body();
    graph.capture_end();
  }
  capture_stream.synchronize();

  auto replay_and_compare = [&](const char* name,
                                const torch::Tensor& query,
                                const torch::Tensor& key,
                                const torch::Tensor& cos,
                                const torch::Tensor& sin) {
    {
      c10_npu::NPUStreamGuard stream_guard(capture_stream);
      graph_query.copy_(query);
      graph_key.copy_(key);
      graph_cos.copy_(cos);
      graph_sin.copy_(sin);
      graph.replay();
    }
    capture_stream.synchronize();

    auto eager_query = query.clone();
    auto eager_key = key.clone();
    kernel::npu::apply_rotary(
        eager_query, eager_key, cos, sin, /*input_layout=*/"BSND");
    c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
        ->synchronizeDevice(0);
    auto query_error = (graph_query_output.to(torch::kFloat32) -
                        eager_query.to(torch::kFloat32))
                           .abs()
                           .cpu();
    auto key_error =
        (graph_key_output.to(torch::kFloat32) - eager_key.to(torch::kFloat32))
            .abs()
            .cpu();
    LOG(INFO) << "Graph rotary replay case=" << name
              << ", q_max_abs=" << query_error.max().item<float>()
              << ", q_mismatches=" << query_error.ne(0).sum().item<int64_t>()
              << ", k_max_abs=" << key_error.max().item<float>()
              << ", k_mismatches=" << key_error.ne(0).sum().item<int64_t>();
    EXPECT_EQ(query_error.ne(0).sum().item<int64_t>(), 0) << name;
    EXPECT_EQ(key_error.ne(0).sum().item<int64_t>(), 0) << name;
  };

  replay_and_compare("unchanged", query_base, key_base, cos_base, sin_base);
  replay_and_compare("changed_inputs_and_positions",
                     query_changed,
                     key_changed,
                     cos_changed,
                     sin_changed);
}

}  // namespace
}  // namespace xllm
