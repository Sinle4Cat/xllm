# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from pathlib import Path
import re
import sys

import pytest


pytest.importorskip("tilelang")

REPO_ROOT = Path(__file__).resolve().parents[4]
XLLM_PYTHON_ROOT = REPO_ROOT / "xllm"
sys.path.insert(0, str(XLLM_PYTHON_ROOT))

from compiler.tilelang.targets.ascend.kernels.mega_gdn_mtp_decode import (  # noqa: E402
    A3_UB_CAPACITY_BYTES,
    DISPATCH_DTYPE,
    OUTPUT_INDICES,
    SUPPORTED_SPECULATIVE_TOKENS,
    MegaGdnMtpDecodeKernel,
    _generated_pto_ub_high_water_bytes,
    _validate_generated_pto_source,
    build_mega_gdn_mtp_decode_kernel,
)
from compiler.tilelang.targets.ascend.abi_entry import (  # noqa: E402
    parse_kernel_abi,
    rename_variant_internal_symbols,
)


EXPECTED_CALL_PARAMETERS = (
    "qkv_handle",
    "z_handle",
    "b_handle",
    "a_handle",
    "conv_weight_handle",
    "conv_state_handle",
    "a_log_handle",
    "dt_bias_handle",
    "ssm_state_handle",
    "read_state_indices_handle",
    "write_state_indices_handle",
    "num_accepted_tokens_handle",
    "norm_weight_handle",
    "conv_out_handle",
    "conv_state_out_handle",
    "ssm_state_out_handle",
    "out_handle",
    "batch_size",
    "stream",
)
EXPECTED_UB_HIGH_WATER_BYTES = {
    1: 156_256,
    2: 158_048,
    3: 159_840,
    4: 161_632,
    5: 163_424,
    8: 168_288,
}
PRODUCTION_PTO_KERNEL = (
    REPO_ROOT
    / "third_party/xllm_ops/xllm_ops/mega_gdn_mtp_decode/op_kernel"
    / "mega_gdn_mtp_decode_pto_kernel.h"
)


@pytest.fixture(scope="module")
def generated_k8_b4_source() -> str:
    return MegaGdnMtpDecodeKernel.generate_source(
        speculative_tokens=8,
        max_batch_size=4,
        num_state_slots=256,
        num_k_heads=8,
        num_v_heads=24,
        dtype=DISPATCH_DTYPE,
    )


def _readout_token_variables(source: str) -> tuple[str, str]:
    producer_match = re.search(
        r"TASSIGN\(readout_cache_temp_0,.*?"
        r"\(\(([A-Za-z_]\w*) \* 128\)",
        source,
    )
    consumer_match = re.search(
        r"TASSIGN\(readout_cache_temp_1,.*?"
        r"\(([A-Za-z_]\w*) \* 128\)",
        source,
    )
    assert producer_match is not None
    assert consumer_match is not None
    return producer_match.group(1), consumer_match.group(1)


def _braced_block_end(source: str, block_start: int) -> int:
    open_brace = source.index("{", block_start)
    depth = 0
    for position in range(open_brace, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return position + 1
    raise AssertionError("unterminated generated C++ block")


def test_registered_pto_specializations_cover_required_k() -> None:
    specs = MegaGdnMtpDecodeKernel.specs()

    assert {spec.target for spec in specs} == {"pto"}
    assert {
        int(spec.specialization["speculative_tokens"]) for spec in specs
    } == set(SUPPORTED_SPECULATIVE_TOKENS)
    assert len({spec.variant_key for spec in specs}) == len(
        SUPPORTED_SPECULATIVE_TOKENS
    )


@pytest.mark.parametrize(
    "speculative_tokens", SUPPORTED_SPECULATIVE_TOKENS
)
def test_builds_tilelang_primfunc_for_each_k(
    speculative_tokens: int,
) -> None:
    kernel = build_mega_gdn_mtp_decode_kernel(
        speculative_tokens=speculative_tokens,
        max_batch_size=1,
        num_state_slots=2,
    )

    assert type(kernel).__name__ == "PrimFunc"


def test_rejects_unvalidated_k() -> None:
    with pytest.raises(ValueError, match="unsupported K"):
        build_mega_gdn_mtp_decode_kernel(
            speculative_tokens=6,
            max_batch_size=1,
            num_state_slots=2,
        )


@pytest.mark.parametrize("aicore_macro", ("__aicore__", "AICORE"))
def test_renames_generated_device_entry_per_variant(
    aicore_macro: str,
) -> None:
    source = (
        f'extern "C" __global__ {aicore_macro} void launch_kernel() {{}}\n'
        "void call(void* stream) {\n"
        "  launch_kernel<<<1, nullptr, stream>>>();\n"
        "}\n"
    )

    renamed = rename_variant_internal_symbols(source, "k1")

    assert renamed.count("launch_kernel__k1") == 2
    assert "void launch_kernel()" not in renamed


@pytest.mark.parametrize(
    "speculative_tokens", SUPPORTED_SPECULATIVE_TOKENS
)
def test_generates_auditable_pto_source_for_each_k(
    speculative_tokens: int,
) -> None:
    source = MegaGdnMtpDecodeKernel.generate_source(
        speculative_tokens=speculative_tokens,
        max_batch_size=1,
        num_state_slots=2,
        num_k_heads=8,
        num_v_heads=24,
        dtype=DISPATCH_DTYPE,
    )

    assert '#include "tl_templates/pto/common.h"' in source
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);" in source
    assert "#if defined(__DAV_VEC__) || defined(__DAV_C220_VEC__)" in source
    assert "TROWEXPAND" in source
    assert "TCOLSUM" in source
    assert "ssm_state_out_handle" in source
    assert 'extern "C" void call(' in source
    assert "token_idx" in source
    assert source.count("TROWEXPAND(") == 2
    assert source.count("TCOLSUM(") == 2
    assert source.count("TCOLEXPAND(") == 1
    assert source.count("compare_scalar(") == 1
    assert source.count("TSEL(") == 1
    assert source.count("TSIGMOID<") == 1
    assert source.count("TMAXS(") == 1
    assert "TADD(gate_softplus, gate_x, gate_abs)" not in source
    assert "TMULS(gate_softplus, gate_softplus" not in source
    assert "condval" not in source
    assert "owner_tile_idx" not in source
    assert "launch_kernel<<<4, nullptr, stream>>>" in source
    assert (
        "for (int32_t value_head_offset = 0; "
        "value_head_offset < 3;"
    ) in source
    conv_out_load_lines = [
        line
        for line in source.splitlines()
        if "copy_gm_to_ub" in line and ">(conv_out_handle" in line
    ]
    conv_out_store_lines = [
        line
        for line in source.splitlines()
        if "copy_ub_to_gm" in line and ">(conv_out_handle" in line
    ]
    assert len(conv_out_load_lines) == 3
    assert len(conv_out_store_lines) == 5
    value_head_loop_position = source.index(
        "for (int32_t value_head_offset = 0; "
        "value_head_offset < 3;"
    )
    norm_weight_load_lines = [
        line
        for line in source.splitlines()
        if "copy_gm_to_ub" in line and ">(norm_weight_handle" in line
    ]
    assert len(norm_weight_load_lines) == 1
    norm_weight_load_position = source.index(norm_weight_load_lines[0])
    if speculative_tokens == 8:
        assert norm_weight_load_position < value_head_loop_position
        assert (
            source.count(
                "TCVT(norm_weight_fp32, norm_weight_half, "
                "RoundMode::CAST_NONE);"
            )
            == 1
        )
        assert (
            "TCVT(weight_fp32, weight_half, RoundMode::CAST_NONE);"
            not in source
        )
    else:
        assert norm_weight_load_position > value_head_loop_position
        assert (
            source.count(
                "TCVT(weight_fp32, weight_half, RoundMode::CAST_NONE);"
            )
            == 1
        )
        assert (
            "TCVT(norm_weight_fp32, norm_weight_half, "
            "RoundMode::CAST_NONE);"
            not in source
        )
    for marker in (
        "TMOV(q_fp32_cache_temp_0",
        "TMOV(k_fp32_cache_temp_0",
    ):
        assert source.count(marker) == 1
        assert source.index(marker) < value_head_loop_position
    if speculative_tokens == 8:
        assert "k_column;" not in source
        assert "TMOV(k_column, " not in source
        assert (
            source.count(
                "TROWEXPAND(broadcast_buf, k_fp32_cache_temp_1);"
            )
            == 1
        )
        assert (
            source.count(
                "TROWEXPAND(broadcast_buf, q_fp32_cache_temp_1);"
            )
            == 1
        )
    else:
        assert "k_column;" in source
        assert source.count("TMOV(k_column, ") == 2
        assert (
            source.count(
                "TROWEXPAND(broadcast_buf, k_column_temp_"
            )
            == 2
        )
    out_load_lines = [
        line
        for line in source.splitlines()
        if "copy_gm_to_ub" in line and ">(out_handle" in line
    ]
    out_store_lines = [
        line
        for line in source.splitlines()
        if "copy_ub_to_gm" in line and ">(out_handle" in line
    ]
    assert out_load_lines == []
    assert len(out_store_lines) == 1
    for marker in (
        "TCVT(readout_half, pred, RoundMode::CAST_RINT);",
        "TMOV(readout_cache_temp_0, readout_half);",
        "TCVT(norm_fp32, readout_cache_temp_1, RoundMode::CAST_NONE);",
    ):
        assert source.count(marker) == 1
    producer_token, consumer_token = _readout_token_variables(source)
    assert (
        source.count(
            f"for (int32_t {producer_token} = 0; "
            f"{producer_token} < {speculative_tokens + 1};"
        )
        == 1
    )
    assert (
        source.count(
            f"for (int32_t {consumer_token} = 0; "
            f"{consumer_token} < {speculative_tokens + 1};"
        )
        == 1
    )
    qk_reduce_positions = [
        position
        for position in range(len(source))
        if source.startswith("TROWSUM(norm_value", position)
    ]
    assert len(qk_reduce_positions) == 2
    assert all(
        position < value_head_loop_position
        for position in qk_reduce_positions
    )
    assert (
        source.count(
            "set_flag_pipeline<PIPE_MTE3, PIPE_MTE2> (7);"
        )
        == 1
    )
    assert (
        source.count(
            "wait_flag_pipeline<PIPE_MTE3, PIPE_MTE2> (7);"
        )
        == 1
    )
    assert _generated_pto_ub_high_water_bytes(source) == (
        EXPECTED_UB_HIGH_WATER_BYTES[speculative_tokens]
    )
    assert (
        _generated_pto_ub_high_water_bytes(source)
        < A3_UB_CAPACITY_BYTES
    )

    abi = parse_kernel_abi(source, "call")
    assert tuple(parameter.name for parameter in abi.parameters) == (
        EXPECTED_CALL_PARAMETERS
    )
    assert OUTPUT_INDICES == (13, 14, 15, 16)


def test_generates_default_b4_key_head_task_map(
    generated_k8_b4_source: str,
) -> None:
    assert "launch_kernel<<<16, nullptr, stream>>>" in generated_k8_b4_source
    assert "condval" not in generated_k8_b4_source
    assert "owner_tile_idx" not in generated_k8_b4_source
    assert (
        _generated_pto_ub_high_water_bytes(generated_k8_b4_source)
        == EXPECTED_UB_HIGH_WATER_BYTES[8]
    )


def test_generated_source_rejects_shifted_final_out_address(
    generated_k8_b4_source: str,
) -> None:
    store_line = next(
        line
        for line in generated_k8_b4_source.splitlines()
        if "copy_ub_to_gm" in line and ">(out_handle" in line
    )
    address_start = store_line.index("out_handle + ") + len(
        "out_handle + "
    )
    address_end = store_line.index(",", address_start)
    shifted_store_line = (
        store_line[:address_start]
        + f"({store_line[address_start:address_end]} + 128)"
        + store_line[address_end:]
    )
    shifted_source = generated_k8_b4_source.replace(
        store_line,
        shifted_store_line,
        1,
    )

    with pytest.raises(RuntimeError, match="incorrect final out address"):
        _validate_generated_pto_source(
            source=shifted_source,
            speculative_tokens=8,
            max_batch_size=4,
            num_k_heads=8,
            num_v_heads=24,
        )


@pytest.mark.parametrize(
    ("producer_name", "consumer_name"),
    (
        ("readout_token", "norm_token"),
        ("shared_token", "shared_token"),
    ),
)
def test_generated_source_accepts_readout_token_renames(
    generated_k8_b4_source: str,
    producer_name: str,
    consumer_name: str,
) -> None:
    producer_token, consumer_token = _readout_token_variables(
        generated_k8_b4_source
    )
    renamed_source = re.sub(
        rf"\b{re.escape(producer_token)}\b",
        producer_name,
        generated_k8_b4_source,
    )
    renamed_source = re.sub(
        rf"\b{re.escape(consumer_token)}\b",
        consumer_name,
        renamed_source,
    )

    _validate_generated_pto_source(
        source=renamed_source,
        speculative_tokens=8,
        max_batch_size=4,
        num_k_heads=8,
        num_v_heads=24,
    )


def test_generated_source_rejects_shifted_readout_cache_base(
    generated_k8_b4_source: str,
) -> None:
    producer_match = re.search(
        r"TASSIGN\(readout_cache_temp_0, (\d+) \+",
        generated_k8_b4_source,
    )
    consumer_match = re.search(
        r"TASSIGN\(readout_cache_temp_1, (\d+) \+",
        generated_k8_b4_source,
    )
    assert producer_match is not None
    assert consumer_match is not None
    assert producer_match.group(1) == consumer_match.group(1)
    shifted_base = str(int(producer_match.group(1)) + 128)
    shifted_source = generated_k8_b4_source.replace(
        producer_match.group(0),
        producer_match.group(0).replace(
            producer_match.group(1), shifted_base, 1
        ),
        1,
    )
    shifted_source = shifted_source.replace(
        consumer_match.group(0),
        consumer_match.group(0).replace(
            consumer_match.group(1), shifted_base, 1
        ),
        1,
    )

    with pytest.raises(RuntimeError, match="does not address readout_cache"):
        _validate_generated_pto_source(
            source=shifted_source,
            speculative_tokens=8,
            max_batch_size=4,
            num_k_heads=8,
            num_v_heads=24,
        )


def test_generated_source_rejects_missing_final_out_dependency(
    generated_k8_b4_source: str,
) -> None:
    final_cast_position = generated_k8_b4_source.index(
        "TCVT(final_half, norm_fp32, RoundMode::CAST_RINT);"
    )
    free_wait = re.search(
        r"wait_flag\(PIPE_V, PIPE_MTE3, EVENT_ID\d+\);",
        generated_k8_b4_source[final_cast_position:],
    )
    assert free_wait is not None
    wait_start = final_cast_position + free_wait.start()
    wait_end = final_cast_position + free_wait.end()
    missing_wait_source = (
        generated_k8_b4_source[:wait_start]
        + generated_k8_b4_source[wait_end:]
    )

    with pytest.raises(RuntimeError, match="missing its MTE3-to-Vector"):
        _validate_generated_pto_source(
            source=missing_wait_source,
            speculative_tokens=8,
            max_batch_size=4,
            num_k_heads=8,
            num_v_heads=24,
        )


def test_generated_source_rejects_whitespace_hidden_out_transfers(
    generated_k8_b4_source: str,
) -> None:
    store_line = next(
        line
        for line in generated_k8_b4_source.splitlines()
        if "copy_ub_to_gm" in line and ">(out_handle" in line
    )
    whitespace_store = store_line.replace(
        ">(out_handle",
        ">( out_handle",
        1,
    )
    whitespace_load = whitespace_store.replace(
        "copy_ub_to_gm",
        "copy_gm_to_ub",
        1,
    )
    for hidden_transfer in (whitespace_store, whitespace_load):
        mutated_source = generated_k8_b4_source.replace(
            store_line,
            f"{hidden_transfer}\n{store_line}",
            1,
        )
        with pytest.raises(RuntimeError, match="readouts through GM"):
            _validate_generated_pto_source(
                source=mutated_source,
                speculative_tokens=8,
                max_batch_size=4,
                num_k_heads=8,
                num_v_heads=24,
            )


def test_generated_source_rejects_final_store_before_norm_and_gate(
    generated_k8_b4_source: str,
) -> None:
    final_cast = "TCVT(final_half, norm_fp32, RoundMode::CAST_RINT);"
    final_cast_position = generated_k8_b4_source.index(final_cast)
    block_start = generated_k8_b4_source.rfind(
        "set_flag(PIPE_MTE3, PIPE_V,",
        0,
        final_cast_position,
    )
    store_position = generated_k8_b4_source.index(
        "copy_ub_to_gm",
        final_cast_position,
    )
    block_end = generated_k8_b4_source.index("\n", store_position) + 1
    final_store_block = generated_k8_b4_source[block_start:block_end]
    without_final_store = (
        generated_k8_b4_source[:block_start]
        + generated_k8_b4_source[block_end:]
    )
    consumer_cast_position = without_final_store.index(
        "TCVT(norm_fp32, readout_cache_temp_1, RoundMode::CAST_NONE);"
    )
    early_store_source = (
        without_final_store[:consumer_cast_position]
        + final_store_block
        + without_final_store[consumer_cast_position:]
    )

    with pytest.raises(RuntimeError, match="before final output conversion"):
        _validate_generated_pto_source(
            source=early_store_source,
            speculative_tokens=8,
            max_batch_size=4,
            num_k_heads=8,
            num_v_heads=24,
        )


def test_production_pto_uses_the_reviewed_state_formulas() -> None:
    source = PRODUCTION_PTO_KERNEL.read_text(encoding="utf-8")

    assert "#pragma unroll" not in source
    assert source.count("if constexpr (SpeculativeTokens == 8)") == 8
    assert source.count("if constexpr (SpeculativeTokens != 8)") == 1
    assert "kUbConvInputHalfPong = 7936" in source
    assert "kUbConvOutputHalfPong = 8192" in source
    assert "static_cast<event_t>(conv_pingpong_flag + 2)" in source
    assert "wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2)" in source
    assert "wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3)" in source
    assert "wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2)" in source
    assert "wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3)" in source
    assert "const int32_t next_token_idx = token_idx + 1" in source
    assert source.count("LoadQkBf16Rows(") == 3
    assert "using QkShape = pto::Shape<1, 1, 1, 2, kHeadDim>" in source
    assert (
        "using QkStride = "
        "pto::Stride<1, 1, 1, pto::DYNAMIC, 1>"
    ) in source
    assert "read_slot * conv_state_stride +" in source
    assert "(accepted - 1) * conv_dim + channel_offset" in source
    assert "read_slot * sequence_length + accepted - 1" in source
    assert "write_slot * sequence_length + token_idx" in source
    assert (
        "static_cast<int64_t>(read_checkpoint) * "
        "ssm_checkpoint_stride"
    ) in source
    assert (
        "static_cast<int64_t>(write_checkpoint) * "
        "ssm_checkpoint_stride"
    ) in source
    assert "ssm_state_handle + read_state_offset" in source
    assert "ssm_state_out_handle + write_state_offset" in source
