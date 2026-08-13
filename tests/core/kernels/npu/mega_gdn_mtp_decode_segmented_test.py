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
import sys

import pytest


pytest.importorskip("tilelang")

REPO_ROOT = Path(__file__).resolve().parents[4]
XLLM_PYTHON_ROOT = REPO_ROOT / "xllm"
sys.path.insert(0, str(XLLM_PYTHON_ROOT))

from compiler.tilelang.targets.ascend.abi_entry import (  # noqa: E402
    parse_kernel_abi,
)
from compiler.tilelang.targets.ascend.kernel_registry import (  # noqa: E402
    _load_registered_kernel_family,
)
from compiler.tilelang.targets.ascend.kernels.mega_gdn_mtp_decode import (  # noqa: E402
    A3_UB_CAPACITY_BYTES,
    SUPPORTED_SPECULATIVE_TOKENS,
    _generated_pto_ub_high_water_bytes,
)
from compiler.tilelang.targets.ascend.kernels.mega_gdn_mtp_decode_segmented import (  # noqa: E402
    CONV_OUTPUT_INDICES,
    NORM_OUTPUT_INDICES,
    RECURRENT_OUTPUT_INDICES,
    build_mega_gdn_mtp_conv_kernel,
    build_mega_gdn_mtp_norm_kernel,
    build_mega_gdn_mtp_recurrent_kernel,
    lower_mega_gdn_mtp_conv_pto,
    lower_mega_gdn_mtp_segmented_pto,
)


EXPECTED_CONV_PARAMETERS = (
    "qkv_handle",
    "b_handle",
    "a_handle",
    "conv_weight_handle",
    "conv_state_handle",
    "a_log_handle",
    "dt_bias_handle",
    "read_state_indices_handle",
    "write_state_indices_handle",
    "num_accepted_tokens_handle",
    "conv_out_handle",
    "conv_state_out_handle",
    "qk_prepared_handle",
    "gate_prepared_handle",
    "batch_size",
    "stream",
)
EXPECTED_RECURRENT_PARAMETERS = (
    "conv_out_handle",
    "qk_prepared_handle",
    "gate_prepared_handle",
    "ssm_state_handle",
    "read_state_indices_handle",
    "write_state_indices_handle",
    "num_accepted_tokens_handle",
    "ssm_state_out_handle",
    "readout_handle",
    "batch_size",
    "stream",
)
EXPECTED_NORM_PARAMETERS = (
    "readout_handle",
    "z_handle",
    "norm_weight_handle",
    "out_handle",
    "batch_size",
    "stream",
)


@pytest.mark.parametrize(
    "speculative_tokens", SUPPORTED_SPECULATIVE_TOKENS
)
def test_builds_all_segmented_primfuncs(
    speculative_tokens: int,
) -> None:
    conv = build_mega_gdn_mtp_conv_kernel(
        speculative_tokens=speculative_tokens,
        max_batch_size=1,
        num_state_slots=2,
    )
    recurrent = build_mega_gdn_mtp_recurrent_kernel(
        speculative_tokens=speculative_tokens,
        max_batch_size=1,
        num_state_slots=2,
    )
    norm = build_mega_gdn_mtp_norm_kernel(
        speculative_tokens=speculative_tokens,
        max_batch_size=1,
        num_state_slots=2,
    )

    assert type(conv).__name__ == "PrimFunc"
    assert type(recurrent).__name__ == "PrimFunc"
    assert type(norm).__name__ == "PrimFunc"


@pytest.mark.parametrize(
    "builder",
    (
        build_mega_gdn_mtp_conv_kernel,
        build_mega_gdn_mtp_recurrent_kernel,
        build_mega_gdn_mtp_norm_kernel,
    ),
)
def test_rejects_unvalidated_k(builder) -> None:
    with pytest.raises(ValueError, match="unsupported K"):
        builder(
            speculative_tokens=6,
            max_batch_size=1,
            num_state_slots=2,
        )


def test_registers_three_six_variant_pto_families() -> None:
    expected_modules = (
        "mega_gdn_mtp_decode_conv",
        "mega_gdn_mtp_decode_recurrent",
        "mega_gdn_mtp_decode_norm",
    )

    for module_name in expected_modules:
        family = _load_registered_kernel_family(module_name)
        assert family is not None
        assert family.kernel_name == module_name
        assert len(family.spec_pairs) == len(
            SUPPORTED_SPECULATIVE_TOKENS
        )
        assert {
            compile_spec.target
            for compile_spec, _ in family.spec_pairs
        } == {"pto"}
        assert {
            compile_spec.specialization["speculative_tokens"]
            for compile_spec, _ in family.spec_pairs
        } == set(SUPPORTED_SPECULATIVE_TOKENS)


@pytest.mark.parametrize(
    "speculative_tokens", SUPPORTED_SPECULATIVE_TOKENS
)
def test_lowers_three_stage_owner_maps_and_contiguous_state(
    speculative_tokens: int,
) -> None:
    (
        _,
        conv_source,
        _,
        recurrent_source,
        _,
        norm_source,
    ) = lower_mega_gdn_mtp_segmented_pto(
        speculative_tokens=speculative_tokens,
        max_batch_size=1,
        num_state_slots=2,
    )

    assert "launch_kernel<<<20, nullptr, stream>>>" in conv_source
    assert "launch_kernel<<<24, nullptr, stream>>>" in recurrent_source
    assert "launch_kernel<<<12, nullptr, stream>>>" in norm_source
    assert "KERNEL_TASK_TYPE_DEFAULT(" not in conv_source
    assert "KERNEL_TASK_TYPE_DEFAULT(" not in recurrent_source
    assert "KERNEL_TASK_TYPE_DEFAULT(" not in norm_source
    assert conv_source.count("set_mask_norm();") == 1
    assert recurrent_source.count("set_mask_norm();") == 1
    assert norm_source.count("set_mask_norm();") == 1
    assert conv_source.count("set_vector_mask(-1, -1);") == 1
    assert recurrent_source.count("set_vector_mask(-1, -1);") == 1
    assert norm_source.count("set_vector_mask(-1, -1);") == 1
    assert "ssm_state_handle" not in conv_source
    assert "qk_prepared_handle" in conv_source
    assert "gate_prepared_handle" in conv_source
    assert "conv_weight_handle" not in recurrent_source
    assert "norm_weight_handle" not in recurrent_source
    assert "state_row_idx" in recurrent_source
    assert "ssm_state_handle" not in norm_source
    assert _generated_pto_ub_high_water_bytes(
        conv_source
    ) < A3_UB_CAPACITY_BYTES
    assert _generated_pto_ub_high_water_bytes(
        recurrent_source
    ) < A3_UB_CAPACITY_BYTES
    assert _generated_pto_ub_high_water_bytes(
        norm_source
    ) < A3_UB_CAPACITY_BYTES

    conv_abi = parse_kernel_abi(conv_source, "call")
    recurrent_abi = parse_kernel_abi(recurrent_source, "call")
    norm_abi = parse_kernel_abi(norm_source, "call")
    assert tuple(
        parameter.name for parameter in conv_abi.parameters
    ) == EXPECTED_CONV_PARAMETERS
    assert tuple(
        parameter.name for parameter in recurrent_abi.parameters
    ) == EXPECTED_RECURRENT_PARAMETERS
    assert tuple(
        parameter.name for parameter in norm_abi.parameters
    ) == EXPECTED_NORM_PARAMETERS
    assert CONV_OUTPUT_INDICES == (10, 11, 12, 13)
    assert RECURRENT_OUTPUT_INDICES == (7, 8)
    assert NORM_OUTPUT_INDICES == (3,)


def test_conv_owner_reuses_weights_across_batch_rows() -> None:
    _, conv_source = lower_mega_gdn_mtp_conv_pto(
        speculative_tokens=1,
        max_batch_size=4,
        num_state_slots=256,
    )

    assert "launch_kernel<<<20, nullptr, stream>>>" in conv_source
    assert "((cid * 256) + (vid * 128))" in conv_source
    assert (
        "for (int32_t batch_idx = 0; batch_idx < 4; ++batch_idx)"
        in conv_source
    )


def test_recurrent_and_norm_use_static_batch_owners() -> None:
    (
        _,
        _,
        _,
        recurrent_source,
        _,
        norm_source,
    ) = lower_mega_gdn_mtp_segmented_pto(
        speculative_tokens=8,
        max_batch_size=4,
        num_state_slots=256,
    )

    assert "launch_kernel<<<96, nullptr, stream>>>" in recurrent_source
    assert "launch_kernel<<<48, nullptr, stream>>>" in norm_source


def test_conv_prepares_unique_qk_and_gate_values_across_batch() -> None:
    _, conv_source = lower_mega_gdn_mtp_conv_pto(
        speculative_tokens=1,
        max_batch_size=4,
        num_state_slots=256,
    )

    assert "launch_kernel<<<20, nullptr, stream>>>" in conv_source
    assert (
        "for (int32_t batch_idx = 0; batch_idx < 4; ++batch_idx)"
        in conv_source
    )
    assert "qk_prepared_handle" in conv_source
    assert "gate_prepared_handle" in conv_source
    assert "conv_out_handle" in conv_source
    assert (
        "TCVT(qk_fp32, conv_half, RoundMode::CAST_NONE)"
        in conv_source
    )
    assert "TCVT(gate_a_half, gate_x, RoundMode::CAST_RINT)" in conv_source
    assert "TCVT(gate_x, gate_a_half, RoundMode::CAST_NONE)" in conv_source
    assert "TCVT(gate_b_half, gate_beta, RoundMode::CAST_RINT)" in conv_source
    assert "TCVT(gate_beta, gate_b_half, RoundMode::CAST_NONE)" in conv_source
