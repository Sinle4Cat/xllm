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

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
PRODUCTION_PTO_KERNEL = (
    REPO_ROOT
    / "third_party/xllm_ops/xllm_ops/mega_gdn_mtp_decode/op_kernel"
    / "mega_gdn_mtp_decode_pto_kernel.h"
)
PRODUCTION_TILING = (
    REPO_ROOT
    / "third_party/xllm_ops/xllm_ops/mega_gdn_mtp_decode/op_host"
    / "mega_gdn_mtp_decode_tiling.cpp"
)
PRODUCTION_ENTRY = (
    REPO_ROOT
    / "third_party/xllm_ops/xllm_ops/mega_gdn_mtp_decode/op_kernel"
    / "mega_gdn_mtp_decode.cpp"
)
SUPPORTED_SPECULATIVE_TOKENS = tuple(range(1, 17))
HEAD_DIM = 128
SSM_HEAD_ELEMENTS = HEAD_DIM * HEAD_DIM
MAX_STATE_SLOTS = 1024
MAX_NUM_V_HEADS = 64
INT32_MAX = (1 << 31) - 1
INT64_MAX = (1 << 63) - 1
A3_UB_CAPACITY_BYTES = 192 * 1024
EXPECTED_UB_HIGH_WATERMARK_BYTES = 187936
EXPECTED_REQUIRED_UB_BYTES = 175360
EXPECTED_QK_GROUP_CACHE_REQUIRED_UB_BYTES = 187936
EXPECTED_DEFERRED_NORM_REQUIRED_UB_BYTES = 182816
_UB_DTYPE_BYTES = {
    "bfloat16_t": 2,
    "float": 4,
    "uint8_t": 1,
}
_UB_ADDRESS_PATTERN = re.compile(
    r"constexpr\s+int32_t\s+(kUb\w+)\s*=\s*(\d+)\s*;"
)
_UB_ADDRESS_ALIAS_PATTERN = re.compile(
    r"^\s*(?:constexpr\s+int32_t\s+)?(\w+)\s*="
    r"\s*[^;]*?\?\s*(\w+)\s*:\s*(\w+)\s*;",
    re.MULTILINE,
)
_UB_TILE_ASSIGNMENT_PATTERN = re.compile(
    r"(?:\w+::)*TileUbData(?:ND|DN)<"
    r"\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,[^>]*)?>"
    r"\s+(\w+)\s*;\s*TASSIGN\(\s*\4\s*,\s*(\w+)\s*\);",
    re.MULTILINE,
)


def _parse_ub_addresses(source: str) -> dict[str, int]:
    return {
        name: int(address)
        for name, address in _UB_ADDRESS_PATTERN.findall(source)
    }


def _parse_ub_footprints(source: str) -> dict[str, int]:
    address_aliases = {
        name: (first_address, second_address)
        for name, first_address, second_address in (
            _UB_ADDRESS_ALIAS_PATTERN.findall(source)
        )
    }
    footprints: dict[str, int] = {}
    for dtype, rows, columns, _, address_expression in (
        _UB_TILE_ASSIGNMENT_PATTERN.findall(source)
    ):
        if dtype not in _UB_DTYPE_BYTES:
            raise ValueError(f"unsupported UB tile dtype: {dtype}")
        footprint = (
            int(rows) * int(columns) * _UB_DTYPE_BYTES[dtype]
        )
        addresses = address_aliases.get(
            address_expression, (address_expression,)
        )
        for address in addresses:
            if address.startswith("kUb"):
                footprints[address] = max(
                    footprints.get(address, 0), footprint
                )
    return footprints


def _parse_required_ub_bytes(
    source: str, constant_name: str = "kRequiredUbBytes"
) -> int:
    match = re.search(
        rf"constexpr\s+uint64_t\s+{constant_name}\s*=\s*(\d+)\s*;",
        source,
    )
    if match is None:
        raise ValueError(
            f"{constant_name} is missing from production tiling"
        )
    return int(match.group(1))


def _parse_static_assert_value(source: str, constant_name: str) -> int:
    match = re.search(
        rf"static_assert\(\s*{constant_name}\s*==\s*(\d+)\s*\);",
        source,
    )
    if match is None:
        raise ValueError(
            f"static assertion for {constant_name} is missing"
        )
    return int(match.group(1))


def _last_ssm_state_start_offset(
    num_state_slots: int,
    sequence_length: int,
    num_v_heads: int,
) -> int:
    return (
        num_state_slots * sequence_length * num_v_heads - 1
    ) * SSM_HEAD_ELEMENTS


def test_k8_nv24_crosses_int32_at_607_state_slots() -> None:
    sequence_length = 8 + 1

    assert (
        _last_ssm_state_start_offset(606, sequence_length, 24)
        <= INT32_MAX
    )
    assert (
        _last_ssm_state_start_offset(607, sequence_length, 24)
        > INT32_MAX
    )


@pytest.mark.parametrize(
    "speculative_tokens", SUPPORTED_SPECULATIVE_TOKENS
)
def test_max_host_ssm_offset_fits_int64(
    speculative_tokens: int,
) -> None:
    offset = _last_ssm_state_start_offset(
        MAX_STATE_SLOTS,
        speculative_tokens + 1,
        64,
    )

    assert offset <= INT64_MAX


def test_max_intermediate_ssm_indices_fit_int32() -> None:
    sequence_length = max(SUPPORTED_SPECULATIVE_TOKENS) + 1
    checkpoint_stride = MAX_NUM_V_HEADS * SSM_HEAD_ELEMENTS
    checkpoint = MAX_STATE_SLOTS * sequence_length - 1
    head_offset = (MAX_NUM_V_HEADS - 1) * SSM_HEAD_ELEMENTS

    assert checkpoint_stride <= INT32_MAX
    assert checkpoint <= INT32_MAX
    assert head_offset <= INT32_MAX
    assert checkpoint * checkpoint_stride + head_offset > INT32_MAX


def test_production_pto_promotes_only_ssm_offset_product() -> None:
    source = PRODUCTION_PTO_KERNEL.read_text(encoding="utf-8")

    for variable_name in (
        "ssm_checkpoint_stride",
        "read_checkpoint",
        "write_checkpoint",
    ):
        assert re.search(
            rf"const\s+int32_t\s+{variable_name}\s*=",
            source,
        )
        assert not re.search(
            rf"const\s+int64_t\s+{variable_name}\s*=",
            source,
        )

    for variable_name in ("read_state_offset", "write_state_offset"):
        assert re.search(
            rf"const\s+int64_t\s+{variable_name}\s*=",
            source,
        )
        assert not re.search(
            rf"const\s+int32_t\s+{variable_name}\s*=",
            source,
        )

    for checkpoint in ("read_checkpoint", "write_checkpoint"):
        assert re.search(
            rf"static_cast<int64_t>\({checkpoint}\)\s*\*"
            rf"\s*ssm_checkpoint_stride",
            source,
        )
        assert not re.search(
            rf"static_cast<int64_t>\(\s*{checkpoint}\s*\*"
            rf"\s*ssm_checkpoint_stride\s*\)",
            source,
        )

    assert re.search(
        r"LoadState\(\s*ssm_state_handle\s*\+\s*read_state_offset",
        source,
    )
    assert re.search(
        r"StoreState\(\s*ssm_state_out_handle\s*\+\s*write_state_offset",
        source,
    )


def test_production_pto_ub_footprint_fits_host_reserve() -> None:
    kernel_source = PRODUCTION_PTO_KERNEL.read_text(encoding="utf-8")
    tiling_source = PRODUCTION_TILING.read_text(encoding="utf-8")
    ub_addresses = _parse_ub_addresses(kernel_source)
    ub_footprints = _parse_ub_footprints(kernel_source)

    assert ub_addresses
    assert (
        ub_addresses.keys() - ub_footprints.keys()
        == {"kUbQkCacheTail"}
    )
    assert ub_footprints.keys() <= ub_addresses.keys()
    assert all(address % 32 == 0 for address in ub_addresses.values())

    regions = sorted(
        (
            ub_addresses[name],
            ub_addresses[name] + footprint,
            name,
        )
        for name, footprint in ub_footprints.items()
    )
    for (_, current_end, current_name), (
        next_start,
        _,
        next_name,
    ) in zip(regions, regions[1:]):
        assert current_end <= next_start, (
            f"UB regions overlap: {current_name} ends at {current_end}, "
            f"but {next_name} starts at {next_start}"
        )

    float_scalar_tile_bytes = 1 * 8 * _UB_DTYPE_BYTES["float"]
    assert ub_footprints["kUbALogStatic"] == float_scalar_tile_bytes
    assert ub_footprints["kUbDtBiasStatic"] == float_scalar_tile_bytes
    assert (
        ub_addresses["kUbKHalf"]
        == ub_addresses["kUbQHalf"] + 128 * _UB_DTYPE_BYTES["bfloat16_t"]
    )
    assert (
        ub_addresses["kUbAHalf"]
        >= ub_addresses["kUbQHalf"]
        + 2 * 128 * _UB_DTYPE_BYTES["bfloat16_t"]
    )

    high_watermark = _parse_static_assert_value(
        kernel_source, "kUbDeferredRowsEnd"
    )
    required_ub_bytes = _parse_required_ub_bytes(tiling_source)
    qk_group_cache_required_ub_bytes = _parse_required_ub_bytes(
        tiling_source, "kQkGroupCacheRequiredUbBytes"
    )
    deferred_norm_required_ub_bytes = _parse_required_ub_bytes(
        tiling_source, "kDeferredNormRequiredUbBytes"
    )

    assert high_watermark == EXPECTED_UB_HIGH_WATERMARK_BYTES
    assert required_ub_bytes == EXPECTED_REQUIRED_UB_BYTES
    assert (
        qk_group_cache_required_ub_bytes
        == EXPECTED_QK_GROUP_CACHE_REQUIRED_UB_BYTES
    )
    assert (
        deferred_norm_required_ub_bytes
        == EXPECTED_DEFERRED_NORM_REQUIRED_UB_BYTES
    )
    assert high_watermark <= A3_UB_CAPACITY_BYTES
    assert required_ub_bytes <= high_watermark
    assert qk_group_cache_required_ub_bytes >= high_watermark
    assert deferred_norm_required_ub_bytes <= high_watermark


def test_pto_recurrent_gates_round_through_bfloat16() -> None:
    kernel_source = PRODUCTION_PTO_KERNEL.read_text(encoding="utf-8")

    assert re.search(
        r"TileUbDataND<bfloat16_t,\s*1,\s*16,\s*1,\s*1>\s*"
        r"rounded_gate_half;\s*"
        r"TASSIGN\(rounded_gate_half,\s*kUbExpA\);",
        kernel_source,
    )
    assert re.search(
        r"TCVT\(rounded_gate_half,\s*scalar_tmp,\s*"
        r"RoundMode::CAST_RINT\);.*?"
        r"TCVT\(scalar_tmp,\s*rounded_gate_half,\s*"
        r"RoundMode::CAST_NONE\);.*?"
        r"TEXP\(scalar_work,\s*scalar_tmp\);",
        kernel_source,
        re.DOTALL,
    )
    assert re.search(
        r"TRECIP\(scalar_work,\s*scalar_tmp\);.*?"
        r"TCVT\(rounded_gate_half,\s*scalar_work,\s*"
        r"RoundMode::CAST_RINT\);.*?"
        r"TCVT\(scalar_work,\s*rounded_gate_half,\s*"
        r"RoundMode::CAST_NONE\);",
        kernel_source,
        re.DOTALL,
    )


def test_pto_k1_to_k16_dispatch_limits_static_performance_keys() -> None:
    tiling_source = PRODUCTION_TILING.read_text(encoding="utf-8")
    entry_source = PRODUCTION_ENTRY.read_text(encoding="utf-8")

    assert re.search(
        r"constexpr\s+int64_t\s+kMaxSequenceLength\s*=\s*17\s*;",
        tiling_source,
    )
    for speculative_tokens in (1, 2, 3, 4, 5, 8):
        assert (
            f"RUN_MTP({speculative_tokens}, false, false, false)"
            in entry_source
        )
    assert "RUN_MTP(8, true, true, false)" in entry_source
    assert "RUN_MTP(8, true, true, true)" in entry_source
    assert "TILING_KEY_IS(308)" in entry_source
    assert "RUN_MTP(0, false, false, false)" in entry_source
    for speculative_tokens in (6, 7, 9):
        assert f"RUN_MTP({speculative_tokens}," not in entry_source
    for speculative_tokens in range(10, 17):
        assert (
            f"RUN_MTP({speculative_tokens}, false, true, false)"
            in entry_source
        )

    assert re.search(
        r"kMinDeferredNormSpeculativeTokens\s*=\s*10\s*;",
        tiling_source,
    )
    assert re.search(
        r"kDeferredNormTilingKeyBase\s*=\s*200\s*;",
        tiling_source,
    )


def test_k8_fast_path_keeps_fixed_nine_row_ub_extent() -> None:
    kernel_source = PRODUCTION_PTO_KERNEL.read_text(encoding="utf-8")

    assert re.search(
        r"constexpr\s+int32_t\s+kQkGroupCacheSequenceLength\s*=\s*9\s*;",
        kernel_source,
    )
    assert "kMaxSequenceLength" not in kernel_source
