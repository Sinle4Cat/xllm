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

import argparse
import ast
import functools
import math
from pathlib import Path
import re
from typing import Any

import tilelang
import tilelang.language as T

from ....common.spec import DispatchField, TilelangKernel, register_kernel


SUPPORTED_SPECULATIVE_TOKENS = (1, 2, 3, 4, 5, 8)
DEFAULT_MAX_BATCH_SIZE = 4
DEFAULT_NUM_STATE_SLOTS = 256
DEFAULT_NUM_K_HEADS = 8
DEFAULT_NUM_V_HEADS = 24
HEAD_DIM = 128
CONV_WIDTH = 4
CONV_CHANNEL_TILE = 128
VEC_NUM = 2
VALUE_TILE = HEAD_DIM // VEC_NUM
GATE_VECTOR_DIM = 64
L2_NORM_EPS = 1e-6
RMS_NORM_EPS = 1e-6
SOFTPLUS_THRESHOLD = 20.0
INPUT_DTYPE = "bfloat16"
DISPATCH_DTYPE = "bf16"
ACCUM_DTYPE = "float"
OUTPUT_INDICES = (13, 14, 15, 16)
A3_UB_CAPACITY_BYTES = 192 * 1024

_PTO_INTRINSIC_COUNTS_PER_TOKEN = {
    "TROWEXPAND(": 2,
    "TCOLSUM(": 2,
    "TCOLEXPAND(": 1,
}
_PTO_UB_DTYPE_BYTES = {
    "bfloat16_t": 2,
    "float": 4,
    "int": 4,
    "uint8_t": 1,
}
_PTO_UB_DECLARATION = re.compile(
    r"TileUbData(?:ND|DN)<\s*(?P<dtype>[^,>]+)\s*,"
    r"\s*(?P<rows>\d+)\s*,\s*(?P<cols>\d+)"
    r"(?:\s*,[^>]*)?>\s+(?P<name>[A-Za-z_]\w*)\s*;"
)
_PTO_UB_ASSIGNMENT = re.compile(
    r"TASSIGN\(\s*(?P<name>[A-Za-z_]\w*)\s*,\s*(?P<offset>\d+)"
)
_PTO_READOUT_PRODUCER_ASSIGNMENT = re.compile(
    r"TASSIGN\(\s*readout_cache_temp_0\s*,\s*(?P<base>\d+)\s*\+\s*"
    r"\(\((?P<token>[A-Za-z_]\w*)\s*\*\s*128\)\s*\+\s*"
    r"\(value_half_idx\s*\*\s*64\)\)\s*\*\s*2\s*\);"
)
_PTO_READOUT_CONSUMER_ASSIGNMENT = re.compile(
    r"TASSIGN\(\s*readout_cache_temp_1\s*,\s*(?P<base>\d+)\s*\+\s*"
    r"\((?P<token>[A-Za-z_]\w*)\s*\*\s*128\)\s*\*\s*2\s*\);"
)
_PTO_OUT_LOAD = re.compile(
    r"tl::ascend_pto::copy_gm_to_ub<[^>\n]+>\(\s*out_handle\b"
)
_PTO_OUT_STORE = re.compile(
    r"tl::ascend_pto::copy_ub_to_gm<(?P<template>[^>\n]+)>\(\s*"
    r"out_handle\s*\+\s*(?P<address>[^,\n]+),\s*"
    r"(?P<ub_offset>\d+),\s*(?P<tail>[^;\n]+)\);"
)
_PTO_FINAL_OUTPUT_PIPELINE = re.compile(
    r"set_flag\(PIPE_MTE3,\s*PIPE_V,\s*"
    r"(?P<ready_event>EVENT_ID\d+)\);\s*"
    r"wait_flag\(PIPE_MTE3,\s*PIPE_V,\s*"
    r"(?P=ready_event)\);\s*"
    r"TCVT\(final_half,\s*norm_fp32,\s*"
    r"RoundMode::CAST_RINT\);\s*"
    r"set_flag\(PIPE_V,\s*PIPE_MTE3,\s*"
    r"(?P<free_event>EVENT_ID\d+)\);\s*"
    r"wait_flag\(PIPE_V,\s*PIPE_MTE3,\s*"
    r"(?P=free_event)\);\s*"
    r"(?:pipe_barrier\(PIPE_MTE3\);\s*)?"
)

PTO_PASS_CONFIGS = {
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_sync": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def _validate_specialization(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int,
    num_v_heads: int,
    dtype: str,
) -> None:
    if speculative_tokens not in SUPPORTED_SPECULATIVE_TOKENS:
        raise ValueError(
            f"unsupported K={speculative_tokens}; expected "
            f"{SUPPORTED_SPECULATIVE_TOKENS}"
        )
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    if num_state_slots < max_batch_size:
        raise ValueError("num_state_slots must cover max_batch_size")
    if num_k_heads < 1 or num_k_heads > 16:
        raise ValueError("num_k_heads must be in [1, 16]")
    if num_k_heads & (num_k_heads - 1):
        raise ValueError("num_k_heads must be a power of two")
    if num_v_heads % num_k_heads != 0:
        raise ValueError("num_v_heads must be divisible by num_k_heads")
    heads_per_k = num_v_heads // num_k_heads
    if heads_per_k < 1 or heads_per_k > 4:
        raise ValueError("num_v_heads / num_k_heads must be in [1, 4]")
    if dtype != DISPATCH_DTYPE:
        raise ValueError(
            f"mega_gdn_mtp_decode only supports dtype={DISPATCH_DTYPE}, "
            f"got {dtype}"
        )


def build_mega_gdn_mtp_decode_kernel(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int = DEFAULT_NUM_K_HEADS,
    num_v_heads: int = DEFAULT_NUM_V_HEADS,
) -> Any:
    """Build the key-head-owned TileLang PTO kernel.

    One AIV owns a batch/key-head pair: its Q/K Conv shards, the corresponding
    value-head Conv shards, recurrent states, checkpoints, and normalized
    outputs. This keeps every producer/consumer dependency local without
    requiring the unsupported cross-AIV ``T.sync_all`` lowering.
    """

    _validate_specialization(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        dtype=DISPATCH_DTYPE,
    )
    sequence_length = speculative_tokens + 1
    conv_state_length = speculative_tokens + 3
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM
    heads_per_k = num_v_heads // num_k_heads
    conv_tiles_per_owner = 2 + heads_per_k
    logical_task_count = max_batch_size * num_k_heads
    kernel_block_count = (
        logical_task_count + VEC_NUM - 1
    ) // VEC_NUM
    q_scale = 1.0 / math.sqrt(HEAD_DIM)
    gate_mask_bytes = GATE_VECTOR_DIM // 8
    token_loop = T.serial
    cache_norm_weight = speculative_tokens == 8
    cache_qk_native = speculative_tokens == 8

    if cache_qk_native:
        qk_cache_shape = [sequence_length, HEAD_DIM, 1]

        @T.macro
        def _store_qk_cache(cache, token_idx, value):
            T.copy(value, cache[token_idx, :, 0])

        @T.macro
        def _broadcast_qk_cache(dst, cache, column, token_idx):
            T.tile.broadcast(dst, cache[token_idx, :, :])

    else:
        qk_cache_shape = [sequence_length, HEAD_DIM]

        @T.macro
        def _store_qk_cache(cache, token_idx, value):
            T.copy(value, cache[token_idx, :])

        @T.macro
        def _broadcast_qk_cache(dst, cache, column, token_idx):
            T.copy(cache[token_idx, :], column[:, 0])
            T.tile.broadcast(dst, column)

    @T.prim_func
    def main(
        qkv: T.Tensor(
            [max_batch_size, sequence_length, conv_dim], INPUT_DTYPE
        ),
        z: T.Tensor(
            [max_batch_size, sequence_length, num_v_heads, HEAD_DIM],
            INPUT_DTYPE,
        ),
        b: T.Tensor(
            [max_batch_size, sequence_length, num_v_heads], INPUT_DTYPE
        ),
        a: T.Tensor(
            [max_batch_size, sequence_length, num_v_heads], INPUT_DTYPE
        ),
        conv_weight: T.Tensor([CONV_WIDTH, conv_dim], INPUT_DTYPE),
        conv_state: T.Tensor(
            [num_state_slots, conv_state_length, conv_dim], INPUT_DTYPE
        ),
        a_log: T.Tensor([num_v_heads], ACCUM_DTYPE),
        dt_bias: T.Tensor([num_v_heads], ACCUM_DTYPE),
        ssm_state: T.Tensor(
            [
                num_state_slots * sequence_length,
                num_v_heads,
                HEAD_DIM,
                HEAD_DIM,
            ],
            ACCUM_DTYPE,
        ),
        read_state_indices: T.Tensor([max_batch_size], "int32"),
        write_state_indices: T.Tensor([max_batch_size], "int32"),
        num_accepted_tokens: T.Tensor([max_batch_size], "int32"),
        norm_weight: T.Tensor([HEAD_DIM], INPUT_DTYPE),
        conv_out: T.Tensor(
            [max_batch_size, sequence_length, conv_dim], INPUT_DTYPE
        ),
        conv_state_out: T.Tensor(
            [num_state_slots, conv_state_length, conv_dim], INPUT_DTYPE
        ),
        ssm_state_out: T.Tensor(
            [
                num_state_slots * sequence_length,
                num_v_heads,
                HEAD_DIM,
                HEAD_DIM,
            ],
            ACCUM_DTYPE,
        ),
        out: T.Tensor(
            [max_batch_size, sequence_length, num_v_heads, HEAD_DIM],
            INPUT_DTYPE,
        ),
        batch_size: T.int32,
    ):
        with T.Kernel(kernel_block_count, is_npu=True) as (cid, vid):
            task_idx = cid * VEC_NUM + vid
            if task_idx < logical_task_count:
                batch_idx = task_idx // num_k_heads
                key_head_idx = task_idx % num_k_heads
                if batch_idx < batch_size:
                    read_slot = read_state_indices[batch_idx]
                    write_slot = write_state_indices[batch_idx]
                    accepted = num_accepted_tokens[batch_idx]
                    value_head_start = key_head_idx * heads_per_k

                    # Phase 1: this owner computes Q, K, and its value-head
                    # Conv shards. All written channel ranges are disjoint.
                    for owner_tile_idx in T.unroll(conv_tiles_per_owner):
                        channel_idx = T.if_then_else(
                            owner_tile_idx < 2,
                            owner_tile_idx * num_k_heads + key_head_idx,
                            2 * num_k_heads
                            + value_head_start
                            + owner_tile_idx
                            - 2,
                        )
                        channel_offset = channel_idx * CONV_CHANNEL_TILE

                        weight0_half = T.alloc_ub(
                            [CONV_CHANNEL_TILE], INPUT_DTYPE
                        )
                        weight1_half = T.alloc_ub(
                            [CONV_CHANNEL_TILE], INPUT_DTYPE
                        )
                        weight2_half = T.alloc_ub(
                            [CONV_CHANNEL_TILE], INPUT_DTYPE
                        )
                        weight3_half = T.alloc_ub(
                            [CONV_CHANNEL_TILE], INPUT_DTYPE
                        )
                        history0_half = T.alloc_ub(
                            [CONV_CHANNEL_TILE], INPUT_DTYPE
                        )
                        history1_half = T.alloc_ub(
                            [CONV_CHANNEL_TILE], INPUT_DTYPE
                        )
                        history2_half = T.alloc_ub(
                            [CONV_CHANNEL_TILE], INPUT_DTYPE
                        )
                        token_half = T.alloc_ub(
                            [CONV_CHANNEL_TILE], INPUT_DTYPE
                        )
                        conv_half = T.alloc_ub(
                            [CONV_CHANNEL_TILE], INPUT_DTYPE
                        )
                        saved_tail0 = T.alloc_ub(
                            [CONV_CHANNEL_TILE], INPUT_DTYPE
                        )
                        saved_tail1 = T.alloc_ub(
                            [CONV_CHANNEL_TILE], INPUT_DTYPE
                        )

                        weight0 = T.alloc_ub(
                            [CONV_CHANNEL_TILE], ACCUM_DTYPE
                        )
                        weight1 = T.alloc_ub(
                            [CONV_CHANNEL_TILE], ACCUM_DTYPE
                        )
                        weight2 = T.alloc_ub(
                            [CONV_CHANNEL_TILE], ACCUM_DTYPE
                        )
                        weight3 = T.alloc_ub(
                            [CONV_CHANNEL_TILE], ACCUM_DTYPE
                        )
                        history0 = T.alloc_ub(
                            [CONV_CHANNEL_TILE], ACCUM_DTYPE
                        )
                        history1 = T.alloc_ub(
                            [CONV_CHANNEL_TILE], ACCUM_DTYPE
                        )
                        history2 = T.alloc_ub(
                            [CONV_CHANNEL_TILE], ACCUM_DTYPE
                        )
                        token_fp32 = T.alloc_ub(
                            [CONV_CHANNEL_TILE], ACCUM_DTYPE
                        )
                        conv_acc = T.alloc_ub(
                            [CONV_CHANNEL_TILE], ACCUM_DTYPE
                        )
                        conv_tmp = T.alloc_ub(
                            [CONV_CHANNEL_TILE], ACCUM_DTYPE
                        )
                        conv_fp32 = T.alloc_ub(
                            [CONV_CHANNEL_TILE], ACCUM_DTYPE
                        )

                        T.copy(
                            conv_weight[0, channel_offset], weight0_half
                        )
                        T.copy(
                            conv_weight[1, channel_offset], weight1_half
                        )
                        T.copy(
                            conv_weight[2, channel_offset], weight2_half
                        )
                        T.copy(
                            conv_weight[3, channel_offset], weight3_half
                        )
                        T.copy(
                            conv_state[
                                read_slot, accepted - 1, channel_offset
                            ],
                            history0_half,
                        )
                        T.copy(
                            conv_state[read_slot, accepted, channel_offset],
                            history1_half,
                        )
                        T.copy(
                            conv_state[
                                read_slot, accepted + 1, channel_offset
                            ],
                            history2_half,
                        )
                        T.tile.cast(
                            weight0,
                            weight0_half,
                            "CAST_NONE",
                            CONV_CHANNEL_TILE,
                        )
                        T.tile.cast(
                            weight1,
                            weight1_half,
                            "CAST_NONE",
                            CONV_CHANNEL_TILE,
                        )
                        T.tile.cast(
                            weight2,
                            weight2_half,
                            "CAST_NONE",
                            CONV_CHANNEL_TILE,
                        )
                        T.tile.cast(
                            weight3,
                            weight3_half,
                            "CAST_NONE",
                            CONV_CHANNEL_TILE,
                        )
                        T.tile.cast(
                            history0,
                            history0_half,
                            "CAST_NONE",
                            CONV_CHANNEL_TILE,
                        )
                        T.tile.cast(
                            history1,
                            history1_half,
                            "CAST_NONE",
                            CONV_CHANNEL_TILE,
                        )
                        T.tile.cast(
                            history2,
                            history2_half,
                            "CAST_NONE",
                            CONV_CHANNEL_TILE,
                        )
                        T.copy(history1_half, saved_tail0)
                        T.copy(history2_half, saved_tail1)

                        for token_idx in token_loop(sequence_length):
                            T.copy(
                                qkv[
                                    batch_idx, token_idx, channel_offset
                                ],
                                token_half,
                            )
                            T.tile.cast(
                                token_fp32,
                                token_half,
                                "CAST_NONE",
                                CONV_CHANNEL_TILE,
                            )
                            T.tile.mul(conv_acc, weight0, history0)
                            T.tile.mul(conv_tmp, weight1, history1)
                            T.tile.add(conv_acc, conv_acc, conv_tmp)
                            T.tile.mul(conv_tmp, weight2, history2)
                            T.tile.add(conv_acc, conv_acc, conv_tmp)
                            T.tile.mul_add_dst(
                                conv_acc, token_fp32, weight3
                            )
                            T.tile.silu(conv_fp32, conv_acc)
                            T.tile.cast(
                                conv_half,
                                conv_fp32,
                                "CAST_RINT",
                                CONV_CHANNEL_TILE,
                            )
                            T.copy(
                                conv_half,
                                conv_out[
                                    batch_idx, token_idx, channel_offset
                                ],
                            )
                            T.copy(
                                token_half,
                                conv_state_out[
                                    write_slot,
                                    token_idx + 2,
                                    channel_offset,
                                ],
                            )
                            T.copy(history1, history0)
                            T.copy(history2, history1)
                            T.copy(token_fp32, history2)

                        T.copy(
                            saved_tail0,
                            conv_state_out[
                                write_slot, 0, channel_offset
                            ],
                        )
                        T.copy(
                            saved_tail1,
                            conv_state_out[
                                write_slot, 1, channel_offset
                            ],
                        )

                    # The same AIV consumes every Conv shard it produced.
                    T.barrier_all()

                    # Phase 2: evaluate this key head's contiguous value-head
                    # gates per token. The fixed physical Vector tile keeps the
                    # padded tail private to this owner.
                    gate_a_log = T.alloc_ub(
                        [GATE_VECTOR_DIM], ACCUM_DTYPE
                    )
                    gate_dt_bias = T.alloc_ub(
                        [GATE_VECTOR_DIM], ACCUM_DTYPE
                    )
                    gate_a_half = T.alloc_ub(
                        [GATE_VECTOR_DIM], INPUT_DTYPE
                    )
                    gate_b_half = T.alloc_ub(
                        [GATE_VECTOR_DIM], INPUT_DTYPE
                    )
                    gate_x = T.alloc_ub(
                        [GATE_VECTOR_DIM], ACCUM_DTYPE
                    )
                    gate_softplus = T.alloc_ub(
                        [GATE_VECTOR_DIM], ACCUM_DTYPE
                    )
                    gate_abs = T.alloc_ub(
                        [GATE_VECTOR_DIM], ACCUM_DTYPE
                    )
                    gate_tmp = T.alloc_ub(
                        [GATE_VECTOR_DIM], ACCUM_DTYPE
                    )
                    gate_beta = T.alloc_ub(
                        [GATE_VECTOR_DIM], ACCUM_DTYPE
                    )
                    gate_cmp_mask = T.alloc_ub(
                        [gate_mask_bytes], "uint8"
                    )
                    decay_cache = T.alloc_ub(
                        [sequence_length, GATE_VECTOR_DIM], ACCUM_DTYPE
                    )
                    beta_cache = T.alloc_ub(
                        [sequence_length, GATE_VECTOR_DIM], ACCUM_DTYPE
                    )

                    T.copy(
                        a_log[
                            value_head_start : value_head_start
                            + heads_per_k
                        ],
                        gate_a_log,
                    )
                    T.copy(
                        dt_bias[
                            value_head_start : value_head_start
                            + heads_per_k
                        ],
                        gate_dt_bias,
                    )
                    T.tile.exp(gate_a_log, gate_a_log)
                    T.tile.mul(gate_a_log, gate_a_log, -1.0)

                    for token_idx in token_loop(sequence_length):
                        T.copy(
                            a[
                                batch_idx,
                                token_idx,
                                value_head_start : value_head_start
                                + heads_per_k,
                            ],
                            gate_a_half,
                        )
                        T.copy(
                            b[
                                batch_idx,
                                token_idx,
                                value_head_start : value_head_start
                                + heads_per_k,
                            ],
                            gate_b_half,
                        )
                        T.tile.cast(
                            gate_x,
                            gate_a_half,
                            "CAST_NONE",
                            GATE_VECTOR_DIM,
                        )
                        T.tile.add(gate_x, gate_x, gate_dt_bias)

                        # Stable softplus from fused_gdn_gating.py:
                        # max(x, 0) + log1p(exp(-abs(x))), with the
                        # threshold branch selected from the original x.
                        T.tile.abs(gate_abs, gate_x)
                        T.tile.mul(gate_tmp, gate_abs, -1.0)
                        T.tile.exp(gate_beta, gate_tmp)
                        T.tile.add(gate_beta, gate_beta, 1.0)
                        T.tile.ln(gate_tmp, gate_beta)
                        T.tile.compare(
                            gate_cmp_mask,
                            gate_x,
                            SOFTPLUS_THRESHOLD,
                            "GT",
                        )
                        T.tile.max(gate_softplus, gate_x, 0.0)
                        T.tile.axpy(
                            gate_softplus, gate_tmp, 1.0
                        )
                        T.tile.select(
                            gate_softplus,
                            gate_cmp_mask,
                            gate_x,
                            gate_softplus,
                            "VSEL_TENSOR_TENSOR_MODE",
                        )

                        T.tile.cast(
                            gate_x,
                            gate_b_half,
                            "CAST_NONE",
                            GATE_VECTOR_DIM,
                        )
                        # Keep sigmoid source and destination separate because
                        # A3 TRECIP does not support an in-place destination.
                        T.tile.sigmoid(gate_beta, gate_x)
                        T.tile.mul(
                            gate_x, gate_a_log, gate_softplus
                        )
                        T.tile.exp(gate_x, gate_x)
                        T.copy(
                            gate_x,
                            decay_cache[token_idx, :],
                        )
                        T.copy(
                            gate_beta,
                            beta_cache[token_idx, :],
                        )

                    T.barrier_all()

                    # All three value heads share this key head's normalized
                    # Q/K. Cache each token once before the value-head loop.
                    read_checkpoint = (
                        read_slot * sequence_length + accepted - 1
                    )
                    q_offset = key_head_idx * HEAD_DIM
                    k_offset = (
                        num_k_heads * HEAD_DIM
                        + key_head_idx * HEAD_DIM
                    )

                    q_half = T.alloc_ub([HEAD_DIM], INPUT_DTYPE)
                    k_half = T.alloc_ub([HEAD_DIM], INPUT_DTYPE)
                    q_fp32 = T.alloc_ub([HEAD_DIM], ACCUM_DTYPE)
                    k_fp32 = T.alloc_ub([HEAD_DIM], ACCUM_DTYPE)
                    q_fp32_cache = T.alloc_ub(qk_cache_shape, ACCUM_DTYPE)
                    k_fp32_cache = T.alloc_ub(qk_cache_shape, ACCUM_DTYPE)
                    norm_square = T.alloc_ub(
                        [1, HEAD_DIM], ACCUM_DTYPE
                    )
                    norm_value = T.alloc_ub([1], ACCUM_DTYPE)

                    for token_idx in token_loop(sequence_length):
                        T.copy(
                            conv_out[batch_idx, token_idx, q_offset],
                            q_half,
                        )
                        T.copy(
                            conv_out[batch_idx, token_idx, k_offset],
                            k_half,
                        )
                        T.tile.cast(
                            q_fp32,
                            q_half,
                            "CAST_NONE",
                            HEAD_DIM,
                        )
                        T.tile.cast(
                            k_fp32,
                            k_half,
                            "CAST_NONE",
                            HEAD_DIM,
                        )

                        T.tile.mul(
                            norm_square[0, :],
                            q_fp32,
                            q_fp32,
                        )
                        T.reduce_sum(
                            norm_square, norm_value, dim=-1
                        )
                        T.tile.add(
                            norm_value,
                            norm_value,
                            L2_NORM_EPS,
                        )
                        T.tile.sqrt(norm_value, norm_value)
                        q_norm = norm_value[0]
                        T.tile.div(q_fp32, q_fp32, q_norm)
                        T.tile.mul(q_fp32, q_fp32, q_scale)
                        _store_qk_cache(
                            q_fp32_cache, token_idx, q_fp32
                        )

                        T.tile.mul(
                            norm_square[0, :],
                            k_fp32,
                            k_fp32,
                        )
                        T.reduce_sum(
                            norm_square, norm_value, dim=-1
                        )
                        T.tile.add(
                            norm_value,
                            norm_value,
                            L2_NORM_EPS,
                        )
                        T.tile.sqrt(norm_value, norm_value)
                        k_norm = norm_value[0]
                        T.tile.div(k_fp32, k_fp32, k_norm)
                        _store_qk_cache(
                            k_fp32_cache, token_idx, k_fp32
                        )

                    # RMSNorm weight is shared by every owned value head and
                    # token. Keep one FP32 copy in UB for the complete task.
                    norm_weight_half = T.alloc_ub(
                        [HEAD_DIM], INPUT_DTYPE
                    )
                    norm_weight_fp32 = T.alloc_ub(
                        [HEAD_DIM], ACCUM_DTYPE
                    )
                    if cache_norm_weight:
                        T.copy(norm_weight[:], norm_weight_half)
                        T.tile.cast(
                            norm_weight_fp32,
                            norm_weight_half,
                            "CAST_NONE",
                            HEAD_DIM,
                        )

                    # Retain one 128x64 state half in UB across all S
                    # recurrent updates for each owned value head.
                    for value_head_offset in T.serial(heads_per_k):
                        value_head_idx = (
                            value_head_start + value_head_offset
                        )
                        # Preserve the original BF16 rounding boundary while
                        # keeping recurrent readouts in UB for RMSNorm.
                        readout_cache = T.alloc_ub(
                            [sequence_length, HEAD_DIM], INPUT_DTYPE
                        )

                        for value_half_idx in T.serial(VEC_NUM):
                            value_offset = value_half_idx * VALUE_TILE
                            value_gm_offset = (
                                2 * num_k_heads * HEAD_DIM
                                + value_head_idx * HEAD_DIM
                                + value_offset
                            )

                            v_half = T.alloc_ub(
                                [VALUE_TILE], INPUT_DTYPE
                            )
                            readout_half = T.alloc_ub(
                                [VALUE_TILE], INPUT_DTYPE
                            )
                            k_column = T.alloc_ub(
                                [HEAD_DIM, 1], ACCUM_DTYPE
                            )
                            v_fp32 = T.alloc_ub(
                                [VALUE_TILE], ACCUM_DTYPE
                            )
                            state_half = T.alloc_ub(
                                [HEAD_DIM, VALUE_TILE], ACCUM_DTYPE
                            )
                            broadcast_buf = T.alloc_ub(
                                [HEAD_DIM, VALUE_TILE], ACCUM_DTYPE
                            )
                            compute_buf = T.alloc_ub(
                                [HEAD_DIM, VALUE_TILE], ACCUM_DTYPE
                            )
                            pred = T.alloc_ub(
                                [1, VALUE_TILE], ACCUM_DTYPE
                            )
                            delta = T.alloc_ub(
                                [1, VALUE_TILE], ACCUM_DTYPE
                            )

                            # TileLang PTO 0.1.4 lowers this 2-D strided GM
                            # slice with an invalid row stride. Keep each row
                            # contiguous and auditable in the semantic JIT.
                            for state_row_idx in T.serial(HEAD_DIM):
                                T.copy(
                                    ssm_state[
                                        read_checkpoint,
                                        value_head_idx,
                                        state_row_idx,
                                        value_offset : value_offset
                                        + VALUE_TILE,
                                    ],
                                    state_half[state_row_idx, :],
                                )

                            for token_idx in token_loop(sequence_length):
                                T.copy(
                                    conv_out[
                                        batch_idx,
                                        token_idx,
                                        value_gm_offset,
                                    ],
                                    v_half,
                                )
                                T.tile.cast(
                                    v_fp32,
                                    v_half,
                                    "CAST_NONE",
                                    VALUE_TILE,
                                )
                                decay = decay_cache[
                                    token_idx, value_head_offset
                                ]
                                beta = beta_cache[
                                    token_idx, value_head_offset
                                ]

                                T.tile.mul(
                                    state_half, state_half, decay
                                )
                                _broadcast_qk_cache(
                                    broadcast_buf,
                                    k_fp32_cache,
                                    k_column,
                                    token_idx,
                                )
                                T.tile.mul(
                                    compute_buf,
                                    state_half,
                                    broadcast_buf,
                                )
                                T.reduce_sum(
                                    compute_buf, pred[0, :], dim=0
                                )
                                T.tile.sub(
                                    delta[0, :],
                                    v_fp32,
                                    pred[0, :],
                                )
                                T.tile.mul(
                                    delta[0, :],
                                    delta[0, :],
                                    beta,
                                )
                                T.tile.broadcast(
                                    compute_buf, delta
                                )
                                T.tile.mul_add_dst(
                                    state_half,
                                    broadcast_buf,
                                    compute_buf,
                                )

                                _broadcast_qk_cache(
                                    broadcast_buf,
                                    q_fp32_cache,
                                    k_column,
                                    token_idx,
                                )
                                T.tile.mul(
                                    compute_buf,
                                    state_half,
                                    broadcast_buf,
                                )
                                T.reduce_sum(
                                    compute_buf, pred[0, :], dim=0
                                )
                                T.tile.cast(
                                    readout_half,
                                    pred[0, :],
                                    "CAST_RINT",
                                    VALUE_TILE,
                                )
                                T.copy(
                                    readout_half,
                                    readout_cache[
                                        token_idx,
                                        value_offset : value_offset
                                        + VALUE_TILE,
                                    ],
                                )
                                write_checkpoint = (
                                    write_slot * sequence_length
                                    + token_idx
                                )
                                for state_row_idx in T.serial(HEAD_DIM):
                                    T.copy(
                                        state_half[state_row_idx, :],
                                        ssm_state_out[
                                            write_checkpoint,
                                            value_head_idx,
                                            state_row_idx,
                                            value_offset : value_offset
                                            + VALUE_TILE,
                                        ],
                                    )

                            # MTE3 must release state_half before the next
                            # value half reloads it through MTE2.
                            T.set_flag("mte3", "mte2", 7)
                            T.wait_flag("mte3", "mte2", 7)

                        # Both BF16 readout halves are now resident in UB.
                        for token_idx in token_loop(sequence_length):
                            z_half = T.alloc_ub(
                                [HEAD_DIM], INPUT_DTYPE
                            )
                            weight_half = T.alloc_ub(
                                [HEAD_DIM], INPUT_DTYPE
                            )
                            final_half = T.alloc_ub(
                                [HEAD_DIM], INPUT_DTYPE
                            )
                            norm_fp32 = T.alloc_ub(
                                [HEAD_DIM], ACCUM_DTYPE
                            )
                            z_fp32 = T.alloc_ub(
                                [HEAD_DIM], ACCUM_DTYPE
                            )
                            gate_fp32 = T.alloc_ub(
                                [HEAD_DIM], ACCUM_DTYPE
                            )
                            weight_fp32 = T.alloc_ub(
                                [HEAD_DIM], ACCUM_DTYPE
                            )
                            square_fp32 = T.alloc_ub(
                                [1, HEAD_DIM], ACCUM_DTYPE
                            )
                            rms = T.alloc_ub([1], ACCUM_DTYPE)

                            T.copy(
                                z[
                                    batch_idx,
                                    token_idx,
                                    value_head_idx,
                                    :,
                                ],
                                z_half,
                            )
                            if not cache_norm_weight:
                                T.copy(norm_weight[:], weight_half)
                            T.tile.cast(
                                norm_fp32,
                                readout_cache[token_idx, :],
                                "CAST_NONE",
                                HEAD_DIM,
                            )
                            T.tile.cast(
                                z_fp32,
                                z_half,
                                "CAST_NONE",
                                HEAD_DIM,
                            )
                            if not cache_norm_weight:
                                T.tile.cast(
                                    weight_fp32,
                                    weight_half,
                                    "CAST_NONE",
                                    HEAD_DIM,
                                )
                            T.tile.mul(
                                square_fp32[0, :],
                                norm_fp32,
                                norm_fp32,
                            )
                            T.reduce_sum(
                                square_fp32, rms, dim=-1
                            )
                            T.tile.div(rms, rms, float(HEAD_DIM))
                            T.tile.add(rms, rms, RMS_NORM_EPS)
                            T.tile.sqrt(rms, rms)
                            T.tile.div(
                                norm_fp32, norm_fp32, rms[0]
                            )
                            if cache_norm_weight:
                                T.tile.mul(
                                    norm_fp32,
                                    norm_fp32,
                                    norm_weight_fp32,
                                )
                            else:
                                T.tile.mul(
                                    norm_fp32,
                                    norm_fp32,
                                    weight_fp32,
                                )
                            T.tile.silu(gate_fp32, z_fp32)
                            T.tile.mul(
                                norm_fp32, norm_fp32, gate_fp32
                            )
                            T.tile.cast(
                                final_half,
                                norm_fp32,
                                "CAST_RINT",
                                HEAD_DIM,
                            )
                            T.copy(
                                final_half,
                                out[
                                    batch_idx,
                                    token_idx,
                                    value_head_idx,
                                    :,
                                ],
                            )

    return main


def _generated_pto_ub_high_water_bytes(source: str) -> int:
    declarations: dict[str, tuple[str, int, int]] = {}
    for match in _PTO_UB_DECLARATION.finditer(source):
        declarations[match.group("name")] = (
            match.group("dtype").strip(),
            int(match.group("rows")),
            int(match.group("cols")),
        )

    high_water_bytes = 0
    missing_declarations: set[str] = set()
    for match in _PTO_UB_ASSIGNMENT.finditer(source):
        name = match.group("name")
        if name not in declarations:
            missing_declarations.add(name)
            continue
        dtype, rows, cols = declarations[name]
        if dtype not in _PTO_UB_DTYPE_BYTES:
            raise RuntimeError(
                "generated MegaGdnMtpDecode PTO source uses an unsupported "
                f"UB dtype: {dtype}"
            )
        tile_bytes = rows * cols * _PTO_UB_DTYPE_BYTES[dtype]
        high_water_bytes = max(
            high_water_bytes,
            int(match.group("offset")) + tile_bytes,
        )

    if missing_declarations:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source has TASSIGN entries "
            f"without auditable TileUbData declarations: "
            f"{sorted(missing_declarations)}"
        )
    if high_water_bytes == 0:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source has no auditable UB "
            "assignments"
        )
    return high_water_bytes


def _cxx_integer_divide(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ValueError("division by zero")
    quotient = abs(numerator) // abs(denominator)
    return -quotient if (numerator < 0) != (denominator < 0) else quotient


def _evaluate_generated_integer_expression(
    expression: str,
    variables: dict[str, int],
) -> int:
    """Evaluate a generated address expression without executing source."""

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"invalid integer expression: {expression}") from error

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"unknown variable: {node.id}")
            return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(
                f"unsupported unary operator: {type(node.op).__name__}"
            )
        if isinstance(node, ast.BinOp):
            lhs = evaluate(node.left)
            rhs = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, ast.Div):
                return _cxx_integer_divide(lhs, rhs)
            if isinstance(node.op, ast.Mod):
                return lhs - _cxx_integer_divide(lhs, rhs) * rhs
            raise ValueError(
                f"unsupported binary operator: {type(node.op).__name__}"
            )
        raise ValueError(
            f"unsupported expression node: {type(node).__name__}"
        )

    return evaluate(parsed.body)


def _generated_serial_loop_count(
    source: str,
    variable: str,
    extent: int,
) -> int:
    escaped_variable = re.escape(variable)
    loop = re.compile(
        rf"for\s*\(\s*int32_t\s+{escaped_variable}\s*=\s*0\s*;\s*"
        rf"{escaped_variable}\s*<\s*{extent}\s*;\s*"
        rf"\+\+{escaped_variable}\s*\)"
    )
    return len(loop.findall(source))


def _validate_generated_readout_cache(
    source: str,
    sequence_length: int,
) -> str:
    producer_assignments = list(
        _PTO_READOUT_PRODUCER_ASSIGNMENT.finditer(source)
    )
    consumer_assignments = list(
        _PTO_READOUT_CONSUMER_ASSIGNMENT.finditer(source)
    )
    producer_cast = "TCVT(readout_half, pred, RoundMode::CAST_RINT);"
    producer_move = "TMOV(readout_cache_temp_0, readout_half);"
    consumer_cast = (
        "TCVT(norm_fp32, readout_cache_temp_1, RoundMode::CAST_NONE);"
    )
    marker_counts = {
        producer_cast: source.count(producer_cast),
        producer_move: source.count(producer_move),
        consumer_cast: source.count(consumer_cast),
        "producer_assignment": len(producer_assignments),
        "consumer_assignment": len(consumer_assignments),
    }
    invalid_marker_counts = {
        marker: count
        for marker, count in marker_counts.items()
        if count != 1
    }
    if invalid_marker_counts:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source did not preserve the BF16 "
            "readout cache handoff: "
            f"{invalid_marker_counts}"
        )

    producer_assignment = producer_assignments[0]
    consumer_assignment = consumer_assignments[0]
    producer_token = producer_assignment.group("token")
    consumer_token = consumer_assignment.group("token")
    expected_loop_counts: dict[str, int] = {}
    for token in (producer_token, consumer_token):
        expected_loop_counts[token] = expected_loop_counts.get(token, 0) + 1
    actual_loop_counts = {
        token: _generated_serial_loop_count(
            source, token, sequence_length
        )
        for token in expected_loop_counts
    }
    if actual_loop_counts != expected_loop_counts:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source has invalid readout token "
            f"loops: actual={actual_loop_counts}, "
            f"expected={expected_loop_counts}"
        )
    if producer_assignment.group("base") != consumer_assignment.group("base"):
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source uses different producer "
            "and consumer readout-cache bases"
        )
    readout_cache_assignments = re.findall(
        r"TASSIGN\(\s*readout_cache\s*,\s*(\d+)\s*\);",
        source,
    )
    if (
        len(readout_cache_assignments) != 1
        or producer_assignment.group("base")
        != readout_cache_assignments[0]
    ):
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source readout producer or "
            "consumer does not address readout_cache"
        )

    producer_cast_position = source.find(producer_cast)
    producer_move_position = source.find(producer_move)
    consumer_cast_position = source.find(consumer_cast)
    if not (
        producer_cast_position
        < producer_assignment.start()
        < producer_move_position
        < consumer_assignment.start()
        < consumer_cast_position
    ):
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source reordered the BF16 "
            "readout-cache producer or consumer"
        )
    return consumer_token


def _validate_generated_out_store(
    source: str,
    consumer_token: str,
    sequence_length: int,
    max_batch_size: int,
    num_k_heads: int,
    num_v_heads: int,
) -> None:
    stores = list(_PTO_OUT_STORE.finditer(source))
    if len(stores) != 1:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source has an invalid final out "
            f"store count: {len(stores)}"
        )
    store = stores[0]

    template_arguments = [
        argument.strip() for argument in store.group("template").split(",")
    ]
    expected_template_prefix = [
        "bfloat16_t",
        "bfloat16_t",
        "1",
        "1",
        "1",
        "1",
        "128",
        "1",
        "1",
    ]
    if (
        len(template_arguments) != 14
        or template_arguments[:9] != expected_template_prefix
        or template_arguments[-3:] != ["1", "1", "128"]
        or [
            argument.strip()
            for argument in store.group("tail").split(",")
        ]
        != ["0", "1", "128"]
    ):
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source final out store is not a "
            "128-lane BF16 transfer"
        )

    final_half_assignments = re.findall(
        r"TASSIGN\(\s*final_half\s*,\s*(\d+)\s*\);",
        source,
    )
    if (
        len(final_half_assignments) != 1
        or int(store.group("ub_offset")) != int(final_half_assignments[0])
    ):
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source final out store does not "
            "read from final_half"
        )

    pipeline_matches = list(_PTO_FINAL_OUTPUT_PIPELINE.finditer(source))
    if (
        len(pipeline_matches) != 1
        or pipeline_matches[0].end() > store.start()
        or source[pipeline_matches[0].end() : store.start()].strip()
    ):
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source final out store is "
            "missing its MTE3-to-Vector or Vector-to-MTE3 dependency"
        )
    consumer_cast = (
        "TCVT(norm_fp32, readout_cache_temp_1, RoundMode::CAST_NONE);"
    )
    final_gate = "TMUL(norm_fp32, norm_fp32, gate_fp32);"
    if (
        source.count(consumer_cast) != 1
        or source.count(final_gate) != 1
        or not (
            source.find(consumer_cast)
            < source.find(final_gate)
            < pipeline_matches[0].start()
            < store.start()
        )
    ):
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source final out store occurs "
            "before final output conversion and gating"
        )

    heads_per_k = num_v_heads // num_k_heads
    address_expression = store.group("address").strip()
    try:
        parsed_address = ast.parse(address_expression, mode="eval")
    except SyntaxError as error:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source has an invalid final out "
            f"address expression: {address_expression}"
        ) from error
    address_names = {
        node.id for node in ast.walk(parsed_address) if isinstance(node, ast.Name)
    }
    allowed_address_names = {
        "cid",
        "vid",
        consumer_token,
        "value_head_offset",
    }
    if (
        consumer_token not in address_names
        or not address_names <= allowed_address_names
    ):
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source final out address uses "
            "unexpected variables: "
            f"actual={sorted(address_names)}, "
            f"allowed={sorted(allowed_address_names)}"
        )
    for batch_idx in range(max_batch_size):
        for key_head_idx in range(num_k_heads):
            logical_task_idx = batch_idx * num_k_heads + key_head_idx
            cid = logical_task_idx // VEC_NUM
            vid = logical_task_idx % VEC_NUM
            for token_idx in range(sequence_length):
                for value_head_offset in range(heads_per_k):
                    value_head_idx = (
                        key_head_idx * heads_per_k + value_head_offset
                    )
                    variables = {
                        "cid": cid,
                        "vid": vid,
                        consumer_token: token_idx,
                        "value_head_offset": value_head_offset,
                    }
                    try:
                        actual_offset = (
                            _evaluate_generated_integer_expression(
                                address_expression,
                                variables,
                            )
                        )
                    except ValueError as error:
                        raise RuntimeError(
                            "generated MegaGdnMtpDecode PTO source has an "
                            "unauditable final out address expression: "
                            f"{address_expression}"
                        ) from error
                    expected_offset = (
                        (
                            batch_idx * sequence_length + token_idx
                        )
                        * num_v_heads
                        + value_head_idx
                    ) * HEAD_DIM
                    if actual_offset != expected_offset:
                        raise RuntimeError(
                            "generated MegaGdnMtpDecode PTO source has an "
                            "incorrect final out address: "
                            f"expression={address_expression}, "
                            f"batch={batch_idx}, key_head={key_head_idx}, "
                            f"token={token_idx}, "
                            f"value_head_offset={value_head_offset}, "
                            f"actual={actual_offset}, "
                            f"expected={expected_offset}"
                        )


def _validate_generated_pto_source(
    source: str,
    speculative_tokens: int,
    max_batch_size: int,
    num_k_heads: int,
    num_v_heads: int,
) -> None:
    required_markers = (
        '#include "tl_templates/pto/common.h"',
        "#include <pto/pto-inst.hpp>",
        "using namespace pto;",
        'extern "C" void call(',
        "ssm_state_out_handle",
        "conv_state_out_handle",
    )
    missing = [marker for marker in required_markers if marker not in source]
    if missing:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source is missing required "
            f"markers: {missing}"
        )

    singleton_markers = (
        "#if defined(__DAV_VEC__) || defined(__DAV_C220_VEC__)",
        "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);",
        'extern "C" void call(',
    )
    invalid_singletons = {
        marker: source.count(marker)
        for marker in singleton_markers
        if source.count(marker) != 1
    }
    if invalid_singletons:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source has invalid singleton "
            f"marker counts: {invalid_singletons}"
        )

    heads_per_k = num_v_heads // num_k_heads
    sequence_length = speculative_tokens + 1
    logical_task_count = max_batch_size * num_k_heads
    kernel_block_count = (
        logical_task_count + VEC_NUM - 1
    ) // VEC_NUM
    value_head_loop_marker = (
        "for (int32_t value_head_offset = 0; "
        f"value_head_offset < {heads_per_k};"
    )
    task_map_markers = (
        f"launch_kernel<<<{kernel_block_count}, nullptr, stream>>>",
        value_head_loop_marker,
        "set_flag_pipeline<PIPE_MTE3, PIPE_MTE2> (7);",
        "wait_flag_pipeline<PIPE_MTE3, PIPE_MTE2> (7);",
        "pto::PadValue::Zero>(a_log_handle",
        "pto::PadValue::Zero>(dt_bias_handle",
    )
    missing_task_map_markers = [
        marker for marker in task_map_markers if marker not in source
    ]
    if missing_task_map_markers:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source is missing task-map "
            f"markers: {missing_task_map_markers}"
        )
    forbidden_task_map_markers = {
        marker: source.count(marker)
        for marker in ("condval", "owner_tile_idx")
        if marker in source
    }
    if forbidden_task_map_markers:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source retained dynamic owner "
            f"expressions: {forbidden_task_map_markers}"
        )

    gate_load_errors: dict[str, list[str]] = {}
    for handle in (
        "a_log_handle",
        "dt_bias_handle",
        "a_handle",
        "b_handle",
    ):
        load_lines = [
            line.strip()
            for line in source.splitlines()
            if "copy_gm_to_ub" in line and f">({handle}" in line
        ]
        if (
            len(load_lines) != 1
            or ", 1, 1, 1, 1, 64, 1, 1," not in load_lines[0]
            or ", 1, 1, 64, pto::PadValue::Zero>" not in load_lines[0]
            or f", 0, 1, {heads_per_k});" not in load_lines[0]
        ):
            gate_load_errors[handle] = load_lines
    if gate_load_errors:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source does not zero-pad each "
            f"Gate input into a full {GATE_VECTOR_DIM}-lane Vector tile: "
            f"{gate_load_errors}"
        )

    value_head_loop_position = source.find(value_head_loop_marker)
    qk_hoist_markers = (
        "TMOV(q_fp32_cache_temp_0",
        "TMOV(k_fp32_cache_temp_0",
    )
    invalid_qk_hoist_markers = {
        marker: source.find(marker)
        for marker in qk_hoist_markers
        if source.count(marker) != 1
        or source.find(marker) >= value_head_loop_position
    }
    qk_reduce_marker = "TROWSUM(norm_value"
    qk_reduce_positions = [
        match.start()
        for match in re.finditer(re.escape(qk_reduce_marker), source)
    ]
    if (
        invalid_qk_hoist_markers
        or len(qk_reduce_positions) != 2
        or any(
            position >= value_head_loop_position
            for position in qk_reduce_positions
        )
    ):
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source did not hoist shared "
            "Q/K normalization before the value-head loop: "
            f"cache_markers={invalid_qk_hoist_markers}, "
            f"reduce_positions={qk_reduce_positions}, "
            f"value_head_loop={value_head_loop_position}"
        )

    out_loads = [
        match.group(0) for match in _PTO_OUT_LOAD.finditer(source)
    ]
    out_stores = [
        match.group(0) for match in _PTO_OUT_STORE.finditer(source)
    ]
    if out_loads or len(out_stores) != 1:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source materialized recurrent "
            "readouts through GM instead of retaining BF16 values in UB: "
            f"out_loads={out_loads}, out_stores={out_stores}"
        )
    consumer_token = _validate_generated_readout_cache(
        source=source,
        sequence_length=sequence_length,
    )
    _validate_generated_out_store(
        source=source,
        consumer_token=consumer_token,
        sequence_length=sequence_length,
        max_batch_size=max_batch_size,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )

    vector_gate_intrinsics = {
        intrinsic: source.count(intrinsic)
        for intrinsic in (
            "compare_scalar(",
            "TMAXS(",
            "TSEL(",
            "TSIGMOID<",
        )
    }
    invalid_vector_gate_intrinsics = {
        intrinsic: count
        for intrinsic, count in vector_gate_intrinsics.items()
        if count != 1
    }
    if invalid_vector_gate_intrinsics:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source has unexpected Vector "
            f"Gate intrinsic counts: {invalid_vector_gate_intrinsics}"
        )
    forbidden_softplus_intrinsics = (
        "TADD(gate_softplus, gate_x, gate_abs)",
        "TMULS(gate_softplus, gate_softplus",
    )
    retained_softplus_intrinsics = [
        marker for marker in forbidden_softplus_intrinsics if marker in source
    ]
    if retained_softplus_intrinsics:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source retained the "
            "negative-infinity-unsafe softplus max formulation: "
            f"{retained_softplus_intrinsics}"
        )

    intrinsic_loop_copies = 1
    invalid_intrinsics = {
        intrinsic: source.count(intrinsic)
        for intrinsic, expected_count in (
            _PTO_INTRINSIC_COUNTS_PER_TOKEN.items()
        )
        if source.count(intrinsic) != expected_count * intrinsic_loop_copies
    }
    if invalid_intrinsics:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source has unexpected recurrent "
            f"intrinsic counts: {invalid_intrinsics}"
        )

    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM
    conv_state_stride = (speculative_tokens + 3) * conv_dim
    checkpoint_elements = num_v_heads * HEAD_DIM * HEAD_DIM
    ssm_slot_stride = sequence_length * checkpoint_elements
    address_markers = (
        f"(read_slot * {conv_state_stride})",
        f"(write_slot * {conv_state_stride})",
        f"(accepted * {conv_dim})",
        f"(read_slot * {ssm_slot_stride})",
        f"(accepted * {checkpoint_elements})",
        f"(write_slot * {ssm_slot_stride})",
    )
    missing_address_markers = [
        marker for marker in address_markers if marker not in source
    ]
    if missing_address_markers:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source is missing state address "
            f"markers: {missing_address_markers}"
        )

    ub_high_water_bytes = _generated_pto_ub_high_water_bytes(source)
    if ub_high_water_bytes > A3_UB_CAPACITY_BYTES:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source exceeds A3 UB capacity: "
            f"{ub_high_water_bytes} > {A3_UB_CAPACITY_BYTES}"
        )


def _mark_generated_kernel_type(source: str) -> str:
    main_body = "{\n  auto cid = get_block_idx();"
    main_end = "}\n\nextern \"C\" __global__ AICORE void launch_kernel"
    if source.count(main_body) != 1 or source.count(main_end) != 1:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source has an unexpected "
            "main_kernel structure"
        )
    source = source.replace(
        main_body,
        "{\n"
        "#if defined(__DAV_VEC__) || defined(__DAV_C220_VEC__)\n"
        "  auto cid = get_block_idx();\n"
        "  set_mask_norm();\n"
        "  set_vector_mask(-1, -1);",
        1,
    )
    source = source.replace(
        main_end,
        "#endif\n}\n\nextern \"C\" __global__ AICORE void launch_kernel",
        1,
    )

    launch_body = "{\n    main_kernel("
    if source.count(launch_body) != 1:
        raise RuntimeError(
            "generated MegaGdnMtpDecode PTO source must contain exactly "
            "one launch_kernel body"
        )
    return source.replace(
        launch_body,
        "{\n"
        "#if !defined(XLLM_TILELANG_PTO_JIT)\n"
        "    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);\n"
        "#endif\n"
        "    main_kernel(",
        1,
    )


def _lower_mega_gdn_mtp_decode_pto(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int,
    num_v_heads: int,
    dtype: str,
) -> tuple[Any, Any, str]:
    _validate_specialization(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        dtype=dtype,
    )
    tilelang.disable_cache()
    tilelang_kernel = build_mega_gdn_mtp_decode_kernel(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    with tilelang.tvm.transform.PassContext(
        opt_level=3,
        config=PTO_PASS_CONFIGS,
    ):
        compiled = tilelang.engine.lower(
            tilelang_kernel,
            target="pto",
            platform="A3",
        )
    source = _mark_generated_kernel_type(compiled.kernel_source)
    _validate_generated_pto_source(
        source,
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    return tilelang_kernel, compiled, source


@functools.lru_cache(maxsize=None)
def mega_gdn_mtp_decode_kernel_jit(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int = DEFAULT_NUM_K_HEADS,
    num_v_heads: int = DEFAULT_NUM_V_HEADS,
) -> Any:
    """Compile the transformed PTO source into a directly callable JIT adapter."""

    tilelang_kernel, compiled, source = _lower_mega_gdn_mtp_decode_pto(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        dtype=DISPATCH_DTYPE,
    )

    # TileLang 0.1.4 has no public source-transform hook on @tilelang.jit.
    # Constructing the adapter from the shared lower result keeps JIT and AOT
    # on the same guarded PTO source instead of compiling the raw lower output.
    from tilelang.jit.adapter import CythonKernelAdapter
    from ..toolchain import compile_pto_jit_library

    library_path = compile_pto_jit_library(source, device="a3")
    return CythonKernelAdapter.from_database(
        params=compiled.params,
        result_idx=list(OUTPUT_INDICES),
        workspace_idx=[],
        auto_gm_idx=[],
        target="pto",
        platform="A3",
        func_or_mod=tilelang_kernel,
        kernel_global_source=source,
        kernel_lib_path=str(library_path),
        verbose=False,
        pass_configs=PTO_PASS_CONFIGS,
    )


@register_kernel
class MegaGdnMtpDecodeKernel(TilelangKernel):
    TARGET = "pto"
    KERNEL_NAME = "mega_gdn_mtp_decode"
    DISPATCH_SCHEMA = [
        DispatchField("speculative_tokens", "int32"),
        DispatchField("max_batch_size", "int32"),
        DispatchField("num_state_slots", "int32"),
        DispatchField("num_k_heads", "int32"),
        DispatchField("num_v_heads", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": (
                f"k{speculative_tokens}_bs{DEFAULT_MAX_BATCH_SIZE}"
                f"_slots{DEFAULT_NUM_STATE_SLOTS}"
                f"_nk{DEFAULT_NUM_K_HEADS}_nv{DEFAULT_NUM_V_HEADS}_bf16"
            ),
            "speculative_tokens": speculative_tokens,
            "max_batch_size": DEFAULT_MAX_BATCH_SIZE,
            "num_state_slots": DEFAULT_NUM_STATE_SLOTS,
            "num_k_heads": DEFAULT_NUM_K_HEADS,
            "num_v_heads": DEFAULT_NUM_V_HEADS,
            "dtype": DISPATCH_DTYPE,
        }
        for speculative_tokens in SUPPORTED_SPECULATIVE_TOKENS
    ]

    @staticmethod
    def generate_source(
        speculative_tokens: int,
        max_batch_size: int,
        num_state_slots: int,
        num_k_heads: int,
        num_v_heads: int,
        dtype: str,
    ) -> str:
        _, _, source = _lower_mega_gdn_mtp_decode_pto(
            speculative_tokens=speculative_tokens,
            max_batch_size=max_batch_size,
            num_state_slots=num_state_slots,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            dtype=dtype,
        )
        return source


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MegaGdnMtpDecode TileLang PTO source."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--speculative-tokens",
        type=int,
        choices=SUPPORTED_SPECULATIVE_TOKENS,
        required=True,
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
    )
    parser.add_argument(
        "--num-state-slots",
        type=int,
        default=DEFAULT_NUM_STATE_SLOTS,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = MegaGdnMtpDecodeKernel.generate_source(
        speculative_tokens=args.speculative_tokens,
        max_batch_size=args.max_batch_size,
        num_state_slots=args.num_state_slots,
        num_k_heads=DEFAULT_NUM_K_HEADS,
        num_v_heads=DEFAULT_NUM_V_HEADS,
        dtype=DISPATCH_DTYPE,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
