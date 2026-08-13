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

"""Experimental three-stage TileLang schedule for MegaGdnMtpDecode.

The launch boundary replaces the unsupported cross-AIV synchronization in
TileLang PTO 0.1.4.  The frontend uses channel-tile owners for Conv and computes
each normalized Q/K row and gate once before the launch boundary.  Recurrent
uses value-head-half owners, and a final value-head-owned stage combines the two
recurrent halves for RMSNorm and Z gating.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import tilelang
import tilelang.language as T

from ....common.spec import DispatchField
from .mega_gdn_mtp_decode import (
    ACCUM_DTYPE,
    A3_UB_CAPACITY_BYTES,
    CONV_CHANNEL_TILE,
    CONV_WIDTH,
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_NUM_K_HEADS,
    DEFAULT_NUM_STATE_SLOTS,
    DEFAULT_NUM_V_HEADS,
    DISPATCH_DTYPE,
    GATE_VECTOR_DIM,
    HEAD_DIM,
    INPUT_DTYPE,
    L2_NORM_EPS,
    PTO_PASS_CONFIGS,
    RMS_NORM_EPS,
    SOFTPLUS_THRESHOLD,
    SUPPORTED_SPECULATIVE_TOKENS,
    VALUE_TILE,
    VEC_NUM,
    _generated_pto_ub_high_water_bytes,
    _mark_generated_kernel_type,
    _validate_specialization,
)


CONV_OUTPUT_INDICES = (10, 11, 12, 13)
RECURRENT_OUTPUT_INDICES = (7, 8)
NORM_OUTPUT_INDICES = (3,)
SEGMENTED_DISPATCH_SCHEMA = [
    DispatchField("speculative_tokens", "int32"),
    DispatchField("max_batch_size", "int32"),
    DispatchField("num_state_slots", "int32"),
    DispatchField("num_k_heads", "int32"),
    DispatchField("num_v_heads", "int32"),
    DispatchField("dtype", "dtype"),
]
SEGMENTED_SPECIALIZATIONS = [
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


def _specialization_dimensions(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int,
    num_v_heads: int,
) -> tuple[int, int, int, int]:
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
    return sequence_length, conv_state_length, conv_dim, heads_per_k


def build_mega_gdn_mtp_conv_kernel(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int = DEFAULT_NUM_K_HEADS,
    num_v_heads: int = DEFAULT_NUM_V_HEADS,
) -> Any:
    """Build the channel-tile-owned causal Conv stage."""

    (
        sequence_length,
        conv_state_length,
        conv_dim,
        heads_per_k,
    ) = _specialization_dimensions(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    conv_tile_count = conv_dim // CONV_CHANNEL_TILE
    kernel_block_count = (conv_tile_count + VEC_NUM - 1) // VEC_NUM
    q_scale = 1.0 / math.sqrt(HEAD_DIM)
    gate_mask_bytes = GATE_VECTOR_DIM // 8

    @T.prim_func
    def main(
        qkv: T.Tensor(
            [max_batch_size, sequence_length, conv_dim], INPUT_DTYPE
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
        read_state_indices: T.Tensor([max_batch_size], "int32"),
        write_state_indices: T.Tensor([max_batch_size], "int32"),
        num_accepted_tokens: T.Tensor([max_batch_size], "int32"),
        conv_out: T.Tensor(
            [max_batch_size, sequence_length, conv_dim], INPUT_DTYPE
        ),
        conv_state_out: T.Tensor(
            [num_state_slots, conv_state_length, conv_dim], INPUT_DTYPE
        ),
        qk_prepared: T.Tensor(
            [
                max_batch_size,
                sequence_length,
                2,
                num_k_heads,
                HEAD_DIM,
            ],
            ACCUM_DTYPE,
        ),
        gate_prepared: T.Tensor(
            [max_batch_size, sequence_length, 2, num_v_heads],
            ACCUM_DTYPE,
        ),
        batch_size: T.int32,
    ):
        with T.Kernel(kernel_block_count, is_npu=True) as (cid, vid):
            channel_idx = cid * VEC_NUM + vid
            if channel_idx < conv_tile_count:
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
                qk_fp32 = T.alloc_ub([HEAD_DIM], ACCUM_DTYPE)
                norm_square = T.alloc_ub(
                    [1, HEAD_DIM], ACCUM_DTYPE
                )
                norm_value = T.alloc_ub([1], ACCUM_DTYPE)

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

                gate_key_head_idx = channel_idx - 2 * num_k_heads
                value_head_start = gate_key_head_idx * heads_per_k
                if (
                    channel_idx >= 2 * num_k_heads
                    and channel_idx < 3 * num_k_heads
                ):
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

                for batch_idx in T.serial(max_batch_size):
                    if batch_idx < batch_size:
                        read_slot = read_state_indices[batch_idx]
                        write_slot = write_state_indices[batch_idx]
                        accepted = num_accepted_tokens[batch_idx]
                        T.copy(
                            conv_state[
                                read_slot,
                                accepted - 1,
                                channel_offset,
                            ],
                            history0_half,
                        )
                        T.copy(
                            conv_state[
                                read_slot, accepted, channel_offset
                            ],
                            history1_half,
                        )
                        T.copy(
                            conv_state[
                                read_slot,
                                accepted + 1,
                                channel_offset,
                            ],
                            history2_half,
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

                        for token_idx in T.serial(sequence_length):
                            T.copy(
                                qkv[
                                    batch_idx,
                                    token_idx,
                                    channel_offset,
                                ],
                                token_half,
                            )
                            T.tile.cast(
                                token_fp32,
                                token_half,
                                "CAST_NONE",
                                CONV_CHANNEL_TILE,
                            )
                            T.tile.mul(
                                conv_acc, weight0, history0
                            )
                            T.tile.mul(
                                conv_tmp, weight1, history1
                            )
                            T.tile.add(
                                conv_acc, conv_acc, conv_tmp
                            )
                            T.tile.mul(
                                conv_tmp, weight2, history2
                            )
                            T.tile.add(
                                conv_acc, conv_acc, conv_tmp
                            )
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
                            if channel_idx < 2 * num_k_heads:
                                qk_kind = channel_idx // num_k_heads
                                key_head_idx = channel_idx % num_k_heads
                                T.tile.cast(
                                    qk_fp32,
                                    conv_half,
                                    "CAST_NONE",
                                    HEAD_DIM,
                                )
                                T.tile.mul(
                                    norm_square[0, :],
                                    qk_fp32,
                                    qk_fp32,
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
                                qk_norm = norm_value[0]
                                T.tile.div(
                                    qk_fp32, qk_fp32, qk_norm
                                )
                                if channel_idx < num_k_heads:
                                    T.tile.mul(
                                        qk_fp32, qk_fp32, q_scale
                                    )
                                T.copy(
                                    qk_fp32,
                                    qk_prepared[
                                        batch_idx,
                                        token_idx,
                                        qk_kind,
                                        key_head_idx,
                                        :,
                                    ],
                                )
                            if (
                                channel_idx >= 2 * num_k_heads
                                and channel_idx < 3 * num_k_heads
                            ):
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
                                T.tile.add(
                                    gate_x, gate_x, gate_dt_bias
                                )
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
                                T.tile.max(
                                    gate_softplus, gate_x, 0.0
                                )
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
                                T.tile.sigmoid(gate_beta, gate_x)
                                T.tile.mul(
                                    gate_x,
                                    gate_a_log,
                                    gate_softplus,
                                )
                                # Match the unfused gate hand-off: g is BF16
                                # before exp(g) enters the recurrent update.
                                T.tile.cast(
                                    gate_a_half,
                                    gate_x,
                                    "CAST_RINT",
                                    GATE_VECTOR_DIM,
                                )
                                T.tile.cast(
                                    gate_x,
                                    gate_a_half,
                                    "CAST_NONE",
                                    GATE_VECTOR_DIM,
                                )
                                T.tile.exp(gate_x, gate_x)
                                # torch.sigmoid(BF16) materializes beta as
                                # BF16 before the FP32 recurrent update.
                                T.tile.cast(
                                    gate_b_half,
                                    gate_beta,
                                    "CAST_RINT",
                                    GATE_VECTOR_DIM,
                                )
                                T.tile.cast(
                                    gate_beta,
                                    gate_b_half,
                                    "CAST_NONE",
                                    GATE_VECTOR_DIM,
                                )
                                T.copy(
                                    gate_x[0:heads_per_k],
                                    gate_prepared[
                                        batch_idx,
                                        token_idx,
                                        0,
                                        value_head_start : value_head_start
                                        + heads_per_k,
                                    ],
                                )
                                T.copy(
                                    gate_beta[0:heads_per_k],
                                    gate_prepared[
                                        batch_idx,
                                        token_idx,
                                        1,
                                        value_head_start : value_head_start
                                        + heads_per_k,
                                    ],
                                )
                            T.copy(
                                conv_half,
                                conv_out[
                                    batch_idx,
                                    token_idx,
                                    channel_offset,
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

    return main


def build_mega_gdn_mtp_recurrent_kernel(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int = DEFAULT_NUM_K_HEADS,
    num_v_heads: int = DEFAULT_NUM_V_HEADS,
) -> Any:
    """Build the value-head-half-owned recurrent/checkpoint stage."""

    (
        sequence_length,
        _,
        conv_dim,
        heads_per_k,
    ) = _specialization_dimensions(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    logical_task_count = max_batch_size * num_v_heads * VEC_NUM
    kernel_block_count = (
        logical_task_count + VEC_NUM - 1
    ) // VEC_NUM

    @T.prim_func
    def main(
        conv_out: T.Tensor(
            [max_batch_size, sequence_length, conv_dim], INPUT_DTYPE
        ),
        qk_prepared: T.Tensor(
            [
                max_batch_size,
                sequence_length,
                2,
                num_k_heads,
                HEAD_DIM,
            ],
            ACCUM_DTYPE,
        ),
        gate_prepared: T.Tensor(
            [max_batch_size, sequence_length, 2, num_v_heads],
            ACCUM_DTYPE,
        ),
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
        ssm_state_out: T.Tensor(
            [
                num_state_slots * sequence_length,
                num_v_heads,
                HEAD_DIM,
                HEAD_DIM,
            ],
            ACCUM_DTYPE,
        ),
        readout: T.Tensor(
            [max_batch_size, sequence_length, num_v_heads, HEAD_DIM],
            INPUT_DTYPE,
        ),
        batch_size: T.int32,
    ):
        with T.Kernel(kernel_block_count, is_npu=True) as (cid, vid):
            task_idx = cid * VEC_NUM + vid
            if task_idx < logical_task_count:
                head_task_idx = task_idx // VEC_NUM
                value_half_idx = task_idx % VEC_NUM
                batch_idx = head_task_idx // num_v_heads
                value_head_idx = head_task_idx % num_v_heads
                if batch_idx < batch_size:
                    key_head_idx = value_head_idx // heads_per_k
                    value_offset = value_half_idx * VALUE_TILE
                    read_slot = read_state_indices[batch_idx]
                    write_slot = write_state_indices[batch_idx]
                    accepted = num_accepted_tokens[batch_idx]
                    read_checkpoint = (
                        read_slot * sequence_length + accepted - 1
                    )
                    v_offset = (
                        2 * num_k_heads * HEAD_DIM
                        + value_head_idx * HEAD_DIM
                        + value_offset
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
                    q_column = T.alloc_ub(
                        [HEAD_DIM, 1], ACCUM_DTYPE
                    )
                    k_column = T.alloc_ub(
                        [HEAD_DIM, 1], ACCUM_DTYPE
                    )
                    v_half = T.alloc_ub([VALUE_TILE], INPUT_DTYPE)
                    readout_half = T.alloc_ub(
                        [VALUE_TILE], INPUT_DTYPE
                    )
                    q_fp32 = T.alloc_ub([HEAD_DIM], ACCUM_DTYPE)
                    k_fp32 = T.alloc_ub([HEAD_DIM], ACCUM_DTYPE)
                    v_fp32 = T.alloc_ub(
                        [VALUE_TILE], ACCUM_DTYPE
                    )
                    pred = T.alloc_ub(
                        [1, VALUE_TILE], ACCUM_DTYPE
                    )
                    delta = T.alloc_ub(
                        [1, VALUE_TILE], ACCUM_DTYPE
                    )

                    for state_row_idx in T.serial(HEAD_DIM):
                        T.copy(
                            ssm_state[
                                read_checkpoint,
                                value_head_idx,
                                state_row_idx,
                                value_offset : value_offset + VALUE_TILE,
                            ],
                            state_half[state_row_idx, :],
                        )

                    for token_idx in T.serial(sequence_length):
                        T.copy(
                            qk_prepared[
                                batch_idx,
                                token_idx,
                                0,
                                key_head_idx,
                                :,
                            ],
                            q_fp32,
                        )
                        T.copy(
                            qk_prepared[
                                batch_idx,
                                token_idx,
                                1,
                                key_head_idx,
                                :,
                            ],
                            k_fp32,
                        )
                        T.copy(
                            conv_out[batch_idx, token_idx, v_offset],
                            v_half,
                        )
                        T.tile.cast(
                            v_fp32,
                            v_half,
                            "CAST_NONE",
                            VALUE_TILE,
                        )

                        decay = gate_prepared[
                            batch_idx,
                            token_idx,
                            0,
                            value_head_idx,
                        ]
                        beta = gate_prepared[
                            batch_idx,
                            token_idx,
                            1,
                            value_head_idx,
                        ]

                        T.tile.mul(
                            state_half, state_half, decay
                        )
                        T.copy(k_fp32, k_column[:, 0])
                        T.copy(q_fp32, q_column[:, 0])
                        T.tile.broadcast(broadcast_buf, k_column)
                        T.tile.mul(
                            compute_buf,
                            state_half,
                            broadcast_buf,
                        )
                        T.reduce_sum(
                            compute_buf, pred[0, :], dim=0
                        )
                        T.tile.sub(
                            delta[0, :], v_fp32, pred[0, :]
                        )
                        T.tile.mul(
                            delta[0, :], delta[0, :], beta
                        )
                        T.tile.broadcast(compute_buf, delta)
                        T.tile.mul_add_dst(
                            state_half,
                            broadcast_buf,
                            compute_buf,
                        )
                        T.tile.broadcast(broadcast_buf, q_column)
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
                            readout[
                                batch_idx,
                                token_idx,
                                value_head_idx,
                                value_offset : value_offset + VALUE_TILE,
                            ],
                        )

                        write_checkpoint = (
                            write_slot * sequence_length + token_idx
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

                    T.set_flag("mte3", "mte2", 7)
                    T.wait_flag("mte3", "mte2", 7)

    return main


def build_mega_gdn_mtp_norm_kernel(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int = DEFAULT_NUM_K_HEADS,
    num_v_heads: int = DEFAULT_NUM_V_HEADS,
) -> Any:
    """Build the value-head-owned RMSNorm/Z-gating stage."""

    (
        sequence_length,
        _,
        _,
        _,
    ) = _specialization_dimensions(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    logical_task_count = max_batch_size * num_v_heads
    kernel_block_count = (
        logical_task_count + VEC_NUM - 1
    ) // VEC_NUM

    @T.prim_func
    def main(
        readout: T.Tensor(
            [max_batch_size, sequence_length, num_v_heads, HEAD_DIM],
            INPUT_DTYPE,
        ),
        z: T.Tensor(
            [max_batch_size, sequence_length, num_v_heads, HEAD_DIM],
            INPUT_DTYPE,
        ),
        norm_weight: T.Tensor([HEAD_DIM], INPUT_DTYPE),
        out: T.Tensor(
            [max_batch_size, sequence_length, num_v_heads, HEAD_DIM],
            INPUT_DTYPE,
        ),
        batch_size: T.int32,
    ):
        with T.Kernel(kernel_block_count, is_npu=True) as (cid, vid):
            task_idx = cid * VEC_NUM + vid
            if task_idx < logical_task_count:
                batch_idx = task_idx // num_v_heads
                value_head_idx = task_idx % num_v_heads
                if batch_idx < batch_size:
                    norm_weight_half = T.alloc_ub(
                        [HEAD_DIM], INPUT_DTYPE
                    )
                    norm_weight_fp32 = T.alloc_ub(
                        [HEAD_DIM], ACCUM_DTYPE
                    )
                    T.copy(norm_weight[:], norm_weight_half)
                    T.tile.cast(
                        norm_weight_fp32,
                        norm_weight_half,
                        "CAST_NONE",
                        HEAD_DIM,
                    )

                    for token_idx in T.serial(sequence_length):
                        readout_half = T.alloc_ub(
                            [HEAD_DIM], INPUT_DTYPE
                        )
                        z_half = T.alloc_ub([HEAD_DIM], INPUT_DTYPE)
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
                        square_fp32 = T.alloc_ub(
                            [1, HEAD_DIM], ACCUM_DTYPE
                        )
                        rms = T.alloc_ub([1], ACCUM_DTYPE)

                        T.copy(
                            readout[
                                batch_idx,
                                token_idx,
                                value_head_idx,
                                :,
                            ],
                            readout_half,
                        )
                        T.copy(
                            z[
                                batch_idx,
                                token_idx,
                                value_head_idx,
                                :,
                            ],
                            z_half,
                        )
                        T.tile.cast(
                            norm_fp32,
                            readout_half,
                            "CAST_NONE",
                            HEAD_DIM,
                        )
                        T.tile.cast(
                            z_fp32,
                            z_half,
                            "CAST_NONE",
                            HEAD_DIM,
                        )
                        T.tile.mul(
                            square_fp32[0, :],
                            norm_fp32,
                            norm_fp32,
                        )
                        T.reduce_sum(square_fp32, rms, dim=-1)
                        T.tile.div(rms, rms, float(HEAD_DIM))
                        T.tile.add(rms, rms, RMS_NORM_EPS)
                        T.tile.sqrt(rms, rms)
                        T.tile.div(norm_fp32, norm_fp32, rms[0])
                        T.tile.mul(
                            norm_fp32,
                            norm_fp32,
                            norm_weight_fp32,
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


def _prepare_generated_mixed_kernel_source(source: str) -> str:
    source = _mark_generated_kernel_type(source)
    task_type_block = (
        "#if !defined(XLLM_TILELANG_PTO_JIT)\n"
        "    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);\n"
        "#endif\n"
    )
    if source.count(task_type_block) != 1:
        raise RuntimeError(
            "generated segmented PTO source has an unexpected mixed-kernel "
            "marker count"
        )
    return source.replace(task_type_block, "", 1)


def _lower_pto_kernel(kernel: Any) -> tuple[Any, str]:
    with tilelang.tvm.transform.PassContext(
        opt_level=3,
        config=PTO_PASS_CONFIGS,
    ):
        compiled = tilelang.engine.lower(
            kernel,
            target="pto",
            platform="A3",
        )
    source = _prepare_generated_mixed_kernel_source(compiled.kernel_source)
    return compiled, source


def _lower_mega_gdn_mtp_segmented_stage_pto(
    stage: str,
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int,
    num_v_heads: int,
    dtype: str,
) -> tuple[Any, str]:
    _validate_specialization(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        dtype=dtype,
    )
    builders = {
        "conv": build_mega_gdn_mtp_conv_kernel,
        "recurrent": build_mega_gdn_mtp_recurrent_kernel,
        "norm": build_mega_gdn_mtp_norm_kernel,
    }
    try:
        builder = builders[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported segmented stage: {stage!r}") from exc

    tilelang.disable_cache()
    kernel = builder(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    compiled, source = _lower_pto_kernel(kernel)
    _validate_segmented_pto_stage_source(
        stage=stage,
        source=source,
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    return compiled, source


def lower_mega_gdn_mtp_conv_pto(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int = DEFAULT_NUM_K_HEADS,
    num_v_heads: int = DEFAULT_NUM_V_HEADS,
    dtype: str = DISPATCH_DTYPE,
) -> tuple[Any, str]:
    """Lower and audit the channel-tile-owned Conv stage."""

    return _lower_mega_gdn_mtp_segmented_stage_pto(
        stage="conv",
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        dtype=dtype,
    )


def lower_mega_gdn_mtp_recurrent_pto(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int = DEFAULT_NUM_K_HEADS,
    num_v_heads: int = DEFAULT_NUM_V_HEADS,
    dtype: str = DISPATCH_DTYPE,
) -> tuple[Any, str]:
    """Lower and audit the value-head-half-owned recurrent stage."""

    return _lower_mega_gdn_mtp_segmented_stage_pto(
        stage="recurrent",
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        dtype=dtype,
    )


def lower_mega_gdn_mtp_norm_pto(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int = DEFAULT_NUM_K_HEADS,
    num_v_heads: int = DEFAULT_NUM_V_HEADS,
    dtype: str = DISPATCH_DTYPE,
) -> tuple[Any, str]:
    """Lower and audit the value-head-owned RMSNorm/Z-gating stage."""

    return _lower_mega_gdn_mtp_segmented_stage_pto(
        stage="norm",
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        dtype=dtype,
    )


def lower_mega_gdn_mtp_segmented_pto(
    speculative_tokens: int,
    max_batch_size: int,
    num_state_slots: int,
    num_k_heads: int = DEFAULT_NUM_K_HEADS,
    num_v_heads: int = DEFAULT_NUM_V_HEADS,
) -> tuple[Any, str, Any, str, Any, str]:
    """Lower all segmented stages and enforce their structural gates."""

    conv_compiled, conv_source = lower_mega_gdn_mtp_conv_pto(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    recurrent_compiled, recurrent_source = lower_mega_gdn_mtp_recurrent_pto(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    norm_compiled, norm_source = lower_mega_gdn_mtp_norm_pto(
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_state_slots=num_state_slots,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    return (
        conv_compiled,
        conv_source,
        recurrent_compiled,
        recurrent_source,
        norm_compiled,
        norm_source,
    )


def _segmented_stage_block_count(
    stage: str,
    speculative_tokens: int,
    max_batch_size: int,
    num_k_heads: int,
    num_v_heads: int,
) -> int:
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM
    logical_tasks = {
        "conv": conv_dim // CONV_CHANNEL_TILE,
        "recurrent": max_batch_size * num_v_heads * VEC_NUM,
        "norm": max_batch_size * num_v_heads,
    }
    try:
        task_count = logical_tasks[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported segmented stage: {stage!r}") from exc
    return (task_count + VEC_NUM - 1) // VEC_NUM


def _validate_segmented_pto_stage_source(
    stage: str,
    source: str,
    speculative_tokens: int,
    max_batch_size: int,
    num_k_heads: int,
    num_v_heads: int,
) -> None:
    sequence_length = speculative_tokens + 1
    if "KERNEL_TASK_TYPE_DEFAULT(" in source:
        raise RuntimeError(
            f"segmented {stage} source unexpectedly overrides kernel type"
        )
    blocks = _segmented_stage_block_count(
        stage=stage,
        speculative_tokens=speculative_tokens,
        max_batch_size=max_batch_size,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    launch = f"launch_kernel<<<{blocks}, nullptr, stream>>>"
    if launch not in source:
        raise RuntimeError(
            f"segmented {stage} source has incorrect owner count"
        )
    high_water_bytes = _generated_pto_ub_high_water_bytes(source)
    if high_water_bytes >= A3_UB_CAPACITY_BYTES:
        raise RuntimeError(
            f"segmented {stage} UB high-water "
            f"{high_water_bytes} exceeds A3 capacity"
        )

    if stage == "conv":
        if "ssm_state_handle" in source:
            raise RuntimeError(
                "segmented Conv source unexpectedly accesses SSM"
            )
        if "qk_prepared_handle" not in source:
            raise RuntimeError(
                "segmented Conv source lost normalized Q/K output"
            )
        if "gate_prepared_handle" not in source:
            raise RuntimeError(
                "segmented Conv source lost gate output"
            )
        return
    if stage == "norm":
        if "ssm_state_handle" in source:
            raise RuntimeError(
                "segmented Norm source unexpectedly accesses SSM"
            )
        if "conv_out_handle" in source:
            raise RuntimeError(
                "segmented Norm source unexpectedly accesses Conv output"
            )
        return

    if "conv_weight_handle" in source:
        raise RuntimeError(
            "segmented recurrent source unexpectedly computes Conv"
        )
    if "norm_weight_handle" in source:
        raise RuntimeError(
            "segmented recurrent source unexpectedly computes Norm"
        )
    for removed_input in (
        "b_handle",
        "a_handle",
        "a_log_handle",
        "dt_bias_handle",
    ):
        if removed_input in source:
            raise RuntimeError(
                "segmented recurrent source unexpectedly recomputes "
                f"Prepare input {removed_input}"
            )
    if "state_row_idx" not in source:
        raise RuntimeError(
            "segmented recurrent source lost its audited state-half copy"
        )
    if source.count("TROWEXPAND(") < 2:
        raise RuntimeError(
            "segmented recurrent source is missing recurrent broadcasts"
        )
    if source.count("TCOLSUM(") < 2:
        raise RuntimeError(
            "segmented recurrent source is missing recurrent reductions"
        )
    if f"< {sequence_length};" not in source:
        raise RuntimeError(
            "segmented recurrent source has no specialized token loop"
        )


def _validate_segmented_pto_sources(
    conv_source: str,
    recurrent_source: str,
    norm_source: str,
    speculative_tokens: int,
    max_batch_size: int,
    num_k_heads: int,
    num_v_heads: int,
) -> None:
    for stage_name, source in (
        ("conv", conv_source),
        ("recurrent", recurrent_source),
        ("norm", norm_source),
    ):
        _validate_segmented_pto_stage_source(
            stage=stage_name,
            source=source,
            speculative_tokens=speculative_tokens,
            max_batch_size=max_batch_size,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate segmented MegaGdnMtpDecode TileLang PTO sources."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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
    (
        _,
        conv_source,
        _,
        recurrent_source,
        _,
        norm_source,
    ) = lower_mega_gdn_mtp_segmented_pto(
        speculative_tokens=args.speculative_tokens,
        max_batch_size=args.max_batch_size,
        num_state_slots=args.num_state_slots,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conv.cpp").write_text(
        conv_source, encoding="utf-8"
    )
    (args.output_dir / "recurrent.cpp").write_text(
        recurrent_source, encoding="utf-8"
    )
    (args.output_dir / "norm.cpp").write_text(
        norm_source, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
