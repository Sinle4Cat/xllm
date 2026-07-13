import math

import tilelang
import tilelang.language as T


NUM_CORES = 20
VEC_NUM = 2
NK = 8
NV = 24
DK = 128
DV = 128
V_PER_K = NV // NK
CONV_DIM = 2 * NK * DK + NV * DV
CONV_WIDTH = 4
CONV_STATE_LEN = CONV_WIDTH - 1
CONV_DIM_PER_TASK = 128
VEC_V = DV // VEC_NUM
L2_NORM_EPS = 1e-6
RMS_NORM_EPS = 1e-6
SOFTPLUS_THRESHOLD = 20.0
SCALE = 1.0 / math.sqrt(DK)
INPUT_DTYPE = "bfloat16"
ACCUM_DTYPE = "float"

PASS_CONFIGS = {
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_sync": True,
    "tl.ascend_auto_cross_core_sync": True,
    "tl.ascend_auto_cv_combine": False,
}


def build_qwen35_gdn_decode_superop_kernel(
    num_cache_slots: int, max_batch_size: int
):
    head_passes = math.ceil(
        max_batch_size * NV / (NUM_CORES * VEC_NUM)
    )

    @T.prim_func
    def main(
        qkv: T.Tensor([max_batch_size, CONV_DIM], INPUT_DTYPE),
        z: T.Tensor([max_batch_size, NV, DV], INPUT_DTYPE),
        b: T.Tensor([max_batch_size, NV], INPUT_DTYPE),
        a: T.Tensor([max_batch_size, NV], INPUT_DTYPE),
        conv_weight: T.Tensor([CONV_WIDTH, CONV_DIM], INPUT_DTYPE),
        conv_state: T.Tensor(
            [num_cache_slots, CONV_STATE_LEN, CONV_DIM], INPUT_DTYPE
        ),
        a_log: T.Tensor([NV], ACCUM_DTYPE),
        dt_bias: T.Tensor([NV], ACCUM_DTYPE),
        ssm_state: T.Tensor([num_cache_slots, NV, DK, DV], ACCUM_DTYPE),
        state_indices: T.Tensor([max_batch_size], "int32"),
        norm_weight: T.Tensor([DV], INPUT_DTYPE),
        conv_out: T.Tensor([max_batch_size, CONV_DIM], INPUT_DTYPE),
        conv_state_out: T.Tensor(
            [num_cache_slots, CONV_STATE_LEN, CONV_DIM], INPUT_DTYPE
        ),
        ssm_state_out: T.Tensor(
            [num_cache_slots, NV, DK, DV], ACCUM_DTYPE
        ),
        out: T.Tensor([max_batch_size, NV, DV], INPUT_DTYPE),
        batch_size: T.int32,
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # Phase 1: all 40 vector cores cover 5120 Conv channels.
            conv_task = cid * VEC_NUM + vid
            d_offset = conv_task * CONV_DIM_PER_TASK

            w_half0 = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)
            w_half1 = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)
            w_half2 = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)
            w_half3 = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)
            hist_half0 = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)
            hist_half1 = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)
            hist_half2 = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)
            x_half = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)
            y_half = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)
            save_half0 = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)
            save_half1 = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)
            save_half2 = T.alloc_ub([CONV_DIM_PER_TASK], INPUT_DTYPE)

            w0 = T.alloc_ub([CONV_DIM_PER_TASK], ACCUM_DTYPE)
            w1 = T.alloc_ub([CONV_DIM_PER_TASK], ACCUM_DTYPE)
            w2 = T.alloc_ub([CONV_DIM_PER_TASK], ACCUM_DTYPE)
            w3 = T.alloc_ub([CONV_DIM_PER_TASK], ACCUM_DTYPE)
            hist0 = T.alloc_ub([CONV_DIM_PER_TASK], ACCUM_DTYPE)
            hist1 = T.alloc_ub([CONV_DIM_PER_TASK], ACCUM_DTYPE)
            hist2 = T.alloc_ub([CONV_DIM_PER_TASK], ACCUM_DTYPE)
            x_fp32 = T.alloc_ub([CONV_DIM_PER_TASK], ACCUM_DTYPE)
            conv_acc = T.alloc_ub([CONV_DIM_PER_TASK], ACCUM_DTYPE)
            conv_tmp = T.alloc_ub([CONV_DIM_PER_TASK], ACCUM_DTYPE)
            conv_y = T.alloc_ub([CONV_DIM_PER_TASK], ACCUM_DTYPE)

            T.copy(conv_weight[0, d_offset], w_half0)
            T.copy(conv_weight[1, d_offset], w_half1)
            T.copy(conv_weight[2, d_offset], w_half2)
            T.copy(conv_weight[3, d_offset], w_half3)
            T.set_flag("mte2", "v", 1)
            T.wait_flag("mte2", "v", 1)

            T.tile.cast(w0, w_half0, "CAST_NONE", CONV_DIM_PER_TASK)
            T.tile.cast(w1, w_half1, "CAST_NONE", CONV_DIM_PER_TASK)
            T.tile.cast(w2, w_half2, "CAST_NONE", CONV_DIM_PER_TASK)
            T.tile.cast(w3, w_half3, "CAST_NONE", CONV_DIM_PER_TASK)

            for batch_idx in T.serial(max_batch_size):
                if batch_idx < batch_size:
                    state_idx = state_indices[batch_idx]
                    T.copy(conv_state[state_idx, 0, d_offset], hist_half0)
                    T.copy(conv_state[state_idx, 1, d_offset], hist_half1)
                    T.copy(conv_state[state_idx, 2, d_offset], hist_half2)
                    T.copy(qkv[batch_idx, d_offset], x_half)
                    T.set_flag("mte2", "v", 1)
                    T.wait_flag("mte2", "v", 1)

                    T.tile.cast(
                        hist0, hist_half0, "CAST_NONE", CONV_DIM_PER_TASK
                    )
                    T.tile.cast(
                        hist1, hist_half1, "CAST_NONE", CONV_DIM_PER_TASK
                    )
                    T.tile.cast(
                        hist2, hist_half2, "CAST_NONE", CONV_DIM_PER_TASK
                    )
                    T.tile.cast(
                        x_fp32, x_half, "CAST_NONE", CONV_DIM_PER_TASK
                    )

                    T.tile.mul(conv_acc, w0, hist0)
                    T.tile.mul(conv_tmp, w1, hist1)
                    T.tile.add(conv_acc, conv_acc, conv_tmp)
                    T.tile.mul(conv_tmp, w2, hist2)
                    T.tile.add(conv_acc, conv_acc, conv_tmp)
                    T.tile.mul_add_dst(conv_acc, x_fp32, w3)
                    T.tile.silu(conv_y, conv_acc)

                    # Preserve the standalone Conv BF16 hand-off exactly.
                    T.tile.cast(
                        y_half, conv_y, "CAST_RINT", CONV_DIM_PER_TASK
                    )
                    T.tile.cast(
                        save_half0, hist1, "CAST_RINT", CONV_DIM_PER_TASK
                    )
                    T.tile.cast(
                        save_half1, hist2, "CAST_RINT", CONV_DIM_PER_TASK
                    )
                    T.tile.cast(
                        save_half2, x_fp32, "CAST_RINT", CONV_DIM_PER_TASK
                    )
                    T.set_flag("v", "mte3", 2)
                    T.wait_flag("v", "mte3", 2)
                    T.copy(y_half, conv_out[batch_idx, d_offset])
                    T.copy(
                        save_half0,
                        conv_state_out[state_idx, 0, d_offset],
                    )
                    T.copy(
                        save_half1,
                        conv_state_out[state_idx, 1, d_offset],
                    )
                    T.copy(
                        save_half2,
                        conv_state_out[state_idx, 2, d_offset],
                    )
                    T.set_flag("mte3", "v", 3)
                    T.wait_flag("mte3", "v", 3)

            T.sync_all()

            # Phase 2: each V-head owner keeps recurrent output in UB and
            # immediately applies gated RMSNorm, avoiding a GM phase boundary.
            v_task = cid * VEC_NUM + vid
            for head_pass in T.serial(head_passes):
                head_task = (
                    v_task + head_pass * NUM_CORES * VEC_NUM
                )
                if head_task < batch_size * NV:
                    batch_idx = head_task // NV
                    v_head_idx = head_task % NV
                    k_head_idx = v_head_idx // V_PER_K
                    state_idx = state_indices[batch_idx]

                    q_half = T.alloc_ub([DK], INPUT_DTYPE)
                    k_half = T.alloc_ub([DK], INPUT_DTYPE)
                    v_half = T.alloc_ub([VEC_V], INPUT_DTYPE)
                    a_half = T.alloc_ub([1], INPUT_DTYPE)
                    b_half = T.alloc_ub([1], INPUT_DTYPE)
                    out_half = T.alloc_ub([VEC_V], INPUT_DTYPE)
                    norm_half = T.alloc_ub([DV], INPUT_DTYPE)
                    z_half = T.alloc_ub([DV], INPUT_DTYPE)
                    weight_half = T.alloc_ub([DV], INPUT_DTYPE)
                    final_half = T.alloc_ub([DV], INPUT_DTYPE)

                    q_fp32 = T.alloc_ub([DK], ACCUM_DTYPE)
                    k_fp32 = T.alloc_ub([DK], ACCUM_DTYPE)
                    k_1d = T.alloc_ub([DK, 1], ACCUM_DTYPE)
                    v_fp32 = T.alloc_ub([VEC_V], ACCUM_DTYPE)
                    h_vec = T.alloc_ub([DK, VEC_V], ACCUM_DTYPE)
                    broadcast_buf = T.alloc_ub([DK, VEC_V], ACCUM_DTYPE)
                    compute_buf = T.alloc_ub([DK, VEC_V], ACCUM_DTYPE)
                    pred = T.alloc_ub([1, VEC_V], ACCUM_DTYPE)
                    delta = T.alloc_ub([1, VEC_V], ACCUM_DTYPE)
                    scalar = T.alloc_ub([1], ACCUM_DTYPE)
                    scalar2 = T.alloc_ub([1], ACCUM_DTYPE)
                    scalar_tmp = T.alloc_ub([1], ACCUM_DTYPE)
                    norm_sq = T.alloc_ub([1, DK], ACCUM_DTYPE)
                    norm_val = T.alloc_ub([1], ACCUM_DTYPE)
                    norm_fp32 = T.alloc_ub([DV], ACCUM_DTYPE)
                    z_fp32 = T.alloc_ub([DV], ACCUM_DTYPE)
                    gate_fp32 = T.alloc_ub([DV], ACCUM_DTYPE)
                    weight_fp32 = T.alloc_ub([DV], ACCUM_DTYPE)
                    square_fp32 = T.alloc_ub([1, DV], ACCUM_DTYPE)
                    rms = T.alloc_ub([1], ACCUM_DTYPE)

                    q_offset = k_head_idx * DK
                    k_offset = NK * DK + k_head_idx * DK
                    T.copy(conv_out[batch_idx, q_offset], q_half)
                    T.copy(conv_out[batch_idx, k_offset], k_half)
                    T.copy(
                        a[batch_idx, v_head_idx : v_head_idx + 1],
                        a_half,
                    )
                    T.copy(
                        b[batch_idx, v_head_idx : v_head_idx + 1],
                        b_half,
                    )
                    T.set_flag("mte2", "v", 4)
                    T.wait_flag("mte2", "v", 4)

                    T.tile.cast(q_fp32, q_half, "CAST_NONE", DK)
                    T.tile.cast(k_fp32, k_half, "CAST_NONE", DK)
                    T.tile.cast(scalar, a_half, "CAST_NONE", 1)
                    T.tile.cast(scalar2, b_half, "CAST_NONE", 1)

                    T.tile.mul(norm_sq[0, :], q_fp32, q_fp32)
                    T.reduce_sum(norm_sq, norm_val, dim=-1)
                    T.tile.add(norm_val, norm_val, L2_NORM_EPS)
                    T.tile.sqrt(norm_val, norm_val)
                    q_norm_scalar = norm_val[0]
                    T.tile.div(q_fp32, q_fp32, q_norm_scalar)
                    T.tile.mul(q_fp32, q_fp32, SCALE)

                    T.tile.mul(norm_sq[0, :], k_fp32, k_fp32)
                    T.reduce_sum(norm_sq, norm_val, dim=-1)
                    T.tile.add(norm_val, norm_val, L2_NORM_EPS)
                    T.tile.sqrt(norm_val, norm_val)
                    k_norm_scalar = norm_val[0]
                    T.tile.div(k_fp32, k_fp32, k_norm_scalar)

                    T.copy(
                        a_log[v_head_idx : v_head_idx + 1], scalar_tmp
                    )
                    T.set_flag("mte2", "v", 5)
                    T.wait_flag("mte2", "v", 5)
                    T.tile.exp(scalar_tmp, scalar_tmp)
                    exp_a = scalar_tmp[0]
                    T.set_flag("v", "mte2", 6)
                    T.wait_flag("v", "mte2", 6)
                    T.copy(
                        dt_bias[v_head_idx : v_head_idx + 1], norm_val
                    )
                    T.set_flag("mte2", "v", 6)
                    T.wait_flag("mte2", "v", 6)
                    x_gate = scalar[0] + norm_val[0]
                    if x_gate > SOFTPLUS_THRESHOLD:
                        norm_val[0] = x_gate
                    else:
                        scalar_tmp[0] = x_gate
                        T.tile.exp(scalar_tmp, scalar_tmp)
                        T.tile.add(scalar_tmp, scalar_tmp, 1.0)
                        T.tile.ln(scalar_tmp, scalar_tmp)
                        norm_val[0] = scalar_tmp[0]
                    scalar_tmp[0] = -exp_a * norm_val[0]
                    T.tile.exp(scalar_tmp, scalar_tmp)
                    decay = scalar_tmp[0]
                    T.tile.sigmoid(scalar2, scalar2)
                    beta_gate = scalar2[0]

                    for half_idx in T.serial(VEC_NUM):
                        v_offset = half_idx * VEC_V
                        v_gm_offset = (
                            2 * NK * DK + v_head_idx * DV + v_offset
                        )
                        T.copy(
                            conv_out[batch_idx, v_gm_offset], v_half
                        )
                        T.copy(
                            ssm_state[
                                state_idx,
                                v_head_idx,
                                :,
                                v_offset : v_offset + VEC_V,
                            ],
                            h_vec,
                        )
                        T.set_flag("mte2", "v", 4)
                        T.wait_flag("mte2", "v", 4)

                        T.tile.cast(v_fp32, v_half, "CAST_NONE", VEC_V)
                        T.tile.mul(h_vec, h_vec, decay)
                        T.copy(k_fp32, k_1d[:, 0])
                        T.tile.broadcast(broadcast_buf, k_1d)
                        T.tile.mul(compute_buf, h_vec, broadcast_buf)
                        T.reduce_sum(compute_buf, pred[0, :], dim=0)
                        T.tile.sub(delta[0, :], v_fp32, pred[0, :])
                        T.tile.mul(delta[0, :], delta[0, :], beta_gate)
                        T.tile.broadcast(compute_buf, delta)
                        T.tile.mul_add_dst(
                            h_vec, broadcast_buf, compute_buf
                        )

                        T.copy(q_fp32, k_1d[:, 0])
                        T.tile.broadcast(broadcast_buf, k_1d)
                        T.tile.mul(compute_buf, h_vec, broadcast_buf)
                        T.reduce_sum(compute_buf, pred[0, :], dim=0)
                        T.tile.cast(
                            out_half, pred[0, :], "CAST_RINT", VEC_V
                        )
                        T.copy(
                            out_half,
                            norm_half[v_offset : v_offset + VEC_V],
                        )
                        T.set_flag("v", "mte3", 5)
                        T.wait_flag("v", "mte3", 5)
                        T.copy(
                            h_vec,
                            ssm_state_out[
                                state_idx,
                                v_head_idx,
                                :,
                                v_offset : v_offset + VEC_V,
                            ],
                        )
                        T.set_flag("mte3", "v", 6)
                        T.wait_flag("mte3", "v", 6)

                    T.copy(z[batch_idx, v_head_idx, :], z_half)
                    T.copy(norm_weight[:], weight_half)
                    T.set_flag("mte2", "v", 7)
                    T.wait_flag("mte2", "v", 7)
                    T.tile.cast(norm_fp32, norm_half, "CAST_NONE", DV)
                    T.tile.cast(z_fp32, z_half, "CAST_NONE", DV)
                    T.tile.cast(
                        weight_fp32, weight_half, "CAST_NONE", DV
                    )
                    T.tile.mul(square_fp32[0, :], norm_fp32, norm_fp32)
                    T.reduce_sum(square_fp32, rms, dim=-1)
                    T.tile.div(rms, rms, float(DV))
                    T.tile.add(rms, rms, RMS_NORM_EPS)
                    T.tile.sqrt(rms, rms)
                    T.tile.div(norm_fp32, norm_fp32, rms[0])
                    T.tile.mul(norm_fp32, norm_fp32, weight_fp32)
                    T.tile.silu(gate_fp32, z_fp32)
                    T.tile.mul(norm_fp32, norm_fp32, gate_fp32)
                    T.tile.cast(final_half, norm_fp32, "CAST_RINT", DV)
                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    T.copy(
                        final_half, out[batch_idx, v_head_idx, :]
                    )

    return main


@tilelang.jit(out_idx=[11, 12, 13, 14], pass_configs=PASS_CONFIGS)
def qwen35_gdn_decode_superop_kernel_jit(
    num_cache_slots: int, max_batch_size: int = 1
):
    return build_qwen35_gdn_decode_superop_kernel(
        num_cache_slots, max_batch_size
    )
