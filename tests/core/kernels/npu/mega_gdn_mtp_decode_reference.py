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

from dataclasses import dataclass

import torch
import torch.nn.functional as F


SUPPORTED_SPECULATIVE_TOKENS = frozenset(range(1, 17))
HEAD_DIM = 128


@dataclass(frozen=True)
class MegaGdnMtpDecodeResult:
    conv_out: torch.Tensor
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    out: torch.Tensor


def _check_tensor(
    tensor: torch.Tensor,
    name: str,
    dtype: torch.dtype,
    rank: int,
) -> None:
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}, got {tensor.dtype}")
    if tensor.dim() != rank:
        raise ValueError(f"{name} must have rank {rank}, got {tensor.dim()}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_state_ownership(
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    num_state_slots: int,
) -> None:
    read_ids = [int(value) for value in read_state_indices.tolist()]
    write_ids = [int(value) for value in write_state_indices.tolist()]
    if any(value < 0 or value >= num_state_slots for value in read_ids):
        raise ValueError("read_state_indices contains an out-of-range slot")
    if any(value < 0 or value >= num_state_slots for value in write_ids):
        raise ValueError("write_state_indices contains an out-of-range slot")
    if len(set(write_ids)) != len(write_ids):
        raise ValueError("write_state_indices must be unique within a batch")

    for batch_idx, write_id in enumerate(write_ids):
        for other_batch_idx, read_id in enumerate(read_ids):
            if batch_idx != other_batch_idx and write_id == read_id:
                raise ValueError(
                    "a write slot cannot be another batch row's read slot"
                )


def _validate_inputs(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[int, int, int, int]:
    _check_tensor(qkv, "qkv", torch.bfloat16, 3)
    _check_tensor(z, "z", torch.bfloat16, 4)
    _check_tensor(b, "b", torch.bfloat16, 3)
    _check_tensor(a, "a", torch.bfloat16, 3)
    _check_tensor(conv_weight, "conv_weight", torch.bfloat16, 2)
    _check_tensor(conv_state, "conv_state", torch.bfloat16, 3)
    _check_tensor(a_log, "a_log", torch.float32, 1)
    _check_tensor(dt_bias, "dt_bias", torch.float32, 1)
    _check_tensor(ssm_state, "ssm_state", torch.float32, 4)
    _check_tensor(read_state_indices, "read_state_indices", torch.int32, 1)
    _check_tensor(write_state_indices, "write_state_indices", torch.int32, 1)
    _check_tensor(num_accepted_tokens, "num_accepted_tokens", torch.int32, 1)
    _check_tensor(norm_weight, "norm_weight", torch.bfloat16, 1)

    batch_size, sequence_length, conv_dim = qkv.shape
    speculative_tokens = sequence_length - 1
    if speculative_tokens not in SUPPORTED_SPECULATIVE_TOKENS:
        raise ValueError(
            f"unsupported K={speculative_tokens}; expected "
            f"{sorted(SUPPORTED_SPECULATIVE_TOKENS)}"
        )
    if norm_weight.numel() != HEAD_DIM:
        raise ValueError(f"norm_weight must contain {HEAD_DIM} elements")
    if z.shape[:2] != (batch_size, sequence_length):
        raise ValueError("z batch/sequence dimensions must match qkv")

    num_v_heads = z.size(2)
    if z.size(3) != HEAD_DIM:
        raise ValueError(f"z head dimension must be {HEAD_DIM}")
    if a.shape != (batch_size, sequence_length, num_v_heads):
        raise ValueError("a shape must be [B, K+1, NV]")
    if b.shape != a.shape:
        raise ValueError("b shape must match a")

    qk_width = conv_dim - num_v_heads * HEAD_DIM
    if qk_width <= 0 or qk_width % (2 * HEAD_DIM) != 0:
        raise ValueError("qkv last dimension does not encode valid Q/K/V heads")
    num_k_heads = qk_width // (2 * HEAD_DIM)
    if (
        num_k_heads < 1
        or num_k_heads > 16
        or num_k_heads & (num_k_heads - 1)
        or num_v_heads % num_k_heads != 0
        or not 1 <= num_v_heads // num_k_heads <= 4
    ):
        raise ValueError("unsupported Q/K/V head geometry")

    num_state_slots = conv_state.size(0)
    if conv_weight.shape != (4, conv_dim):
        raise ValueError("conv_weight shape must be [4, conv_dim]")
    if conv_state.shape != (
        num_state_slots,
        speculative_tokens + 3,
        conv_dim,
    ):
        raise ValueError("conv_state shape must be [N, K+3, conv_dim]")
    if a_log.shape != (num_v_heads,) or dt_bias.shape != (num_v_heads,):
        raise ValueError("a_log and dt_bias must have shape [NV]")
    if ssm_state.shape != (
        num_state_slots * sequence_length,
        num_v_heads,
        HEAD_DIM,
        HEAD_DIM,
    ):
        raise ValueError("ssm_state checkpoint layout is invalid")
    for name, tensor in (
        ("read_state_indices", read_state_indices),
        ("write_state_indices", write_state_indices),
        ("num_accepted_tokens", num_accepted_tokens),
    ):
        if tensor.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]")

    accepted = num_accepted_tokens.to(torch.int64)
    if bool(torch.any(accepted < 1)) or bool(torch.any(accepted > sequence_length)):
        raise ValueError("num_accepted_tokens entries must be in [1, K+1]")
    _validate_state_ownership(
        read_state_indices, write_state_indices, num_state_slots
    )
    return batch_size, sequence_length, num_k_heads, num_v_heads


def _l2_normalize(value: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + 1e-6)


def _compute_recurrent_gates(
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = (
        -torch.exp(a_log) * F.softplus(a.float() + dt_bias)
    ).to(torch.bfloat16).float()
    decay = torch.exp(g)
    beta = torch.sigmoid(b.float()).to(torch.bfloat16).float()
    return decay, beta


def mega_gdn_mtp_decode_reference(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_out_override: torch.Tensor | None = None,
) -> MegaGdnMtpDecodeResult:
    batch_size, sequence_length, num_k_heads, num_v_heads = _validate_inputs(
        qkv,
        z,
        b,
        a,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        ssm_state,
        read_state_indices,
        write_state_indices,
        num_accepted_tokens,
        norm_weight,
    )
    speculative_tokens = sequence_length - 1
    conv_dim = qkv.size(2)
    state_stride = sequence_length
    if conv_out_override is not None:
        _check_tensor(
            conv_out_override,
            "conv_out_override",
            torch.bfloat16,
            3,
        )
        if conv_out_override.shape != qkv.shape:
            raise ValueError(
                "conv_out_override shape must match qkv"
            )

    conv_state_out = conv_state.clone()
    ssm_state_out = ssm_state.clone()
    conv_outputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []

    read_conv_snapshots = conv_state.index_select(
        0, read_state_indices.to(torch.int64)
    ).clone()
    read_ssm_indices = (
        read_state_indices.to(torch.int64) * state_stride
        + num_accepted_tokens.to(torch.int64)
        - 1
    )
    read_ssm_snapshots = ssm_state.index_select(0, read_ssm_indices).clone()

    for batch_idx in range(batch_size):
        write_slot = int(write_state_indices[batch_idx])
        accepted = int(num_accepted_tokens[batch_idx])
        read_conv = read_conv_snapshots[batch_idx]
        history = read_conv[accepted - 1 : accepted + 2].float()
        token_conv_outputs: list[torch.Tensor] = []
        for token_idx in range(sequence_length):
            token = qkv[batch_idx, token_idx].float()
            conv_acc = (
                (history * conv_weight[:3].float()).sum(dim=0)
                + token * conv_weight[3].float()
            )
            conv_fp32 = conv_acc * torch.reciprocal(
                torch.exp(-conv_acc) + 1.0
            )
            conv_bfloat16 = conv_fp32.to(torch.bfloat16)
            token_conv_outputs.append(conv_bfloat16)
            history = torch.cat((history[1:], token.unsqueeze(0)), dim=0)

        batch_conv = torch.stack(token_conv_outputs)
        if conv_out_override is not None:
            batch_conv = conv_out_override[batch_idx].clone()
        conv_outputs.append(batch_conv)
        conv_state_out[write_slot, :2] = read_conv[accepted : accepted + 2]
        conv_state_out[write_slot, 2 : speculative_tokens + 3] = qkv[batch_idx]

        q = batch_conv[:, : num_k_heads * HEAD_DIM].reshape(
            sequence_length, num_k_heads, HEAD_DIM
        )
        k = batch_conv[
            :, num_k_heads * HEAD_DIM : 2 * num_k_heads * HEAD_DIM
        ].reshape(sequence_length, num_k_heads, HEAD_DIM)
        v = batch_conv[:, 2 * num_k_heads * HEAD_DIM :].reshape(
            sequence_length, num_v_heads, HEAD_DIM
        )
        q = _l2_normalize(q.float()) / HEAD_DIM**0.5
        k = _l2_normalize(k.float())
        repeats = num_v_heads // num_k_heads
        q = q.repeat_interleave(repeats, dim=1)
        k = k.repeat_interleave(repeats, dim=1)

        state = read_ssm_snapshots[batch_idx].float()
        token_outputs: list[torch.Tensor] = []
        for token_idx in range(sequence_length):
            decay, beta = _compute_recurrent_gates(
                a[batch_idx, token_idx],
                b[batch_idx, token_idx],
                a_log,
                dt_bias,
            )
            state = state * decay[:, None, None]
            prediction = torch.einsum("hkv,hk->hv", state, k[token_idx])
            delta = (v[token_idx].float() - prediction) * beta[:, None]
            state = state + torch.einsum("hk,hv->hkv", k[token_idx], delta)
            readout = torch.einsum("hkv,hk->hv", state, q[token_idx])

            checkpoint = write_slot * state_stride + token_idx
            ssm_state_out[checkpoint] = state

            norm_input = readout.to(torch.bfloat16).float()
            rms_inv = torch.rsqrt(
                norm_input.square().mean(dim=-1, keepdim=True) + 1e-6
            )
            norm_output = norm_input * rms_inv * norm_weight.float()
            norm_output = norm_output * F.silu(z[batch_idx, token_idx].float())
            token_outputs.append(norm_output.to(torch.bfloat16))
        outputs.append(torch.stack(token_outputs))

    return MegaGdnMtpDecodeResult(
        conv_out=torch.stack(conv_outputs).reshape(
            batch_size, sequence_length, conv_dim
        ),
        conv_state=conv_state_out,
        ssm_state=ssm_state_out,
        out=torch.stack(outputs),
    )
