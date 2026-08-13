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
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from mega_gdn_mtp_decode_reference import (  # noqa: E402
    HEAD_DIM,
    SUPPORTED_SPECULATIVE_TOKENS,
    _compute_recurrent_gates,
    mega_gdn_mtp_decode_reference,
)


def _make_inputs(
    speculative_tokens: int,
    batch_size: int = 1,
    num_k_heads: int = 1,
    num_v_heads: int = 1,
    num_state_slots: int = 2,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260729 + speculative_tokens)
    sequence_length = speculative_tokens + 1
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM
    return {
        "qkv": torch.randn(
            batch_size,
            sequence_length,
            conv_dim,
            dtype=torch.bfloat16,
            generator=generator,
        ),
        "z": torch.randn(
            batch_size,
            sequence_length,
            num_v_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            generator=generator,
        ),
        "b": torch.randn(
            batch_size,
            sequence_length,
            num_v_heads,
            dtype=torch.bfloat16,
            generator=generator,
        ),
        "a": torch.randn(
            batch_size,
            sequence_length,
            num_v_heads,
            dtype=torch.bfloat16,
            generator=generator,
        ),
        "conv_weight": torch.randn(
            4, conv_dim, dtype=torch.bfloat16, generator=generator
        ),
        "conv_state": torch.randn(
            num_state_slots,
            speculative_tokens + 3,
            conv_dim,
            dtype=torch.bfloat16,
            generator=generator,
        ),
        "a_log": torch.randn(
            num_v_heads, dtype=torch.float32, generator=generator
        ),
        "dt_bias": torch.randn(
            num_v_heads, dtype=torch.float32, generator=generator
        ),
        "ssm_state": torch.randn(
            num_state_slots * sequence_length,
            num_v_heads,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            generator=generator,
        ),
        "read_state_indices": torch.zeros(batch_size, dtype=torch.int32),
        "write_state_indices": torch.ones(batch_size, dtype=torch.int32),
        "num_accepted_tokens": torch.full(
            (batch_size,), sequence_length, dtype=torch.int32
        ),
        "norm_weight": torch.randn(
            HEAD_DIM, dtype=torch.bfloat16, generator=generator
        ),
    }


@pytest.mark.parametrize(
    "speculative_tokens", sorted(SUPPORTED_SPECULATIVE_TOKENS)
)
def test_supports_required_k_values(speculative_tokens: int) -> None:
    inputs = _make_inputs(speculative_tokens)
    result = mega_gdn_mtp_decode_reference(**inputs)
    sequence_length = speculative_tokens + 1
    conv_dim = inputs["qkv"].size(2)

    assert result.conv_out.shape == (1, sequence_length, conv_dim)
    assert result.out.shape == (1, sequence_length, 1, HEAD_DIM)
    assert result.conv_out.dtype == torch.bfloat16
    assert result.out.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "speculative_tokens", sorted(SUPPORTED_SPECULATIVE_TOKENS)
)
def test_conv_extended_window_layout(speculative_tokens: int) -> None:
    inputs = _make_inputs(speculative_tokens)
    accepted = (speculative_tokens + 2) // 2
    inputs["num_accepted_tokens"].fill_(accepted)
    source_state = inputs["conv_state"].clone()
    result = mega_gdn_mtp_decode_reference(**inputs)

    expected = torch.cat(
        (
            source_state[0, accepted : accepted + 2],
            inputs["qkv"][0],
        ),
        dim=0,
    )
    torch.testing.assert_close(result.conv_state[1], expected, rtol=0, atol=0)


def test_prefix_fork_keeps_shared_state_unchanged() -> None:
    inputs = _make_inputs(
        speculative_tokens=4,
        batch_size=2,
        num_state_slots=3,
    )
    inputs["read_state_indices"] = torch.tensor([0, 0], dtype=torch.int32)
    inputs["write_state_indices"] = torch.tensor([1, 2], dtype=torch.int32)
    inputs["num_accepted_tokens"] = torch.tensor([1, 5], dtype=torch.int32)
    conv_before = inputs["conv_state"].clone()
    ssm_before = inputs["ssm_state"].clone()

    result = mega_gdn_mtp_decode_reference(**inputs)

    torch.testing.assert_close(result.conv_state[0], conv_before[0], rtol=0, atol=0)
    torch.testing.assert_close(
        result.ssm_state[:5], ssm_before[:5], rtol=0, atol=0
    )
    assert not torch.equal(result.conv_state[1], conv_before[1])
    assert not torch.equal(result.conv_state[2], conv_before[2])


def test_same_slot_reads_checkpoint_before_overwrite() -> None:
    inputs = _make_inputs(speculative_tokens=2, num_state_slots=1)
    inputs["read_state_indices"].zero_()
    inputs["write_state_indices"].zero_()
    inputs["num_accepted_tokens"].fill_(2)
    inputs["ssm_state"][0].fill_(1.0)
    inputs["ssm_state"][1].fill_(2.0)
    inputs["ssm_state"][2].fill_(3.0)

    from_same_slot = mega_gdn_mtp_decode_reference(**inputs)

    fork_inputs = {name: value.clone() for name, value in inputs.items()}
    fork_inputs["conv_state"] = torch.cat(
        (inputs["conv_state"], inputs["conv_state"].clone()), dim=0
    )
    fork_inputs["ssm_state"] = torch.cat(
        (inputs["ssm_state"], inputs["ssm_state"].clone()), dim=0
    )
    fork_inputs["write_state_indices"].fill_(1)
    from_fork = mega_gdn_mtp_decode_reference(**fork_inputs)

    torch.testing.assert_close(
        from_same_slot.out, from_fork.out, rtol=0, atol=0
    )
    torch.testing.assert_close(
        from_same_slot.ssm_state[:3],
        from_fork.ssm_state[3:],
        rtol=0,
        atol=0,
    )


def test_zero_inputs_produce_zero_outputs() -> None:
    inputs = _make_inputs(speculative_tokens=1)
    for name in ("qkv", "z", "b", "a", "conv_weight", "conv_state", "ssm_state"):
        inputs[name].zero_()
    inputs["a_log"].zero_()
    inputs["dt_bias"].zero_()
    inputs["norm_weight"].fill_(1)

    result = mega_gdn_mtp_decode_reference(**inputs)

    assert torch.count_nonzero(result.conv_out) == 0
    assert torch.count_nonzero(result.out) == 0
    assert torch.count_nonzero(result.ssm_state[2:4]) == 0


def test_recurrent_gates_match_small_op_bfloat16_rounding() -> None:
    a = torch.tensor([-0.75, 0.25], dtype=torch.bfloat16)
    b = torch.tensor([-1.25, 0.75], dtype=torch.bfloat16)
    a_log = torch.tensor([-0.3, 0.7], dtype=torch.float32)
    dt_bias = torch.tensor([-0.2, 0.1], dtype=torch.float32)

    decay, beta = _compute_recurrent_gates(a, b, a_log, dt_bias)

    unrounded_g = -torch.exp(a_log) * torch.nn.functional.softplus(
        a.float() + dt_bias
    )
    expected_decay = torch.exp(unrounded_g.to(torch.bfloat16).float())
    unrounded_beta = torch.sigmoid(b.float())
    expected_beta = unrounded_beta.to(torch.bfloat16).float()
    torch.testing.assert_close(decay, expected_decay, rtol=0, atol=0)
    torch.testing.assert_close(beta, expected_beta, rtol=0, atol=0)
    assert not torch.equal(decay, torch.exp(unrounded_g))
    assert not torch.equal(beta, unrounded_beta)


def test_conv_output_override_drives_recurrent_reference() -> None:
    inputs = _make_inputs(speculative_tokens=1)
    baseline = mega_gdn_mtp_decode_reference(**inputs)
    conv_override = torch.zeros_like(baseline.conv_out)

    overridden = mega_gdn_mtp_decode_reference(
        **inputs,
        conv_out_override=conv_override,
    )

    torch.testing.assert_close(
        overridden.conv_out, conv_override, rtol=0, atol=0
    )
    torch.testing.assert_close(
        overridden.conv_state, baseline.conv_state, rtol=0, atol=0
    )
    assert not torch.equal(overridden.ssm_state, baseline.ssm_state)
    assert not torch.equal(overridden.out, baseline.out)


def test_rejects_invalid_conv_output_override() -> None:
    inputs = _make_inputs(speculative_tokens=1)
    inputs["conv_out_override"] = torch.zeros(
        1,
        1,
        inputs["qkv"].size(2),
        dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError, match="conv_out_override shape"):
        mega_gdn_mtp_decode_reference(**inputs)


@pytest.mark.parametrize("speculative_tokens", [0, 17, 32])
def test_rejects_unvalidated_k_values(speculative_tokens: int) -> None:
    inputs = _make_inputs(speculative_tokens=1)
    sequence_length = speculative_tokens + 1
    conv_dim = inputs["qkv"].size(2)
    inputs["qkv"] = torch.zeros(
        1, sequence_length, conv_dim, dtype=torch.bfloat16
    )
    inputs["z"] = torch.zeros(
        1, sequence_length, 1, HEAD_DIM, dtype=torch.bfloat16
    )
    inputs["a"] = torch.zeros(1, sequence_length, 1, dtype=torch.bfloat16)
    inputs["b"] = torch.zeros(1, sequence_length, 1, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="unsupported K"):
        mega_gdn_mtp_decode_reference(**inputs)


@pytest.mark.parametrize("accepted", [0, 4])
def test_rejects_invalid_accepted_count(accepted: int) -> None:
    inputs = _make_inputs(speculative_tokens=2)
    inputs["num_accepted_tokens"].fill_(accepted)

    with pytest.raises(ValueError, match="num_accepted_tokens"):
        mega_gdn_mtp_decode_reference(**inputs)


def test_rejects_duplicate_write_slots() -> None:
    inputs = _make_inputs(
        speculative_tokens=1,
        batch_size=2,
        num_state_slots=2,
    )
    inputs["read_state_indices"] = torch.tensor([0, 0], dtype=torch.int32)
    inputs["write_state_indices"] = torch.tensor([1, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="write_state_indices must be unique"):
        mega_gdn_mtp_decode_reference(**inputs)


@pytest.mark.parametrize(
    ("index_name", "invalid_slot"),
    (
        ("read_state_indices", -1),
        ("read_state_indices", 2),
        ("write_state_indices", -1),
        ("write_state_indices", 2),
    ),
)
def test_rejects_out_of_range_state_slots(
    index_name: str,
    invalid_slot: int,
) -> None:
    inputs = _make_inputs(speculative_tokens=1, num_state_slots=2)
    inputs[index_name].fill_(invalid_slot)

    with pytest.raises(ValueError, match=rf"{index_name}.*out-of-range"):
        mega_gdn_mtp_decode_reference(**inputs)


def test_rejects_write_to_another_batch_rows_read_slot() -> None:
    inputs = _make_inputs(
        speculative_tokens=1,
        batch_size=2,
        num_state_slots=3,
    )
    inputs["read_state_indices"] = torch.tensor([0, 1], dtype=torch.int32)
    inputs["write_state_indices"] = torch.tensor([1, 2], dtype=torch.int32)

    with pytest.raises(
        ValueError,
        match="write slot cannot be another batch row's read slot",
    ):
        mega_gdn_mtp_decode_reference(**inputs)
