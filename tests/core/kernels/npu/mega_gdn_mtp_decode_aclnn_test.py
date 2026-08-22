# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import ctypes
import os
from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from mega_gdn_mtp_decode_reference import (  # noqa: E402
    HEAD_DIM,
    MegaGdnMtpDecodeResult,
    SUPPORTED_SPECULATIVE_TOKENS,
    mega_gdn_mtp_decode_reference,
)


_ACL_FLOAT = 0
_ACL_INT32 = 3
_ACL_BF16 = 27
_ACL_FORMAT_ND = 2
_ACLNN_LIBRARY_ENV = "MEGA_GDN_MTP_ACLNN_LIB"


def _make_inputs(
    speculative_tokens: int,
    num_state_slots: int = 2,
    num_k_heads: int = 1,
    num_v_heads: int = 1,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(
        20260729
        + speculative_tokens
        + 10 * num_k_heads
        + 100 * num_v_heads
    )
    sequence_length = speculative_tokens + 1
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM

    def random_bfloat16(*shape: int) -> torch.Tensor:
        return (
            torch.randn(*shape, dtype=torch.float32, generator=generator)
            .mul_(0.1)
            .to(torch.bfloat16)
        )

    inputs = {
        "qkv": random_bfloat16(batch_size, sequence_length, conv_dim),
        "z": random_bfloat16(
            batch_size, sequence_length, num_v_heads, HEAD_DIM
        ),
        "b": random_bfloat16(
            batch_size, sequence_length, num_v_heads
        ),
        "a": random_bfloat16(
            batch_size, sequence_length, num_v_heads
        ),
        "conv_weight": random_bfloat16(4, conv_dim),
        "conv_state": random_bfloat16(
            num_state_slots, sequence_length + 2, conv_dim
        ),
        "a_log": torch.full((num_v_heads,), -1.0, dtype=torch.float32),
        "dt_bias": torch.zeros(num_v_heads, dtype=torch.float32),
        "ssm_state": torch.randn(
            num_state_slots * sequence_length,
            num_v_heads,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            generator=generator,
        ).mul_(0.1),
        "read_state_indices": torch.zeros(batch_size, dtype=torch.int32),
        "write_state_indices": torch.ones(batch_size, dtype=torch.int32),
        "num_accepted_tokens": torch.ones(
            batch_size, dtype=torch.int32
        ),
        "norm_weight": random_bfloat16(HEAD_DIM),
    }
    return inputs


class _AclnnMegaGdnMtpDecode:
    def __init__(self, library_path: str) -> None:
        self._torch_npu = pytest.importorskip("torch_npu")
        if not self._torch_npu.npu.is_available():
            pytest.skip("Ascend NPU is not available")

        self._base_lib = ctypes.CDLL(
            "libnnopbase.so", mode=ctypes.RTLD_GLOBAL
        )
        self._custom_lib = ctypes.CDLL(
            library_path, mode=ctypes.RTLD_GLOBAL
        )
        self._configure_functions()

    def _configure_functions(self) -> None:
        self._base_lib.aclCreateTensor.restype = ctypes.c_void_p
        self._base_lib.aclCreateTensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self._base_lib.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        self._base_lib.aclDestroyTensor.restype = ctypes.c_int
        self._base_lib.InitHugeMemThreadLocal.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
        ]
        self._base_lib.InitHugeMemThreadLocal.restype = ctypes.c_int
        self._base_lib.ReleaseHugeMem.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
        ]
        self._base_lib.UnInitHugeMemThreadLocal.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
        ]

        self._get_workspace_size = (
            self._custom_lib.aclnnMegaGdnMtpDecodeGetWorkspaceSize
        )
        self._get_workspace_size.restype = ctypes.c_int
        self._get_workspace_size.argtypes = (
            [ctypes.c_void_p] * 13
            + [ctypes.c_bool]
            + [ctypes.c_void_p] * 4
            + [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
            ]
        )
        self._run = self._custom_lib.aclnnMegaGdnMtpDecode
        self._run.restype = ctypes.c_int
        self._run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]

    def _create_acl_tensor(
        self,
        tensor: torch.Tensor,
        keepers: list[tuple[object, object, object]],
    ) -> int:
        dtype_map = {
            torch.bfloat16: _ACL_BF16,
            torch.float32: _ACL_FLOAT,
            torch.int32: _ACL_INT32,
        }
        dims = (ctypes.c_int64 * tensor.dim())(*tensor.shape)
        strides = (ctypes.c_int64 * tensor.dim())(*tensor.stride())
        storage_dims = (ctypes.c_int64 * 1)(
            tensor.untyped_storage().nbytes() // tensor.element_size()
        )
        keepers.append((dims, strides, storage_dims))
        acl_tensor = self._base_lib.aclCreateTensor(
            dims,
            tensor.dim(),
            dtype_map[tensor.dtype],
            strides,
            tensor.storage_offset(),
            _ACL_FORMAT_ND,
            storage_dims,
            1,
            ctypes.c_void_p(tensor.untyped_storage().data_ptr()),
        )
        if not acl_tensor:
            raise RuntimeError("aclCreateTensor failed")
        return acl_tensor

    def run(
        self,
        inputs: dict[str, torch.Tensor],
        in_place: bool,
    ) -> MegaGdnMtpDecodeResult:
        device_inputs = {
            name: tensor.to("npu") for name, tensor in inputs.items()
        }
        return self.run_device(device_inputs, in_place)

    def run_device(
        self,
        device_inputs: dict[str, torch.Tensor],
        in_place: bool,
    ) -> MegaGdnMtpDecodeResult:
        return self.run_device_many(device_inputs, in_place, launches=1)

    def run_device_many(
        self,
        device_inputs: dict[str, torch.Tensor],
        in_place: bool,
        launches: int,
    ) -> MegaGdnMtpDecodeResult:
        if launches < 1:
            raise ValueError("launches must be positive")
        qkv = device_inputs["qkv"]
        z = device_inputs["z"]
        conv_state = device_inputs["conv_state"]
        ssm_state = device_inputs["ssm_state"]
        conv_out = torch.full_like(qkv, 5.0)
        conv_state_out = (
            conv_state if in_place else conv_state.clone()
        )
        ssm_state_out = ssm_state if in_place else ssm_state.clone()
        out = torch.full_like(z, 5.0)
        tensors = [
            qkv,
            z,
            device_inputs["b"],
            device_inputs["a"],
            device_inputs["conv_weight"],
            conv_state,
            device_inputs["a_log"],
            device_inputs["dt_bias"],
            ssm_state,
            device_inputs["read_state_indices"],
            device_inputs["write_state_indices"],
            device_inputs["num_accepted_tokens"],
            device_inputs["norm_weight"],
            conv_out,
            conv_state_out,
            ssm_state_out,
            out,
        ]

        status = self._base_lib.InitHugeMemThreadLocal(None, False)
        if status != 0:
            raise RuntimeError(
                f"InitHugeMemThreadLocal failed with status {status}"
            )

        keepers: list[tuple[object, object, object]] = []
        acl_tensors: list[int] = []
        try:
            acl_tensors = [
                self._create_acl_tensor(tensor, keepers)
                for tensor in tensors
            ]
            executors: list[ctypes.c_void_p] = []
            workspace_bytes = 0
            for _ in range(launches):
                workspace_size = ctypes.c_uint64(0)
                executor = ctypes.c_void_p()
                status = self._get_workspace_size(
                    *acl_tensors[:13],
                    True,
                    *acl_tensors[13:],
                    ctypes.byref(workspace_size),
                    ctypes.byref(executor),
                )
                if status != 0:
                    raise RuntimeError(
                        "aclnnMegaGdnMtpDecodeGetWorkspaceSize failed "
                        f"with status {status}"
                    )
                executors.append(executor)
                workspace_bytes = max(
                    workspace_bytes, workspace_size.value
                )
            workspace = torch.empty(
                (workspace_bytes,), dtype=torch.uint8, device="npu"
            )
            stream = self._torch_npu.npu.current_stream()
            for executor in executors:
                status = self._run(
                    ctypes.c_void_p(workspace.data_ptr()),
                    workspace_bytes,
                    executor,
                    ctypes.c_void_p(stream.npu_stream),
                )
                if status != 0:
                    raise RuntimeError(
                        "aclnnMegaGdnMtpDecode failed "
                        f"with status {status}"
                    )
            self._torch_npu.npu.synchronize()
        finally:
            for acl_tensor in acl_tensors:
                destroy_status = self._base_lib.aclDestroyTensor(acl_tensor)
                if destroy_status != 0:
                    raise RuntimeError(
                        "aclDestroyTensor failed "
                        f"with status {destroy_status}"
                    )
            self._base_lib.ReleaseHugeMem(None, False)
            self._base_lib.UnInitHugeMemThreadLocal(None, False)

        return MegaGdnMtpDecodeResult(
            conv_out=conv_out.cpu(),
            conv_state=conv_state_out.cpu(),
            ssm_state=ssm_state_out.cpu(),
            out=out.cpu(),
        )


@pytest.fixture(scope="module")
def aclnn_kernel() -> _AclnnMegaGdnMtpDecode:
    library_path = os.environ.get(_ACLNN_LIBRARY_ENV)
    if not library_path:
        pytest.skip(f"{_ACLNN_LIBRARY_ENV} is not set")
    if not Path(library_path).is_file():
        pytest.fail(f"{_ACLNN_LIBRARY_ENV} does not exist: {library_path}")
    if not os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        pytest.fail("ASCEND_CUSTOM_OPP_PATH is not set")
    return _AclnnMegaGdnMtpDecode(library_path)


def _assert_matches_reference(
    actual: MegaGdnMtpDecodeResult,
    expected: MegaGdnMtpDecodeResult,
) -> None:
    torch.testing.assert_close(
        actual.conv_out, expected.conv_out, rtol=8.0e-3, atol=1.0e-6
    )
    torch.testing.assert_close(
        actual.conv_state, expected.conv_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.ssm_state, expected.ssm_state, rtol=5.0e-3, atol=2.5e-5
    )
    torch.testing.assert_close(
        actual.out, expected.out, rtol=5.0e-3, atol=2.0e-2
    )


def _checkpoint_cases() -> list[object]:
    cases: list[object] = []
    for speculative_tokens in sorted(SUPPORTED_SPECULATIVE_TOKENS):
        sequence_length = speculative_tokens + 1
        accepted_values = {
            1,
            (sequence_length + 1) // 2,
            sequence_length,
        }
        for accepted in sorted(accepted_values):
            for same_slot in (False, True):
                ownership = "same" if same_slot else "fork"
                cases.append(
                    pytest.param(
                        speculative_tokens,
                        accepted,
                        same_slot,
                        id=(
                            f"K{speculative_tokens}-accepted{accepted}-"
                            f"{ownership}"
                        ),
                    )
                )
    return cases


def _qwen35_head_geometry_cases() -> list[object]:
    model_geometries = (
        ("0p8b-2b", 16, 16),
        ("4b-9b-35b", 16, 32),
        ("27b", 16, 48),
        ("122b-397b", 16, 64),
    )
    cases: list[object] = []
    for model_group, global_num_k_heads, global_num_v_heads in model_geometries:
        for tensor_parallel_size in (1, 2, 4, 8, 16):
            cases.append(
                pytest.param(
                    global_num_k_heads // tensor_parallel_size,
                    global_num_v_heads // tensor_parallel_size,
                    id=f"{model_group}-tp{tensor_parallel_size}",
                )
            )
    return cases


def test_identity_state_and_zero_conv(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
) -> None:
    inputs = _make_inputs(speculative_tokens=1, num_state_slots=1)
    for name in (
        "qkv",
        "z",
        "b",
        "a",
        "conv_weight",
        "conv_state",
    ):
        inputs[name].zero_()
    inputs["a_log"].fill_(-100.0)
    inputs["dt_bias"].zero_()
    inputs["ssm_state"].fill_(1.0)
    inputs["read_state_indices"].zero_()
    inputs["write_state_indices"].zero_()
    inputs["num_accepted_tokens"].fill_(1)
    inputs["norm_weight"].fill_(1.0)

    actual = aclnn_kernel.run(inputs, in_place=False)

    assert torch.count_nonzero(actual.conv_out) == 0
    assert torch.count_nonzero(actual.conv_state) == 0
    assert torch.count_nonzero(actual.out) == 0
    torch.testing.assert_close(
        actual.ssm_state,
        torch.ones_like(actual.ssm_state),
        rtol=0,
        atol=0,
    )


def test_decay_matches_small_op_bfloat16_rounding(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
) -> None:
    inputs = _make_inputs(speculative_tokens=1)
    for name in (
        "qkv",
        "z",
        "b",
        "conv_weight",
        "conv_state",
    ):
        inputs[name].zero_()
    inputs["a"].fill_(-0.75)
    inputs["a_log"].fill_(-0.3)
    inputs["dt_bias"].fill_(-0.2)
    inputs["ssm_state"].fill_(1.0)
    inputs["num_accepted_tokens"].fill_(2)

    actual = aclnn_kernel.run(inputs, in_place=False)

    unrounded_g = -torch.exp(inputs["a_log"]) * torch.nn.functional.softplus(
        inputs["a"][0, 0].float() + inputs["dt_bias"]
    )
    expected_decay = torch.exp(unrounded_g.to(torch.bfloat16).float())[0]
    unrounded_decay = torch.exp(unrounded_g)[0]
    actual_decay = actual.ssm_state[2, 0, 0, 0]
    torch.testing.assert_close(actual_decay, expected_decay, rtol=0, atol=0)
    assert abs(actual_decay - unrounded_decay).item() > 1.0e-5


def test_beta_matches_small_op_bfloat16_rounding(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
) -> None:
    inputs = _make_inputs(speculative_tokens=1)
    for name in (
        "qkv",
        "z",
        "a",
        "conv_weight",
        "conv_state",
        "ssm_state",
    ):
        inputs[name].zero_()
    inputs["qkv"][0, 0, HEAD_DIM] = 1.0
    inputs["qkv"][0, 0, 2 * HEAD_DIM] = 1.0
    inputs["conv_weight"][3].fill_(1.0)
    inputs["b"].fill_(-1.25)
    inputs["a_log"].fill_(-100.0)
    inputs["dt_bias"].zero_()
    inputs["num_accepted_tokens"].fill_(2)

    expected = mega_gdn_mtp_decode_reference(**inputs)
    actual = aclnn_kernel.run(inputs, in_place=False)

    rounded_beta = (
        torch.sigmoid(inputs["b"][0, 0].float())
        .to(torch.bfloat16)
        .float()[0]
    )
    unrounded_beta = torch.sigmoid(inputs["b"][0, 0].float())[0]
    expected_value = expected.ssm_state[2, 0, 0, 0]
    unrounded_value = expected_value * unrounded_beta / rounded_beta
    actual_value = actual.ssm_state[2, 0, 0, 0]
    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    assert abs(actual_value - unrounded_value).item() > 1.0e-5


@pytest.mark.parametrize(
    "speculative_tokens", sorted(SUPPORTED_SPECULATIVE_TOKENS)
)
@pytest.mark.parametrize("in_place", [False, True])
def test_matches_cpu_reference(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
    speculative_tokens: int,
    in_place: bool,
) -> None:
    inputs = _make_inputs(speculative_tokens)
    inputs["num_accepted_tokens"].fill_(
        (speculative_tokens + 2) // 2
    )
    expected = mega_gdn_mtp_decode_reference(**inputs)
    actual = aclnn_kernel.run(inputs, in_place=in_place)

    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize(
    "speculative_tokens", sorted(SUPPORTED_SPECULATIVE_TOKENS)
)
def test_matches_production_head_geometry(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
    speculative_tokens: int,
) -> None:
    inputs = _make_inputs(
        speculative_tokens,
        num_k_heads=8,
        num_v_heads=24,
    )
    inputs["num_accepted_tokens"].fill_(
        (speculative_tokens + 2) // 2
    )
    expected = mega_gdn_mtp_decode_reference(**inputs)
    actual = aclnn_kernel.run(inputs, in_place=True)

    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize("same_slot", [False, True])
def test_k8_b4_production_geometry(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
    same_slot: bool,
) -> None:
    batch_size = 4
    num_state_slots = batch_size if same_slot else batch_size + 1
    inputs = _make_inputs(
        speculative_tokens=8,
        num_state_slots=num_state_slots,
        num_k_heads=8,
        num_v_heads=24,
        batch_size=batch_size,
    )
    if same_slot:
        state_indices = torch.arange(batch_size, dtype=torch.int32)
        inputs["read_state_indices"] = state_indices
        inputs["write_state_indices"] = state_indices.clone()
    else:
        inputs["read_state_indices"].zero_()
        inputs["write_state_indices"] = torch.arange(
            1, batch_size + 1, dtype=torch.int32
        )
    inputs["num_accepted_tokens"] = torch.tensor(
        (1, 3, 6, 9), dtype=torch.int32
    )

    expected = mega_gdn_mtp_decode_reference(**inputs)
    actual = aclnn_kernel.run(inputs, in_place=True)

    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize(
    ("num_k_heads", "num_v_heads"), _qwen35_head_geometry_cases()
)
@pytest.mark.parametrize(
    "speculative_tokens", sorted(SUPPORTED_SPECULATIVE_TOKENS)
)
@pytest.mark.parametrize("same_slot", [False, True])
def test_qwen35_family_head_geometries(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
    num_k_heads: int,
    num_v_heads: int,
    speculative_tokens: int,
    same_slot: bool,
) -> None:
    inputs = _make_inputs(
        speculative_tokens,
        num_state_slots=1 if same_slot else 2,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
    )
    inputs["read_state_indices"].zero_()
    inputs["write_state_indices"].fill_(0 if same_slot else 1)
    inputs["num_accepted_tokens"].fill_(
        (speculative_tokens + 2) // 2
    )

    expected = mega_gdn_mtp_decode_reference(**inputs)
    actual = aclnn_kernel.run(inputs, in_place=True)

    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize("state_slot", range(6))
def test_k8_qwen35_27b_tp4_nonzero_same_slot(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
    state_slot: int,
) -> None:
    inputs = _make_inputs(
        speculative_tokens=8,
        num_state_slots=6,
        num_k_heads=4,
        num_v_heads=12,
    )
    inputs["read_state_indices"].fill_(state_slot)
    inputs["write_state_indices"].fill_(state_slot)
    inputs["num_accepted_tokens"].fill_(5)

    expected = mega_gdn_mtp_decode_reference(**inputs)
    actual = aclnn_kernel.run(inputs, in_place=True)

    _assert_matches_reference(actual, expected)


def test_k8_qwen35_27b_tp4_persistent_state_and_request_reset(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
) -> None:
    inputs = _make_inputs(
        speculative_tokens=8,
        num_state_slots=6,
        num_k_heads=4,
        num_v_heads=12,
    )
    state_slot = 5
    inputs["read_state_indices"].fill_(state_slot)
    inputs["write_state_indices"].fill_(state_slot)
    initial_conv_state = inputs["conv_state"].clone()
    initial_ssm_state = inputs["ssm_state"].clone()
    device_inputs = {
        name: tensor.to("npu") for name, tensor in inputs.items()
    }

    for accepted in (5, 3, 9):
        inputs["num_accepted_tokens"].fill_(accepted)
        device_inputs["num_accepted_tokens"].fill_(accepted)
        expected = mega_gdn_mtp_decode_reference(**inputs)
        actual = aclnn_kernel.run_device(device_inputs, in_place=True)
        _assert_matches_reference(actual, expected)
        inputs["conv_state"] = expected.conv_state
        inputs["ssm_state"] = expected.ssm_state

    inputs["conv_state"] = initial_conv_state
    inputs["ssm_state"] = initial_ssm_state
    inputs["num_accepted_tokens"].fill_(5)
    device_inputs["conv_state"].copy_(initial_conv_state.to("npu"))
    device_inputs["ssm_state"].copy_(initial_ssm_state.to("npu"))
    device_inputs["num_accepted_tokens"].fill_(5)
    aclnn_kernel._torch_npu.npu.synchronize()

    expected = mega_gdn_mtp_decode_reference(**inputs)
    actual = aclnn_kernel.run_device(device_inputs, in_place=True)
    _assert_matches_reference(actual, expected)


def test_k2_qwen35_27b_tp4_persistent_checkpoint_zero_is_finite(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
) -> None:
    inputs = _make_inputs(
        speculative_tokens=2,
        num_state_slots=1,
        num_k_heads=4,
        num_v_heads=12,
    )
    inputs["read_state_indices"].zero_()
    inputs["write_state_indices"].zero_()
    inputs["num_accepted_tokens"].fill_(1)
    device_inputs = {
        name: tensor.to("npu") for name, tensor in inputs.items()
    }

    for round_index in range(64):
        if round_index > 0:
            for name in ("qkv", "z", "a", "b"):
                inputs[name].neg_()
                device_inputs[name].neg_()

        expected = mega_gdn_mtp_decode_reference(**inputs)
        actual = aclnn_kernel.run_device(device_inputs, in_place=True)

        for output_name in ("conv_out", "conv_state", "ssm_state", "out"):
            output = getattr(actual, output_name)
            assert torch.isfinite(output).all(), (
                f"{output_name} became non-finite at round {round_index}"
            )
        _assert_matches_reference(actual, expected)
        inputs["conv_state"] = expected.conv_state
        inputs["ssm_state"] = expected.ssm_state


def test_k2_qwen35_27b_tp4_checkpoint_zero_without_intermediate_sync(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
) -> None:
    launches = 64
    inputs = _make_inputs(
        speculative_tokens=2,
        num_state_slots=1,
        num_k_heads=4,
        num_v_heads=12,
    )
    inputs["read_state_indices"].zero_()
    inputs["write_state_indices"].zero_()
    inputs["num_accepted_tokens"].fill_(1)
    device_inputs = {
        name: tensor.to("npu") for name, tensor in inputs.items()
    }

    for _ in range(launches):
        expected = mega_gdn_mtp_decode_reference(**inputs)
        inputs["conv_state"] = expected.conv_state
        inputs["ssm_state"] = expected.ssm_state

    actual = aclnn_kernel.run_device_many(
        device_inputs, in_place=True, launches=launches
    )

    for output_name in ("conv_out", "conv_state", "ssm_state", "out"):
        output = getattr(actual, output_name)
        assert torch.isfinite(output).all(), (
            f"{output_name} became non-finite after {launches} launches"
        )
    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize(
    ("num_k_heads", "num_v_heads"),
    [
        pytest.param(4, 12, id="key108-qwen35-27b-tp4"),
        pytest.param(8, 24, id="key208-qwen35-27b-tp2"),
    ],
)
def test_k8_future_tokens_do_not_change_accepted_checkpoint(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
    num_k_heads: int,
    num_v_heads: int,
) -> None:
    speculative_tokens = 8
    sequence_length = speculative_tokens + 1
    batch_size = 4
    first_a = _make_inputs(
        speculative_tokens,
        num_state_slots=batch_size,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        batch_size=batch_size,
    )
    first_b = {
        name: tensor.clone() for name, tensor in first_a.items()
    }
    state_indices = torch.arange(batch_size, dtype=torch.int32)
    for inputs in (first_a, first_b):
        inputs["read_state_indices"] = state_indices.clone()
        inputs["write_state_indices"] = state_indices.clone()
        inputs["num_accepted_tokens"] = torch.tensor(
            (1, 3, 6, 9), dtype=torch.int32
        )
    for name in ("qkv", "z", "a", "b"):
        first_b[name][:, 1:].neg_()

    first_result_a = aclnn_kernel.run(first_a, in_place=True)
    first_result_b = aclnn_kernel.run(first_b, in_place=True)

    torch.testing.assert_close(
        first_result_a.conv_out[:, 0],
        first_result_b.conv_out[:, 0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first_result_a.out[:, 0],
        first_result_b.out[:, 0],
        rtol=0,
        atol=0,
    )
    for state_slot in range(batch_size):
        checkpoint = state_slot * sequence_length
        torch.testing.assert_close(
            first_result_a.ssm_state[checkpoint],
            first_result_b.ssm_state[checkpoint],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            first_result_a.conv_state[state_slot, :3],
            first_result_b.conv_state[state_slot, :3],
            rtol=0,
            atol=0,
        )

    second_a = _make_inputs(
        speculative_tokens,
        num_state_slots=batch_size,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        batch_size=batch_size,
    )
    second_b = {
        name: tensor.clone() for name, tensor in second_a.items()
    }
    second_a["conv_state"] = first_result_a.conv_state
    second_a["ssm_state"] = first_result_a.ssm_state
    second_b["conv_state"] = first_result_b.conv_state
    second_b["ssm_state"] = first_result_b.ssm_state
    for inputs in (second_a, second_b):
        inputs["read_state_indices"] = state_indices.clone()
        inputs["write_state_indices"] = state_indices.clone()
        inputs["num_accepted_tokens"].fill_(1)

    second_result_a = aclnn_kernel.run(second_a, in_place=True)
    second_result_b = aclnn_kernel.run(second_b, in_place=True)
    for output_name in ("conv_out", "conv_state", "ssm_state", "out"):
        torch.testing.assert_close(
            getattr(second_result_a, output_name),
            getattr(second_result_b, output_name),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("speculative_tokens", "accepted", "same_slot"),
    _checkpoint_cases(),
)
def test_checkpoint_and_prefix_fork_semantics(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
    speculative_tokens: int,
    accepted: int,
    same_slot: bool,
) -> None:
    num_state_slots = 1 if same_slot else 2
    inputs = _make_inputs(speculative_tokens, num_state_slots)
    inputs["read_state_indices"].zero_()
    inputs["write_state_indices"].fill_(0 if same_slot else 1)
    inputs["num_accepted_tokens"].fill_(accepted)
    expected = mega_gdn_mtp_decode_reference(**inputs)
    actual = aclnn_kernel.run(inputs, in_place=True)

    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize(
    "speculative_tokens", sorted(SUPPORTED_SPECULATIVE_TOKENS)
)
@pytest.mark.parametrize("same_slot", [False, True])
def test_k1_to_k16_is_deterministic(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
    speculative_tokens: int,
    same_slot: bool,
) -> None:
    num_state_slots = 1 if same_slot else 2
    first_inputs = _make_inputs(speculative_tokens, num_state_slots)
    second_inputs = _make_inputs(speculative_tokens, num_state_slots)
    for inputs in (first_inputs, second_inputs):
        inputs["read_state_indices"].zero_()
        inputs["write_state_indices"].fill_(0 if same_slot else 1)
        inputs["num_accepted_tokens"].fill_(
            (speculative_tokens + 2) // 2
        )

    first = aclnn_kernel.run(first_inputs, in_place=True)
    second = aclnn_kernel.run(second_inputs, in_place=True)

    for output_name in ("conv_out", "conv_state", "ssm_state", "out"):
        torch.testing.assert_close(
            getattr(first, output_name),
            getattr(second, output_name),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("batch_size", "same_slot", "accepted_values"),
    [
        pytest.param(4, True, (1, 3, 6, 9), id="B4-same"),
        pytest.param(4, False, (1, 3, 6, 9), id="B4-prefix-fork"),
        pytest.param(
            8,
            True,
            (1, 2, 3, 4, 6, 7, 8, 9),
            id="B8-same",
        ),
        pytest.param(
            8,
            False,
            (1, 2, 3, 4, 6, 7, 8, 9),
            id="B8-prefix-fork",
        ),
    ],
)
def test_batched_checkpoint_and_prefix_fork_semantics(
    aclnn_kernel: _AclnnMegaGdnMtpDecode,
    batch_size: int,
    same_slot: bool,
    accepted_values: tuple[int, ...],
) -> None:
    num_state_slots = batch_size if same_slot else batch_size + 1
    inputs = _make_inputs(
        speculative_tokens=8,
        num_state_slots=num_state_slots,
        batch_size=batch_size,
    )
    if same_slot:
        state_indices = torch.arange(batch_size, dtype=torch.int32)
        inputs["read_state_indices"] = state_indices
        inputs["write_state_indices"] = state_indices.clone()
    else:
        inputs["read_state_indices"].zero_()
        inputs["write_state_indices"] = torch.arange(
            1, batch_size + 1, dtype=torch.int32
        )
    inputs["num_accepted_tokens"] = torch.tensor(
        accepted_values, dtype=torch.int32
    )
    expected = mega_gdn_mtp_decode_reference(**inputs)
    actual = aclnn_kernel.run(inputs, in_place=True)

    _assert_matches_reference(actual, expected)
