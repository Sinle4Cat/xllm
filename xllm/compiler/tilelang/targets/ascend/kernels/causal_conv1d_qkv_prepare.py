#!/usr/bin/env python3

# Copyright 2025-2026 The xLLM Authors.
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

import argparse
import importlib.util
from pathlib import Path

import tilelang
import tilelang.language as T

from compiler.tilelang.common.spec import (
    DispatchField,
    TilelangKernel,
    register_kernel,
)
from compiler.tilelang.targets.ascend.kernels.utils import (
    DEFAULT_ASCEND_PASS_CONFIGS,
    detect_vec_core_num,
)
from scripts.logger import logger


DEFAULT_NUM_K_HEADS = 8
DEFAULT_NUM_V_HEADS = 24
DEFAULT_HEAD_DIM = 128
DEFAULT_MAX_TOKENS = 8192
DEFAULT_EPS = 1e-6
VEC_NUM = 2
MAX_VEC_CORE_NUM = detect_vec_core_num()


def build_causal_conv1d_qkv_prepare_kernel(
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    compile_max_tokens: int,
):
    if num_k_heads <= 0 or num_v_heads <= 0 or head_dim <= 0:
        raise ValueError("head counts and head_dim must be positive")
    if compile_max_tokens <= 0:
        raise ValueError("compile_max_tokens must be positive")
    if MAX_VEC_CORE_NUM <= 0 or MAX_VEC_CORE_NUM % VEC_NUM != 0:
        raise ValueError(
            f"vector core count({MAX_VEC_CORE_NUM}) must be positive and even"
        )

    total_heads = 2 * num_k_heads + num_v_heads
    qk_heads = 2 * num_k_heads
    qk_elems = qk_heads * head_dim
    v_elems = num_v_heads * head_dim
    cubecore_block_num = MAX_VEC_CORE_NUM // VEC_NUM
    task_num = cubecore_block_num * VEC_NUM

    @T.prim_func
    def causal_conv1d_qkv_prepare_kernel(
        mixed_qkv: T.Tensor(
            (compile_max_tokens, total_heads, head_dim), "bfloat16"
        ),
        q_out: T.Tensor(
            (compile_max_tokens, num_k_heads, head_dim), "float16"
        ),
        k_out: T.Tensor(
            (compile_max_tokens, num_k_heads, head_dim), "float16"
        ),
        v_out: T.Tensor(
            (compile_max_tokens, num_v_heads, head_dim), "float16"
        ),
        num_tokens: T.int32,
        eps: T.float32,
    ):
        with T.Kernel(cubecore_block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            tokens_per_core = (num_tokens + task_num - 1) // task_num
            token_start = task_id * tokens_per_core
            tokens_left = T.if_then_else(
                num_tokens > token_start,
                num_tokens - token_start,
                0,
            )
            active_tokens = T.if_then_else(
                tokens_left < tokens_per_core,
                tokens_left,
                tokens_per_core,
            )

            with T.Scope("V"):
                heads_bf16 = T.alloc_shared(
                    (qk_heads, head_dim), "bfloat16"
                )
                heads_fp16 = T.alloc_shared(
                    (qk_heads, head_dim), "float16"
                )
                heads_fp32 = T.alloc_shared(
                    (qk_heads, head_dim), "float32"
                )
                square_fp32 = T.alloc_shared(
                    (qk_heads, head_dim), "float32"
                )
                inv_norm = T.alloc_shared((qk_heads, 1), "float32")
                inv_norm_2d = T.alloc_shared(
                    (qk_heads, head_dim), "float32"
                )
                value_bf16 = T.alloc_shared(
                    (num_v_heads, head_dim), "bfloat16"
                )
                value_fp32 = T.alloc_shared(
                    (num_v_heads, head_dim), "float32"
                )
                value_fp16 = T.alloc_shared(
                    (num_v_heads, head_dim), "float16"
                )

                for local_token in T.serial(active_tokens):
                    token = token_start + local_token

                    T.copy(mixed_qkv[token, 0, 0], heads_bf16)
                    T.tile.cast(
                        heads_fp32, heads_bf16, "CAST_NONE", qk_elems
                    )
                    T.tile.mul(square_fp32, heads_fp32, heads_fp32)
                    T.reduce_sum(square_fp32, inv_norm, dim=-1)
                    T.tile.add(inv_norm, inv_norm, eps)
                    T.tile.sqrt(inv_norm, inv_norm)
                    T.tile.broadcast(inv_norm_2d, inv_norm)
                    T.tile.div(heads_fp32, heads_fp32, inv_norm_2d)
                    T.tile.cast(
                        heads_bf16, heads_fp32, "CAST_RINT", qk_elems
                    )
                    T.tile.cast(
                        heads_fp32, heads_bf16, "CAST_NONE", qk_elems
                    )
                    T.tile.cast(
                        heads_fp16, heads_fp32, "CAST_RINT", qk_elems
                    )
                    T.copy(
                        heads_fp16[0:num_k_heads, 0:head_dim],
                        q_out[token, 0:num_k_heads, 0:head_dim],
                    )
                    T.copy(
                        heads_fp16[
                            num_k_heads:qk_heads, 0:head_dim
                        ],
                        k_out[token, 0:num_k_heads, 0:head_dim],
                    )

                    T.copy(
                        mixed_qkv[token, 2 * num_k_heads, 0], value_bf16
                    )
                    T.tile.cast(
                        value_fp32, value_bf16, "CAST_NONE", v_elems
                    )
                    T.tile.cast(
                        value_fp16, value_fp32, "CAST_RINT", v_elems
                    )
                    T.copy(value_fp16, v_out[token, 0, 0])

    return causal_conv1d_qkv_prepare_kernel


@tilelang.jit(pass_configs=DEFAULT_ASCEND_PASS_CONFIGS)
def causal_conv1d_qkv_prepare_kernel_jit(
    num_tokens: int,
    num_k_heads: int = DEFAULT_NUM_K_HEADS,
    num_v_heads: int = DEFAULT_NUM_V_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
):
    return build_causal_conv1d_qkv_prepare_kernel(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim=head_dim,
        compile_max_tokens=num_tokens,
    )


@register_kernel
class CausalConv1dQkvPrepareKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("num_k_heads", "int32"),
        DispatchField("num_v_heads", "int32"),
        DispatchField("head_dim", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "hk8_hv24_d128",
            "num_k_heads": DEFAULT_NUM_K_HEADS,
            "num_v_heads": DEFAULT_NUM_V_HEADS,
            "head_dim": DEFAULT_HEAD_DIM,
        }
    ]

    @staticmethod
    def generate_source(
        num_k_heads: int,
        num_v_heads: int,
        head_dim: int,
    ) -> str:
        tilelang.disable_cache()
        tilelang_kernel = build_causal_conv1d_qkv_prepare_kernel(
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_dim=head_dim,
            compile_max_tokens=DEFAULT_MAX_TOKENS,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def _normalize_ref(x):
    import torch

    x_fp32 = x.to(torch.float32)
    norm = torch.rsqrt(
        torch.sum(x_fp32 * x_fp32, dim=-1, keepdim=True) + DEFAULT_EPS
    )
    return (x_fp32 * norm).to(torch.bfloat16).to(torch.float16)


def _production_l2norm_ref(x):
    import torch

    module_path = (
        Path(__file__).resolve().parents[6]
        / "third_party/torch_npu_ops/triton_npu/triton_src/test_l2norm.py"
    )
    spec = importlib.util.spec_from_file_location("xllm_test_l2norm", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load production L2Norm reference: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.l2norm_fwd(x.contiguous()).to(torch.float16)


def _run_ref_check(
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
) -> None:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logger.warning(
            "Skip causal_conv1d_qkv_prepare reference check: NPU unavailable"
        )
        return

    torch.manual_seed(59)
    device = torch.device("npu")
    total_heads = 2 * num_k_heads + num_v_heads
    mixed_qkv = torch.randn(
        (num_tokens, total_heads, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    q_out = torch.empty(
        (num_tokens, num_k_heads, head_dim),
        device=device,
        dtype=torch.float16,
    )
    k_out = torch.empty_like(q_out)
    v_out = torch.empty(
        (num_tokens, num_v_heads, head_dim),
        device=device,
        dtype=torch.float16,
    )
    kernel = causal_conv1d_qkv_prepare_kernel_jit(
        num_tokens=num_tokens,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim=head_dim,
    )
    kernel(mixed_qkv, q_out, k_out, v_out, num_tokens, DEFAULT_EPS)
    torch.npu.synchronize()

    q_input = mixed_qkv[:, :num_k_heads]
    k_input = mixed_qkv[:, num_k_heads : 2 * num_k_heads]
    q_torch_ref = _normalize_ref(q_input)
    k_torch_ref = _normalize_ref(k_input)
    q_ref = _production_l2norm_ref(q_input)
    k_ref = _production_l2norm_ref(k_input)
    v_ref = mixed_qkv[:, 2 * num_k_heads :].to(torch.float16)
    for name, output, reference in (
        ("q", q_out, q_ref),
        ("k", k_out, k_ref),
        ("v", v_out, v_ref),
    ):
        max_abs = (output - reference).abs().max().item()
        mismatch = torch.count_nonzero(output != reference).item()
        logger.info(
            f"{name}: max_abs={max_abs}, mismatch={mismatch}/{output.numel()}"
        )
        if not torch.allclose(output, reference, atol=2e-3, rtol=2e-3):
            raise AssertionError(f"{name} output does not match reference")
    for name, production, torch_ref in (
        ("q", q_ref, q_torch_ref),
        ("k", k_ref, k_torch_ref),
    ):
        logger.info(
            f"production-vs-torch {name}: max_abs="
            f"{(production - torch_ref).abs().max().item()}, mismatch="
            f"{torch.count_nonzero(production != torch_ref).item()}/{production.numel()}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=2048)
    parser.add_argument("--num-k-heads", type=int, default=DEFAULT_NUM_K_HEADS)
    parser.add_argument("--num-v-heads", type=int, default=DEFAULT_NUM_V_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--ref-check", action="store_true")
    parser.add_argument("--dump-source", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kernel = causal_conv1d_qkv_prepare_kernel_jit(
        num_tokens=args.num_tokens,
        num_k_heads=args.num_k_heads,
        num_v_heads=args.num_v_heads,
        head_dim=args.head_dim,
    )
    if args.dump_source is not None:
        args.dump_source.write_text(kernel.get_kernel_source(), encoding="utf-8")
    if args.ref_check:
        _run_ref_check(
            args.num_tokens,
            args.num_k_heads,
            args.num_v_heads,
            args.head_dim,
        )


if __name__ == "__main__":
    main()
