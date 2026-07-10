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


DEFAULT_NUM_HEADS = 24
DEFAULT_HEAD_K_DIM = 128
DEFAULT_HEAD_V_DIM = 128
COPY_CHUNK = 1024
VEC_NUM = 2
MAX_VEC_CORE_NUM = detect_vec_core_num()


def build_final_state_cache_store_kernel(
    *,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
):
    state_elems = num_heads * head_k_dim * head_v_dim
    if state_elems <= 0 or state_elems % COPY_CHUNK != 0:
        raise ValueError(
            f"state elements({state_elems}) must be a positive multiple of "
            f"{COPY_CHUNK}"
        )
    if MAX_VEC_CORE_NUM <= 0 or MAX_VEC_CORE_NUM % VEC_NUM != 0:
        raise ValueError(
            f"vector core count({MAX_VEC_CORE_NUM}) must be positive and even"
        )

    cubecore_block_num = MAX_VEC_CORE_NUM // VEC_NUM
    task_num = cubecore_block_num * VEC_NUM
    num_chunks = state_elems // COPY_CHUNK

    @T.prim_func
    def final_state_cache_store_kernel(
        final_state: T.Tensor((state_elems,), "float16"),
        cache_slot: T.Tensor((state_elems,), "float32"),
    ):
        with T.Kernel(cubecore_block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            chunks_per_core = (num_chunks + task_num - 1) // task_num
            chunk_start = task_id * chunks_per_core
            chunks_left = T.if_then_else(
                num_chunks > chunk_start,
                num_chunks - chunk_start,
                0,
            )
            active_chunks = T.if_then_else(
                chunks_left < chunks_per_core,
                chunks_left,
                chunks_per_core,
            )

            with T.Scope("V"):
                state_half = T.alloc_shared((COPY_CHUNK,), "float16")
                state_float = T.alloc_shared((COPY_CHUNK,), "float32")

                for local_chunk in T.serial(active_chunks):
                    offset = (chunk_start + local_chunk) * COPY_CHUNK
                    T.copy(final_state[offset], state_half)
                    T.tile.cast(
                        state_float,
                        state_half,
                        "CAST_NONE",
                        COPY_CHUNK,
                    )
                    T.copy(state_float, cache_slot[offset])

    return final_state_cache_store_kernel


@tilelang.jit(pass_configs=DEFAULT_ASCEND_PASS_CONFIGS)
def final_state_cache_store_kernel_jit(
    num_heads: int = DEFAULT_NUM_HEADS,
    head_k_dim: int = DEFAULT_HEAD_K_DIM,
    head_v_dim: int = DEFAULT_HEAD_V_DIM,
):
    return build_final_state_cache_store_kernel(
        num_heads=num_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )


@register_kernel
class FinalStateCacheStoreKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("num_heads", "int32"),
        DispatchField("head_k_dim", "int32"),
        DispatchField("head_v_dim", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "h24_dk128_dv128",
            "num_heads": DEFAULT_NUM_HEADS,
            "head_k_dim": DEFAULT_HEAD_K_DIM,
            "head_v_dim": DEFAULT_HEAD_V_DIM,
        }
    ]

    @staticmethod
    def generate_source(
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
    ) -> str:
        tilelang.disable_cache()
        tilelang_kernel = build_final_state_cache_store_kernel(
            num_heads=num_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def _run_ref_check(
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> None:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logger.warning(
            "Skip final_state_cache_store reference check: NPU is not available"
        )
        return

    torch.manual_seed(47)
    device = torch.device("npu")
    shape = (num_heads, head_k_dim, head_v_dim)
    final_state = torch.randn(shape, device=device, dtype=torch.float16)
    cache = torch.zeros((3, *shape), device=device, dtype=torch.float32)
    cache_slot = cache[1]

    kernel = final_state_cache_store_kernel_jit(
        num_heads=num_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )
    kernel(final_state.view(-1), cache_slot.view(-1))
    torch.npu.synchronize()

    if not torch.equal(cache_slot, final_state.to(torch.float32)):
        raise AssertionError("cache slot does not exactly match FP32 reference")
    if torch.count_nonzero(cache[0]).item() != 0:
        raise AssertionError("cache slot before target was modified")
    if torch.count_nonzero(cache[2]).item() != 0:
        raise AssertionError("cache slot after target was modified")
    logger.info(
        "final_state_cache_store exactly matches reference for "
        f"shape={shape}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--head-k-dim", type=int, default=DEFAULT_HEAD_K_DIM)
    parser.add_argument("--head-v-dim", type=int, default=DEFAULT_HEAD_V_DIM)
    parser.add_argument("--ref-check", action="store_true")
    parser.add_argument("--dump-source", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kernel = final_state_cache_store_kernel_jit(
        num_heads=args.num_heads,
        head_k_dim=args.head_k_dim,
        head_v_dim=args.head_v_dim,
    )
    if args.dump_source is not None:
        args.dump_source.write_text(kernel.get_kernel_source(), encoding="utf-8")
    if args.ref_check:
        _run_ref_check(args.num_heads, args.head_k_dim, args.head_v_dim)


if __name__ == "__main__":
    main()
