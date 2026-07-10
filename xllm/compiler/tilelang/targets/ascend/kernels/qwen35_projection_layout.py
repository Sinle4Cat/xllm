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


DEFAULT_DTYPE = "bf16"
DEFAULT_QKV_SIZE = 5120
DEFAULT_Z_SIZE = 3072
DEFAULT_NUM_HEADS = 24
DEFAULT_MAX_ROWS = 64
COPY_CHUNK = 512
VEC_NUM = 2
MAX_VEC_CORE_NUM = detect_vec_core_num()


def build_qwen35_projection_layout_kernel(
    *,
    qkv_size: int,
    z_size: int,
    num_heads: int,
    compile_max_rows: int,
):
    if qkv_size <= 0 or qkv_size % COPY_CHUNK != 0:
        raise ValueError(
            f"qkv_size({qkv_size}) must be a positive multiple of {COPY_CHUNK}"
        )
    if z_size <= 0 or z_size % COPY_CHUNK != 0:
        raise ValueError(
            f"z_size({z_size}) must be a positive multiple of {COPY_CHUNK}"
        )
    if num_heads <= 0:
        raise ValueError(f"num_heads({num_heads}) must be positive")
    if compile_max_rows <= 0:
        raise ValueError(f"compile_max_rows({compile_max_rows}) must be positive")
    if MAX_VEC_CORE_NUM <= 0 or MAX_VEC_CORE_NUM % VEC_NUM != 0:
        raise ValueError(
            f"vector core count({MAX_VEC_CORE_NUM}) must be positive and even"
        )

    dtype = "bfloat16"
    projection_size = qkv_size + z_size + 2 * num_heads
    qkv_chunks = qkv_size // COPY_CHUNK
    z_chunks = z_size // COPY_CHUNK
    tasks_per_row = qkv_chunks + z_chunks + 2
    cubecore_block_num = MAX_VEC_CORE_NUM // VEC_NUM
    task_num = cubecore_block_num * VEC_NUM

    @T.prim_func
    def qwen35_projection_layout_kernel(
        projection: T.Tensor((compile_max_rows, projection_size), dtype),
        qkv: T.Tensor((compile_max_rows, qkv_size), dtype),
        z: T.Tensor((compile_max_rows, z_size), dtype),
        b: T.Tensor((compile_max_rows, num_heads), dtype),
        a: T.Tensor((compile_max_rows, num_heads), dtype),
        num_rows: T.int32,
    ):
        with T.Kernel(cubecore_block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            total_tasks = num_rows * tasks_per_row
            tasks_per_core = (total_tasks + task_num - 1) // task_num
            task_start = task_id * tasks_per_core
            tasks_left = T.if_then_else(
                total_tasks > task_start,
                total_tasks - task_start,
                0,
            )
            active_tasks = T.if_then_else(
                tasks_left < tasks_per_core,
                tasks_left,
                tasks_per_core,
            )

            with T.Scope("V"):
                chunk_ub = T.alloc_shared((COPY_CHUNK,), dtype)
                heads_ub = T.alloc_shared((num_heads,), dtype)

                for local_task in T.serial(active_tasks):
                    linear_task = task_start + local_task
                    row = linear_task // tasks_per_row
                    segment = linear_task % tasks_per_row

                    with T.If(segment < qkv_chunks):
                        with T.Then():
                            offset = segment * COPY_CHUNK
                            T.copy(projection[row, offset], chunk_ub)
                            T.copy(chunk_ub, qkv[row, offset])

                    with T.If(segment >= qkv_chunks):
                        with T.Then():
                            with T.If(segment < qkv_chunks + z_chunks):
                                with T.Then():
                                    z_segment = segment - qkv_chunks
                                    z_offset = z_segment * COPY_CHUNK
                                    T.copy(
                                        projection[row, qkv_size + z_offset],
                                        chunk_ub,
                                    )
                                    T.copy(chunk_ub, z[row, z_offset])

                    with T.If(segment == qkv_chunks + z_chunks):
                        with T.Then():
                            T.copy(
                                projection[row, qkv_size + z_size],
                                heads_ub,
                            )
                            T.copy(heads_ub, b[row, 0])

                    with T.If(segment == qkv_chunks + z_chunks + 1):
                        with T.Then():
                            T.copy(
                                projection[row, qkv_size + z_size + num_heads],
                                heads_ub,
                            )
                            T.copy(heads_ub, a[row, 0])

    return qwen35_projection_layout_kernel


@tilelang.jit(pass_configs=DEFAULT_ASCEND_PASS_CONFIGS)
def qwen35_projection_layout_kernel_jit(
    num_rows: int,
    qkv_size: int = DEFAULT_QKV_SIZE,
    z_size: int = DEFAULT_Z_SIZE,
    num_heads: int = DEFAULT_NUM_HEADS,
):
    return build_qwen35_projection_layout_kernel(
        qkv_size=qkv_size,
        z_size=z_size,
        num_heads=num_heads,
        compile_max_rows=num_rows,
    )


@register_kernel
class Qwen35ProjectionLayoutKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("qkv_size", "int32"),
        DispatchField("z_size", "int32"),
        DispatchField("num_heads", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "qkv5120_z3072_h24_bf16",
            "qkv_size": DEFAULT_QKV_SIZE,
            "z_size": DEFAULT_Z_SIZE,
            "num_heads": DEFAULT_NUM_HEADS,
            "dtype": DEFAULT_DTYPE,
        }
    ]

    @staticmethod
    def generate_source(
        qkv_size: int,
        z_size: int,
        num_heads: int,
        dtype: str,
    ) -> str:
        if dtype != DEFAULT_DTYPE:
            raise ValueError(
                f"qwen35_projection_layout only supports dtype={DEFAULT_DTYPE}, "
                f"got {dtype}"
            )
        tilelang.disable_cache()
        tilelang_kernel = build_qwen35_projection_layout_kernel(
            qkv_size=qkv_size,
            z_size=z_size,
            num_heads=num_heads,
            compile_max_rows=DEFAULT_MAX_ROWS,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def _run_ref_check(
    num_rows: int,
    qkv_size: int,
    z_size: int,
    num_heads: int,
) -> None:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logger.warning(
            "Skip qwen35_projection_layout reference check: NPU is not available"
        )
        return

    torch.manual_seed(31)
    device = torch.device("npu")
    projection_size = qkv_size + z_size + 2 * num_heads
    projection = torch.randn(
        (num_rows, projection_size), device=device, dtype=torch.bfloat16
    )
    qkv = torch.empty((num_rows, qkv_size), device=device, dtype=torch.bfloat16)
    z = torch.empty((num_rows, z_size), device=device, dtype=torch.bfloat16)
    b = torch.empty((num_rows, num_heads), device=device, dtype=torch.bfloat16)
    a = torch.empty_like(b)

    kernel = qwen35_projection_layout_kernel_jit(
        num_rows=num_rows,
        qkv_size=qkv_size,
        z_size=z_size,
        num_heads=num_heads,
    )
    kernel(projection, qkv, z, b, a, num_rows)
    torch.npu.synchronize()

    expected = torch.split(
        projection,
        [qkv_size, z_size, num_heads, num_heads],
        dim=-1,
    )
    for name, output, reference in zip(
        ("qkv", "z", "b", "a"),
        (qkv, z, b, a),
        expected,
        strict=True,
    ):
        if not torch.equal(output, reference):
            raise AssertionError(f"{name} output does not exactly match reference")
    logger.info(
        "qwen35_projection_layout outputs exactly match torch reference for "
        f"shape=({num_rows}, {projection_size})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TileLang AscendC source for qwen35 projection layout."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qkv-size", type=int, default=DEFAULT_QKV_SIZE)
    parser.add_argument("--z-size", type=int, default=DEFAULT_Z_SIZE)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument("--skip-ref-check", action="store_true")
    parser.add_argument("--ref-num-rows", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Qwen35ProjectionLayoutKernel.generate_source(
        qkv_size=args.qkv_size,
        z_size=args.z_size,
        num_heads=args.num_heads,
        dtype=args.dtype,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")

    if not args.skip_ref_check:
        _run_ref_check(
            num_rows=args.ref_num_rows,
            qkv_size=args.qkv_size,
            z_size=args.z_size,
            num_heads=args.num_heads,
        )


if __name__ == "__main__":
    main()
