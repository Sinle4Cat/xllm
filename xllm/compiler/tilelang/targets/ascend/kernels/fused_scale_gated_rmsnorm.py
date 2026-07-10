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
DEFAULT_HEAD_SIZE = 128
DEFAULT_MAX_ROWS = 262144
ROWS_PER_ITER = 8
VEC_NUM = 2
MAX_VEC_CORE_NUM = detect_vec_core_num()
SUPPORTED_HEAD_SIZES = (DEFAULT_HEAD_SIZE,)


def build_fused_scale_gated_rmsnorm_kernel(
    *,
    head_size: int,
    compile_max_rows: int,
):
    if head_size not in SUPPORTED_HEAD_SIZES:
        raise ValueError(
            "fused_scale_gated_rmsnorm only supports head_size in "
            f"{SUPPORTED_HEAD_SIZES}, got {head_size}"
        )
    if compile_max_rows <= 0:
        raise ValueError(f"compile_max_rows({compile_max_rows}) must be > 0")
    if MAX_VEC_CORE_NUM <= 0 or MAX_VEC_CORE_NUM % VEC_NUM != 0:
        raise ValueError(
            f"vector core count({MAX_VEC_CORE_NUM}) must be positive and even"
        )

    input_dtype = "float16"
    output_dtype = "bfloat16"
    acc_dtype = "float32"
    cubecore_block_num = MAX_VEC_CORE_NUM // VEC_NUM
    task_num = cubecore_block_num * VEC_NUM
    tile_elements = ROWS_PER_ITER * head_size

    @T.prim_func
    def fused_scale_gated_rmsnorm_kernel(
        x: T.Tensor((compile_max_rows, head_size), input_dtype),
        gate: T.Tensor((compile_max_rows, head_size), output_dtype),
        weight: T.Tensor((head_size,), output_dtype),
        output: T.Tensor((compile_max_rows, head_size), output_dtype),
        num_rows: T.int32,
        eps: T.float32,
        scale: T.float32,
    ):
        with T.Kernel(cubecore_block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            total_chunks = (num_rows + ROWS_PER_ITER - 1) // ROWS_PER_ITER
            chunks_per_task = (total_chunks + task_num - 1) // task_num
            chunk_start = task_id * chunks_per_task
            chunks_left = T.if_then_else(
                total_chunks > chunk_start,
                total_chunks - chunk_start,
                0,
            )
            num_chunks = T.if_then_else(
                chunks_left < chunks_per_task,
                chunks_left,
                chunks_per_task,
            )

            with T.Scope("V"):
                weight_bf16_ub = T.alloc_shared((1, head_size), output_dtype)
                weight_base_fp32_ub = T.alloc_shared((1, head_size), acc_dtype)
                weight_fp32_ub = T.alloc_shared(
                    (ROWS_PER_ITER, head_size), acc_dtype
                )
                x_fp16_ub = T.alloc_shared(
                    (ROWS_PER_ITER, head_size), input_dtype
                )
                gate_bf16_ub = T.alloc_shared(
                    (ROWS_PER_ITER, head_size), output_dtype
                )
                output_bf16_ub = T.alloc_shared(
                    (ROWS_PER_ITER, head_size), output_dtype
                )
                x_fp32_ub = T.alloc_shared(
                    (ROWS_PER_ITER, head_size), acc_dtype
                )
                gate_fp32_ub = T.alloc_shared(
                    (ROWS_PER_ITER, head_size), acc_dtype
                )
                silu_fp32_ub = T.alloc_shared(
                    (ROWS_PER_ITER, head_size), acc_dtype
                )
                square_fp32_ub = T.alloc_shared(
                    (ROWS_PER_ITER, head_size), acc_dtype
                )
                inv_rms_ub = T.alloc_shared((ROWS_PER_ITER, 1), acc_dtype)
                inv_rms_2d_ub = T.alloc_shared(
                    (ROWS_PER_ITER, head_size), acc_dtype
                )

                T.copy(weight[0], weight_bf16_ub[0, :head_size])
                T.tile.cast(
                    weight_base_fp32_ub,
                    weight_bf16_ub,
                    "CAST_NONE",
                    head_size,
                )
                for row in T.serial(ROWS_PER_ITER):
                    T.copy(
                        weight_base_fp32_ub[0, :head_size],
                        weight_fp32_ub[row, :head_size],
                    )

                for chunk_idx in T.serial(num_chunks):
                    base_row = (chunk_start + chunk_idx) * ROWS_PER_ITER
                    remaining = T.if_then_else(
                        num_rows > base_row,
                        num_rows - base_row,
                        0,
                    )
                    valid_rows = T.if_then_else(
                        remaining >= ROWS_PER_ITER,
                        ROWS_PER_ITER,
                        remaining,
                    )

                    with T.If(valid_rows == ROWS_PER_ITER):
                        with T.Then():
                            T.copy(x[base_row, 0], x_fp16_ub)
                            T.copy(gate[base_row, 0], gate_bf16_ub)
                        with T.Else():
                            for row in T.serial(valid_rows):
                                T.copy(
                                    x[base_row + row, 0],
                                    x_fp16_ub[row, :head_size],
                                )
                                T.copy(
                                    gate[base_row + row, 0],
                                    gate_bf16_ub[row, :head_size],
                                )

                    T.tile.cast(
                        x_fp32_ub, x_fp16_ub, "CAST_NONE", tile_elements
                    )
                    T.tile.mul(x_fp32_ub, x_fp32_ub, scale)
                    # Preserve the original Mega output scale + BF16 cast
                    # before RMSNorm while keeping the value in UB.
                    T.tile.cast(
                        output_bf16_ub,
                        x_fp32_ub,
                        "CAST_RINT",
                        tile_elements,
                    )
                    T.tile.cast(
                        x_fp32_ub,
                        output_bf16_ub,
                        "CAST_NONE",
                        tile_elements,
                    )
                    T.tile.cast(
                        gate_fp32_ub,
                        gate_bf16_ub,
                        "CAST_NONE",
                        tile_elements,
                    )
                    T.tile.silu(silu_fp32_ub, gate_fp32_ub)

                    T.tile.mul(square_fp32_ub, x_fp32_ub, x_fp32_ub)
                    T.reduce_sum(square_fp32_ub, inv_rms_ub, dim=-1)
                    T.tile.mul(inv_rms_ub, inv_rms_ub, 1.0 / head_size)
                    T.tile.add(inv_rms_ub, inv_rms_ub, eps)
                    T.tile.sqrt(inv_rms_ub, inv_rms_ub)
                    T.tile.broadcast(inv_rms_2d_ub, inv_rms_ub)

                    T.tile.div(x_fp32_ub, x_fp32_ub, inv_rms_2d_ub)
                    T.tile.mul(x_fp32_ub, x_fp32_ub, weight_fp32_ub)
                    T.tile.mul(x_fp32_ub, x_fp32_ub, silu_fp32_ub)
                    T.tile.cast(
                        output_bf16_ub,
                        x_fp32_ub,
                        "CAST_RINT",
                        tile_elements,
                    )

                    with T.If(valid_rows == ROWS_PER_ITER):
                        with T.Then():
                            T.copy(output_bf16_ub, output[base_row, 0])
                        with T.Else():
                            for row in T.serial(valid_rows):
                                T.copy(
                                    output_bf16_ub[row, :head_size],
                                    output[base_row + row, :head_size],
                                )

    return fused_scale_gated_rmsnorm_kernel


@tilelang.jit(pass_configs=DEFAULT_ASCEND_PASS_CONFIGS)
def fused_scale_gated_rmsnorm_kernel_jit(
    num_rows: int,
    head_size: int = DEFAULT_HEAD_SIZE,
):
    return build_fused_scale_gated_rmsnorm_kernel(
        head_size=head_size,
        compile_max_rows=num_rows,
    )


@register_kernel
class FusedScaleGatedRmsnormKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("head_size", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"hs{head_size}_bf16",
            "head_size": head_size,
            "dtype": DEFAULT_DTYPE,
        }
        for head_size in SUPPORTED_HEAD_SIZES
    ]

    @staticmethod
    def generate_source(head_size: int, dtype: str) -> str:
        if dtype != DEFAULT_DTYPE:
            raise ValueError(
                "fused_scale_gated_rmsnorm only supports "
                f"dtype={DEFAULT_DTYPE}, got {dtype}"
            )
        tilelang.disable_cache()
        tilelang_kernel = build_fused_scale_gated_rmsnorm_kernel(
            head_size=head_size,
            compile_max_rows=DEFAULT_MAX_ROWS,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def _torch_reference(x, gate, weight, eps: float, scale: float):
    import torch
    import torch.nn.functional as F

    scaled = (x.to(torch.float32) * scale).to(torch.bfloat16)
    scaled_fp32 = scaled.to(torch.float32)
    inv_rms = (scaled_fp32.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    return (
        scaled_fp32
        * inv_rms
        * weight.to(torch.float32)
        * F.silu(gate.to(torch.float32))
    ).to(torch.bfloat16)


def _run_ref_check(num_rows: int, head_size: int, eps: float, scale: float) -> None:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logger.warning(
            "Skip fused_scale_gated_rmsnorm reference check: NPU is not available"
        )
        return

    torch.manual_seed(23)
    device = torch.device("npu")
    x = torch.randn((num_rows, head_size), device=device, dtype=torch.float16)
    gate = torch.randn(
        (num_rows, head_size), device=device, dtype=torch.bfloat16
    )
    weight = torch.randn((head_size,), device=device, dtype=torch.bfloat16)
    output = torch.empty_like(gate)

    kernel = fused_scale_gated_rmsnorm_kernel_jit(
        num_rows=num_rows,
        head_size=head_size,
    )
    kernel(x, gate, weight, output, num_rows, eps, scale)
    torch.npu.synchronize()

    reference = _torch_reference(x, gate, weight, eps, scale)
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
    logger.info(
        "fused_scale_gated_rmsnorm output matches torch reference for "
        f"shape=({num_rows}, {head_size})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TileLang AscendC source for fused_scale_gated_rmsnorm."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-size", type=int, default=DEFAULT_HEAD_SIZE)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument("--skip-ref-check", action="store_true")
    parser.add_argument("--ref-num-rows", type=int, default=17)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--scale", type=float, default=DEFAULT_HEAD_SIZE**-0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = FusedScaleGatedRmsnormKernel.generate_source(
        head_size=args.head_size,
        dtype=args.dtype,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")

    if not args.skip_ref_check:
        _run_ref_check(
            num_rows=args.ref_num_rows,
            head_size=args.head_size,
            eps=args.eps,
            scale=args.scale,
        )


if __name__ == "__main__":
    main()
