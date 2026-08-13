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

"""AOT family for the value-head-owned MTP RMSNorm/Z-gating stage."""

from ....common.spec import TilelangKernel, register_kernel
from .mega_gdn_mtp_decode_segmented import (
    SEGMENTED_DISPATCH_SCHEMA,
    SEGMENTED_SPECIALIZATIONS,
    lower_mega_gdn_mtp_norm_pto,
)


@register_kernel
class MegaGdnMtpDecodeNormKernel(TilelangKernel):
    TARGET = "pto"
    KERNEL_NAME = "mega_gdn_mtp_decode_norm"
    DISPATCH_SCHEMA = SEGMENTED_DISPATCH_SCHEMA
    SPECIALIZATIONS = SEGMENTED_SPECIALIZATIONS

    @staticmethod
    def generate_source(
        speculative_tokens: int,
        max_batch_size: int,
        num_state_slots: int,
        num_k_heads: int,
        num_v_heads: int,
        dtype: str,
    ) -> str:
        _, source = lower_mega_gdn_mtp_norm_pto(
            speculative_tokens=speculative_tokens,
            max_batch_size=max_batch_size,
            num_state_slots=num_state_slots,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            dtype=dtype,
        )
        return source
