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
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[5]
XLLM_PYTHON_ROOT = REPO_ROOT / "xllm"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(XLLM_PYTHON_ROOT))

from compiler.tilelang.common.cache import compute_cache_key  # noqa: E402
from compiler.tilelang.common.spec import KernelCompileSpec  # noqa: E402
from compiler.tilelang.targets.ascend import build as ascend_build  # noqa: E402
from compiler.tilelang.targets.ascend import (  # noqa: E402
    abi_entry,
    kernel_family_builder,
    toolchain,
)


def _make_tilelang_root(base: Path) -> Path:
    tilelang_root = base / "tilelang"
    (tilelang_root / "tilelang/engine").mkdir(parents=True)
    (tilelang_root / "lib").mkdir(parents=True)
    (tilelang_root / "src/tl_templates/ascend").mkdir(parents=True)
    (tilelang_root / "src/tl_templates/pto").mkdir(parents=True)
    (tilelang_root / "3rdparty/pto-isa/include/pto").mkdir(parents=True)
    (tilelang_root / "tilelang/engine/lower.py").write_text(
        "# lowering v1\n",
        encoding="utf-8",
    )
    (tilelang_root / "lib/libtilelang.so").write_bytes(b"tilelang-so-v1")
    (tilelang_root / "src/tl_templates/ascend/common.h").write_text(
        "// ascend template\n",
        encoding="utf-8",
    )
    (tilelang_root / "src/tl_templates/pto/common.h").write_text(
        "#include <pto/pto-inst.hpp>\n",
        encoding="utf-8",
    )
    (tilelang_root / "3rdparty/pto-isa/include/pto/pto-inst.hpp").write_text(
        "// PTO v1\n",
        encoding="utf-8",
    )
    return tilelang_root


def _make_cann_root(base: Path) -> Path:
    npu_home = base / "cann"
    (npu_home / "include/acl").mkdir(parents=True, exist_ok=True)
    (npu_home / "include/runtime").mkdir(parents=True, exist_ok=True)
    (npu_home / "version.info").write_text(
        "cann v1\n",
        encoding="utf-8",
    )
    (npu_home / "include/acl/acl.h").write_text(
        "// acl v1\n",
        encoding="utf-8",
    )
    (npu_home / "include/runtime/rt_ffts.h").write_text(
        "// runtime v1\n",
        encoding="utf-8",
    )
    return npu_home


def _make_context(
    tilelang_root: Path,
    *,
    device: str,
) -> toolchain.AscendBuildContext:
    npu_home = _make_cann_root(tilelang_root.parent)
    cann_include = npu_home / "include"
    return toolchain.AscendBuildContext(
        device=device,
        bisheng_arch="dav-2201",
        bisheng_executable="/fake/bisheng",
        bisheng_compile_flags=tuple(
            toolchain.build_bisheng_compile_flags(device, "dav-2201")
        ),
        toolchain_options={
            "device": device,
            "bisheng_arch": "dav-2201",
        },
        fingerprint={
            "target": "ascend",
            "tl_root": str(tilelang_root),
            "npu_home_path": str(cann_include.parent),
            "bisheng_executable": "/fake/bisheng",
            "bisheng_arch": "dav-2201",
        },
        include_dirs=[str(cann_include), str(tilelang_root / "src")],
    )


def _make_family(kernel_name: str, target: str) -> SimpleNamespace:
    compile_spec = SimpleNamespace(target=target)
    return SimpleNamespace(
        kernel_name=kernel_name,
        spec_pairs=[(compile_spec, object())],
    )


def _install_build_fakes(
    monkeypatch: pytest.MonkeyPatch,
    context: toolchain.AscendBuildContext,
    families: list[SimpleNamespace],
    built_kernel_names: list[str],
) -> list[list[str] | None]:
    requested_kernel_names: list[list[str] | None] = []

    monkeypatch.setattr(
        ascend_build,
        "find_required_executable",
        lambda _name: context.bisheng_executable,
    )
    monkeypatch.setattr(
        ascend_build,
        "resolve_build_context",
        lambda **_kwargs: context,
    )

    def fake_get_default_families(
        kernel_names: list[str] | None,
    ) -> list[SimpleNamespace]:
        requested_kernel_names.append(kernel_names)
        if kernel_names is None:
            return list(families)
        return [
            family
            for family in families
            if family.kernel_name in kernel_names
        ]

    def fake_build_kernel_family(
        family: SimpleNamespace,
        **kwargs,
    ) -> str:
        target = family.spec_pairs[0][0].target
        kernel_family_builder._variant_toolchain_options(
            kwargs["context"],
            target,
        )
        built_kernel_names.append(family.kernel_name)
        return family.kernel_name

    monkeypatch.setattr(
        ascend_build,
        "get_default_families",
        fake_get_default_families,
    )
    monkeypatch.setattr(
        ascend_build,
        "_build_kernel_family",
        fake_build_kernel_family,
    )
    return requested_kernel_names


@pytest.mark.parametrize("device", ["a2", "a5"])
def test_default_build_skips_pto_on_unsupported_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    tilelang_root = _make_tilelang_root(tmp_path)
    context = _make_context(tilelang_root, device=device)
    families = [
        _make_family("rope", "ascend"),
        _make_family("mega_gdn_mtp_decode", "pto"),
    ]
    built_kernel_names: list[str] = []
    requested = _install_build_fakes(
        monkeypatch,
        context,
        families,
        built_kernel_names,
    )

    ascend_build.build_kernels(tmp_path / "output", device=device)

    assert requested == [None]
    assert built_kernel_names == ["rope"]


@pytest.mark.parametrize("device", ["a2", "a5"])
def test_explicit_pto_request_still_raises_on_unsupported_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    tilelang_root = _make_tilelang_root(tmp_path)
    context = _make_context(tilelang_root, device=device)
    pto_family = _make_family("mega_gdn_mtp_decode", "pto")
    built_kernel_names: list[str] = []
    requested = _install_build_fakes(
        monkeypatch,
        context,
        [pto_family],
        built_kernel_names,
    )

    with pytest.raises(ValueError, match="requires one of devices"):
        ascend_build.build_kernels(
            tmp_path / "output",
            kernel_names=["mega_gdn_mtp_decode"],
            device=device,
        )

    assert requested == [["mega_gdn_mtp_decode"]]
    assert built_kernel_names == []


def test_default_a3_build_keeps_pto_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tilelang_root = _make_tilelang_root(tmp_path)
    context = _make_context(tilelang_root, device="a3")
    families = [
        _make_family("rope", "ascend"),
        _make_family("mega_gdn_mtp_decode", "pto"),
    ]
    built_kernel_names: list[str] = []
    _install_build_fakes(
        monkeypatch,
        context,
        families,
        built_kernel_names,
    )

    ascend_build.build_kernels(tmp_path / "output", device="a3")

    assert built_kernel_names == ["rope", "mega_gdn_mtp_decode"]


def test_pto_aot_include_precedes_cann_include(tmp_path: Path) -> None:
    tilelang_root = _make_tilelang_root(tmp_path)
    context = _make_context(tilelang_root, device="a3")

    include_dirs = kernel_family_builder._variant_include_dirs(
        context,
        "pto",
    )

    bundled_pto_include = str(
        tilelang_root / "3rdparty/pto-isa/include"
    )
    cann_include = context.include_dirs[0]
    assert include_dirs[0] == bundled_pto_include
    assert include_dirs.index(bundled_pto_include) < include_dirs.index(
        cann_include
    )
    assert include_dirs.count(bundled_pto_include) == 1


def test_pto_aot_uses_validated_cce_compile_flags(tmp_path: Path) -> None:
    tilelang_root = _make_tilelang_root(tmp_path)
    context = _make_context(tilelang_root, device="a3")

    compile_flags = kernel_family_builder._variant_compile_flags(
        context,
        "pto",
    )

    assert "-xcce" in compile_flags
    assert "-xasc" not in compile_flags
    assert "-std=gnu++17" in compile_flags
    assert "-cce-aicore-stack-size=0x8000" in compile_flags
    assert "-cce-aicore-function-stack-size=0x8000" in compile_flags
    assert "-cce-aicore-record-overflow=true" in compile_flags
    assert "-cce-aicore-addr-transform" in compile_flags
    assert "-cce-aicore-dcci-insert-for-scalar=false" in compile_flags
    assert "-DL2_CACHE_HINT" in compile_flags


def _make_compile_spec() -> KernelCompileSpec:
    return KernelCompileSpec(
        target="pto",
        kernel_name="mega_gdn_mtp_decode",
        module_name="mega_gdn_mtp_decode",
        variant_key="k1",
        specialization={"speculative_tokens": 1},
    )


def test_internal_symbols_are_unique_and_fit_pto_runtime_limit() -> None:
    source = (
        'extern "C" __global__ AICORE void launch_kernel() {}\n'
        "void launch_kernel_tiling() {}\n"
    )
    conv_spec = _make_compile_spec()
    recurrent_spec = KernelCompileSpec(
        target="pto",
        kernel_name="mega_gdn_mtp_decode_recurrent",
        module_name="mega_gdn_mtp_decode_recurrent",
        variant_key="k1",
        specialization={"speculative_tokens": 1},
    )

    conv_suffix = kernel_family_builder._variant_internal_symbol_suffix(
        conv_spec
    )
    recurrent_suffix = (
        kernel_family_builder._variant_internal_symbol_suffix(
            recurrent_spec
        )
    )
    conv_source = abi_entry.rename_variant_internal_symbols(
        source, conv_suffix
    )
    recurrent_source = abi_entry.rename_variant_internal_symbols(
        source, recurrent_suffix
    )

    assert conv_suffix != recurrent_suffix
    assert f"launch_kernel__{conv_suffix}" in conv_source
    assert f"launch_kernel__{recurrent_suffix}" in recurrent_source
    assert len(f"launch_kernel__{conv_suffix}_mix_aiv") <= 63
    assert len(f"launch_kernel__{recurrent_suffix}_mix_aiv") <= 63


def test_bisheng_content_and_version_invalidate_cache_key(
    tmp_path: Path,
) -> None:
    tilelang_root = _make_tilelang_root(tmp_path)
    npu_home = _make_cann_root(tmp_path)
    bisheng = tmp_path / "bisheng"
    dependency = tmp_path / "dependency.py"
    dependency.write_text("# dependency\n", encoding="utf-8")
    bisheng.write_bytes(b"compiler-v1")

    fingerprint_v1 = toolchain.build_fingerprint(
        str(bisheng),
        "dav-2201",
        tilelang_root=tilelang_root,
        npu_home_path=npu_home,
        bisheng_version="bisheng 1",
    )
    cache_key_v1 = compute_cache_key(
        _make_compile_spec(),
        fingerprint_v1,
        [dependency],
    )

    bisheng.write_bytes(b"compiler-v2")
    fingerprint_new_binary = toolchain.build_fingerprint(
        str(bisheng),
        "dav-2201",
        tilelang_root=tilelang_root,
        npu_home_path=npu_home,
        bisheng_version="bisheng 1",
    )
    cache_key_new_binary = compute_cache_key(
        _make_compile_spec(),
        fingerprint_new_binary,
        [dependency],
    )
    fingerprint_new_version = toolchain.build_fingerprint(
        str(bisheng),
        "dav-2201",
        tilelang_root=tilelang_root,
        npu_home_path=npu_home,
        bisheng_version="bisheng 2",
    )
    cache_key_new_version = compute_cache_key(
        _make_compile_spec(),
        fingerprint_new_version,
        [dependency],
    )

    assert fingerprint_v1["bisheng_sha256"] != (
        fingerprint_new_binary["bisheng_sha256"]
    )
    assert cache_key_v1 != cache_key_new_binary
    assert cache_key_new_binary != cache_key_new_version


@pytest.mark.parametrize(
    ("relative_path", "updated_content"),
    [
        ("tilelang/engine/lower.py", b"# lowering v2\n"),
        ("lib/libtilelang.so", b"tilelang-so-v2"),
        ("../cann/include/acl/acl.h", b"// acl v2\n"),
        ("../cann/include/runtime/rt_ffts.h", b"// runtime v2\n"),
        ("../cann/version.info", b"cann v2\n"),
    ],
)
def test_lowering_and_cann_changes_invalidate_cache_key(
    tmp_path: Path,
    relative_path: str,
    updated_content: bytes,
) -> None:
    tilelang_root = _make_tilelang_root(tmp_path)
    npu_home = _make_cann_root(tmp_path)
    bisheng = tmp_path / "bisheng"
    dependency = tmp_path / "dependency.py"
    bisheng.write_bytes(b"compiler-v1")
    dependency.write_text("# dependency\n", encoding="utf-8")

    fingerprint_v1 = toolchain.build_fingerprint(
        str(bisheng),
        "dav-2201",
        tilelang_root=tilelang_root,
        npu_home_path=npu_home,
        bisheng_version="bisheng 1",
    )
    cache_key_v1 = compute_cache_key(
        _make_compile_spec(),
        fingerprint_v1,
        [dependency],
    )
    changed_path = tilelang_root / relative_path
    changed_path.write_bytes(updated_content)
    fingerprint_v2 = toolchain.build_fingerprint(
        str(bisheng),
        "dav-2201",
        tilelang_root=tilelang_root,
        npu_home_path=npu_home,
        bisheng_version="bisheng 1",
    )
    cache_key_v2 = compute_cache_key(
        _make_compile_spec(),
        fingerprint_v2,
        [dependency],
    )

    assert fingerprint_v1 != fingerprint_v2
    assert cache_key_v1 != cache_key_v2


def test_pto_header_change_invalidates_cache_key(tmp_path: Path) -> None:
    tilelang_root = _make_tilelang_root(tmp_path)
    context = _make_context(tilelang_root, device="a3")
    dependency = tmp_path / "dependency.py"
    dependency.write_text("# dependency\n", encoding="utf-8")
    pto_header = (
        tilelang_root / "3rdparty/pto-isa/include/pto/pto-inst.hpp"
    )

    fingerprint_v1 = kernel_family_builder._variant_fingerprint(
        context,
        "pto",
    )
    cache_key_v1 = compute_cache_key(
        _make_compile_spec(),
        fingerprint_v1,
        [dependency],
    )
    pto_header.write_text("// PTO v2\n", encoding="utf-8")
    fingerprint_v2 = kernel_family_builder._variant_fingerprint(
        context,
        "pto",
    )
    cache_key_v2 = compute_cache_key(
        _make_compile_spec(),
        fingerprint_v2,
        [dependency],
    )

    assert fingerprint_v1["external_headers"] != (
        fingerprint_v2["external_headers"]
    )
    assert cache_key_v1 != cache_key_v2
