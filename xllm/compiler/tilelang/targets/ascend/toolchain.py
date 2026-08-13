from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.logger import logger

from ...common.toolchain import (
    find_required_executable,
    require_env,
    resolve_tilelang_root,
    run_checked,
)
from .kernels.utils import DEFAULT_ASCEND_BISHENG_ARCH

TILELANG_BISHENG_COMMON_FLAGS = [
    "-O2",
    "-std=c++17",
    "-xasc",
    "-fPIC",
    "-Wno-macro-redefined",
    "-Wno-ignored-attributes",
    "-Wno-non-c-typedef-for-linkage",
    "-DBACKEND_HYBM",
]

TILELANG_BISHENG_A5_FLAGS = [
    "-DREGISTER_BASE",
    "-D__DAV_C310__",
    "-O2",
    "-std=gnu++17",
    "-xcce",
    "--cce-auto-sync",
    "--cce-mask-opt",
    "-mllvm",
    "-cce-aicore-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-function-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-record-overflow=true",
    "-mllvm",
    "-cce-aicore-addr-transform",
    "-mllvm",
    "-cce-aicore-jump-expand=true",
    "-mllvm",
    "-cce-aicore-dcci-insert-for-scalar=false",
    "-DL2_CACHE_HINT",
    "-fPIC",
    "-Wno-macro-redefined",
    "-Wno-ignored-attributes",
    "-Wno-non-c-typedef-for-linkage",
    "-DBACKEND_HYBM",
]

TILELANG_PTO_BISHENG_COMMON_FLAGS = [
    "-O2",
    "-std=gnu++17",
    "-xcce",
    "-mllvm",
    "-cce-aicore-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-function-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-record-overflow=true",
    "-mllvm",
    "-cce-aicore-addr-transform",
    "-mllvm",
    "-cce-aicore-dcci-insert-for-scalar=false",
    "-DL2_CACHE_HINT",
    "-fPIC",
    "-Wno-macro-redefined",
    "-Wno-ignored-attributes",
    "-Wno-non-c-typedef-for-linkage",
    "-DMEMORY_BASE",
]

TILELANG_PTO_JIT_BISHENG_COMMON_FLAGS = [
    "-O2",
    "-std=gnu++17",
    "-xcce",
    "-fPIC",
    "-Wno-macro-redefined",
    "-Wno-ignored-attributes",
    "-DMEMORY_BASE",
    "-DXLLM_TILELANG_PTO_JIT",
]

ASCEND_DEVICE_TO_BISHENG_ARCH = {
    "a2": DEFAULT_ASCEND_BISHENG_ARCH,
    "a3": DEFAULT_ASCEND_BISHENG_ARCH,
    "a5": "dav-c310",
}

PTO_DEVICE_TO_BISHENG_ARCH = {
    "a3": "dav-c220",
}

_HEADER_SUFFIXES = frozenset({".def", ".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp"})


@dataclass(frozen=True)
class AscendBuildContext:
    device: str | None
    bisheng_arch: str
    bisheng_executable: str
    bisheng_compile_flags: tuple[str, ...]
    toolchain_options: dict[str, Any]
    fingerprint: dict[str, Any]
    include_dirs: list[str]


def normalize_ascend_device(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = device.strip().lower()
    if not normalized:
        return None
    if normalized not in ASCEND_DEVICE_TO_BISHENG_ARCH:
        supported = ", ".join(sorted(ASCEND_DEVICE_TO_BISHENG_ARCH))
        raise ValueError(
            f"Unsupported Ascend TileLang device {device!r}. Expected one of: "
            f"{supported}"
        )
    return normalized


def resolve_bisheng_arch(device: str | None) -> tuple[str | None, str]:
    normalized_device = normalize_ascend_device(device)
    if normalized_device is None:
        logger.warning(
            "TileLang Ascend build did not receive --device. Falling back "
            f"to default bisheng_arch={DEFAULT_ASCEND_BISHENG_ARCH}. Prefer "
            "running via xLLM main build path or pass `--device npu` explicitly."
        )
        return None, DEFAULT_ASCEND_BISHENG_ARCH
    return normalized_device, ASCEND_DEVICE_TO_BISHENG_ARCH[normalized_device]


def build_toolchain_options(device: str | None, bisheng_arch: str) -> dict[str, str]:
    toolchain_options = {"bisheng_arch": bisheng_arch}
    if device is not None:
        toolchain_options["device"] = device
    return toolchain_options


def build_bisheng_compile_flags(
    device: str | None, bisheng_arch: str
) -> list[str]:
    if device == "a5":
        return [
            f"--cce-aicore-arch={bisheng_arch}",
            *TILELANG_BISHENG_A5_FLAGS,
        ]
    return [f"--npu-arch={bisheng_arch}", *TILELANG_BISHENG_COMMON_FLAGS]


def resolve_npu_home_path() -> str:
    for env_name in ("NPU_HOME_PATH", "NPU_TOOLKIT_HOME"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    for candidate in (
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/ascend-toolkit",
    ):
        if Path(candidate).exists():
            return candidate

    raise RuntimeError(
        "Required NPU toolkit root is not set. Expected NPU_HOME_PATH or "
        "NPU_TOOLKIT_HOME, or a standard install path under "
        "/usr/local/Ascend/ascend-toolkit."
    )


def bisheng_include_dirs() -> list[str]:
    tl_root = require_env("TL_ROOT")
    npu_home_path = resolve_npu_home_path()
    return [
        f"{npu_home_path}/include",
        f"{npu_home_path}/include/experiment/runtime",
        f"{npu_home_path}/include/experiment/msprof",
        f"{npu_home_path}/compiler/tikcpp",
        f"{npu_home_path}/compiler/tikcpp/tikcfw",
        f"{npu_home_path}/compiler/tikcpp/tikcfw/impl",
        f"{npu_home_path}/compiler/tikcpp/tikcfw/interface",
        f"{tl_root}/3rdparty/catlass/include",
        f"{tl_root}/3rdparty/shmem/include",
        f"{tl_root}/3rdparty/shmem/src/device",
        f"{tl_root}/src",
    ]


def resolve_pto_include_dir(
    tilelang_root: str | Path | None = None,
) -> Path:
    root = (
        Path(tilelang_root).resolve()
        if tilelang_root is not None
        else resolve_tilelang_root()
    )
    include_dir = root / "3rdparty" / "pto-isa" / "include"
    umbrella_header = include_dir / "pto" / "pto-inst.hpp"
    if not umbrella_header.is_file():
        raise RuntimeError(
            "TileLang bundled PTO headers are incomplete. Expected: "
            f"{umbrella_header}"
        )
    return include_dir


def _header_tree_sha256(root: str | Path) -> str:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise RuntimeError(
            f"Required TileLang header directory is missing: {root_path}"
        )

    header_paths = sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in _HEADER_SUFFIXES
    )
    if not header_paths:
        raise RuntimeError(
            f"Required TileLang header directory contains no headers: {root_path}"
        )

    digest = hashlib.sha256()
    for path in header_paths:
        digest.update(path.relative_to(root_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_set_sha256(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def tilelang_lowering_fingerprint(tilelang_root: str | Path) -> dict[str, str]:
    root = Path(tilelang_root).resolve()
    source_paths = [
        path
        for path in root.rglob("*.py")
        if "3rdparty" not in path.relative_to(root).parts
        and ".git" not in path.relative_to(root).parts
    ]
    for version_name in ("VERSION", "VERSION.txt", "version.txt"):
        version_path = root / version_name
        if version_path.is_file():
            source_paths.append(version_path)
    for library_root in (root / "lib", root / "build" / "lib"):
        if library_root.is_dir():
            source_paths.extend(library_root.rglob("*.so"))
    if not source_paths:
        raise RuntimeError(
            f"TileLang lowering fingerprint found no source files under: {root}"
        )
    return {
        "tilelang_lowering_sha256": _path_set_sha256(root, source_paths),
    }


def cann_toolkit_fingerprint(npu_home_path: str | Path) -> dict[str, str]:
    root = Path(npu_home_path).resolve()
    acl_headers = root / "include" / "acl"
    runtime_headers = root / "include" / "runtime"
    fingerprint = {
        "cann_acl_headers_sha256": _header_tree_sha256(acl_headers),
        "cann_runtime_headers_sha256": _header_tree_sha256(runtime_headers),
    }
    version_paths = [
        path
        for path in (
            root / "version.info",
            root / "share/info/acl_extend/version.info",
            root / "share/info/runtime/version.info",
            root / "share/info/bisheng-compiler/version.info",
        )
        if path.is_file()
    ]
    if version_paths:
        fingerprint["cann_versions_sha256"] = _path_set_sha256(
            root, version_paths
        )
    return fingerprint


def target_header_fingerprint(
    tilelang_root: str | Path,
    target: str,
) -> dict[str, str]:
    root = Path(tilelang_root).resolve()
    if target == "ascend":
        template_dir = root / "src" / "tl_templates" / "ascend"
        return {
            "tilelang_template_headers_sha256": _header_tree_sha256(template_dir)
        }
    if target == "pto":
        template_dir = root / "src" / "tl_templates" / "pto"
        pto_include_dir = resolve_pto_include_dir(root)
        return {
            "tilelang_template_headers_sha256": _header_tree_sha256(template_dir),
            "pto_include_dir": str(pto_include_dir),
            "pto_isa_headers_sha256": _header_tree_sha256(pto_include_dir),
        }
    raise ValueError(f"Unsupported Ascend TileLang kernel target: {target}")


def _read_bisheng_version(bisheng_executable: str | Path) -> str:
    executable = str(Path(bisheng_executable).resolve())
    try:
        result = subprocess.run(
            [executable, "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable:{type(exc).__name__}"

    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    return f"returncode={result.returncode}\n{output}"


def build_bisheng_fingerprint(
    bisheng_executable: str | Path,
    *,
    version: str | None = None,
) -> dict[str, str]:
    executable = Path(bisheng_executable).resolve()
    if not executable.is_file():
        raise RuntimeError(f"Required bisheng executable is missing: {executable}")
    return {
        "bisheng_executable": str(executable),
        "bisheng_sha256": _file_sha256(executable),
        "bisheng_version": (
            version if version is not None else _read_bisheng_version(executable)
        ),
    }


def build_fingerprint(
    bisheng_executable: str,
    bisheng_arch: str,
    *,
    tilelang_root: str | Path | None = None,
    npu_home_path: str | Path | None = None,
    bisheng_version: str | None = None,
) -> dict[str, str]:
    tl_root = (
        Path(tilelang_root).resolve()
        if tilelang_root is not None
        else Path(require_env("TL_ROOT")).resolve()
    )
    resolved_npu_home = (
        Path(npu_home_path).resolve()
        if npu_home_path is not None
        else Path(resolve_npu_home_path()).resolve()
    )
    fingerprint = {
        "target": "ascend",
        "tl_root": str(tl_root),
        "npu_home_path": str(resolved_npu_home),
        "bisheng_arch": bisheng_arch,
    }
    fingerprint.update(tilelang_lowering_fingerprint(tl_root))
    fingerprint.update(cann_toolkit_fingerprint(resolved_npu_home))
    fingerprint.update(
        build_bisheng_fingerprint(
            bisheng_executable,
            version=bisheng_version,
        )
    )
    return fingerprint


def resolve_build_context(device: str | None, bisheng_executable: str) -> AscendBuildContext:
    normalized_device, bisheng_arch = resolve_bisheng_arch(device)
    fingerprint = build_fingerprint(bisheng_executable, bisheng_arch)
    if normalized_device is not None:
        fingerprint["device"] = normalized_device
    return AscendBuildContext(
        device=normalized_device,
        bisheng_arch=bisheng_arch,
        bisheng_executable=bisheng_executable,
        bisheng_compile_flags=tuple(
            build_bisheng_compile_flags(normalized_device, bisheng_arch)
        ),
        toolchain_options=build_toolchain_options(normalized_device, bisheng_arch),
        fingerprint=fingerprint,
        include_dirs=bisheng_include_dirs(),
    )


def compile_pto_jit_library(source: str, device: str = "a3") -> Path:
    normalized_device = normalize_ascend_device(device)
    if normalized_device not in PTO_DEVICE_TO_BISHENG_ARCH:
        supported = ", ".join(sorted(PTO_DEVICE_TO_BISHENG_ARCH))
        raise ValueError(
            "TileLang PTO JIT compilation requires one of devices: "
            f"{supported}; got {device!r}"
        )

    tilelang_root = resolve_tilelang_root()
    pto_include_dir = resolve_pto_include_dir(tilelang_root)
    npu_home_path = Path(resolve_npu_home_path())
    output_dir = Path(tempfile.mkdtemp(prefix="xllm_tilelang_pto_jit_"))
    source_path = output_dir / "kernel.cpp"
    library_path = output_dir / "kernel.so"
    source_path.write_text(source, encoding="utf-8")

    include_dirs = [
        pto_include_dir,
        tilelang_root / "src",
        npu_home_path / "include",
        npu_home_path / "include/experiment/msprof",
        npu_home_path / "include/experiment/runtime",
        npu_home_path / "pkg_inc",
        npu_home_path / "pkg_inc/runtime",
        npu_home_path / "pkg_inc/profiling",
    ]
    driver_include = Path("/usr/local/Ascend/driver/kernel/inc")
    if driver_include.is_dir():
        include_dirs.append(driver_include)

    compile_cmd = [
        find_required_executable("bisheng"),
        f"--cce-aicore-arch={PTO_DEVICE_TO_BISHENG_ARCH[normalized_device]}",
        *TILELANG_PTO_JIT_BISHENG_COMMON_FLAGS,
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        *[f"-I{path}" for path in include_dirs],
        f"-L{npu_home_path / 'lib64'}",
        "-lruntime",
        "-lstdc++",
        "-lascendcl",
        "-lm",
        "-ltiling_api",
        "-lplatform",
        "-lc_sec",
        "-ldl",
        "--shared",
        str(source_path),
        "-o",
        str(library_path),
    ]
    run_checked(compile_cmd)
    return library_path
