"""Native LightGBM and XGBoost C libraries, built from pinned sources.

The versions come from uv.lock, so each C library matches the Python package
that trained the models, and every source is pinned here by commit SHA
(LightGBM moved to github.com/lightgbm-org/LightGBM in 4.7.0). The build uses
the released flags: Release, ``-march=native``, OpenMP. Libraries go to a
git-ignored cache, ``experiments/.cache/native``, and ``native.json`` there
records each library's path, hash, version, commit and compiler; the execution
manifest passes paths and hashes to ``sweep_bench``.
"""

import os
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import formats as fm
from .paths import Paths

# Sources by (library, version). A version uv.lock pins without an entry here
# is an error: pin its commit first.
SOURCES: dict[str, dict[str, dict[str, str]]] = {
    "lightgbm": {
        "4.7.0": {
            "url": "https://github.com/lightgbm-org/LightGBM.git",
            "tag": "v4.7.0",
            "commit": "8f7036f03627054d5a54a6f965b13f4b9ff2cb63",
        },
    },
    "xgboost": {
        "3.4.1": {
            "url": "https://github.com/dmlc/xgboost.git",
            "tag": "v3.4.1",
            "commit": "6fe8c547bdd21c73e4555d85b087d9260595d30d",
        },
    },
}
LIBRARIES = ("lightgbm", "xgboost")
SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
OUTPUT = {"lightgbm": f"lib_lightgbm{SUFFIX}", "xgboost": f"libxgboost{SUFFIX}"}
FLAGS = "-march=native"


def locked_version(paths: Paths, lib: str) -> str:
    """The version uv.lock pins for the Python package of ``lib``. On Linux the
    XGBoost package is xgboost-cpu, on macOS xgboost; both are listed."""
    lock = tomllib.loads((paths.repo / "uv.lock").read_text())
    names = {"lightgbm": ["lightgbm"], "xgboost": ["xgboost-cpu", "xgboost"]}[lib]
    if lib == "xgboost" and sys.platform != "linux":
        names.reverse()
    versions = {p["name"]: p["version"] for p in lock["package"]}
    for name in names:
        if name in versions:
            return versions[name]
    raise KeyError(f"uv.lock has no {' or '.join(names)}")


def source(paths: Paths, lib: str) -> dict[str, str]:
    version = locked_version(paths, lib)
    try:
        return {"version": version, **SOURCES[lib][version]}
    except KeyError:
        raise KeyError(
            f"uv.lock pins {lib} {version}, which has no pinned source commit; "
            f"add it to treewalker_exp.native.SOURCES"
        ) from None


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{(r.stdout + r.stderr)[-4000:]}")
    return r.stdout.strip()


def _first_line(cmd: list[str]) -> str:
    try:
        return _run(cmd).splitlines()[0]
    except OSError, RuntimeError, IndexError:
        return "unknown"


def cache_dir(paths: Paths) -> Path:
    return paths.experiments / ".cache" / "native"


def records(paths: Paths) -> dict[str, dict[str, Any]]:
    index = cache_dir(paths) / "native.json"
    return fm.read_json(index) if index.exists() else {}


def _cmake_args() -> list[str]:
    args = [
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_FLAGS={FLAGS}",
        f"-DCMAKE_CXX_FLAGS={FLAGS}",
        "-DUSE_OPENMP=ON",
    ]
    if sys.platform == "darwin":
        # Homebrew's libomp, a macOS prerequisite.
        prefix = Path("/opt/homebrew/opt/libomp")
        if shutil.which("brew"):
            prefix = Path(_run(["brew", "--prefix", "libomp"]))
        args.append(f"-DOpenMP_ROOT={prefix}")
    return args


def identity(src_rec: dict[str, str]) -> dict[str, Any]:
    """What a build depends on, resolved before any reuse: the source, the CMake
    settings, the C and C++ compilers CMake will pick (CC and CXX, else cc and c++)
    with their versions and targets, CMake's version and the host CPU."""
    from .baselines import host_cpu, probe_compiler

    def compiler(env: str, name: str) -> dict[str, str]:
        wanted = os.environ.get(env) or name
        path = os.path.realpath(shutil.which(wanted) or wanted)
        if not os.access(path, os.X_OK):
            return {"path": path, "version": "missing", "target": ""}
        return probe_compiler(path)

    return {
        "url": src_rec["url"],
        "tag": src_rec["tag"],
        "commit": src_rec["commit"],
        "cmake_args": _cmake_args(),
        "cc": compiler("CC", "cc"),
        "cxx": compiler("CXX", "c++"),
        "cmake": _first_line(["cmake", "--version"]),
        "host_cpu": host_cpu(),
        "platform": sys.platform,
    }


def build(paths: Paths, lib: str, jobs: int | None = None, force: bool = False) -> dict[str, Any]:
    """Build one library unless the cache holds it with the same build identity."""
    src_rec = source(paths, lib)
    ident = identity(src_rec)
    key = fm.sha256_json(ident)
    root = cache_dir(paths) / f"{lib}-{src_rec['version']}-{src_rec['commit'][:12]}"
    out = root / OUTPUT[lib]
    record_path = root / "native.json"
    if not force and record_path.exists() and out.exists():
        rec = fm.read_json(record_path)
        if rec.get("key") == key and rec.get("sha256") == fm.sha256_file(out):
            _index(paths, lib, rec)
            return rec
    src, build_dir = root / "src", root / "build"
    if not (src / ".git").exists():
        shutil.rmtree(src, ignore_errors=True)
        src.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--quiet",
                "--depth",
                "1",
                "--branch",
                src_rec["tag"],
                "--recurse-submodules",
                "--shallow-submodules",
                src_rec["url"],
                str(src),
            ]
        )
    head = _run(["git", "rev-parse", "HEAD"], cwd=src)
    if head != src_rec["commit"]:
        raise RuntimeError(
            f"{src_rec['url']} {src_rec['tag']} is {head}, pinned {src_rec['commit']}"
        )
    shutil.rmtree(build_dir, ignore_errors=True)
    _run(["cmake", "-S", str(src), "-B", str(build_dir), *_cmake_args()])
    target = "_lightgbm" if lib == "lightgbm" else "xgboost"
    n = str(jobs or os.cpu_count() or 4)
    _run(["cmake", "--build", str(build_dir), "--target", target, "--parallel", n])
    # LightGBM writes its library to the source root, XGBoost to lib/.
    built = src / OUTPUT[lib] if lib == "lightgbm" else src / "lib" / OUTPUT[lib]
    if not built.exists():
        raise FileNotFoundError(f"the build did not produce {built}")
    shutil.copy2(built, out)
    cc = _run(["cmake", "-LA", "-N", str(build_dir)])
    compiler = next(
        (ln.split("=", 1)[1] for ln in cc.splitlines() if ln.startswith("CMAKE_CXX_COMPILER:")),
        "unknown",
    )
    rec = {
        "key": key,
        "identity": ident,
        "library": lib,
        "version": src_rec["version"],
        "url": src_rec["url"],
        "tag": src_rec["tag"],
        "commit": head,
        "path": str(out),
        "sha256": fm.sha256_file(out),
        "cmake_args": _cmake_args(),
        "compiler": {"path": compiler, "version": _first_line([compiler, "--version"])},
        "cmake": _first_line(["cmake", "--version"]),
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    fm.write_json(record_path, rec)
    _index(paths, lib, rec)
    return rec


def _index(paths: Paths, lib: str, rec: dict[str, Any]) -> None:
    index = records(paths)
    index[lib] = rec
    fm.write_json(cache_dir(paths) / "native.json", index)
