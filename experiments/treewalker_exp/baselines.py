"""Compiled baselines: tl2cgen for both frameworks, lleaves for LightGBM.

Each baseline is compiled once per model, compiler and target, next to the
model. ``<baseline>.json`` records the compile identity, resolved before any
reuse: the model's hash, the settings, the compiler's path, version and
target triple, the host CPU, and the generator's version (tl2cgen) or the
script and its lock (lleaves, whose lock pins llvmlite and so its LLVM). A
library is reused only when that identity and the library's hash match.
Changing a recipe is a separately measured baseline update, so the recipes
here are the released ones.
"""

import contextlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import formats as fm

# The released lleaves recipe (fblocksize 34 and O3 are lleaves's own
# defaults; fp-contract on, not compile.py's former "fast").
LLEAVES = {"fblocksize": 34, "opt_level": "3", "fp_contract": "on"}
SCHEMA = 1


@contextlib.contextmanager
def _quiet_stderr() -> Iterator[None]:
    """Silence C-level writes to fd 2 from native libraries."""
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    return (r.stdout + r.stderr).strip()


def _which(name: str, configured: str = "") -> str:
    path = configured or shutil.which(name)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{name} not found; put it on PATH or pass its path")
    return os.path.realpath(path)


def host_cpu() -> str:
    """The host CPU's model name, as the OS reports it."""
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def probe_compiler(path: str) -> dict[str, str]:
    """A C compiler's resolved path, version line and default target triple."""
    return {
        "path": path,
        "version": _run([path, "--version"]).splitlines()[0],
        "target": _run([path, "-dumpmachine"]),
    }


def probe_llc(path: str) -> dict[str, str]:
    """llc's version, default target and the CPU that -mcpu=native resolves to."""
    out = {"path": path, "version": "", "default_target": "", "host_cpu": ""}
    for line in _run([path, "--version"]).splitlines():
        text = line.strip()
        if "version" in text.lower() and not out["version"]:
            out["version"] = text
        elif text.startswith("Default target:"):
            out["default_target"] = text.split(":", 1)[1].strip()
        elif text.startswith("Host CPU:"):
            out["host_cpu"] = text.split(":", 1)[1].strip()
    return out


def _dist_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def identity_key(identity: dict[str, Any]) -> str:
    return fm.sha256_json(identity)


def reusable(record: Path, lib: Path, key: str) -> bool:
    """A library is reused only when its record carries the same compile
    identity and the file still has the recorded hash."""
    if not (record.exists() and lib.exists()):
        return False
    doc = fm.read_json(record)
    return doc.get("key") == key and doc.get("sha256") == fm.sha256_file(lib)


def tl2cgen_settings(n_trees: int, nthread: int) -> dict[str, Any]:
    # CFLAGS is set as the released harness did. Neither tl2cgen nor its gcc
    # command reads it, so these libraries target the generic CPU; PR 3's
    # baseline update passes -march=native through export_lib's options.
    return {
        "toolchain": "gcc",
        "params": {"parallel_comp": n_trees},
        "options": None,
        "env": {"CFLAGS": "-march=native", "CXXFLAGS": "-march=native"},
        "nthread": nthread,
    }


def tl2cgen_identity(model_sha: str, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "baseline": "tl2cgen",
        "model_sha256": model_sha,
        "settings": settings,
        "tl2cgen": _dist_version("tl2cgen"),
        "treelite": _dist_version("treelite"),
        # tl2cgen runs `gcc -c -O3 ... -fPIC -std=c99` from PATH, with no -march,
        # so the compiler's default target decides the code.
        "compiler": probe_compiler(_which(settings["toolchain"])),
        "host_cpu": host_cpu(),
    }


def compile_tl2cgen(fw_dir: Path, nthread: int, force: bool = False) -> dict[str, Any]:
    import tl2cgen
    import treelite

    model = fw_dir / "model_treelite.bin"
    lib, record = fw_dir / "tl2cgen.so", fw_dir / "tl2cgen.json"
    model_sha = fm.sha256_file(model)
    tl_model = treelite.Model.deserialize(str(model))
    settings = tl2cgen_settings(tl_model.num_tree, nthread)
    identity = tl2cgen_identity(model_sha, settings)
    key = identity_key(identity)
    if not force and reusable(record, lib, key):
        return fm.read_json(record)

    old = {k: os.environ.get(k) for k in settings["env"]}
    os.environ.update({k: f"{v} {old[k] or ''}".strip() for k, v in settings["env"].items()})
    t0 = time.perf_counter()
    try:
        with _quiet_stderr():
            tl2cgen.export_lib(
                tl_model,
                toolchain=settings["toolchain"],
                libpath=str(lib),
                params=settings["params"],
                nthread=nthread,
                verbose=False,
            )
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    doc = {
        "baseline": "tl2cgen",
        "model": model.name,
        "key": key,
        "identity": identity,
        "effective": {"compiler_command": "gcc -c -O3 -o {obj} {src} -fPIC -std=c99"},
        "seconds": round(time.perf_counter() - t0, 3),
        "sha256": fm.sha256_file(lib),
    }
    fm.write_json(record, doc)
    return doc


def lleaves_request(fw_dir: Path, n_jobs: int, llc: str, clang: str) -> dict[str, Any]:
    return {
        "model": str(fw_dir / "model_native.txt"),
        "output": str(fw_dir / "lleaves.so"),
        **LLEAVES,
        "n_jobs": n_jobs,
        "target_cpu": "native",
        "relocation_model": "pic",
        "use_fp64": True,
        "llc": llc,
        "clang": clang,
    }


def lleaves_identity(script: Path, model_sha: str, request: dict[str, Any]) -> dict[str, Any]:
    lock = script.with_name(script.name + ".lock")
    settings = {k: v for k, v in request.items() if k not in ("model", "output")}
    return {
        "schema_version": SCHEMA,
        "baseline": "lleaves",
        "model_sha256": model_sha,
        "settings": settings,
        "compile_py_sha256": fm.sha256_file(script),
        "compile_lock_sha256": fm.sha256_file(lock),
        "llc": probe_llc(request["llc"]),
        "clang": probe_compiler(request["clang"]),
        "host_cpu": host_cpu(),
    }


def compile_lleaves(
    script: Path, fw_dir: Path, n_jobs: int, llc: str = "", clang: str = "", force: bool = False
) -> dict[str, Any]:
    """Run compile.py under its own script lock with a structured request."""
    model = fw_dir / "model_native.txt"
    lib, record = fw_dir / "lleaves.so", fw_dir / "lleaves.json"
    model_sha = fm.sha256_file(model)
    # Resolve the tools here, so the identity names the ones compile.py runs.
    req = lleaves_request(fw_dir, n_jobs, _which("llc", llc), _which("clang", clang))
    identity = lleaves_identity(script, model_sha, req)
    key = identity_key(identity)
    if not force and reusable(record, lib, key):
        return fm.read_json(record)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(req, f)
    try:
        r = subprocess.run(
            ["uv", "run", "--quiet", "--locked", "--script", str(script), f.name],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        Path(f.name).unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile.py failed:\n{r.stderr[-2000:]}")
    result = json.loads(r.stdout)
    doc = {
        "baseline": "lleaves",
        "model": model.name,
        "key": key,
        "identity": identity,
        "request": req,
        "effective": result["settings"],
        "seconds": result["seconds"],
        "sha256": result["sha256"],
    }
    fm.write_json(record, doc)
    return doc


def summary(doc: dict[str, Any]) -> dict[str, Any]:
    """What a cell records about a compiled baseline."""
    return {
        "library": "tl2cgen.so" if doc["baseline"] == "tl2cgen" else "lleaves.so",
        "sha256": doc["sha256"],
        "key": doc["key"],
        "identity": doc["identity"],
        "effective": doc["effective"],
    }
