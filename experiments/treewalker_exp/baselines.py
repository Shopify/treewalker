"""Compiled baselines: tl2cgen for both frameworks, lleaves for LightGBM, and
QuickScorer's XML for XGBoost.

Each baseline is compiled once per model, compiler and target, next to the
model. ``<baseline>.json`` records the compile identity, resolved before any
reuse: the model's hash, the settings, the compiler's path, version and
target triple, the host CPU, and the generator's version (tl2cgen) or the
script and its lock (lleaves, whose lock pins llvmlite and so its LLVM). A
library is reused only when that identity and the library's hash match.
Changing a recipe is a separately measured baseline update.

PR 3's baseline update, measured on the next full run:

- tl2cgen compiles for the host CPU: ``options=["-march=native"]`` reaches
  its compiler command, where the released ``CFLAGS`` never did (finding
  0.19). ``parallel_comp`` comes from grids.toml ``[baselines]``, set by the
  pilot.
- lleaves objects come from llvmlite's own LLVM, pinned by compile.py's lock,
  linked by the system C compiler; no LLVM install.
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

# The lleaves recipe (fblocksize 34 and O3 are lleaves's own defaults;
# fp-contract on, LLVM's default). compile.py splits the trees into one chunk per
# job, so n_jobs shapes the library and belongs to the recipe: 4 chunks, however
# many compiles run alongside.
LLEAVES_CHUNKS = 4
LLEAVES = {"fblocksize": 34, "opt_level": "3", "fp_contract": "on", "n_jobs": LLEAVES_CHUNKS}
# tl2cgen compiles for the host CPU.
TL2CGEN_OPTIONS = ["-march=native"]
SCHEMA = 2
# QuickScorer's masks are u128, so a tree may have at most 128 leaves.
QUICKSCORER_MAX_LEAVES = 128


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


# Nodes per tl2cgen source file. At -O3, gcc's memory grows with a file's size:
# a 4.0M-node model (Expedia, T=1000, L=16) in 32 files, 126,000 nodes each,
# took 60 GB in one cc1 and was killed on a 120 GB VM. 8,000 nodes per file
# keeps the parallel compiles far below that, and models up to 256,000 nodes,
# such as T=500 at L=8, still get the pilot's 32 files.
MAX_NODES_PER_FILE = 8000


def parallel_comp(setting: int | str, n_trees: int, total_nodes: int = 0) -> int:
    """Source files for tl2cgen: ``"trees"`` is one per tree, the released recipe;
    a file count is raised until no file holds more than ``MAX_NODES_PER_FILE``
    nodes, and never exceeds the tree count."""
    if setting == "trees":
        return n_trees
    return min(max(int(setting), -(-total_nodes // MAX_NODES_PER_FILE)), n_trees)


def tl2cgen_settings(
    n_trees: int, files: int | str = "trees", total_nodes: int = 0
) -> dict[str, Any]:
    """What a tl2cgen library depends on. Its compile thread count does not shape
    the library, so it is not part of its identity."""
    return {
        "toolchain": "gcc",
        "params": {"parallel_comp": parallel_comp(files, n_trees, total_nodes)},
        "max_nodes_per_file": MAX_NODES_PER_FILE,
        "options": TL2CGEN_OPTIONS,
    }


def tl2cgen_identity(model_sha: str, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "baseline": "tl2cgen",
        "model_sha256": model_sha,
        "settings": settings,
        "tl2cgen": _dist_version("tl2cgen"),
        "treelite": _dist_version("treelite"),
        # tl2cgen runs `gcc -c -O3 ... -fPIC -std=c99 {options}` from PATH.
        "compiler": probe_compiler(_which(settings["toolchain"])),
        "host_cpu": host_cpu(),
    }


def compile_tl2cgen(
    fw_dir: Path,
    nthread: int,
    force: bool = False,
    files: int | str = "trees",
    lib_name: str = "tl2cgen.so",
) -> dict[str, Any]:
    import tl2cgen
    import treelite

    model = fw_dir / "model_treelite.bin"
    lib = fw_dir / lib_name
    record = lib.with_suffix(".json")
    model_sha = fm.sha256_file(model)
    tl_model = treelite.Model.deserialize(str(model))
    total_nodes = sum(
        int(tl_model.get_tree_accessor(i).get_field("num_nodes")[0])
        for i in range(tl_model.num_tree)
    )
    settings = tl2cgen_settings(tl_model.num_tree, files, total_nodes)
    identity = tl2cgen_identity(model_sha, settings)
    key = identity_key(identity)
    if not force and reusable(record, lib, key):
        return fm.read_json(record)

    t0 = time.perf_counter()
    with _quiet_stderr():
        tl2cgen.export_lib(
            tl_model,
            toolchain=settings["toolchain"],
            libpath=str(lib),
            params=settings["params"],
            nthread=nthread,
            verbose=False,
            options=list(settings["options"]),
        )
    options = " ".join([*settings["options"], "-lm"])
    doc = {
        "baseline": "tl2cgen",
        "model": model.name,
        "key": key,
        "identity": identity,
        "effective": {
            "compiler_command": f"gcc -c -O3 -o {{obj}} {{src}} -fPIC -std=c99 {options}",
            "link_command": f"gcc -shared -O3 -o {{lib}} {{objects}} -std=c99 {options}",
            "source_files": settings["params"]["parallel_comp"],
            "nthread": nthread,
        },
        "seconds": round(time.perf_counter() - t0, 3),
        "sha256": fm.sha256_file(lib),
    }
    fm.write_json(record, doc)
    return doc


def lleaves_request(fw_dir: Path, cc: str) -> dict[str, Any]:
    return {
        "model": str(fw_dir / "model_native.txt"),
        "output": str(fw_dir / "lleaves.so"),
        **LLEAVES,
        "target_cpu": "native",
        "relocation_model": "pic",
        "use_fp64": True,
        "cc": cc,
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
        "cc": probe_compiler(request["cc"]),
        "host_cpu": host_cpu(),
    }


def compile_lleaves(
    script: Path, fw_dir: Path, cc: str = "", force: bool = False
) -> dict[str, Any]:
    """Run compile.py under its own script lock with a structured request."""
    model = fw_dir / "model_native.txt"
    lib, record = fw_dir / "lleaves.so", fw_dir / "lleaves.json"
    model_sha = fm.sha256_file(model)
    # Resolve the linker here, so the identity names the one compile.py runs.
    req = lleaves_request(fw_dir, _which("cc", cc))
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


# --- QuickScorer -----------------------------------------------------------------


def _f32_below(t: float) -> float:
    """The next float32 below float32 ``t``."""
    import numpy as np

    return float(np.nextafter(np.float32(t), np.float32(-np.inf), dtype=np.float32))


def _xml_number(x: float) -> str:
    """The shortest decimal that parses back to the same float32."""
    import numpy as np

    return np.format_float_positional(np.float32(x), unique=True, trim="-")


def _xml_node(arrays: tuple, i: int, pos: str = "") -> str:
    """Node ``i`` of a tree's (left, right, feature, threshold, leaf) arrays."""
    import numpy as np

    left, right, feat, thr, leaf = arrays
    attr = f' pos="{pos}"' if pos else ""
    if left[i] < 0:
        return f"<split{attr}><output>{_xml_number(leaf[i])}</output></split>"
    t32 = np.float32(thr[i])
    if float(t32) != thr[i]:
        raise ValueError(f"threshold {thr[i]!r} is not a float32")
    return (
        f"<split{attr}><feature>{int(feat[i]) + 1}</feature>"
        f"<threshold>{_xml_number(_f32_below(float(t32)))}</threshold>"
        f"{_xml_node(arrays, int(left[i]), 'left')}{_xml_node(arrays, int(right[i]), 'right')}"
        "</split>"
    )


def quickscorer_xml(tl_model: Any) -> tuple[str | None, str]:
    """An XGBoost model as QuickScorer XML (the ``<ranker>`` ensemble format), or
    None and the reason it cannot be one.

    XGBoost sends ``x < t`` left and QuickScorer ``x <= t``, so each threshold
    becomes the next float32 below it; that is exact because XGBoost's
    thresholds and inputs are float32. Leaves are float32 already. Categorical
    splits have no QuickScorer form, and a tree may have at most 128 leaves.
    Missing values go left at every QuickScorer split whatever the learned
    default; validation decides per cell.
    """
    import numpy as np

    trees = []
    for t in range(tl_model.num_tree):
        acc = tl_model.get_tree_accessor(t)
        left = acc.get_field("cleft")
        right = acc.get_field("cright")
        if int((left < 0).sum()) > QUICKSCORER_MAX_LEAVES:
            return None, f"tree {t} has {int((left < 0).sum())} leaves; QuickScorer takes 128"
        node_type = acc.get_field("node_type")
        if np.any((left >= 0) & (node_type == 2)):
            return None, f"tree {t} has categorical splits; QuickScorer has none"
        ops = acc.get_field("cmp")
        if np.any((left >= 0) & (ops != 2)):  # Operator::kLT
            return None, f"tree {t} has a comparison other than <"
        feat = acc.get_field("split_index")
        thr = np.asarray(acc.get_field("threshold"), dtype=np.float64)
        leaf = np.asarray(acc.get_field("leaf_value"), dtype=np.float64)

        if left[0] < 0:
            return None, f"tree {t} is a single leaf; QuickScorer needs a split at the root"
        arrays = (left, right, feat, thr, leaf)
        trees.append(f'<tree id="{t + 1}" weight="1">{_xml_node(arrays, 0)}</tree>')
    body = "\n".join(trees)
    return f'<?xml version="1.0"?>\n<ranker>\n<ensemble>\n{body}\n</ensemble>\n</ranker>\n', "ready"


def write_quickscorer(fw_dir: Path, force: bool = False) -> dict[str, Any]:
    """Write ``quickscorer.xml`` for an XGBoost model, or record why there is none."""
    import treelite

    model = fw_dir / "model_treelite.bin"
    xml, record = fw_dir / "quickscorer.xml", fw_dir / "quickscorer.json"
    identity = {
        "schema_version": SCHEMA,
        "baseline": "quickscorer",
        "model_sha256": fm.sha256_file(model),
        "rule": "threshold = next float32 below t; at most 128 leaves; no categorical splits",
    }
    key = identity_key(identity)
    if not force and record.exists():
        old = fm.read_json(record)
        if old.get("key") == key and (old["status"] != "ready" or reusable(record, xml, key)):
            return old
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    text, status = quickscorer_xml(treelite.Model.deserialize(str(model)))
    doc: dict[str, Any] = {"baseline": "quickscorer", "key": key, "identity": identity}
    if text is None:
        xml.unlink(missing_ok=True)
        doc["status"] = f"unsupported: {status}"
    else:
        fm.write_bytes(xml, text.encode())
        doc.update(status="ready", sha256=fm.sha256_file(xml), effective={"format": "xml"})
    fm.write_json(record, doc)
    return doc


def model_nodes(fw_dir: Path) -> int:
    """A model's node count, what orders the compiles: from its model.json, else
    counted from its Treelite model, else 0."""
    with contextlib.suppress(OSError, KeyError, ValueError):
        return int(fm.read_json(fw_dir / "model.json")["structure"]["total_nodes"])
    model = fw_dir / "model_treelite.bin"
    if not model.exists():
        return 0
    import treelite

    m = treelite.Model.deserialize(str(model))
    return sum(int(m.get_tree_accessor(i).get_field("num_nodes")[0]) for i in range(m.num_tree))


def compile_cost(tool: str, tl2cgen_threads: int) -> int:
    """Compiler processes a compile keeps busy: tl2cgen its threads, lleaves one
    per chunk, QuickScorer's XML none but its own."""
    return {"tl2cgen": tl2cgen_threads, "lleaves": LLEAVES_CHUNKS}.get(tool, 1)


def run_budgeted(pool: Any, tasks: list[Any], cost: Any, budget: int, submit: Any) -> Iterator:
    """Run ``tasks`` in order on ``pool``, starting each as soon as the running
    tasks' costs leave room for it within ``budget`` (a task costing more runs
    alone). Order is kept strictly, so a large task is never starved by smaller
    ones behind it. Yields ``(task, future)`` as each finishes."""
    from concurrent.futures import FIRST_COMPLETED, wait

    pending = list(tasks)
    running: dict[Any, tuple[Any, int]] = {}
    used = 0
    while pending or running:
        while pending:
            c = min(cost(pending[0]), budget)
            if running and used + c > budget:
                break
            task = pending.pop(0)
            running[submit(pool, task)] = (task, c)
            used += c
        done, _ = wait(running, return_when=FIRST_COMPLETED)
        for fut in done:
            task, c = running.pop(fut)
            used -= c
            yield task, fut


LIBRARY = {"tl2cgen": "tl2cgen.so", "lleaves": "lleaves.so", "quickscorer": "quickscorer.xml"}


def summary(doc: dict[str, Any]) -> dict[str, Any]:
    """What a cell records about a compiled baseline."""
    return {
        "library": LIBRARY[doc["baseline"]],
        "status": doc.get("status", "ready"),
        "sha256": doc.get("sha256"),
        "key": doc["key"],
        "identity": doc["identity"],
        "effective": doc.get("effective"),
    }
