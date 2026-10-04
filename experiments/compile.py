# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "lleaves @ git+https://github.com/siboehm/lleaves@v1.4.1",
#     # 0.47 is the last llvmlite on LLVM 20: its IR must parse in the VMs'
#     # LLVM 20 llc (infra/scripts/startup.sh).
#     "llvmlite<0.48",
# ]
#
# [[tool.uv.index]]
# name = "pypi"
# url = "https://pypi.org/simple"
# ///
"""Compile a LightGBM model into an lleaves shared library.

lleaves generates LLVM IR through llvmlite. The IR is split into one module
for ``forest_root`` and per-worker chunks of trees, each chunk is compiled
with ``llc`` in parallel, and ``clang`` links the objects. This avoids
llvmlite's in-process optimization, which hangs on models of 500 trees or
more.

treewalker-exp runs this script as a subprocess, with its own lock
(compile.py.lock), so the project's uv.lock does not govern lleaves:

    uv run --locked --script experiments/compile.py REQUEST.json

The request holds every setting, and the script prints the effective settings
and the library's SHA-256 as JSON. Tools come from the request or PATH.
"""

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import llvmlite
import llvmlite.binding as llvm
import llvmlite.ir
from lleaves.compiler.ast import parse_to_ast
from lleaves.compiler.codegen import gen_forest

_SUBPROCESS_TIMEOUT = 300  # seconds per subprocess invocation


@dataclass(frozen=True, slots=True)
class Request:
    model: str  # LightGBM model_native.txt
    output: str  # the shared library to write
    fblocksize: int
    opt_level: str  # llc -O
    fp_contract: str  # llc -fp-contract
    n_jobs: int  # parallel llc processes, and the number of tree chunks
    target_cpu: str = "native"
    relocation_model: str = "pic"
    use_fp64: bool = True
    llc: str = ""  # empty: from PATH
    clang: str = ""  # empty: from PATH


def _find_tool(name: str, configured: str) -> str:
    if configured:
        if not os.path.isfile(configured):
            raise FileNotFoundError(f"configured {name} does not exist: {configured}")
        return configured
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(f"{name} is not on PATH; pass it in the compile request")
    return found


def _tool_version(tool: str) -> str:
    out = subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=5).stdout
    return next((line.strip() for line in out.splitlines() if "version" in line.lower()), "unknown")


# --- IR generation and splitting ---------------------------------------------------

_DEFINE_RE = re.compile(r"^define ", re.MULTILINE)
_ATTR_RE = re.compile(r"^attributes #", re.MULTILINE)
_SIG_RE = re.compile(
    r'define\s+(?:(?:private|hidden|internal|dso_local)\s+)?(\w+)\s+@"?([^"(]+)"?\s*\(([^)]*)\)'
)


@dataclass(slots=True)
class ForestIR:
    preamble: str
    forest_root_block: str
    tree_blocks: list[str]
    tree_names: list[str]
    ret_type: str
    param_types: str
    attributes: str
    target_triple: str


def _split_ir(ir_text: str) -> tuple[str, str, list[str], str]:
    """Split IR into (preamble, forest_root block, tree blocks, attributes)."""
    attr_starts = [m.start() for m in _ATTR_RE.finditer(ir_text)]
    attributes = ir_text[attr_starts[0] :] if attr_starts else ""
    body = ir_text[: attr_starts[0]] if attr_starts else ir_text
    starts = [m.start() for m in _DEFINE_RE.finditer(body)]
    if len(starts) < 2:
        raise ValueError(f"expected at least 2 define blocks, found {len(starts)}")
    root = body[starts[0] : starts[1]]
    if "forest_root" not in root.split("\n", 1)[0]:
        raise ValueError(f"first define is not forest_root: {root.split(chr(10), 1)[0]}")
    ends = [*starts[2:], len(body)]
    trees = [body[s:e] for s, e in zip(starts[1:], ends, strict=True)]
    return body[: starts[0]], root, trees, attributes


def _signature(tree_block: str) -> tuple[str, str, str]:
    """(function name, return type, parameter types) of a tree's define."""
    first = tree_block.split("\n", 1)[0]
    m = _SIG_RE.match(first)
    if not m:
        raise ValueError(f"cannot parse tree signature: {first}")
    params = [p.strip().split("%")[0].strip() for p in m.group(3).split(",") if p.strip()]
    return m.group(2), m.group(1), ", ".join(params)


def generate_ir(model: str, *, use_fp64: bool, fblocksize: int) -> ForestIR:
    forest = parse_to_ast(model)
    forest.raw_score = False
    module = llvmlite.ir.Module(name="forest")
    gen_forest(
        forest, module, fblocksize=fblocksize, froot_func_name="forest_root", use_fp64=use_fp64
    )
    triple = llvm.get_process_triple()
    module.triple = triple
    preamble, root, trees, attributes = _split_ir(str(module))
    sigs = [_signature(t) for t in trees]
    return ForestIR(
        preamble, root, trees, [s[0] for s in sigs], sigs[0][1], sigs[0][2], attributes, triple
    )


def _root_module(ir: ForestIR) -> str:
    decls = [f'declare hidden {ir.ret_type} @"{n}"({ir.param_types})\n' for n in ir.tree_names]
    return "".join([ir.preamble, *decls, "\n", ir.forest_root_block, ir.attributes])


def _chunk_module(ir: ForestIR, blocks: list[str]) -> str:
    """Trees get hidden visibility, so the root module can link to them."""

    def fix(block: str) -> str:
        if "define private " in block:
            return block.replace("define private ", "define hidden ", 1)
        return block.replace("define internal ", "define hidden ", 1)

    return "".join([ir.preamble, *(fix(b) for b in blocks), ir.attributes])


# --- compilation -------------------------------------------------------------------


def _llc(ir_text: str, obj: Path, llc_args: list[str]) -> None:
    try:
        r = subprocess.run(
            [*llc_args, "-o", str(obj)],
            input=ir_text.encode(),
            capture_output=True,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"llc timed out for {obj.name}") from e
    if r.returncode != 0:
        raise RuntimeError(f"llc failed for {obj.name}:\n{r.stderr.decode()}")


def _link(clang: str, objs: list[Path], lib: Path) -> list[str]:
    tmp = lib.with_suffix(lib.suffix + ".tmp")
    cmd = [clang, "-shared", "-o", str(tmp), *map(str, objs)]
    r = subprocess.run(cmd, capture_output=True, check=False, timeout=_SUBPROCESS_TIMEOUT)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"link failed:\n{r.stderr.decode()}")
    tmp.replace(lib)
    return [clang, "-shared", "-o", str(lib), "<objects>"]


def compile_model(req: Request) -> dict:
    t0 = time.perf_counter()
    llc = _find_tool("llc", req.llc)
    clang = _find_tool("clang", req.clang)
    ir = generate_ir(req.model, use_fp64=req.use_fp64, fblocksize=req.fblocksize)
    llc_args = [
        llc,
        "-",
        f"-O{req.opt_level}",
        f"-relocation-model={req.relocation_model}",
        "-filetype=obj",
        f"-mcpu={req.target_cpu}",
        f"-fp-contract={req.fp_contract}",
    ]
    n = len(ir.tree_blocks)
    size = max(1, (n + req.n_jobs - 1) // req.n_jobs)
    chunks = [ir.tree_blocks[i : i + size] for i in range(0, n, size)]
    modules = [("root", _root_module(ir))]
    modules += [(f"chunk_{i}", _chunk_module(ir, c)) for i, c in enumerate(chunks)]
    lib = Path(req.output)
    lib.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lleaves_") as tmpdir:
        objs = [Path(tmpdir) / f"{name}.o" for name, _ in modules]
        with ThreadPoolExecutor(max_workers=req.n_jobs) as pool:
            futures = {
                pool.submit(_llc, text, obj, llc_args): name
                for (name, text), obj in zip(modules, objs, strict=True)
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    raise RuntimeError(f"compilation failed for chunk {futures[fut]}") from e
        link_cmd = _link(clang, objs, lib)
    digest = hashlib.sha256(lib.read_bytes()).hexdigest()
    return {
        "output": str(lib),
        "sha256": digest,
        "seconds": round(time.perf_counter() - t0, 3),
        "n_trees": n,
        "n_chunks": len(chunks),
        "settings": {
            **asdict(req),
            "llc": llc,
            "clang": clang,
            "llc_version": _tool_version(llc),
            "clang_version": _tool_version(clang),
            "llc_command": [*llc_args, "-o", "<object>"],
            "link_command": link_cmd,
            "target_triple": ir.target_triple,
            "lleaves": importlib.metadata.version("lleaves"),
            "llvmlite": llvmlite.__version__,
            "llvmlite_llvm": ".".join(map(str, llvm.llvm_version_info)),
        },
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: compile.py REQUEST.json", file=sys.stderr)
        return 2
    req = Request(**json.loads(Path(sys.argv[1]).read_text()))
    print(json.dumps(compile_model(req), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
