# /// script
# requires-python = ">=3.14"
# dependencies = ["lleaves @ git+https://github.com/siboehm/lleaves@v1.4.1", "pyarrow", "numpy"]
# ///

import contextlib
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import llvmlite.binding as llvm
import llvmlite.ir
from lleaves.compiler.ast import parse_to_ast
from lleaves.compiler.codegen import gen_forest

_SUBPROCESS_TIMEOUT = 300  # seconds per subprocess invocation


@dataclass
class CompileConfig:
    target_triple: str = ""
    target_cpu: str = "native"
    opt_level: str = "2"
    fp_contract: str = "fast"
    relocation_model: str = "pic"
    code_model: str = ""
    use_fp64: bool = True
    fblocksize: int = 2
    n_jobs: int = 0
    clang: str = ""
    llc: str = ""
    opt: str = ""
    use_opt: bool = False
    opt_passes: str = ""  # e.g. "default<O2>,mergefunc". Empty = default<O{opt_level}>


@dataclass
class ForestIR:
    """Parsed and split LLVM IR, ready for compilation with any backend flags."""

    preamble: str
    forest_root_block: str
    tree_blocks: list[str]
    tree_names: list[str]
    ret_type: str
    param_types_str: str
    attributes_suffix: str
    n_trees: int
    ir_size_mb: float
    use_fp64: bool
    fblocksize: int
    model_stem: str
    model_dir: Path


NATIVE = CompileConfig()

LINUX_X86_64 = CompileConfig(
    target_triple="x86_64-unknown-linux-gnu",
    target_cpu="x86-64-v3",
)

LINUX_X86_64_AVX512 = CompileConfig(
    target_triple="x86_64-unknown-linux-gnu",
    target_cpu="x86-64-v4",
)

LINUX_AARCH64 = CompileConfig(
    target_triple="aarch64-unknown-linux-gnu",
    target_cpu="neoverse-n1",
)

MACOS_ARM64 = CompileConfig(
    target_triple="arm64-apple-macosx",
    target_cpu="apple-m1",
)

WASM32 = CompileConfig(
    target_triple="wasm32-unknown-unknown",
    target_cpu="generic",
)


# ---------------------------------------------------------------------------
# Tool resolution
# ---------------------------------------------------------------------------


def _find_tool(name: str, cfg_path: str = "") -> str:
    """Resolve an LLVM tool. Priority: config > env var > Homebrew > PATH."""
    if cfg_path:
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Configured {name} path does not exist: {cfg_path}")
        return cfg_path
    from_env = os.environ.get(name.upper())
    if from_env and os.path.isfile(from_env):
        return from_env
    if platform.system() == "Darwin":
        brew_path = f"/opt/homebrew/opt/llvm/bin/{name}"
        if os.path.isfile(brew_path):
            return brew_path
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"Cannot find '{name}'. Install LLVM or set {name.upper()}=/path/to/{name}")


def _find_lib(name: str, patterns: list[str], cfg_path: str = "") -> str:
    """Find a library by glob patterns. Returns last match (latest version)."""
    if cfg_path and os.path.isfile(cfg_path):
        return cfg_path
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[-1]
    raise FileNotFoundError(f"Cannot find {name}")


def _tool_version(tool_path: str) -> str:
    """Extract a one-line version string from an LLVM tool."""
    try:
        out = subprocess.run([tool_path, "--version"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "version" in line.lower():
                return line.strip()
    except Exception:
        pass
    return "unknown"


def _lib_ext(target_triple: str) -> str:
    """Shared library extension from target triple or host OS."""
    if target_triple:
        t = target_triple.lower()
        if "wasm" in t:
            return ".wasm"
        if "darwin" in t or "apple" in t or "macos" in t:
            return ".dylib"
        if "windows" in t or "win32" in t:
            return ".dll"
        return ".so"
    if platform.system() == "Darwin":
        return ".dylib"
    if platform.system() == "Windows":
        return ".dll"
    return ".so"


# ---------------------------------------------------------------------------
# IR splitting
# ---------------------------------------------------------------------------

_DEFINE_RE = re.compile(r"^define ", re.MULTILINE)
_ATTR_RE = re.compile(r"^attributes #", re.MULTILINE)
_SIG_RE = re.compile(r'define\s+(?:(?:private|hidden|internal|dso_local)\s+)?(\w+)\s+@"?([^"(]+)"?\s*\(([^)]*)\)')


def _split_ir(ir_text: str) -> tuple[str, str, list[str], str]:
    """Split IR text into (preamble, forest_root_block, [tree_block, ...], attributes_suffix)."""
    attr_starts = [m.start() for m in _ATTR_RE.finditer(ir_text)]
    if attr_starts:
        attributes_suffix = ir_text[attr_starts[0] :]
        ir_body = ir_text[: attr_starts[0]]
    else:
        attributes_suffix = ""
        ir_body = ir_text

    define_starts = [m.start() for m in _DEFINE_RE.finditer(ir_body)]
    if len(define_starts) < 2:
        raise ValueError(f"Expected >=2 define blocks, found {len(define_starts)}")

    preamble = ir_body[: define_starts[0]]
    forest_root_block = ir_body[define_starts[0] : define_starts[1]]

    first_line = forest_root_block.split("\n", 1)[0]
    if "forest_root" not in first_line:
        raise ValueError(f"First define is not forest_root: {first_line}")

    tree_blocks = []
    for i in range(1, len(define_starts)):
        start = define_starts[i]
        end = define_starts[i + 1] if i + 1 < len(define_starts) else len(ir_body)
        tree_blocks.append(ir_body[start:end])

    return preamble, forest_root_block, tree_blocks, attributes_suffix


def _extract_tree_signature(tree_block: str) -> tuple[str, str, str]:
    """Extract (func_name, return_type, param_types_str) from a tree define block."""
    first_line = tree_block.split("\n", 1)[0]
    m = _SIG_RE.match(first_line)
    if not m:
        raise ValueError(f"Cannot parse tree signature: {first_line}")
    ret_type, func_name, raw_params = m.group(1), m.group(2), m.group(3)
    param_types = [p.strip().split("%")[0].strip() for p in raw_params.split(",") if p.strip()]
    return func_name, ret_type, ", ".join(param_types)


def _fix_linkage(tree_block: str) -> str:
    """Change private/internal linkage to hidden visibility for cross-module linking."""
    if "define private " in tree_block:
        return tree_block.replace("define private ", "define hidden ", 1)
    return tree_block.replace("define internal ", "define hidden ", 1)


def _build_root_module(preamble, forest_root_block, tree_names, ret_type, param_types_str, attributes_suffix):
    """Build root module: preamble + hidden declarations for trees + forest_root + attributes."""
    lines = [preamble]
    lines.extend(f'declare hidden {ret_type} @"{name}"({param_types_str})\n' for name in tree_names)
    lines.append("\n")
    lines.append(forest_root_block)
    if attributes_suffix:
        lines.append(attributes_suffix)
    return "".join(lines)


def _build_chunk_module(preamble: str, tree_blocks: list[str], attributes_suffix: str) -> str:
    """Build a chunk module: preamble + tree definitions with hidden visibility + attributes."""
    parts = [preamble]
    parts.extend(_fix_linkage(block) for block in tree_blocks)
    if attributes_suffix:
        parts.append(attributes_suffix)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Chunk compilation + linking
# ---------------------------------------------------------------------------


def _compile_chunk(ir_text: str, obj_path: str, *, llc_args: list[str], opt_args: list[str] | None = None) -> None:
    """Compile an IR chunk to an object file, optionally piping through opt first."""
    llc_cmd = [*llc_args, "-o", obj_path]

    if opt_args:
        # Pipe chain: IR text → opt (bitcode stdout) → llc (bitcode stdin) → .o
        try:
            opt_proc = subprocess.Popen(opt_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            llc_proc = subprocess.Popen(llc_cmd, stdin=opt_proc.stdout, stderr=subprocess.PIPE)
            opt_proc.stdout.close()  # critical for SIGPIPE propagation

            with contextlib.suppress(BrokenPipeError):
                opt_proc.stdin.write(ir_text.encode())
            opt_proc.stdin.close()

            llc_stderr = llc_proc.communicate(timeout=_SUBPROCESS_TIMEOUT)[1]
            opt_stderr = opt_proc.communicate(timeout=_SUBPROCESS_TIMEOUT)[1]
            opt_rc = opt_proc.returncode
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"opt|llc pipeline timed out for {obj_path}") from e

        if opt_rc != 0:
            raise RuntimeError(f"opt failed for {obj_path}:\n{opt_stderr.decode()}")
        if llc_proc.returncode != 0:
            raise RuntimeError(f"llc failed for {obj_path}:\n{llc_stderr.decode()}")
    else:
        try:
            result = subprocess.run(
                llc_cmd, input=ir_text.encode(), capture_output=True, check=False, timeout=_SUBPROCESS_TIMEOUT
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"llc timed out for {obj_path}") from e
        if result.returncode != 0:
            raise RuntimeError(f"llc failed for {obj_path}:\n{result.stderr.decode()}")


def _link(
    clang: str,
    target_triple: str,
    obj_paths: list[Path],
    lib_path: Path,
    *,
    emit_relocs: bool = False,
    profile_runtime: str = "",
) -> None:
    """Link object files into a shared library. Writes atomically via temp+rename."""
    tmp_path = lib_path.with_suffix(lib_path.suffix + ".tmp")
    if "wasm" in target_triple.lower():
        cmd = [
            clang,
            f"--target={target_triple}",
            "-nostdlib",
            "-Wl,--no-entry",
            "-Wl,--export=forest_root",
            "-Wl,--allow-undefined",
            "-o",
            str(tmp_path),
        ]
    else:
        cmd = [clang, "-shared", "-o", str(tmp_path)]
        if target_triple:
            cmd.extend([f"--target={target_triple}", "-fuse-ld=lld"])
    if emit_relocs:
        cmd.append("-Wl,--emit-relocs")
    cmd.extend(str(p) for p in obj_paths)
    if profile_runtime:
        if platform.system() == "Linux":
            cmd.extend(["-Wl,--whole-archive", profile_runtime, "-Wl,--no-whole-archive"])
        else:
            cmd.append(profile_runtime)

    try:
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=_SUBPROCESS_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"linker timed out after {_SUBPROCESS_TIMEOUT}s") from e
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"link failed:\n{result.stderr.decode()}")
    tmp_path.rename(lib_path)


# ---------------------------------------------------------------------------
# Profiling subprocess (shared by PGO and BOLT)
# ---------------------------------------------------------------------------

_PROFILE_SCRIPT = (
    "import ctypes, time, numpy as np, pyarrow.feather as pf\n"
    "table = pf.read_table({data_path!r})\n"
    "n_rows, n_features = table.num_rows, table.num_columns\n"
    "np_dtype = np.float64 if {use_fp64} else np.float32\n"
    "ct_dtype = ctypes.c_double if {use_fp64} else ctypes.c_float\n"
    "data = np.empty((n_rows, n_features), dtype=np_dtype)\n"
    "for i, col in enumerate(table.columns):\n"
    "    data[:, i] = col.to_numpy(zero_copy_only=False).astype(np_dtype)\n"
    "results = np.zeros(n_rows, dtype=np_dtype)\n"
    "lib = ctypes.CDLL({lib_path!r})\n"
    "lib.forest_root.restype = None\n"
    "lib.forest_root.argtypes = [ctypes.POINTER(ct_dtype), ctypes.POINTER(ct_dtype), ctypes.c_int32, ctypes.c_int32]\n"
    "data_ptr = data.ctypes.data_as(ctypes.POINTER(ct_dtype))\n"
    "results_ptr = results.ctypes.data_as(ctypes.POINTER(ct_dtype))\n"
    "t0 = time.perf_counter()\n"
    "lib.forest_root(data_ptr, results_ptr, 0, n_rows)\n"
    "t1 = time.perf_counter()\n"
    "print(f'{{n_rows}} {{t1 - t0:.4f}}')\n"
)


def _run_forest_subprocess(
    lib_path: str, data_path: str, use_fp64: bool, *, env: dict | None = None
) -> tuple[int, float, float]:
    """Run forest_root on feather data in a subprocess. Returns (n_rows, predict_time, wall_time)."""
    script = _PROFILE_SCRIPT.format(data_path=data_path, lib_path=lib_path, use_fp64=use_fp64)
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT, env=env
    )
    wall_time = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f"Profiling subprocess failed:\n{result.stderr}")
    parts = result.stdout.strip().split()
    return int(parts[0]), float(parts[1]), wall_time


# ---------------------------------------------------------------------------
# PGO helpers
# ---------------------------------------------------------------------------

_PROFILE_RT_PATTERNS = [
    "/opt/homebrew/opt/llvm/lib/clang/*/lib/darwin/libclang_rt.profile_osx.a",
    "/opt/homebrew/Cellar/llvm/*/lib/clang/*/lib/darwin/libclang_rt.profile_osx.a",
    "/usr/lib/llvm-*/lib/clang/*/lib/linux/libclang_rt.profile-x86_64.a",
    "/usr/lib/llvm-*/lib/clang/*/lib/linux/libclang_rt.profile-aarch64.a",
]

_BOLT_RT_PATTERNS = [
    "/usr/lib/llvm-*/lib/libbolt_rt_instr.a",
    "/usr/local/lib/libbolt_rt_instr.a",
]


def _strip_binary(lib_path: str, *, quiet: bool = False) -> None:
    """Strip relocations, debug sections, and BOLT's original code backup sections."""
    objcopy = shutil.which("llvm-objcopy") or (shutil.which("objcopy") if platform.system() == "Linux" else None)
    if not objcopy:
        return
    size_before = Path(lib_path).stat().st_size

    # Remove BOLT's backup sections (.bolt.org.*) and standard unneeded sections
    cmd = [objcopy, "--strip-unneeded"]
    for section in [".bolt.org.text", ".bolt.org.rodata", ".bolt.org.eh_frame", ".bolt.org.eh_frame_hdr"]:
        cmd.extend(["--remove-section", section])
    cmd.extend([lib_path, lib_path])

    result = subprocess.run(cmd, capture_output=True, check=False, timeout=_SUBPROCESS_TIMEOUT)
    if result.returncode != 0:
        return
    size_after = Path(lib_path).stat().st_size
    if not quiet:
        saved = (size_before - size_after) / (1024 * 1024)
        print(f"  Strip:                 {size_after / (1024 * 1024):.1f} MB (saved {saved:.1f} MB)", flush=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_ir(
    model_txt_path: str,
    *,
    use_fp64: bool = True,
    fblocksize: int = 2,
    target_triple: str = "",
    quiet: bool = False,
) -> ForestIR:
    """Parse a LightGBM model and generate split LLVM IR modules."""
    model_path = Path(model_txt_path)
    target = target_triple or llvm.get_process_triple()

    t0 = time.perf_counter()
    forest = parse_to_ast(str(model_path))
    forest.raw_score = False
    t1 = time.perf_counter()
    if not quiet:
        print(f"  [ir 1/3] Parse model:   {t1 - t0:.2f}s  ({len(forest.trees)} trees)", flush=True)

    ir_module = llvmlite.ir.Module(name="forest")
    gen_forest(forest, ir_module, fblocksize=fblocksize, froot_func_name="forest_root", use_fp64=use_fp64)
    ir_module.triple = target
    t2 = time.perf_counter()
    if not quiet:
        print(f"  [ir 2/3] Generate IR:   {t2 - t1:.2f}s", flush=True)

    ir_text = str(ir_module)
    preamble, forest_root_block, tree_blocks, attributes_suffix = _split_ir(ir_text)
    sigs = [_extract_tree_signature(tb) for tb in tree_blocks]
    tree_names = [name for name, _, _ in sigs]

    t3 = time.perf_counter()
    if not quiet:
        print(
            f"  [ir 3/3] Split IR:      {t3 - t2:.2f}s  ({len(ir_text) / 1e6:.1f} MB, {len(tree_blocks)} trees)",
            flush=True,
        )

    return ForestIR(
        preamble=preamble,
        forest_root_block=forest_root_block,
        tree_blocks=tree_blocks,
        tree_names=tree_names,
        ret_type=sigs[0][1],
        param_types_str=sigs[0][2],
        attributes_suffix=attributes_suffix,
        n_trees=len(tree_blocks),
        ir_size_mb=len(ir_text) / 1e6,
        use_fp64=use_fp64,
        fblocksize=fblocksize,
        model_stem=model_path.stem,
        model_dir=model_path.parent,
    )


def compile_ir(
    ir: ForestIR,
    cfg: CompileConfig,
    *,
    output_path: str | Path | None = None,
    quiet: bool = False,
    profile_runtime: str = "",
    pgo_instrument: bool = False,
    pgo_profdata: str = "",
    emit_relocs: bool = False,
) -> str:
    """Compile pre-generated ForestIR into a shared library."""
    if cfg.target_triple and cfg.target_cpu == "native":
        raise ValueError("target_cpu='native' cannot be used with an explicit target_triple.")

    n_jobs = cfg.n_jobs or os.cpu_count() or 4
    llc = _find_tool("llc", cfg.llc)
    clang = _find_tool("clang", cfg.clang)
    opt_tool = _find_tool("opt", cfg.opt) if cfg.use_opt else ""

    if not quiet:
        tools = f"llc={_tool_version(llc)}, clang={_tool_version(clang)}"
        if opt_tool:
            tools += f", opt={_tool_version(opt_tool)}"
        print(f"  Tools: {tools}", flush=True)

    if output_path is not None:
        lib_path = Path(output_path)
    else:
        suffix = "fp64" if ir.use_fp64 else "fp32"
        lib_path = ir.model_dir / f"{ir.model_stem}.{suffix}{_lib_ext(cfg.target_triple)}"
    lib_path.parent.mkdir(parents=True, exist_ok=True)

    # Build llc args (shared across all chunks, -o appended per chunk)
    llc_opt = "0" if pgo_instrument else cfg.opt_level
    llc_args = [
        llc,
        "-",
        f"-O{llc_opt}",
        f"-relocation-model={cfg.relocation_model}",
        "-filetype=obj",
        f"-mcpu={cfg.target_cpu}",
        f"-fp-contract={cfg.fp_contract}",
    ]
    if cfg.target_triple:
        llc_args.append(f"-mtriple={cfg.target_triple}")
    if cfg.code_model:
        llc_args.append(f"-code-model={cfg.code_model}")

    # Build opt args (None if opt not used)
    opt_args = None
    if opt_tool:
        if pgo_instrument:
            opt_args = [opt_tool, "-", "-passes=pgo-instr-gen,instrprof", "-o", "-"]
        else:
            passes = cfg.opt_passes or f"default<O{cfg.opt_level}>"
            opt_args = [opt_tool, "-", f"-passes={passes}", "-o", "-"]
            if pgo_profdata:
                opt_args.insert(2, f"-pgo-test-profile-file={pgo_profdata}")

    # Split trees into chunks and build IR modules
    chunk_size = max(1, (ir.n_trees + n_jobs - 1) // n_jobs)
    chunks = [ir.tree_blocks[i : i + chunk_size] for i in range(0, ir.n_trees, chunk_size)]
    root_ir = _build_root_module(
        ir.preamble,
        ir.forest_root_block,
        ir.tree_names,
        ir.ret_type,
        ir.param_types_str,
        ir.attributes_suffix,
    )
    chunk_irs = [_build_chunk_module(ir.preamble, c, ir.attributes_suffix) for c in chunks]

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="lleaves_") as tmpdir:
        tmp = Path(tmpdir)
        all_modules = [("root", root_ir)] + [(f"chunk_{i}", ir_str) for i, ir_str in enumerate(chunk_irs)]
        obj_paths = [tmp / f"{name}.o" for name, _ in all_modules]

        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(_compile_chunk, ir_str, str(obj), llc_args=llc_args, opt_args=opt_args): name
                for (name, ir_str), obj in zip(all_modules, obj_paths, strict=True)
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    raise RuntimeError(f"Compilation failed for chunk '{futures[fut]}'") from e

        _link(clang, cfg.target_triple, obj_paths, lib_path, emit_relocs=emit_relocs, profile_runtime=profile_runtime)

    t1 = time.perf_counter()
    if not quiet:
        tags = f"-O{cfg.opt_level}"
        if opt_tool:
            tags += "+opt"
        if pgo_instrument:
            tags += "+pgo-instrument"
        elif pgo_profdata:
            tags += "+pgo-use"
        print(f"  Compile+link:          {t1 - t0:.2f}s  (n={n_jobs}, {tags})  ->  {lib_path}", flush=True)

    return str(lib_path)


def compile_ir_pgo(
    ir: ForestIR,
    cfg: CompileConfig,
    profile_data_path: str,
    *,
    output_path: str | Path | None = None,
    quiet: bool = False,
    emit_relocs: bool = False,
) -> str:
    """Three-phase PGO compilation: instrument → profile → optimize."""
    if output_path is not None:
        final_path = Path(output_path)
    else:
        suffix = "fp64" if ir.use_fp64 else "fp32"
        final_path = ir.model_dir / f"{ir.model_stem}.{suffix}.pgo{_lib_ext(cfg.target_triple)}"

    profile_rt = _find_lib("LLVM profile runtime", _PROFILE_RT_PATTERNS)
    if not quiet:
        print(f"  Profile runtime: {profile_rt}", flush=True)

    t_all = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pgo_") as pgo_dir:
        pgo_tmp = Path(pgo_dir)

        # Phase 1: Instrument
        instr_lib = pgo_tmp / f"instrumented{_lib_ext(cfg.target_triple)}"
        if not quiet:
            print("  [pgo 1/3] Instrument:", flush=True)
        compile_ir(
            ir,
            replace(cfg, use_opt=True),
            output_path=instr_lib,
            quiet=quiet,
            profile_runtime=profile_rt,
            pgo_instrument=True,
        )

        # Phase 2: Profile
        raw_profile = pgo_tmp / "default.profraw"
        env = {**os.environ, "LLVM_PROFILE_FILE": str(raw_profile)}
        n_rows, predict_time, wall_time = _run_forest_subprocess(
            str(instr_lib), profile_data_path, ir.use_fp64, env=env
        )
        if not raw_profile.exists():
            raise RuntimeError(f"Profile data not written to {raw_profile}")
        if not quiet:
            prof_mb = raw_profile.stat().st_size / (1024 * 1024)
            print(
                f"  [pgo 2/3] Profile:     {wall_time:.2f}s  ({n_rows:,} rows, predict={predict_time:.2f}s, {prof_mb:.1f} MB profraw)",
                flush=True,
            )

        # Merge profdata
        profdata_tool = _find_tool("llvm-profdata")
        profdata = pgo_tmp / "merged.profdata"
        result = subprocess.run(
            [profdata_tool, "merge", "-output", str(profdata), str(raw_profile)],
            capture_output=True,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(f"llvm-profdata merge failed:\n{result.stderr.decode()}")

        # Phase 3: Optimize with PGO data
        if not quiet:
            print("  [pgo 3/3] Optimize:", flush=True)
        lib_path = compile_ir(
            ir,
            replace(cfg, use_opt=True),
            output_path=final_path,
            quiet=quiet,
            pgo_profdata=str(profdata),
            emit_relocs=emit_relocs,
        )

    t_total = time.perf_counter() - t_all
    if not quiet:
        print(f"  PGO total:             {t_total:.2f}s  ->  {final_path}", flush=True)
    return lib_path


def bolt_optimize(
    input_lib: str,
    profile_data_path: str,
    *,
    output_path: str | Path | None = None,
    use_fp64: bool = True,
    quiet: bool = False,
) -> str:
    """Post-link BOLT optimization: instrument → profile → optimize. Linux/ELF only."""
    if platform.system() != "Linux":
        raise RuntimeError("BOLT optimization requires Linux/ELF binaries")

    bolt = _find_tool("llvm-bolt")
    merge_tool = _find_tool("merge-fdata")

    input_path = Path(input_lib)
    if output_path is not None:
        final_path = Path(output_path)
    else:
        final_path = input_path.with_suffix(".bolt" + input_path.suffix)

    # Ensure BOLT runtime is available
    rt_path = _find_lib("BOLT runtime", _BOLT_RT_PATTERNS)
    rt_default = Path("/usr/local/lib/libbolt_rt_instr.a")
    if not rt_default.exists():
        with contextlib.suppress(OSError):
            rt_default.symlink_to(rt_path)

    if not quiet:
        print(f"  BOLT: {_tool_version(bolt)}", flush=True)

    t_all = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="bolt_") as bolt_dir:
        bolt_tmp = Path(bolt_dir)

        # Phase 1: Instrument
        fdata_dir = bolt_tmp / "fdata"
        fdata_dir.mkdir()
        instr_lib = str(bolt_tmp / f"instrumented{input_path.suffix}")
        t0 = time.perf_counter()
        result = subprocess.run(
            [
                bolt,
                input_lib,
                "-instrument",
                "-o",
                instr_lib,
                "-instrumentation-file-append-pid",
                f"-instrumentation-file={fdata_dir / 'prof.fdata'}",
            ],
            capture_output=True,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(f"llvm-bolt instrument failed:\n{result.stderr.decode()}")
        if not quiet:
            size = Path(instr_lib).stat().st_size / (1024 * 1024)
            print(f"  [bolt 1/3] Instrument: {time.perf_counter() - t0:.2f}s  ({size:.1f} MB)", flush=True)

        # Phase 2: Profile
        n_rows, predict_time, wall_time = _run_forest_subprocess(instr_lib, profile_data_path, use_fp64)
        fdata_files = sorted(glob.glob(str(fdata_dir / "prof.fdata*")))
        if not fdata_files:
            raise RuntimeError(f"No BOLT fdata files generated in {fdata_dir}")
        if not quiet:
            total_size = sum(Path(f).stat().st_size for f in fdata_files) / (1024 * 1024)
            print(
                f"  [bolt 2/3] Profile:    {wall_time:.2f}s  ({n_rows:,} rows, predict={predict_time:.2f}s, {total_size:.1f} MB fdata)",
                flush=True,
            )

        # Merge fdata
        merged_fdata = bolt_tmp / "merged.fdata"
        if len(fdata_files) == 1:
            Path(fdata_files[0]).rename(merged_fdata)
        else:
            with open(merged_fdata, "w") as out_f:
                result = subprocess.run(
                    [merge_tool, *fdata_files],
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=_SUBPROCESS_TIMEOUT,
                )
            if result.returncode != 0:
                raise RuntimeError(f"merge-fdata failed:\n{result.stderr.decode()}")

        # Phase 3: Optimize
        t0 = time.perf_counter()
        result = subprocess.run(
            [
                bolt,
                input_lib,
                "-o",
                str(final_path),
                f"-data={merged_fdata}",
                "-reorder-blocks=ext-tsp",
                "-reorder-functions=cdsort",
                "-split-functions",
                "-split-all-cold",
                "-dyno-stats",
            ],
            capture_output=True,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(f"llvm-bolt optimize failed:\n{result.stderr.decode()}")
        if not quiet:
            size = final_path.stat().st_size / (1024 * 1024)
            print(f"  [bolt 3/3] Optimize:   {time.perf_counter() - t0:.2f}s  ({size:.1f} MB)", flush=True)

    _strip_binary(str(final_path), quiet=quiet)

    t_total = time.perf_counter() - t_all
    if not quiet:
        print(f"  BOLT total:            {t_total:.2f}s  ->  {final_path}", flush=True)
    return str(final_path)


def validate_model(model_txt_path: str, lib_path: str, use_fp64: bool = True, n_rows: int = 100) -> None:
    """Validate compiled library predictions match LightGBM's Python predict."""
    import ctypes

    import lightgbm as lgb
    import numpy as np

    model = lgb.Booster(model_file=model_txt_path)
    n_features = model.num_feature()
    np_dtype = np.float64 if use_fp64 else np.float32
    ct_dtype = ctypes.c_double if use_fp64 else ctypes.c_float
    rng = np.random.default_rng(42)
    data = rng.random((n_rows, n_features)).astype(np_dtype)

    lgb_preds = model.predict(data, num_threads=os.cpu_count() or 4).astype(np_dtype)

    lib = ctypes.CDLL(lib_path)
    lib.forest_root.restype = None
    lib.forest_root.argtypes = [ctypes.POINTER(ct_dtype), ctypes.POINTER(ct_dtype), ctypes.c_int32, ctypes.c_int32]
    compiled_preds = np.zeros(n_rows, dtype=np_dtype)
    lib.forest_root(
        data.ctypes.data_as(ctypes.POINTER(ct_dtype)),
        compiled_preds.ctypes.data_as(ctypes.POINTER(ct_dtype)),
        0,
        n_rows,
    )

    max_diff = float(np.max(np.abs(lgb_preds - compiled_preds)))
    tolerance = 1e-6 if use_fp64 else 1e-4
    if max_diff > tolerance:
        raise AssertionError(
            f"Predictions differ by {max_diff:.2e} (tolerance={tolerance:.0e}, {'fp64' if use_fp64 else 'fp32'})"
        )
    print(f"  Validation:            PASSED (max_diff={max_diff:.2e}, n={n_rows})", flush=True)


def compile_model(
    model_txt_path: str,
    cfg: CompileConfig | None = None,
    *,
    validate: bool = True,
    pgo: str | None = None,
    bolt: str | None = None,
) -> str:
    """Compile a LightGBM model .txt into a shared library.

    Set pgo to a feather file path to enable PGO (implies use_opt=True).
    Set bolt to a feather file path to apply BOLT post-link optimization (Linux/ELF only).
    """
    if cfg is None:
        cfg = NATIVE

    t0 = time.perf_counter()
    ir = generate_ir(model_txt_path, use_fp64=cfg.use_fp64, fblocksize=cfg.fblocksize, target_triple=cfg.target_triple)

    if pgo:
        lib_path = compile_ir_pgo(ir, cfg, pgo, emit_relocs=bool(bolt))
    else:
        lib_path = compile_ir(ir, cfg, emit_relocs=bool(bolt))

    if bolt:
        lib_path = bolt_optimize(lib_path, bolt, use_fp64=cfg.use_fp64)

    t1 = time.perf_counter()
    target = cfg.target_triple or llvm.get_process_triple()
    print(f"  Total:                 {t1 - t0:.2f}s", flush=True)
    print(f"  Target:                {target} ({cfg.target_cpu})", flush=True)

    if validate and not cfg.target_triple:
        try:
            validate_model(model_txt_path, lib_path, use_fp64=cfg.use_fp64)
        except ImportError:
            print("  Validation:            SKIPPED (lightgbm not installed)", flush=True)

    return lib_path


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/model.txt"
    do_validate = "--no-validate" not in sys.argv
    cfg = WASM32 if "--wasm" in sys.argv else NATIVE

    if "--opt" in sys.argv:
        cfg = replace(cfg, use_opt=True)

    if "--opt-passes" in sys.argv:
        idx = sys.argv.index("--opt-passes")
        cfg = replace(cfg, use_opt=True, opt_passes=sys.argv[idx + 1])

    pgo_data = None
    if "--pgo" in sys.argv:
        cfg = replace(cfg, use_opt=True)
        idx = sys.argv.index("--pgo")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
            pgo_data = sys.argv[idx + 1]
        else:
            pgo_data = "outputs/fold_cache/fold_1_val_X.feather"

    bolt_data = None
    if "--bolt" in sys.argv:
        idx = sys.argv.index("--bolt")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
            bolt_data = sys.argv[idx + 1]
        else:
            bolt_data = pgo_data or "outputs/fold_cache/fold_1_val_X.feather"

    compile_model(model_path, cfg, validate=do_validate, pgo=pgo_data, bolt=bolt_data)
