"""Compare the production functions of two commits on a target and CPU.

Builds the post-LTO assembly of a binary from each commit (`git archive` into a
temporary directory, then `cargo rustc --release ... -- --emit asm` with
`-C target-cpu=CPU`, in a target directory per commit under `target/asm-identity`, which
later runs reuse), and compares every function of the two after normalizing what
changes when nothing in a function does:

- symbols are demangled (`c++filt`, which reads Rust's v0 and legacy manglings) and
  written `«path»`, so that they compare by path, not by mangled bytes whose
  backreferences shift whenever a path's length does;
- local labels, constant pools, jump and switch tables, anonymous constants, Apple's
  linker-optimization hints, crate hashes and LTO's numeric suffixes on duplicated
  local symbols are renumbered away;
- LLVM's MergeFunctions keeps one of identical functions under a name chosen by
  order, so functions only one side has are paired by identical bodies.

`--map FILE` renames A's paths first: lines `old => new` of path prefixes (e.g.
`treewalker_gbdt::parser:: => treewalker_gbdt::load::parse::`), applied in one
longest-first pass, for commits that moved code between modules.

The default binary is `examples/walk_asm.rs`, which compiles every production walk;
`--bench` builds `sweep_bench` with the external baselines (and `pmu` on Linux targets)
and reports its timed regions separately: `TimedMethod::serve` and `batch`, every
adapter's `predict_timed` and `convert_only`, `time_mode` and the timer, and every
`treewalker_gbdt` function. Cross targets need no linker: the assembly is written
before the link fails. `--asm A.s B.s` compares two files that are already built.

Exit 0 when nothing differs (with `--bench`, when no timed function differs), else 1.

    python3 experiments/scripts/asm_identity.py HEAD~1 HEAD
    python3 experiments/scripts/asm_identity.py A B --target x86_64-unknown-linux-gnu \
        --cpu emeraldrapids
    python3 experiments/scripts/asm_identity.py A B --bench --features research
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# A mangled Rust symbol (v0 `_R`, legacy `_ZN`, with Mach-O's extra underscore), not
# inside another identifier, and LLVM's suffix after it.
MANGLED = re.compile(r"(?<![A-Za-z0-9_$.])(_{0,2}(?:_R|_ZN)[A-Za-z0-9_$]+)((?:\.[A-Za-z0-9_$]+)*)")
LABELS = [
    (re.compile(r"\.?(L|l)tmp\d+"), "Ltmp"),
    (re.compile(r"\.?LBB\d+_\d+"), "LBB"),
    (re.compile(r"l_anon\.[0-9a-f]+\.\d+|\.Lanon\.[0-9a-f]+\.\d+"), "anon"),
    (re.compile(r"\.?L__unnamed_\d+|l___unnamed_\d+"), "unnamed"),
    (re.compile(r"Lloh\d+|Lfunc_(begin|end)\d+|\.?Lexception\d+|GCC_except_table\d+"), "L"),
    (re.compile(r"[l.]?L?CPI\d+_\d+|\.?LJTI\d+_\d+"), "L"),
    (
        re.compile(r"(l_switch\.table\.|\.Lswitch\.table\.)(\S+?)\.\d+(?=@|\(|\.rel|$|\s|,)"),
        r"\1\2",
    ),
    (re.compile(r"\bh[0-9a-f]{16}\b"), "H"),
    (re.compile(r"\.(?:llvm\.)?\d+»"), "»"),
]
FUNC = re.compile(r"^(«[^»]*»|_?main):")
SKIP = (".loc", ".cfi", "//", ";", "#", "@", ".p2align", ".file")
END = (".cfi_endproc", ".Lfunc_end", "Lfunc_end", ".size", ".section", ".subsections_via_symbols")
TIMED = re.compile(
    r"TimedMethod>::(serve|batch)»|::predict_timed»|::convert_only»|driver::(timing::)?time_mode»|"
    r"timer::|^«(<)?treewalker_gbdt::"
)


def demangle(text: str) -> str:
    """`text` with every mangled symbol replaced by `«path»`, through one `c++filt` call."""
    symbols = sorted({m.group(1) for m in MANGLED.finditer(text)})
    if not symbols:
        return text
    # Mach-O adds an underscore to every symbol, ELF does not: give c++filt each
    # symbol with one.
    out = subprocess.run(
        ["c++filt"],
        input="\n".join("_" + s.lstrip("_") for s in symbols) + "\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if len(out) != len(symbols):
        raise SystemExit("c++filt returned a different number of lines")
    names = dict(zip(symbols, out, strict=True))
    return MANGLED.sub(lambda m: f"«{names[m.group(1)]}{m.group(2)}»", text)


def functions(text: str, renames: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Every function of demangled assembly `text`, by name, as normalized lines."""
    if renames:
        table = dict(renames)
        pattern = re.compile("|".join(re.escape(k) for k in sorted(table, key=len, reverse=True)))
        text = pattern.sub(lambda m: table[m.group(0)], text)
    out: dict[str, list[str]] = {}
    cur: list[str] | None = None
    for line in text.splitlines():
        m = FUNC.match(line)
        if m:
            name = re.sub(r"\.(?:llvm\.)?\d+»", "»", m.group(1))
            while name in out:
                name += "'"
            cur = out[name] = []
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.startswith(END):
            cur = None
            continue
        if not s or s.startswith(SKIP):
            continue
        for pattern_, repl in LABELS:
            s = pattern_.sub(repl, s)
        cur.append(s)
    return out


def pair_merged(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict[str, list[str]]:
    """`b` with the functions only it has renamed to A's of identical body, repeated
    while that pairs more (renaming a callee can make its callers identical)."""
    while True:
        by_body = {tuple(v): k for k, v in a.items() if k not in b}
        pairs = {k: by_body[tuple(v)] for k, v in b.items() if k not in a and tuple(v) in by_body}
        if not pairs:
            return b
        for new, old in pairs.items():
            print(f"  merged alike: {new[:100]} ~ {old[:100]}")
        pattern = re.compile("|".join(re.escape(k) for k in pairs))

        def rename(m: re.Match[str], table: dict[str, str] = pairs) -> str:
            return table[m.group(0)]

        b = {pairs.get(k, k): [pattern.sub(rename, x) for x in v] for k, v in b.items()}


def build(rev: str, args: argparse.Namespace, work: Path) -> str:
    """The post-LTO assembly of `rev`'s binary for the target and CPU."""
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", f"{rev}^{{commit}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    src = work / commit
    src.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", rev], capture_output=True, check=True
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(src)], input=archive, check=True)
    target = args.target or ""
    linux = "linux" in target
    env = dict(os.environ, RUSTFLAGS=f"-C target-cpu={args.cpu}")
    if linux and sys.platform == "darwin":
        # Parquet's zstd is C; compile it for Linux with clang and the SDK's headers.
        sdk = subprocess.run(
            ["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=True
        ).stdout.strip()
        arch = target.split("-")[0]
        key = target.replace("-", "_")
        extra = " -D__arm64__" if arch == "aarch64" else ""
        env[f"CC_{key}"] = "clang"
        env[f"CFLAGS_{key}"] = f"--target={target} -isystem {sdk}/usr/include{extra}"
    if args.bench:
        features = ["external-bench", *(["pmu"] if linux else []), *args.features]
        what = ["-p", "treewalker-bench", "--bin", "sweep_bench"]
        glob, sub, main_rs = (
            "sweep_bench-*.s",
            "deps",
            "experiments/benchmarks/src/bin/sweep_bench.rs",
        )
    else:
        features = list(args.features)
        what = ["-p", "treewalker-gbdt", "--example", "walk_asm"]
        glob, sub, main_rs = "walk_asm-*.s", "examples", "examples/walk_asm.rs"
    # One target directory per commit: cargo's fingerprints of path packages rest on
    # file times, so two source trees must not share one.
    target_dir = args.target_dir / commit
    cmd = ["cargo", "rustc", "-q", "--release", *what, "--target-dir", str(target_dir)]
    cmd += ["--features", ",".join(features)] if features else []
    cmd += ["--target", target] if target else []
    # A newer binary source makes cargo link, and write the assembly, again.
    (src / main_rs).touch()
    out = target_dir / target / "release" / sub
    for old in out.glob(glob):
        old.unlink()
    built = subprocess.run([*cmd, "--", "--emit", "asm"], cwd=src, env=env, capture_output=True)
    files = sorted(out.glob(glob), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.stderr.write(built.stderr.decode(errors="replace"))
        raise SystemExit(f"{rev}: no assembly was written")
    return files[-1].read_text()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("a", help="the base commit, or with --asm its assembly file")
    p.add_argument("b", help="the other commit, or with --asm its assembly file")
    p.add_argument("--target", help="a target triple; the host's by default")
    p.add_argument("--cpu", default="native", help="-C target-cpu (default: native)")
    p.add_argument("--bench", action="store_true", help="sweep_bench and its timed regions")
    p.add_argument("--features", nargs="*", default=[], help="extra package features")
    p.add_argument("--map", type=Path, help="lines `old => new`: path renames for A")
    p.add_argument("--asm", action="store_true", help="A and B are assembly files")
    p.add_argument(
        "--target-dir",
        type=Path,
        default=ROOT / "target" / "asm-identity",
        help="where each commit's target directory goes (default: target/asm-identity)",
    )
    args = p.parse_args(argv)
    renames: list[tuple[str, str]] = []
    if args.map:
        for line in args.map.read_text().splitlines():
            if "=>" in line and not line.lstrip().startswith("#"):
                old, new = (x.strip() for x in line.split("=>", 1))
                renames.append((old, new))
    if args.asm:
        text_a, text_b = Path(args.a).read_text(), Path(args.b).read_text()
    else:
        work = Path(tempfile.mkdtemp(prefix="asm-identity-"))
        try:
            text_a, text_b = build(args.a, args, work), build(args.b, args, work)
        finally:
            shutil.rmtree(work)
    a = functions(demangle(text_a), renames)
    b = pair_merged(a, functions(demangle(text_b), []))
    bad = 0
    if args.bench:
        timed = sorted(k for k in set(a) | set(b) if TIMED.search(k))
        for k in timed:
            if k not in a or k not in b:
                print(f"  TIMED only {'A' if k in a else 'B'}: {k[:200]}")
                bad += 1
            elif a[k] != b[k]:
                print(f"  TIMED differs: {k[:200]}")
                bad += 1
        print(f"timed: {len(timed)} functions; {bad} differ or are missing")
    common = [k for k in a if k in b]
    diff = [k for k in common if a[k] != b[k]]
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    print(
        f"all: {len(a)} / {len(b)} functions, {len(common)} common; {len(diff)} differ; "
        f"{len(only_a)} only in A; {len(only_b)} only in B"
    )
    for tag, names in (("differs", diff), ("only A", only_a), ("only B", only_b)):
        for k in names:
            print(f"  {tag}: {k[:200]}")
    if args.bench:
        return 1 if bad else 0
    return 1 if diff or only_a or only_b else 0


if __name__ == "__main__":
    sys.exit(main())
