"""Check that the production walk loops spill exactly where the lanes model says.

The walk's lane count comes from a rule (`lanes` in `src/rules/lanes.rs`): on AArch64 the
most lanes whose nodes and cursors fit the registers beside the loop's invariants and a
step's temporaries, so that no lane spills; on x86-64 the fewest lanes that reach the
core's issue bound, which spills the lane values that do not fit (`spilled` there). This
script builds `examples/walk_asm.rs` with the release profile (fat LTO) for a target and
CPU, emits its post-LTO assembly, finds each walk function's main loop (the widest
backward branch) and counts the instructions in it that store to the stack and that load
from it. A spilled lane is stored and reloaded every step; a spilled invariant is only
reloaded, with an address that does not depend on the lane's previous step.

LLVM's allocator weighs uses, and an invariant is used by every lane's step while a
lane's node or cursor is used by one, so when the registers run short it spills lanes
first; the model therefore counts the invariants too. A walk's lanes and rate come from
its symbol (the rule's lanes, K = 0, from its word stores per round). The check fails when
a loop stores to the stack and the model has every value in registers, or stores nothing
and the model has values that do not fit (`MODEL` below mirrors `src/rules/lanes.rs`).

    python3 experiments/scripts/walk_spills.py [--target TRIPLE] [--cpu CPU] [--lanes K]

`--lanes K` builds with `TREEWALKER_LANES=K` (every walk at K lanes) to see where the
register file runs out. Cross targets need their standard library (rustup target
add); the assembly is emitted without linking.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WALK = re.compile(r"4walk4walk")
LABEL = re.compile(r"^([.\w$]+):")
# Each lane's step stores its node's word 1 at its cursor, an 8-byte scaled index.
WORD_STORE = re.compile(
    r"^str\s+x\d+, \[x\d+, [wx]\d+, (uxtw|lsl) #3\]|^movq\s+%r\w+, \(%r\w+,%r\w+,8\)"
)


def functions(asm: str) -> dict[str, list[str]]:
    """Each walk function's lines, from its label to the next function's."""
    out: dict[str, list[str]] = {}
    cur: str | None = None
    for line in asm.splitlines():
        m = LABEL.match(line)
        if m and not re.match(r"^\.?L", m.group(1)) and not line.startswith(("ltmp", "Ltmp")):
            cur = m.group(1) if WALK.search(m.group(1)) else None
            if cur:
                out[cur] = []
            continue
        if cur is not None:
            s = line.strip()
            if s.startswith((".cfi_endproc", ".Lfunc_end")):
                cur = None
            elif s and not s.startswith(("#", ";", "//")):
                out[cur].append(s)
    return out


def main_loop(body: list[str]) -> list[str]:
    """The instructions of the widest backward branch's range."""
    labels = {}
    for i, s in enumerate(body):
        m = LABEL.match(s)
        if m:
            labels[m.group(1)] = i
    best: tuple[int, int] | None = None
    for i, s in enumerate(body):
        parts = re.split(r"[\s,]+", s)
        target = parts[-1] if parts else ""
        if parts and (parts[0].startswith(("j", "b", "cb", "tb"))) and target in labels:
            lo = labels[target]
            if lo < i and (best is None or i - lo > best[1] - best[0]):
                best = (lo, i)
    if best is None:
        return []
    return [s for s in body[best[0] : best[1] + 1] if not LABEL.match(s) and not s.startswith(".")]


B62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CONSTS = re.compile(r"4walk4walk(.*?)((?:K(?:j[0-9a-f]*_|b[01]_|B[0-9a-zA-Z]*_))+)E")


def consts(name: str) -> tuple[str, list[int | bool]] | None:
    """The walk's items and its consts (K, VARYING, D, R) from a v0 symbol, its
    back-references resolved: K is 0 for the rule's lanes."""
    m = CONSTS.search(name)
    if not m:
        return None
    body = name.lstrip("_")[1:]
    item = "roots" if m.group(1).startswith("m") else "items"
    out: list[int | bool] = []
    for tok in re.findall(r"K(j[0-9a-f]*_|b[01]_|B[0-9a-zA-Z]*_)", m.group(2)):
        if tok[0] == "B":
            v = 0
            for ch in tok[1:-1]:
                v = v * 62 + B62.index(ch)
            ref = re.match(r"j[0-9a-f]*_|b[01]_", body[v + 1 if tok[1:-1] else 0 :])
            if not ref:
                return None
            tok = ref.group(0)
        out.append(int(tok[1:-1] or "0", 16) if tok[0] == "j" else tok[1] == "1")
    return item, out[:4]


def describe(name: str) -> str:
    """The walk's stops, items, prefetch distance and rate, from a v0 symbol."""
    c = consts(name)
    if not c:
        return name[-40:]
    item, (_, varying, d, r) = c
    return f"{'varying' if varying else 'constant':8} {item:5} distance {d:2} rate {r}"


# As in src/rules/lanes.rs: registers, a step's temporaries, invariants, and whether a
# walk with varying stops holds its lane count in a register (AArch64's `csel`) and
# keeps one temporary fewer (its `tst` sets only the flags).
MODEL = {
    "aarch64-apple": (29, 5, 5, True),
    "aarch64": (30, 5, 5, True),
    "x86_64-apple": (14, 3, 6, False),
    "x86_64": (15, 3, 6, False),
}


def model(target: str | None) -> tuple[int, int, int, bool]:
    import platform

    t = target or f"{platform.machine().replace('arm64', 'aarch64')}-{sys.platform}"
    arch = "x86_64" if t.startswith("x86_64") else "aarch64"
    apple = "apple" in t or "darwin" in t
    return MODEL[f"{arch}-apple" if apple else arch]


def spilled(varying: bool, lanes: int, target: str | None) -> int:
    """The lane values that do not fit the registers beside the invariants and a step's
    temporaries, as `spilled` in src/rules/lanes.rs."""
    regs, temps, base, select = model(target)
    invariants = base + varying + (varying and select)
    temporaries = temps - (varying and select)
    return max(0, 2 * lanes + invariants + temporaries - regs)


def stack(s: str) -> bool:
    return bool(re.search(r"\[sp\b|\[x29\b|\(%rsp\)|\(%rbp\)", s))


def stack_store(s: str) -> bool:
    """Whether the instruction writes the stack: an AArch64 store to [sp] or [x29], or
    an x86 instruction whose destination, its last AT&T operand, is a stack slot."""
    op = s.split(None, 1)[0]
    if re.match(r"st", op):
        return bool(re.search(r"\[sp\b|\[x29\b", s))
    if op.startswith(("cmp", "test", "bt", "prefetch", "ucomi")):
        return False
    return bool(re.search(r"(^\w+\s+|,\s*)-?\w*\(%r[sb]p\)\s*$", s))


def build(target: str | None, cpu: str, lanes: int | None) -> Path:
    env = dict(os.environ)
    env["RUSTFLAGS"] = f"-C target-cpu={cpu}"
    if lanes:
        env["TREEWALKER_LANES"] = str(lanes)
    out = ROOT / "target" / "walk-spills"
    cmd = ["cargo", "rustc", "--quiet", "--release", "--example", "walk_asm"]
    cmd += ["--target-dir", str(out)]
    if target:
        cmd += ["--target", target]
    cmd += ["--", "--emit", "asm"]
    # An up-to-date build emits no assembly, so touch the example to rebuild it. A
    # cross target's link fails after the assembly is written; only a missing
    # assembly file is an error.
    os.utime(ROOT / "examples" / "walk_asm.rs")
    run = subprocess.run(cmd, cwd=ROOT, env=env, check=False, capture_output=True, text=True)
    examples = out / (target or "") / "release" / "examples"
    found = sorted(examples.glob("walk_asm-*.s"), key=lambda p: p.stat().st_mtime)
    if not found:
        sys.exit(f"no assembly was emitted:\n{run.stderr}")
    return found[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target")
    ap.add_argument("--cpu", default="native")
    ap.add_argument("--lanes", type=int)
    ap.add_argument("--asm", type=Path, help="check this assembly instead of building")
    a = ap.parse_args(argv)
    path = a.asm or build(a.target, a.cpu, a.lanes)
    text = path.read_text()
    walks = functions(text)
    if a.target is None and a.asm is not None:
        # Name the assembly's target by its register syntax.
        a.target = "x86_64-unknown-linux-gnu" if "%rsp" in text else "aarch64-apple-darwin"
    if not walks:
        print(f"{path}: no walk functions found")
        return 1
    failed = 0
    for name, body in sorted(walks.items()):
        loop = main_loop(body)
        stores = sum(1 for s in loop if stack_store(s))
        loads = sum(1 for s in loop if stack(s)) - stores
        words = sum(1 for s in loop if WORD_STORE.match(s))
        c = consts(name)
        varying, rate = (bool(c[1][1]), int(c[1][3])) if c else (False, 1)
        lanes = (int(c[1][0]) or words // rate) if c else words
        expect = spilled(varying, lanes, a.target)
        wrong = (stores > 0) != (expect > 0)
        failed += wrong
        flag = "WRONG" if wrong else "spills" if stores else "ok"
        print(
            f"{flag:6} {lanes:2} lanes, {len(loop):4} instructions, "
            f"{stores:2} stack stores, {loads:2} stack loads, model {expect} values spilled  "
            f"{describe(name)}"
        )
    print(
        f"{len(walks)} walk loops, {failed} storing to the stack where the model has every "
        f"value in registers or not storing where it spills ({path.name})"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
