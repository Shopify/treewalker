"""Building and running sweep_bench, the Rust benchmark runner."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import Paths

# Hardware counters read perf_event, which only Linux has.
FEATURES = "external-bench,pmu" if sys.platform == "linux" else "external-bench"


def build(paths: Paths, features: str = FEATURES, rustflags: str = "-C target-cpu=native") -> Path:
    """Build sweep_bench in release mode, for the host CPU by default."""
    env = {**os.environ, "RUSTFLAGS": rustflags}
    cmd = [
        "cargo",
        "build",
        "--manifest-path",
        "experiments/benchmarks/Cargo.toml",
        "--target-dir",
        "target",
        "--release",
        "--bin",
        "sweep_bench",
        "--features",
        features,
    ]
    subprocess.run(cmd, cwd=paths.repo, check=True, env=env)
    return paths.sweep_bench


def pinned(cmd: list[str], core: int | None) -> list[str]:
    """Pin to one core with taskset where it exists (Linux)."""
    if core is None or not shutil.which("taskset"):
        return cmd
    return ["taskset", "-c", str(core), *cmd]


def validate_cell(binary: Path, cell_dir: Path, set_: str = "treewalker") -> tuple[bool, str]:
    """TreeWalker's outputs against production predict and the stage oracle."""
    cmd = [str(binary), "validate", str(cell_dir), "--set", set_]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    out = r.stdout + r.stderr
    line = next((ln for ln in out.splitlines() if ln.startswith("VALIDATE")), out[-400:])
    return r.returncode == 0, line.strip()
