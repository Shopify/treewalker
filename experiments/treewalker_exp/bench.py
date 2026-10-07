"""Building and running sweep_bench, the Rust benchmark runner."""

import os
import subprocess
from pathlib import Path

from .paths import Paths

FEATURES = "external-bench"
TOL_F64 = 1e-14


def build(paths: Paths, features: str = FEATURES) -> Path:
    """Build sweep_bench in release mode for the host CPU."""
    env = {**os.environ, "RUSTFLAGS": "-C target-cpu=native"}
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


def validate_scenario_cell(binary: Path, model_dir: Path, cell_dir: Path) -> tuple[bool, str]:
    """TreeWalker against the cell's GTIL reference, within 1e-14."""
    cmd = [
        str(binary),
        str(model_dir),
        "--data-dir",
        str(cell_dir),
        "--group-offsets",
        str(cell_dir / "group_offsets.bin"),
        "--validate",
        str(cell_dir / "reference.bin"),
        "--tol",
        str(TOL_F64),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    out = r.stdout + r.stderr
    line = next((ln for ln in out.splitlines() if "VALIDATE" in ln), out[-400:])
    return "VALIDATE PASS" in out, line.strip()
