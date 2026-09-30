#!/usr/bin/env python3
"""Prepare artifacts for experiment E1 (scenario-analysis benchmark).

Dataset (FROZEN 2026-07-24, OpenML 42477): UCI Default of Credit Card Clients
(Yeh & Lien 2009), 30,000 rows, 23 numeric features x1..x23, binary target y
(22.1% positive). Downloaded as ARFF over plain https; the ARFF body is CSV
after the ``@data`` line. A leading ``id`` column is dropped.

Workload = grouped what-if scoring. One base row from the test split is
expanded into G variants. The k perturbed features vary across variants
(deterministic multiplicative perturbation, log-uniform multipliers in
[0.5, 2.0] via ``np.random.default_rng(42)``; variant 0 = unperturbed base
row). ALL other features are bit-identical across the group (hard engine
invariant). This is the canonical credit-stress-testing scenario.

Perturbable pool (frozen): x1 (credit limit), x12-x17 (bill amounts),
x18-x23 (payment amounts) = 13 numeric features. Nested k sets
k in {1,2,4,8} are ordered by descending LightGBM gain importance within the
pool, so k=1 subset k=2 subset k=4 subset k=8.

Grid: k in {1,2,4,8} x G in {4,16,64,128} = 16 cells. One LightGBM model
(T=500, L=8, num_leaves=255, binary objective, seed 42) is shared across all
cells. Each cell writes its own walker_config.json (varying_mask = the k-set,
max_group_width = G), test_data.bin, group_offsets.bin, and a GTIL f64
reference (reference.bin) for the correctness gate.

Reuses utils.py helpers: train_lgb_model, write_raw_f64, write_group_offsets,
build_sweep_bench/find_sweep_bench. prepare.py is NOT modified.

Usage:
    uv run python3 paper/experiments/scripts/prepare_scenario.py            # all 16 cells
    uv run python3 paper/experiments/scripts/prepare_scenario.py --cell k1_G4   # hard-gate cell
    uv run python3 paper/experiments/scripts/prepare_scenario.py --no-validate  # skip sweep_bench gate
    uv run python3 paper/experiments/scripts/prepare_scenario.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ARTIFACTS_DIR,
    DATA_DIR,
    PROJECT_ROOT,
    build_sweep_bench,
    train_lgb_model,
    write_group_offsets,
    write_raw_f64,
)

# ---------------------------------------------------------------------------
# macOS treelite bootstrap (libomp rpath fix)
# ---------------------------------------------------------------------------
# treelite's bundled libtreelite.dylib references @rpath/libomp.dylib, which is
# not on the default macOS loader path. dyld caches the environment at process
# launch, so an in-process os.environ change is too late: re-exec the script
# with DYLD_LIBRARY_PATH pointing at the homebrew libomp install. This is a
# local-machine convenience only; the final-numbers platform is Linux.
def _bootstrap_libomp() -> None:
    if sys.platform != "darwin":
        return
    cand = Path("/opt/homebrew/opt/libomp/lib/libomp.dylib")
    if not cand.exists():
        # Intel macs / other locations.
        cand = Path("/usr/local/opt/libomp/lib/libomp.dylib")
    if not cand.exists():
        return
    libdir = str(cand.parent)
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    if libdir in existing.split(":"):
        return
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = (
        libdir if not existing else f"{libdir}:{existing}"
    )
    # Re-exec so dyld picks up the new library path at launch.
    os.execvpe(sys.executable, [sys.executable, "-u", *sys.argv], env)


_bootstrap_libomp()

import lightgbm as lgb  # noqa: E402
import treelite  # noqa: E402


# ---------------------------------------------------------------------------
# Frozen spec
# ---------------------------------------------------------------------------

ARFF_URL = (
    "https://openml.org/data/v1/download/21854402/"
    "default-of-credit-card-clients.arff"
)
CACHE_NAME = "default-of-credit-card-clients.arff"

N_TREES = 500
MAX_DEPTH = 8
NUM_LEAVES = 2 ** MAX_DEPTH - 1  # 255, matches utils.train_lgb_model convention
SEED = 42
TEST_FRAC = 0.2
TOL_F64 = 1e-14

# Number of base rows (groups) per cell. Plenty for the block protocol
# (>=11 blocks x 12 batches) while keeping the largest cell (G=128) modest:
# 2000 x 128 x 23 x 8B ~ 47MB.
N_BASE = 2000

# Grid: k in {1,2,4,8} x G in {4,16,64,128}. The 3x3 core is generated first.
K_VALUES = [1, 2, 4, 8]
G_VALUES = [4, 16, 64, 128]
CORE_K = [1, 4, 8]
CORE_G = [4, 16, 128]

# Perturbable pool (frozen): x1 (credit limit), x12-x17 (bill amounts),
# x18-x23 (payment amounts). Feature names are x1..x23 (0-indexed 0..22 after
# dropping the id column): x1->0, x12->11, x17->16, x18->17, x23->22.
PERTURBABLE_INDICES = (
    [0]
    + list(range(11, 17))   # x12..x17 -> 11..16
    + list(range(17, 23))   # x18..x23 -> 17..22
)
assert len(PERTURBABLE_INDICES) == 13

OUT_ROOT = ARTIFACTS_DIR / "scenario_credit"
MODEL_DIR = OUT_ROOT / "lightgbm"          # shared trained model
CELLS_DIR = OUT_ROOT / "cells"             # one subdir per (k, G)

FEATURE_NAMES = [f"x{i}" for i in range(1, 24)]  # x1..x23
N_FEATURES = 23


# ---------------------------------------------------------------------------
# Dataset loading (cached like the other loaders)
# ---------------------------------------------------------------------------

def load_credit_arff() -> tuple[np.ndarray, np.ndarray]:
    """Download (cached) and parse the OpenML ARFF. Returns (X, y).

    Drops the leading ``id`` column. X is float64 [30000, 23], y is float64
    [30000] in {0, 1}.
    """
    cache_dir = Path(tempfile.gettempdir()) / "treewalker_datasets"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / CACHE_NAME
    if not cache_path.exists():
        print(f"  Downloading {CACHE_NAME} from OpenML...", file=sys.stderr)
        import urllib.request
        urllib.request.urlretrieve(ARFF_URL, str(cache_path))

    text = cache_path.read_text()
    # Split off the @data body.
    marker = "@data"
    idx = text.lower().find(marker)
    if idx < 0:
        raise ValueError("ARFF has no @data marker")
    body = text[idx + len(marker):]
    # Parse CSV body (strip quotes/whitespace, skip blank lines).
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        rows.append([float(v) for v in line.split(",")])
    arr = np.array(rows, dtype=np.float64)

    # Columns: id, x1..x23, y  (25 total). Drop id (col 0); y is last.
    if arr.shape[1] != 25:
        raise ValueError(
            f"Expected 25 columns (id, x1..x23, y), got {arr.shape[1]}"
        )
    X = arr[:, 1:24]          # x1..x23
    y = arr[:, 24]            # target
    return X, y


# ---------------------------------------------------------------------------
# Train / test split (deterministic, seed 42)
# ---------------------------------------------------------------------------

def split_train_test(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    n_test = max(1, int(n * TEST_FRAC))
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Nested k feature sets (descending LightGBM gain importance within pool)
# ---------------------------------------------------------------------------

def ordered_perturbable_pool(bst: lgb.Booster) -> list[int]:
    """Return perturbable feature indices sorted by descending gain importance."""
    gains = bst.feature_importance(importance_type="gain")
    scored = [(gains[i], i) for i in PERTURBABLE_INDICES]
    scored.sort(key=lambda t: (-t[0], t[1]))  # desc gain, ties by index
    return [i for _, i in scored]


def k_set(ordered_pool: list[int], k: int) -> list[int]:
    """Top-k perturbable feature indices (nested by construction)."""
    return sorted(ordered_pool[:k])


# ---------------------------------------------------------------------------
# Scenario group construction
# ---------------------------------------------------------------------------

def build_scenario_groups(
    base_X: np.ndarray, perturb_idx: list[int], G: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand base rows into G-variant groups.

    Variant 0 = unperturbed base row. Variants 1..G-1 multiply the k perturbed
    features by log-uniform multipliers in [0.5, 2.0] (2**Uniform(-1, 1)).
    Multipliers are drawn with a fresh ``default_rng(42)`` per cell so every
    cell is independently reproducible. All non-perturbed features are
    bit-identical across the group.

    Returns (grouped_X [N_BASE*G, 23], group_offsets [N_BASE+1] uint64).

    Note: a perturbed feature whose base value is 0 stays 0 across variants
    (multiplicative on 0); TreeWalker correctly treats it as constant for that
    group. Recorded in the config for transparency.
    """
    n_base = base_X.shape[0]
    k = len(perturb_idx)
    rng = np.random.default_rng(SEED)
    # Multipliers for variants 1..G-1, shape (n_base, G-1, k).
    mult = np.power(2.0, rng.uniform(-1.0, 1.0, size=(n_base, G - 1, k)))

    out = np.empty((n_base * G, base_X.shape[1]), dtype=np.float64)
    for b in range(n_base):
        base = base_X[b]
        row0 = b * G
        out[row0] = base  # variant 0 = unperturbed
        for v in range(1, G):
            row = out[row0 + v]
            row[:] = base
            for j, fi in enumerate(perturb_idx):
                row[fi] = base[fi] * mult[b, v - 1, j]

    offsets = np.concatenate([[0], np.cumsum([G] * n_base)]).astype(np.uint64)
    return out, offsets


# ---------------------------------------------------------------------------
# Model training + treelite export (shared across all cells)
# ---------------------------------------------------------------------------

def train_and_export(force: bool) -> lgb.Booster:
    """Train the single LightGBM model and export treelite .json/.bin."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    native = MODEL_DIR / "model_native.txt"
    bin_path = MODEL_DIR / "model_treelite.bin"
    json_path = MODEL_DIR / "model_treelite.json"

    if not force and native.exists() and bin_path.exists() and json_path.exists():
        print("  model: exists, skipping train/export", file=sys.stderr)
        return lgb.Booster(model_file=str(native))

    X, y = load_credit_arff()
    print(f"  data: {X.shape[0]} rows, {X.shape[1]} features, "
          f"{y.mean():.1%} positive", file=sys.stderr)
    train_X, train_y, _, _ = split_train_test(X, y)
    print(f"  split: {train_X.shape[0]} train rows", file=sys.stderr)

    print(f"  training LightGBM T={N_TREES} L={MAX_DEPTH} "
          f"num_leaves={NUM_LEAVES}...", file=sys.stderr)
    bst = train_lgb_model(
        train_X, train_y, FEATURE_NAMES,
        n_trees=N_TREES, max_depth=MAX_DEPTH, cat_indices=[],
    )
    bst.save_model(str(native))

    print("  exporting treelite .bin + .json...", file=sys.stderr)
    tl_model = treelite.frontend.load_lightgbm_model(str(native))
    json_path.write_text(tl_model.dump_as_json())
    bin_path.write_bytes(tl_model.serialize_bytes())
    return bst


# ---------------------------------------------------------------------------
# Per-cell artifact generation
# ---------------------------------------------------------------------------

def gtil_reference(tl_model: treelite.Model, grouped_X: np.ndarray) -> np.ndarray:
    """GTIL f64 reference probabilities for the grouped rows.

    Mirrors prepare.py: GTIL evaluates f64 thresholds and applies sigmoid for
    the binary objective, matching TreeWalker exactly within TOL_F64.
    """
    preds = treelite.gtil.predict(tl_model, grouped_X).flatten().astype(np.float64)
    return preds


def any_zero_group_fraction(base_X: np.ndarray, perturb_idx: list[int]) -> float:
    """Fraction of base rows (groups) with ANY selected (perturbed) feature == 0.

    This is a per-GROUP any-zero fraction, not a per-entry fraction: a group
    counts once if at least one of its k selected features is zero on that base
    row. Under multiplicative perturbation such zero entries stay zero across
    all G variants, so d_v is measured on the realized (non-zero) dimensions.
    """
    if not perturb_idx:
        return 0.0
    sub = base_X[:, perturb_idx]
    return float(np.mean(np.any(sub == 0.0, axis=1)))


def write_cell(
    cell_name: str,
    grouped_X: np.ndarray,
    offsets: np.ndarray,
    perturb_idx: list[int],
    G: int,
    k: int,
    any_zero_frac: float,
    tl_model: treelite.Model,
    force: bool,
) -> Path:
    """Write one cell's artifacts and return the cell directory."""
    cell_dir = CELLS_DIR / cell_name
    cell_dir.mkdir(parents=True, exist_ok=True)

    td = cell_dir / "test_data.bin"
    go = cell_dir / "group_offsets.bin"
    wc = cell_dir / "walker_config.json"
    ref = cell_dir / "reference.bin"

    if not force and td.exists() and go.exists() and wc.exists() and ref.exists():
        print(f"    {cell_name}: artifacts exist, skipping", file=sys.stderr)
        return cell_dir

    write_raw_f64(td, grouped_X)
    write_group_offsets(go, offsets)

    config = {
        "n_features": N_FEATURES,
        "max_group_width": G,
        "feature_names": FEATURE_NAMES,
        "varying_features": perturb_idx,
        "mono_inc_features": [],
        "mono_dec_features": [],
        # Provenance (not read by the engine; recorded for the spec).
        "scenario": {
            "k": k,
            "G": G,
            "perturbable_features": perturb_idx,
            "perturbable_feature_names": [FEATURE_NAMES[i] for i in perturb_idx],
            "base_rows": int(offsets.shape[0] - 1),
            # Per-group ANY-zero fraction: share of groups (base rows) with at
            # least one selected feature == 0 (NOT a per-entry zero fraction).
            # Such entries stay zero under multiplicative shocks.
            "any_zero_group_fraction": any_zero_frac,
            "perturbation": "multiplicative log-uniform [0.5, 2.0], seed 42",
        },
    }
    wc.write_text(json.dumps(config, indent=2))

    preds = gtil_reference(tl_model, grouped_X)
    write_raw_f64(ref, preds.reshape(-1, 1))  # u64 n, u64 1, n f64
    return cell_dir


# ---------------------------------------------------------------------------
# Correctness gate via sweep_bench --validate
# ---------------------------------------------------------------------------

def validate_cell(cell_dir: Path, binary: Path) -> tuple[bool, float]:
    """Run sweep_bench --validate: TreeWalker vs GTIL reference at TOL_F64."""
    ref_path = cell_dir / "reference.bin"
    go_path = cell_dir / "group_offsets.bin"
    if not ref_path.exists():
        print(f"    {cell_dir.name}: no reference.bin, cannot validate",
              file=sys.stderr)
        return False, float("inf")
    cmd = [
        str(binary),
        str(MODEL_DIR),
        "--data-dir", str(cell_dir),
        "--group-offsets", str(go_path),
        "--validate", str(ref_path),
        "--tol", str(TOL_F64),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"    {cell_dir.name}: validate timed out", file=sys.stderr)
        return False, float("inf")
    # sweep_bench prints "VALIDATE PASS max_delta=..." or "VALIDATE FAIL ..."
    out = (r.stdout + r.stderr)
    delta = float("inf")
    for line in out.splitlines():
        if "max_delta=" in line:
            try:
                delta = float(line.split("max_delta=")[1].split()[0])
            except (ValueError, IndexError):
                pass
        if "VALIDATE PASS" in line:
            return True, delta
    print(f"    {cell_dir.name}: validate FAIL (rc={r.returncode})\n"
          f"      {out[-400:]}", file=sys.stderr)
    return False, delta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cell_name(k: int, G: int) -> str:
    return f"k{k}_G{G}"


def ordered_cells() -> list[tuple[int, int]]:
    """Core 3x3 {1,4,8}x{4,16,128} first, then the remaining 7 cells."""
    core = [(k, G) for k in CORE_K for G in CORE_G]
    rest = [(k, G) for k in K_VALUES for G in G_VALUES if (k, G) not in set(core)]
    return core + rest


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare E1 scenario-analysis artifacts")
    ap.add_argument("--cell", default=None,
                    help="Prepare only one cell, e.g. k1_G4 (hard-gate cell)")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip sweep_bench --validate correctness gate")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing artifacts")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, file=sys.stderr)
    print("E1 scenario-analysis: UCI Default of Credit Card Clients", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # 1. Train + export the shared model.
    bst = train_and_export(args.force)
    tl_model = treelite.Model.deserialize(str(MODEL_DIR / "model_treelite.bin"))

    # 2. Nested k feature sets (deterministic given the model).
    pool = ordered_perturbable_pool(bst)
    k_sets = {k: k_set(pool, k) for k in K_VALUES}
    print(f"\nPerturbable pool ordered by gain importance:", file=sys.stderr)
    for k in K_VALUES:
        names = [FEATURE_NAMES[i] for i in k_sets[k]]
        print(f"  k={k}: {k_sets[k]} -> {names}", file=sys.stderr)

    # 3. Base rows from the test split (first N_BASE, deterministic split).
    X, y = load_credit_arff()
    _, _, test_X, _ = split_train_test(X, y)
    base_X = test_X[:N_BASE]
    print(f"\nBase rows: {N_BASE} (of {test_X.shape[0]} test rows)", file=sys.stderr)

    # 4. Build sweep_bench binary (needed for the correctness gate).
    binary = build_sweep_bench() if not args.no_validate else None

    # 5. Select cells.
    if args.cell:
        wanted = [args.cell]
    else:
        wanted = [cell_name(k, G) for (k, G) in ordered_cells()]

    name_to_kg = {cell_name(k, G): (k, G) for k in K_VALUES for G in G_VALUES}
    print(f"\nCells: {wanted}", file=sys.stderr)

    # 6. Generate + validate each cell.
    failures: list[str] = []
    for name in wanted:
        if name not in name_to_kg:
            print(f"  unknown cell {name}, skipping", file=sys.stderr)
            continue
        k, G = name_to_kg[name]
        perturb_idx = k_sets[k]
        print(f"\n  {name} (k={k}, G={G}):", file=sys.stderr)
        grouped_X, offsets = build_scenario_groups(base_X, perturb_idx, G)
        any_zero_frac = any_zero_group_fraction(base_X, perturb_idx)
        print(f"    grouped: {grouped_X.shape[0]} rows, {len(offsets)-1} groups, "
              f"varying={perturb_idx}, any_zero_group_frac={any_zero_frac:.3f}", file=sys.stderr)
        cell_dir = write_cell(name, grouped_X, offsets, perturb_idx, G, k,
                              any_zero_frac, tl_model, args.force)

        if args.no_validate:
            continue
        ok, delta = validate_cell(cell_dir, binary)
        if ok:
            print(f"    VALIDATE PASS max_delta={delta:.2e} (tol={TOL_F64:.0e})",
                  file=sys.stderr)
        else:
            failures.append(name)
            print(f"    VALIDATE FAIL {name} max_delta={delta:.2e}", file=sys.stderr)
            # Hard gate: a single --cell failure aborts.
            if args.cell:
                sys.exit(1)

    print(f"\nDone. Artifacts under {OUT_ROOT}", file=sys.stderr)
    if failures:
        print(f"  {len(failures)} cell(s) FAILED correctness: {failures}",
              file=sys.stderr)
        sys.exit(1)
    print("  all cells validated within TOL_F64", file=sys.stderr)


if __name__ == "__main__":
    main()
