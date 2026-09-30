#!/usr/bin/env python3
"""Prepare all artifacts for the TreeWalker benchmark sweep.

Supports both survival datasets (discrete hazard expansion with fixed panels)
and CTR datasets (session-based with variable-length groups). For each dataset
and parameter combo, trains LightGBM and XGBoost models, exports treelite JSON
and binary, and writes reference predictions.

Granular idempotency: each artifact is checked independently. If a model exists
but treelite .bin is missing, only the .bin is regenerated — nothing is retrained.

Usage:
    uv run python3 paper/experiments/scripts/prepare.py
    uv run python3 paper/experiments/scripts/prepare.py --datasets support
    uv run python3 paper/experiments/scripts/prepare.py --datasets expedia --prepare-groups
    uv run python3 paper/experiments/scripts/prepare.py --grid all --prepare-groups
    uv run python3 paper/experiments/scripts/prepare.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product as iprod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    ABLATION_ANCHORS_B,
    ARTIFACTS_DIR,
    DATA_DIR,
    DATASETS,
    DEFAULT_HORIZON,
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_TREES,
    GRID1_MAX_DEPTH,
    GRID1_N_TREES,
    GROUP_DISTRIBUTIONS,
    DatasetSpec,
    PreparedData,
    build_cat_encoders,
    build_sweep_bench,
    build_walker_config,
    compile_lleaves,
    compile_tl2cgen,
    compute_bin_edges,
    discretize_test,
    discretize_train,
    factorial_param_combos,
    load_pycox_dataset,
    read_group_offsets,
    split_raw_patients,
    train_lgb_model,
    train_xgb_model,
    unique_param_combos,
    write_group_offsets,
    write_raw_f64,
    xgb_feature_types,
)


# ---------------------------------------------------------------------------
# CTR-specific: session loading and expansion
# ---------------------------------------------------------------------------

LABEL_COL = "click_bool"


def load_ctr_data(
    spec: DatasetSpec,
    max_sessions: int | None,
    seed: int,
    test_frac: float,
    n_sample_rows: int | None = None,
) -> tuple:
    """Load CTR parquet, filter sessions, split train/test by session.

    Returns (train_df, test_df, session_col) as polars DataFrames sorted by
    session id. train_df/test_df contain ALL_FEATURES + LABEL_COL columns.
    """
    import polars as pl

    data_path = Path(spec.parquet_path)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent.parent.parent / data_path

    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    constant_features = spec.constant_features or []
    varying_features = spec.varying_features or []
    all_features = constant_features + varying_features

    print(f"  Loading {data_path.name}...", file=sys.stderr)
    df = pl.read_parquet(data_path)
    if n_sample_rows:
        df = df.head(n_sample_rows)
    print(f"  Raw: {len(df):,} rows, {len(df.columns)} columns", file=sys.stderr)

    # Filter to searches with 2-128 properties (u128 row mask limit).
    session_sizes = df.group_by("srch_id").agg(pl.len().alias("_n"))
    valid = session_sizes.filter(
        (pl.col("_n") >= 2) & (pl.col("_n") <= 128)
    )
    print(f"  Sessions 2-128 rows: {len(valid):,} of {len(session_sizes):,} "
          f"({100 * len(valid) / len(session_sizes):.1f}%)",
          file=sys.stderr)

    # Filter sessions where constant features actually vary.
    df_valid = df.join(valid.select("srch_id"), on="srch_id")
    n_before = df_valid["srch_id"].n_unique()
    for col in constant_features:
        varying_sids = df_valid.group_by("srch_id").agg(
            pl.col(col).drop_nulls().n_unique().alias("_nuniq")
        ).filter(pl.col("_nuniq") > 1).select("srch_id")
        if len(varying_sids) > 0:
            n_dropped = len(varying_sids)
            df_valid = df_valid.join(varying_sids, on="srch_id", how="anti")
            print(f"    Dropped {n_dropped} sessions where {col} varies",
                  file=sys.stderr)

    valid = df_valid.group_by("srch_id").agg(pl.len().alias("_n")).select("srch_id")
    df = df.join(valid, on="srch_id")
    n_after = len(valid)
    print(f"  After constancy filter: {n_after:,} sessions "
          f"(dropped {n_before - n_after})", file=sys.stderr)

    # Subsample sessions if requested.
    rng = np.random.default_rng(seed)
    unique_sids = df["srch_id"].unique().sort().to_list()
    n_sessions = len(unique_sids)
    if max_sessions and n_sessions > max_sessions:
        idx = rng.choice(n_sessions, max_sessions, replace=False)
        keep_sids = [unique_sids[i] for i in idx]
        df = df.filter(pl.col("srch_id").is_in(keep_sids))
        unique_sids = df["srch_id"].unique().sort().to_list()
        n_sessions = len(unique_sids)
        print(f"  Subsampled to {n_sessions:,} sessions", file=sys.stderr)

    df = df.sort("srch_id")

    # Train/test split by session.
    n_test = max(1, int(n_sessions * test_frac))
    perm = rng.permutation(n_sessions)
    test_set = set(perm[:n_test].tolist())
    sid_to_idx = dict(zip(unique_sids, range(n_sessions)))
    df = df.with_columns(
        pl.col("srch_id").replace_strict(sid_to_idx, return_dtype=pl.UInt32).alias("_sidx")
    )
    is_test = df["_sidx"].is_in(list(test_set))
    train_df = df.filter(~is_test).sort("srch_id")
    test_df = df.filter(is_test).sort("srch_id")

    n_train_sessions = n_sessions - n_test
    print(f"  Split: {n_train_sessions:,} train, {n_test:,} test sessions",
          file=sys.stderr)
    print(f"  Features: {len(all_features)} ({len(constant_features)} constant "
          f"[{len(constant_features)/len(all_features):.0%}], "
          f"{len(varying_features)} varying)", file=sys.stderr)

    return train_df, test_df


def expand_ctr(
    train_df,
    test_df,
    spec: DatasetSpec,
    n_trees: int,
    max_depth: int,
) -> PreparedData:
    """Build feature matrices and group_offsets from CTR session DataFrames.

    Returns PreparedData with variable-length groups (group_offsets != None).
    """
    constant_features = spec.constant_features or []
    varying_features = spec.varying_features or []
    all_features = constant_features + varying_features
    n_features = len(all_features)
    varying_indices = list(range(len(constant_features), n_features))

    train_X = train_df.select(all_features).to_numpy().astype(np.float64)
    train_y = train_df[LABEL_COL].to_numpy().astype(np.float64)
    test_X = test_df.select(all_features).to_numpy().astype(np.float64)

    # Build group offsets for test set.
    test_sids = test_df["srch_id"].to_numpy()
    changes = np.where(test_sids[:-1] != test_sids[1:])[0] + 1
    test_offsets = np.concatenate([[0], changes, [len(test_sids)]]).astype(np.uint64)
    n_test_sessions = len(test_offsets) - 1
    max_panel = int(np.diff(test_offsets).max())

    config = {
        "n_features": n_features,
        "max_group_width": max_panel,
        "feature_names": all_features,
        "varying_features": varying_indices,
        "mono_inc_features": [],
        "mono_dec_features": [],
    }

    print(f"    data: {train_X.shape[0]:,} train rows, {test_X.shape[0]:,} test rows "
          f"({n_test_sessions:,} sessions, max_panel={max_panel}), "
          f"{n_features} features ({len(constant_features)/n_features:.0%} const)",
          file=sys.stderr)

    return PreparedData(
        train_X=train_X,
        train_y=train_y,
        test_X=test_X,
        feature_names=all_features,
        config=config,
        cat_indices=[],
        group_offsets=test_offsets,
        n_obs=n_test_sessions,
        horizon=1,
    )


# ---------------------------------------------------------------------------
# Survival-specific: load + expand
# ---------------------------------------------------------------------------

def expand_survival(
    train_df,
    test_df,
    spec: DatasetSpec,
    cat_encoders: dict[str, dict],
    n_trees: int,
    max_depth: int,
    horizon: int,
) -> PreparedData:
    """Discretize survival data into fixed-panel expanded matrices.

    Returns PreparedData with group_offsets=None (fixed panel_length = horizon).
    """
    duration = train_df[spec.duration_col].to_numpy().astype(float)
    event = train_df[spec.event_col].to_numpy().astype(float)
    bin_edges = compute_bin_edges(duration, event, horizon)
    actual_horizon = len(bin_edges) - 1

    train_X, train_y, feature_names, info = discretize_train(
        train_df, spec, actual_horizon, bin_edges, cat_encoders,
    )
    test_X = discretize_test(test_df, spec, actual_horizon, bin_edges, cat_encoders)
    n_test_obs = test_X.shape[0] // actual_horizon
    cat_indices = info.get("cat_indices", [])

    config = build_walker_config(feature_names, info)

    print(f"    data: {train_X.shape[0]} train rows, {test_X.shape[0]} test rows "
          f"({n_test_obs} obs x {actual_horizon} steps), "
          f"{info['n_total_features']} features ({info['constant_frac']:.0%} const)",
          file=sys.stderr)

    return PreparedData(
        train_X=train_X,
        train_y=train_y,
        test_X=test_X,
        feature_names=feature_names,
        config=config,
        cat_indices=cat_indices,
        group_offsets=None,
        n_obs=n_test_obs,
        horizon=actual_horizon,
    )


# ---------------------------------------------------------------------------
# Unified combo pipeline (granular idempotency)
# ---------------------------------------------------------------------------

def prepare_combo(
    data: PreparedData,
    n_trees: int,
    max_depth: int,
    out_dir: Path,
    force: bool = False,
    skip_compiled: bool = True,
) -> None:
    """Prepare artifacts for one (dataset, param combo). Each artifact checked independently."""
    import treelite

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data files ---
    if force or not (out_dir / "test_data.bin").exists():
        write_raw_f64(out_dir / "test_data.bin", data.test_X)
    if force or not (out_dir / "train_data.bin").exists():
        train_with_label = np.column_stack([data.train_X, data.train_y])
        write_raw_f64(out_dir / "train_data.bin", train_with_label)
    if force or not (out_dir / "walker_config.json").exists():
        with open(out_dir / "walker_config.json", "w") as f:
            json.dump(data.config, f, indent=2)
    if data.group_offsets is not None:
        if force or not (out_dir / "group_offsets.bin").exists():
            write_group_offsets(out_dir / "group_offsets.bin", data.group_offsets)

    feature_names = data.feature_names
    cat_indices = data.cat_indices
    n_cat = len(cat_indices)

    # --- Per-framework: train, export, predict ---
    for fw in ["lightgbm", "xgboost"]:
        fw_dir = out_dir / fw
        fw_dir.mkdir(exist_ok=True)
        native = fw_dir / ("model_native.txt" if fw == "lightgbm" else "model_native.json")

        # Train (skip if native model exists)
        if force or not native.exists():
            _train_model(fw, data, n_trees, max_depth, fw_dir)

        if not native.exists():
            print(f"    {fw}: native model missing after train, skipping export",
                  file=sys.stderr)
            continue

        # Treelite export (skip individually)
        json_path = fw_dir / "model_treelite.json"
        bin_path = fw_dir / "model_treelite.bin"
        if force or not json_path.exists() or not bin_path.exists():
            _export_treelite(fw, native, json_path, bin_path)

        # Reference predictions
        pred_path = fw_dir / "predictions.npy"
        if force or not pred_path.exists():
            _generate_predictions(fw, native, data, pred_path)

        print(f"    {fw}: {n_trees} trees, {n_cat} cat features", file=sys.stderr)


def _train_model(
    fw: str,
    data: PreparedData,
    n_trees: int,
    max_depth: int,
    fw_dir: Path,
) -> None:
    """Train and save native model."""
    if fw == "lightgbm":
        bst = train_lgb_model(
            data.train_X, data.train_y, data.feature_names,
            n_trees, max_depth, data.cat_indices,
        )
        bst.save_model(str(fw_dir / "model_native.txt"))
    else:
        bst = train_xgb_model(
            data.train_X, data.train_y, data.feature_names,
            n_trees, max_depth, data.cat_indices,
        )
        bst.save_model(str(fw_dir / "model_native.json"))


def _export_treelite(
    fw: str,
    native: Path,
    json_path: Path,
    bin_path: Path,
) -> None:
    """Export treelite JSON and binary from a native model."""
    import treelite

    if fw == "lightgbm":
        tl_model = treelite.frontend.load_lightgbm_model(str(native))
    else:
        tl_model = treelite.frontend.load_xgboost_model(str(native))

    json_path.write_text(tl_model.dump_as_json())
    bin_path.write_bytes(tl_model.serialize_bytes())


def _generate_predictions(
    fw: str,
    native: Path,
    data: PreparedData,
    pred_path: Path,
) -> None:
    """Generate reference predictions from a native model."""
    import treelite

    if fw == "lightgbm":
        # LightGBM reference via GTIL (f64 thresholds — exact match with TW).
        tl_model = treelite.frontend.load_lightgbm_model(str(native))
        preds = treelite.gtil.predict(tl_model, data.test_X).flatten().astype(np.float64)
    else:
        # XGBoost reference via native predict (not GTIL).
        # Treelite internally stores XGBoost thresholds as f32 and GTIL evaluates
        # in f32. TreeWalker reads the JSON (f64-promoted thresholds) and evaluates
        # in f64, so GTIL is not a valid f64 reference. Native XGBoost predict is
        # the ground truth for the f32 model.
        import xgboost as xgb
        feature_types = xgb_feature_types(data.feature_names, data.cat_indices)
        dmat = xgb.DMatrix(
            data.test_X,
            feature_names=data.feature_names,
            feature_types=feature_types,
            enable_categorical=bool(data.cat_indices),
        )
        bst = xgb.Booster()
        bst.load_model(str(native))
        preds = bst.predict(dmat).astype(np.float64)

    np.save(pred_path, preds)


# ---------------------------------------------------------------------------
# Grid 4: Group distribution generation (CTR only)
# ---------------------------------------------------------------------------

def generate_group_offsets(
    base_offsets: np.ndarray,
    dist_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Generate group offsets for a specific distribution from existing test data.

    base_offsets: the empirical cumulative offsets [0, end0, end1, ...]
    dist_name: one of GROUP_DISTRIBUTIONS
    Returns (new_offsets, selected_session_indices) or None if unsatisfiable.
    """
    rng = np.random.default_rng(seed)
    sizes = np.diff(base_offsets).astype(int)

    if dist_name == "empirical":
        return base_offsets.copy(), np.arange(len(sizes))

    if dist_name.startswith("fixed"):
        target_size = int(dist_name[5:])
        mask = sizes == target_size
        if mask.sum() == 0:
            print(f"    {dist_name}: no sessions of size {target_size}", file=sys.stderr)
            return None
        selected = np.where(mask)[0]
        new_sizes = sizes[selected]
        new_offsets = np.concatenate([[0], np.cumsum(new_sizes)]).astype(np.uint64)
        return new_offsets, selected

    if dist_name.startswith("geom"):
        mean_k = int(dist_name[4:])
        p = 1.0 / mean_k
        n_target = len(sizes)
        target_sizes = np.clip(rng.geometric(p, size=n_target), 2, 32)
        size_to_indices: dict[int, list[int]] = {}
        for i, s in enumerate(sizes):
            size_to_indices.setdefault(int(s), []).append(i)

        selected = []
        for ts in target_sizes:
            ts = int(ts)
            best = None
            for delta in range(0, 31):
                for candidate in [ts + delta, ts - delta]:
                    if 1 <= candidate <= 32 and candidate in size_to_indices and size_to_indices[candidate]:
                        best = candidate
                        break
                if best is not None:
                    break
            if best is None:
                continue
            idx = rng.choice(size_to_indices[best])
            selected.append(idx)

        if not selected:
            return None
        selected = list(dict.fromkeys(selected))  # order-preserving dedup
        selected = np.array(selected)
        new_sizes = sizes[selected]
        new_offsets = np.concatenate([[0], np.cumsum(new_sizes)]).astype(np.uint64)
        return new_offsets, selected

    if dist_name == "bimodal2_16":
        idx_2 = np.where(sizes == 2)[0]
        idx_16 = np.where(sizes == 16)[0]
        if len(idx_2) == 0 or len(idx_16) == 0:
            print(f"    bimodal2_16: insufficient sessions (size-2={len(idx_2)}, "
                  f"size-16={len(idx_16)})", file=sys.stderr)
            return None
        n_each = min(len(idx_2), len(idx_16))
        selected = np.concatenate([
            rng.choice(idx_2, n_each, replace=False),
            rng.choice(idx_16, n_each, replace=False),
        ])
        rng.shuffle(selected)
        new_sizes = sizes[selected]
        new_offsets = np.concatenate([[0], np.cumsum(new_sizes)]).astype(np.uint64)
        return new_offsets, selected

    print(f"    Unknown distribution: {dist_name}", file=sys.stderr)
    return None


def _load_test_data_raw(path: Path) -> np.ndarray:
    """Load test_data.bin as a 2D numpy array."""
    raw = path.read_bytes()
    n_rows = int.from_bytes(raw[:8], "little")
    n_cols = int.from_bytes(raw[8:16], "little")
    return np.frombuffer(raw[16:], dtype="<f8").reshape(n_rows, n_cols)


def prepare_group_distributions(
    base_dir: Path,
    seed: int,
    force: bool = False,
) -> None:
    """Generate per-distribution subdirectories with matching test data and offsets.

    Each non-empirical distribution selects a subset of sessions from the base
    test data. We write a self-contained subdirectory per distribution:
        base_dir/{dist}/test_data.bin       -- row subset matching selected sessions
        base_dir/{dist}/group_offsets.bin   -- cumulative offsets into the subset
        base_dir/{dist}/walker_config.json  -- copy from parent (max_group_width updated)
    The empirical distribution reuses the parent directory directly.
    """
    base_offsets_path = base_dir / "group_offsets.bin"
    if not base_offsets_path.exists():
        print(f"  No group_offsets.bin in {base_dir}, skipping group generation",
              file=sys.stderr)
        return

    base_offsets = read_group_offsets(base_offsets_path)
    test_data = _load_test_data_raw(base_dir / "test_data.bin")
    base_config_path = base_dir / "walker_config.json"

    print(f"  Generating group distributions from {len(base_offsets) - 1} sessions:",
          file=sys.stderr)

    for dist in GROUP_DISTRIBUTIONS:
        dist_dir = base_dir / dist
        if not force and (dist_dir / "group_offsets.bin").exists():
            print(f"    {dist}: exists, skipping", file=sys.stderr)
            continue

        result = generate_group_offsets(base_offsets, dist, seed)
        if result is None:
            continue
        new_offsets, selected_sessions = result

        # Extract rows belonging to the selected sessions from the original data.
        row_indices = []
        for sid in selected_sessions:
            start = int(base_offsets[sid])
            end = int(base_offsets[sid + 1])
            row_indices.extend(range(start, end))
        subset_data = test_data[row_indices]

        # Compute max panel length for the subset config.
        new_sizes = np.diff(new_offsets).astype(int)
        max_panel = int(new_sizes.max()) if len(new_sizes) > 0 else 1

        # Write self-contained distribution directory.
        dist_dir.mkdir(parents=True, exist_ok=True)
        write_raw_f64(dist_dir / "test_data.bin", subset_data)
        write_group_offsets(dist_dir / "group_offsets.bin", new_offsets)

        # Write walker_config with updated max_group_width.
        with open(base_config_path) as f:
            config = json.load(f)
        config["max_group_width"] = max_panel
        with open(dist_dir / "walker_config.json", "w") as f:
            json.dump(config, f, indent=2)

        n_groups = len(new_offsets) - 1
        mean_size = float(new_sizes.mean()) if n_groups > 0 else 0
        print(f"    {dist}: {n_groups} groups, {subset_data.shape[0]} rows, "
              f"mean_size={mean_size:.1f}, max_panel={max_panel}",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _ctr_factorial_combos() -> list[tuple[int, int]]:
    """All T x L combos for CTR Grid 1 cells (no horizon axis)."""
    return sorted(iprod(GRID1_N_TREES, GRID1_MAX_DEPTH))


def _combos_for_spec(
    spec: DatasetSpec,
    grid: str,
    explicit_combos: list[tuple] | None,
) -> list[tuple]:
    """Return parameter combos for a dataset. Survival combos are (T, L, H);
    CTR combos are (T, L). When explicit_combos is set, use those directly."""
    if explicit_combos is not None:
        if spec.dataset_type == "ctr":
            # Strip horizon from explicit combos for CTR
            return sorted({(nt, md) for nt, md, *_ in explicit_combos})
        return explicit_combos

    if spec.dataset_type == "ctr":
        if grid == "factorial" or grid == "all":
            return _ctr_factorial_combos()
        return [(DEFAULT_N_TREES, DEFAULT_MAX_DEPTH)]

    # Survival
    if grid == "all":
        return sorted(set(factorial_param_combos()) | set(ABLATION_ANCHORS_B))
    if grid == "factorial":
        return factorial_param_combos()
    return unique_param_combos()


def _parse_combos_arg(combos_str: str) -> list[tuple[int, ...]]:
    """Parse 'T,L,H;T,L,H;...' or 'T,L;T,L;...' into list of tuples."""
    result = []
    for part in combos_str.split(";"):
        part = part.strip()
        if not part:
            continue
        vals = tuple(int(x.strip()) for x in part.split(","))
        if len(vals) not in (2, 3):
            raise ValueError(f"Expected T,L or T,L,H but got {part!r}")
        result.append(vals)
    return result


def _compile_one(task: tuple) -> str:
    """Compile a single tl2cgen or lleaves .so (picklable for ProcessPoolExecutor)."""
    model, so, fw, tool, nthread = task  # fw is vestigial for tl2cgen after Model.deserialize migration
    if tool == "tl2cgen":
        ok = compile_tl2cgen(model, so, nthread=nthread)
    else:
        ok = compile_lleaves(model, so, nthread=nthread)
    return f"{'ok' if ok else 'FAIL'}: {so.relative_to(ARTIFACTS_DIR)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark artifacts")
    parser.add_argument("--dry-run", action="store_true",
                        help="One dataset, one param combo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Filter by dataset name")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing artifacts")
    parser.add_argument("--grid", choices=["ofat", "factorial", "all"], default="ofat",
                        help="Parameter grid: ofat (13), factorial (96/16), "
                             "all (factorial + ablation anchors)")
    parser.add_argument("--combos", type=str, default=None,
                        help="Specific combos: 'T,L,H;T,L,H;...' (overrides --grid)")
    parser.add_argument("--prepare-groups", action="store_true",
                        help="Generate Grid 4 group distributions (CTR datasets only)")
    parser.add_argument("--skip-compiled", action="store_true",
                        help="Skip tl2cgen/lleaves compilation")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only build sweep_bench and compile tl2cgen/lleaves (skip training)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel compile workers (default: min(ncpu//2, 4))")
    parser.add_argument("--max-sessions", type=int, default=50_000,
                        help="Max sessions for CTR datasets (default: 50000)")
    args = parser.parse_args()

    explicit_combos = _parse_combos_arg(args.combos) if args.combos else None

    if args.dry_run:
        specs = DATASETS[:1]
    else:
        specs = [ds for ds in DATASETS if not args.datasets or ds.name in args.datasets]

    # Print summary
    total_combos = 0
    for spec in specs:
        combos = _combos_for_spec(spec, args.grid, explicit_combos)
        if args.dry_run:
            combos = combos[:1]
        total_combos += len(combos)
    print(f"Preparing {len(specs)} datasets, {total_combos} total artifact sets "
          f"(grid={args.grid})", file=sys.stderr)

    for spec in specs:
        if args.compile_only:
            break
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Dataset: {spec.name} ({spec.dataset_type})", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        combos = _combos_for_spec(spec, args.grid, explicit_combos)
        if args.dry_run:
            combos = combos[:1]

        # Deterministic per-dataset seed
        name_hash = int.from_bytes(spec.name.encode(), "big") % (2**31)
        dataset_seed = args.seed + name_hash

        if spec.dataset_type == "ctr":
            # CTR: load sessions once, expand per combo
            n_sample_rows = 500_000 if args.dry_run else None
            max_sessions = 2_000 if args.dry_run else args.max_sessions
            train_df, test_df = load_ctr_data(
                spec, max_sessions, dataset_seed, args.test_frac, n_sample_rows,
            )

            for nt, md in combos:
                dir_name = spec.param_dir_name(nt, md, 0)
                out_dir = ARTIFACTS_DIR / spec.name / dir_name

                if not args.force and _all_artifacts_exist(out_dir, has_groups=True):
                    print(f"  {dir_name}: all artifacts exist, skipping", file=sys.stderr)
                else:
                    print(f"\n  {dir_name} (trees={nt}, depth={md}):", file=sys.stderr)
                    try:
                        data = expand_ctr(train_df, test_df, spec, nt, md)
                        prepare_combo(data, nt, md, out_dir, args.force,
                                      args.skip_compiled)
                    except Exception as e:
                        print(f"    FAILED: {e}", file=sys.stderr)
                        continue

                # Generate group distributions if requested
                if args.prepare_groups:
                    prepare_group_distributions(out_dir, dataset_seed, args.force)

        else:
            # Survival: load patients, split once, expand per combo
            df = load_pycox_dataset(spec)
            print(f"  Raw: {df.shape[0]} patients, {df.shape[1]} columns",
                  file=sys.stderr)

            dataset_rng = np.random.default_rng(dataset_seed)
            train_df, test_df = split_raw_patients(df, args.test_frac, dataset_rng)
            cat_encoders = build_cat_encoders(train_df, spec)
            print(f"  Split: {len(train_df)} train, {len(test_df)} test patients",
                  file=sys.stderr)

            for nt, md, h in combos:
                dir_name = spec.param_dir_name(nt, md, h)
                out_dir = ARTIFACTS_DIR / spec.name / dir_name

                if not args.force and _all_artifacts_exist(out_dir, has_groups=False):
                    print(f"  {dir_name}: all artifacts exist, skipping",
                          file=sys.stderr)
                    continue

                print(f"\n  {dir_name} (trees={nt}, depth={md}, horizon={h}):",
                      file=sys.stderr)
                try:
                    data = expand_survival(
                        train_df, test_df, spec, cat_encoders, nt, md, h,
                    )
                    prepare_combo(data, nt, md, out_dir, args.force,
                                  args.skip_compiled)
                except Exception as e:
                    print(f"    FAILED: {e}", file=sys.stderr)

    # --- Build sweep_bench binary (Rust, target-cpu=native) ---
    build_sweep_bench()

    # --- Compile tl2cgen & lleaves for all param combos ---
    if not args.skip_compiled:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        compile_tasks = []  # (model, so, framework, tool, nthread)
        ncpu = os.cpu_count() or 8
        n_workers = args.workers or min(ncpu // 2, 4)
        threads_per_worker = max(1, ncpu // n_workers)

        for spec in specs:
            combos = _combos_for_spec(spec, args.grid, explicit_combos)
            if args.dry_run:
                combos = combos[:1]
            for combo in combos:
                if spec.dataset_type == "ctr":
                    nt, md = combo
                    dir_name = spec.param_dir_name(nt, md, 0)
                else:
                    nt, md, h = combo
                    dir_name = spec.param_dir_name(nt, md, h)
                combo_dir = ARTIFACTS_DIR / spec.name / dir_name
                lgb_dir = combo_dir / "lightgbm"
                xgb_dir = combo_dir / "xgboost"
                if lgb_dir.exists() and (lgb_dir / "model_native.txt").exists():
                    compile_tasks.append((lgb_dir / "model_treelite.bin", lgb_dir / "tl2cgen.so", "lightgbm", "tl2cgen", threads_per_worker))
                    compile_tasks.append((lgb_dir / "model_native.txt", lgb_dir / "lleaves.so", "lightgbm", "lleaves", threads_per_worker))
                if xgb_dir.exists() and (xgb_dir / "model_native.json").exists():
                    compile_tasks.append((xgb_dir / "model_treelite.bin", xgb_dir / "tl2cgen.so", "xgboost", "tl2cgen", threads_per_worker))

        # Filter out already-compiled (idempotent skip)
        pending = [t for t in compile_tasks if not t[1].exists()]
        print(f"\nCompiling tl2cgen/lleaves: {len(pending)} pending, "
              f"{len(compile_tasks) - len(pending)} already done "
              f"({n_workers} workers × {threads_per_worker} threads)",
              file=sys.stderr)

        n_workers = min(n_workers, len(pending)) or 1
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_compile_one, t): t for t in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    result = fut.result()
                except Exception as e:
                    task = futures[fut]
                    result = f"FAIL: {task[1].relative_to(ARTIFACTS_DIR)}: {e}"
                print(f"  [{i}/{len(pending)}] {result}", file=sys.stderr)
    else:
        print("\nSkipping tl2cgen/lleaves compilation (--skip-compiled)", file=sys.stderr)

    print(f"\nArtifacts written to {ARTIFACTS_DIR}", file=sys.stderr)
    if args.dry_run:
        print("Dry run complete.", file=sys.stderr)


def _all_artifacts_exist(out_dir: Path, has_groups: bool) -> bool:
    """Quick check: do the core artifacts for a combo exist?

    This is a fast-path skip — prepare_combo does its own granular checks.
    Returns True only if ALL expected artifacts are present.
    """
    if not (out_dir / "walker_config.json").exists():
        return False
    if not (out_dir / "test_data.bin").exists():
        return False
    if has_groups and not (out_dir / "group_offsets.bin").exists():
        return False
    for fw, native_name in [("lightgbm", "model_native.txt"),
                            ("xgboost", "model_native.json")]:
        fw_dir = out_dir / fw
        if not (fw_dir / native_name).exists():
            return False
        if not (fw_dir / "model_treelite.json").exists():
            return False
        if not (fw_dir / "model_treelite.bin").exists():
            return False
        if not (fw_dir / "predictions.npy").exists():
            return False
    return True


if __name__ == "__main__":
    main()
