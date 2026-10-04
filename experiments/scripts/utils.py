"""Shared utilities for TreeWalker benchmarking pipeline.

Provides: dataset loading, discrete hazard expansion, train/test splitting,
model training, artifact I/O, and inference timing functions.
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import contextlib
from dataclasses import replace

import lightgbm as lgb
import numpy as np
import polars as pl


@contextlib.contextmanager
def _suppress_fd_stderr():
    """Suppress C-level stderr (fd 2) writes from native libraries."""
    sys.stderr.flush()
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)

# Thread-pinning env vars — set these before importing LightGBM/XGBoost
# to enforce single-threaded execution during benchmarking.
SINGLE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def pin_threads_single() -> None:
    """Force all numerical backends to single-threaded mode (for benchmarking)."""
    for var in SINGLE_THREAD_ENV_VARS:
        os.environ[var] = "1"

# ---------------------------------------------------------------------------
# Incremental CSV (crash-safe per-cell persistence)
# ---------------------------------------------------------------------------

class IncrementalCSV:
    """Crash-safe incremental CSV writer with dedup-on-resume.

    After each cell, rewrites the entire CSV atomically via write-to-temp +
    os.replace. On restart, loads existing rows and skips completed cells.
    """

    def __init__(self, path: Path, key_cols: list[str]) -> None:
        self.path = path
        self.key_cols = key_cols
        self.rows: list[dict] = []
        self._completed: set[tuple] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        import polars as pl
        try:
            df = pl.read_csv(self.path)
        except (pl.exceptions.ComputeError, pl.exceptions.NoDataError,
                UnicodeDecodeError, ValueError):
            return  # corrupted or empty — start fresh
        if not all(c in df.columns for c in self.key_cols):
            backup = self.path.with_suffix(".csv.bak")
            self.path.rename(backup)
            print(f"  Incompatible CSV, backed up to {backup.name}", file=sys.stderr)
            return
        self.rows = df.to_dicts()
        for row in self.rows:
            self._completed.add(self._key(row))
        print(f"  Resumed {len(self.rows)} existing rows from {self.path.name}",
              file=sys.stderr)

    def _key(self, row: dict) -> tuple:
        return tuple(row.get(c) for c in self.key_cols)

    def is_done(self, **kv: object) -> bool:
        return tuple(kv.get(c) for c in self.key_cols) in self._completed

    def append(self, row: dict) -> None:
        self.rows.append(row)
        self._completed.add(self._key(row))
        self._flush()

    def extend(self, rows: list[dict]) -> None:
        for row in rows:
            self.rows.append(row)
            self._completed.add(self._key(row))
        self._flush()

    def _flush(self) -> None:
        if not self.rows:
            return
        import polars as pl
        import tempfile
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(
                suffix=".csv.tmp", dir=self.path.parent, prefix=f".{self.path.stem}_"
            )
            os.close(fd)
            pl.DataFrame(self.rows).write_csv(tmp)
            os.replace(tmp, self.path)
        except Exception:
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)
            raise

    def __len__(self) -> int:
        return len(self.rows)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent
DATA_DIR = RUN_DIR / "data"
ARTIFACTS_DIR = RUN_DIR / "artifacts"
PROJECT_ROOT = RUN_DIR.parent

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

DEFAULT_N_TREES = 500
DEFAULT_MAX_DEPTH = 8
DEFAULT_HORIZON = 16

N_TREES_SWEEP = [50, 100, 250, 500, 750, 1000]
MAX_DEPTH_SWEEP = [4, 6, 8, 10]
HORIZON_SWEEP = [1, 2, 4, 8, 16]

# ---------------------------------------------------------------------------
# Grid 1: Full factorial
# ---------------------------------------------------------------------------

GRID1_N_TREES = [50, 500, 1000, 2000]
GRID1_MAX_DEPTH = [2, 4, 8, 16]
GRID1_HORIZON = [1, 2, 4, 8, 16, 32, 64, 128]

# ---------------------------------------------------------------------------
# Grid 3: Ablation anchors
# ---------------------------------------------------------------------------

# Grid 3 anchors: chosen from the Grid 1 (T, L, H) lattice so every anchor
# also has Grid 1 coverage.  B' anchors get the full 2^6 = 64-combo cross;
# the rest use within-group powerset (15 combos).
#
# Archetype mapping (from benchmark_plan.md):
#   boundary / low-amortization:  (50, 4, 1)
#   small-group / medium-model:   (500, 8, 4)
#   reference:                    (500, 8, 16)
#   large / deep:                 (1000, 16, 32)
#   optional shallow-large-G:     (500, 4, 16)
ABLATION_ANCHORS_B = [
    (50, 4, 1),      # boundary
    (500, 8, 4),     # small-group
    (500, 8, 16),    # reference
    (1000, 16, 32),  # large / deep
    (500, 4, 16),    # shallow-large-G
    (500, 8, 64),    # wide-group (u64 mask)
    (500, 8, 128),   # widest-group (u128 mask)
]
ABLATION_ANCHORS_B_PRIME = [
    (500, 8, 16),    # reference — full 64-combo cross
    (1000, 16, 32),  # large / deep — full 64-combo cross
]

RUNTIME_FLAGS = ["precompute", "unsplit", "monotonic"]
PARSE_FLAGS = ["tree_ordering", "prefix_grouping", "bitset_intern"]

# ---------------------------------------------------------------------------
# Grid 4: Group distributions
# ---------------------------------------------------------------------------

# Grid 4 distributions: ~6 diagnostic shapes chosen to separate
# "mean group size matters" from "distribution shape matters too."
#
# From benchmark_plan.md:
#   fixed8, fixed16, fixed32       — mean-only baseline
#   geometric (mean ~8-10)         — right-skewed
#   bimodal matched-mean           — bimodal at same mean as geometric
#   empirical                      — real Expedia distribution
GROUP_DISTRIBUTIONS = [
    "fixed8", "fixed16", "fixed32",
    "geom8", "bimodal2_16", "empirical",
]

# Full set kept for backward compat with any existing artifacts.
GROUP_DISTRIBUTIONS_FULL = [
    "fixed1", "fixed2", "fixed4", "fixed8", "fixed16", "fixed32",
    "geom4", "geom8", "bimodal2_16", "empirical",
]

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    name: str
    duration_col: str = ""
    event_col: str = ""
    dataset_type: str = "survival"          # "survival" or "ctr"
    drop_cols: list[str] | None = None      # columns to remove before training
    cat_cols: list[str] | None = None       # categorical features (passed to LGB/XGB)
    binary_cols: list[str] | None = None    # binary features (treated as categorical)
    # CTR-specific fields
    parquet_path: str = ""
    constant_features: list[str] | None = None
    varying_features: list[str] | None = None

    def param_dir_name(self, nt: int, md: int, h: int) -> str:
        """Directory name for a parameter combo (CTR omits horizon)."""
        if self.dataset_type == "ctr":
            return f"nt{nt}_md{md}"
        return f"nt{nt}_md{md}_h{h}"


@dataclass
class PreparedData:
    """Uniform interface for expanded feature matrices (survival or CTR)."""
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    feature_names: list[str]
    config: dict
    cat_indices: list[int]
    group_offsets: np.ndarray | None
    n_obs: int
    horizon: int


# Feature types from original dataset documentation:
#   SUPPORT:  Knaus et al. 1995 (SUPPORT study), Katzman et al. 2018
#   FLCHAIN:  Dispenzieri et al. 2012, R survival::flchain
DATASETS = [
    DatasetSpec(
        "support", "duration", "event",
        cat_cols=["x2", "x3", "x6"],            # num.co (0-9), race (6), cancer (3)
        binary_cols=["x1", "x4", "x5"],         # sex, diabetes, dementia
    ),
    DatasetSpec(
        "flchain", "duration", "event",
        drop_cols=["rownames"],                  # row ID, no predictive content
        cat_cols=["sample.yr", "flc.grp"],       # enrollment year (9), FLC decile (10)
        binary_cols=["sex", "mgus"],             # sex (M/F), prior MGUS (0/1)
    ),
    DatasetSpec(
        "expedia", dataset_type="ctr",
        parquet_path="experiments/data/expedia.parquet",
        constant_features=[
            "site_id",
            "visitor_location_country_id",
            "prop_country_id",
            "srch_destination_id",
            "srch_length_of_stay",
            "srch_booking_window",
            "srch_adults_count",
            "srch_children_count",
            "srch_room_count",
            "srch_saturday_night_bool",
            "random_bool",
        ],
        varying_features=[
            "prop_id",
            "prop_starrating",
            "prop_review_score",
            "prop_brand_bool",
            "prop_location_score1",
            "prop_location_score2",
            "prop_log_historical_price",
            "position",
            "price_usd",
            "promotion_flag",
        ],
    ),
]


def unique_param_combos() -> list[tuple[int, int, int]]:
    """Return deduplicated (n_trees, max_depth, horizon) tuples for the sweep."""
    combos = set()
    for nt in N_TREES_SWEEP:
        combos.add((nt, DEFAULT_MAX_DEPTH, DEFAULT_HORIZON))
    for md in MAX_DEPTH_SWEEP:
        combos.add((DEFAULT_N_TREES, md, DEFAULT_HORIZON))
    for h in HORIZON_SWEEP:
        combos.add((DEFAULT_N_TREES, DEFAULT_MAX_DEPTH, h))
    return sorted(combos)


def param_dir_name(n_trees: int, max_depth: int, horizon: int) -> str:
    """Directory name for a parameter combo."""
    return f"nt{n_trees}_md{max_depth}_h{horizon}"


def sweep_axis_for(nt: int, md: int, h: int) -> str:
    """Determine which sweep axis a param combo belongs to."""
    axes = []
    if md == DEFAULT_MAX_DEPTH and h == DEFAULT_HORIZON:
        axes.append("n_trees")
    if nt == DEFAULT_N_TREES and h == DEFAULT_HORIZON:
        axes.append("max_depth")
    if nt == DEFAULT_N_TREES and md == DEFAULT_MAX_DEPTH:
        axes.append("horizon")
    return ",".join(axes) if axes else "default"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pycox_dataset(spec: DatasetSpec) -> pl.DataFrame:
    """Load a survival dataset from source URLs (no pycox/torch dependency)."""
    cache_dir = Path(tempfile.gettempdir()) / "treewalker_datasets"
    cache_dir.mkdir(exist_ok=True)

    if spec.name in ("support", "metabric", "gbsg"):
        return _load_deepsurv_dataset(spec.name, cache_dir)
    elif spec.name == "flchain":
        return _load_flchain(cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {spec.name}")


def _load_deepsurv_dataset(name: str, cache_dir: Path) -> pl.DataFrame:
    import h5py

    urls = {
        "support": "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/support/support_train_test.h5",
        "metabric": "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/metabric/metabric_IHC4_clinical_train_test.h5",
        "gbsg": "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/gbsg/gbsg_cancer_train_test.h5",
    }

    h5_path = cache_dir / f"{name}.h5"
    if not h5_path.exists():
        print(f"  Downloading {name} from DeepSurv...", file=sys.stderr)
        import requests
        r = requests.get(urls[name])
        r.raise_for_status()
        h5_path.write_bytes(r.content)

    frames = []
    with h5py.File(h5_path, "r") as f:
        for split in f:
            x = f[split]["x"][:]
            t = f[split]["t"][:]
            e = f[split]["e"][:]
            colnames = [f"x{i}" for i in range(x.shape[1])]
            df_split = pl.DataFrame(
                {col: x[:, i] for i, col in enumerate(colnames)}
            ).with_columns(
                pl.Series("duration", t),
                pl.Series("event", e),
            )
            frames.append(df_split)

    return pl.concat(frames)


def _load_flchain(cache_dir: Path) -> pl.DataFrame:
    csv_path = cache_dir / "flchain.csv"
    if not csv_path.exists():
        print("  Downloading flchain from Rdatasets...", file=sys.stderr)
        import requests
        url = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv"
        r = requests.get(url)
        r.raise_for_status()
        csv_path.write_text(r.text)

    df = pl.read_csv(csv_path)
    df = df.drop(["chapter", "Unnamed: 0"], strict=False)
    df = df.filter(pl.col("creatinine").is_not_null())
    df = df.with_columns(
        pl.when(pl.col("sex") == "M").then(1.0).otherwise(0.0).alias("sex")
    )

    for col in ["sample.yr", "flc.grp"]:
        if col in df.columns:
            uniq = sorted(df[col].drop_nulls().unique().to_list())
            mapping = {v: float(i) for i, v in enumerate(uniq)}
            df = df.with_columns(
                pl.col(col).replace_strict(mapping, default=None).cast(pl.Float64).alias(col)
            )

    # Cast all remaining columns to Float64
    df = df.with_columns(
        pl.col(c).cast(pl.Float64, strict=False) for c in df.columns
    )

    df = df.rename({"futime": "duration", "death": "event"})
    return df


# ---------------------------------------------------------------------------
# Train/test split (on raw patients, BEFORE discretization)
# ---------------------------------------------------------------------------

def split_raw_patients(
    df: pl.DataFrame,
    test_frac: float,
    rng: np.random.Generator,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split raw patient DataFrame into train/test by patient index."""
    n = len(df)
    n_test = max(1, int(n * test_frac))
    perm = rng.permutation(n)
    train_idx = sorted(perm[n_test:])
    test_idx = sorted(perm[:n_test])
    return df[train_idx], df[test_idx]


# ---------------------------------------------------------------------------
# Discrete hazard expansion
# ---------------------------------------------------------------------------

def compute_bin_edges(duration: np.ndarray, event: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based time bin edges from event times."""
    event_times = duration[event > 0]
    if len(event_times) < n_bins:
        edges = np.linspace(duration.min(), duration.max(), n_bins + 1)
    else:
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(event_times, quantiles))
    return edges


def _build_tv_features(t: int, horizon: int, x0: float) -> list[float]:
    """Build the 6 TV features for time step t."""
    h1 = max(horizon - 1, 1)
    time_frac = t / h1
    remaining = h1 - t
    remaining_frac = remaining / h1
    return [
        float(t),                          # time_step (mono inc)
        time_frac ** 2,                    # time_step_sq (mono inc)
        float(remaining),                  # remaining_steps (mono dec)
        remaining_frac ** 2,               # remaining_frac_sq (mono dec)
        x0 * time_frac,                   # interaction_x0_t (non-mono)
        math.sin(math.pi * time_frac),    # hazard_bump (non-mono)
    ]


TV_FEATURE_NAMES = [
    "time_step", "time_step_sq",
    "remaining_steps", "remaining_frac_sq",
    "interaction_x0_t", "hazard_bump",
]

N_TV_FEATURES = len(TV_FEATURE_NAMES)


def _sanitize_feature_name(name: str) -> str:
    return name.replace(".", "_").replace(" ", "_").replace("[", "_").replace("]", "_")


def build_cat_encoders(
    train_df: pl.DataFrame,
    spec: DatasetSpec,
) -> dict[str, dict]:
    """Fit category encoders on the TRAIN set only.

    Returns {col_name: {raw_value: code}} mapping. Applied to both train
    and test via _prepare_covariates. Unseen test categories map to NaN
    (handled by default_left in tree splits).
    """
    cat_set = set(spec.cat_cols or []) | set(spec.binary_cols or [])
    encoders = {}
    for col in train_df.columns:
        if col not in cat_set:
            continue
        uniq = sorted(train_df[col].drop_nulls().unique().to_list())
        encoders[col] = {v: i for i, v in enumerate(uniq)}
    return encoders


def _prepare_covariates(
    df: pl.DataFrame,
    spec: DatasetSpec,
    cat_encoders: dict[str, dict],
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Clean covariates using spec-defined types. Returns (cov_df, covariate_cols, cat_cols).

    Categorical and binary columns use encodings from cat_encoders (fit on full data).
    Continuous columns are float64.
    """
    exclude = {spec.duration_col, spec.event_col}
    if spec.drop_cols:
        exclude.update(spec.drop_cols)
    covariate_cols = [c for c in df.columns if c not in exclude]

    # Only multi-class categoricals are passed as categorical to LGB/XGB.
    # Binary features stay numeric (threshold at 0.5 is simpler than a 1-bit bitset).
    cat_set = set(spec.cat_cols or [])
    cat_cols = [c for c in covariate_cols if c in cat_set]

    cov_df = df.select(covariate_cols)
    exprs = []
    for col in cov_df.columns:
        if col in cat_encoders:
            mapping = cat_encoders[col]
            exprs.append(
                pl.col(col).replace_strict(mapping, default=None).cast(pl.Float64).alias(col)
            )
        else:
            exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    cov_df = cov_df.with_columns(exprs)

    return cov_df, covariate_cols, cat_cols


def discretize_train(
    df: pl.DataFrame,
    spec: DatasetSpec,
    horizon: int,
    bin_edges: np.ndarray,
    cat_encoders: dict[str, dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Expand train patients to variable-length panel (up to event/censor bin).

    Returns X, y, feature_names, info.
    """
    duration = df[spec.duration_col].to_numpy().astype(float)
    event = df[spec.event_col].to_numpy().astype(float)

    if cat_encoders is None:
        cat_encoders = build_cat_encoders(df, spec)
    cov_df, covariate_cols, cat_cols = _prepare_covariates(df, spec, cat_encoders)

    # Filter rows with any null in covariates or duration/event
    any_null_mask = cov_df.select(pl.any_horizontal(pl.all().is_null())).to_series().to_numpy()
    valid_mask = ~any_null_mask & ~np.isnan(duration) & ~np.isnan(event)

    cov_df = cov_df.filter(pl.Series(valid_mask))
    duration = duration[valid_mask]
    event = event[valid_mask]

    n_obs = len(cov_df)
    n_covariates = len(covariate_cols)
    covariates = cov_df.to_numpy().astype(np.float64)

    actual_bins = len(bin_edges) - 1
    bin_idx = np.clip(np.digitize(duration, bin_edges[1:]), 0, actual_bins - 1)

    rows_X, rows_y = [], []
    for i in range(n_obs):
        max_bin = bin_idx[i]
        obs_covs = covariates[i]
        for t in range(max_bin + 1):
            row = list(obs_covs) + _build_tv_features(t, horizon, obs_covs[0])
            rows_X.append(row)
            rows_y.append(1.0 if t == max_bin and event[i] > 0 else 0.0)

    X = np.array(rows_X, dtype=np.float64)
    y = np.array(rows_y, dtype=np.float64)

    feature_names = [_sanitize_feature_name(c) for c in covariate_cols] + TV_FEATURE_NAMES
    n_total = n_covariates + N_TV_FEATURES
    tv_indices = list(range(n_covariates, n_total))
    mono_inc_indices = [n_covariates, n_covariates + 1]         # time_step, time_step_sq
    mono_dec_indices = [n_covariates + 2, n_covariates + 3]     # remaining_steps, remaining_frac_sq
    cat_indices = [i for i, c in enumerate(covariate_cols) if c in cat_cols]

    info = {
        "n_obs": n_obs,
        "n_covariates": n_covariates,
        "n_total_features": n_total,
        "horizon": horizon,
        "n_rows": len(rows_X),
        "event_rate": float(event.mean()),
        "constant_frac": n_covariates / n_total,
        "tv_indices": tv_indices,
        "mono_inc_indices": mono_inc_indices,
        "mono_dec_indices": mono_dec_indices,
        "cat_indices": cat_indices,
        "bin_edges": bin_edges.tolist(),
    }
    return X, y, feature_names, info


def discretize_test(
    df: pl.DataFrame,
    spec: DatasetSpec,
    horizon: int,
    bin_edges: np.ndarray,
    cat_encoders: dict[str, dict] | None = None,
) -> np.ndarray:
    """Expand test patients to full-horizon panel (exactly h rows per patient).

    Returns X only (no labels needed for inference).
    """
    if cat_encoders is None:
        cat_encoders = build_cat_encoders(df, spec)
    cov_df, _, _ = _prepare_covariates(df, spec, cat_encoders)

    any_null_mask = cov_df.select(pl.any_horizontal(pl.all().is_null())).to_series().to_numpy()
    cov_df = cov_df.filter(pl.Series(~any_null_mask))
    covariates = cov_df.to_numpy().astype(np.float64)
    n_obs = len(cov_df)

    rows = []
    for i in range(n_obs):
        obs_covs = covariates[i]
        for t in range(horizon):
            row = list(obs_covs) + _build_tv_features(t, horizon, obs_covs[0])
            rows.append(row)

    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Walker config
# ---------------------------------------------------------------------------

def build_walker_config(feature_names: list[str], info: dict) -> dict:
    """Build walker_config.json dict."""
    return {
        "n_features": info["n_total_features"],
        "max_group_width": info["horizon"],
        "feature_names": feature_names,
        "varying_features": info["tv_indices"],
        "mono_inc_features": info["mono_inc_indices"],
        "mono_dec_features": info["mono_dec_indices"],
    }


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lgb_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    feature_names: list[str],
    n_trees: int = 500,
    max_depth: int = 8,
    cat_indices: list[int] | None = None,
) -> lgb.Booster:
    ds = lgb.Dataset(
        train_X, label=train_y,
        feature_name=feature_names,
        categorical_feature=[feature_names[i] for i in cat_indices] if cat_indices else "auto",
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 2 ** max_depth - 1,
        "max_depth": max_depth,
        "learning_rate": 0.05,
        "verbose": -1,
        "seed": 42,
    }
    return lgb.train(params, ds, num_boost_round=n_trees)


def xgb_feature_types(
    feature_names: list[str], cat_indices: list[int] | None,
) -> list[str] | None:
    """Build XGBoost feature_types list: 'c' for categoricals, 'q' for numeric."""
    if not cat_indices:
        return None
    return ["c" if i in cat_indices else "q" for i in range(len(feature_names))]


def train_xgb_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    feature_names: list[str],
    n_trees: int = 500,
    max_depth: int = 8,
    cat_indices: list[int] | None = None,
):
    import xgboost as xgb
    feature_types = xgb_feature_types(feature_names, cat_indices)
    dtrain = xgb.DMatrix(
        train_X, label=train_y,
        feature_names=feature_names,
        feature_types=feature_types,
        enable_categorical=bool(cat_indices),
    )
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": max_depth,
        "learning_rate": 0.05,
        "tree_method": "hist",
        "seed": 42,
    }
    return xgb.train(params, dtrain, num_boost_round=n_trees)


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------

def write_raw_f64(path: Path, data: np.ndarray) -> None:
    """Write a 2D float64 array: u64 n_rows, u64 n_cols, then flat LE f64."""
    n_rows, n_cols = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", n_rows, n_cols))
        f.write(data.astype("<f8").tobytes())


def write_group_offsets(path: Path, offsets: np.ndarray) -> None:
    """Write variable-length group offsets for sweep_bench --group-offsets.

    Format: u64 n_groups, then (n_groups + 1) u64 LE cumulative row offsets.
    offsets[0] must be 0 and offsets must be monotonically increasing.
    """
    offsets = np.asarray(offsets, dtype=np.uint64)
    assert offsets[0] == 0, "first offset must be 0"
    n_groups = len(offsets) - 1
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n_groups))
        f.write(offsets.astype("<u8").tobytes())


def read_group_offsets(path: Path) -> np.ndarray:
    """Read group_offsets.bin → numpy array of (n_groups + 1) cumulative offsets."""
    raw = path.read_bytes()
    n_groups = struct.unpack("<Q", raw[:8])[0]
    offsets = np.frombuffer(raw[8:], dtype="<u8")
    assert len(offsets) == n_groups + 1
    return offsets



# ---------------------------------------------------------------------------
# Benchmark runner (TreeWalker via sweep_bench binary)
# ---------------------------------------------------------------------------

def find_sweep_bench() -> Path:
    """Return the sweep_bench binary path, building it if necessary."""
    binary = PROJECT_ROOT / "target" / "release" / "sweep_bench"
    if not binary.exists():
        build_sweep_bench()
    return binary


def build_sweep_bench() -> Path:
    """Build the sweep_bench binary with -C target-cpu=native, QuickScorer, and external baselines."""
    binary = PROJECT_ROOT / "target" / "release" / "sweep_bench"
    print("Building sweep_bench (release, target-cpu=native, external-bench)...", file=sys.stderr)
    env = {
        **os.environ,
        "PATH": f"{Path.home() / '.cargo/bin'}:{os.environ.get('PATH', '')}",
        "RUSTFLAGS": "-C target-cpu=native",
    }
    subprocess.run(
        ["cargo", "build", "--manifest-path", "experiments/benchmarks/Cargo.toml",
         "--target-dir", "target", "--release", "--bin", "sweep_bench",
         "--features", "external-bench"],
        cwd=str(PROJECT_ROOT),
        check=True,
        env=env,
    )
    return binary


def run_sweep_bench(
    binary: Path,
    model_dir: Path,
    data_dir: Path | None = None,
    iters: int = 101,
    group_offsets_path: Path | None = None,
    max_time_secs: float | None = None,
) -> list[dict] | None:
    """Run sweep_bench, return parsed JSON results."""
    cmd = [str(binary), str(model_dir), "--iters", str(iters)]
    if data_dir is not None:
        cmd.extend(["--data-dir", str(data_dir)])
    if group_offsets_path is not None:
        cmd.extend(["--group-offsets", str(group_offsets_path)])
    if max_time_secs is not None:
        cmd.extend(["--max-time-secs", str(max_time_secs)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        if result.returncode != 0:
            print(f"  sweep_bench failed: {result.stderr[-1000:]}", file=sys.stderr)
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"  benchmark error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Timing helpers — all return (median, p5, p95) in us per observation
# ---------------------------------------------------------------------------

def timing_summary(times_us: list[float]) -> tuple[float, float, float]:
    """Return (median, p5, p95) from a list of per-obs us timings."""
    p5, median, p95 = np.percentile(times_us, [5, 50, 95])
    return float(median), float(p5), float(p95)


def _timed_loop(
    predict_fn: Callable, iters: int, n_obs: int, warmup: int = 1,
) -> list[float]:
    """Time predict_fn over iters iterations with warmup. Returns per-obs us.

    GC is disabled during the timed region to prevent collection cycles
    from contaminating individual iteration timings.
    """
    import gc
    for _ in range(warmup):
        predict_fn()
    gc.collect()
    gc.disable()
    try:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            predict_fn()
            times.append((time.perf_counter_ns() - t0) / 1000 / n_obs)
    finally:
        gc.enable()
    return times


def _group_boundaries(
    n_obs: int, horizon: int, group_offsets: np.ndarray | None,
) -> list[tuple[int, int]]:
    """Return (start, end) row pairs for each observation.

    For variable-width groups, `group_offsets` is the cumulative offset
    array (length n_obs + 1).  For fixed-width groups, boundaries are
    computed from `horizon`.
    """
    if group_offsets is not None:
        return [(int(group_offsets[g]), int(group_offsets[g + 1]))
                for g in range(n_obs)]
    return [(g * horizon, (g + 1) * horizon) for g in range(n_obs)]


def time_lgb_predict(
    bst: lgb.Booster, test_X: np.ndarray, horizon: int, iters: int,
    n_obs: int | None = None,
    warmup: int = 1,
    group_offsets: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Time LightGBM prediction, one group at a time."""
    if n_obs is None:
        n_obs = test_X.shape[0] // horizon
    bounds = _group_boundaries(n_obs, horizon, group_offsets)

    def _predict():
        for start, end in bounds:
            bst.predict(test_X[start:end], num_threads=1)

    return timing_summary(_timed_loop(_predict, iters, n_obs, warmup=warmup))


def time_xgb_predict(
    bst, test_X: np.ndarray, horizon: int, iters: int,
    feature_names: list[str] | None = None,
    n_obs: int | None = None,
    warmup: int = 1,
    group_offsets: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Time XGBoost prediction via inplace_predict, one group at a time."""
    if n_obs is None:
        n_obs = test_X.shape[0] // horizon
    bounds = _group_boundaries(n_obs, horizon, group_offsets)

    def _predict():
        for start, end in bounds:
            bst.inplace_predict(test_X[start:end])

    return timing_summary(_timed_loop(_predict, iters, n_obs, warmup=warmup))


def compile_tl2cgen(model_path: Path, so_path: Path,
                    nthread: int = 0) -> bool:
    """Compile a treelite binary model to a tl2cgen shared library.

    model_path must be a treelite binary file (model_treelite.bin),
    created by treelite.Model.serialize() or serialize_bytes().

    Idempotent — skips if so_path already exists.
    nthread=0 (default) uses all available cores.
    """
    if so_path.exists():
        print(f"    tl2cgen: {so_path.name} exists, skipping", file=sys.stderr)
        return True
    try:
        import treelite
        import tl2cgen
    except (ImportError, OSError):
        print("    tl2cgen: not installed, skipping", file=sys.stderr)
        return False

    try:
        nthread = nthread or os.cpu_count() or 4
        # Set CFLAGS so tl2cgen's internal GCC invocation uses -march=native.
        # tl2cgen.export_lib() does not accept arch flags directly.
        old_cflags = os.environ.get("CFLAGS", "")
        old_cxxflags = os.environ.get("CXXFLAGS", "")
        os.environ["CFLAGS"] = f"-march=native {old_cflags}".strip()
        os.environ["CXXFLAGS"] = f"-march=native {old_cxxflags}".strip()
        try:
            with _suppress_fd_stderr():
                tl_model = treelite.Model.deserialize(str(model_path))
                tl2cgen.export_lib(
                    tl_model, toolchain="gcc", libpath=str(so_path),
                    params={"parallel_comp": tl_model.num_tree},
                    nthread=nthread, verbose=False,
                )
        finally:
            os.environ["CFLAGS"] = old_cflags
            os.environ["CXXFLAGS"] = old_cxxflags
        print(f"    tl2cgen: compiled {so_path.name} (-march=native)", file=sys.stderr)
        return True
    except Exception as e:
        print(f"    tl2cgen: compile failed: {e}", file=sys.stderr)
        return False


def compile_lleaves(model_txt_path: Path, so_path: Path,
                    nthread: int = 0) -> bool:
    """Compile a LightGBM model to an lleaves shared library.

    Uses the DHM chunked compilation pipeline: generate LLVM IR via lleaves,
    split into per-tree chunks, compile each chunk in parallel via llc
    subprocesses, then link with clang. This avoids llvmlite's in-process
    LLVM optimization which hangs on large models (500+ trees).

    Idempotent — skips if so_path already exists.
    nthread=0 (default) uses all available cores.
    """
    if so_path.exists():
        print(f"    lleaves: {so_path.name} exists, skipping", file=sys.stderr)
        return True

    try:
        # Import the DHM compile module (lives alongside this file or in dhm)
        compile_mod_path = Path(__file__).parent / "compile.py"
        if not compile_mod_path.exists():
            # Fallback: try lleaves.Model.compile() directly
            print("    lleaves: compile.py not found, using lleaves.Model.compile()",
                  file=sys.stderr)
            import lleaves
            model = lleaves.Model(model_file=str(model_txt_path))
            model.compile(cache=str(so_path))
            print(f"    lleaves: compiled {so_path.name}", file=sys.stderr)
            return True

        import importlib.util
        spec = importlib.util.spec_from_file_location("compile", compile_mod_path)
        compile_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compile_mod)

        t0 = time.perf_counter()
        ir = compile_mod.generate_ir(str(model_txt_path), fblocksize=34, quiet=True)
        # Use O3 backend optimization via llc (no opt pass — avoids IR version
        # mismatch between llvmlite's bundled LLVM and the system opt binary).
        # fblocksize=34 and opt_level="3" match lleaves defaults.
        cfg = replace(compile_mod.NATIVE, opt_level="3",
                      fp_contract="on", n_jobs=nthread or 0)
        compile_mod.compile_ir(ir, cfg, output_path=so_path, quiet=True)
        elapsed = time.perf_counter() - t0
        print(f"    lleaves: compiled {so_path.name} ({elapsed:.1f}s, {ir.n_trees} trees)",
              file=sys.stderr)
        return True
    except Exception as e:
        print(f"    lleaves compile failed: {e}", file=sys.stderr)
        return False


def try_tl2cgen_predict(
    so_path: Path, test_X: np.ndarray, horizon: int, iters: int,
    n_obs: int | None = None,
    warmup: int = 1,
    group_offsets: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    """Time tl2cgen prediction, one group at a time.

    Each group pays the real DMatrix-construction + predict cost.
    """
    try:
        import tl2cgen
    except (ImportError, OSError):
        return None

    if not so_path.exists():
        print(f"  tl2cgen: {so_path} not found, skipping", file=sys.stderr)
        return None

    try:
        predictor = tl2cgen.Predictor(str(so_path), nthread=1)
        if n_obs is None:
            n_obs = test_X.shape[0] // horizon
        bounds = _group_boundaries(n_obs, horizon, group_offsets)

        def _predict():
            for start, end in bounds:
                dmat = tl2cgen.DMatrix(test_X[start:end])
                predictor.predict(dmat)

        return timing_summary(_timed_loop(_predict, iters, n_obs, warmup=warmup))
    except Exception as e:
        print(f"  tl2cgen predict failed: {e}", file=sys.stderr)
        return None


def try_lleaves_predict(
    so_path: Path, test_X: np.ndarray, horizon: int, iters: int,
    n_obs: int | None = None,
    warmup: int = 1,
    group_offsets: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    """Time lleaves prediction via ctypes, one group at a time.

    The .so exports `forest_root(data*, results*, start, end)` which
    already accepts row ranges, so per-group iteration is natural.
    """
    import ctypes

    if not so_path.exists():
        print(f"  lleaves: {so_path} not found, skipping", file=sys.stderr)
        return None

    try:
        lib = ctypes.CDLL(str(so_path))
        lib.forest_root.restype = None
        lib.forest_root.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.c_int32,
        ]

        n_rows = test_X.shape[0]
        if n_obs is None:
            n_obs = n_rows // horizon
        bounds = _group_boundaries(n_obs, horizon, group_offsets)
        data = np.ascontiguousarray(test_X, dtype=np.float64)
        results = np.zeros(n_rows, dtype=np.float64)
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        results_ptr = results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        def _predict():
            for start, end in bounds:
                lib.forest_root(data_ptr, results_ptr, start, end)

        return timing_summary(_timed_loop(
            _predict, iters, n_obs, warmup=warmup,
        ))
    except Exception as e:
        print(f"  lleaves failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def factorial_param_combos() -> list[tuple[int, int, int]]:
    """All T × L × H combos for Grid 1. Returns 252 tuples."""
    from itertools import product
    return sorted(product(GRID1_N_TREES, GRID1_MAX_DEPTH, GRID1_HORIZON))


def ablation_combos(anchor: tuple[int, int, int], is_full_cross: bool) -> list[dict[str, bool]]:
    """Generate ablation flag combos for an anchor config.

    is_full_cross=False: within-group powerset (8 runtime + 7 parse-time = 15
    combos). Each group's flags vary independently; cross-group interactions
    are not explored.
    is_full_cross=True: 2^6 = 64 combos (all runtime × all parse-time).
    """
    from itertools import product as iprod

    all_flags = RUNTIME_FLAGS + PARSE_FLAGS

    if is_full_cross:
        combos = []
        for bits in iprod([False, True], repeat=len(all_flags)):
            combos.append(dict(zip(all_flags, bits)))
        return combos

    # One-at-a-time within each group, cross between groups.
    def powerset_flags(flags: list[str]) -> list[dict[str, bool]]:
        result = []
        for bits in iprod([False, True], repeat=len(flags)):
            result.append(dict(zip(flags, bits)))
        return result

    runtime_combos = powerset_flags(RUNTIME_FLAGS)   # 8
    parse_combos = powerset_flags(PARSE_FLAGS)        # 8
    combos = []
    for rt in runtime_combos:
        combos.append({**{f: False for f in all_flags}, **rt})
    for pt in parse_combos:
        if all(not v for v in pt.values()):
            continue  # skip all-false (already in runtime combos)
        combos.append({**{f: False for f in all_flags}, **pt})
    return combos


def run_sweep_bench_ablation(
    binary: Path,
    model_dir: Path,
    data_dir: Path | None = None,
    ablation: dict[str, bool] | None = None,
    iters: int = 11,
    warmup: int = 3,
    group_offsets_path: Path | None = None,
    emit_extended: bool = True,
    max_time_secs: float | None = None,
) -> dict | None:
    """Run sweep_bench with --disable-* flags constructed from ablation dict.

    Returns a single result dict (first element of the JSON array), or None on error.
    """
    cmd = [str(binary), str(model_dir), "--iters", str(iters), "--warmup", str(warmup)]
    if data_dir is not None:
        cmd.extend(["--data-dir", str(data_dir)])
    if group_offsets_path is not None:
        cmd.extend(["--group-offsets", str(group_offsets_path)])
    if emit_extended:
        cmd.append("--emit-extended")
    if max_time_secs is not None:
        cmd.extend(["--max-time-secs", str(max_time_secs)])

    flag_map = {
        "precompute": "--disable-precompute",
        "unsplit": "--disable-unsplit",
        "monotonic": "--disable-monotonic",
        "tree_ordering": "--disable-tree-ordering",
        "prefix_grouping": "--disable-prefix-grouping",
        "bitset_intern": "--disable-bitset-intern",
    }

    if ablation is None or not any(ablation.values()):
        cmd.extend(["--mode", "baseline"])
    elif ablation:
        for key, enabled in ablation.items():
            if enabled and key in flag_map:
                cmd.append(flag_map[key])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        if result.returncode != 0:
            print(f"  sweep_bench failed: {result.stderr[-1000:]}", file=sys.stderr)
            return None
        results = json.loads(result.stdout)
        return results[0] if results else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"  benchmark error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Measurement protocol
# ---------------------------------------------------------------------------

@dataclass
class MeasurementProtocol:
    """Controls benchmark repetition.

    The 'legacy' fields (warmup, iters, repeats) are used by the old
    full-dataset protocol.  The 'block' fields drive the new blocked
    paired measurement protocol via sweep_bench --block-mode.
    """
    # Legacy protocol fields
    warmup: int = 5
    iters: int = 30
    repeats: int = 3
    max_time_secs: float | None = None

    # Block protocol fields
    protocol_name: str = "legacy"
    seed: int = 42
    pool_frac: float = 0.25
    pool_max_groups: int = 4096
    n_batches: int = 12
    target_sample_ms: float = 75.0
    min_blocks: int = 11
    max_blocks: int = 21
    precision_target_pct: float = 3.0
    collect_stats: bool = False
    use_full_dataset: bool = False
    sentinel_period: int = 25

    @classmethod
    def standard(cls) -> MeasurementProtocol:
        return cls(warmup=5, iters=30, repeats=3, protocol_name="legacy")

    @classmethod
    def quick(cls) -> MeasurementProtocol:
        return cls(warmup=3, iters=11, repeats=1, protocol_name="legacy")

    @classmethod
    def fast_sweep(cls) -> MeasurementProtocol:
        return cls(
            protocol_name="fast_sweep",
            seed=42,
            pool_frac=0.25,
            pool_max_groups=4096,
            n_batches=12,
            target_sample_ms=75.0,
            min_blocks=11,
            max_blocks=21,
            precision_target_pct=3.0,
            collect_stats=False,
            use_full_dataset=False,
            sentinel_period=25,
            # Legacy fields (used for external baselines)
            warmup=3, iters=11, repeats=1,
        )

    @classmethod
    def final_anchor(cls) -> MeasurementProtocol:
        return cls(
            protocol_name="final_anchor",
            seed=42,
            pool_frac=0.50,
            pool_max_groups=8192,
            n_batches=24,
            target_sample_ms=150.0,
            min_blocks=21,
            max_blocks=41,
            precision_target_pct=1.75,
            collect_stats=False,
            use_full_dataset=False,
            sentinel_period=25,
            # Legacy fields
            warmup=5, iters=30, repeats=1,
        )


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def detect_architecture() -> str:
    """Return 'intel' or 'arm' based on platform."""
    import platform
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm"
    return "intel"


# ---------------------------------------------------------------------------
# Block protocol: benchmark pool construction
# ---------------------------------------------------------------------------

def build_benchmark_pool(
    n_total_groups: int,
    protocol: MeasurementProtocol,
    group_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Build a deterministic benchmark pool of group indices.

    For fixed-width datasets (group_offsets is None), samples uniformly.
    For variable-width datasets, samples stratified by group width to
    preserve the empirical width distribution.

    Returns sorted array of group indices.
    """
    rng = np.random.default_rng(protocol.seed)
    pool_size = min(
        protocol.pool_max_groups,
        math.ceil(protocol.pool_frac * n_total_groups),
    )
    pool_size = min(pool_size, n_total_groups)
    # For small datasets, use all groups to avoid sampling bias.
    if n_total_groups <= 500:
        pool_size = n_total_groups

    if protocol.use_full_dataset:
        return np.arange(n_total_groups, dtype=np.uint64)

    if group_offsets is None:
        # Fixed-width: uniform sample.
        indices = rng.choice(n_total_groups, size=pool_size, replace=False)
        return np.sort(indices).astype(np.uint64)

    # Variable-width: stratified sample by group width.
    widths = np.diff(group_offsets)
    unique_widths = np.unique(widths)
    sampled = []
    for w in unique_widths:
        group_ids = np.where(widths == w)[0]
        # Proportional allocation.
        n_sample = max(1, round(len(group_ids) / n_total_groups * pool_size))
        n_sample = min(n_sample, len(group_ids))
        chosen = rng.choice(group_ids, size=n_sample, replace=False)
        sampled.extend(chosen)

    # Trim or pad to exact pool_size.
    sampled = np.array(sampled, dtype=np.uint64)
    if len(sampled) > pool_size:
        sampled = rng.choice(sampled, size=pool_size, replace=False)
    return np.sort(sampled)


def write_pool_manifest(path: Path, pool: np.ndarray) -> None:
    """Write a pool manifest: u64 count, then count × u64 group indices."""
    pool = np.asarray(pool, dtype=np.uint64)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pool)))
        f.write(pool.astype("<u8").tobytes())


# ---------------------------------------------------------------------------
# Block protocol: result parsing and adaptive stopping
# ---------------------------------------------------------------------------

def parse_block_results(jsonl_output: str) -> list[dict]:
    """Parse JSONL output from sweep_bench --block-mode."""
    records = []
    for line in jsonl_output.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("{") is False:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def paired_log_speedup(
    records: list[dict],
    baseline_mode: str = "baseline",
    target_mode: str = "full",
) -> tuple[float, float, float]:
    """Compute paired speedup from block records.

    Returns (speedup_estimate, ci_low, ci_high) where speedup = t_target / t_baseline.
    Uses paired log-ratios per block with bootstrap 95% CI.
    """
    # Group by block_id.
    blocks: dict[int, dict[str, float]] = {}
    for r in records:
        bid = r["block_id"]
        if bid not in blocks:
            blocks[bid] = {}
        blocks[bid][r["mode"]] = r["us_per_obs"]

    # Extract paired log-ratios.
    log_ratios = []
    for bid in sorted(blocks):
        b = blocks[bid]
        if baseline_mode in b and target_mode in b:
            d = math.log(b[target_mode]) - math.log(b[baseline_mode])
            log_ratios.append(d)

    if len(log_ratios) < 3:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.array(log_ratios)
    point = float(np.exp(np.mean(arr)))

    # Bootstrap 95% CI.
    rng = np.random.default_rng(12345)
    n_boot = 10_000
    boot_means = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return (point, float(np.exp(lo)), float(np.exp(hi)))


def speedup_ci_half_width_pct(ci_low: float, ci_high: float) -> float:
    """CI half-width as percentage of the estimate midpoint."""
    if math.isnan(ci_low) or math.isnan(ci_high):
        return float("inf")
    mid = (ci_low + ci_high) / 2.0
    if mid == 0:
        return float("inf")
    return (ci_high - ci_low) / 2.0 / mid * 100.0


def should_stop_adaptive(
    records: list[dict],
    baseline_mode: str,
    target_mode: str,
    precision_target_pct: float,
    min_blocks: int,
) -> bool:
    """Check whether to stop adaptively based on paired speedup CI."""
    # Count completed blocks.
    block_ids = set(r["block_id"] for r in records)
    if len(block_ids) < min_blocks:
        return False

    _, ci_lo, ci_hi = paired_log_speedup(records, baseline_mode, target_mode)
    half_width = speedup_ci_half_width_pct(ci_lo, ci_hi)
    return half_width < precision_target_pct


# ---------------------------------------------------------------------------
# Block protocol: running sweep_bench in block mode
# ---------------------------------------------------------------------------

def run_sweep_bench_blocked(
    binary: Path,
    model_dir: Path,
    data_dir: Path | None = None,
    modes: str = "baseline,full",
    pool_manifest_path: Path | None = None,
    group_offsets_path: Path | None = None,
    protocol: MeasurementProtocol | None = None,
) -> list[dict] | None:
    """Run sweep_bench in --block-mode and return parsed block records.

    If pool_manifest_path is None, sweep_bench uses all groups.
    Returns list of BlockRecord dicts, or None on error.
    """
    if protocol is None:
        protocol = MeasurementProtocol.fast_sweep()

    cmd = [
        str(binary), str(model_dir),
        "--block-mode",
        "--modes", modes,
        "--n-batches", str(protocol.n_batches),
        "--target-ms", str(protocol.target_sample_ms),
        "--min-blocks", str(protocol.min_blocks),
        "--max-blocks", str(protocol.max_blocks),
    ]
    if data_dir is not None:
        cmd.extend(["--data-dir", str(data_dir)])
    if pool_manifest_path is not None:
        cmd.extend(["--pool", str(pool_manifest_path)])
    if group_offsets_path is not None:
        cmd.extend(["--group-offsets", str(group_offsets_path)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"  sweep_bench block-mode failed: {result.stderr[-500:]}", file=sys.stderr)
            return None
        return parse_block_results(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"  block-mode error: {e}", file=sys.stderr)
        return None


def run_quickscorer_bench(
    binary: Path,
    model_dir: Path,
    qs_model_path: Path,
    data_dir: Path | None = None,
    group_offsets_path: Path | None = None,
    warmup: int = 3,
    iters: int = 21,
    min_iters: int = 11,
    max_time_secs: float | None = 10.0,
) -> dict | None:
    """Run QuickScorer baseline (per-observation). Returns parsed JSON or None.

    sweep_bench iterates over groups and scores each row within a group,
    so the returned latency_us_per_obs reflects the real per-group cost
    including loop and function-call overhead.
    """
    cmd = [
        str(binary), str(model_dir),
        "--quickscorer", str(qs_model_path),
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--min-iters", str(min_iters),
    ]
    if data_dir is not None:
        cmd.extend(["--data-dir", str(data_dir)])
    if group_offsets_path is not None:
        cmd.extend(["--group-offsets", str(group_offsets_path)])
    if max_time_secs is not None:
        cmd.extend(["--max-time-secs", str(max_time_secs)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in result.stdout.strip().splitlines():
            if '"quickscorer"' in line:
                return json.loads(line)
    except Exception as e:
        print(f"  QuickScorer failed: {e}", file=sys.stderr)
    return None


def summarize_block_results(records: list[dict]) -> dict[str, dict]:
    """Summarize block records into per-mode statistics.

    Returns {mode_name: {median_us, p5_us, p95_us, n_blocks}}.
    """
    from collections import defaultdict
    mode_timings: dict[str, list[float]] = defaultdict(list)
    for r in records:
        mode_timings[r["mode"]].append(r["us_per_obs"])

    summary = {}
    for mode, timings in mode_timings.items():
        arr = np.array(timings)
        summary[mode] = {
            "median_us": float(np.median(arr)),
            "p5_us": float(np.percentile(arr, 5)),
            "p95_us": float(np.percentile(arr, 95)),
            "n_blocks": len(arr),
        }
    return summary


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------

def schedule_sentinels(n_cells: int, period: int = 25) -> set[int]:
    """Return cell indices after which a sentinel should be rerun."""
    return {i for i in range(period - 1, n_cells, period)}


def check_sentinel_drift(
    sentinel_timings: list[float],
    threshold_pct: float = 3.0,
) -> tuple[bool, float]:
    """Check if sentinel timings have drifted beyond threshold.

    Returns (drifted, max_deviation_pct).
    """
    if len(sentinel_timings) < 2:
        return (False, 0.0)
    baseline = sentinel_timings[0]
    if baseline == 0:
        return (False, 0.0)
    max_dev = max(abs(t - baseline) / baseline * 100.0 for t in sentinel_timings[1:])
    return (max_dev > threshold_pct, max_dev)
