"""The four datasets: download, load, split, and the survival training expansion.

SUPPORT, FLCHAIN and credit are public and fetched with pooch against pinned
SHA-256s into ``experiments/data/raw/``. Expedia may not be
redistributed: ``fetch-expedia`` converts the user's Kaggle download, and
``check_expedia`` compares its content fingerprint with the file behind the
paper.

Every function here reproduced the released preparation exactly at the
package refactor (the byte-identical check), so its arithmetic and row order
are deliberate. One change since is intended: ``split_expedia`` filters only
on the search-level features, so cross-border searches, where
``prop_country_id`` varies, stay in the pool.
"""

import hashlib
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .paths import Paths


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --- Sources -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Source:
    url: str
    sha256: str
    filename: str


SOURCES = {
    "support": Source(
        "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/"
        "experiments/data/support/support_train_test.h5",
        "e398f49930e4efc4ecd675a72ac547f0d1ae25ccf13f6a1024cded12ea88a83e",
        "support.h5",
    ),
    "flchain": Source(
        "https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv",
        "a96bcc58addb4c127e5012c5974aec8c7daf66123ddb104f309a78f158daa563",
        "flchain.csv",
    ),
    # UCI Default of Credit Card Clients (Yeh & Lien 2009), OpenML 42477,
    # CC BY 4.0, DOI 10.24432/C55S3H. Frozen 2026-07-24.
    "credit": Source(
        "https://openml.org/data/v1/download/21854402/default-of-credit-card-clients.arff",
        "6b621ff9f006c875dcda4ea4163db3574773caa89f06bb36b81973af8062a3f7",
        "default-of-credit-card-clients.arff",
    ),
}


def fetch(paths: Paths, name: str) -> Path:
    import pooch

    src = SOURCES[name]
    return Path(
        pooch.retrieve(
            src.url,
            known_hash=f"sha256:{src.sha256}",
            fname=src.filename,
            path=paths.raw,
            progressbar=False,
        )
    )


def source_record(name: str) -> dict[str, str]:
    src = SOURCES[name]
    return {"url": src.url, "sha256": src.sha256}


# --- Survival: SUPPORT and FLCHAIN --------------------------------------------------


@dataclass(frozen=True, slots=True)
class SurvivalSpec:
    name: str
    duration_col: str = "duration"
    event_col: str = "event"
    drop_cols: tuple[str, ...] = ()
    cat_cols: tuple[str, ...] = ()  # passed to LightGBM and XGBoost as categorical
    binary_cols: tuple[str, ...] = ()  # encoded like categoricals, trained as numeric


# Feature types from the original documentation: SUPPORT (Knaus et al. 1995,
# Katzman et al. 2018) and FLCHAIN (Dispenzieri et al. 2012, R survival::flchain).
SURVIVAL = {
    "support": SurvivalSpec(
        "support",
        cat_cols=("x2", "x3", "x6"),  # num.co (0-9), race (6), cancer (3)
        binary_cols=("x1", "x4", "x5"),  # sex, diabetes, dementia
    ),
    "flchain": SurvivalSpec(
        "flchain",
        drop_cols=("rownames",),  # row ID, no predictive content
        cat_cols=("sample.yr", "flc.grp"),  # enrollment year (9), FLC decile (10)
        binary_cols=("sex", "mgus"),  # sex (M/F), prior MGUS (0/1)
    ),
}

TV_FEATURE_NAMES = [
    "time_step",  # increasing
    "time_step_sq",  # increasing
    "remaining_steps",  # decreasing
    "remaining_frac_sq",  # decreasing
    "interaction_x0_t",  # x0 * time_fraction, not monotonic
    "hazard_bump",  # not monotonic
]
N_TV = len(TV_FEATURE_NAMES)
INTERACTION = TV_FEATURE_NAMES.index("interaction_x0_t")


def _load_support(path: Path) -> pl.DataFrame:
    import h5py

    frames = []
    with h5py.File(path, "r") as f:
        for split in f:
            x = f[split]["x"][:]
            t = f[split]["t"][:]
            e = f[split]["e"][:]
            cols = {f"x{i}": x[:, i] for i in range(x.shape[1])}
            frames.append(
                pl.DataFrame(cols).with_columns(pl.Series("duration", t), pl.Series("event", e))
            )
    return pl.concat(frames)


def _load_flchain(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    df = df.drop(["chapter", "Unnamed: 0"], strict=False)
    df = df.filter(pl.col("creatinine").is_not_null())
    df = df.with_columns(pl.when(pl.col("sex") == "M").then(1.0).otherwise(0.0).alias("sex"))
    for col in ["sample.yr", "flc.grp"]:
        if col in df.columns:
            uniq = sorted(df[col].drop_nulls().unique().to_list())
            mapping = {v: float(i) for i, v in enumerate(uniq)}
            df = df.with_columns(
                pl.col(col).replace_strict(mapping, default=None).cast(pl.Float64).alias(col)
            )
    df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False) for c in df.columns)
    return df.rename({"futime": "duration", "death": "event"})


def load_survival(paths: Paths, name: str) -> pl.DataFrame:
    path = fetch(paths, name)
    return _load_support(path) if name == "support" else _load_flchain(path)


def split_patients(
    df: pl.DataFrame, test_frac: float, rng: np.random.Generator
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n = len(df)
    n_test = max(1, int(n * test_frac))
    perm = rng.permutation(n)
    return df[sorted(perm[n_test:])], df[sorted(perm[:n_test])]


def bin_edges(duration: np.ndarray, event: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile time-bin edges of the event times. Ties merge bins, so a
    requested horizon can realize fewer steps."""
    event_times = duration[event > 0]
    if len(event_times) < n_bins:
        return np.linspace(duration.min(), duration.max(), n_bins + 1)
    return np.unique(np.percentile(event_times, np.linspace(0, 100, n_bins + 1)))


def sanitize(name: str) -> str:
    return name.replace(".", "_").replace(" ", "_").replace("[", "_").replace("]", "_")


@dataclass(slots=True)
class Covariates:
    """One row per patient, nulls removed, categories encoded."""

    names: list[str]  # raw column names, in feature order
    values: np.ndarray  # (patients, covariates) float64
    cat_indices: list[int]
    rows: np.ndarray  # each patient's row in its split
    duration: np.ndarray | None = None
    event: np.ndarray | None = None


@dataclass(slots=True)
class SurvivalSplit:
    spec: SurvivalSpec
    seed: int
    test_frac: float
    n_patients: int
    train_df: pl.DataFrame
    test_df: pl.DataFrame
    encoders: dict[str, dict] = field(default_factory=dict)

    def covariates(self, which: str) -> Covariates:
        """Train covariates drop patients with a null covariate, duration or
        event; test covariates drop patients with a null covariate."""
        df = self.train_df if which == "train" else self.test_df
        spec = self.spec
        exclude = {spec.duration_col, spec.event_col, *spec.drop_cols}
        names = [c for c in df.columns if c not in exclude]
        exprs = []
        for col in names:
            if col in self.encoders:
                exprs.append(
                    pl.col(col)
                    .replace_strict(self.encoders[col], default=None)
                    .cast(pl.Float64)
                    .alias(col)
                )
            else:
                exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
        cov = df.select(names).with_columns(exprs)
        valid = ~cov.select(pl.any_horizontal(pl.all().is_null())).to_series().to_numpy()
        duration = event = None
        if which == "train":
            duration = df[spec.duration_col].to_numpy().astype(float)
            event = df[spec.event_col].to_numpy().astype(float)
            valid &= ~np.isnan(duration) & ~np.isnan(event)
            duration, event = duration[valid], event[valid]
        cov = cov.filter(pl.Series(valid))
        cat = set(spec.cat_cols)
        return Covariates(
            names=names,
            values=cov.to_numpy().astype(np.float64),
            cat_indices=[i for i, c in enumerate(names) if c in cat],
            rows=np.nonzero(valid)[0],
            duration=duration,
            event=event,
        )

    def edges(self, horizon: int) -> np.ndarray:
        duration = self.train_df[self.spec.duration_col].to_numpy().astype(float)
        event = self.train_df[self.spec.event_col].to_numpy().astype(float)
        return bin_edges(duration, event, horizon)


def dataset_seed(name: str, seed: int) -> int:
    """The released per-dataset seed: the base seed plus a hash of the name."""
    return seed + int.from_bytes(name.encode(), "big") % (2**31)


def split_survival(paths: Paths, name: str, seed: int, test_frac: float) -> SurvivalSplit:
    spec = SURVIVAL[name]
    df = load_survival(paths, name)
    ds_seed = dataset_seed(name, seed)
    train_df, test_df = split_patients(df, test_frac, np.random.default_rng(ds_seed))
    # Category encoders are fit on the train split only; unseen test categories
    # become null and drop out with the other nulls.
    cat_set = {*spec.cat_cols, *spec.binary_cols}
    encoders = {
        col: {v: i for i, v in enumerate(sorted(train_df[col].drop_nulls().unique().to_list()))}
        for col in train_df.columns
        if col in cat_set
    }
    log(f"  {name}: {len(df)} patients, split {len(train_df)} train / {len(test_df)} test")
    return SurvivalSplit(spec, ds_seed, test_frac, len(df), train_df, test_df, encoders)


def tv_table(horizon: int) -> np.ndarray:
    """The time-varying features for steps 0..horizon-1, except the
    interaction column, which holds the time fraction to multiply x0 by.

    Built with Python floats, one step at a time, as the released expansion
    did, so every value is bit-identical."""
    h1 = max(horizon - 1, 1)
    rows = []
    for t in range(horizon):
        frac = t / h1
        remaining = h1 - t
        rem_frac = remaining / h1
        rows.append(
            [float(t), frac**2, float(remaining), rem_frac**2, frac, math.sin(math.pi * frac)]
        )
    return np.array(rows, dtype=np.float64).reshape(horizon, N_TV)


def tv_features(steps: np.ndarray, x0: np.ndarray, horizon: int) -> np.ndarray:
    """Time-varying features for each (step, x0) pair."""
    tv = tv_table(horizon)[steps]
    tv[:, INTERACTION] = x0 * tv[:, INTERACTION]
    return tv


def feature_names(cov: Covariates) -> list[str]:
    return [sanitize(c) for c in cov.names] + TV_FEATURE_NAMES


def expand_train(
    split: SurvivalSplit, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Covariates]:
    """Each train patient contributes one row per step up to its event or
    censoring bin; the label is 1 on the event step."""
    cov = split.covariates("train")
    assert cov.duration is not None and cov.event is not None
    horizon = len(edges) - 1
    last = np.clip(np.digitize(cov.duration, edges[1:]), 0, horizon - 1)
    counts = last + 1
    patient = np.repeat(np.arange(len(counts)), counts)
    starts = np.cumsum(counts) - counts
    steps = np.arange(int(counts.sum())) - np.repeat(starts, counts)
    X = np.hstack([cov.values[patient], tv_features(steps, cov.values[patient, 0], horizon)])
    y = ((steps == last[patient]) & (cov.event[patient] > 0)).astype(np.float64)
    return X, y, cov


# --- Expedia (ranking) -------------------------------------------------------------

EXPEDIA_LABEL = "click_bool"
# Feature columns, in the order the models are trained on.
EXPEDIA_FEATURES = [
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
]
# Search-level features: one value per search, so constant across its
# candidates. prop_country_id is a property attribute and varies: it is the
# same for every candidate of 99.44% of searches, but 2,236 of the 399,344
# searches with 2-128 candidates span countries.
EXPEDIA_SESSION = [
    "site_id",
    "visitor_location_country_id",
    "srch_destination_id",
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
    "random_bool",
]
EXPEDIA_VARYING = [f for f in EXPEDIA_FEATURES if f not in EXPEDIA_SESSION]

# Column order and dtypes of the parquet the paper was run on.
EXPEDIA_SCHEMA: dict[str, pl.DataType] = {
    "srch_id": pl.UInt32(),
    "site_id": pl.UInt8(),
    "visitor_location_country_id": pl.UInt8(),
    "prop_country_id": pl.UInt8(),
    "prop_id": pl.UInt32(),
    "prop_starrating": pl.UInt8(),
    "prop_review_score": pl.Float32(),
    "prop_brand_bool": pl.Boolean(),
    "prop_location_score1": pl.Float32(),
    "prop_location_score2": pl.Float32(),
    "prop_log_historical_price": pl.Float32(),
    "position": pl.UInt8(),
    "price_usd": pl.Float32(),
    "promotion_flag": pl.Boolean(),
    "srch_destination_id": pl.UInt16(),
    "srch_length_of_stay": pl.UInt8(),
    "srch_booking_window": pl.UInt16(),
    "srch_adults_count": pl.UInt8(),
    "srch_children_count": pl.UInt8(),
    "srch_room_count": pl.UInt8(),
    "srch_saturday_night_bool": pl.Boolean(),
    "random_bool": pl.Boolean(),
    "click_bool": pl.Boolean(),
}
EXPEDIA_ROWS = 9_917_530
EXPEDIA_SESSIONS = 399_344
EXPEDIA_FINGERPRINT = "471367a93277c91df3d99a815584c5f45a8acec2f17d84643f876989e29395a8"


def expedia_fingerprint(df: pl.DataFrame) -> str:
    """Library-version-independent content hash: names, dtypes, null masks, values."""
    h = hashlib.sha256()
    for c in df.columns:
        s = df[c]
        h.update(c.encode())
        h.update(str(s.dtype).encode())
        h.update(s.is_null().to_numpy().tobytes())
        v = s.fill_null(False) if s.dtype == pl.Boolean else s.fill_null(0)
        h.update(np.ascontiguousarray(v.to_numpy()).tobytes())
    return h.hexdigest()


def check_expedia(df: pl.DataFrame) -> tuple[bool, str]:
    rows, sessions = df.height, df["srch_id"].n_unique()
    fp = expedia_fingerprint(df)
    log(f"  expedia: rows={rows:,} sessions={sessions:,} fingerprint={fp}")
    ok = rows == EXPEDIA_ROWS and sessions == EXPEDIA_SESSIONS and fp == EXPEDIA_FINGERPRINT
    return ok, fp


def convert_expedia(src: Path, out: Path) -> bool:
    """Convert Kaggle's train.csv, or its data.zip, to the paper's parquet.

    The zip uses Deflate64, which Python's zipfile cannot read, so it is
    extracted with Info-ZIP ``unzip``. Booleans are 0/1 in the CSV and missing
    values the literal string NULL."""
    with tempfile.TemporaryDirectory() as tmp:
        if src.suffix == ".zip":
            if shutil.which("unzip") is None:
                raise RuntimeError("unzip not found; extract train.csv and pass its path")
            subprocess.run(
                ["unzip", "-o", "-q", "-j", str(src), "train.csv", "-d", tmp], check=True
            )
            src = Path(tmp) / "train.csv"
        read_types = {c: (pl.UInt8 if t == pl.Boolean else t) for c, t in EXPEDIA_SCHEMA.items()}
        df = pl.read_csv(
            src, columns=list(EXPEDIA_SCHEMA), schema_overrides=read_types, null_values="NULL"
        )
        df = df.select([pl.col(c).cast(t) for c, t in EXPEDIA_SCHEMA.items()])
    ok, _ = check_expedia(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return ok


@dataclass(slots=True)
class RankingSplit:
    seed: int
    test_frac: float
    max_sessions: int
    fingerprint: str
    fingerprint_ok: bool
    train_df: pl.DataFrame
    test_df: pl.DataFrame


def split_expedia(paths: Paths, seed: int, test_frac: float, max_sessions: int) -> RankingSplit:
    """Sessions of 2-128 candidates whose search-level features are constant,
    subsampled to ``max_sessions`` and split by session."""
    if not paths.expedia.exists():
        raise FileNotFoundError(f"{paths.expedia} not found; run treewalker-exp fetch-expedia")
    ds_seed = dataset_seed("expedia", seed)
    df = pl.read_parquet(paths.expedia)
    ok, fp = check_expedia(df)
    if not ok:
        log("  expedia: fingerprint MISMATCH, results will not reproduce the paper's")

    sizes = df.group_by("srch_id").agg(pl.len().alias("_n"))
    valid = sizes.filter((pl.col("_n") >= 2) & (pl.col("_n") <= 128))
    df_valid = df.join(valid.select("srch_id"), on="srch_id")
    for col in EXPEDIA_SESSION:
        varying = (
            df_valid.group_by("srch_id")
            .agg(pl.col(col).drop_nulls().n_unique().alias("_nuniq"))
            .filter(pl.col("_nuniq") > 1)
            .select("srch_id")
        )
        if len(varying) > 0:
            df_valid = df_valid.join(varying, on="srch_id", how="anti")
    valid = df_valid.group_by("srch_id").agg(pl.len().alias("_n")).select("srch_id")
    df = df.join(valid, on="srch_id")

    rng = np.random.default_rng(ds_seed)
    sids = df["srch_id"].unique().sort().to_list()
    if max_sessions and len(sids) > max_sessions:
        idx = rng.choice(len(sids), max_sessions, replace=False)
        df = df.filter(pl.col("srch_id").is_in([sids[i] for i in idx]))
        sids = df["srch_id"].unique().sort().to_list()
    n = len(sids)
    df = df.sort("srch_id")

    n_test = max(1, int(n * test_frac))
    test_set = set(rng.permutation(n)[:n_test].tolist())
    df = df.with_columns(
        pl.col("srch_id")
        .replace_strict(dict(zip(sids, range(n), strict=True)), return_dtype=pl.UInt32)
        .alias("_sidx")
    )
    is_test = df["_sidx"].is_in(list(test_set))
    log(f"  expedia: {n} sessions, split {n - n_test} train / {n_test} test")
    return RankingSplit(
        ds_seed,
        test_frac,
        max_sessions,
        fp,
        ok,
        df.filter(~is_test).sort("srch_id"),
        df.filter(is_test).sort("srch_id"),
    )


def session_offsets(srch_id: np.ndarray) -> np.ndarray:
    changes = np.where(srch_id[:-1] != srch_id[1:])[0] + 1
    return np.concatenate([[0], changes, [len(srch_id)]]).astype(np.uint64)


# --- Credit (what-if) --------------------------------------------------------------

CREDIT_FEATURES = [f"x{i}" for i in range(1, 24)]  # x1..x23
CREDIT_SEED = 42
CREDIT_TEST_FRAC = 0.2


def load_credit(paths: Paths) -> tuple[np.ndarray, np.ndarray]:
    """The OpenML ARFF body is CSV after ``@data``. Columns: id, x1..x23, y."""
    text = fetch(paths, "credit").read_text()
    idx = text.lower().find("@data")
    if idx < 0:
        raise ValueError("ARFF has no @data marker")
    rows = []
    for line in text[idx + len("@data") :].splitlines():
        line = line.strip()
        if line and not line.startswith("%"):
            rows.append([float(v) for v in line.split(",")])
    arr = np.array(rows, dtype=np.float64)
    if arr.shape[1] != 25:
        raise ValueError(f"expected 25 columns (id, x1..x23, y), got {arr.shape[1]}")
    return arr[:, 1:24], arr[:, 24]


@dataclass(slots=True)
class CreditSplit:
    seed: int
    test_frac: float
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray


def split_credit(paths: Paths) -> CreditSplit:
    X, y = load_credit(paths)
    rng = np.random.default_rng(CREDIT_SEED)
    n_test = max(1, int(X.shape[0] * CREDIT_TEST_FRAC))
    perm = rng.permutation(X.shape[0])
    test, train = perm[:n_test], perm[n_test:]
    log(f"  credit: {X.shape[0]} rows, split {len(train)} train / {len(test)} test")
    return CreditSplit(CREDIT_SEED, CREDIT_TEST_FRAC, X[train], y[train], X[test])
