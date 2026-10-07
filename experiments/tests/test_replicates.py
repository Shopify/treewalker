"""Seed replicates, expedia-filled, and the identities of every existing model and cell."""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from treewalker_exp import datasets as ds
from treewalker_exp import grids
from treewalker_exp import prepare as prep
from treewalker_exp import workloads as wl
from treewalker_exp.grids import Cell, Model
from treewalker_exp.paths import Paths

GRIDS = Path(__file__).resolve().parents[1] / "grids.toml"
NEW_WORKLOADS = {"panel-replicates", "whatif-credit-replicates", "ranking-replicates"}
NEW_WORKLOADS.add("ranking-filled")


def replicate(model: Model, k: int = 1) -> Model:
    return Model(
        model.dataset,
        model.n_trees,
        model.max_depth,
        model.horizon,
        replicate=k,
        replicate_fraction=0.8,
    )


# --- synthetic splits, in place of the datasets -------------------------------------


def survival_split(n: int = 61) -> ds.SurvivalSplit:
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "x0": rng.uniform(20, 80, n),
            "x1": rng.normal(size=n),
            "x2": rng.integers(0, 3, n).astype(np.float64),
            "duration": rng.uniform(1, 100, n),
            "event": rng.integers(0, 2, n).astype(np.float64),
        }
    )
    return ds.SurvivalSplit(ds.SurvivalSpec("support"), 7, 0.2, n, df[:49], df[49:], {})


def ranking_frame(sessions: range, rng: np.random.Generator, below_min: bool) -> pl.DataFrame:
    cols: dict[str, list] = {c: [] for c in ["srch_id", *ds.EXPEDIA_FEATURES, ds.EXPEDIA_LABEL]}
    for s in sessions:
        n = int(rng.integers(2, 7))
        session = {c: float(rng.integers(0, 5)) for c in ds.EXPEDIA_SESSION}
        for _ in range(n):
            cols["srch_id"].append(s)
            for c in ds.EXPEDIA_FEATURES:
                cols[c].append(session[c] if c in session else float(rng.integers(0, 9)))
            cols[ds.EXPEDIA_LABEL].append(float(rng.integers(0, 2)))
    df = pl.DataFrame(cols, schema={c: pl.Float64 for c in cols} | {"srch_id": pl.Int64})
    score = rng.uniform(0.5, 5.0, df.height)
    score[::3] = np.nan  # missing, as in prop_location_score2
    if below_min:
        score[1] = -5.0  # below every training value: kept
    review = df["prop_review_score"].to_numpy().copy()
    review[::7] = np.nan
    return df.with_columns(
        pl.Series("prop_location_score2", score).fill_nan(None),
        pl.Series("prop_review_score", review).fill_nan(None),
    )


def ranking_split() -> ds.RankingSplit:
    rng = np.random.default_rng(1)
    train = ranking_frame(range(0, 60, 2), rng, below_min=False)
    test = ranking_frame(range(1, 21, 2), rng, below_min=True)
    return ds.RankingSplit(11, 0.2, 0, "synthetic", False, train, test)


def credit_split() -> ds.CreditSplit:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(250, 23)) * 1000
    y = (X[:, 0] + X[:, 11] + rng.normal(size=250) * 500 > 0).astype(np.float64)
    return ds.CreditSplit(42, 0.2, X[:200], y[:200], X[200:])


@pytest.fixture
def ctx(tmp_path):
    c = prep.Context(
        Paths(tmp_path, tmp_path / "artifacts"),
        {"seed": 42, "test_frac": 0.2, "expedia_max_sessions": 0},
    )
    c._survival["support"] = survival_split()
    c._ranking = ranking_split()
    c._credit = credit_split()
    return c


# --- the sampler ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 10, 49, 7099, 24000, 40000])
def test_sampler_keeps_exactly_the_fraction_deterministically(n):
    seed = ds.replicate_seed(1886352030, 1)
    idx = ds.sample_entities(n, 0.8, seed)
    assert len(idx) == round(0.8 * n) == ds.replicate_size(n, 0.8)
    assert len(np.unique(idx)) == len(idx)  # without replacement
    assert np.all(np.diff(idx) > 0) and (len(idx) == 0 or (idx[0] >= 0 and idx[-1] < n))
    assert np.array_equal(idx, ds.sample_entities(n, 0.8, seed))
    if n >= 10:
        assert not np.array_equal(idx, ds.sample_entities(n, 0.8, ds.replicate_seed(1886352030, 2)))
    assert ds.replicate_size(7099, 0.8) == 5679 and ds.replicate_size(24000, 0.8) == 19200


def test_sampler_seed_is_its_own_stream():
    assert ds.replicate_seed(5, 1) == [5, ds.REPLICATE_STREAM, 1]
    assert ds.REPLICATE_STREAM != wl.SESSION_ORDER_STREAM
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ds.replicate_size(10, bad)


# --- replicates: entity-level samples, the released test data ---------------------------


def expected_rows(rows: np.ndarray, dataset_seed: int, k: int) -> np.ndarray:
    entities = np.unique(rows)
    chosen = entities[ds.sample_entities(len(entities), 0.8, ds.replicate_seed(dataset_seed, k))]
    return np.isin(rows, chosen)


def assert_whole_entities(rows: np.ndarray, keep: np.ndarray) -> None:
    for e in np.unique(rows):
        mine = keep[rows == e]
        assert mine.all() or not mine.any(), f"entity {e} split across the sample"


def test_survival_replicate_samples_whole_patients(ctx):
    released = Model("support", 5, 2, 4)
    td = ctx.train_data(released)
    rep = ctx.train_data(replicate(released))
    patients = ds.train_patients(td.extra["train_cov"], td.extra["edges"])
    keep = expected_rows(patients, 7, 1)
    assert_whole_entities(patients, keep)
    assert np.array_equal(rep.X, td.X[keep]) and np.array_equal(rep.y, td.y[keep])
    r = rep.identity["replicate"]
    assert r["entities"] == 49 and r["sampled"] == round(0.8 * 49) == len(np.unique(patients[keep]))
    assert r == {
        "index": 1,
        "fraction": 0.8,
        "seed": [7, ds.REPLICATE_STREAM, 1],
        "unit": "patient",
        "entities": 49,
        "sampled": 39,
        "rows": int(keep.sum()),
        "released_train_data_sha256": td.sha256,
    }
    # The released split and time bins; another k, another sample.
    assert rep.split == td.split and rep.horizon == td.horizon
    assert np.array_equal(rep.extra["edges"], td.extra["edges"])
    assert ctx.train_data(replicate(released, 2)).sha256 not in (rep.sha256, td.sha256)


def test_replicates_are_deterministic(ctx, tmp_path):
    again = prep.Context(ctx.paths, {"seed": 42, "test_frac": 0.2, "expedia_max_sessions": 0})
    again._survival["support"] = survival_split()
    again._ranking, again._credit = ranking_split(), credit_split()
    for m in (Model("support", 5, 2, 4), Model("expedia", 5, 2), Model("credit", 5, 2)):
        assert ctx.train_data(replicate(m, 3)).sha256 == again.train_data(replicate(m, 3)).sha256


def test_ranking_replicate_samples_whole_sessions(ctx):
    released = Model("expedia", 5, 2)
    td = ctx.train_data(released)
    rep = ctx.train_data(replicate(released))
    sids = ctx.ranking().train_df["srch_id"].to_numpy()
    keep = expected_rows(sids, 11, 1)
    assert_whole_entities(sids, keep)
    assert np.array_equal(rep.X, td.X[keep], equal_nan=True)
    assert rep.identity["replicate"]["unit"] == "session"
    assert rep.identity["replicate"]["sampled"] == round(0.8 * 30) == len(np.unique(sids[keep]))


def test_credit_replicate_samples_rows(ctx):
    rep = ctx.train_data(replicate(Model("credit", 5, 2)))
    assert rep.X.shape == (160, 23) and rep.identity["replicate"]["unit"] == "row"
    keep = expected_rows(np.arange(200), 42, 1)
    assert np.array_equal(rep.X, ctx.credit().train_X[keep])


def hashes(ctx, cell, td):
    return prep.build_workload(ctx, cell, td, {"key": "unused"}).hashes


@pytest.mark.parametrize(
    ("workload", "generator", "model", "params"),
    [
        ("panel", "panel-v2", Model("support", 5, 2, 4), ()),
        ("ranking", "ranking-sessions-v2", Model("expedia", 5, 2), ()),
        (
            "ranking-cohort",
            "ranking-cohort-v2",
            Model("expedia", 5, 2),
            (("min_candidates", 4), ("size", 2)),
        ),
    ],
)
def test_replicates_time_the_released_test_data(ctx, workload, generator, model, params):
    released = Cell(workload, generator, model, "lightgbm", params)
    rep = Cell(workload, generator, replicate(model), "lightgbm", params)
    want = hashes(ctx, released, ctx.train_data(model))
    got = hashes(ctx, rep, ctx.train_data(rep.model))
    assert got == want
    assert rep.dir(ctx.paths.artifacts) != released.dir(ctx.paths.artifacts)


def test_whatif_replicates_take_the_released_models_features(ctx):
    model = Model("credit", 5, 2)
    params = (("k", 2), ("G", 4))
    released = Cell("whatif-credit-core", "whatif-v2", model, "lightgbm", params)
    rep = Cell("whatif-credit-replicates", "whatif-v2", replicate(model), "lightgbm", params)
    # The replicate's build trains the released model it takes the features from.
    built = prep.build_workload(ctx, rep, ctx.train_data(rep.model), {"key": "unused"})
    assert built.workload.meta["perturbation"]["ranked_by"] == "credit/nt5_md2/lightgbm"
    assert built.hashes == hashes(ctx, released, ctx.train_data(model))
    assert (model.dir(ctx.paths.artifacts) / "lightgbm" / "model.json").exists()


def test_a_replicate_is_a_new_model_with_its_own_key(ctx):
    model = Model("support", 5, 2, 4)
    rep = replicate(model)
    assert rep.id == "support/nt5_md2_h4_r1" and rep.released == model == model.released
    a = prep.model_identity(model, "lightgbm", ctx.train_data(model))
    b = prep.model_identity(rep, "lightgbm", ctx.train_data(rep))
    assert set(b) - set(a) == {"replicate"} and b["replicate"]["fraction"] == 0.8
    assert {k: v for k, v in b.items() if k not in ("replicate", "train_data_sha256")} == {
        k: v for k, v in a.items() if k != "train_data_sha256"
    }


# --- expedia-filled ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lo", "fill"),
    [(0.0, -1.0), (0.25, -0.75), (-0.5, -1.5), (5.0, 0.0), (-3.0, -6.0), (16777217.0, 0.0)],
)
def test_fill_constant_is_below_the_minimum_in_f32(lo, fill):
    X = np.array([[lo], [lo + 2.0], [np.nan]])
    got = ds.fill_constants(X, ["c"])
    assert got[0] == fill
    assert np.float32(got[0]) < np.float32(lo)


def test_fill_constants_reject_what_cannot_be_encoded():
    with pytest.raises(ValueError):
        ds.fill_constants(np.array([[np.nan], [np.nan]]), ["empty"])
    with pytest.raises(ValueError):
        ds.fill_constants(np.array([[-2e38]]), ["f32 overflow"])


def test_fill_missing_replaces_only_missing_values():
    X = np.array([[1.0, np.nan], [np.nan, -9.0]])
    got = ds.fill_missing(X, np.array([-1.0, -2.0]))
    assert np.array_equal(got, [[1.0, -2.0], [-1.0, -9.0]]) and np.isnan(X[0, 1])


def test_expedia_filled_uses_the_training_minimum(ctx):
    released = ctx.train_data(Model("expedia", 5, 2))
    filled = ctx.train_data(Model("expedia-filled", 5, 2))
    assert not np.isnan(filled.X).any()
    nan = np.isnan(released.X)
    assert np.array_equal(filled.X[~nan], released.X[~nan])
    fills, record = ctx.expedia_fills()
    for name in ("prop_location_score2", "prop_review_score"):
        j = ds.EXPEDIA_FEATURES.index(name)
        lo = np.nanmin(released.X[:, j])
        assert fills[j] == lo - max(1.0, abs(lo)) == record["fill"][name]
        assert np.all(filled.X[nan[:, j], j] == fills[j])
        assert np.float32(fills[j]) < np.float32(lo)  # distinct after XGBoost's f32 conversion
    assert set(record["fill"]) == {"prop_location_score2", "prop_review_score"}
    assert filled.identity == {"missing_values": record}
    a = prep.model_identity(Model("expedia", 5, 2), "xgboost", released)
    b = prep.model_identity(Model("expedia-filled", 5, 2), "xgboost", filled)
    assert set(b) - set(a) == {"missing_values"} and b["split"] == a["split"]


def test_expedia_filled_sessions_are_expedias_filled(ctx):
    base, filled = ctx.sessions("expedia"), ctx.sessions("expedia-filled")
    fills, record = ctx.expedia_fills()
    j = ds.EXPEDIA_FEATURES.index("prop_location_score2")
    test = ctx.ranking().test_df["prop_location_score2"].to_numpy()
    assert np.nanmin(test) == -5.0 < fills[j]  # the constant is the training split's
    nan = np.isnan(base.X)
    assert nan.any() and not np.isnan(filled.X).any()
    assert np.array_equal(filled.X[~nan], base.X[~nan])  # -5 kept
    assert np.array_equal(filled.X[nan], fills[np.nonzero(nan)[1]])
    assert np.array_equal(filled.offsets, base.offsets)
    assert np.array_equal(filled.entities, base.entities)
    assert filled.config == base.config  # the varying features carry over
    assert filled.meta["missing_values"] == record
    assert wl.contracts(filled)["ok"]


def test_expedia_filled_has_no_whatif(ctx):
    m = Model("expedia-filled", 5, 2)
    with pytest.raises(ValueError):
        ctx.whatif_draws(m, ctx.train_data(m))


# --- identities of the existing models and cells -----------------------------------------


# The fields of every model identity at 8f4a9f9, before replicates and expedia-filled.
MODEL_IDENTITY_FIELDS = {
    "schema_version",
    "dataset",
    "framework",
    "source",
    "split",
    "n_trees",
    "max_depth",
    "horizon",
    "train",
    "categorical_features",
    "feature_order",
    "libraries",
    "train_data_sha256",
}


def test_released_models_keep_their_identity_fields(ctx):
    for m in (Model("support", 5, 2, 4), Model("expedia", 5, 2), Model("credit", 5, 2)):
        td = ctx.train_data(m)
        assert td.identity == {}
        assert set(prep.model_identity(m, "lightgbm", td)) == MODEL_IDENTITY_FIELDS


# sha256 of every suite's cells (ID, directory, workload, generator) at 8f4a9f9,
# leaving out the workloads added since.
SUITES_AT_8F4A9F9 = {
    "factorial": (1094, "3f44ce977469023c798105506b818952f9daed43a7aa2a493c83ed83de5fe2cf"),
    "ablation": (36, "68ba9c8ec70bdf7d173e8fa88eeceaeb331eaf0b1da4d0ffbfa27064b5dccf42"),
    "scenario-v1": (16, "04d464a93b06089049b0e5b002c4f69c844440b014c0818714ad2ba498628075"),
    "acceptance": (17, "44bdadcb250065276eca37972a656e537302d8e763998d8c356d960f3b3e5f3a"),
    "smoke": (31, "30a667eb2b8be7abf433820b624ac530e2791f452d89aaf406dc52a8f17d0872"),
}


def test_existing_cells_keep_their_ids_and_order():
    doc = grids.load(GRIDS)
    for name, (n, digest) in SUITES_AT_8F4A9F9.items():
        cells = [c for c in grids.suite(doc, name).cells if c.workload not in NEW_WORKLOADS]
        text = "\n".join(f"{c.id} {c.dir(Path('A'))} {c.workload} {c.generator}" for c in cells)
        assert (len(cells), hashlib.sha256(text.encode()).hexdigest()) == (n, digest), name
        assert all(c.model.replicate == 0 and c.model.replicate_fraction == 0.0 for c in cells)


def test_factorial_adds_the_replicates_and_expedia_filled():
    s = grids.suite(grids.load(GRIDS), "factorial")
    new = [c for c in s.cells if c.workload in NEW_WORKLOADS]
    assert s.cells[-len(new) :] == new  # appended: existing cells keep their order
    models = {(c.model, c.framework) for c in new}
    assert len(new) == len(models) == 40
    reps = [c for c in new if c.model.replicate]
    assert len(reps) == 24
    assert {c.model.released.id for c in reps} == {
        "support/nt500_md4_h16",
        "credit/nt500_md4",
        "expedia/nt500_md8",
    }
    assert {c.model.replicate for c in reps} == {1, 2, 3, 4}
    assert {c.workload_id for c in reps if c.model.dataset == "credit"} == {"whatif-v2-k4-G16"}
    released_ids = {c.id for c in s.cells if c.workload not in NEW_WORKLOADS}
    for c in reps:  # the released cell is in the suite
        assert c.id.replace(f"_r{c.model.replicate}/", "/") in released_ids
    filled = [c for c in new if c.model.dataset == "expedia-filled"]
    assert len(filled) == 16 and {c.workload_id for c in filled} == {"sessions"}
    assert {(c.model.n_trees, c.model.max_depth) for c in filled} == {
        (t, d) for t in (50, 500, 1000, 2000) for d in (2, 4)
    }
    assert all(s.plan_for(c)["methods"] == "factorial" for c in new)


@pytest.mark.parametrize(
    "bad",
    [
        {"replicates": [1, 2]},
        {"replicates": [0], "replicate_fraction": 0.8},
        {"replicates": [1, 1], "replicate_fraction": 0.8},
        {"replicates": [1], "replicate_fraction": 1.0},
        {"replicate_fraction": 0.8},
        {"replicates": [1], "replicate_fraction": 0.8, "generator": "whatif-v1"},
    ],
)
def test_replicate_settings_are_checked(bad):
    w = {"generator": "panel-v2", "datasets": ["support"], "n_trees": [5], "max_depth": [2]}
    with pytest.raises(ValueError):
        grids._workload_cells("w", {**w, **bad}, {"frameworks": ["lightgbm"]})


def test_one_id_cannot_name_two_models():
    doc = grids.load(GRIDS)
    w = dict(doc["workloads"]["panel-replicates"], replicate_fraction=0.5)
    doc["workloads"]["panel-replicates-half"] = w
    doc["suites"]["factorial"]["workloads"].append("panel-replicates-half")
    with pytest.raises(ValueError, match="names two models"):
        grids.suite(doc, "factorial")
