"""The categorical scale recipe is separate from the paper's numeric Expedia cells."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from treewalker_exp import datasets as ds
from treewalker_exp import grids, results
from treewalker_exp import train as tr
from treewalker_exp import workloads as wl
from treewalker_exp.paths import Paths
from treewalker_exp.prepare import cells, context, models

GRIDS = Path(__file__).resolve().parents[1] / "grids.toml"
FEATURES = [
    "site_id",
    "visitor_location_country_id",
    "srch_destination_id",
    "srch_length_of_stay",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_booking_window",
    "srch_saturday_night_bool",
    "random_bool",
    "price_usd",
]


def scale_cell():
    return grids.suite(grids.load(GRIDS), "categorical-scale").cells[0]


def test_scale_cell_has_a_leaf_limited_identity_and_its_own_suite():
    doc = grids.load(GRIDS)
    suite = grids.suite(doc, "categorical-scale")
    assert len(suite.cells) == 1
    cell = suite.cells[0]
    assert cell.id == "expedia-cat/nt600_md32_nl4096/lightgbm/cohort16-n16-g40000"
    assert cell.param == {"min_candidates": 16, "size": 16, "max_groups": 40_000}
    assert suite.plan_for(cell).methods == "treewalker"
    assert not any(c.model.dataset == "expedia-cat" for c in grids.suite(doc, "factorial").cells)
    assert replace(cell.model, num_leaves=2048).id != cell.model.id
    assert results.design(cell.id)["G"] == 16
    for leaves, frameworks in [
        ([1], ["lightgbm"]),
        ([4096], ["xgboost"]),
        ([4096, 4096], ["lightgbm"]),
    ]:
        doc["workloads"][cell.workload].update(num_leaves=leaves, frameworks=frameworks)
        with pytest.raises(ValueError, match="num_leaves"):
            grids.workload_cells(doc, cell.workload)


def test_scale_training_parameters_do_not_change_released_models():
    cell = scale_cell()
    recipe = tr.params("lightgbm", 600, 32, dataset="expedia-cat", num_leaves=4096)
    assert recipe == {
        "num_boost_round": 600,
        "params": {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 4096,
            "max_depth": 32,
            "learning_rate": 0.05,
            "cat_l2": 1.0,
            "min_data_per_group": 50,
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
            "seed": 42,
        },
    }
    assert cell.model.num_leaves == 4096
    numeric = tr.params("lightgbm", 600, 32)["params"]
    assert numeric["seed"] == 42 and numeric["learning_rate"] == 0.05
    assert numeric["num_leaves"] == 2**32 - 1
    with pytest.raises(ValueError, match="LightGBM only"):
        tr.params("xgboost", 600, 32, dataset="expedia-cat")


@pytest.fixture
def categorical_context(tmp_path):
    paths = Paths(tmp_path)
    paths.expedia.parent.mkdir(parents=True)
    # Non-contiguous IDs distinguish a permutation of IDs from their indices;
    # prices expose any shuffle of candidates. One search is nonuniform, one too
    # short, and one too long. Category IDs are deliberately not dense.
    lengths = [16] * 100 + [16, 1, 129]
    sids = np.repeat(np.arange(len(lengths)) * 3 + 7, lengths)
    data: dict[str, Any] = {name: sids % 5 for name in ds.EXPEDIA_SESSION}
    data.update(
        srch_id=sids,
        price_usd=np.arange(len(sids), dtype=float),
        click_bool=(sids % 2 == 0),
        srch_destination_id=100 + sids % 13,
    )
    data["site_id"] = data["site_id"].copy()
    data["site_id"][1600] += 1
    pl.DataFrame(data).write_parquet(paths.expedia)
    return context.Context(paths, {"seed": 999, "test_frac": 0.5, "expedia_max_sessions": 2})


def test_categorical_split_features_order_and_cohort(categorical_context):
    ctx = categorical_context
    r = ctx.ranking("expedia-cat")
    seed = ds.dataset_seed("expedia", 999)
    assert r.seed == seed and r.test_frac == 0.2 and r.max_sessions == 0
    ids = np.arange(100) * 3 + 7
    expected = sorted(np.random.default_rng(seed).permutation(ids)[:20])
    assert sorted(r.test_df["srch_id"].unique()) == expected
    assert set(r.train_df["srch_id"]).isdisjoint(expected)
    td = ctx.train_data(scale_cell().model)
    assert td.names == FEATURES and td.cat == list(range(7))
    assert td.X[:, 2].min() >= 100  # no category recoding
    sessions = ctx.sessions("expedia-cat")
    assert sessions.config["varying_features"] == [10]
    assert sessions.config["mono_inc_features"] == sessions.config["mono_dec_features"] == []
    assert np.all(np.diff(sessions.X[:, 10].reshape(-1, 16), axis=1) == 1)
    cohort = wl.ranking_cohort(sessions, 16, [16], max_groups=3)[16]
    assert cohort.entities.tolist() == expected[:3]
    np.testing.assert_array_equal(cohort.X, sessions.X[:48])
    assert wl.contracts(cohort)["ok"]


def test_tiny_categorical_cell_prepares_and_resumes(categorical_context):
    ctx = categorical_context
    cell = scale_cell()
    cell = replace(cell, model=replace(cell.model, n_trees=2, max_depth=4, num_leaves=8))
    td = ctx.train_data(cell.model)
    mdoc = models.ensure_model(ctx, cell.model, cell.framework, td, False)
    assert mdoc["train"]["params"]["num_leaves"] == 8
    assert mdoc["categorical_features"] == FEATURES[:7]
    doc = cells.ensure_cell(ctx, cell, td, mdoc, False)
    assert doc["status"] == "ready" and doc["contracts"]["ok"]
    assert doc["load_options"] == {"max_rank_table_bytes": 4 * 1024**3}
    assert doc["grouping"]["n_groups"] == 20
    assert {"model_treelite_bin", "test_data", "group_offsets", "entities", "oracle"} <= set(
        doc["files"]
    )
    assert cells.ensure_cell(ctx, cell, td, mdoc, False)["key"] == doc["key"]


def test_capped_cohorts_have_distinct_ids_and_paths(tmp_path):
    capped = scale_cell()
    uncapped = replace(capped, params=tuple((k, v) for k, v in capped.params if k != "max_groups"))
    smaller = replace(
        capped, params=tuple((k, 10 if k == "max_groups" else v) for k, v in capped.params)
    )
    assert uncapped.workload_id == "cohort16-n16"
    assert {c.workload_id for c in (capped, uncapped, smaller)} == {
        "cohort16-n16",
        "cohort16-n16-g10",
        "cohort16-n16-g40000",
    }
    assert len({c.dir(tmp_path) for c in (capped, uncapped, smaller)}) == 3


def test_existing_expedia_cell_identity_name_and_params_are_unchanged(monkeypatch):
    doc = grids.load(GRIDS)
    cell = next(
        c
        for c in grids.workload_cells(doc, "ranking-cohort")
        if c.model.n_trees == 500
        and c.model.max_depth == 8
        and c.framework == "lightgbm"
        and c.param["size"] == 16
    )
    assert cell.id == "expedia/nt500_md8/lightgbm/cohort32-n16"
    assert cell.model.name == "nt500_md8"
    assert cell.params == (("min_candidates", 32), ("size", 16))
    expected_train = {
        "params": {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 255,
            "max_depth": 8,
            "learning_rate": 0.05,
            "verbose": -1,
            "seed": 42,
        },
        "num_boost_round": 500,
    }
    assert tr.params("lightgbm", 500, 8, dataset="expedia") == expected_train
    monkeypatch.setattr(tr, "library_versions", lambda fw: {"lightgbm": "reference-version"})
    td = context.TrainData(
        np.zeros((1, 1)),
        np.zeros(1),
        ["price_usd"],
        [],
        {"seed": 42, "unit": "session"},
        {"fingerprint": "source-hash"},
        None,
        "train-hash",
    )
    assert models.model_identity(cell.model, "lightgbm", td) == {
        "schema_version": 1,
        "dataset": "expedia",
        "framework": "lightgbm",
        "source": {"fingerprint": "source-hash"},
        "split": {"seed": 42, "unit": "session"},
        "n_trees": 500,
        "max_depth": 8,
        "horizon": None,
        "train": expected_train,
        "categorical_features": [],
        "feature_order": ["price_usd"],
        "libraries": {"lightgbm": "reference-version"},
        "train_data_sha256": "train-hash",
    }
    w = wl.Workload(np.zeros((16, 1)), None, {"max_group_width": 16}, np.array([0]))
    built = cells.Built(w, {}, {"test_data": "data-hash", "walker_config": "config-hash"})
    assert cells.cell_identity(cell, {"key": "model-key"}, built) == {
        "schema_version": 1,
        "id": "expedia/nt500_md8/lightgbm/cohort32-n16",
        "workload": "ranking-cohort",
        "generator": "ranking-cohort-v2",
        "dataset": "expedia",
        "framework": "lightgbm",
        "n_trees": 500,
        "max_depth": 8,
        "params": {"min_candidates": 32, "size": 16},
        "model_key": "model-key",
        "data_sha256": {"test_data": "data-hash", "walker_config": "config-hash"},
    }


def test_file_order_truncation_boundaries_and_missing_values(tmp_path):
    paths = Paths(tmp_path)
    paths.expedia.parent.mkdir(parents=True)
    # All four boundaries occur in the held-out split. The final searches are
    # too long, too short, and nonuniform, respectively; none may enter it.
    lengths = [15, 16, 17, 128] * 25 + [129, 1, 16]
    sids = np.repeat(np.arange(len(lengths)) * 3 + 7, lengths)
    data = {name: sids % 5 for name in FEATURES[:-1]}
    df = pl.DataFrame(
        {
            **data,
            "srch_id": sids,
            "price_usd": np.arange(len(sids), dtype=float),
            "click_bool": sids % 2 == 0,
        }
    ).with_row_index("_row")
    bad_sid = 102 * 3 + 7
    df = df.with_columns(
        pl.when((pl.col("srch_id") == bad_sid) & (pl.col("_row") == len(sids) - 1))
        .then(999)
        .otherwise(pl.col("site_id"))
        .alias("site_id")
    )
    # Interleave searches and reverse their candidates relative to numeric price
    # order: sorting by a feature or by a post-join position cannot hide here.
    df = df[np.random.default_rng(9).permutation(len(df))]
    seed = ds.dataset_seed("expedia", 999)
    expected_ids = sorted(np.random.default_rng(seed).permutation(np.arange(100) * 3 + 7)[:20])
    missing_sid = next(sid for sid in expected_ids if lengths[(sid - 7) // 3] >= 16)
    source_rows = df.filter(pl.col("srch_id") == missing_sid)["_row"].to_list()
    df = df.with_columns(
        pl.when(pl.col("srch_id") == missing_sid)
        .then(None)
        .otherwise(pl.col("srch_room_count"))
        .alias("srch_room_count"),
        pl.when(pl.col("_row") == source_rows[0])
        .then(None)
        .when(pl.col("_row") == source_rows[1])
        .then(float("nan"))
        .otherwise(pl.col("price_usd"))
        .alias("price_usd"),
    ).drop("_row")
    df.write_parquet(paths.expedia)
    ctx = context.Context(paths, {"seed": 999, "test_frac": 0.5, "expedia_max_sessions": 2})
    split = ctx.ranking("expedia-cat")
    assert sorted(split.test_df["srch_id"].unique()) == expected_ids
    assert {15, 16, 17, 128} == set(split.test_df.group_by("srch_id").len()["len"])
    assert set(split.train_df["srch_id"]) | set(split.test_df["srch_id"]) == set(
        np.arange(100) * 3 + 7
    )
    sessions = ctx.sessions("expedia-cat")
    expected_cohort = [
        sid for sid in expected_ids if len(df.filter(pl.col("srch_id") == sid)) >= 16
    ]
    cohort = wl.ranking_cohort(sessions, 16, [16])[16]
    assert cohort.entities.tolist() == expected_cohort
    for g, sid in enumerate(expected_cohort):
        expected = (
            df.filter(pl.col("srch_id") == sid).head(16).select(FEATURES).to_numpy().astype(float)
        )
        np.testing.assert_array_equal(cohort.X[g * 16 : (g + 1) * 16], expected)
    missing_index = int(np.flatnonzero(sessions.entities == missing_sid)[0])
    missing_group = sessions.X[
        sessions.group_offsets[missing_index] : sessions.group_offsets[missing_index + 1]
    ]
    assert np.isnan(missing_group[:, 6]).all()
    assert np.isnan(missing_group[:2, 10]).all()  # null and NaN both survive as missing
    assert wl.contracts(cohort)["ok"]
