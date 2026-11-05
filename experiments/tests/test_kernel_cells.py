"""The kernel suite's cells: single rows, layout calibration and derived models."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import treelite
from test_replicates import credit_split, ranking_split, survival_split

from treewalker_exp import datasets as ds
from treewalker_exp import formats as fm
from treewalker_exp import grids, manifest
from treewalker_exp import prepare as prep
from treewalker_exp import train as tr
from treewalker_exp import workloads as wl
from treewalker_exp.grids import Cell, Model, Suite
from treewalker_exp.paths import Paths
from treewalker_exp.prepare import cells as pcells
from treewalker_exp.prepare import context as pcontext
from treewalker_exp.prepare import fixtures as pfixtures
from treewalker_exp.prepare import models as pmodels

GRIDS = Path(__file__).resolve().parents[1] / "grids.toml"
KERNEL = {
    "kernel-g1",
    "kernel-layout",
    "kernel-derived",
    "kernel-derived-g1",
    "kernel-grid-credit",
    "kernel-grid-flchain",
    "kernel-whatif-credit",
}
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "import"


@pytest.fixture
def ctx(tmp_path):
    c = pcontext.Context(
        Paths(tmp_path, tmp_path / "artifacts"),
        {"seed": 42, "test_frac": 0.2, "expedia_max_sessions": 0},
    )
    c._survival["support"] = survival_split()
    c._ranking = ranking_split()
    c._credit = credit_split()
    return c


# --- grids ----------------------------------------------------------------------------


def test_the_kernel_suite_has_the_new_cells_only():
    doc = grids.load(GRIDS)
    s = grids.suite(doc, "kernel")
    ids = {c.id for c in s.cells}
    assert len(s.cells) == 60 and {c.workload for c in s.cells} == KERNEL
    for d in ("support/nt500_md8_h16", "flchain/nt500_md8_h16", "credit/nt500_md8"):
        for fw in ("lightgbm", "xgboost"):
            assert {f"{d}/{fw}/g1-independent", f"{d}/{fw}/g1-file"} <= ids
    assert "expedia/nt500_md8/xgboost/g1-file" in ids
    assert "flchain/nt500_md8_h16/xgboost/panel-share45" in ids
    for m in ("pooled", "wide"):
        for w in ("panel", "g1-independent"):
            assert f"support/nt500_md8_h16_{m}/lightgbm/{w}" in ids
    assert {s.plan_for(c).methods for c in s.cells} == {"treewalker"}
    ids_of = {c.id: [v.id for v in s.plan_for(c).variants] for c in s.cells}
    policy = ["all-on", "lanes=4", "lanes=8"]
    assert ids_of["support/nt500_md8_h16/lightgbm/panel-share45"] == policy
    assert ids_of["credit/nt500_md8/xgboost/g1-file"] == [*policy, "disable_g1_constant"]
    rows = [c for c in s.cells if c.generator == "rows-v1"]
    assert {c.param["rows"] for c in rows} == {20_000}
    derived = [c for c in s.cells if c.model.derive]
    assert {c.model.trained.id for c in derived} == {"support/nt500_md8_h16"}
    # Groups past 1,024 rows, on runtime-width masks: two grid sizes per model and framework.
    grid = {c.id for c in s.cells if c.generator == "whatif-grid-v1"}
    assert grid == {
        f"{m}/{fw}/whatif-grid-v1-k2-G{g}"
        for m in ("credit/nt500_md8", "flchain/nt50_md4_h16")
        for fw in ("lightgbm", "xgboost")
        for g in (2048, 4096)
    }
    # Scattered leaves: credit what-ifs at k = 1 and 8, G = 128, both frameworks.
    scattered = {c.id for c in s.cells if c.workload == "kernel-whatif-credit"}
    assert scattered == {
        f"credit/nt500_md8/{fw}/whatif-v2-k{k}-G128"
        for fw in ("lightgbm", "xgboost")
        for k in (1, 8)
    }
    # No other suite names a kernel cell.
    for name in doc["suites"]:
        if name != "kernel":
            assert not KERNEL & {c.workload for c in grids.suite(doc, name).cells}, name


def test_derived_models_plan_treewalker_in_any_suite():
    doc = grids.load(GRIDS)
    doc["suites"]["x"] = {"description": "", "modes": ["serving"], "workloads": ["kernel-derived"]}
    s = grids.suite(doc, "x")
    assert {s.plan_for(c).methods for c in s.cells} == {"treewalker"}


@pytest.mark.parametrize(
    "bad",
    [
        {"generator": "rows-v1", "row_order": ["independent"]},  # no rows
        {"generator": "rows-v1", "row_order": ["shuffled"], "rows": 10},
        {"generator": "rows-v1", "row_order": [], "rows": 10},
        {"generator": "rows-v1", "row_order": ["file", "file"], "rows": 10},
        {"generator": "rows-v1", "rows": 0},
        {"varying_share": [0.4, 0.401]},
        {"varying_share": [1.0]},
        {"varying_share": [1]},
        {"varying_share": [0.4], "generator": "whatif-v2", "k": [1], "G": [4]},
        {"derive": ["pooled"]},
        {"derive": []},
        {"derive": ["wide-sums"], "replicates": [1], "replicate_fraction": 0.8},
    ],
)
def test_kernel_settings_are_checked(bad):
    w = {"generator": "panel-v2", "datasets": ["support"], "n_trees": [5], "max_depth": [2]}
    with pytest.raises(ValueError):
        grids._workload_cells("w", {**w, **bad}, {"frameworks": ["lightgbm"]})


def test_kernel_cell_names():
    m = Model("support", 5, 2, 4, derive="wide-sums")
    assert m.id == "support/nt5_md2_h4_wide" and m.trained == Model("support", 5, 2, 4)
    rows = Cell("w", "rows-v1", m, "xgboost", (("row_order", "file"), ("rows", 10)))
    assert rows.id == "support/nt5_md2_h4_wide/xgboost/g1-file"
    share = Cell("w", "panel-v2", m.trained, "lightgbm", (("varying_share", 0.35),))
    assert share.workload_id == "panel-share35"


# --- workloads --------------------------------------------------------------------------


def source(n_groups=30, width=4):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n_groups * width, 3))
    config = wl.walker_config(["a", "b", "c"], width, [2], [2], [])
    meta = {"generator": "panel-v2", "grouping": {"kind": "panel", "width": width}}
    return wl.Workload(X, None, config, np.arange(100, 100 + n_groups), meta)


def test_single_rows_in_file_order_keep_the_sources_order():
    src = source()
    w = wl.single_rows(src, "file", 50)
    assert np.array_equal(w.X, src.X[:50]) and w.offsets is None
    assert np.array_equal(w.entities, np.repeat(np.arange(100, 130), 4)[:50])
    assert w.config["max_group_width"] == 1
    assert w.config["varying_features"] == w.config["mono_inc_features"] == []
    assert w.meta["row_order"] == "file" and "row_order" not in w.meta["seeds"]
    assert wl.contracts(w)["ok"]
    assert len(wl.single_rows(src, "file", 1000).X) == 120  # every row the source has


def test_independent_rows_are_a_fixed_seed_shuffle():
    src = source()
    w = wl.single_rows(src, "independent", 50)
    rows = np.random.default_rng(wl.ROWS_SEED).permutation(120)[:50]
    assert np.array_equal(w.X, src.X[rows])
    assert np.array_equal(w.entities, np.repeat(src.entities, 4)[rows])
    assert np.array_equal(w.X, wl.single_rows(src, "independent", 50).X)
    assert not np.array_equal(rows, np.sort(rows))
    assert w.meta["seeds"]["row_order"] == wl.ROWS_SEED
    assert w.meta["source"] == {
        "generator": "panel-v2",
        "grouping": {"kind": "panel", "width": 4},
        "rows": 120,
    }
    with pytest.raises(ValueError):
        wl.single_rows(src, "shuffled", 50)


def test_varying_for_share_takes_the_closest_subset():
    counts = np.array([40, 30, 20, 6, 3, 1, 0])
    assert wl.varying_for_share(counts, [0], 0.4) == []
    # 40 of 100 nodes. 74 is 30 + 3 + 1; 75 is missed by one either way, 30 + 6 = 76
    # and 30 + 3 + 1 = 74, and the lower subset index wins.
    assert wl.varying_for_share(counts, [0], 0.74) == [1, 4, 5]
    assert wl.varying_for_share(counts, [0], 0.75) == [1, 3]
    assert wl.varying_for_share(counts, [0], 0.66) == [2, 3]
    assert 6 not in wl.varying_for_share(counts, [0], 1.0)  # never split on
    with pytest.raises(ValueError):
        wl.varying_for_share(np.ones(30, dtype=np.int64), [0], 0.5)


def categorical_split(n: int = 1500) -> ds.SurvivalSplit:
    """A survival split whose x2 is a category the models split on."""
    rng = np.random.default_rng(4)
    x2 = rng.integers(0, 4, n).astype(np.float64)
    df = pl.DataFrame(
        {
            "x0": rng.uniform(20, 80, n),
            "x1": rng.normal(size=n),
            "x2": x2,
            "duration": rng.uniform(1, 100, n) * np.array([0.2, 1.0, 0.5, 3.0])[x2.astype(int)],
            "event": rng.integers(0, 2, n).astype(np.float64),
        }
    )
    spec = ds.SurvivalSpec("support", cat_cols=("x2",))
    return ds.SurvivalSplit(spec, 7, 0.2, n, df[: n * 4 // 5], df[n * 4 // 5 :], {})


# --- prepared cells ---------------------------------------------------------------------


def built(ctx, cell):
    m = cell.model.trained
    td = ctx.train_data(m)
    pmodels.ensure_model(ctx, m, cell.framework, td, force=False)
    if cell.model.derive:
        trained = pmodels.ensure_model(ctx, m, cell.framework, td, force=False)
        pmodels.ensure_derived(ctx, cell.model, cell.framework, trained, force=False)
    return pcells.build_workload(ctx, cell, td, {"key": "unused"}).workload


@pytest.mark.parametrize(
    ("model", "source_generator", "params"),
    [
        (Model("support", 5, 2, 4), "panel-v2", ()),
        (Model("expedia", 5, 2), "ranking-sessions-v2", ()),
        (Model("credit", 5, 2), "whatif-v2", (("k", 4), ("G", 64))),
    ],
)
def test_single_rows_come_from_the_datasets_group_workload(ctx, model, source_generator, params):
    for order in grids.ROW_ORDERS:
        cell = Cell("kernel-g1", "rows-v1", model, "lightgbm", (("row_order", order), ("rows", 40)))
        w = built(ctx, cell)
        src = built(ctx, Cell("x", source_generator, model, "lightgbm", params))
        want = wl.single_rows(src, order, 40)
        assert np.array_equal(w.X, want.X, equal_nan=True)
        assert np.array_equal(w.entities, want.entities)
        assert w.meta["source"]["spec"] == wl.ROWS_SOURCE[model.dataset]
        assert cell.dir(ctx.paths.artifacts) == ctx.paths.artifacts / model.id / "lightgbm" / (
            "cells/g1-" + order
        )


def test_a_layout_panel_declares_constant_covariates_varying(ctx):
    model = Model("support", 20, 3, 4)
    panel = built(ctx, Cell("panel", "panel-v2", model, "xgboost"))
    cell = Cell("kernel-layout", "panel-v2", model, "xgboost", (("varying_share", 0.5),))
    w = built(ctx, cell)
    tl = treelite.Model.deserialize(
        str(model.dir(ctx.paths.artifacts) / "xgboost/model_treelite.bin")
    )
    counts = tr.internal_nodes(tl)
    layout = w.meta["layout"]
    share = wl.varying_share(counts, w.config["varying_features"])
    assert layout["base_share"] < share == layout["varying_share"]
    assert abs(share - 0.5) < 0.5 - layout["base_share"]
    assert set(panel.config["varying_features"]) < set(w.config["varying_features"])
    names = w.config["feature_names"]
    assert set(layout["declared_varying"]) == {
        names[f] for f in set(w.config["varying_features"]) - set(panel.config["varying_features"])
    }
    assert np.array_equal(w.X, panel.X) and wl.contracts(w)["ok"]
    for k in ("mono_inc_features", "mono_dec_features", "max_group_width"):
        assert w.config[k] == panel.config[k]


def test_internal_nodes_match_the_json_dump():
    tl = treelite.Model.deserialize(str(FIXTURES / "random_forest.bin"))
    doc = fm.read_json(FIXTURES / "random_forest.json")
    want = np.zeros(2, dtype=np.int64)
    for t in doc["trees"]:
        for n in t["nodes"]:
            if "left_child" in n:
                want[n["split_feature_id"]] += 1
    assert np.array_equal(tr.internal_nodes(tl), want) and want.sum() > 0


def test_sum_headroom_follows_the_exact_sum_rule():
    # Leaves 0.25, -0.5, 0.5 and -1.0 over 2 trees: bits 2^0 down to 2^-2, plus 2.
    identity = treelite.Model.deserialize(str(FIXTURES / "identity.bin"))
    assert tr.sum_headroom(identity) == 126 - (1 + 2 + 2)
    # 1e300 and 1e-300: no i128 fixed point holds both.
    assert tr.sum_headroom(pfixtures._fallback_sums_model()) < 0


def test_derived_models_edit_the_trained_export(ctx):
    ctx._survival["support"] = categorical_split()
    trained = Model("support", 20, 3, 4)
    td = ctx.train_data(trained)
    assert td.cat == [2]
    art = ctx.paths.artifacts
    for fw in ("lightgbm", "xgboost"):
        tdoc = pmodels.ensure_model(ctx, trained, fw, td, force=False)
        tl = treelite.Model.deserialize(str(trained.dir(art) / fw / "model_treelite.bin"))
        cats = np.concatenate(
            [tl.get_tree_accessor(i).get_field("category_list") for i in range(20)]
        )
        assert len(cats) and cats.max() < 32
        X = built(ctx, Cell("panel", "panel-v2", trained, fw)).X
        dtype = np.float32 if tl.input_type == "float32" else np.float64

        pooled = Model("support", 20, 3, 4, derive="pooled-categories")
        doc = pmodels.ensure_derived(ctx, pooled, fw, tdoc, force=False)
        assert doc["derived_from"] == {"id": trained.id, "key": tdoc["key"]}
        assert doc["key"] != tdoc["key"] and set(doc["files"]) == {"treelite_bin"}
        pl = treelite.Model.deserialize(str(pooled.dir(art) / fw / "model_treelite.bin"))
        pcats = np.concatenate(
            [pl.get_tree_accessor(i).get_field("category_list") for i in range(20)]
        )
        assert np.array_equal(pcats, cats + pmodels.POOLED_OFFSET)
        w = built(ctx, Cell("kernel-derived", "panel-v2", pooled, fw))
        assert np.array_equal(w.X[:, 2], X[:, 2] + pmodels.POOLED_OFFSET)
        assert np.array_equal(np.delete(w.X, 2, axis=1), np.delete(X, 2, axis=1))
        same = treelite.gtil.predict(pl, w.X.astype(dtype))
        assert np.array_equal(same, treelite.gtil.predict(tl, X.astype(dtype)))

        wide = Model("support", 20, 3, 4, derive="wide-sums")
        doc = pmodels.ensure_derived(ctx, wide, fw, tdoc, force=False)
        assert doc["sum_headroom_bits"]["trained"] > 0 >= doc["sum_headroom_bits"]["derived"]
        wd = treelite.Model.deserialize(str(wide.dir(art) / fw / "model_treelite.bin"))
        for i in (8, 9, 19):
            a, b = tl.get_tree_accessor(i), wd.get_tree_accessor(i)
            scale = 2.0**pmodels.WIDE_SHIFT if i % 10 == 9 else 1.0
            assert np.array_equal(b.get_field("leaf_value"), a.get_field("leaf_value") * scale)
        assert np.array_equal(built(ctx, Cell("kernel-derived", "panel-v2", wide, fw)).X, X)
        # Reused while its key matches.
        assert pmodels.ensure_derived(ctx, wide, fw, tdoc, force=False) == doc


def test_kernel_cells_prepare_and_enter_the_manifest(ctx):
    ctx._survival["support"] = categorical_split()
    ctx.paths.grids.parent.mkdir(parents=True, exist_ok=True)
    ctx.paths.grids.write_text("schema = 1\n")
    trained = Model("support", 20, 3, 4)
    cells = [
        Cell("kernel-g1", "rows-v1", trained, "lightgbm", (("row_order", "file"), ("rows", 40))),
        Cell("kernel-layout", "panel-v2", trained, "lightgbm", (("varying_share", 0.45),)),
        Cell(
            "kernel-derived-g1",
            "rows-v1",
            Model("support", 20, 3, 4, derive="pooled-categories"),
            "xgboost",
            (("row_order", "independent"), ("rows", 40)),
        ),
        Cell(
            "kernel-derived", "panel-v2", Model("support", 20, 3, 4, derive="wide-sums"), "xgboost"
        ),
    ]
    report = prep.prepare(ctx, cells)
    assert report.cells == {"ready": 4}, report.failed
    docs = [fm.read_cell(c.dir(ctx.paths.artifacts) / "cell.json") for c in cells]
    assert [d.get("row_order") for d in docs] == ["file", None, "independent", None]
    # Model statistics under each cell's own varying features.
    assert docs[0]["model_stats"]["varying_share"] == 0.0
    assert docs[0]["model_stats"]["walk"]["always_split"]["phi"] == 0.0
    share = docs[1]["model_stats"]["varying_share"]
    assert share == docs[1]["layout"]["varying_share"] > 0
    walk = docs[1]["model_stats"]["walk"]
    assert (
        walk["never_split"]["phi"] > 0
        and walk["never_split"]["lambda"] < walk["always_split"]["lambda"]
    )
    assert docs[1]["layout"]["target_share"] == 0.45 and "row_order" not in docs[1]
    assert docs[2]["derived"] == {"model": "pooled-categories", "shifted": [2]}
    assert docs[3]["derivation"]["name"] == "wide-sums" and docs[3]["model"]["dir"].startswith(
        "../.."
    )
    entries = manifest.build(ctx.paths, Suite("kernel", "", cells))["cells"]
    assert [e["status"] for e in entries] == ["ready"] * 4
    assert [e.get("row_order") for e in entries] == ["file", None, "independent", None]
    # A second pass reuses everything.
    again = prep.prepare(ctx, cells)
    assert again.cells == {"ready": 4}
    assert [fm.read_cell(c.dir(ctx.paths.artifacts) / "cell.json")["key"] for c in cells] == [
        d["key"] for d in docs
    ]


def three_leaves():
    """One tree: x0 <= 0 is a leaf; otherwise x1 <= 0 picks one of two leaves."""
    from treelite.model_builder import Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation

    b = ModelBuilder(
        threshold_type="float64",
        leaf_output_type="float64",
        metadata=Metadata(
            num_feature=2,
            task_type="kRegressor",
            average_tree_output=False,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=TreeAnnotation(num_tree=1, target_id=[0], class_id=[0]),
        postprocessor=PostProcessorFunc(name="identity"),
        base_scores=[0.0],
    )
    b.start_tree()
    for node, feature, children in [(0, 0, (1, 2)), (2, 1, (3, 4))]:
        b.start_node(node)
        b.numerical_test(
            feature_id=feature,
            threshold=0.0,
            default_left=True,
            opname="<=",
            left_child_key=children[0],
            right_child_key=children[1],
        )
        b.end_node()
    for node in (1, 3, 4):
        b.start_node(node)
        b.leaf(float(node))
        b.end_node()
    b.end_tree()
    return b.commit()


def test_walk_shares_bound_the_events_per_step():
    tl = three_leaves()
    # No training counts: halves at every split. Never splitting, a group visits
    # the root (varying), a leaf and the x1 split half the time each, and their
    # leaves a quarter each: 2.5 steps, 1 varying and 1 leaf.
    got = tr.walk_shares(tl, {0})
    assert got["never_split"] == {"phi": 0.4, "lambda": 0.4, "events_per_step": 0.8}
    # Always splitting, both sides get the whole group: 4 steps, 1 varying, 2 leaves.
    assert got["always_split"] == {"phi": 0.25, "lambda": 0.5, "events_per_step": 0.75}
    none = tr.walk_shares(tl, set())
    assert (
        none["never_split"]
        == none["always_split"]
        == {
            "phi": 0.0,
            "lambda": 0.4,
            "events_per_step": 0.4,
        }
    )
    stats = pmodels.model_stats(tl, {0})
    assert stats["varying_share"] == 0.5 and stats["walk"] == got
    cfg = wl.walker_config(["a", "b", "c"], 4, [0], [2], [])
    assert pmodels.config_varying(cfg) == {0, 2}
