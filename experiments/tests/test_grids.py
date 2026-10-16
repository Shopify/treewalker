"""Suites resolve into cells and per-cell run plans."""

from pathlib import Path

import pytest

from treewalker_exp import grids

GRIDS = Path(__file__).resolve().parents[1] / "grids.toml"


@pytest.fixture(scope="module")
def doc():
    return grids.load(GRIDS)


def test_ablation_variants_cross_only_at_the_interaction_anchors(doc):
    s = grids.suite(doc, "ablation")
    sizes = {c.id: len(s.plan[c.id]["variants"]) for c in s.cells}
    # 6 runtime variants, 6 kernel variants and 4 load flags alone; at the
    # interaction anchors every runtime variant crosses the 8 combinations of the
    # first three load flags. Single rows' constant walk only at G = 1.
    assert sizes["support/nt500_md8_h16/lightgbm/panel"] == 6 * 8 + 1 + 6
    assert sizes["flchain/nt1000_md16_h32/xgboost/panel"] == 55
    assert sizes["support/nt50_md4_h1/lightgbm/panel"] == 6 + 6 + 1 + 4
    assert sizes["support/nt500_md4_h16/lightgbm/panel"] == 6 + 6 + 4
    # Expedia at an interaction anchor's (T, L) crosses too.
    assert sizes["expedia/nt500_md8/lightgbm/sessions"] == 55
    assert sizes["expedia/nt50_md4/lightgbm/sessions"] == 16
    assert sum(sizes.values()) == 1048  # 948 before the kernel variants
    g1 = [v for v in s.plan["support/nt50_md4_h1/xgboost/panel"]["variants"] if v["runtime"]]
    assert {"runtime": ["disable_g1_constant"], "load": []} in g1
    kernel = [v for v in s.plan["expedia/nt500_md8/xgboost/sessions"]["variants"] if v["load"]]
    assert not [v for v in kernel if any("=" in f or "dispatch" in f for f in v["runtime"])]
    plan = s.plan["support/nt500_md8_h16/lightgbm/panel"]
    assert plan["methods"] == "ablation"
    dedup = [v for v in plan["variants"] if "disable_predicate_dedup" in v["load"]]
    assert dedup == [{"runtime": [], "load": ["disable_predicate_dedup"]}]


def test_variant_ids_match_the_runner():
    v = grids.variant(["disable_unsplit", "disable_monotonic"], ["disable_tree_ordering"])
    assert grids._variant_id(v) == "disable_monotonic+disable_unsplit|disable_tree_ordering"
    assert grids._variant_id(grids.variant()) == "all-on"
    assert (
        grids._variant_id(grids.variant((), ["disable_bitset_intern"])) == "|disable_bitset_intern"
    )
    with pytest.raises(ValueError):
        grids.variant(["disable_everything"])
    # The kernel's tokens sort as Ablation::id() sorts them.
    v = grids.variant(["lanes=4", "disable_unsplit"])
    assert grids._variant_id(v) == "disable_unsplit+lanes=4"
    assert (
        grids._variant_id(grids.variant(["lanes=3", "disable_monotonic"]))
        == "disable_monotonic+lanes=3"
    )
    for bad in (
        ["lanes=13"],
        ["distance=4"],
        ["rate=2"],
        ["layout=2e"],
        ["lanes=1", "lanes=2"],
        ["disable_exact_sums"],
    ):
        with pytest.raises(ValueError):
            grids.variant(bad)


def test_acceptance_plans_the_cells_and_fixtures(doc):
    s = grids.suite(doc, "acceptance")
    ids = {c.id for c in s.cells}
    assert "support/nt500_md4_h16/lightgbm/panel" in ids
    assert "fixtures/average_base/treelite/import" in ids
    lgb = s.plan_for(next(c for c in s.cells if c.id.startswith("support") and "lightgbm" in c.id))
    assert lgb["methods"] == "factorial"
    assert len(lgb["variants"]) == 2
    fixture = s.plan_for(next(c for c in s.cells if c.generator == "fixture-v1"))
    assert fixture["methods"] == "treewalker"


def test_factorial_cells_default_to_every_method(doc):
    s = grids.suite(doc, "factorial")
    c = s.cells[0]
    assert s.plan_for(c) == {
        "methods": "factorial",
        "variants": [],
        "modes": ["serving"],
    }
    assert grids.run_config(doc)["batch_rows"] > 0


def test_suites_declare_their_modes(doc):
    # Batch mode only in the ablation suites; every other suite serves only.
    for name in doc["suites"]:
        s = grids.suite(doc, name)
        ablation = name in ("ablation", "camera-ready-ablation")
        want = ["serving", "batch"] if ablation else ["serving"]
        assert {tuple(s.plan_for(c)["modes"]) for c in s.cells} == {tuple(want)}, name
    for bad in ({}, {"modes": []}, {"modes": ["warm"]}, {"modes": ["batch", "batch"]}):
        with pytest.raises(ValueError):
            grids.suite_modes("x", bad)
    assert grids.suite_modes("x", {"modes": ["batch", "serving"]}) == ["batch", "serving"]


def test_the_validation_suite_names_real_cells_and_its_own_settings(doc):
    s = grids.suite(doc, "validation")
    ids = {c.id for c in s.cells}
    assert "support/nt500_md4_h16/lightgbm/panel" in ids and len(ids) == 30
    assert {
        "support/nt500_md4_h16_r1/xgboost/panel",
        "expedia-filled/nt500_md4/xgboost/sessions",
    } < ids
    # The validation cells are the factorial's own cells, so their keys match.
    factorial = {c.id: c for c in grids.suite(doc, "factorial").cells}
    for c in s.cells:
        assert factorial[c.id] == c, c.id
    assert "flchain/nt1000_md2_h8/xgboost/panel" in ids
    run = grids.run_config(doc, (), "validation")
    assert run["xgboost_process_arches"] == ["x86_64", "aarch64"]
    assert run["sentinel_every"] == 10
    assert grids.run_config(doc, ("sentinel_every=5",), "validation")["sentinel_every"] == 5
    assert grids.run_config(doc)["sentinel_every"] == 25
    bad = {**doc, "suites": {**doc["suites"], "x": {"run": {"no_such": 1}}}}
    with pytest.raises(ValueError):
        grids.run_config(bad, (), "x")


def test_run_config_overrides_keep_keys_and_types():
    doc = grids.load(GRIDS)
    base = grids.run_config(doc)
    run = grids.run_config(doc, ("target_batches=6", "hardware_counters=false", "precision_pct=2"))
    assert run["target_batches"] == 6
    assert run["hardware_counters"] is False
    assert run["precision_pct"] == 2.0
    changed = {"target_batches", "hardware_counters", "precision_pct"}
    assert {k: v for k, v in run.items() if k not in changed} == {
        k: v for k, v in base.items() if k not in changed
    }
    for bad in ("no_such_key=1", "target_batches", "hardware_counters=1", "batch_rows=big"):
        with pytest.raises(ValueError):
            grids.run_config(doc, (bad,))


def test_the_sentinel_resolves_against_the_artifacts(doc, tmp_path):
    from treewalker_exp import formats as fm
    from treewalker_exp import manifest as mf
    from treewalker_exp.grids import sentinel_cell
    from treewalker_exp.paths import Paths

    run = grids.run_config(doc)
    paths = Paths(tmp_path, tmp_path / "artifacts")
    cell = grids.find_cell(doc, run["sentinel_cell"])
    assert cell is not None and cell.id == "support/nt500_md4_h16/lightgbm/panel"
    # Not prepared: recorded with a reason, never silently dropped.
    entry = mf.sentinel(paths, doc, run)
    assert entry["status"] == "skipped" and "not prepared" in entry["reason"]
    fm.write_cell(cell.dir(paths.artifacts) / "cell.json", {"status": "ready", "key": "k1"})
    entry = mf.sentinel(paths, doc, run)
    assert entry == {
        "id": cell.id,
        "dir": "support/nt500_md4_h16/lightgbm/cells/panel",
        "status": "ready",
        "key": "k1",
        **mf.SENTINEL_PLAN,
    }
    # A ready cell whose recorded file no longer has its hash is stale.
    fm.write_cell(
        cell.dir(paths.artifacts) / "cell.json",
        {"status": "ready", "key": "k1", "files": {"x": {"path": "x.bin", "sha256": "0" * 64}}},
    )
    (cell.dir(paths.artifacts) / "x.bin").write_bytes(b"changed")
    entry = mf.sentinel(paths, doc, run)
    assert entry["status"] == "stale" and entry["reason"] == "status stale"
    assert mf.sentinel(paths, doc, {**run, "sentinel_every": 0}) is None
    missing = mf.sentinel(paths, doc, {**run, "sentinel_cell": "no/such/cell"})
    assert missing["status"] == "skipped"
    # compile-baselines covers the sentinel's model in every suite.
    assert sentinel_cell(doc).id == cell.id


def test_camera_ready_suites_turn_exact_sums_off_and_keep_the_factorial():
    doc = grids.load(Path(__file__).parents[1] / "grids.toml")
    assert grids.run_config(doc, (), "factorial")["exact_sums"] is True
    assert grids.run_config(doc, (), "factorial")["min_rounds"] == 3
    for name in ("camera-ready", "camera-ready-ablation", "scenario-v1"):
        run = grids.run_config(doc, (), name)
        assert run["exact_sums"] is False and run["min_rounds"] == 4, name
    cr = grids.suite(doc, "camera-ready")
    gens = {c.generator for c in cr.cells}
    assert "whatif-v2" not in gens and not any(c.model.replicate for c in cr.cells)
    assert {c.model.horizon for c in cr.cells if c.model.dataset == "flchain"} >= {1, 128, 1024}
    ab = grids.suite(doc, "camera-ready-ablation")
    assert ab.name == "camera-ready-ablation"
    assert {c.id for c in ab.cells} == {c.id for c in grids.suite(doc, "ablation").cells}
    for plan in ab.plan.values():
        assert plan["methods"] == "ablation"
        assert not any("disable_exact_sums" in v["runtime"] for v in plan["variants"])
    sc = grids.suite(doc, "scenario-v1")
    assert len(sc.cells) == 16
    assert {tuple(v["runtime"]) for p in sc.plan.values() for v in p["variants"]} == {
        (),
        ("disable_varying_precompute",),
    }
