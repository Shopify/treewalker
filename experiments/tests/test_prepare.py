"""Resume rules: a matching manifest is reused; any changed input rebuilds."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from treewalker_exp import formats as fm
from treewalker_exp import prepare as prep
from treewalker_exp import train as tr
from treewalker_exp import workloads as wl
from treewalker_exp.grids import Cell, Model
from treewalker_exp.paths import Paths

NAMES = ["a", "b", "c"]


def train_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + rng.normal(size=300) > 0).astype(np.float64)
    sha = hashlib.sha256(prep.matrix_bytes(np.column_stack([X, y]))).hexdigest()
    return prep.TrainData(
        X, y, NAMES, [], {"seed": seed, "test_frac": 0.2, "unit": "row"}, {"x": "y"}, None, sha
    )


@pytest.fixture
def ctx(tmp_path):
    paths = Paths(tmp_path, tmp_path / "artifacts")
    return prep.Context(paths, {"seed": 42, "test_frac": 0.2, "expedia_max_sessions": 0})


@pytest.fixture
def calls(monkeypatch):
    """Count training runs and reference predictions."""
    n = {"train": 0, "predict": 0}
    train, predict = tr.train, tr.Reference.predict

    def counted_train(*a, **k):
        n["train"] += 1
        return train(*a, **k)

    def counted_predict(self, X):
        n["predict"] += 1
        return predict(self, X)

    monkeypatch.setattr(tr, "train", counted_train)
    monkeypatch.setattr(tr.Reference, "predict", counted_predict)
    return n


MODEL = Model("credit", 5, 2)


def test_model_reused_only_when_its_manifest_matches(ctx, calls, monkeypatch):
    td = train_data()
    first = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    assert calls["train"] == 1
    assert prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)["key"] == first["key"]
    assert calls["train"] == 1  # reused

    prep.ensure_model(ctx, MODEL, "lightgbm", train_data(seed=1), force=False)
    assert calls["train"] == 2  # different training data

    td = train_data()
    prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    assert calls["train"] == 3
    versions = tr.library_versions("lightgbm")
    monkeypatch.setattr(tr, "library_versions", lambda fw: {**versions, "lightgbm": "9.9.9"})
    prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    assert calls["train"] == 4  # a library version changed

    prep.ensure_model(ctx, Model("credit", 6, 2), "lightgbm", td, force=False)
    assert calls["train"] == 5  # another model, another directory

    fw_dir = MODEL.dir(ctx.paths.artifacts) / "lightgbm"
    (fw_dir / "model_treelite.bin").write_bytes(b"tampered")
    prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    assert calls["train"] == 6  # a file no longer has its recorded hash

    prep.ensure_model(ctx, MODEL, "lightgbm", td, force=True)
    assert calls["train"] == 7


def test_json_export_only_for_listed_models(ctx, calls):
    td = train_data()
    doc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    fw_dir = MODEL.dir(ctx.paths.artifacts) / "lightgbm"
    assert not (fw_dir / "model_treelite.json").exists()
    assert (fw_dir / "model_treelite.bin").exists()
    assert doc["treelite_json"] == {"status": "skipped: not a test model"}
    assert "treelite_json" not in doc["files"]

    ctx.json_models = frozenset({MODEL.id})
    doc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=True)
    assert calls["train"] == 2
    assert doc["treelite_json"]["status"] == "written"
    assert doc["treelite_json"]["path"] == "model_treelite.json"
    assert (fw_dir / "model_treelite.json").exists()

    ctx.json_models = frozenset()
    prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    assert not (fw_dir / "model_treelite.json").exists()  # no stale dump left behind


def test_retraining_drops_compiled_baselines(ctx, calls):
    td = train_data()
    prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    fw_dir = MODEL.dir(ctx.paths.artifacts) / "lightgbm"
    (fw_dir / "tl2cgen.so").write_bytes(b"old library")
    prep.ensure_model(ctx, MODEL, "lightgbm", train_data(seed=2), force=False)
    assert not (fw_dir / "tl2cgen.so").exists()


CELL = Cell("whatif-credit-full", "whatif-v2", MODEL, "lightgbm", (("k", 1), ("G", 4)))


@pytest.fixture
def workload(monkeypatch):
    """A synthetic what-if workload in place of the dataset-backed generator."""
    state = {"shift": 0.0}

    def build(ctx, cell, td, mdoc):
        base = train_data().X[:20] + state["shift"]
        X = np.repeat(base, 4, axis=0)
        X[1::4, 0] += 1.0
        offsets = np.arange(21, dtype=np.uint64) * 4
        meta = {"generator": "whatif-v2", "grouping": {"kind": "whatif", "G": 4}}
        w = wl.Workload(X, offsets, wl.walker_config(NAMES, 4, [0], [], []), np.arange(20), meta)
        return prep.write_workload(w, cell.dir(ctx.paths.artifacts))

    monkeypatch.setattr(prep, "build_workload", build)
    return state


def test_cell_reused_only_when_its_manifest_matches(ctx, calls, workload):
    td = train_data()
    mdoc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    doc = prep.ensure_cell(ctx, CELL, td, mdoc, force=False)
    assert doc["status"] == "ready" and calls["predict"] == 1
    cell_dir = CELL.dir(ctx.paths.artifacts)
    assert fm.read_cell(cell_dir / "cell.json")["key"] == doc["key"]

    assert prep.ensure_cell(ctx, CELL, td, mdoc, force=False)["key"] == doc["key"]
    assert calls["predict"] == 1  # reused

    workload["shift"] = 0.5
    changed = prep.ensure_cell(ctx, CELL, td, mdoc, force=False)
    assert changed["key"] != doc["key"] and calls["predict"] == 2  # new data

    (cell_dir / "predictions.npy").write_bytes(b"tampered")
    prep.ensure_cell(ctx, CELL, td, mdoc, force=False)
    assert calls["predict"] == 3  # a file lost its recorded hash

    retrained = prep.ensure_model(ctx, MODEL, "lightgbm", train_data(seed=3), force=False)
    assert prep.ensure_cell(ctx, CELL, td, retrained, force=False)["model_key"] == retrained["key"]
    assert calls["predict"] == 4  # a new model


def test_contract_violation_fails_the_cell(ctx, calls, monkeypatch, workload):
    td = train_data()
    mdoc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    build = prep.build_workload

    def broken(ctx, cell, td, mdoc):
        built = build(ctx, cell, td, mdoc)
        built.workload.config["varying_features"] = []  # feature 0 varies but is declared constant
        return built

    monkeypatch.setattr(prep, "build_workload", broken)
    doc = prep.ensure_cell(ctx, CELL, td, mdoc, force=False)
    assert doc["status"] == "failed" and calls["predict"] == 0
    assert doc["contracts"]["violations"]


FIXTURES = Path(__file__).parent / "fixtures" / "manifests"


def as_written_by_old_code(name, model_doc, cell_doc):
    """An old-format manifest pair from fixtures/manifests/NAME, with the
    values that depend on this machine's libraries (keys, file hashes)
    taken from a fresh run of the same model and cell."""
    old_model = fm.read_json(FIXTURES / name / "model.json")
    old_cell = fm.read_json(FIXTURES / name / "cell.json")
    old_model.update({k: model_doc[k] for k in ("key", "libraries", "train_data_sha256")})
    for role, rec in model_doc["files"].items():
        old_model["files"][role] = rec
    # Before PREP_POLICY a cell's key was its content key.
    old_cell.update(key=cell_doc["content_key"], model_key=model_doc["key"])
    old_cell["data_sha256"] = cell_doc["data_sha256"]
    old_cell["model"]["key"] = model_doc["key"]
    for role, rec in cell_doc["files"].items():
        old_cell["files"][role] = rec
    return old_model, old_cell


@pytest.mark.parametrize("name", ["predicate-off-by-one", "no-policy"])
def test_old_manifests_are_reevaluated_on_resume(ctx, calls, workload, name):
    td = train_data()
    mdoc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    cdoc = prep.ensure_cell(ctx, CELL, td, mdoc, force=False)
    fw_dir = MODEL.dir(ctx.paths.artifacts) / "lightgbm"
    cell_dir = CELL.dir(ctx.paths.artifacts)
    old_model, old_cell = as_written_by_old_code(name, mdoc, cdoc)
    fm.write_json(fw_dir / "model.json", old_model)
    fm.write_cell(cell_dir / "cell.json", old_cell)
    if name == "predicate-off-by-one":
        assert old_cell["status"] == "unsupported" and old_cell["limits"]
        assert "varying_predicates" in old_cell["model"]
        assert "treelite_json_bytes" in old_model

    # An ordinary resume: no --force, no retraining, no new references.
    mdoc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    doc = prep.ensure_cell(ctx, CELL, td, mdoc, force=False)
    assert calls == {"train": 1, "predict": 1}
    assert doc["status"] == "ready" and doc["limits"] == [] and doc["warnings"] == []
    assert doc["prep_policy"] == prep.PREP_POLICY and doc["key"] == cdoc["key"]
    assert doc["model"]["varying_predicates_upper_bound"] == 4
    assert "varying_predicates" not in doc["model"]
    assert "model_treelite_json" not in doc["files"]
    model = fm.read_json(fw_dir / "model.json")
    assert model["prep_policy"] == prep.PREP_POLICY
    assert model["treelite_json"] == {"status": "skipped: not a test model"}
    assert "treelite_json_bytes" not in model and "treelite_json" not in model["files"]
    assert not (fw_dir / "model_treelite.json").exists()
    assert fm.read_cell(cell_dir / "cell.json") == doc


def test_allowlist_changes_never_retrain(ctx, calls):
    td = train_data()
    prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    fw_dir = MODEL.dir(ctx.paths.artifacts) / "lightgbm"
    json_path = fw_dir / "model_treelite.json"
    assert not json_path.exists()

    ctx.json_models = frozenset({MODEL.id})  # added: exported from the .bin
    doc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    assert calls["train"] == 1 and json_path.exists()
    assert doc["treelite_json"]["status"] == "written"
    assert doc["treelite_json"]["sha256"] == fm.sha256_file(json_path)
    assert fm.read_json(fw_dir / "model.json") == doc

    json_path.unlink()  # lost: exported again
    prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    assert calls["train"] == 1 and json_path.exists()

    ctx.json_models = frozenset()  # removed: deleted
    doc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    assert calls["train"] == 1 and not json_path.exists()
    assert doc["treelite_json"] == {"status": "skipped: not a test model"}


def test_refresh_deletes_a_stale_dump_over_the_limit(ctx, calls, monkeypatch):
    td = train_data()
    ctx.json_models = frozenset({MODEL.id})
    doc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    fw_dir = MODEL.dir(ctx.paths.artifacts) / "lightgbm"
    json_path = fw_dir / "model_treelite.json"
    json_path.write_text("stale")  # no longer matches its record
    monkeypatch.setattr(tr, "MAX_JSON_BYTES", doc["treelite_json"]["bytes"] - 1)
    doc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    assert calls["train"] == 1 and not json_path.exists()
    assert doc["treelite_json"]["status"].startswith("skipped: over")

    json_path.write_text("stale")  # a dump left behind once the record says skipped
    doc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    assert calls["train"] == 1 and not json_path.exists()
    assert doc["treelite_json"]["status"].startswith("skipped: over")
