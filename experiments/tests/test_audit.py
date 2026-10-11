"""audit-f32: native XGBoost's f32 raw margins against the stage oracle."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from treewalker_exp import audit
from treewalker_exp import formats as fm
from treewalker_exp import train as tr


@dataclass
class _Model:
    path: Path

    def dir(self, artifacts: Path) -> Path:
        return self.path


@dataclass
class _Cell:
    id: str
    model: _Model
    path: Path

    def dir(self, artifacts: Path) -> Path:
        return self.path


def xgboost_cell(tmp_path):
    """A 60-tree XGBoost model and a prepared cell of 100 groups of 4 rows, with its
    stage oracle."""
    import xgboost as xgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = X[:, 0] + 0.1 * rng.normal(size=400)
    bst = xgb.train(
        {"max_depth": 4, "eta": 0.3, "nthread": 1},
        xgb.DMatrix(X, label=y),
        num_boost_round=60,
    )
    (tmp_path / "xgboost").mkdir()
    bst.save_model(str(tmp_path / "xgboost" / "model_native.json"))
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    fm.write_matrix(cell_dir / "test_data.bin", X)
    offsets = np.arange(0, 401, 4, dtype=np.int64)  # 100 groups of 4
    rows, _ = tr.oracle(
        tr.load_treelite("xgboost", tmp_path / "xgboost" / "model_native.json"), X, offsets
    )
    fm.write_matrix(cell_dir / "oracle.bin", rows)
    names = {
        "model_native": "../xgboost/model_native.json",
        "oracle": "oracle.bin",
        "test_data": "test_data.bin",
    }
    record = {
        k: {"path": v, "sha256": fm.sha256_file((cell_dir / v).resolve())} for k, v in names.items()
    }
    fm.write_json(cell_dir / "cell.json", {"files": record})
    return _Cell("t/nt60_md4/xgboost/panel", _Model(tmp_path), cell_dir), rows


def test_native_xgboost_margins_are_within_f32_of_the_oracle(tmp_path):
    cell, rows = xgboost_cell(tmp_path)
    assert audit.unusable(cell, tmp_path) == []

    got = audit.audit_cell(cell, tmp_path)
    assert got["rows"] == len(rows) and 0 < got["rows"] <= 512
    # Native XGBoost sums 60 leaves in f32: off the correctly rounded reference by
    # at most a few f32 ulps of the margin, never more.
    assert 0 < got["max_rel"] < 60 * np.finfo(np.float32).eps


def test_a_file_from_another_prep_is_unusable(tmp_path):
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    (tmp_path / "xgboost").mkdir()
    for name in ("oracle.bin", "test_data.bin"):
        (cell_dir / name).write_bytes(b"x")
    (tmp_path / "xgboost" / "model_native.json").write_text("{}")
    names = {
        "model_native": "../xgboost/model_native.json",
        "oracle": "oracle.bin",
        "test_data": "test_data.bin",
    }
    record = {
        k: {"path": v, "sha256": fm.sha256_file((cell_dir / v).resolve())} for k, v in names.items()
    }
    fm.write_json(cell_dir / "cell.json", {"files": record})
    (cell_dir / "test_data.bin").write_bytes(b"y")  # replaced since the cell was prepared
    cell = _Cell("t/nt60_md4/xgboost/panel", _Model(tmp_path), cell_dir)
    assert audit.unusable(cell, tmp_path) == [(cell_dir / "test_data.bin").resolve()]


def test_closer_tie_and_farther_are_counted_against_the_same_reference():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    native = np.array([1.5, 2.0, 3.0, 4.5])
    treewalker = np.array([1.0, 2.0, 3.25, 3.5])
    got = audit.compare(treewalker, native, ref)
    assert (got["closer"], got["tie"], got["farther"]) == (1, 2, 1)
    assert got["differ"] == 2 and got["max_abs"] == 0.5


def write_run(runs, name, cell, stages=None, oracle_sha=None, exact_sums=False):
    """A packed run holding ``cell``'s manifest and, given, its stages table."""
    import polars as pl

    run = runs / name
    run.mkdir(parents=True)
    sha = oracle_sha or fm.read_json(cell.path / "cell.json")["files"]["oracle"]["sha256"]
    oracle = {"status": "pass", "rows": 400, "exact_sums": exact_sums}
    manifest = {"cell": {"files": {"oracle": {"sha256": sha}}}, "validation": {"oracle": oracle}}
    fm.write_json(run / "cells.json", {cell.id: manifest})
    if stages is not None:
        rows, margin = stages
        n = len(rows)
        pl.DataFrame(
            {"suite": ["s"] * n, "cell": [cell.id] * n, "group": rows // 4, "row": rows}
            | {"tree_sum": margin, "raw_margin": margin, "output": margin},
        ).with_columns(pl.col("cell").cast(pl.Categorical)).write_parquet(run / "stages.parquet")
    return run


def test_treewalkers_margins_come_from_the_runs_stages(tmp_path, monkeypatch):
    (tmp_path / "artifacts").mkdir()
    cell, oracle = xgboost_cell(tmp_path / "artifacts")
    rows, ref = oracle[:, 0].astype(np.int64), oracle[:, 2]
    artifacts, runs = tmp_path / "artifacts", tmp_path / "runs"
    # TreeWalker's margins: the reference itself, but 1e-9 off on the first 5 rows,
    # in reverse order in the table (the audit sorts by row).
    tw = ref.copy()
    tw[:5] += 1e-9
    write_run(runs, "s-x86_64", cell, (rows[::-1], tw[::-1]))
    write_run(runs, "s-aarch64", cell)  # an older run: no stages table

    stages = audit.run_stages(runs / "s-x86_64", [cell], artifacts)
    got_rows, got_margin, exact = stages[cell.id]
    assert np.array_equal(got_rows, rows) and np.array_equal(got_margin, tw) and exact is False
    assert audit.run_stages(runs / "s-aarch64", [cell], artifacts)[cell.id] == (
        "s-aarch64 has no stages table (output schema unrecorded)"
    )
    for schema, why in [
        (5, "s-aarch64 has no stages table (output schema 5)"),
        (6, "s-aarch64 lacks its required stages table (output schema 6)"),
    ]:
        fm.write_json(runs / "s-aarch64" / "run.json", {"output_schema": schema})
        assert audit.run_stages(runs / "s-aarch64", [cell], artifacts)[cell.id] == why
    (runs / "s-aarch64" / "run.json").unlink()
    assert audit.run_stages(runs / "s-none", [cell], artifacts)[cell.id] == "no run s-none"
    other = write_run(tmp_path / "other", "s-x86_64", cell, (rows, tw), oracle_sha="0" * 64)
    why = audit.run_stages(other, [cell], artifacts)[cell.id]
    assert why == "s-x86_64 ran it from another oracle"

    got = audit.audit_cell(cell, artifacts, {"intel": stages[cell.id], "arm": "why"})
    t = got["treewalker"]["intel"]
    native_exact = got["rows"] - got["differ"]
    assert t["closer"] + t["tie"] + t["farther"] == got["rows"] == len(rows)
    assert t["differ"] == 5 and abs(t["max_abs"] - 1e-9) < 1e-12 and t["exact_sums"] is False
    # Off the first 5 rows TreeWalker is exact: closer wherever native is not.
    assert t["tie"] >= native_exact - 5 and t["closer"] >= got["differ"] - 5
    assert got["treewalker"]["arm"] == {"missing": "why"}
    assert 0 < got["agree_f32"] <= got["rows"]

    monkeypatch.setattr(audit, "cells", lambda doc, suite: [cell])
    lines = audit.report(runs, artifacts, {}, "s")
    text = "\n".join(lines)
    assert "intel 1 of 1 cells passed, 400 rows (0 cells bit for bit, 1 with exact sums off" in text
    assert f"TreeWalker closer to the reference on {t['closer']:,}" in text
    assert "TreeWalker against native XGBoost, arm: none (s-aarch64 has no stages table" in text
    assert lines[-1].startswith(f"| {cell.id} | {len(rows)} |")
    assert lines[-1].endswith(f" | {t['closer']}/{t['tie']}/{t['farther']} | - |")
