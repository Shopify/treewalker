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


def test_native_xgboost_margins_are_within_f32_of_the_oracle(tmp_path):
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
    cell = _Cell("t/nt60_md4/xgboost/panel", _Model(tmp_path), cell_dir)
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
