import json
import struct

import numpy as np
import pytest

from treewalker_exp import formats as fm


def test_matrix_round_trip(tmp_path):
    X = np.array([[1.0, -0.0, np.nan], [np.inf, 2.5, 1e-300]])
    p = tmp_path / "test_data.bin"
    fm.write_matrix(p, X)
    raw = p.read_bytes()
    assert struct.unpack("<QQ", raw[:16]) == (2, 3)
    Y = fm.read_matrix(p)
    assert Y.shape == (2, 3)
    assert np.array_equal(X.view("<u8"), np.asarray(Y).view("<u8"))  # bit for bit


def test_matrix_rejects_truncated_file(tmp_path):
    p = tmp_path / "test_data.bin"
    fm.write_matrix(p, np.ones((4, 2)))
    p.write_bytes(p.read_bytes()[:-8])
    with pytest.raises(ValueError):
        fm.read_matrix(p)


def test_group_offsets_round_trip(tmp_path):
    p = tmp_path / "group_offsets.bin"
    fm.write_group_offsets(p, np.array([0, 3, 4, 10]))
    assert struct.unpack("<Q", p.read_bytes()[:8]) == (3,)
    assert fm.read_group_offsets(p).tolist() == [0, 3, 4, 10]


@pytest.mark.parametrize("bad", [[1, 2], [0, 2, 2], [0, 3, 1]])
def test_group_offsets_contract(tmp_path, bad):
    with pytest.raises(ValueError):
        fm.write_group_offsets(tmp_path / "o.bin", np.array(bad))


def test_walker_config_bytes(tmp_path):
    cfg = {"n_features": 2, "max_group_width": 4, "varying_features": [1]}
    p = tmp_path / "walker_config.json"
    fm.write_walker_config(p, cfg)
    assert p.read_bytes() == json.dumps(cfg, indent=2).encode()  # no trailing newline
    assert fm.read_walker_config(p) == cfg


def test_cell_and_predictions_round_trip(tmp_path):
    doc = {"schema_version": 1, "id": "a/b", "k": None}
    fm.write_cell(tmp_path / "cell.json", doc)
    assert fm.read_cell(tmp_path / "cell.json") == doc
    preds = np.linspace(0, 1, 7)
    fm.write_predictions(tmp_path / "predictions.npy", preds)
    assert np.array_equal(fm.read_predictions(tmp_path / "predictions.npy"), preds)


def test_atomic_write_leaves_no_temp_files(tmp_path):
    fm.write_matrix(tmp_path / "x.bin", np.zeros((1, 1)))
    assert [p.name for p in tmp_path.iterdir()] == ["x.bin"]


def test_sha256_json_is_canonical():
    assert fm.sha256_json({"a": 1, "b": [1, 2]}) == fm.sha256_json({"b": [1, 2], "a": 1})
