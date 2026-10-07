"""The execution manifest reports a cell ready only while its files are intact."""

import numpy as np
from test_prepare import MODEL, NAMES, calls, ctx, train_data  # noqa: F401

from treewalker_exp import manifest
from treewalker_exp import prepare as prep
from treewalker_exp import workloads as wl
from treewalker_exp.grids import Cell, Suite

CELLS = [
    Cell("whatif-credit-full", "whatif-v2", MODEL, fw, (("k", 1), ("G", 4)))
    for fw in ("lightgbm", "xgboost")
]


def test_a_cell_whose_shared_file_changed_is_stale(ctx, calls, monkeypatch):  # noqa: F811
    """Two frameworks' cells share the workload's test data; rebuilding one
    rewrites it, and the other may no longer report ready."""
    state = {"shift": 0.0}

    def build(ctx, cell, td, mdoc):
        X = np.repeat(train_data().X[:20] + state["shift"], 4, axis=0)
        X[1::4, 0] += 1.0
        offsets = np.arange(21, dtype=np.uint64) * 4
        meta = {"generator": "whatif-v2", "grouping": {"kind": "whatif", "G": 4}}
        w = wl.Workload(X, offsets, wl.walker_config(NAMES, 4, [0], [], []), np.arange(20), meta)
        return prep.write_workload(w, MODEL.dir(ctx.paths.artifacts) / "shared")

    monkeypatch.setattr(prep, "build_workload", build)
    ctx.paths.grids.parent.mkdir(parents=True, exist_ok=True)
    ctx.paths.grids.write_text("schema = 1\n")
    td = train_data()
    for c in CELLS:
        mdoc = prep.ensure_model(ctx, MODEL, c.framework, td, force=False)
        prep.ensure_cell(ctx, c, td, mdoc, force=False)
    suite = Suite("t", "", CELLS)
    assert [c["status"] for c in manifest.build(ctx.paths, suite)["cells"]] == ["ready", "ready"]

    state["shift"] = 0.5
    mdoc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
    prep.ensure_cell(ctx, CELLS[0], td, mdoc, force=False)
    doc = manifest.build(ctx.paths, suite)
    assert [c["status"] for c in doc["cells"]] == ["ready", "stale"]
    assert doc["counts"] == {"ready": 1, "stale": 1}
