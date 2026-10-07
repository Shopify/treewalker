"""Write model.json and cell.json with an older treewalker-exp checkout, for
the synthetic model and what-if cell of test_prepare.py.

    PYTHONPATH=OLD_CHECKOUT/experiments python make_old_manifests.py OUT_DIR

MAX_PREDICATES is set to the cell's predicate count, so a version whose rule
rejected a count equal to the limit marks the cell unsupported.

- predicate-off-by-one/: the first treewalker-exp manifests, whose predicate
  rule rejected a count equal to the limit;
- no-policy/: the next version, which only warned but had no PREP_POLICY.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from treewalker_exp import prepare as prep
from treewalker_exp import train as tr
from treewalker_exp import workloads as wl
from treewalker_exp.grids import Cell, Model
from treewalker_exp.paths import Paths

NAMES = ["a", "b", "c"]
MODEL = Model("credit", 5, 2)
CELL = Cell("whatif-credit-full", "whatif-v2", MODEL, "lightgbm", (("k", 1), ("G", 4)))


def train_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + rng.normal(size=300) > 0).astype(np.float64)
    sha = hashlib.sha256(prep.matrix_bytes(np.column_stack([X, y]))).hexdigest()
    return prep.TrainData(
        X, y, NAMES, [], {"seed": seed, "test_frac": 0.2, "unit": "row"}, {"x": "y"}, None, sha
    )


def build(ctx, cell, td, mdoc):
    base = train_data().X[:20]
    X = np.repeat(base, 4, axis=0)
    X[1::4, 0] += 1.0
    offsets = np.arange(21, dtype=np.uint64) * 4
    meta = {"generator": "whatif-v2", "grouping": {"kind": "whatif", "G": 4}}
    w = wl.Workload(X, offsets, wl.walker_config(NAMES, 4, [0], [], []), np.arange(20), meta)
    return prep.write_workload(w, cell.dir(ctx.paths.artifacts))


prep.build_workload = build
tmp = Path(tempfile.mkdtemp())
ctx = prep.Context(
    Paths(tmp, tmp / "artifacts"), {"seed": 42, "test_frac": 0.2, "expedia_max_sessions": 0}
)
td = train_data()
mdoc = prep.ensure_model(ctx, MODEL, "lightgbm", td, force=False)
fw_dir = MODEL.dir(ctx.paths.artifacts) / "lightgbm"
n = tr.varying_predicates(ctx.tl_model(fw_dir / "model_treelite.bin"), {0})
tr.MAX_PREDICATES = n
cdoc = prep.ensure_cell(ctx, CELL, td, mdoc, force=False)
print(n, cdoc["status"], cdoc["limits"], cdoc.get("warnings"), file=sys.stderr)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "model.json").write_text(json.dumps(mdoc, indent=2) + "\n")
(out / "cell.json").write_text(json.dumps(cdoc, indent=2) + "\n")
