"""Preparation: models, workloads and cells, each rebuilt unless its manifest matches.

Three identities stay separate:

- a model (``<model>/<framework>/model.json``) is keyed by dataset, split,
  training parameters, library versions and the training data's hash;
- a workload is its data: entity selection, G, k and feature classification,
  keyed by the hashes of its files;
- a cell (``cell.json``) is one workload on one model.

Prep reuses a model or cell only when the stored key matches the one it would
build and every file still has its recorded hash. Anything else is rebuilt.
"""

from dataclasses import dataclass, field

import numpy as np

from .. import datasets as ds
from ..grids import Cell, Model
from .cells import ensure_cell
from .context import Context
from .fixtures import ensure_fixture
from .models import ensure_derived, ensure_model
from .store import put_matrix

log = ds.log


# --- the run ---------------------------------------------------------------------------


@dataclass(slots=True)
class Report:
    models: int = 0
    cells: dict[str, int] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)

    def add(self, cell_id: str, status: str) -> None:
        self.cells[status] = self.cells.get(status, 0) + 1
        if status == "failed":
            self.failed.append(cell_id)


def prepare(ctx: Context, cells: list[Cell], force: bool = False) -> Report:
    by_model: dict[Model, list[Cell]] = {}
    for c in cells:
        by_model.setdefault(c.model, []).append(c)
    # The slowest models first, so limits and memory problems surface early.
    order = sorted(by_model, key=lambda m: (-m.cost(), m))
    report = Report()
    for i, model in enumerate(order, 1):
        model_cells = by_model[model]
        log(f"[{i}/{len(order)}] {model.id}: {len(model_cells)} cells")
        done: set[str] = set()
        if model.fixture is not None:
            for c in model_cells:
                report.add(c.id, ensure_fixture(ctx, c)["status"])
            continue
        try:
            trained = model.trained
            td = ctx.train_data(trained)
            if trained.layout == "standard":
                train = np.column_stack([td.X, td.y])
                put_matrix(trained.dir(ctx.paths.artifacts) / "train_data.bin", train)
            frameworks = sorted({c.framework for c in model_cells})
            mdocs = {fw: ensure_model(ctx, trained, fw, td, force) for fw in frameworks}
            if model.derive:
                mdocs = {fw: ensure_derived(ctx, model, fw, mdocs[fw], force) for fw in frameworks}
            report.models += len(frameworks)
            for c in sorted(model_cells, key=lambda c: c.id):
                doc = ensure_cell(ctx, c, td, mdocs[c.framework], force)
                report.add(c.id, doc["status"])
                done.add(c.id)
                log(f"    {c.id}: {doc['status']}")
        except Exception as e:
            log(f"  FAILED {model.id}: {e}")
            for c in model_cells:
                if c.id not in done:
                    report.add(c.id, "failed")
    return report
