"""What a suite will cost before any VM is provisioned.

Counts come from the grids: models (a model shared by several cells counts
once), compilations and cells. Sizes come from each cell's prepared
`cell.json`; a seed replicate or a filled-Expedia cell, which times its twin's
test data, takes its twin's when its own is not here. A cell with neither is an
error naming it, and the suite then has no total. Samples on disk and wall time
scale from a finished pilot run (the validation deployment): its bytes per
sample, and its wall seconds per timed row-tree-level (both modes, warm passes
and validation included), since a row's cost grows with the trees and their
depth. The memory figure covers prepared models only.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import formats as fm
from . import grids
from .paths import Paths

# Methods a cell times besides its research variants: production, the full walk,
# and up to five more (tl2cgen, lleaves, native, QuickScorer, chunked128) in the
# factorial set; production alone in the ablation set.
OTHER_METHODS = {"factorial": 7, "ablation": 1, "treewalker": 2}


@dataclass(slots=True)
class Pilot:
    bytes_per_sample: float
    # Wall seconds per timed row, per tree and depth level.
    secs_per_row_tree_level: float

    @classmethod
    def from_run(cls, run_dir: Path) -> Pilot:
        """From a packed run; bytes per sample are the packed samples table's."""
        import polars as pl

        from . import pack

        pack.require_packed(run_dir)
        path = run_dir / "samples.parquet"
        s = pl.read_parquet(path, columns=["cell", "rows"])
        rows = dict(s.group_by("cell").agg(pl.col("rows").sum()).iter_rows())
        secs, work = 0.0, 0.0
        for cid, m in fm.read_json(run_dir / pack.CELLS).items():
            secs += m["seconds"]
            depth = max(int(m["cell"].get("max_depth", 1)), 1)
            work += float(rows.get(cid, 0)) * m["model"]["trees"] * depth
        return cls(
            bytes_per_sample=path.stat().st_size / max(len(s), 1),
            secs_per_row_tree_level=secs / max(work, 1.0),
        )


@dataclass(slots=True)
class Budget:
    suite: str
    models: int
    lightgbm: int
    xgboost: int
    cells: int
    # Cells sized, and of them, those sized from their data twin's cell.json.
    sized: int = 0
    from_twins: int = 0
    # Cells, and the sentinel, with no size: no total then.
    missing: list[str] = field(default_factory=list)
    groups: float = 0.0
    rows: float = 0.0
    samples: float = 0.0
    timed_rows: float = 0.0
    # Cells whose pool exceeds the row cap, timed as a group sample.
    sampled: int = 0
    # Rows a round times per method, summed over cells: the sample's, or the pool's.
    round_rows: float = 0.0
    child_processes: int = 0
    # Timed rows times trees times depth: what wall time scales with.
    work: float = 0.0
    disk_bytes: float = 0.0
    ram_bytes: float = 0.0
    sample_bytes: float | None = None
    hours: float | None = None
    notes: list[str] = field(default_factory=list)


def timed_fraction(rows: float, groups: float, run: dict[str, Any]) -> float:
    """The share of a pool a round times: 1, or above the row cap the sample's
    inclusion probability (sweep_bench's sample_groups), the cap's share of the
    rows raised to keep min_groups_per_round groups."""
    cap = int(run.get("max_rows_per_round", 0))
    if cap == 0 or rows <= cap or groups <= 0:
        return 1.0
    floor = max(int(run.get("min_groups_per_round", 0)), 1) / groups
    return min(1.0, max(cap / rows, floor))


def size_of(cd: dict[str, Any]) -> tuple[int, int, int]:
    g = cd["grouping"]
    return g["rows"], g["n_groups"], len(cd["feature_order"])


def data_twin(c: grids.Cell) -> grids.Cell | None:
    """The cell whose test data this one times, so whose size it has: a seed
    replicate's released cell, and an expedia-filled cell's Expedia cell (the
    same sessions with the missing values filled). None for any other cell."""
    model = c.model.released
    if model.dataset == "expedia-filled":
        model = replace(model, dataset="expedia")
    twin = replace(c, model=model)
    return twin if twin.id != c.id else None


def cell_size(paths: Paths, c: grids.Cell) -> tuple[tuple[int, int, int], bool] | None:
    """A cell's (rows, groups, features) from its cell.json, else its data
    twin's, and whether the twin's was used; None without either."""
    path = c.dir(paths.artifacts) / "cell.json"
    if path.exists():
        return size_of(fm.read_cell(path)), False
    twin = data_twin(c)
    if twin is not None and (twin.dir(paths.artifacts) / "cell.json").exists():
        return size_of(fm.read_cell(twin.dir(paths.artifacts) / "cell.json")), True
    return None


def estimate(
    paths: Paths, doc: dict[str, Any], name: str, pilot: Pilot | None, arch: str | None = None
) -> Budget:
    """The suite's budget. ``arch`` (std::env::consts::ARCH) adds XGBoost's extra
    processes where the run settings enable them, beside the sentinels."""
    run = grids.run_config(doc, (), name)
    rounds = int(run.get("min_rounds", 3))
    batch_rows = int(run.get("batch_rows", 1024))
    sv = grids.suite(doc, name)
    models = {(c.model, c.framework) for c in sv.cells if c.framework != "treelite"}
    b = Budget(
        suite=name,
        models=len(models),
        lightgbm=sum(1 for _, fw in models if fw == "lightgbm"),
        xgboost=sum(1 for _, fw in models if fw == "xgboost"),
        cells=len(sv.cells),
    )
    model_bytes: dict[Path, int] = {}
    for c in sv.cells:
        model = c.model.dir(paths.artifacts) / c.framework / "model_treelite.bin"
        if model.exists():
            model_bytes[model] = model.stat().st_size
    largest_model = max(model_bytes.values(), default=0)
    for c in sv.cells:
        known = cell_size(paths, c)
        if known is None:
            b.missing.append(c.id)
            continue
        (rows, groups, nf), twin = known
        b.sized += 1
        b.from_twins += twin
        plan = sv.plan_for(c)
        m = len(plan["variants"]) + OTHER_METHODS[plan["methods"]]
        modes = plan["modes"]
        # Serving calls, batch calls, or both, per the suite's modes.
        # A pool above the row cap times its sample's rows.
        f = timed_fraction(rows, groups, run)
        b.sampled += int(f < 1.0)
        trows, tgroups = f * rows, f * groups
        calls = tgroups * ("serving" in modes) + -(-trows // batch_rows) * ("batch" in modes)
        b.rows += rows
        b.groups += groups
        b.round_rows += trows
        b.samples += m * rounds * calls
        b.timed_rows += m * rounds * len(modes) * trows
        level = c.model.n_trees * max(c.model.max_depth, 1)
        b.work += m * rounds * len(modes) * trows * level
        # XGBoost alone in its extra processes: a warm-up round and a timed one.
        children = arch in run.get("xgboost_process_arches", []) if arch else False
        if c.framework == "xgboost" and plan["methods"] == "factorial" and children:
            k = int(run.get("xgboost_processes", 0))
            b.work += k * 2 * len(modes) * trows * level
            b.samples += k * calls
            b.timed_rows += k * len(modes) * trows
            b.child_processes += k
        data = 8 * rows * nf
        b.disk_bytes += data
        b.ram_bytes = max(b.ram_bytes, largest_model * (1 + len(plan["variants"])) + data)
    every = int(run.get("sentinel_every", 0))
    if every:
        # The sentinel: a warm-up and a measure, then one per `every` cells, each
        # the acceptance cell with every method, serving only.
        sentinels = 2 + b.cells // every
        sid = str(run.get("sentinel_cell"))
        sc = grids.find_cell(doc, sid)
        sentinel = cell_size(paths, sc) if sc is not None else None
        if sc is not None and sentinel is not None:
            (srows_all, sgroups_all, _), _ = sentinel
            level = sc.model.n_trees * max(sc.model.max_depth, 1)
            m = OTHER_METHODS["factorial"]
            # Above the row cap the sentinel too times its sample.
            sf = timed_fraction(srows_all, sgroups_all, run)
            srows, sgroups = sf * srows_all, sf * sgroups_all
            b.work += sentinels * m * rounds * srows * level
            b.samples += sentinels * m * rounds * sgroups
            b.timed_rows += sentinels * m * rounds * srows
            b.notes.append(f"{sentinels} sentinels, at least")
        else:
            b.missing.append(f"{sid} (the sentinel)")
    if b.sampled:
        b.notes.append(
            f"{b.sampled} cells above the row cap time a group sample: "
            f"{b.round_rows:,.0f} rows a round over all cells against {b.rows:,.0f} in the pools"
        )
    if b.child_processes:
        b.notes.append(
            f"{b.child_processes} XGBoost extra processes; their start-ups (loading the "
            "cell and the library) are not in the estimate"
        )
    b.disk_bytes += sum(model_bytes.values())
    if pilot is not None and not b.missing:
        b.sample_bytes = b.samples * pilot.bytes_per_sample
        b.hours = b.work * pilot.secs_per_row_tree_level / 3600
    return b


def describe(b: Budget, rounds: int) -> list[str]:
    lines = [
        f"{b.suite}: {b.models} models ({b.lightgbm} LightGBM, {b.xgboost} XGBoost), "
        f"{b.cells} cells: {b.sized} of {b.cells} sized from their cell.json "
        f"({b.from_twins} from their data twin's)",
        f"  compilations: {b.lightgbm + b.xgboost} tl2cgen, {b.lightgbm} lleaves, "
        f"{b.xgboost} QuickScorer XML",
    ]
    if b.missing:
        shown = ", ".join(b.missing[:10]) + (", ..." if len(b.missing) > 10 else "")
        return [*lines, f"  ERROR: {len(b.missing)} without a prepared size, so no total: {shown}"]
    lines += [
        f"  groups ~{b.groups:,.0f}, rows ~{b.rows:,.0f}; at least {b.samples:,.0f} samples "
        f"({rounds} rounds)",
        f"  artifacts on disk >= {b.disk_bytes / 1e9:.2f} GB; "
        f"largest prepared cell in RAM ~{b.ram_bytes / 1e9:.2f} GB",
    ]
    if b.sample_bytes is not None and b.hours is not None:
        lines.append(f"  samples, packed, ~{b.sample_bytes / 1e9:.2f} GB")
        lines.append(f"  wall time ~{b.hours:.1f} h at the pilot's rate, {rounds} rounds")
    return lines + [f"  note: {n}" for n in b.notes]
