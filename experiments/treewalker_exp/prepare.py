"""Preparation: train each model and write its workloads' data and references.

As in the released scripts, a model is trained only when its native file is
missing, an export or reference only when its file is missing, and ``force``
rebuilds everything. Data files are rewritten whenever their content differs.
"""

import hashlib
import io
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import datasets as ds
from . import formats as fm
from . import train as tr
from . import workloads as wl
from .grids import SURVIVAL, Cell, Model
from .paths import Paths

log = ds.log


def matrix_bytes(X: np.ndarray) -> bytes:
    return struct.pack("<QQ", *X.shape) + np.ascontiguousarray(X, dtype="<f8").tobytes()


def _put(path: Path, data: bytes) -> None:
    """Write ``data`` unless the file already holds it."""
    digest = hashlib.sha256(data).hexdigest()
    if not (path.exists() and path.stat().st_size == len(data) and fm.sha256_file(path) == digest):
        fm.write_bytes(path, data)


def put_matrix(path: Path, X: np.ndarray) -> None:
    _put(path, matrix_bytes(X))


def put_offsets(path: Path, offsets: np.ndarray) -> None:
    o = np.asarray(offsets, dtype="<u8")
    fm.check_offsets(o)
    _put(path, struct.pack("<Q", len(o) - 1) + o.tobytes())


def put_walker_config(path: Path, config: dict[str, Any]) -> None:
    _put(path, json.dumps(config, indent=2).encode())


def put_npy(path: Path, array: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, array)
    _put(path, buf.getvalue())


@dataclass(slots=True)
class TrainData:
    X: np.ndarray
    y: np.ndarray
    names: list[str]
    cat: list[int]
    horizon: int | None  # realized steps, survival only


class Context:
    """Datasets and splits, loaded once per run and shared by every model."""

    def __init__(self, paths: Paths, defaults: dict[str, Any]):
        self.paths = paths
        self.seed = defaults["seed"]
        self.test_frac = defaults["test_frac"]
        self.max_sessions = defaults["expedia_max_sessions"]
        self._survival: dict[str, ds.SurvivalSplit] = {}
        self._train: dict[tuple, TrainData] = {}
        self._ranking: ds.RankingSplit | None = None
        self._sessions: wl.Workload | None = None
        self._credit: ds.CreditSplit | None = None

    def survival(self, name: str) -> ds.SurvivalSplit:
        if name not in self._survival:
            self._survival[name] = ds.split_survival(self.paths, name, self.seed, self.test_frac)
        return self._survival[name]

    def ranking(self) -> ds.RankingSplit:
        if self._ranking is None:
            self._ranking = ds.split_expedia(
                self.paths, self.seed, self.test_frac, self.max_sessions
            )
        return self._ranking

    def credit(self) -> ds.CreditSplit:
        if self._credit is None:
            self._credit = ds.split_credit(self.paths)
        return self._credit

    def sessions(self) -> wl.Workload:
        if self._sessions is None:
            self._sessions = wl.ranking_sessions(self.ranking())
        return self._sessions

    def train_data(self, model: Model) -> TrainData:
        key = (model.dataset, model.horizon)
        if key in self._train:
            return self._train[key]
        if model.dataset in SURVIVAL:
            assert model.horizon is not None
            split = self.survival(model.dataset)
            edges = split.edges(model.horizon)
            X, y, cov = ds.expand_train(split, edges)
            td = TrainData(X, y, ds.feature_names(cov), cov.cat_indices, len(edges) - 1)
        elif model.dataset == "expedia":
            r = self.ranking()
            X = r.train_df.select(ds.EXPEDIA_FEATURES).to_numpy().astype(np.float64)
            y = r.train_df[ds.EXPEDIA_LABEL].to_numpy().astype(np.float64)
            td = TrainData(X, y, list(ds.EXPEDIA_FEATURES), [], None)
        else:
            c = self.credit()
            td = TrainData(c.train_X, c.train_y, list(ds.CREDIT_FEATURES), [], None)
        self._train[key] = td
        return td


def ensure_model(ctx: Context, model: Model, fw: str, td: TrainData, force: bool) -> None:
    fw_dir = model.dir(ctx.paths.artifacts) / fw
    fw_dir.mkdir(parents=True, exist_ok=True)
    native = fw_dir / tr.NATIVE_NAME[fw]
    if force or not native.exists():
        log(f"  {model.id}/{fw}: training on {td.X.shape[0]:,} rows")
        tr.train(fw, td.X, td.y, td.names, td.cat, model.n_trees, model.max_depth, native)
    json_path, bin_path = fw_dir / "model_treelite.json", fw_dir / "model_treelite.bin"
    if force or not json_path.exists() or not bin_path.exists():
        tr.export_treelite(fw, native, json_path, bin_path)


def prepare_cell(ctx: Context, cell: Cell, td: TrainData, force: bool) -> None:
    art = ctx.paths.artifacts
    fw_dir = cell.model.dir(art) / cell.framework
    native = fw_dir / tr.NATIVE_NAME[cell.framework]
    match cell.generator:
        case "panel-v1" | "ranking-sessions-v1":
            if cell.generator == "panel-v1":
                assert td.horizon is not None
                w = wl.panel(ctx.survival(cell.model.dataset).covariates("test"), td.horizon)
            else:
                w = ctx.sessions()
            model_dir = cell.model.dir(art)
            put_matrix(model_dir / "test_data.bin", w.X)
            put_walker_config(model_dir / "walker_config.json", w.config)
            if w.offsets is not None:
                put_offsets(model_dir / "group_offsets.bin", w.offsets)
            pred_path = fw_dir / "predictions.npy"
            if force or not pred_path.exists():
                ref = tr.Reference(cell.framework, native, td.names, td.cat)
                put_npy(pred_path, ref.predict(np.asarray(w.X)))
        case "whatif-v1":
            prepare_whatif_v1(ctx, cell, force)
        case _:
            raise ValueError(cell.generator)


def prepare_whatif_v1(ctx: Context, cell: Cell, force: bool) -> None:
    """The released credit scenario cell, written exactly as before."""
    import lightgbm as lgb
    import treelite

    k, G = cell.param["k"], cell.param["G"]
    fw_dir = cell.model.dir(ctx.paths.artifacts) / "lightgbm"
    cell_dir = cell.dir(ctx.paths.artifacts)
    files = ["test_data.bin", "group_offsets.bin", "walker_config.json", "reference.bin"]
    if not force and all((cell_dir / f).exists() for f in files):
        return
    bst = lgb.Booster(model_file=str(fw_dir / tr.NATIVE_NAME["lightgbm"]))
    pool = wl.whatif_v1_ranked_pool(bst.feature_importance(importance_type="gain"))
    perturb = sorted(pool[:k])
    base_X = ctx.credit().test_X[: wl.N_WHATIF_BASE]
    X, offsets = wl.whatif_v1(base_X, perturb, G)
    names = list(ds.CREDIT_FEATURES)
    config = {
        **wl.walker_config(names, G, perturb, [], []),
        # Provenance, not read by the engine.
        "scenario": {
            "k": k,
            "G": G,
            "perturbable_features": perturb,
            "perturbable_feature_names": [names[i] for i in perturb],
            "base_rows": int(offsets.shape[0] - 1),
            # Share of groups with at least one selected feature equal to 0;
            # such entries stay 0 under multiplicative shocks.
            "any_zero_group_fraction": wl.any_zero_group_fraction(base_X, perturb),
            "perturbation": "multiplicative log-uniform [0.5, 2.0], seed 42",
        },
    }
    put_matrix(cell_dir / "test_data.bin", X)
    put_offsets(cell_dir / "group_offsets.bin", offsets)
    put_walker_config(cell_dir / "walker_config.json", config)
    tl = treelite.Model.deserialize(str(fw_dir / "model_treelite.bin"))
    put_matrix(cell_dir / "reference.bin", tr.gtil(tl, X).reshape(-1, 1))


@dataclass(slots=True)
class Report:
    models: int = 0
    cells: int = 0
    failed: list[str] = field(default_factory=list)


def prepare(ctx: Context, cells: list[Cell], force: bool = False) -> Report:
    by_model: dict[Model, list[Cell]] = {}
    for c in cells:
        by_model.setdefault(c.model, []).append(c)
    # The slowest models first, so memory problems surface early.
    order = sorted(by_model, key=lambda m: (-m.cost(), m))
    report = Report()
    for i, model in enumerate(order, 1):
        model_cells = by_model[model]
        log(f"[{i}/{len(order)}] {model.id}: {len(model_cells)} cells")
        try:
            td = ctx.train_data(model)
            if model.layout == "standard":
                train = np.column_stack([td.X, td.y])
                put_matrix(model.dir(ctx.paths.artifacts) / "train_data.bin", train)
            for fw in sorted({c.framework for c in model_cells}):
                ensure_model(ctx, model, fw, td, force)
                report.models += 1
            for c in sorted(model_cells, key=lambda c: c.id):
                prepare_cell(ctx, c, td, force)
                report.cells += 1
        except Exception as e:
            log(f"  FAILED {model.id}: {e}")
            report.failed.append(model.id)
    return report
