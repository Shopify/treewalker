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

import hashlib
import resource
import struct
import sys
import time
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

SCHEMA = 1
log = ds.log


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


# --- in-memory content hashes, matching the files the writers produce ----------------


def matrix_bytes(X: np.ndarray) -> bytes:
    return struct.pack("<QQ", *X.shape) + np.ascontiguousarray(X, dtype="<f8").tobytes()


def offsets_bytes(offsets: np.ndarray) -> bytes:
    o = np.asarray(offsets, dtype="<u8")
    return struct.pack("<Q", len(o) - 1) + o.tobytes()


def _put(path: Path, data: bytes) -> str:
    """Write ``data`` unless the file already holds it; return its hash."""
    digest = hashlib.sha256(data).hexdigest()
    if not (path.exists() and path.stat().st_size == len(data) and fm.sha256_file(path) == digest):
        fm.write_bytes(path, data)
    return digest


def put_matrix(path: Path, X: np.ndarray) -> str:
    return _put(path, matrix_bytes(X))


def put_offsets(path: Path, offsets: np.ndarray) -> str:
    fm.check_offsets(np.asarray(offsets, dtype=np.uint64))
    return _put(path, offsets_bytes(offsets))


def put_walker_config(path: Path, config: dict[str, Any]) -> str:
    import json

    return _put(path, json.dumps(config, indent=2).encode())


def put_npy(path: Path, array: np.ndarray) -> str:
    import io

    buf = io.BytesIO()
    np.save(buf, array)
    return _put(path, buf.getvalue())


def files_match(base: Path, files: dict[str, dict[str, str]]) -> bool:
    for rec in files.values():
        p = base / rec["path"]
        if not p.exists() or fm.sha256_file(p) != rec["sha256"]:
            return False
    return True


def rel(path: Path, start: Path) -> str:
    import os

    return os.path.relpath(path, start)


# --- model inputs ------------------------------------------------------------------


@dataclass(slots=True)
class TrainData:
    X: np.ndarray
    y: np.ndarray
    names: list[str]
    cat: list[int]
    split: dict[str, Any]
    source: dict[str, Any]
    horizon: dict[str, int] | None
    sha256: str  # of train_data.bin: features plus the label column
    extra: dict[str, Any] = field(default_factory=dict)
    # Fields only some models' identities have (IDENTITY_EXTRAS), added only when
    # present, so every other model keeps its key.
    identity: dict[str, Any] = field(default_factory=dict)


# replicate: a seed replicate's entity sample; missing_values: expedia-filled's encoding.
IDENTITY_EXTRAS = ("replicate", "missing_values")
# Datasets built from the Expedia split, one session per group.
RANKING = ("expedia", "expedia-filled")


class Context:
    """Datasets, splits and draws, loaded once per run and shared by every model."""

    def __init__(
        self, paths: Paths, defaults: dict[str, Any], json_models: frozenset[str] = frozenset()
    ):
        self.paths = paths
        self.json_models = json_models  # model IDs that also get a Treelite JSON export
        self.seed = defaults["seed"]
        self.test_frac = defaults["test_frac"]
        self.max_sessions = defaults["expedia_max_sessions"]
        self._survival: dict[str, ds.SurvivalSplit] = {}
        self._train: dict[tuple, TrainData] = {}
        self._ranking: ds.RankingSplit | None = None
        self._fills: tuple[np.ndarray, dict[str, Any]] | None = None
        self._sessions: dict[str, wl.Workload] = {}
        self._credit: ds.CreditSplit | None = None
        self._draws: dict[tuple, wl.WhatIfDraws] = {}
        self._tl: dict[Path, Any] = {}

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

    def expedia_fills(self) -> tuple[np.ndarray, dict[str, Any]]:
        """expedia-filled's constants, one per feature, from the training split,
        and their record: the rule, the constants of the columns with missing
        values, and how many each split has."""
        if self._fills is None:
            r = self.ranking()
            names = list(ds.EXPEDIA_FEATURES)
            train = r.train_df.select(names).to_numpy().astype(np.float64)
            test = r.test_df.select(names).to_numpy().astype(np.float64)
            fills = ds.fill_constants(train, names)
            n_train, n_test = np.isnan(train).sum(axis=0), np.isnan(test).sum(axis=0)
            cols = [j for j in range(len(names)) if n_train[j] or n_test[j]]
            record = {
                "rule": ds.FILL_RULE,
                "fill": {names[j]: float(fills[j]) for j in cols},
                "missing": {
                    "train": {names[j]: int(n_train[j]) for j in cols},
                    "test": {names[j]: int(n_test[j]) for j in cols},
                },
            }
            self._fills = fills, record
        return self._fills

    def sessions(self, dataset: str = "expedia") -> wl.Workload:
        """Every test session; expedia-filled's are Expedia's with the missing
        values filled, in the same order, with the same walker config."""
        if dataset not in self._sessions:
            if dataset == "expedia":
                w = wl.ranking_sessions(self.ranking())
            elif dataset == "expedia-filled":
                base = self.sessions("expedia")
                fills, record = self.expedia_fills()
                meta = {**base.meta, "missing_values": record}
                X = ds.fill_missing(base.X, fills)
                w = wl.Workload(X, base.offsets, dict(base.config), base.entities, meta)
            else:
                raise ValueError(f"{dataset} has no sessions")
            self._sessions[dataset] = w
        return self._sessions[dataset]

    def tl_model(self, bin_path: Path) -> Any:
        import treelite

        if bin_path not in self._tl:
            self._tl = {bin_path: treelite.Model.deserialize(str(bin_path))}
        return self._tl[bin_path]

    def train_data(self, model: Model) -> TrainData:
        key = (model.dataset, model.horizon, model.replicate, model.replicate_fraction)
        source: dict[str, Any]
        if key in self._train:
            return self._train[key]
        if model.replicate:
            td = self.replicate_data(model)
            self._train[key] = td
            return td
        identity: dict[str, Any] = {}
        if model.dataset in SURVIVAL:
            assert model.horizon is not None
            split = self.survival(model.dataset)
            edges = split.edges(model.horizon)
            X, y, cov = ds.expand_train(split, edges)
            names, cat = ds.feature_names(cov), cov.cat_indices
            split_rec = {"seed": split.seed, "test_frac": split.test_frac, "unit": "patient"}
            source = ds.source_record(model.dataset)
            horizon = {"requested": model.horizon, "realized": len(edges) - 1}
            extra = {"edges": edges, "train_cov": cov}
        elif model.dataset in RANKING:
            r = self.ranking()
            X = r.train_df.select(ds.EXPEDIA_FEATURES).to_numpy().astype(np.float64)
            if model.dataset == "expedia-filled":
                fills, identity["missing_values"] = self.expedia_fills()
                X = ds.fill_missing(X, fills)
            y = r.train_df[ds.EXPEDIA_LABEL].to_numpy().astype(np.float64)
            names, cat = list(ds.EXPEDIA_FEATURES), []
            split_rec = {
                "seed": r.seed,
                "test_frac": r.test_frac,
                "max_sessions": r.max_sessions,
                "unit": "session",
                "search_level_features": list(ds.EXPEDIA_SESSION),
            }
            source = {
                "file": "experiments/data/expedia.parquet",
                "fingerprint": r.fingerprint,
                "matches_paper": r.fingerprint_ok,
            }
            horizon, extra = None, {}
        elif model.dataset == "credit":
            c = self.credit()
            X, y = c.train_X, c.train_y
            names, cat = list(ds.CREDIT_FEATURES), []
            split_rec = {"seed": c.seed, "test_frac": c.test_frac, "unit": "row"}
            source = ds.source_record("credit")
            horizon, extra = None, {}
        else:
            raise ValueError(f"unknown dataset {model.dataset}")
        sha = hashlib.sha256(matrix_bytes(np.column_stack([X, y]))).hexdigest()
        td = TrainData(X, y, names, cat, split_rec, source, horizon, sha, extra, identity)
        self._train[key] = td
        return td

    def train_entities(self, model: Model, td: TrainData) -> np.ndarray:
        """Each training row's entity: patient, session or row."""
        if model.dataset in SURVIVAL:
            return ds.train_patients(td.extra["train_cov"], td.extra["edges"])
        if model.dataset in RANKING:
            return self.ranking().train_df["srch_id"].to_numpy().astype(np.int64)
        return np.arange(len(td.X), dtype=np.int64)

    def replicate_data(self, model: Model) -> TrainData:
        """A seed replicate's training data: the released model's rows of a seeded
        sample of the training split's entities, without replacement. The split,
        the survival time bins and the test data are the released model's."""
        released = self.train_data(model.released)
        rows = self.train_entities(model.released, released)
        if len(rows) != len(released.X):
            raise AssertionError(f"{model.id}: {len(rows)} entity labels, {len(released.X)} rows")
        entities = np.unique(rows)
        seed = ds.replicate_seed(int(released.split["seed"]), model.replicate)
        chosen = entities[ds.sample_entities(len(entities), model.replicate_fraction, seed)]
        keep = np.isin(rows, chosen)
        X, y = released.X[keep], released.y[keep]
        sha = hashlib.sha256(matrix_bytes(np.column_stack([X, y]))).hexdigest()
        replicate = {
            "index": model.replicate,
            "fraction": model.replicate_fraction,
            "seed": seed,
            "unit": released.split["unit"],
            "entities": len(entities),
            "sampled": len(chosen),
            "rows": int(keep.sum()),
            "released_train_data_sha256": released.sha256,
        }
        return TrainData(
            X,
            y,
            released.names,
            released.cat,
            released.split,
            released.source,
            released.horizon,
            sha,
            released.extra,
            {**released.identity, "replicate": replicate},
        )

    def whatif_draws(self, model: Model, td: TrainData) -> wl.WhatIfDraws:
        if model.replicate:  # a replicate times the released model's cells
            model = model.released
            td = self.train_data(model)
        horizon = td.horizon["realized"] if td.horizon else None
        key = (model.dataset, horizon)
        if key in self._draws:
            return self._draws[key]
        if model.dataset in SURVIVAL:
            split = self.survival(model.dataset)
            cov = split.covariates("test")
            train = td.extra["train_cov"]
            pool = list(range(len(cov.names)))
            d = wl.whatif_v2_draws(
                model.dataset, cov.values, cov.rows, train.values, pool, td.names, horizon
            )
        elif model.dataset == "expedia":
            r = self.ranking()
            test = r.test_df.select(ds.EXPEDIA_FEATURES).to_numpy().astype(np.float64)
            pool = [
                i for i, n in enumerate(ds.EXPEDIA_FEATURES) if n not in wl.EXPEDIA_NOT_PERTURBED
            ]
            d = wl.whatif_v2_draws(
                "expedia", test, np.arange(len(test)), td.X, pool, td.names, None
            )
        elif model.dataset == "credit":
            c = self.credit()
            d = wl.whatif_v2_draws(
                "credit",
                c.test_X,
                np.arange(len(c.test_X)),
                c.train_X,
                wl.CREDIT_POOL,
                td.names,
                None,
            )
        else:
            raise ValueError(f"whatif-v2 has no draws for {model.dataset}")
        self._draws[key] = d
        return d


# --- models --------------------------------------------------------------------------


# The version of the rules behind every derived record: a model's limits and
# JSON export, a cell's status, contract checks and preflight warnings. Cell
# keys include it and model.json records it, so a change to the rules
# re-evaluates every model and cell on resume; models are not retrained, and
# a cell's data and references are reused when only the rules changed.
# Policy 3 adds the stage oracle (oracle.bin) to every cell.
PREP_POLICY = 3


def model_identity(model: Model, fw: str, td: TrainData) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "dataset": model.dataset,
        "framework": fw,
        "source": td.source,
        "split": td.split,
        "n_trees": model.n_trees,
        "max_depth": model.max_depth,
        "horizon": td.horizon,
        "train": tr.params(fw, model.n_trees, model.max_depth),
        "categorical_features": [td.names[i] for i in td.cat],
        "feature_order": td.names,
        "libraries": tr.library_versions(fw),
        "train_data_sha256": td.sha256,
        **{k: td.identity[k] for k in IDENTITY_EXTRAS if k in td.identity},
    }


# Compiled baselines built from an older model would otherwise be picked up
# by the runner next to the new one.
STALE_BASELINES = ("tl2cgen.so", "tl2cgen.json", "lleaves.so", "lleaves.json")


def ensure_model(ctx: Context, model: Model, fw: str, td: TrainData, force: bool) -> dict[str, Any]:
    fw_dir = model.dir(ctx.paths.artifacts) / fw
    json = model.id in ctx.json_models
    identity = model_identity(model, fw, td)
    key = fm.sha256_json(identity)
    doc_path = fw_dir / "model.json"
    if not force and doc_path.exists():
        doc = fm.read_json(doc_path)
        if doc.get("key") == key and files_match(fw_dir, model_files(doc)):
            log(f"  {model.id}/{fw}: model matches its manifest, reused")
            return refresh_model(fw_dir, doc, json)
        log(f"  {model.id}/{fw}: manifest differs, retraining")
    fw_dir.mkdir(parents=True, exist_ok=True)
    for name in STALE_BASELINES:
        (fw_dir / name).unlink(missing_ok=True)

    native = fw_dir / tr.NATIVE_NAME[fw]
    log(f"  {model.id}/{fw}: training on {td.X.shape[0]:,} rows")
    t0 = time.perf_counter()
    tr.train(fw, td.X, td.y, td.names, td.cat, model.n_trees, model.max_depth, native)
    t1 = time.perf_counter()
    tl, json_bytes = tr.export_treelite(
        fw, native, fw_dir / "model_treelite.json", fw_dir / "model_treelite.bin", json
    )
    t2 = time.perf_counter()
    structure = tr.structure(tl)
    del tl
    files = {"native": native.name, "treelite_bin": "model_treelite.bin"}
    file_recs = {k: {"path": v, "sha256": fm.sha256_file(fw_dir / v)} for k, v in files.items()}
    doc = {
        **identity,
        "key": key,
        "prep_policy": PREP_POLICY,
        "files": file_recs,
        "structure": structure,
        "limits": tr.model_limits(structure),
        "treelite_json": json_record(fw_dir, json_bytes),
        "prep": {
            "train_seconds": round(t1 - t0, 3),
            "export_seconds": round(t2 - t1, 3),
            "peak_rss_bytes": peak_rss_bytes(),
            "bytes": sum((fw_dir / v).stat().st_size for v in files.values()),
        },
    }
    fm.write_json(doc_path, doc)
    log(
        f"    {structure['trees']} trees, {structure['total_nodes']:,} nodes "
        f"(largest tree {structure['max_tree_nodes']:,}), trained in {t1 - t0:.1f}s"
    )
    for v in doc["limits"]:
        log(f"    LIMIT: {v}")
    return doc


def model_files(doc: dict[str, Any]) -> dict[str, dict[str, str]]:
    """The files that make the model. The JSON dump is derived from the
    binary export and tracked on its own."""
    return {k: v for k, v in doc["files"].items() if k != "treelite_json"}


def json_record(fw_dir: Path, size: int | None) -> dict[str, Any]:
    if size is None:
        return {"status": "skipped: not a test model"}
    path = fw_dir / "model_treelite.json"
    if size > tr.MAX_JSON_BYTES:
        log(f"    JSON export skipped: {size} bytes, over the loader's limit")
        return {"bytes": size, "status": f"skipped: over the loader's limit of {tr.MAX_JSON_BYTES}"}
    return {"bytes": size, "status": "written", "path": path.name, "sha256": fm.sha256_file(path)}


# Fields of model.json written before PREP_POLICY, replaced by treelite_json.
LEGACY_MODEL_FIELDS = ("treelite_json_bytes", "treelite_json_loadable")


def refresh_model(fw_dir: Path, doc: dict[str, Any], json: bool) -> dict[str, Any]:
    """Bring a reused model's derived records up to the current policy without
    retraining: the import limits, and the JSON dump, which is exported from
    the binary when the model is listed and deleted when it is not."""
    new = {
        **{k: v for k, v in doc.items() if k not in LEGACY_MODEL_FIELDS},
        "prep_policy": PREP_POLICY,
        "files": model_files(doc),
        "limits": tr.model_limits(doc["structure"]),
    }
    path = fw_dir / "model_treelite.json"
    rec = doc.get("treelite_json", {})
    current = rec.get("status") == "written" and path.exists()
    current = current and fm.sha256_file(path) == rec.get("sha256")
    if json and rec.get("status", "").startswith("skipped: over"):
        path.unlink(missing_ok=True)  # over the loader's limit: no dump, stale or not
    elif json and not current:
        import treelite

        log(f"    exporting the JSON dump from {fw_dir.name}/model_treelite.bin")
        tl = treelite.Model.deserialize(str(fw_dir / "model_treelite.bin"))
        dumped = tl.dump_as_json().encode()
        if len(dumped) <= tr.MAX_JSON_BYTES:
            fm.write_bytes(path, dumped)
        else:
            path.unlink(missing_ok=True)  # a stale dump must not outlive its record
        new["treelite_json"] = json_record(fw_dir, len(dumped))
    elif not json:
        path.unlink(missing_ok=True)
        new["treelite_json"] = json_record(fw_dir, None)
    if new != doc:
        fm.write_json(fw_dir / "model.json", new)
    return new


# --- cells -----------------------------------------------------------------------------


@dataclass(slots=True)
class Built:
    """A workload on disk: where each file is, and its hash."""

    workload: wl.Workload
    files: dict[str, Path]  # role -> path
    hashes: dict[str, str]  # role -> sha256
    legacy_reference: Path | None = None  # whatif-v1's reference.bin


def write_workload(w: wl.Workload, data_dir: Path, offsets: bool = True) -> Built:
    files = {
        "test_data": data_dir / "test_data.bin",
        "walker_config": data_dir / "walker_config.json",
        "entities": data_dir / "entities.npy",
    }
    hashes = {
        "test_data": put_matrix(files["test_data"], w.X),
        "walker_config": put_walker_config(files["walker_config"], w.config),
        "entities": put_npy(files["entities"], w.entities),
    }
    if offsets and w.offsets is not None:
        files["group_offsets"] = data_dir / "group_offsets.bin"
        hashes["group_offsets"] = put_offsets(files["group_offsets"], w.offsets)
    return Built(w, files, hashes)


def build_workload(ctx: Context, cell: Cell, td: TrainData, mdoc: dict[str, Any]) -> Built:
    art = ctx.paths.artifacts
    model_dir = cell.model.dir(art)
    p = cell.param
    match cell.generator:
        case "panel-v2":
            assert td.horizon is not None
            cov = ctx.survival(cell.model.dataset).covariates("test")
            w = wl.panel(cov, td.horizon["realized"])
            w.meta["horizon"] = td.horizon
            return write_workload(w, model_dir)
        case "ranking-sessions-v2":
            return write_workload(ctx.sessions(cell.model.dataset), model_dir)
        case "ranking-cohort-v2":
            sessions = ctx.sessions(cell.model.dataset)
            subsets = wl.ranking_cohort(sessions, p["min_candidates"], [p["size"]])
            return write_workload(
                subsets[p["size"]], art / cell.model.dataset / "workloads" / cell.workload_id
            )
        case "whatif-v2":
            draws = ctx.whatif_draws(cell.model, td)
            # A replicate times its released cell's data: the features are the
            # released model's most-split ones, so that model must exist.
            released = cell.model.released
            if cell.model.replicate:
                ensure_model(ctx, released, cell.framework, ctx.train_data(released), False)
            native = released.dir(art) / cell.framework / tr.NATIVE_NAME[cell.framework]
            counts = tr.split_counts(cell.framework, native, len(td.names))
            features = wl.whatif_v2_features(draws, counts, p["k"])
            w = wl.whatif_v2(draws, features, p["k"], p["G"])
            if cell.model.replicate:
                w.meta["perturbation"]["ranked_by"] = f"{released.id}/{cell.framework}"
            if td.horizon:
                w.meta["horizon"] = td.horizon
            w.meta["perturbation"]["split_counts"] = {
                td.names[i]: int(counts[i]) for i in draws.pool
            }
            return write_workload(w, cell.dir(art))
        case "whatif-v1":
            return build_whatif_v1(ctx, cell, mdoc)
    raise ValueError(cell.generator)


def build_whatif_v1(ctx: Context, cell: Cell, mdoc: dict[str, Any]) -> Built:
    """The released credit scenario cell, written exactly as before."""
    import lightgbm as lgb

    k, G = cell.param["k"], cell.param["G"]
    fw_dir = cell.model.dir(ctx.paths.artifacts) / "lightgbm"
    bst = lgb.Booster(model_file=str(fw_dir / tr.NATIVE_NAME["lightgbm"]))
    pool = wl.whatif_v1_ranked_pool(bst.feature_importance(importance_type="gain"))
    perturb = sorted(pool[:k])
    base_X = ctx.credit().test_X[: wl.N_WHATIF_BASE]
    X, offsets = wl.whatif_v1(base_X, perturb, G)
    any_zero = wl.any_zero_group_fraction(base_X, perturb)
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
            "any_zero_group_fraction": any_zero,
            "perturbation": "multiplicative log-uniform [0.5, 2.0], seed 42",
        },
    }
    meta = {
        "generator": "whatif-v1",
        "grouping": {"kind": "whatif", "G": G},
        "k": {"requested": k, "selected": len(perturb)},
        "perturbation": {
            "features": perturb,
            "feature_names": [names[i] for i in perturb],
            "pool": [names[i] for i in wl.CREDIT_POOL],
            "ranking": "LightGBM gain importance",
            "shock": "multiplicative 2**Uniform(-1, 1)",
            "any_zero_group_fraction": any_zero,
        },
        "seeds": {"generator": wl.WHATIF_V1_SEED},
    }
    w = wl.Workload(X, offsets, config, np.arange(len(base_X), dtype=np.int64), meta)
    built = write_workload(w, cell.dir(ctx.paths.artifacts))
    built.legacy_reference = cell.dir(ctx.paths.artifacts) / "reference.bin"
    return built


def cell_identity(cell: Cell, mdoc: dict[str, Any], built: Built) -> dict[str, Any]:
    """What a cell's content depends on; the key adds PREP_POLICY."""
    return {
        "schema_version": SCHEMA,
        "id": cell.id,
        "workload": cell.workload,
        "generator": cell.generator,
        "dataset": cell.model.dataset,
        "framework": cell.framework,
        "n_trees": cell.model.n_trees,
        "max_depth": cell.model.max_depth,
        "params": cell.param,
        "model_key": mdoc["key"],
        "data_sha256": dict(sorted(built.hashes.items())),
    }


def ensure_cell(
    ctx: Context, cell: Cell, td: TrainData, mdoc: dict[str, Any], force: bool
) -> dict[str, Any]:
    art = ctx.paths.artifacts
    cell_dir = cell.dir(art)
    fw_dir = cell.model.dir(art) / cell.framework
    built = build_workload(ctx, cell, td, mdoc)
    identity = cell_identity(cell, mdoc, built)
    content_key = fm.sha256_json(identity)
    key = fm.sha256_json({**identity, "prep_policy": PREP_POLICY})
    doc_path = cell_dir / "cell.json"
    old: dict[str, Any] = {}
    if not force and doc_path.exists():
        old = fm.read_cell(doc_path)
        # A model's JSON dump is derived and may come and go with the export
        # policy; it does not decide whether a cell is current.
        kept = {r: f for r, f in old["files"].items() if r != "model_treelite_json"}
        if not files_match(cell_dir, kept):
            old = {}
        elif old.get("key") == key:
            return old
        # Cells from before PREP_POLICY have no content_key; their key was
        # the content key.
        elif old.get("content_key", old.get("key")) != content_key:
            old = {}

    w = built.workload
    contract = wl.contracts(w)
    varying = set(w.config["varying_features"]) | set(w.config["mono_inc_features"])
    varying |= set(w.config["mono_dec_features"])
    tl = ctx.tl_model(fw_dir / "model_treelite.bin")
    predicates = tr.varying_predicates(tl, varying)
    limits = tr.model_limits(mdoc["structure"])
    warnings = tr.predicate_warning(predicates)

    files = dict(built.files)
    hashes = dict(built.hashes)
    reuse = {r: old["files"][r] for r in ("predictions", "reference") if r in old.get("files", {})}
    if (
        contract["ok"]
        and "predictions" in reuse
        and (built.legacy_reference is None or "reference" in reuse)
    ):
        # Only the rules changed: the references on disk are still current.
        for role, rec in reuse.items():
            files[role] = cell_dir / rec["path"]
            hashes[role] = rec["sha256"]
        status = "unsupported" if limits else "ready"
    elif contract["ok"]:
        X = np.asarray(w.X)
        if built.legacy_reference is not None:
            ref = tr.gtil(tl, X)
            files["reference"] = built.legacy_reference
            hashes["reference"] = put_matrix(built.legacy_reference, ref.reshape(-1, 1))
        native = fw_dir / tr.NATIVE_NAME[cell.framework]
        if cell.generator in ("panel-v2", "ranking-sessions-v2"):
            pred_path = fw_dir / "predictions.npy"  # where the released runner reads it
        else:
            pred_path = cell_dir / "predictions.npy"
        if built.legacy_reference is None or cell.framework != "lightgbm":
            preds = tr.Reference(cell.framework, native, td.names, td.cat).predict(X)
        else:
            preds = ref
        files["predictions"] = pred_path
        hashes["predictions"] = put_npy(pred_path, preds)
        status = "unsupported" if limits else "ready"
    else:
        status = "failed"
        for v in contract["violations"]:
            log(f"    CONTRACT: {cell.id}: {v}")
    oracle = None
    if contract["ok"]:
        rows, oracle = tr.oracle(tl, np.asarray(w.X), w.group_offsets)
        files["oracle"] = cell_dir / "oracle.bin"
        hashes["oracle"] = put_matrix(files["oracle"], rows)

    for role, rec in model_files(mdoc).items():
        files[f"model_{role}"] = fw_dir / rec["path"]
        hashes[f"model_{role}"] = rec["sha256"]

    offsets = w.group_offsets
    cfg = w.config
    names = cfg["feature_names"]
    k = w.meta.get("k")
    if k is not None:
        k = {**k, "realized_varying_columns": len(contract["varying_columns"])}
    doc = {
        **identity,
        "prep_policy": PREP_POLICY,
        "content_key": content_key,
        "key": key,
        "status": status,
        "horizon": w.meta.get("horizon", td.horizon),
        "grouping": {**w.meta["grouping"], **wl.sizes(offsets)},
        "k": k,
        "feature_order": names,
        "features": {
            "varying": [names[i] for i in cfg["varying_features"]],
            "mono_inc": [names[i] for i in cfg["mono_inc_features"]],
            "mono_dec": [names[i] for i in cfg["mono_dec_features"]],
            "realized_varying": [names[i] for i in contract["varying_columns"]],
        },
        "perturbation": w.meta.get("perturbation"),
        "seeds": {"split": mdoc["split"]["seed"], **w.meta.get("seeds", {})},
        "split": mdoc["split"],
        "train": mdoc["train"],
        **{k: mdoc[k] for k in IDENTITY_EXTRAS if k in mdoc},
        "libraries": mdoc["libraries"],
        "model": {
            "key": mdoc["key"],
            "dir": rel(fw_dir, cell_dir),
            "structure": mdoc["structure"],
            "varying_predicates_upper_bound": predicates,
        },
        "contracts": {k_: v for k_, v in contract.items() if k_ != "varying_columns"},
        "limits": limits,
        "warnings": warnings,
        "oracle": oracle,
        "files": {
            role: {"path": rel(files[role], cell_dir), "sha256": hashes[role]}
            for role in sorted(files)
        },
        "baselines": old.get("baselines", {}),
    }
    fm.write_cell(doc_path, doc)
    return doc


# --- fixture cells ---------------------------------------------------------------------


# Cells built from the import fixtures for runner coverage their shape lacks.
# fallback_sums: identity.bin's model with leaves 1e300 and 1e-300, 2^-997 apart,
# so no fixed point holds both and exact_sums() is false (tests/import.rs builds
# the same model). wide_pieces: sigmoid_f64 over groups across the 1,024-row piece
# boundary.
WIDE_PIECES = [1023, 1024, 1025, 2049]


def _fallback_sums_model() -> Any:
    from treelite.model_builder import (
        Metadata,
        ModelBuilder,
        PostProcessorFunc,
        TreeAnnotation,
    )

    builder = ModelBuilder(
        threshold_type="float64",
        leaf_output_type="float64",
        metadata=Metadata(
            num_feature=2,
            task_type="kRegressor",
            average_tree_output=False,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=TreeAnnotation(num_tree=2, target_id=[0, 0], class_id=[0, 0]),
        postprocessor=PostProcessorFunc(name="identity"),
        base_scores=[3.0],
    )
    for tree, (left, right) in enumerate([(1e300, -0.5), (1e-300, -1.0)]):
        builder.start_tree()
        builder.start_node(0)
        builder.numerical_test(
            feature_id=tree,
            threshold=1.0,
            default_left=tree == 0,
            opname="<=",
            left_child_key=1,
            right_child_key=2,
        )
        builder.end_node()
        for node, val in [(1, left), (2, right)]:
            builder.start_node(node)
            builder.leaf(val)
            builder.end_node()
        builder.end_tree()
    return builder.commit()


def fixture_inputs(src: Path, name: str) -> tuple[bytes, np.ndarray, dict[str, Any], Any]:
    """A fixture cell's model bytes, data, walker config and group offsets (None
    for groups of the configured width)."""
    X = np.array(fm.read_matrix(src / "data.bin"))
    config = fm.read_walker_config(src / "walker_config.json")
    if name == "fallback_sums":
        return _fallback_sums_model().serialize_bytes(), X, config, None
    if name == "wide_pieces":
        rows = []
        for g, n in enumerate(WIDE_PIECES):
            r = X[np.arange(n) % X.shape[0]].copy()
            r[:, 1] = float(g)  # feature 1 is constant within a group
            rows.append(r)
        offsets = np.concatenate([[0], np.cumsum(WIDE_PIECES)]).astype(np.uint64)
        config = {**config, "max_group_width": max(WIDE_PIECES)}
        return (src / "sigmoid_f64.bin").read_bytes(), np.vstack(rows), config, offsets
    return (src / f"{name}.bin").read_bytes(), X, config, None


def ensure_fixture(ctx: Context, cell: Cell) -> dict[str, Any]:
    """An import fixture as a cell: its model, the fixtures' data (seven groups of
    128 rows, feature 1 constant within each) and Treelite GTIL references."""
    import treelite

    name = cell.model.fixture
    assert name is not None
    src = ctx.paths.repo / "tests" / "fixtures" / "import"
    art = ctx.paths.artifacts
    fw_dir, cell_dir = cell.model.dir(art) / cell.framework, cell.dir(art)
    model_bin = fw_dir / "model_treelite.bin"
    model_bytes, X, config, offsets = fixture_inputs(src, name)
    files = {"model_treelite_bin": model_bin}
    hashes = {"model_treelite_bin": _put(model_bin, model_bytes)}
    if (src / f"{name}.json").exists():
        _put(fw_dir / "model_treelite.json", (src / f"{name}.json").read_bytes())
    config["feature_names"] = ["x0", "x1"]
    n_groups = len(offsets) - 1 if offsets is not None else X.shape[0] // config["max_group_width"]
    meta: dict[str, Any] = {"generator": "fixture-v1", "grouping": {"kind": "fixture"}}
    w = wl.Workload(X, offsets, config, np.arange(n_groups, dtype=np.int64), meta)
    built = write_workload(w, cell_dir)
    files.update(built.files)
    hashes.update(built.hashes)
    tl = treelite.Model.deserialize(str(model_bin))
    ref_path = src / f"{name}_reference.bin"
    if ref_path.exists() and offsets is None and name != "fallback_sums":
        ref = np.array(fm.read_matrix(ref_path)).ravel()
    else:
        ref = tr.gtil(tl, X)
    files["predictions"] = cell_dir / "predictions.npy"
    hashes["predictions"] = put_npy(files["predictions"], ref)
    rows, oracle = tr.oracle(tl, X, w.group_offsets)
    files["oracle"] = cell_dir / "oracle.bin"
    hashes["oracle"] = put_matrix(files["oracle"], rows)
    identity = {
        "schema_version": SCHEMA,
        "id": cell.id,
        "workload": cell.workload,
        "generator": cell.generator,
        "dataset": "fixtures",
        "framework": cell.framework,
        "fixture": name,
        "data_sha256": dict(sorted(built.hashes.items())),
        "model_sha256": hashes["model_treelite_bin"],
    }
    key = fm.sha256_json({**identity, "prep_policy": PREP_POLICY})
    contract = wl.contracts(w)
    doc = {
        **identity,
        "prep_policy": PREP_POLICY,
        "key": key,
        "status": "ready" if contract["ok"] else "failed",
        "grouping": {**meta["grouping"], **wl.sizes(w.group_offsets)},
        "feature_order": config["feature_names"],
        "model": {
            "key": hashes["model_treelite_bin"],
            "dir": rel(fw_dir, cell_dir),
            "source": f"tests/fixtures/import ({name})",
        },
        "contracts": {k: v for k, v in contract.items() if k != "varying_columns"},
        "oracle": oracle,
        "files": {
            role: {"path": rel(files[role], cell_dir), "sha256": hashes[role]}
            for role in sorted(files)
        },
        "baselines": {},
    }
    fm.write_cell(cell_dir / "cell.json", doc)
    return doc


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
            td = ctx.train_data(model)
            if model.layout == "standard":
                train = np.column_stack([td.X, td.y])
                put_matrix(model.dir(ctx.paths.artifacts) / "train_data.bin", train)
            frameworks = sorted({c.framework for c in model_cells})
            mdocs = {fw: ensure_model(ctx, model, fw, td, force) for fw in frameworks}
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
