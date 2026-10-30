"""What every model and cell is built from: datasets, splits, training data and draws,
loaded once per run and shared."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import datasets as ds
from .. import formats as fm
from .. import workloads as wl
from ..grids import SURVIVAL, Model
from ..paths import Paths
from .models import model_stats

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
    # Fields only some models' identities have (models.IDENTITY_EXTRAS), added only when
    # present, so every other model keeps its key.
    identity: dict[str, Any] = field(default_factory=dict)


# Datasets built from the Expedia split, one session per group.
RANKING = ("expedia", "expedia-filled", "expedia-cat")


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
        self._categorical: ds.RankingSplit | None = None
        self._fills: tuple[np.ndarray, dict[str, Any]] | None = None
        self._sessions: dict[str, wl.Workload] = {}
        self._credit: ds.CreditSplit | None = None
        self._draws: dict[tuple, wl.WhatIfDraws] = {}
        self._tl: dict[Path, Any] = {}
        self._stats: dict[tuple, dict[str, Any]] = {}

    def survival(self, name: str) -> ds.SurvivalSplit:
        if name not in self._survival:
            self._survival[name] = ds.split_survival(self.paths, name, self.seed, self.test_frac)
        return self._survival[name]

    def ranking(self, dataset: str = "expedia") -> ds.RankingSplit:
        if dataset == "expedia-cat":
            if self._categorical is None:
                self._categorical = ds.split_expedia(
                    self.paths, self.seed, 0.2, 0, source_order=True
                )
            return self._categorical
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
            elif dataset == "expedia-cat":
                w = wl.ranking_sessions(self.ranking(dataset), categorical=True)
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

    def model_stats(self, bin_path: Path, varying: set[int]) -> dict[str, Any]:
        """``model_stats`` of the model at ``bin_path`` under a varying set."""
        st = bin_path.stat()
        key = (bin_path, st.st_size, st.st_mtime_ns, tuple(sorted(varying)))
        if key not in self._stats:
            self._stats[key] = model_stats(self.tl_model(bin_path), varying)
        return self._stats[key]

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
            r = self.ranking(model.dataset)
            names = list(
                ds.EXPEDIA_CAT_FEATURES if model.dataset == "expedia-cat" else ds.EXPEDIA_FEATURES
            )
            X = r.train_df.select(names).to_numpy().astype(np.float64)
            if model.dataset == "expedia-filled":
                fills, identity["missing_values"] = self.expedia_fills()
                X = ds.fill_missing(X, fills)
            y = r.train_df[ds.EXPEDIA_LABEL].to_numpy().astype(np.float64)
            cat = list(range(len(ds.EXPEDIA_CAT))) if model.dataset == "expedia-cat" else []
            split_rec = {
                "seed": r.seed,
                "test_frac": r.test_frac,
                "max_sessions": r.max_sessions,
                "unit": "session",
                "search_level_features": list(ds.EXPEDIA_SESSION),
                **({"candidate_order": "source-file"} if model.dataset == "expedia-cat" else {}),
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
        sha = hashlib.sha256(fm.matrix_bytes(np.column_stack([X, y]))).hexdigest()
        td = TrainData(X, y, names, cat, split_rec, source, horizon, sha, extra, identity)
        self._train[key] = td
        return td

    def train_entities(self, model: Model, td: TrainData) -> np.ndarray:
        """Each training row's entity: patient, session or row."""
        if model.dataset in SURVIVAL:
            return ds.train_patients(td.extra["train_cov"], td.extra["edges"])
        if model.dataset in RANKING:
            return self.ranking(model.dataset).train_df["srch_id"].to_numpy().astype(np.int64)
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
        sha = hashlib.sha256(fm.matrix_bytes(np.column_stack([X, y]))).hexdigest()
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
