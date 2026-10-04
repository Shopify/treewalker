"""Workload generators: what each group of rows is, and what varies inside it.

Generators are versioned. A workload records its generator, so a change to
one is a new version, never a silent edit:

- ``panel-v1``: every test patient over the full horizon, one group per patient.
- ``ranking-sessions-v1``: every Expedia test session, one group per session.
- ``whatif-v1``: the released credit scenario generator: gain-ranked
  features with multiplicative shocks.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import datasets as ds

N_WHATIF_BASE = 2000


@dataclass(slots=True)
class Workload:
    X: np.ndarray
    offsets: np.ndarray | None  # None: fixed-width groups of config["max_group_width"]
    config: dict[str, Any]  # walker_config.json
    entities: np.ndarray  # one ID per group: patient, session or base row
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def group_offsets(self) -> np.ndarray:
        if self.offsets is not None:
            return self.offsets
        width = self.config["max_group_width"]
        return np.arange(0, self.X.shape[0] + 1, width, dtype=np.uint64)


def walker_config(
    names: list[str], width: int, varying: list[int], inc: list[int], dec: list[int]
) -> dict[str, Any]:
    return {
        "n_features": len(names),
        "max_group_width": width,
        "feature_names": names,
        "varying_features": varying,
        "mono_inc_features": inc,
        "mono_dec_features": dec,
    }


def sizes(offsets: np.ndarray) -> dict[str, Any]:
    s = np.diff(offsets.astype(np.int64))
    return {
        "n_groups": len(s),
        "rows": int(s.sum()),
        "size_min": int(s.min()),
        "size_mean": float(s.mean()),
        "size_max": int(s.max()),
    }


# --- panel ---------------------------------------------------------------------


def panel(cov: ds.Covariates, horizon: int) -> Workload:
    """Each test patient over all ``horizon`` steps."""
    n = len(cov.values)
    patient = np.repeat(np.arange(n), horizon)
    steps = np.tile(np.arange(horizon), n)
    X = np.hstack([cov.values[patient], ds.tv_features(steps, cov.values[patient, 0], horizon)])
    nc = len(cov.names)
    config = walker_config(
        ds.feature_names(cov),
        horizon,
        varying=list(range(nc, nc + ds.N_TV)),
        inc=[nc, nc + 1],  # time_step, time_step_sq
        dec=[nc + 2, nc + 3],  # remaining_steps, remaining_frac_sq
    )
    meta = {"generator": "panel-v1", "grouping": {"kind": "panel", "width": horizon}}
    return Workload(X, None, config, cov.rows.astype(np.int64), meta)


# --- ranking -------------------------------------------------------------------


def ranking_sessions(split: ds.RankingSplit) -> Workload:
    test = split.test_df
    X = test.select(ds.EXPEDIA_FEATURES).to_numpy().astype(np.float64)
    sids = test["srch_id"].to_numpy()
    offsets = ds.session_offsets(sids)
    nc = len(ds.EXPEDIA_CONSTANT)
    config = walker_config(
        ds.EXPEDIA_FEATURES,
        int(np.diff(offsets).max()),
        varying=list(range(nc, len(ds.EXPEDIA_FEATURES))),
        inc=[],
        dec=[],
    )
    meta = {"generator": "ranking-sessions-v1", "grouping": {"kind": "sessions"}}
    return Workload(X, offsets, config, sids[offsets[:-1].astype(np.int64)].astype(np.int64), meta)


# --- what-if, v1 -----------------------------------------------------------------

# Perturbable credit pool (frozen): x1 (credit limit), x12-x17 (bill amounts)
# and x18-x23 (payment amounts), 0-indexed after dropping the id column.
CREDIT_POOL = [0, *range(11, 17), *range(17, 23)]
WHATIF_V1_SEED = 42


def whatif_v1_ranked_pool(gains: np.ndarray) -> list[int]:
    """The pool by descending LightGBM gain importance, ties by index."""
    return [i for _, i in sorted(((gains[i], i) for i in CREDIT_POOL), key=lambda t: (-t[0], t[1]))]


def whatif_v1(base_X: np.ndarray, perturb: list[int], G: int) -> tuple[np.ndarray, np.ndarray]:
    """Variant 0 is the base row; variants 1..G-1 multiply the perturbed
    features by 2**Uniform(-1, 1), from a fresh generator per cell. A base
    value of 0 stays 0, so that feature is constant for the group."""
    n_base, k = base_X.shape[0], len(perturb)
    rng = np.random.default_rng(WHATIF_V1_SEED)
    mult = np.power(2.0, rng.uniform(-1.0, 1.0, size=(n_base, G - 1, k)))
    out = np.repeat(base_X, G, axis=0)
    rows = out.reshape(n_base, G, -1)
    cols = np.asarray(perturb, dtype=np.int64)
    rows[:, 1:, cols] = base_X[:, None, cols] * mult
    offsets = np.concatenate([[0], np.cumsum([G] * n_base)]).astype(np.uint64)
    return out, offsets


def any_zero_group_fraction(base_X: np.ndarray, perturb: list[int]) -> float:
    """Share of base rows with at least one perturbed feature equal to 0."""
    if not perturb:
        return 0.0
    return float(np.mean(np.any(base_X[:, perturb] == 0.0, axis=1)))
