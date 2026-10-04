"""Workload generators: what each group of rows is, and what varies inside it.

Generators are versioned. A workload records its generator, so a change to
one is a new version, never a silent edit:

- ``panel-v2``: every test patient over the full horizon, one group per patient.
  v1 declared ``interaction_x0_t`` non-monotonic; it increases along a panel
  whenever x0 >= 0.
- ``ranking-sessions-v2``: every Expedia test session, one group per session,
  with its candidates in one seeded random order. v1 kept the file's order,
  sorted by ``prop_id``, a feature the models split on: rows that a ``prop_id``
  split sends the same way were contiguous, so the rows reaching a leaf formed
  fewer, longer runs, and exact sums update once per run. Requests do not
  arrive sorted by property.
- ``ranking-cohort-v2``: the controlled size curve. Sessions with at least 32
  candidates; the size-n subset is the first n candidates of the session's
  seeded order, so the subsets nest, 4 ⊂ 8 ⊂ 16 ⊂ 32, and keep one order
  across sizes, because row order decides how exact sums group rows into runs.
- ``whatif-v1``: the released credit scenario generator, kept for the
  byte-identical check: gain-ranked features with multiplicative shocks.
- ``whatif-v2``: a targeted stress test, not a representative distribution of
  production requests. See ``WHATIF_V2``.
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
    inc = [nc, nc + 1]  # time_step, time_step_sq
    # interaction_x0_t = x0 * time fraction. x0 is fixed within a patient and
    # the fraction increases, so it increases whenever x0 >= 0; x0 is age in
    # SUPPORT and FLCHAIN.
    x0 = cov.values[:, 0]
    if np.all(x0[~np.isnan(x0)] >= 0):
        inc.append(nc + ds.INTERACTION)
    config = walker_config(
        ds.feature_names(cov),
        horizon,
        varying=list(range(nc, nc + ds.N_TV)),
        inc=inc,
        dec=[nc + 2, nc + 3],  # remaining_steps, remaining_frac_sq
    )
    meta = {"generator": "panel-v2", "grouping": {"kind": "panel", "width": horizon}}
    return Workload(X, None, config, cov.rows.astype(np.int64), meta)


# --- ranking -------------------------------------------------------------------


# Seed stream for the candidates' order, apart from the split's own draws.
SESSION_ORDER_STREAM = 1


def session_order(offsets: np.ndarray, seed: int) -> np.ndarray:
    """Row indices that put each session's candidates in a seeded random order;
    sessions stay where they are."""
    off = offsets.astype(np.int64)
    rng = np.random.default_rng([seed, SESSION_ORDER_STREAM])
    return np.concatenate(
        [off[g] + rng.permutation(int(off[g + 1] - off[g])) for g in range(len(off) - 1)]
    )


def ranking_sessions(split: ds.RankingSplit) -> Workload:
    test = split.test_df
    sids = test["srch_id"].to_numpy()
    offsets = ds.session_offsets(sids)
    order = session_order(offsets, split.seed)
    X = test.select(ds.EXPEDIA_FEATURES).to_numpy().astype(np.float64)[order]
    config = walker_config(
        ds.EXPEDIA_FEATURES,
        int(np.diff(offsets).max()),
        varying=[ds.EXPEDIA_FEATURES.index(f) for f in ds.EXPEDIA_VARYING],
        inc=[],
        dec=[],
    )
    meta = {
        "generator": "ranking-sessions-v2",
        "grouping": {"kind": "sessions"},
        "seeds": {"order": [split.seed, SESSION_ORDER_STREAM]},
    }
    return Workload(X, offsets, config, sids[offsets[:-1].astype(np.int64)].astype(np.int64), meta)


def ranking_cohort(
    sessions: Workload, min_candidates: int, sizes_: list[int]
) -> dict[int, Workload]:
    """Nested subsets of every session with ``min_candidates`` or more: the
    size-n subset is each session's first n candidates, in the sessions' own
    (seeded) order."""
    if max(sizes_) > min_candidates:
        raise ValueError("subset sizes cannot exceed the cohort's minimum session size")
    off = sessions.group_offsets.astype(np.int64)
    widths = np.diff(off)
    cohort = np.nonzero(widths >= min_candidates)[0]
    out = {}
    for n in sizes_:
        rows = np.concatenate([off[g] + np.arange(n) for g in cohort])
        offsets = np.arange(len(cohort) + 1, dtype=np.uint64) * n
        config = dict(sessions.config, max_group_width=n)
        meta = {
            "generator": "ranking-cohort-v2",
            "grouping": {"kind": "cohort-subset", "size": n, "min_candidates": min_candidates},
            "seeds": sessions.meta.get("seeds", {}),
        }
        out[n] = Workload(
            np.ascontiguousarray(sessions.X[rows]),
            offsets,
            config,
            sessions.entities[cohort],
            meta,
        )
    return out


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


# --- what-if, v2 -----------------------------------------------------------------

WHATIF_V2_SEED = 42
WHATIF_V2_G_MAX = 128

# Expedia identifiers, and the logging flag that marks a randomized result
# order, are not covariates a request would change.
EXPEDIA_NOT_PERTURBED = {
    "site_id",
    "visitor_location_country_id",
    "prop_country_id",
    "srch_destination_id",
    "prop_id",
    "random_bool",
}

WHATIF_V2: dict[str, dict[str, str]] = {
    "credit": {
        "entity": "a test row",
        "pool": "the frozen monetary pool: x1 (credit limit), x12-x17 (bill amounts), "
        "x18-x23 (payment amounts)",
        "donors": "training rows",
        "categorical": "none",
        "missing": "none in the data",
        "time_step": "not a survival dataset",
    },
    "support": {
        "entity": "a test patient",
        "pool": "every raw covariate; the six time features are derived, never drawn",
        "donors": "training patients, one row each, not expanded rows: expanded rows "
        "are length-biased toward patients who live longer",
        "categorical": "codes are copied from the donor, so every value was seen in training",
        "missing": "none: patients with a missing covariate were dropped before the split",
        "time_step": "one seeded step per entity, fixed within its group; "
        "interaction_x0_t is recomputed per row and varies when x0 is perturbed",
    },
    "expedia": {
        "entity": "a test candidate row (one session, one property)",
        "pool": "every feature except the identifiers site_id, visitor_location_country_id, "
        "prop_country_id, srch_destination_id and prop_id, and the logging flag random_bool",
        "donors": "training candidate rows",
        "categorical": "none declared; integer codes are copied from the donor",
        "missing": "copied from the donor: a missing value is a value of the training "
        "distribution, routed by each split's default direction",
        "time_step": "not a survival dataset",
    },
}
WHATIF_V2["flchain"] = WHATIF_V2["support"]

WHATIF_V2_RULES = (
    "Per scenario, one donor drawn uniformly from the training entities supplies "
    "every perturbed feature jointly. Scenario 0 is the base entity. Features are "
    "the model's most-split eligible covariates, ties by feature index, nested "
    "across k; features the model never splits on are not selected, so the "
    "selected k can be smaller than requested. Base entities, donors and time "
    "steps are drawn once per dataset and shared by every model, k and G: the "
    "size-G group is the first G scenarios of the size-128 group."
)


@dataclass(slots=True)
class WhatIfDraws:
    """Everything random in whatif-v2, drawn once per dataset."""

    dataset: str
    base: np.ndarray  # (n_base, n_covariates) base entities
    base_ids: np.ndarray
    donors: np.ndarray  # (n_donors, n_covariates)
    donor_index: np.ndarray  # (n_base, G_max - 1)
    steps: np.ndarray | None  # survival only: one step per base entity
    horizon: int | None
    pool: list[int]
    names: list[str]  # feature names, in model order
    seed: int


def whatif_v2_draws(
    dataset: str,
    base: np.ndarray,
    base_ids: np.ndarray,
    donors: np.ndarray,
    pool: list[int],
    names: list[str],
    horizon: int | None,
) -> WhatIfDraws:
    seed = ds.dataset_seed(dataset, WHATIF_V2_SEED)
    rng = np.random.default_rng(seed)
    n_base = min(N_WHATIF_BASE, len(base))
    chosen = rng.choice(len(base), n_base, replace=False)
    steps = rng.integers(0, horizon, n_base) if horizon else None
    donor_index = rng.integers(0, len(donors), size=(n_base, WHATIF_V2_G_MAX - 1))
    return WhatIfDraws(
        dataset,
        base[chosen],
        base_ids[chosen].astype(np.int64),
        donors,
        donor_index,
        steps,
        horizon,
        pool,
        names,
        seed,
    )


def whatif_v2_features(draws: WhatIfDraws, split_counts: np.ndarray, k: int) -> list[int]:
    ranked = sorted(
        (i for i in draws.pool if split_counts[i] > 0), key=lambda i: (-split_counts[i], i)
    )
    return ranked[:k]


def whatif_v2(draws: WhatIfDraws, features: list[int], k: int, G: int) -> Workload:
    if G > WHATIF_V2_G_MAX:
        raise ValueError(f"G={G} exceeds the drawn maximum {WHATIF_V2_G_MAX}")
    n_base, n_cov = draws.base.shape
    groups = np.repeat(draws.base[:, None, :], G, axis=1)
    cols = np.asarray(sorted(features), dtype=np.int64)
    if len(cols):
        donors = draws.donors[draws.donor_index[:, : G - 1]]  # (n_base, G-1, n_cov)
        groups[:, 1:, cols] = donors[:, :, cols]
    X = groups.reshape(n_base * G, n_cov)
    varying = [int(c) for c in cols]
    if draws.horizon is not None:
        assert draws.steps is not None
        steps = np.repeat(draws.steps, G)
        X = np.hstack([X, ds.tv_features(steps, X[:, 0], draws.horizon)])
        if 0 in varying:
            varying.append(n_cov + ds.INTERACTION)
    offsets = np.arange(n_base + 1, dtype=np.uint64) * G
    config = walker_config(draws.names, G, varying, [], [])
    meta = {
        "generator": "whatif-v2",
        "grouping": {"kind": "whatif", "G": G},
        "k": {"requested": k, "selected": len(cols)},
        "perturbation": {
            "features": varying,
            "feature_names": [draws.names[i] for i in varying],
            "perturbed_covariates": [draws.names[i] for i in cols],
            "pool": [draws.names[i] for i in draws.pool],
            "spec": WHATIF_V2[draws.dataset],
            "rules": WHATIF_V2_RULES,
            "time_step": "seeded per entity" if draws.horizon else None,
        },
        "seeds": {"generator": draws.seed},
    }
    return Workload(np.ascontiguousarray(X), offsets, config, draws.base_ids, meta)


# --- contracts -------------------------------------------------------------------


def _same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a == b) | (np.isnan(a) & np.isnan(b))


def contracts(w: Workload) -> dict[str, Any]:
    """Check what the walker config promises about every group.

    Constant features must be equal across each group's rows (NaN equals NaN).
    Monotonic features must not decrease (or increase) between consecutive
    non-missing values of a group; missing values are allowed, since a split's
    default direction routes them, and are counted.
    """
    X = w.X
    off = w.group_offsets.astype(np.int64)
    widths = np.diff(off)
    first = np.repeat(off[:-1], widths)
    varies = ~_same(X, X[first]).all(axis=0)  # per column: differs within some group
    cfg = w.config
    inc, dec = cfg["mono_inc_features"], cfg["mono_dec_features"]
    declared = set(cfg["varying_features"]) | set(inc) | set(dec)
    violations = [
        f"constant feature {cfg['feature_names'][c]} differs within a group"
        for c in np.nonzero(varies)[0]
        if int(c) not in declared
    ]
    group = np.repeat(np.arange(len(widths)), widths)
    missing = {}
    for f, sign in [(f, 1.0) for f in inc] + [(f, -1.0) for f in dec]:
        col = X[:, f]
        ok = ~np.isnan(col)
        if (n_missing := int((~ok).sum())) > 0:
            missing[cfg["feature_names"][f]] = n_missing
        v, g = col[ok], group[ok]
        same_group = g[1:] == g[:-1]
        if np.any(sign * (v[1:] - v[:-1])[same_group] < 0):
            kind = "increasing" if sign > 0 else "decreasing"
            violations.append(f"{kind} feature {cfg['feature_names'][f]} is not monotonic")
    return {
        "ok": not violations,
        "violations": violations,
        "missing_in_monotonic": missing,
        "varying_columns": [int(c) for c in np.nonzero(varies)[0]],
    }
