import math

import numpy as np
import pytest

from treewalker_exp import datasets as ds
from treewalker_exp import workloads as wl


def released_tv(t, horizon, x0):
    """The released per-row expansion (utils.py _build_tv_features)."""
    h1 = max(horizon - 1, 1)
    frac = t / h1
    remaining = h1 - t
    return [
        float(t),
        frac**2,
        float(remaining),
        (remaining / h1) ** 2,
        x0 * frac,
        math.sin(math.pi * frac),
    ]


@pytest.mark.parametrize("horizon", [1, 2, 16, 104, 1008])
def test_tv_features_bit_identical_to_released(horizon):
    rng = np.random.default_rng(0)
    steps = rng.integers(0, horizon, 50)
    x0 = rng.normal(size=50) * 100
    got = ds.tv_features(steps, x0, horizon)
    want = np.array([released_tv(int(t), horizon, x) for t, x in zip(steps, x0, strict=True)])
    assert np.array_equal(got.view("<u8"), want.view("<u8"))


def cov(n=5, c=3):
    rng = np.random.default_rng(1)
    return ds.Covariates(
        names=[f"x{i}" for i in range(c)],
        values=rng.normal(size=(n, c)),
        cat_indices=[],
        rows=np.arange(n),
    )


def test_panel_matches_released_expansion():
    c = cov()
    w = wl.panel(c, 8)
    want = np.array(
        [list(c.values[i]) + released_tv(t, 8, c.values[i, 0]) for i in range(5) for t in range(8)]
    )
    assert np.array_equal(np.asarray(w.X).view("<u8"), want.view("<u8"))
    assert w.config["varying_features"] == list(range(3, 3 + ds.N_TV))


def test_panel_contracts_hold():
    w = wl.panel(cov(), 8)
    assert w.X.shape == (40, 3 + ds.N_TV)
    c = wl.contracts(w)
    assert c["ok"], c
    assert w.config["mono_inc_features"] == [3, 4]  # x0 < 0 for some patients
    assert w.config["mono_dec_features"] == [5, 6]


def test_panel_declares_the_interaction_increasing_when_x0_is_nonnegative():
    c = cov()
    c.values[:, 0] = np.abs(c.values[:, 0])
    c.values[1, 0] = np.nan  # a missing x0 leaves its rows missing, not decreasing
    c.values[2, 0] = 0.0  # constant 0 along the panel is monotonic too
    w = wl.panel(c, 8)
    assert w.config["mono_inc_features"] == [3, 4, 3 + ds.INTERACTION]
    assert wl.contracts(w)["ok"]


def test_contracts_catch_violations():
    w = wl.panel(cov(), 4)
    X = np.array(w.X)
    X[1, 0] += 1.0  # a constant covariate changes inside group 0
    X[6, 3] = -1.0  # time_step decreases inside group 1
    bad = wl.Workload(X, None, w.config, w.entities)
    c = wl.contracts(bad)
    assert not c["ok"]
    assert len(c["violations"]) == 2


def test_contracts_allow_missing_in_monotonic():
    w = wl.panel(cov(), 4)
    X = np.array(w.X)
    X[2, 3] = np.nan
    c = wl.contracts(wl.Workload(X, None, w.config, w.entities))
    assert c["ok"]
    assert c["missing_in_monotonic"] == {"time_step": 1}


def sessions(widths):
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.uint64)
    n = int(offsets[-1])
    X = np.column_stack([np.repeat(np.arange(len(widths)), widths), np.arange(n)]).astype(float)
    cfg = wl.walker_config(["sid", "row"], max(widths), [1], [], [])
    return wl.Workload(X, offsets, cfg, np.arange(100, 100 + len(widths)))


def test_cohort_is_shared_nested_and_keeps_the_session_order():
    s = sessions([40, 5, 32, 37, 31, 64])
    out = wl.ranking_cohort(s, 32, [4, 8, 16, 32])
    cohort = [0, 2, 3, 5]
    off = s.group_offsets.astype(int)
    for n, w in out.items():
        assert w.entities.tolist() == [100 + g for g in cohort]  # one cohort for every size
        assert np.all(np.diff(w.group_offsets.astype(int)) == n)
        groups = w.X[:, 1].reshape(len(cohort), n)
        # The first n candidates of each session, in the session's order: so the
        # sizes nest, and a row has the same position at every size.
        want = np.array([np.arange(off[g], off[g] + n) for g in cohort])
        assert np.array_equal(groups, want)
        assert np.all(w.X[:, 0].reshape(len(cohort), n) == np.array(cohort)[:, None])


def test_session_order_shuffles_within_sessions_only():
    offsets = np.array([0, 40, 45, 77, 114], dtype=np.uint64)
    order = wl.session_order(offsets, seed=5)
    off = offsets.astype(int)
    for g in range(len(off) - 1):
        rows = order[off[g] : off[g + 1]]
        assert sorted(rows.tolist()) == list(range(off[g], off[g + 1]))  # same session
    assert not np.array_equal(order, np.arange(off[-1]))  # reordered
    assert np.array_equal(order, wl.session_order(offsets, seed=5))  # seeded
    assert not np.array_equal(order, wl.session_order(offsets, seed=6))


def expedia_split(widths):
    import polars as pl

    sid = np.repeat(np.arange(len(widths)), widths)
    n = len(sid)
    cols = {f: np.zeros(n) for f in ds.EXPEDIA_FEATURES}
    cols["prop_id"] = np.arange(n, dtype=float)  # sorted, as in the file
    test = pl.DataFrame({"srch_id": sid, **cols})
    return ds.RankingSplit(9, 0.2, 0, "fp", True, test.head(0), test)


def test_ranking_sessions_shuffle_candidates_and_declare_search_level_constants():
    w = wl.ranking_sessions(expedia_split([6, 40, 3]))
    names = w.config["feature_names"]
    varying = {names[i] for i in w.config["varying_features"]}
    assert varying == set(ds.EXPEDIA_VARYING) and "prop_country_id" in varying
    assert set(names) - varying == set(ds.EXPEDIA_SESSION)
    prop = w.X[:, names.index("prop_id")]
    off = w.group_offsets.astype(int)
    assert sorted(prop.tolist()) == list(range(49))
    assert not np.all(np.diff(prop[off[1] : off[2]]) > 0)  # no longer sorted by prop_id
    for g in range(3):  # each session keeps its own candidates
        assert sorted(prop[off[g] : off[g + 1]].tolist()) == list(range(off[g], off[g + 1]))
    assert w.meta["generator"] == "ranking-sessions-v2"


def draws(horizon=None, n_cov=4):
    rng = np.random.default_rng(3)
    names = [f"x{i}" for i in range(n_cov)] + (ds.TV_FEATURE_NAMES if horizon else [])
    return wl.whatif_v2_draws(
        "support" if horizon else "credit",
        base=rng.normal(size=(30, n_cov)),
        base_ids=np.arange(30),
        donors=rng.normal(size=(50, n_cov)),
        pool=list(range(n_cov)),
        names=names,
        horizon=horizon,
    )


def test_whatif_v2_feature_ranking_and_realized_k():
    d = draws()
    counts = np.array([5, 0, 9, 5])
    assert wl.whatif_v2_features(d, counts, 2) == [2, 0]  # most-split, ties by index
    assert wl.whatif_v2_features(d, counts, 8) == [2, 0, 3]  # never-split features skipped
    w = wl.whatif_v2(d, wl.whatif_v2_features(d, counts, 8), 8, 4)
    assert w.meta["k"] == {"requested": 8, "selected": 3}
    assert len(wl.contracts(w)["varying_columns"]) == 3


def test_whatif_v2_draws_coupled_across_G_and_k():
    d = draws()
    big = wl.whatif_v2(d, [0, 1, 2], 4, 128).X.reshape(30, 128, -1)
    small = wl.whatif_v2(d, [0, 1, 2], 4, 16).X.reshape(30, 16, -1)
    assert np.array_equal(small, big[:, :16])  # size-G group = first G scenarios
    fewer = wl.whatif_v2(d, [0, 2], 2, 16).X.reshape(30, 16, -1)
    assert np.array_equal(fewer[..., [0, 2]], small[..., [0, 2]])  # same draws, nested k
    assert np.array_equal(fewer[..., 1], np.repeat(d.base[:, 1:2], 16, axis=1))


def test_whatif_v2_scenario0_and_joint_donor():
    d = draws()
    g = wl.whatif_v2(d, [1, 3], 2, 8).X.reshape(30, 8, -1)
    assert np.array_equal(g[:, 0], d.base)
    donor = d.donors[d.donor_index[:, :7]]
    assert np.array_equal(g[:, 1:, [1, 3]], donor[:, :, [1, 3]])


def test_whatif_v2_survival_recomputes_interaction():
    d = draws(horizon=16, n_cov=4)
    nc = 4
    inter = nc + ds.INTERACTION
    w = wl.whatif_v2(d, [0], 1, 8)
    assert inter in w.config["varying_features"]
    X = w.X.reshape(30, 8, -1)
    steps = d.steps
    frac = ds.tv_table(16)[steps, ds.INTERACTION]
    assert np.array_equal(X[:, :, inter], X[:, :, 0] * frac[:, None])
    for col in range(nc, nc + ds.N_TV):  # the time step is fixed within a group
        if col != inter:
            assert np.all(X[:, :, col] == X[:, :1, col])
    assert wl.contracts(w)["ok"]
    other = wl.whatif_v2(d, [2], 1, 8)
    assert inter not in other.config["varying_features"]
    assert wl.contracts(other)["ok"]


def test_whatif_v1_matches_released_loop():
    rng = np.random.default_rng(5)
    base = rng.uniform(0, 10, size=(6, 5))
    base[0, 1] = 0.0
    perturb, G = [1, 3], 4
    got, offsets = wl.whatif_v1(base, perturb, G)
    r = np.random.default_rng(wl.WHATIF_V1_SEED)
    mult = np.power(2.0, r.uniform(-1.0, 1.0, size=(6, G - 1, 2)))
    want = np.empty((6 * G, 5))
    for b in range(6):
        want[b * G] = base[b]
        for v in range(1, G):
            want[b * G + v] = base[b]
            for j, fi in enumerate(perturb):
                want[b * G + v, fi] = base[b, fi] * mult[b, v - 1, j]
    assert np.array_equal(got.view("<u8"), want.view("<u8"))
    assert offsets.tolist() == list(range(0, 6 * G + 1, G))
