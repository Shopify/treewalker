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
    assert w.config["mono_inc_features"] == [3, 4]
    assert w.config["mono_dec_features"] == [5, 6]


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
