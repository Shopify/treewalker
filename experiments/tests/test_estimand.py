"""The camera-ready estimand: v1's nested medians over blocks and their interval."""

import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import polars as pl
import pytest

from treewalker_exp import analysis, estimand, grids, paths, results


def v1_percentile(sorted_values, p):
    """v1's src/bench/timing.rs:24-31: Rust's f64::round is half away from zero."""
    if not sorted_values:
        return 0.0
    x = p / 100.0 * (len(sorted_values) - 1)
    idx = int(Decimal(x).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return sorted_values[min(idx, len(sorted_values) - 1)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10, 11, 20, 21, 36, 101])
@pytest.mark.parametrize("p", [5.0, 50.0, 95.0])
def test_rank_index_is_v1s_nearest_rank(n, p):
    values = list(range(n))
    assert estimand.nearest_rank(np.array(values, dtype=float), p) == v1_percentile(values, p)


def test_the_median_of_an_even_count_is_the_upper_one():
    assert estimand.nearest_rank(np.array([1.0, 2.0, 3.0, 4.0]), 50) == 3.0
    assert estimand.nearest_rank(np.array([4.0, 1.0]), 50) == 4.0


def test_the_latency_is_the_median_of_block_medians_not_the_pooled_median():
    # Two blocks of three groups: medians 2 and 10, so the latency is the upper
    # median of the block medians, 10, while the pooled median of six is 9.
    ticks = np.array([1, 2, 3, 9, 10, 11])
    block = np.array([0, 0, 0, 1, 1, 1])
    vals = estimand.block_values(ticks, block)
    assert list(vals) == [2.0, 10.0]
    assert estimand.nested_median(vals)[0] == 10.0


def brute_force(ticks, block, ent, rnd, we, wr):
    """The nested median with every drawn entity and round repeated, literally."""
    medians = []
    for b in np.unique(block):
        sel = block == b
        vals = np.repeat(ticks[sel], we[ent[sel]].astype(int))
        if len(vals):
            medians += [estimand.nearest_rank(vals, 50)] * int(wr[rnd[sel][0]])
    return estimand.nearest_rank(np.array(medians), 50) if medians else math.nan


def test_the_bootstrap_recomputes_the_nested_medians_of_each_draw():
    rng = np.random.default_rng(7)
    for trial in range(60):
        n_ent, n_rounds, n_batches = (int(x) for x in rng.integers(1, [9, 5, 4]))
        batch_of = rng.integers(0, n_batches, n_ent)
        rows = [
            (int(rng.integers(1, 30)), r * n_batches + batch_of[e], e, r)
            for r in range(n_rounds)
            for e in range(n_ent)
            if rng.random() < 0.9
        ]
        if not rows:
            continue
        ticks, block, ent, rnd = (np.array(c) for c in zip(*rows, strict=True))
        m = estimand.Method(ticks, block, ent, rnd)
        we, wr = analysis.resample_weights(n_ent, n_rounds, 40, trial)
        got = m.boot(we, wr, chunk=11)
        want = [brute_force(ticks, block, ent, rnd, we[i], wr[i]) for i in range(40)]
        np.testing.assert_array_equal(got, want)
        assert (
            estimand.nested_median(m.point())[0]
            == estimand.nested_median(estimand.block_values(ticks, block))[0]
        )


def strata(sizes, timed, fraction):
    out = []
    for b in sorted({int(np.log2(s)) for s in sizes}):
        in_bin = [i for i, s in enumerate(sizes) if int(np.log2(s)) == b]
        out.append(
            {
                "min_size": 1 << b,
                "max_size": (2 << b) - 1,
                "pool": len(in_bin),
                "sampled": sum(timed[i] for i in in_bin),
            }
        )
    return {"fraction": fraction, "strata": out}


def test_a_systematic_equal_probability_sample_passes():
    sizes = [1] * 10 + [3] * 10 + [5] * 5
    # fraction 0.3: expected 3, 3 and 1.5, systematic quotas 3, 3, 2 (prefix 6 of 6, 8 of 7.5).
    timed = [i < 3 for i in range(10)] + [i < 3 for i in range(10)] + [i < 2 for i in range(5)]
    groups = pl.DataFrame({"rows": sizes, "timed": timed})
    estimand.check_equal_probability("c", {"sample": strata(sizes, timed, 0.3)}, groups)


def test_unequal_inclusion_probabilities_fail_loudly():
    sizes = [1] * 10 + [3] * 10
    # Every small group, few large ones: the first prefix takes 10 of an expected 5.
    timed = [True] * 10 + [i < 0 for i in range(10)]
    groups = pl.DataFrame({"rows": sizes, "timed": timed})
    with pytest.raises(ValueError, match="unequal inclusion"):
        estimand.check_equal_probability("c", {"sample": strata(sizes, timed, 0.5)}, groups)
    # A record that disagrees with the groups table.
    record = strata(sizes, [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5, 0.5)
    with pytest.raises(ValueError, match="disagrees"):
        estimand.check_equal_probability("c", {"sample": record}, groups)
    # Untimed groups without a sample record.
    with pytest.raises(ValueError, match="without a sample record"):
        estimand.check_equal_probability("c", {}, groups)


def synthetic_samples(rounds=6, batches=3, per_batch=8, seed=0):
    """Two methods timing the same 24 one-row groups each round: TreeWalker at
    100 ticks and the full walk at 300, with group and round noise."""
    rng = np.random.default_rng(seed)
    rows = []
    group_effect = rng.lognormal(0, 0.3, batches * per_batch)
    for r in range(rounds):
        shock = rng.lognormal(0, 0.05)
        for b in range(batches):
            for g in range(b * per_batch, (b + 1) * per_batch):
                for method, variant, base in (
                    ("treewalker", "all-on", 100),
                    ("treewalker_fullwalk", "-", 300),
                ):
                    t = base * group_effect[g] * shock * rng.lognormal(0, 0.02)
                    rows.append(("x/c", method, variant, "serving", r * batches + b, b, g, r, t))
    s = pl.DataFrame(
        rows,
        schema=["cell", "method", "variant", "mode", "block", "batch", "group", "repetition", "t"],
        orient="row",
    ).with_columns(
        ticks=pl.col("t").round().cast(pl.UInt64),
        n_groups=pl.lit(1, pl.UInt32),
        rows=pl.lit(1, pl.UInt32),
    )
    n = batches * per_batch
    groups = pl.DataFrame(
        {"cell": ["x/c"] * n, "group": range(n), "entity": range(100, 100 + n)}
    ).with_columns(rows=pl.lit(1), timed=pl.lit(True))
    return s.drop("t"), groups


def test_block_estimates_ratio_of_latencies_with_a_covering_interval():
    s, groups = synthetic_samples()
    est = estimand.block_estimates(s, groups, {"x/c": {}}, n_boot=200)
    tw, fw = (
        est[("x/c", "treewalker", "all-on", "serving")],
        est[("x/c", "treewalker_fullwalk", "-", "serving")],
    )
    assert tw["blocks"] == fw["blocks"] == 18
    assert fw["ratio"] == pytest.approx(fw["latency"] / tw["latency"])
    assert fw["lo"] < fw["ratio"] < fw["hi"]
    assert fw["lo"] < 3.0 < fw["hi"]
    assert fw["p5"] <= fw["latency"] <= fw["p95"]
    assert tw["ratio"] == 1.0 and tw["lo"] == tw["hi"] == 1.0


def test_batch_mode_takes_ticks_over_groups_per_block_without_an_interval():
    s = pl.DataFrame(
        {
            "cell": ["x/c"] * 4,
            "method": ["treewalker", "treewalker", "treewalker_fullwalk", "treewalker_fullwalk"],
            "variant": ["all-on", "all-on", "-", "-"],
            "mode": ["batch"] * 4,
            "block": [0, 1, 0, 1],
            "group": [0, 0, 0, 0],
            "repetition": [0, 1, 0, 1],
            "n_groups": [4, 4, 4, 4],
            "ticks": [400, 480, 1200, 1600],
        }
    )
    groups = pl.DataFrame({"cell": ["x/c"] * 4, "group": range(4), "entity": range(4)})
    groups = groups.with_columns(rows=pl.lit(1), timed=pl.lit(True))
    est = estimand.block_estimates(s, groups, {"x/c": {}})
    fw = est[("x/c", "treewalker_fullwalk", "-", "batch")]
    # Block values 300 and 400 (upper median 400) over 100 and 120 (upper 120).
    assert fw["latency"] == 400 and fw["ratio"] == pytest.approx(400 / 120) and fw["lo"] is None


def test_scenario_cells_parse_with_the_released_model():
    d = results.design("scenario_credit/lightgbm/whatif-v1-k8-G4")
    assert (d["dataset"], d["T"], d["L"], d["k"], d["G"], d["family"]) == (
        "scenario_credit",
        500,
        8,
        8,
        4,
        "whatif",
    )
    w = grids.load(paths.Paths(paths.find_repo()).grids)["workloads"]["whatif-v1"]
    assert (w["n_trees"], w["max_depth"]) == (
        [results.SCENARIO_MODEL[0]],
        [results.SCENARIO_MODEL[1]],
    )


def test_the_campaign_defaults_to_the_camera_ready_runs_where_present(tmp_path):
    assert results.campaign(tmp_path).name == "final"
    for arch in results.MACHINES.values():
        (tmp_path / f"camera-ready-{arch}").mkdir()
    c = results.campaign(tmp_path)
    assert c.name == "camera-ready" and c.scenario == "scenario-v1"
    assert c.ablation == ("camera-ready-ablation-1", "camera-ready-ablation-2")
    assert results.campaign(tmp_path, "final").factorial == "factorial"
    with pytest.raises(ValueError):
        results.campaign(tmp_path, "nope")
