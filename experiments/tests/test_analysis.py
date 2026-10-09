"""The acceptance estimators on synthetic samples with known answers."""

from pathlib import Path

import numpy as np

from treewalker_exp import analysis


def test_bootstrap_ratio_is_row_weighted_and_covers_the_truth():
    rng = np.random.default_rng(1)
    n_ent, reps = 200, 4
    entity = np.repeat(np.arange(n_ent), reps)
    rows = np.repeat(rng.integers(1, 40, n_ent), reps)
    base = rows * 100.0 * rng.lognormal(0, 0.05, entity.size)
    other = base * 2.5 * rng.lognormal(0, 0.05, entity.size)
    ticks = np.column_stack([base, other])
    est, lo, hi = analysis.bootstrap_ratio(ticks, entity, n_boot=400)
    assert lo < 2.5 < hi
    assert est == other.sum() / base.sum()


def blocks(rows):
    """XGBoost blocks: (process, block, batch, ticks per row, instructions per row)."""
    import polars as pl

    return pl.DataFrame(
        {
            "process": [r[0] for r in rows],
            "block": [r[1] for r in rows],
            "batch": [r[2] for r in rows],
            "repetition": [r[1] // 12 for r in rows],
            "ticks": [r[3] * 100.0 for r in rows],
            "rows": [100] * len(rows),
            "instructions": [None if r[4] is None else r[4] * 100.0 for r in rows],
        }
    )


def faster_labels(r):
    """Per block, sorted by process and block: True, False, or None outside mixed batches."""
    return r["labels"].sort("process", "block")["faster"].to_list()


def test_modes_switching_within_a_process_split_by_instructions():
    # support/nt500_md2_h8 serving in the shakedown: process 0 switches between
    # 78.6k instructions a row (15.6k ticks) and 71.7k (18.1k ticks), 1.097x
    # apart, which a per-process 1.10 gap merged.
    rng = np.random.default_rng(0)
    rows = []
    for blk in range(48):
        fast = blk < 12 or blk >= 36
        ipr, tpr = (78_600, 15_600) if fast else (71_700, 18_100)
        rows.append((0, blk, blk % 12, tpr * rng.normal(1, 0.003), ipr * rng.normal(1, 0.001)))
    for k in (1, 2, 3):  # children, one round each, in the slow mode
        rows += [(k, b, b, 18_100 * rng.normal(1, 0.003), 71_700.0) for b in range(12)]
    r = analysis.xgboost_faster_mode(blocks(rows))
    assert r["outcome"] == "two modes" and r["mixed_share"] == 1.0
    assert abs(r["faster"] / 15_600 - 1) < 0.01
    assert abs(r["gap_pct"] - 9.6) < 0.5 and abs(r["time_gap_pct"] - 16.0) < 1
    # The higher-instruction cluster is the faster one, in every batch.
    assert r["lower_faster_share"] == 0.0
    assert r["processes_mixed"] == 1 and r["processes_all_faster"] == 0
    assert r["processes"] == 4


def test_without_counters_there_is_no_mode_analysis():
    # The variance check's credit serving modes, 7.5% apart in time, one mode a
    # process: without counters they cannot be told from noise or batch effects.
    rng = np.random.default_rng(1)
    rows = []
    for k in range(8):
        base = 7_440 if k % 2 else 8_130
        for rnd in range(3 if k == 0 else 1):
            rows += [(k, rnd * 12 + b, b, base * rng.normal(1, 0.002), None) for b in range(12)]
    r = analysis.xgboost_faster_mode(blocks(rows))
    assert r["outcome"] == "no counters" and "faster" not in r and "labels" not in r
    # Each process weighs the same in the expected time; process 0 times 3 rounds.
    per = [np.mean([x[3] for x in rows if x[0] == k]) for k in range(8)]
    assert abs(r["expected"] / np.mean(per) - 1) < 1e-12
    assert "no counters, no mode analysis" in analysis.xgboost_line("serving", r, 1.0)
    # On Linux (x86 or Arm), where counters are required: unverifiable.
    r = analysis.xgboost_faster_mode(blocks(rows), counters_required=True)
    assert r["outcome"] == "unverifiable"


def test_a_fast_child_sets_the_headline_over_a_slow_main_process():
    import polars as pl

    rows = [(0, b, b % 12, 200.0, 54_000.0) for b in range(36)]
    rows += [(1, b, b, 100.0, 29_000.0) for b in range(12)]
    r = analysis.xgboost_faster_mode(blocks(rows))
    assert r["faster"] == 100.0 and r["processes_all_faster"] == 1
    # The mean over processes, not over blocks: process 0 times three rounds.
    assert r["expected"] == (200 + 100) / 2
    assert r["weights"] == [(b, 100.0) for b in range(12)]
    # TreeWalker over the same batches and weights.
    tw = pl.DataFrame({"batch": [0, 1, 2], "ticks": [30.0, 80.0, 999.0], "rows": [3, 4, 1]})
    assert analysis.batch_weighted(tw, [(0, 1.0), (1, 3.0)]) == (10 + 3 * 20) / 4


FIXTURE = Path(__file__).parent / "fixtures" / "xgboost_switching_blocks.csv"


def test_modes_switching_within_rounds_on_real_blocks():
    # support/nt1000_md2_h8 serving, the shakedown's Intel factorial: process 0
    # switches 39 times over 120 blocks, between 132.1k and 147.7k instructions a
    # row, so batches hold different mode mixes. Batch-median centring labelled
    # 4 blocks faster against a true 47 or 73. Block 0 (round 0, batch 0) reads
    # 138.3k: it switched mid-block. Where rounds leave it beside slow blocks only
    # (rounds 0, 7 and 9), batch 0 splits with the partial block as its faster
    # cluster, which read 1.4% slow; its gap is off the median gap, so batch 0 is
    # left out as unknown, as it is wherever the partial block skews its gap.
    import itertools

    import polars as pl

    real = pl.read_csv(FIXTURE).with_columns(process=pl.lit(0))
    truth_split = analysis.split_modes(
        np.log((real["instructions"] / real["rows"]).to_numpy()), np.log(1.02)
    )
    tpr = (real["ticks"] / real["rows"]).to_numpy()
    fast_label = int(tpr[truth_split == 1].mean() < tpr[truth_split == 0].mean())
    real = real.with_columns(truth=pl.Series(truth_split == fast_label))
    for rounds in [tuple(range(10)), *itertools.combinations(range(10), 3)]:
        sub = real.filter(pl.col("repetition").is_in(rounds))
        r = analysis.xgboost_faster_mode(sub.drop("truth"))
        mixed = sub.group_by("batch").agg(pl.col("truth").n_unique() == 2)
        mixed = sorted(mixed.filter(pl.col("truth"))["batch"].to_list())
        if not mixed:
            assert r["outcome"] == "no second mode visible", rounds
            continue
        # The mixed batches are those holding both true modes, and their labels
        # are the truth; the rest are unknown.
        assert r["outcome"] == "two modes", rounds
        got = [k for k, _ in r["weights"]]
        odd = r["odd_gap_batches"]
        assert set(odd) <= {0} and (rounds != (0, 7, 9) or odd == [0]), rounds
        assert set(got) == set(mixed) - set(odd) and r["lower_faster_share"] == 0.0, rounds
        lab = r["labels"].join(sub.select("block", "truth"), on="block")
        known = lab.filter(pl.col("faster").is_not_null())
        assert (known["faster"] == known["truth"]).all(), rounds
        assert known["batch"].is_in(got).all(), rounds
        fast = sub.filter(pl.col("truth") & pl.col("batch").is_in(got))
        per = fast.group_by("batch").agg(t=pl.col("ticks").sum() / pl.col("rows").sum())
        per = per.join(pl.DataFrame(r["weights"], schema=["batch", "w"], orient="row"), on="batch")
        want = float((per["t"] * per["w"]).sum() / per["w"].sum())
        # The partial block shows: off its cluster's level beside other blocks of
        # it, or alone in its cluster with an odd gap.
        assert (r["off_level"] + len(odd) > 0) == (0 in rounds), rounds
        assert abs(r["faster"] / want - 1) < 1e-12, rounds


def test_mode_shares_that_differ_by_batch_are_labelled_exactly():
    # rev-gpt's synthetic: processes 0-3 fast in batches 0-5, 4-6 in batches
    # 6-11; batch 6-11 rows cost twice as much; the modes are 7.5% apart in time.
    rows, truth = [], []
    for p in range(7):
        for rnd in range(3 if p == 0 else 1):
            for b in range(12):
                fast = (p <= 3) if b < 6 else (p >= 4)
                base = 100 if b < 6 else 200
                rows.append(
                    (p, rnd * 12 + b, b, base if fast else 1.075 * base, 29_000 if fast else 54_000)
                )
                truth.append(fast)
    r = analysis.xgboost_faster_mode(blocks(rows))
    assert r["outcome"] == "two modes" and faster_labels(r) == truth
    # Each batch's rows weigh the same: (100 + 200) / 2, not the faster blocks
    # pooled (133.3), which weighs a batch by how many faster blocks it holds.
    assert abs(r["faster"] - 150.0) < 1e-9
    assert r["lower_faster_share"] == 1.0 and r["processes_mixed"] == 7


def test_unread_blocks_leave_the_rest_by_instructions():
    rows = [(0, b, b % 12, 200.0, 54_000.0) for b in range(36)]
    rows += [(1, b, b, 100.0, 29_000.0) for b in range(12)]
    rows[5] = (0, 5, 5, 200.0, None)
    r = analysis.xgboost_faster_mode(blocks(rows))
    assert r["outcome"] == "two modes" and r["unread_blocks"] == 1
    assert r["faster"] == 100.0 and len(r["labels"]) == 47


def misoriented(time_gap=1.075):
    """rev-gpt's round-3 case: the low-instruction label costs `time_gap` more in
    every batch, but its batch mix puts it in the cheap batches more often."""
    rows, true_fast = [], []
    for p in range(7):
        for rnd in range(3 if p == 0 else 1):
            for b in range(12):
                low = (p <= 3) if b < 6 else (p >= 4)
                base = 100 if b < 6 else 200
                rows.append(
                    (p, rnd * 12 + b, b, base * (time_gap if low else 1), 29_000 if low else 54_000)
                )
                true_fast.append(not low)
    return rows, true_fast


def test_the_faster_mode_is_oriented_inside_batches():
    rows, truth = misoriented()
    r = analysis.xgboost_faster_mode(blocks(rows))
    assert r["outcome"] == "two modes" and faster_labels(r) == truth
    assert abs(r["faster"] - 150.0) < 1e-9 and r["lower_faster_share"] == 0.0
    assert abs(r["time_gap_pct"] - 7.5) < 1e-6
    rows, truth = misoriented(1.015)
    r = analysis.xgboost_faster_mode(blocks(rows))
    assert faster_labels(r) == truth and abs(r["time_gap_pct"] - 1.5) < 1e-6


def test_an_orientation_that_flips_across_batches_takes_each_batchs_faster_cluster():
    # The low-instruction cluster is 10% slower in batches 0-5 and 10% faster in
    # 6-11. Each batch takes its faster cluster, 100; either cluster averaged over
    # the batches would read 105.
    rows, truth = [], []
    for p in range(4):
        for b in range(12):
            low = p % 2 == 0
            slow = low == (b < 6)
            rows.append((p, b, b, 110.0 if slow else 100.0, 29_000 if low else 54_000))
            truth.append(not slow)
    r = analysis.xgboost_faster_mode(blocks(rows))
    assert r["outcome"] == "two modes" and faster_labels(r) == truth
    assert r["faster"] == 100.0 and r["lower_faster_share"] == 0.5
    assert r["processes_mixed"] == 4 and r["processes_all_faster"] == 0
    line = analysis.xgboost_line("serving", r, 1.0)
    assert "the lower-instruction cluster faster in 50% of them" in line


def test_a_cell_without_a_mixed_batch_shows_no_second_mode_and_flags_its_span():
    def cell(level, tpr=lambda b: 100.0):
        return [
            (p, rnd * 12 + b, b, tpr(b), level(b))
            for p in range(7)
            for rnd in range(3 if p == 0 else 1)
            for b in range(12)
        ]

    # Batch levels 0.1% apart: no flag. The children read 2% slow: the headline
    # is process 0's time, timed like TreeWalker, with the mean over processes
    # beside it.
    rows = cell(lambda b: 71_700 * (1 + 0.001 * b))
    rows = [(p, k, b, t * (1.02 if p else 1.0), i) for p, k, b, t, i in rows]
    r = analysis.xgboost_faster_mode(blocks(rows))
    assert r["outcome"] == "no second mode visible" and not r["span_flag"]
    assert abs(r["span_pct"] - 1.1) < 1e-9 and r["faster"] == r["process0"] == 100.0
    assert abs(r["expected"] - (100 + 6 * 102) / 7) < 1e-9
    line = analysis.xgboost_line("serving", r, 1.0)
    assert "process 0's time 100.0000 us/row (timed like TreeWalker)" in line
    assert "expected over processes 101.7143 us/row (the mean of 7)" in line
    assert r["weights"] == [(b, 100.0) for b in range(12)]
    assert all(x is None for x in faster_labels(r))
    # rev-gpt's round-6 and round-7 attacks: no batch mixed, batches 0-5 slow and
    # 6-11 fast, or one batch fast, 9.6% apart in instructions. Nothing inside a
    # batch tells modes from batch effects, so the all-process time (the mixture)
    # shows, with the span flagged.
    for fast in (set(range(6, 12)), {11}, set(range(11))):
        r = analysis.xgboost_faster_mode(
            blocks(
                cell(
                    lambda b, f=fast: 78_600 if b in f else 71_700,
                    lambda b, f=fast: 100.0 if b in f else 107.5,
                )
            )
        )
        assert r["outcome"] == "no second mode visible" and r["span_flag"], fast
        assert abs(r["span_pct"] - 9.623) < 1e-2 and r["faster"] == r["process0"]
        line = analysis.xgboost_line("serving", r, 1.0)
        assert "(FLAG: above 2%; content or a mode every process shares)" in line
    # One stray block on its own side of a batch is not a mode.
    rows = cell(lambda b: 71_700.0)
    rows[5] = (*rows[5][:4], 78_600.0)
    assert analysis.xgboost_faster_mode(blocks(rows))["outcome"] == "no second mode visible"


def test_a_partial_block_near_the_faster_level_and_a_counter_outlier():
    # rev-gpt's round-9 cases: full slow (71,700 instructions a row, 107.5 ticks)
    # and fast (78,600, 100) blocks in every batch from the children, and
    # process 0's round-0 block of each batch in between (75,300, 103.5), nearer
    # the fast level. It joins the faster cluster: at most a quarter of the time
    # gap in its batch, and counted off level.
    def rows(first):
        out = []
        for p in range(7):
            for rnd in range(3 if p == 0 else 1):
                for b in range(12):
                    if p == 0 and rnd == 0:
                        out.append((p, b, b, *first(b)))
                    elif p % 2:
                        out.append((p, rnd * 12 + b, b, 100.0, 78_600.0))
                    else:
                        out.append((p, rnd * 12 + b, b, 107.5, 71_700.0))
        return out

    r = analysis.xgboost_faster_mode(blocks(rows(lambda b: (103.5, 75_300.0))))
    assert r["outcome"] == "two modes" and r["odd_gap_batches"] == []
    assert 100.0 < r["faster"] < 100.0 + 7.5 / 4 and r["off_level"] == 12
    # A first block far above both levels (95,000) puts both modes on one side
    # of batch 0's largest gap: its gap is off the median, so batch 0 is left
    # out as unknown and the headline is the faster mode's.
    r = analysis.xgboost_faster_mode(
        blocks(rows(lambda b: (101.0, 95_000.0) if b == 0 else (100.0, 78_600.0)))
    )
    assert r["odd_gap_batches"] == [0] and r["faster"] == 100.0
    assert [k for k, _ in r["weights"]] == list(range(1, 12)) and r["mixed_share"] == 11 / 12
    assert all(x is None for x in r["labels"].filter(r["labels"]["batch"] == 0)["faster"])
    assert "left out as unknown" in analysis.xgboost_line("serving", r, 1.0)


def unmatched(mixed, effect=0.12):
    """rev-gpt's round-4 case: 12 batches, `mixed` holding both modes; the others
    hold one each, whose instructions carry a batch effect of +-`effect`. The
    fast mode (78,600 instructions a row) costs 100 ticks a row, the slow 107.5."""
    rows, truth = [], []
    for p in range(7):
        for rnd in range(3 if p == 0 else 1):
            for b in range(12):
                low = (p % 2 == 0) if b < mixed else (b < (mixed + 12) // 2)
                fx = 1 if b < mixed else ((1 + effect) if low else (1 - effect))
                ins = (71_700 if low else 78_600) * fx
                rows.append((p, rnd * 12 + b, b, 107.5 if low else 100.0, ins))
                truth.append(not low)
    return rows, truth


def test_single_mode_batches_are_unknown():
    # rev-gpt's round-4 case: pure batches with +-12% instruction offsets.
    for mixed in (6, 10):
        rows, truth = unmatched(mixed)
        r = analysis.xgboost_faster_mode(blocks(rows))
        got = faster_labels(r)
        assert all(g is None for g, row in zip(got, rows, strict=True) if row[2] >= mixed)
        assert all(g == t for g, t in zip(got, truth, strict=True) if g is not None), mixed
        assert r["faster"] == 100.0 and r["mixed_share"] == mixed / 12


def test_pure_batches_a_gap_away_are_unknown_not_the_other_mode():
    # rev-gpt's round-5 attack: each pure batch's offset is one gap, plus or
    # minus 0.2%, so it lands on the wrong level; the faster mode costs 100 in
    # every batch, the slower 107.5.
    for mixed in (1, 2, 6, 10, 12):
        for perturb in (0.0, 0.002, -0.002):
            rows, truth = [], []
            for p in range(7):
                for rnd in range(3 if p == 0 else 1):
                    for b in range(12):
                        low = (p % 2 == 0) if b < mixed else b < (mixed + 12) // 2
                        shift = (78_600 / 71_700 if low else 71_700 / 78_600) * np.exp(perturb)
                        fx = 1 if b < mixed else shift
                        ins = (71_700 if low else 78_600) * fx
                        rows.append((p, rnd * 12 + b, b, 107.5 if low else 100.0, ins))
                        truth.append(not low)
            r = analysis.xgboost_faster_mode(blocks(rows))
            got = faster_labels(r)
            assert all(g == t for g, t in zip(got, truth, strict=True) if g is not None)
            assert sum(g is not None for g in got) == 9 * mixed, (mixed, perturb)
            assert r["faster"] == 100.0 and r["mixed_share"] == mixed / 12


def test_split_modes_needs_two_values_a_side():
    v = np.log(np.array([1.0, 1.001, 1.002, 1.5]))
    assert analysis.split_modes(v, np.log(1.02)).tolist() == [0, 0, 0, 0]
    v = np.log(np.array([1.0, 1.001, 1.5, 1.501]))
    assert analysis.split_modes(v, np.log(1.02)).tolist() == [0, 0, 1, 1]


def test_a_shared_round_shock_is_one_observation():
    # Every entity has the same three round ratios: three observations, however
    # many entities; the interval must not shrink as entities grow.
    widths = []
    for n_ent in (10, 1000):
        entity = np.repeat(np.arange(n_ent), 3)
        rounds = np.tile(np.arange(3), n_ent)
        ticks = np.column_stack([np.ones(3 * n_ent), np.tile([1.0, 2.0, 3.0], n_ent)])
        _, lo, hi = analysis.bootstrap_ratio(ticks, entity, 300, rounds=rounds)
        widths.append(hi - lo)
    assert widths[1] > 0.5 and widths[1] > 0.5 * widths[0]


def test_sentinel_lines_show_failures_flags_and_the_first_load():
    def rec(index, role, cells, status="ok", **totals):
        return {
            "index": index,
            "role": role,
            "cells_done": cells,
            "unix_time": 0,
            "status": status,
            "totals": {k: list(v) for k, v in totals.items()},
        }

    run = {
        "run": {"sentinel_drift_pct": 3.0},
        "sentinel": {
            "cell": "c",
            "records": [
                rec(0, "warm-up", 0, lgb=(1140, 10), tw=(500, 10)),
                rec(1, "measure", 0, lgb=(1000, 10), tw=(500, 10)),
                rec(2, "measure", 25, lgb=(960, 10), tw=(510, 10)),
                rec(3, "measure", 50, status="failed: x"),
                rec(4, "warm-up", 50, status="failed: y"),
                rec(5, "measure", 50, lgb=(1035, 10), tw=(0, 0)),
            ],
        },
    }
    text = "\n".join(analysis.sentinel_lines(run))
    assert "reference 1, drift above 3.0%" in text
    assert "1 measure after 0 cells: ok; first-load effect lgb +14.0%, tw +0.0%" in text
    assert "2 measure after 25 cells: ok; flagged lgb -4.0%\n" in text
    assert "failed: x" in text and "failed: y" in text
    # 3.5% is flagged; no first load after a failed warm-up.
    assert (
        "5 measure after 50 cells: ok; flagged lgb +3.5%" in text
        and "first-load" not in text.split("\n")[-1]
    )
    t, ref = analysis.sentinel_table(run["sentinel"], 3.0)
    assert ref == 1 and len(t) == 10
    # Failed attempts keep a row, with a null key and null values; the
    # reference's own drift and a key with no rows are null, never NaN.
    failed = t.filter(t["sentinel"] == 3).row(0, named=True)
    assert failed["key"] is None and failed["ticks_per_row"] is None and failed["ratio"] is None
    assert t.filter(t["sentinel"] == 1)["ratio"].null_count() == 2
    tw5 = t.filter((t["sentinel"] == 5) & (t["key"] == "tw")).row(0, named=True)
    assert tw5["ticks_per_row"] is None and tw5["flagged"] is None
    assert analysis.sentinel_lines({"sentinel": {"skipped": "why"}}) == ["sentinel: none (why)"]
    # Exactly 3% either way is within; just beyond is flagged.
    edge = {
        "records": [
            rec(0, "measure", 0, a=(100, 1), b=(100, 1), c=(100, 1), d=(100, 1)),
            rec(1, "measure", 1, a=(97, 1), b=(103, 1), c=(9699, 100), d=(10301, 100)),
        ]
    }
    t, _ = analysis.sentinel_table(edge, 3.0)
    flags = dict(t.filter(t["sentinel"] == 1).select("key", "flagged").rows())
    assert flags == {"a": False, "b": False, "c": True, "d": True}


def test_conversion_share_pools_ticks_per_method():
    passes = [
        {"method": "quickscorer", "conversion_ticks": 10, "ticks": 100},
        {"method": "quickscorer", "conversion_ticks": 30, "ticks": 100},
        {"method": "other", "conversion_ticks": 0, "ticks": 0},
    ]
    assert analysis.conversion_share(passes) == {"quickscorer": 0.2}


def test_bootstrap_resamples_entities_not_repetitions():
    # Two entities with very different ratios: the interval must reflect them.
    entity = np.array([0] * 50 + [1] * 50)
    ticks = np.column_stack([np.ones(100), np.r_[np.full(50, 1.0), np.full(50, 3.0)]])
    _, lo, hi = analysis.bootstrap_ratio(ticks, entity, n_boot=400)
    assert hi - lo > 0.5
