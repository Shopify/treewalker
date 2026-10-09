"""pack, summarize and validation-readout end to end, on a small Linux-like run with
counters and XGBoost children (rev-opus n44, round 9)."""

from pathlib import Path

import polars as pl

from treewalker_exp import analysis, pack, readout
from treewalker_exp import formats as fm

BATCHES, ROUNDS, CHILDREN = 4, 3, 3


def write_cell(run: Path, cell: str, xgboost, rounds: int = ROUNDS) -> None:
    """One serving cell of 8 one-row groups in 4 batches: TreeWalker in process 0
    at 10 ticks a row, XGBoost in process 0 for `rounds` rounds and in three
    children for one; `xgboost(process)` gives its (ticks, instructions) a row."""
    samples, hw = [], []
    for method, procs in (("treewalker", [0]), ("xgboost_native", range(CHILDREN + 1))):
        variant = "all-on" if method == "treewalker" else "-"
        for p in procs:
            for r in range(rounds if p == 0 else 1):
                for b in range(BATCHES):
                    block = r * BATCHES + b
                    tpr, ipr = (10.0, 50.0) if method == "treewalker" else xgboost(p)
                    for g in (2 * b, 2 * b + 1):
                        samples.append((cell, method, variant, "serving", p, block, b, r, g, tpr))
                    hw.append((cell, method, variant, "serving", p, block, int(ipr * 2), "ok"))
    d = run / "cells" / cell.replace("/", "_")
    d.mkdir(parents=True)
    pl.DataFrame(
        samples,
        schema=[
            "cell",
            "method",
            "variant",
            "mode",
            "process",
            "block",
            "batch",
            "repetition",
            "group",
            "ticks",
        ],
        orient="row",
    ).with_columns(
        pl.col("ticks").cast(pl.UInt64),
        rows=pl.lit(1, pl.UInt32),
        position=pl.lit(0, pl.UInt32),
    ).write_parquet(d / "samples.parquet")
    pl.DataFrame(
        hw,
        schema=["cell", "method", "variant", "mode", "process", "block", "instructions", "status"],
        orient="row",
    ).with_columns(pl.col("instructions").cast(pl.UInt64)).write_parquet(d / "hw.parquet")
    pl.DataFrame({"cell": [cell] * 8, "group": range(8), "entity": range(8)}).write_parquet(
        d / "groups.parquet"
    )
    pl.DataFrame({"cell": [cell] * 8, "group": range(8)}).write_parquet(d / "counters.parquet")
    stop = {"rounds": rounds, "blocks_per_round": BATCHES, "reason": "precision"}
    procs = [{"process": p, "mode": "serving", "status": "ok"} for p in range(CHILDREN + 1)]
    fm.write_json(
        d / "manifest.json",
        {
            "id": cell,
            "groups": 8,
            "rows": 8,
            "timed_groups": 8,
            "excluded": [],
            "validation": {"oracle": {"status": "pass", "rows": 8}},
            "stop": {"serving": {**stop, "probes": [], "conversion": []}},
            "xgboost_processes": {"count": CHILDREN, "processes": procs},
            "seconds": 1.0,
            "model": {"trees": 50},
            "cell": {"max_depth": 4},
        },
    )


def write_run(run: Path, rounds: int = ROUNDS) -> None:
    def record(index, role):
        return {
            "index": index,
            "role": role,
            "cells_done": 0,
            "unix_time": 0,
            "status": "ok",
            "totals": {"treewalker/all-on/serving": [100 if role == "measure" else 115, 10]},
        }

    fm.write_json(
        run / "run.json",
        {
            "timer": {
                "calibration": {
                    "hz": 1e9,
                    "counter": "tsc",
                    "source": "cpuid",
                    "overhead": {"p50": 20},
                }
            },
            "host": {"os": "linux", "arch": "x86_64", "cpu": "fake"},
            "run": {"sentinel_drift_pct": 3.0},
            "sentinel": {
                "cell": "s",
                "every": 25,
                "records": [record(0, "warm-up"), record(1, "measure")],
            },
        },
    )
    # Two modes: child 1 runs the fast mode (more instructions, 90 ticks a row).
    write_cell(
        run, "c/two/xgboost/panel", lambda p: (90.0, 110.0) if p == 1 else (100.0, 100.0), rounds
    )
    # One mode; the children read 2% slow.
    write_cell(run, "c/one/xgboost/panel", lambda p: (102.0 if p else 100.0, 100.0), rounds)
    pack.pack_run(run)


def test_summarize_and_the_readout_on_a_run_with_counters_and_children(tmp_path):
    write_run(tmp_path)
    text = "\n".join(analysis.summarize(tmp_path, n_boot=20))
    # Two modes: child 1's faster cluster in every batch, against TreeWalker.
    assert "two modes in 4 of 4 batches (100%); faster 0.0900 us/row" in text
    assert "[faster mode, all processes]" in text and "    0.0900    9.000" in text
    # No second mode visible: process 0's time, with the mean over processes beside it.
    assert "no second mode visible in 4 batches; process 0's time 0.1000 us/row" in text
    assert "expected over processes 0.1015 us/row (the mean of 4)" in text
    assert "[no second mode visible]" in text and "    0.1000   10.000" in text
    assert "first-load effect treewalker/all-on/serving +15.0%" in text

    lines = readout.readout(tmp_path)
    text = "\n".join(lines)
    assert "round-0 rule at 1%: holds" in text
    assert "xgboost_native           1.0000 [1.0000, 1.0000] over 2; 0 off" in text
    assert "c/two/xgboost/panel serving: two modes, mixed 4 of 4 (100%)" in text
    assert "0:1/5 1:1/5 2:1/5 3:1/5" in text
    # Child 1 runs the mode process 0 never does; the others match it.
    two = "c/two/xgboost/panel serving process"
    assert f"{two} 1: - (no batch where process 0 shares this child's mode)" in text
    assert f"{two} 2: 1.000 of process 0's later rounds, on 4 of its 4 blocks" in text
    assert "c/one/xgboost/panel serving process 3: 1.020 of process 0's later rounds" in text
    assert "measured rate:" in text


def test_a_one_round_run_has_no_round_zero_evidence(tmp_path):
    # rev-gpt's N18: process 0 times one round only, so no method has a later
    # period to compare round 0 with.
    write_run(tmp_path, rounds=1)
    text = "\n".join(readout.readout(tmp_path))
    assert "round-0 rule at 1%: not decided" in text
    assert "no evidence: treewalker, xgboost_native" in text
