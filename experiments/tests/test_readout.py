"""The validation readout's checks on known samples."""

import polars as pl

from treewalker_exp import analysis, readout


def samples(rows):
    cols = ["cell", "method", "variant", "mode", "process", "batch", "repetition", "ticks", "rows"]
    return pl.DataFrame(rows, schema=cols, orient="row")


def test_round_zero_against_the_later_rounds():
    rows = [
        ("c", "m", "-", "serving", 0, b, r, 110 if r == 0 else 100, 1)
        for r in range(3)
        for b in range(2)
    ]
    rows += [("c", "m", "-", "serving", 1, 0, 0, 999, 1)]  # a child: not counted
    r = readout.round_zero(samples(rows))
    assert r["ratio"].to_list() == [1.1]


def xgboost(rows):
    """XGBoost samples and blocks from (process, repetition, batch, ticks, instructions)
    per one-row block, two batches."""
    df = samples(
        [("c", "xgboost_native", "-", "serving", p, b, r, t, 1) for p, r, b, t, _ in rows]
    ).with_columns(block=pl.col("repetition") * 2 + pl.col("batch"))
    blocks = df.select("process", "block", "batch", "repetition", "ticks", "rows").with_columns(
        instructions=pl.Series([float(x[4]) for x in rows])
    )
    return df, analysis.xgboost_faster_mode(blocks)


def test_children_against_process_zero_on_the_same_batch_and_cluster():
    # rev-gpt's case: process 0 costs 200 in round 0 and 100 later; the child
    # costs 200 in its one round: against the later rounds, 2.
    rows = [(0, r, b, 200 if r == 0 else 100, 1000) for r in range(3) for b in range(2)]
    rows += [(1, 0, b, 200, 1000) for b in range(2)]
    df, r = xgboost(rows)
    assert r["outcome"] == "no second mode visible"
    ch = readout.children(df, r["labels"])
    assert ch["ratio"].to_list() == [2.0] and ch["matched"].to_list() == [2]
    # Process 0 in the slow mode (1,000 instructions a row) in round 0, fast
    # (1,100) later; child 1 fast, child 2 slow. Child 2 shares no batch and
    # cluster with process 0's later rounds.
    rows = [
        (0, r, b, 120 if r == 0 else 100, 1000 if r == 0 else 1100)
        for r in range(3)
        for b in range(2)
    ]
    rows += [(1, 0, b, 101, 1100) for b in range(2)]
    rows += [(2, 0, b, 125, 1000) for b in range(2)]
    df, r = xgboost(rows)
    assert r["outcome"] == "two modes"
    ch = readout.children(df, r["labels"])
    assert ch["ratio"].to_list() == [1.01, None] and ch["matched"].to_list() == [2, 0]
    # Process 0 switches inside a batch, so its round 0 is left out of the rule.
    assert readout.process_zero_switches(r["labels"])
    lines = readout.xgboost_lines("c", "serving", r)
    assert "two modes, mixed 2 of 2 (100%)" in lines[0]
    assert lines[1].endswith("0:3/2 1:3/2")


def test_the_position_profile_is_net_of_content():
    rows = []
    for r in range(4):
        for b in range(4):
            content = 1 + b  # batch b costs 1 + b for every method
            for m, var, base in (("treewalker", "all-on", 10), ("lleaves", "-", 20)):
                t = base * content
                if m == "lleaves" and r == 0:
                    t *= 1.5  # round 0 slow for lleaves only
                if m == "lleaves" and b == 0:
                    t *= 1.1  # its first timed batch 10% slow, net of content
                rows.append(("c", m, var, "serving", 0, b, r, t, 1))
    pos = readout.position_profile(samples(rows)).filter(pl.col("method") == "lleaves")
    assert abs(pos["first"][0] / pos["warm"][0] - 1.1) < 0.06


def test_the_round_zero_rule_names_the_methods_over_it_and_their_side():
    (line,) = readout.round_zero_rule({"a": [1.0, 1.004], "b": [1.03, 1.02], "c": [0.97]})
    assert "FIRES" in line and "b 1.0250 (round 0 slower)" in line
    assert "c 0.9700 (round 0 faster)" in line and "a 1.00" not in line
    assert "holds" in readout.round_zero_rule({"a": [1.0]})[0]
    # Exactly 1% does not exceed it; just above does.
    assert "holds" in readout.round_zero_rule({"a": [1.01], "b": [0.99]})[0]
    assert "FIRES" in readout.round_zero_rule({"a": [1.0101]})[0]
    # A method with no evidence is never counted as holding.
    (line,) = readout.round_zero_rule({"a": [1.0], "xgboost_native": []})
    assert "not decided" in line and "no evidence: xgboost_native" in line
    (line,) = readout.round_zero_rule({})
    assert "not decided" in line and "no evidence: every method" in line
