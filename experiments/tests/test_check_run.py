"""check-run: the validity gates and the warnings before reading any result."""

import copy

import pytest

from treewalker_exp import check_run as cr
from treewalker_exp import formats as fm
from treewalker_exp import paths


def method(name, variant="-", serving="predict_group", one_row=None, validated=None, prec="F64"):
    return {
        "method": name,
        "variant": variant,
        "interfaces": {"serving": serving},
        "interface_one_row": one_row or serving,
        "validated": validated or [serving],
        "precision": prec,
    }


def check(name, iface, variant="-", rows=64, passed=True, mode="serving", defect=None):
    c = {
        "method": name,
        "variant": variant,
        "mode": mode,
        "interface": iface,
        "rows": rows,
        "passed": passed,
    }
    return c | ({"defect": defect} if defect is not None else {})


DEFECT = {
    "name": "tl2cgen-f64-missing-alias: Entry union int missing vs double fvalue",
    "affected_rows": 2,
    "affected_differ": 1,
    "affected_max_abs": 0.1,
}


def cell(**extra):
    m = {
        "rows": 64,
        "required_methods": ["treewalker"],
        "stop": {"serving": {"rounds": 2}},
        "excluded": [],
        "methods": [
            method("treewalker", "all-on"),
            method("tl2cgen", serving="predict", validated=["Batch", "predict loop"]),
        ],
        "validation": {
            "checks": [
                check("treewalker", "predict_group", "all-on"),
                check("tl2cgen", "Batch", defect=DEFECT),
                check("tl2cgen", "predict loop", defect=DEFECT),
            ],
            "oracle": {"status": "pass", "rows": 16, "exact_sums": False},
        },
        "exact_sums": {
            "setting": False,
            "forests": {
                "exact_sums_off": {"exact_sums": False},
                "chunked128": {"exact_sums": False},
            },
        },
    }
    return m | extra


def test_a_cell_validated_on_its_timed_rows_passes():
    # tl2cgen times its one-row predict, which its loop candidate validated.
    assert cr.validation_failures(cell()) == []


def test_validation_gaps_are_named():
    m = cell()
    m["validation"]["checks"][0]["rows"] = 32
    m["validation"]["checks"][2]["passed"] = False
    del m["validation"]["checks"][2]["defect"]
    got = "; ".join(cr.validation_failures(m))
    assert "treewalker/all-on serving predict_group: not validated on the 64 timed rows" in got
    assert "tl2cgen/- serving predict: a check of the timed interface failed" in got
    assert "no record of the rows attributed to the defect" in got
    # A sampled cell is validated on the sample's rows.
    assert cr.validation_failures(cell(sample={"rows": 32})) != []
    # An excluded method's failed checks do not count.
    m = cell()
    m["validation"]["checks"].append(check("quickscorer", "score", passed=False))
    m["excluded"] = [{"method": "quickscorer", "variant": "-", "reason": "outputs differ"}]
    assert cr.validation_failures(m) == []


def test_every_forest_must_carry_the_runs_exact_sums():
    assert cr.exact_sums_failures(cell(), False) == []
    m = cell()
    m["exact_sums"]["forests"]["chunked128"]["exact_sums"] = True
    assert cr.exact_sums_failures(m, False) == ["forest chunked128 reports exact_sums = True"]
    assert "the run True" in cr.exact_sums_failures(cell(), True)[0]
    on = cell(exact_sums={"setting": True, "forests": {"": {"exact_sums": False}}})
    on["validation"]["oracle"]["exact_sums"] = False
    # On: every forest as production reports, here off (leaves without exact sums).
    assert cr.exact_sums_failures(on, True) == []
    assert cr.exact_sums_failures(cell(exact_sums=None), False) == ["no exact_sums record"]


def write_tables(d, cells, schema=6, skip=()):
    """The tables a packed run holds: 4 one-row timed groups per cell, two rounds of
    serving samples per timed method, except the (cell, method) pairs in ``skip``."""
    import polars as pl

    samples, groups = [], []
    for cid, m in cells.items():
        groups += [(cid, g, g, 16, True) for g in range(4)]
        for meth in m.get("methods") or []:
            if (cid, meth["method"]) in skip:
                continue
            for r in range(2):
                samples += [
                    (cid, meth["method"], meth["variant"], "serving", 0, r, r, g, 1, 16)
                    for g in range(4)
                ]
    pl.DataFrame(
        samples,
        schema=[
            *("cell", "method", "variant", "mode", "process", "repetition", "block"),
            *("group", "n_groups", "rows"),
        ],
        orient="row",
    ).write_parquet(d / "samples.parquet")
    pl.DataFrame(
        groups, schema=["cell", "group", "entity", "rows", "timed"], orient="row"
    ).write_parquet(d / "groups.parquet")
    for t in ("counters", "hw", *(("stages",) if schema >= 6 else ())):
        pl.DataFrame({"cell": ["x"]}).write_parquet(d / f"{t}.parquet")


def write_run(
    tmp_path, cells, run=None, failed=(), schema=6, sentinel=None, suite="camera-ready", skip=()
):
    d = tmp_path / "camera-ready-x86_64"
    d.mkdir()
    fm.write_json(d / "cells.json", cells)
    info = {"suite": suite, "output_schema": schema, "run": run or {"exact_sums": False}}
    info |= {"sentinel": sentinel} if sentinel else {}
    fm.write_json(d / "run.json", info | {"manifest": {"cells": len(cells)}})
    for name in failed:
        (d / "failed").mkdir(exist_ok=True)
        (d / "failed" / name).write_text("{}")
    write_tables(d, cells, schema, skip)
    return d


def test_gates_pass_and_fail(tmp_path):
    ok_run = write_run(tmp_path, {"a/c": cell()})
    lines, ok = cr.gates(ok_run)
    assert ok, lines
    assert any("a/c: 2 rows, 1 differ" in line for line in lines)
    bad = cell()
    bad["exact_sums"]["forests"]["exact_sums_off"]["exact_sums"] = True
    other = tmp_path / "x"
    other.mkdir()
    lines, ok = cr.gates(write_run(other, {"a/c": bad}, failed=["b_c.json"]))
    assert not ok
    text = "\n".join(lines)
    assert "FAIL failed cells: 1 unresolved" in text and "FAIL exact sums (False): 1 cells" in text
    # A runner older than schema 6, without the setting: said, not gated, outside
    # the paper's suites.
    old = tmp_path / "old"
    old.mkdir()
    lines, ok = cr.gates(
        write_run(old, {"a/c": cell()}, run={"min_rounds": 3}, schema=5, suite="factorial")
    )
    assert ok and any("not recorded" in line for line in lines)


def gate(tmp_path, name, cells, **kw):
    d = tmp_path / name
    d.mkdir()
    lines, ok = cr.gates(write_run(d, cells, **kw), kw.get("expect"))
    return "\n".join(lines), ok


def test_gates_fail_closed(tmp_path):
    # No timed method, a failed empty oracle, no samples.
    empty = cell(methods=[])
    empty["validation"]["oracle"] = {"status": "fail", "rows": 0, "exact_sums": False}
    text, ok = gate(tmp_path, "empty", {"a/c": empty})
    assert not ok and "no timed method" in text and "stage oracle 'fail'" in text
    assert "required method treewalker not timed" in text
    # A timed method without samples.
    text, ok = gate(tmp_path, "nosamples", {"a/c": cell()}, skip={("a/c", "tl2cgen")})
    assert not ok and "tl2cgen/-/serving: no samples" in text
    # A failed sentinel attempt is a failed cell.
    records = [{"index": 0, "role": "measure", "status": "failed: the stage oracle did not pass"}]
    text, ok = gate(tmp_path, "sentinel", {"a/c": cell()}, sentinel={"records": records})
    assert not ok and "1 sentinel attempts not ok" in text
    # Schema 6 needs the stages table and the exact_sums setting.
    d = tmp_path / "stages"
    d.mkdir()
    run = write_run(d, {"a/c": cell()})
    (run / "stages.parquet").unlink()
    lines, ok = cr.gates(run)
    assert not ok and "tables missing: stages" in "\n".join(lines)
    text, ok = gate(tmp_path, "nosetting", {"a/c": cell()}, run={"min_rounds": 4}, suite="x")
    assert not ok and "records no setting" in text
    # The paper's suites need exact sums off, however consistent the run is.
    on = cell(exact_sums={"setting": True, "forests": {"": {"exact_sums": True}}})
    on["validation"]["oracle"]["exact_sums"] = True
    text, ok = gate(tmp_path, "on", {"a/c": on}, run={"exact_sums": True})
    assert not ok and "needs False" in text
    text, ok = gate(tmp_path, "on-elsewhere", {"a/c": on}, run={"exact_sums": True}, suite="x")
    assert ok, text


def test_an_unchosen_failing_candidate_is_a_diagnostic_not_a_failure():
    m = cell()
    m["methods"].append(method("lightgbm_native", serving="fast", validated=["fast"]))
    m["validation"]["checks"] += [
        check("lightgbm_native", "fast"),
        check("lightgbm_native", "slow", passed=False),
    ]
    assert cr.validation_failures(m) == []
    assert cr.diagnostics(m) == ["lightgbm_native/- serving slow (not chosen)"]
    # A failed check of the interface it timed still fails.
    m["validation"]["checks"].append(check("lightgbm_native", "fast", passed=False))
    assert cr.validation_failures(m) == [
        "lightgbm_native/- serving fast: a check of the timed interface failed"
    ]


def sentinel(drifts):
    """run.json's sentinel: a reference measure, then one measure per drift."""
    records = [
        {
            "index": i,
            "role": "measure",
            "status": "ok",
            "cells_done": i,
            "unix_time": i,
            "totals": {"treewalker/all-on/serving": [round(1000 * (1 + d / 100)), 10]},
        }
        for i, d in enumerate([0.0, *drifts])
    ]
    return {"sentinel": {"records": records}}


@pytest.mark.parametrize(
    ("drifts", "warn"),
    [
        ([0.5, -1.0, 1.5], False),
        ([2.5, 0.0, 0.5, 2.5, 0.0], False),  # isolated sentinels above 2%, median within
        ([0.0, 2.5, 2.6, 0.0], True),  # two consecutive above 2%
        ([3.0, 3.0, -3.0, 2.1, 0.0], True),  # median above 2%
    ],
)
def test_sentinel_warnings(drifts, warn):
    text = "\n".join(cr.sentinel_warnings(sentinel(drifts)))
    assert ("WARN" in text) == warn, text


def test_released_e1_reads_the_tagged_csv():
    released = cr.released_e1(paths.find_repo())
    if released is None:
        pytest.skip("the neurips2026 tag is not available")
    assert len(released) == 16 and released[(1, 4)] == pytest.approx(31.836)


def test_e1_warns_beyond_ten_percent(tmp_path, monkeypatch):
    run = tmp_path / "scenario-v1-x86_64"
    run.mkdir()
    cells = {
        "scenario_credit/lightgbm/whatif-v1-k1-G4": {},
        "scenario_credit/lightgbm/whatif-v1-k8-G4": {},
    }
    fm.write_json(run / "cells.json", cells)
    fm.write_json(run / "run.json", {"timer": {"calibration": {"hz": 1e6}}})
    lat = {
        ("scenario_credit/lightgbm/whatif-v1-k1-G4", "treewalker_fullwalk", "-"): 105.0,
        ("scenario_credit/lightgbm/whatif-v1-k8-G4", "treewalker_fullwalk", "-"): 120.0,
    }
    monkeypatch.setattr(cr, "latencies", lambda run, cells: copy.deepcopy(lat))
    lines = cr.e1_warnings(run, {(1, 4): 100.0, (8, 4): 100.0})
    assert len(lines) == 1 and "k8-G4" in lines[0] and "+20.0%" in lines[0]


def test_nothing_measured_is_never_ok(tmp_path, monkeypatch):
    # Every sentinel attempt failed.
    records = [
        {"index": i, "role": role, "status": "failed: no native library", "totals": {}}
        | {"cells_done": 0, "unix_time": 0}
        for i, role in enumerate(("warm-up", "measure"))
    ]
    text = "\n".join(cr.sentinel_warnings({"sentinel": {"records": records}}))
    assert "ok" not in text and "no measured drift (2 attempts, 2 failed)" in text
    assert "WARN sentinel attempt 1 (measure): failed: no native library" in text
    # E1 with no cell to compare.
    run = tmp_path / "scenario-v1-x86_64"
    run.mkdir()
    fm.write_json(run / "cells.json", {"scenario_credit/lightgbm/whatif-v1-k1-G4": {}})
    fm.write_json(run / "run.json", {"timer": {"calibration": {"hz": 1e6}}})
    monkeypatch.setattr(cr, "latencies", lambda run, cells: {})
    assert cr.e1_warnings(run, {(1, 4): 100.0}) == [
        "  WARN E1: no cell compared (1 scenario cells in the run)"
    ]


def test_an_unused_f64_tl2cgen_call_error_needs_no_defect_record():
    # The runner records a failed call without a defect: it had no outputs.
    m = cell()
    m["validation"]["checks"] = [c for c in m["validation"]["checks"] if c["interface"] != "Batch"]
    m["validation"]["checks"].append(
        check("tl2cgen", "Batch", passed=False, mode="-") | {"against": "call", "error": "x"}
    )
    assert cr.validation_failures(m) == []
    assert cr.diagnostics(m) == ["tl2cgen/- - Batch (not chosen)"]
    # The timed interface's compared checks still need it.
    del m["validation"]["checks"][1]["defect"]
    assert any("no record of the rows" in f for f in cr.validation_failures(m))


def rewrite_samples(run, edit):
    import polars as pl

    s = pl.read_parquet(run / "samples.parquet")
    edit(s).write_parquet(run / "samples.parquet")


def coverage_case(tmp_path, name, edit, cells=None):
    d = tmp_path / name
    d.mkdir()
    run = write_run(d, cells or {"a/c": cell()})
    rewrite_samples(run, edit)
    return cr.sample_failures(run, fm.read_json(run / "cells.json")), cr.gates(run)[1]


def test_sample_coverage_is_checked_by_round_and_group_id(tmp_path):
    import polars as pl

    tw = pl.col("method") == "treewalker"
    got, ok = coverage_case(tmp_path, "intact", lambda s: s)
    assert got == {} and ok
    # A whole round of TreeWalker deleted, the manifest still recording two.
    got, ok = coverage_case(
        tmp_path, "round", lambda s: s.filter(~(tw & (pl.col("repetition") == 1)))
    )
    assert not ok and "treewalker/all-on/serving: 1 rounds sampled, 2 recorded" in got["a/c"]
    # Every TreeWalker group ID replaced by 0, the sample count unchanged.
    got, ok = coverage_case(
        tmp_path,
        "ids",
        lambda s: s.with_columns(group=pl.when(tw).then(0).otherwise(pl.col("group"))),
    )
    assert not ok and any("4 samples of 1 groups" in f for f in got["a/c"])
    # A sample whose rows differ from its group's.
    got, ok = coverage_case(
        tmp_path,
        "rows",
        lambda s: s.with_columns(
            rows=pl.when(tw & (pl.col("group") == 0)).then(1).otherwise(pl.col("rows"))
        ),
    )
    assert not ok and any("rows differ" in f for f in got["a/c"])


def test_samples_of_an_excluded_method_fail_and_never_reach_results(tmp_path):
    import polars as pl

    from treewalker_exp import analysis

    m = cell()
    m["excluded"] = [{"method": "quickscorer", "variant": "-", "reason": "outputs differ"}]

    def add_quickscorer(s):
        return pl.concat(
            [
                s,
                s.filter(pl.col("method") == "treewalker").with_columns(
                    method=pl.lit("quickscorer"), variant=pl.lit("-")
                ),
            ]
        )

    got, ok = coverage_case(tmp_path, "qs", add_quickscorer, {"a/c": m})
    assert not ok and got["a/c"] == [
        "quickscorer/-/serving: samples of a method the cell does not time"
    ]
    s = add_quickscorer(
        pl.DataFrame({"cell": ["a/c"], "method": ["treewalker"], "variant": ["all-on"]})
    )
    with pytest.raises(ValueError, match="does not time"):
        analysis.refuse_untimed(s, {"a/c": m})


def test_a_successful_sentinel_needs_its_packed_evidence(tmp_path):
    records = [{"index": 0, "role": "measure", "status": "ok", "totals": {}}]
    text, ok = gate(tmp_path, "nosentinel", {"a/c": cell()}, sentinel={"records": records})
    assert not ok and "sentinel 000: recorded ok, no packed manifest" in text


def test_batch_rounds_cover_the_timed_groups_count_and_rows(tmp_path):
    import polars as pl

    m = cell(stop={"batch": {"rounds": 2, "blocks_per_round": 1}})
    for meth in m["methods"]:
        meth["interfaces"] = {"batch": meth["interfaces"]["serving"]}
    for c in m["validation"]["checks"]:
        c["mode"] = "batch"

    def to_batch(s):
        # One call of all 4 groups per round.
        first = s.filter(pl.col("group") == 0)
        return first.with_columns(mode=pl.lit("batch"), n_groups=pl.lit(4), rows=pl.lit(64))

    got, _ok = coverage_case(tmp_path, "batch", to_batch, {"a/c": m})
    assert got == {}, got
    got, _ = coverage_case(
        tmp_path, "batch-short", lambda s: to_batch(s).with_columns(n_groups=pl.lit(3)), {"a/c": m}
    )
    assert any("calls cover 3 groups" in f for f in got["a/c"])


def invocations(run, *attempts):
    """Record one invocation per ``(done, failed)`` pair of cell-ID lists."""
    info = fm.read_json(run / "run.json")
    info["invocations"] = [
        {"cells_done": [{"id": c} for c in done], "cells_failed": [{"id": c} for c in failed]}
        for done, failed in attempts
    ]
    fm.write_json(run / "run.json", info)


def test_a_failure_a_later_attempt_resolved_is_listed_not_gated(tmp_path):
    d = tmp_path / "resolved"
    d.mkdir()
    run = write_run(d, {"a/c": cell()})
    (run / "failed").mkdir()
    # The cell failed in invocation 0 and finished in invocation 1.
    invocations(run, ([], ["a/c"]), (["a/c"], []))
    fm.write_json(run / "failed" / "a_2fc.json", {"id": "a/c", "invocation": 0, "reason": "x"})
    lines, ok = cr.gates(run)
    text = "\n".join(lines)
    assert ok, text
    assert "resolved: 1 failed cells finished on a later attempt (a_2fc.json)" in text
    # A failed cell no later attempt finished still gates.
    fm.write_json(run / "failed" / "b_2fc.json", {"id": "b/c", "invocation": 0, "reason": "x"})
    lines, ok = cr.gates(run)
    assert not ok and "FAIL failed cells: 1 unresolved (b_2fc.json)" in "\n".join(lines)


def test_a_finished_result_older_than_the_failure_does_not_resolve_it(tmp_path):
    """A rerun of a changed cell that fails leaves the earlier result in cells.json;
    only an attempt after the failure resolves it (rnewman, PR 15)."""
    d = tmp_path / "replaced"
    d.mkdir()
    run = write_run(d, {"a/c": cell()})
    (run / "failed").mkdir()
    fm.write_json(run / "failed" / "a_2fc.json", {"id": "a/c", "invocation": 1, "reason": "x"})
    invocations(run, (["a/c"], []), ([], ["a/c"]))
    assert cr.completeness(run) == ["failed cell a_2fc.json"]
    lines, ok = cr.gates(run)
    assert not ok and "FAIL failed cells: 1 unresolved (a_2fc.json)" in "\n".join(lines)
    # An invocation cut short records nothing: a failure in it, at index 1 after
    # one listed invocation, then the cell finishing in the next one, at index 1.
    invocations(run, (["a/c"], []), (["a/c"], []))
    assert cr.completeness(run) == []
    # A failure in an invocation cut short after the cell last finished stays open.
    invocations(run, (["a/c"], []))
    assert cr.completeness(run) == ["failed cell a_2fc.json"]
    # A record without its invocation counts from the last one that listed the failure.
    fm.write_json(run / "failed" / "a_2fc.json", {"id": "a/c", "reason": "x"})
    invocations(run, ([], ["a/c"]), (["a/c"], []))
    assert cr.completeness(run) == []
    invocations(run, (["a/c"], []), ([], ["a/c"]))
    assert cr.completeness(run) == ["failed cell a_2fc.json"]
    invocations(run, (["a/c"], []))
    assert cr.completeness(run) == ["failed cell a_2fc.json"]


def test_a_failure_only_an_invocation_lists_needs_a_later_finish(tmp_path):
    # A cell that failed before it was timed (a hash mismatch) writes no failed/.
    d = tmp_path / "listed"
    d.mkdir()
    run = write_run(d, {"a/c": cell()})
    invocations(run, (["a/c"], []), ([], ["a/c"]))
    assert cr.completeness(run) == ["failed cell a/c (invocation 1)"]
    invocations(run, ([], ["a/c"]), (["a/c"], []))
    assert cr.completeness(run) == []


def test_a_sentinel_attempt_retried_with_success_is_resolved():
    # The runner numbers every attempt anew; a retry is a later attempt of the role.
    records = [
        {"index": 0, "role": "warm-up", "status": "failed: x"},
        {"index": 1, "role": "measure", "status": "failed: y"},
        {"index": 2, "role": "measure", "status": "ok"},
        {"index": 3, "role": "measure", "status": "failed: z"},
    ]
    open_, retried = cr.open_sentinel_failures(records)
    assert [r["index"] for r in open_] == [0, 3]  # no later ok warm-up; nothing after 3
    assert [r["index"] for r in retried] == [1]


def test_completeness_is_the_no_failed_cell_rule(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    run = write_run(d, {"a/c": cell()})
    assert cr.completeness(run) == []
    (run / "failed").mkdir()
    invocations(run, ([], ["a/c"]), (["a/c"], []))
    fm.write_json(run / "failed" / "a_2fc.json", {"id": "a/c", "invocation": 0})  # resolved
    assert cr.completeness(run) == []
    fm.write_json(run / "failed" / "b_2fc.json", {"id": "b/c", "invocation": 0})  # unresolved
    assert cr.completeness(run) == ["failed cell b_2fc.json"]
    (run / "failed" / "b_2fc.json").unlink()
    info = fm.read_json(run / "run.json")
    info["manifest"]["cells"] = 2
    info["sentinel"] = {"records": [{"index": 4, "role": "measure", "status": "failed: q"}]}
    fm.write_json(run / "run.json", info)
    assert cr.completeness(run) == [
        "1 of 2 cells missing",
        "sentinel attempt 4 (measure) failed: q",
    ]
    assert cr.completeness(tmp_path / "absent") == ["absent, not packed"]
