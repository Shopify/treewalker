"""Re-timing: cells with wide intervals, their manifests, and the longer attempt."""

from pathlib import Path

import polars as pl
import pytest

from treewalker_exp import formats as fm
from treewalker_exp import results, retime


def rows(cell, rounds, half_pct, method="treewalker_fullwalk", arch="intel", mode="serving"):
    speedup = 2.0
    return {
        "arch": arch,
        "cell": cell,
        "mode": mode,
        "method": method,
        "variant": "-",
        "speedup": speedup,
        "lo": speedup * (1 - half_pct / 100),
        "hi": speedup * (1 + half_pct / 100),
        "rounds": rounds,
    }


def test_flagged_lists_each_cell_and_mode_above_the_threshold_once():
    df = pl.DataFrame(
        [
            rows("a", 4, 3.0),
            rows("a", 4, 5.0, method="lightgbm_native"),
            rows("b", 6, 1.9),
            rows("c", 4, 2.5),
            # The reference against itself has a degenerate interval: never flagged.
            {**rows("d", 4, 9.0, method="treewalker"), "variant": "all-on"},
        ]
    )
    f = retime.flagged(df)
    assert f["cell"].to_list() == ["a", "c"]
    a = f.row(0, named=True)
    assert (a["method"], a["comparisons"]) == ("lightgbm_native", 2)
    assert a["half_pct"] == pytest.approx(5.0)
    assert retime.flagged(df, threshold_pct=1.0).height == 3


def test_manifests_double_each_cells_rounds_one_manifest_per_round_count():
    manifest = {
        "suite": "camera-ready",
        "run": {"min_rounds": 4, "max_rounds": 10, "seed": 42},
        "counts": {"ready": 3},
        "cells": [{"id": c, "status": "ready"} for c in ("a", "b", "c")],
    }
    f = retime.flagged(
        pl.DataFrame([rows("a", 4, 3.0), rows("b", 6, 3.0), rows("b", 5, 3.0, mode="batch")])
    )
    docs = retime.manifests(manifest, f, "camera-ready", 2.0)
    assert sorted(docs) == ["camera-ready-retime-r12", "camera-ready-retime-r8"]
    r8 = docs["camera-ready-retime-r8"]
    assert [c["id"] for c in r8["cells"]] == ["a"]
    assert (r8["run"]["min_rounds"], r8["run"]["max_rounds"], r8["run"]["seed"]) == (8, 8, 42)
    assert r8["retime"] == {"of": "camera-ready", "threshold_pct": 2.0, "rounds": 4, "cells": ["a"]}
    assert docs["camera-ready-retime-r12"]["counts"] == {"ready": 1}
    assert manifest["run"]["min_rounds"] == 4  # the original is untouched
    with pytest.raises(ValueError, match="not in the manifest"):
        retime.manifests({**manifest, "cells": []}, f, "camera-ready", 2.0)


def test_the_longer_attempt_is_reported_per_cell_and_mode():
    first = pl.DataFrame(
        [
            {**rows("a", 4, 3.0), "speedup": 2.0, "attempt": "x-x86_64"},
            {**rows("a", 4, 3.0, mode="batch"), "speedup": 2.5, "attempt": "x-x86_64"},
            {**rows("b", 4, 1.0), "speedup": 3.0, "attempt": "x-x86_64"},
        ]
    )
    second = pl.DataFrame([{**rows("a", 8, 1.0), "speedup": 2.2, "attempt": "x-retime-r8-x86_64"}])
    got = results.longest([first, second]).sort("cell", "mode")
    assert got.select("cell", "mode", "speedup", "attempt", "attempts").rows() == [
        ("a", "batch", 2.5, "x-x86_64", 1),
        ("a", "serving", 2.2, "x-retime-r8-x86_64", 2),
        ("b", "serving", 3.0, "x-x86_64", 1),
    ]


def test_attempts_finds_the_runs_retimes_only(tmp_path):
    for name in (
        "camera-ready-x86_64",
        "camera-ready-retime-r8-x86_64",
        "camera-ready-retime-r12-x86_64",
        "camera-ready-ablation-1-retime-r8-x86_64",
        "camera-ready-retime-r8-aarch64",
    ):
        (tmp_path / name).mkdir()
    names = [p.name for p in results.attempts(tmp_path, "camera-ready", "intel")]
    assert names == [
        "camera-ready-x86_64",
        "camera-ready-retime-r12-x86_64",
        "camera-ready-retime-r8-x86_64",
    ]


def test_paper_against_is_research_all_on_or_monotonics_precompute_off():
    from treewalker_exp.estimand import paper_against

    r = "treewalker_research"
    assert paper_against(r, "disable_unsplit") == (r, "all-on")
    assert paper_against(r, "|disable_tree_ordering") == (r, "all-on")
    assert paper_against(r, "disable_monotonic+disable_varying_precompute") == (
        r,
        "disable_varying_precompute",
    )
    assert paper_against(
        r, "disable_monotonic+disable_varying_precompute|disable_tree_ordering"
    ) == (
        r,
        "disable_varying_precompute|disable_tree_ordering",
    )
    assert paper_against(r, "all-on") is None
    assert paper_against("treewalker", "all-on") is None


def counterexample():
    """A balanced example: production and the
    research build agree within a round shock that moves the ablation's two sides
    in opposite directions, so the effect's interval is wide while every interval
    against production is narrow."""
    from treewalker_exp import estimand

    cid = "support/nt500_md8_h16/lightgbm/panel"
    rows = []
    for r in range(4):
        shock = 0.018 if r % 2 else -0.018
        for b in range(2):
            for g in range(4 * b, 4 * (b + 1)):
                for m, v, t in [
                    ("treewalker", "all-on", 1000),
                    ("treewalker_research", "all-on", 1000 * (1 + shock)),
                    ("treewalker_research", "disable_unsplit", 2000 * (1 - shock)),
                    ("treewalker_research", "disable_varying_precompute", 1500),
                    (
                        "treewalker_research",
                        "disable_monotonic+disable_varying_precompute",
                        1500 * (1 + 2 * shock),
                    ),
                ]:
                    rows.append((cid, m, v, "serving", r * 2 + b, b, g, r, round(t), 1))
    s = pl.DataFrame(
        rows,
        schema=[
            *("cell", "method", "variant", "mode", "block", "batch", "group", "repetition"),
            *("ticks", "n_groups"),
        ],
        orient="row",
    )
    g = pl.DataFrame(
        {"cell": [cid] * 8, "group": range(8), "entity": range(8), "rows": 16, "timed": True}
    )
    est = estimand.block_estimates(s, g, {cid: {}}, n_boot=1000)
    return pl.DataFrame(
        [
            dict(zip(["cell", "method", "variant", "mode"], k, strict=True))
            | {"arch": "intel", "rounds": 4, "speedup": v["ratio"]}
            | {x: v.get(x) for x in ("lo", "hi", "effect", "effect_lo", "effect_hi")}
            | {"effect_against": v.get("effect_against")}
            for k, v in est.items()
        ]
    )


def test_retime_tests_the_ablation_contrasts_from_paired_draws():
    df = counterexample()
    hw = retime.half_widths(df)
    speed = hw.filter(
        (pl.col("contrast") == "treewalker/all-on")
        & pl.col("variant").is_in(["all-on", "disable_unsplit"])
    )
    assert speed["half_pct"].max() < 2.0  # both sides against production: narrow
    unsplit = hw.filter(
        (pl.col("variant") == "disable_unsplit")
        & (pl.col("contrast") == "treewalker_research/all-on")
    ).row(0, named=True)
    assert unsplit["half_pct"] > 3.0
    mono = hw.filter(pl.col("variant") == "disable_monotonic+disable_varying_precompute")
    assert "treewalker_research/disable_varying_precompute" in mono["contrast"].to_list()
    f = retime.flagged(df.filter(~pl.col("variant").str.contains("precompute")))
    assert f.height == 1 and f.row(0, named=True)["contrast"] == "treewalker_research/all-on"


def run_record(xgb_sha, settings=None):
    return {
        "suite": "camera-ready",
        "manifest": {"artifacts_dir": "/a", "grids_sha256": "g", "source": {"commit": "c"}},
        "natives": {"xgboost": {"sha256": xgb_sha}, "lightgbm": {"sha256": "l"}},
        "tl2cgen_runtime": {"sha256": "t"},
        "run": {"min_rounds": 4, "max_rounds": 10, "exact_sums": False} | (settings or {}),
        "system_info": {"machine_type": "c4"},
        "host": {"arch": "x86_64"},
    }


def exec_manifest(xgb_sha, settings=None):
    return {
        "suite": "camera-ready",
        "artifacts_dir": "/a",
        "grids_sha256": "g",
        "source": {"commit": "c"},
        "native": {"xgboost": {"sha256": xgb_sha}, "lightgbm": {"sha256": "l"}},
        "tl2cgen_runtime": {"sha256": "t"},
        "run": {"min_rounds": 4, "max_rounds": 10, "exact_sums": False} | (settings or {}),
        "cells": [{"id": "a", "key": "k", "status": "ready"}],
    }


def test_a_manifest_must_be_the_one_the_machines_run_executed():
    cells = {"a": {"cell": {"key": "k"}}}
    assert retime.manifest_problems(exec_manifest("intel"), run_record("intel"), cells) == []
    # The other machine's manifest: its native libraries differ.
    got = retime.manifest_problems(exec_manifest("intel"), run_record("arm"), cells)
    assert got == ["native xgboost: manifest 'intel', run 'arm'"]
    # A non-default setting must match; rounds may differ.
    got = retime.manifest_problems(
        exec_manifest("x", {"exact_sums": True, "min_rounds": 8}), run_record("x"), cells
    )
    assert got == ["run setting exact_sums: manifest True, run False"]
    got = retime.manifest_problems(
        exec_manifest("x"), run_record("x"), {"a": {"cell": {"key": "z"}}}
    )
    assert got == ["cell a: the manifest's key differs from the run's"]


def test_plan_writes_one_machines_manifest_and_a_campaign_script(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "camera-ready-x86_64"
    run.mkdir(parents=True)
    fm.write_json(run / "run.json", run_record("intel"))
    fm.write_json(run / "cells.json", {"a": {"cell": {"key": "k"}}})
    fm.write_json(tmp_path / "m.json", exec_manifest("intel"))
    monkeypatch.setattr(
        results, "speedups_one", lambda run, arch: pl.DataFrame([rows("a", 4, 3.0)])
    )
    lines = retime.plan(
        tmp_path / "runs", "camera-ready", "intel", tmp_path / "m.json", tmp_path / "o"
    )
    doc = fm.read_json(tmp_path / "o" / "camera-ready-retime-r8-x86_64.json")
    assert doc["native"]["xgboost"]["sha256"] == "intel" and doc["run"]["min_rounds"] == 8
    script = (tmp_path / "o" / "camera-ready-retime-r8-x86_64.sh").read_text()
    assert "retime-run" in script and "--core 0 --require-pmu" in script
    assert "--run-id camera-ready-retime-r8-x86_64" in script
    assert any("1 (cell, mode) pairs" in line for line in lines)
    fm.write_json(tmp_path / "arm.json", exec_manifest("arm"))
    with pytest.raises(ValueError, match="native xgboost"):
        retime.plan(
            tmp_path / "runs", "camera-ready", "intel", tmp_path / "arm.json", tmp_path / "o"
        )


REAL = Path(__file__).parents[1] / "data" / "runs" / "factorial-x86_64"
REF = "support/nt500_md8_h16/lightgbm/panel"


def real_pair(tmp_path, edit_run=None, edit_cell=None):
    """The final run's Intel run.json and its SUPPORT reference cell as a parent and,
    edited, as an attempt with twice the rounds."""
    import copy

    info = fm.read_json(REAL / "run.json")
    cell = fm.read_json(REAL / "cells.json")[REF]
    out = []
    for name, edit_r, edit_c in (("parent", None, None), ("attempt", edit_run, edit_cell)):
        i, c = copy.deepcopy(info), copy.deepcopy(cell)
        if name == "attempt":
            i["run"] = {**i["run"], "min_rounds": 6, "max_rounds": 6}
        if edit_r:
            edit_r(i)
        if edit_c:
            edit_c(c["cell"])
        d = tmp_path / name
        d.mkdir()
        fm.write_json(d / "run.json", i)
        fm.write_json(d / "cells.json", {REF: c})
        out.append(d)
    return out


@pytest.mark.skipif(not (REAL / "run.json").exists(), reason="the final run's data (git lfs)")
@pytest.mark.parametrize(
    ("edit_run", "edit_cell", "expect"),
    [
        (lambda i: i["source"].update(commit="other"), None, "source.commit"),
        (lambda i: i.update(binary_sha256="other"), None, "binary_sha256"),
        (lambda i: i["host"].update(pinned_cpus=[0, 1]), None, "host.pinned_cpus"),
        (lambda i: i["host"].update(rustflags="-C target-cpu=generic"), None, "host.rustflags"),
        (lambda i: i["host"].update(rustc="rustc 1.98.0"), None, "host.rustc"),
        (lambda i: i["system_info"].update(image="other"), None, "system_info.image"),
        (lambda i: i["system_info"].update(zone="other"), None, "system_info.zone"),
        (lambda i: i["natives"]["xgboost"].update(sha256="other"), None, "natives.xgboost"),
        (lambda i: i["run"].update(exact_sums=False), None, "run setting exact_sums"),
        (None, lambda c: c["baselines"]["tl2cgen"].update(sha256="other"), "baselines differs"),
        (None, lambda c: c["baselines"]["lleaves"].update(sha256="other"), "baselines differs"),
        (None, lambda c: c["files"]["oracle"].update(sha256="other"), "files differs"),
    ],
)
def test_an_attempt_must_run_the_parents_program_and_protocol(
    tmp_path, edit_run, edit_cell, expect
):
    parent, attempt = real_pair(tmp_path, edit_run, edit_cell)
    problems = retime.attempt_problems(parent, attempt)
    assert any(expect in p for p in problems), problems


@pytest.mark.skipif(not (REAL / "run.json").exists(), reason="the final run's data (git lfs)")
def test_an_attempt_must_run_on_the_parents_vm(tmp_path):
    # The final run recorded no instance, so its re-times cannot be shown to be its own.
    parent, attempt = real_pair(tmp_path)
    problems = retime.attempt_problems(parent, attempt)
    assert problems and all("unrecorded" in p for p in problems)

    def on(parent_vm, attempt_vm):
        d = tmp_path / f"{parent_vm}-{attempt_vm}"
        d.mkdir()
        parent, attempt = real_pair(
            d, lambda i: i["system_info"].update(instance_id=attempt_vm, hostname=attempt_vm)
        )
        info = fm.read_json(parent / "run.json")
        info["system_info"] |= {"instance_id": parent_vm, "hostname": parent_vm}
        fm.write_json(parent / "run.json", info)
        return retime.attempt_problems(parent, attempt)

    assert on("vm", "vm") == []
    assert any("another VM" in p for p in on("vm", "other"))


def test_a_refused_attempt_leaves_the_original_reported(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for name in ("camera-ready-x86_64", "camera-ready-retime-r8-x86_64", "camera-ready-aarch64"):
        (runs / name).mkdir(parents=True)
        fm.write_json(runs / name / "cells.json", {"a": {}})

    def fake(run, arch, n_boot):
        r = 8 if "retime" in run.name else 4
        return pl.DataFrame([rows("a", r, 1.0, arch=arch) | {"speedup": 2.0 if r == 4 else 3.0}])

    monkeypatch.setattr(results, "_speedups", fake)
    monkeypatch.setattr(retime, "attempt_problems", lambda p, a: ["binary_sha256: differs"])
    df = results.speedups(runs, "camera-ready").filter(pl.col("arch") == "intel")
    assert df["speedup"].to_list() == [2.0] and df["attempt"].to_list() == ["camera-ready-x86_64"]
    assert df["retime"].to_list()[0].startswith("re-time refused: camera-ready-retime-r8-x86_64")
    monkeypatch.setattr(retime, "attempt_problems", lambda p, a: [])
    df = results.speedups(runs, "camera-ready").filter(pl.col("arch") == "intel")
    assert df["speedup"].to_list() == [3.0] and df["retime"].to_list() == [None]


def test_the_plan_writes_every_setting_the_parent_resolved(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "camera-ready-x86_64"
    run.mkdir(parents=True)
    fm.write_json(run / "run.json", run_record("intel") | {"output_schema": 6})
    fm.write_json(run / "cells.json", {"a": {"cell": {"key": "k"}}})
    m = exec_manifest("intel")
    del m["run"]["exact_sums"]  # omitted: the runner's default would turn it on
    fm.write_json(tmp_path / "m.json", m)
    monkeypatch.setattr(
        results, "speedups_one", lambda run, arch: pl.DataFrame([rows("a", 4, 3.0)])
    )
    retime.plan(tmp_path / "runs", "camera-ready", "intel", tmp_path / "m.json", tmp_path / "o")
    doc = fm.read_json(tmp_path / "o" / "camera-ready-retime-r8-x86_64.json")
    assert doc["run"]["exact_sums"] is False
    script = (tmp_path / "o" / "camera-ready-retime-r8-x86_64.sh").read_text()
    assert "--parent camera-ready-x86_64" in script and "--require-pmu" in script


def test_dispositions_are_rendered_and_counted(tmp_path, monkeypatch, capsys):
    from treewalker_exp import numbers, tables

    runs = tmp_path / "runs"
    stop = {"a": {"stop": {"serving": {"rounds": 4}}}, "b": {"stop": {"serving": {"rounds": 4}}}}
    for name, cells in (
        ("camera-ready-x86_64", stop),
        ("camera-ready-retime-r8-x86_64", {"a": {"stop": {"serving": {"rounds": 8}}}}),
        ("camera-ready-retime-r9-x86_64", {"b": {"stop": {"serving": {"rounds": 8}}}}),
        ("camera-ready-aarch64", stop),
    ):
        (runs / name).mkdir(parents=True)
        fm.write_json(runs / name / "cells.json", cells)
    fm.write_json(
        tmp_path / "retime" / "camera-ready-retime-r8-x86_64.json",
        {"retime": {"cells": ["a", "b"]}},
    )

    def problems(parent, attempt):
        return ["binary_sha256: attempt 'x', parent 'y'"] if "r9" in attempt.name else []

    monkeypatch.setattr(retime, "attempt_problems", problems)
    rows = retime.dispositions(runs, "camera-ready", "intel")
    assert [(r["cell"], r["selected"]) for r in rows] == [
        ("a", "camera-ready-retime-r8-x86_64"),
        ("b", "camera-ready-x86_64"),
    ]
    c = results.Campaign("t", "camera-ready", ())
    text = tables.retime_table(runs, c, "intel")
    assert (
        "| a | serving | camera-ready-x86_64 (4) | camera-ready-retime-r8-x86_64 (8, accepted)"
        in text
    )
    assert "(8, refused) | camera-ready-x86_64 | binary_sha256" in text
    assert "No re-timed cell." in tables.retime_table(runs, c, "arm")
    (intel, arm) = numbers.retime_counts(runs, c)
    assert intel[1] == "2; 2; 1; 1"
    assert arm[1] == "unknown; 0; 0; 0 (no re-time run)"
    err = capsys.readouterr().err
    assert "warning: re-timing on intel: 1 (cell, mode) pairs refused;" in err
    assert "arm" not in err
