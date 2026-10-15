"""results and figures: cell designs, the TikZ heatmap, and the work decomposition."""

import pytest

from treewalker_exp import figures, results


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        (
            "support/nt500_md8_h16/lightgbm/panel",
            {
                "family": "panel",
                "T": 500,
                "L": 8,
                "horizon": 16,
                "G": 16,
                "k": None,
                "replicate": None,
            },
        ),
        ("flchain/nt2000_md16_h1024/xgboost/panel", {"family": "panel-long", "G": 1024}),
        ("expedia/nt500_md8/lightgbm/sessions", {"family": "sessions", "G": None, "horizon": None}),
        ("expedia/nt500_md8/xgboost/cohort32-n8", {"family": "cohort", "G": 8}),
        ("credit/nt500_md8/lightgbm/whatif-v2-k4-G128", {"family": "whatif", "k": 4, "G": 128}),
        ("support/nt500_md4_h16_r2/lightgbm/panel", {"family": "panel", "replicate": 2}),
        ("expedia-filled/nt50_md4/lightgbm/sessions", {"family": "sessions-filled"}),
    ],
)
def test_design_reads_the_cell_id(cell, expected):
    got = results.design(cell)
    assert {k: got[k] for k in expected} == expected


def test_heatmap_tex_has_every_cell_and_contrasting_text():
    cells = {
        (ds, t, depth): 1.0 + i / 10
        for i, (ds, t, depth) in enumerate(
            (ds, t, depth)
            for ds in ("flchain", "support", "expedia")
            for t in (50, 500, 1000, 2000)
            for depth in (2, 4, 8, 16)
        )
    }
    tex = figures.heatmap_tex(cells)
    assert tex.count(r"\node[cell,") == 48
    assert "text=white" in tex and "text=black" in tex  # the darkest and the lightest
    assert tex.startswith("% Auto-generated") and tex.rstrip().endswith(r"\end{tikzpicture}")


def test_numbers_round_half_up_as_the_paper_prints():
    from treewalker_exp import numbers

    assert (
        numbers.r(2.25, 1) == "2.3" and numbers.r(2.35, 1) == "2.4" and numbers.r(44.5, 0) == "45"
    )
    assert numbers.rng(1.04, 2.96, 1) == "1.0-3.0"


def test_decomposition_is_theorem_1s_per_row_work_over_g_t_l():
    import polars as pl

    # One group of 16 rows, T=500, L=8: C + V + Q = 8000 per group, plus G T leaf writes.
    row = {
        "cell": "support/nt500_md8_h16/lightgbm/panel",
        "dataset": "support",
        "framework": "lightgbm",
        "family": "panel",
        "T": 500,
        "L": 8,
        "horizon": 16,
        "groups": 2,
        "rows": 32,
        "constant_steps": 6000,
        "varying_splits": 2000,
        "precompute_row_evals": 8000,
        "partition_row_evals": 16000,
    }
    work = pl.DataFrame(
        [{**row, "variant": "all-on"}, {**row, "variant": "disable_varying_precompute"}]
    )
    w, asym = figures.decomposition(pl.concat([work, work.with_columns(dataset=pl.lit("flchain"))]))
    got = dict(w.filter(pl.col("dataset") == "support").select("evaluator", "per_row_ratio").rows())
    assert got["precompute"] == pytest.approx((8000 + 16 * 500) / 16 / (500 * 8))
    assert got["trace"] == pytest.approx((12000 + 16 * 500) / 16 / (500 * 8))
    # d_v from the trace evaluator's partition row evaluations at the largest G.
    assert asym[("support", "trace")] == pytest.approx((16000 / (16 * 500 * 2) + 1) / 8)
    assert asym[("support", "precompute")] == 1 / 8


def work_row(fw, variant, v, b, groups=2, ds="support", h=16):
    return {
        "dataset": ds,
        "framework": fw,
        "T": 500,
        "L": 8,
        "horizon": h,
        "cell": f"{ds}/{fw}",
        "variant": variant,
        "varying_splits": v,
        "recursive_calls": b,
        "unsplit_skips": v - b,
        "groups": groups,
    }


def test_unsplit_reports_u_and_the_ablated_paths_calls_beyond_it():
    import polars as pl

    from treewalker_exp import numbers

    # Default: V 100, B 60, so U 40; disabled: 104 recursive calls, E 44, 4 beyond U.
    work = pl.DataFrame(
        [work_row("lightgbm", "all-on", 100, 60), work_row("lightgbm", "disable_unsplit", 104, 104)]
    )
    ((what, value),) = numbers.unsplit(work)
    assert "SUPPORT" in what and "lightgbm" in what
    assert value == "20.0 (40.0% of V); E 22.0, 10.00% beyond U"
    broken = work.with_columns(unsplit_skips=pl.lit(1, pl.Int64))
    with pytest.raises(ValueError, match="unsplit skips differ"):
        numbers.unsplit(broken)


def test_the_tl2cgen_defect_table_lists_cells_with_affected_rows():
    from treewalker_exp import tables

    def check(method, affected, differ, err, rows=100):
        defect = {
            "name": "tl2cgen-f64-missing-alias: Entry union int missing vs double fvalue",
            "affected_rows": affected,
            "affected_differ": differ,
            "affected_max_abs": err,
        }
        return {"method": method, "rows": rows, "defect": defect}

    f64 = [{"method": "tl2cgen", "precision": "F64"}]
    manifests = {
        "b/cell": {
            "methods": f64,
            "validation": {"checks": [check("tl2cgen", 3, 2, 0.5), check("tl2cgen", 3, 1, 0.1)]},
        },
        "a/cell": {"methods": f64, "validation": {"checks": [check("tl2cgen", 0, 0, 0.0)]}},
        "c/cell": {"validation": {"checks": [{"method": "treewalker", "rows": 100}]}},
    }
    assert tables.tl2cgen_defect(manifests) == [
        {
            "cell": "b/cell",
            "affected_rows": 3,
            "differing_rows": 2,
            "largest_error": 0.5,
            "rows": 100,
        }
    ]
    text = tables.tl2cgen_defect_table(manifests, "intel")
    assert "1 of the 2 cells where tl2cgen ran a float64 (LightGBM) model" in text
    assert "| b/cell | 100 | 3 | 2 | 0.5 |" in text


def test_the_scenario_table_restores_the_trace_columns_with_d_v_plus_1_over_l():
    import polars as pl

    from treewalker_exp import tables

    cell = "scenario_credit/lightgbm/whatif-v1-k1-G4"
    d = results.design(cell)
    base = {"arch": "intel", "cell": cell, "mode": "serving", **d}
    fac = pl.DataFrame(
        [
            base
            | {"method": "treewalker", "variant": "all-on", "latency_us": 10.0}
            | {"speedup": 1.0, "lo": 1.0, "hi": 1.0},
            base
            | {"method": "treewalker_fullwalk", "variant": "-", "latency_us": 20.0}
            | {"speedup": 2.0, "lo": 1.9, "hi": 2.1},
        ]
    )
    # 2 groups of 4 rows: G*T = 2000 per group; trace: 4000 partition evaluations a
    # group, so d_v = 2 and the prediction (d_v + 1) / L = 3 / 8.
    counts = {"constant_steps": 0, "varying_splits": 0, "precompute_row_evals": 0}
    work = pl.DataFrame(
        [
            {**d, "cell": cell, "variant": v, "groups": 2, "rows": 8, **counts}
            | {"partition_row_evals": p}
            for v, p in (("all-on", 0), ("disable_varying_precompute", 8000))
        ]
    )
    text = tables.scenario_table(fac, work, "intel")
    assert "predicted (d_v+1)/L | d_v |" in text and "d+1" not in text
    assert text.rstrip().endswith("| 0.375 | 0.375 | 2.00 |")


def ablation_runs(effects):
    """Speedups rows of two ablation runs at the SUPPORT anchor: research all-on at 1
    us and disable_unsplit at 1 + effect, per (run, framework)."""
    import polars as pl

    rows = []
    for (run, fw), e in effects.items():
        base = {"arch": "intel", "run": run, "framework": fw, "mode": "serving"}
        base |= {"method": "treewalker_research", "dataset": "support", "T": 500, "L": 8}
        base |= {"horizon": 16}
        rows += [
            base | {"variant": "all-on", "latency_us": 1.0},
            base | {"variant": "disable_unsplit", "latency_us": 1.0 + e},
        ]
    return pl.DataFrame(rows)


def test_the_between_run_spread_tells_frameworks_from_runs_apart():
    from treewalker_exp import numbers, tables

    # Two cases with the same pooled 0-20% range and mean.
    same = ablation_runs({("r1", "lightgbm"): 0.0, ("r1", "xgboost"): 0.2})
    same = same.vstack(ablation_runs({("r2", "lightgbm"): 0.0, ("r2", "xgboost"): 0.2}))
    moved = ablation_runs({("r1", "lightgbm"): 0.0, ("r1", "xgboost"): 0.0})
    moved = moved.vstack(ablation_runs({("r2", "lightgbm"): 0.2, ("r2", "xgboost"): 0.2}))
    for df, spread in ((same, 0.0), (moved, 20.0)):
        e = figures.ablation_effects(df)
        s = figures.ablation_spread(e)
        assert [x["between_runs_pp"] for x in s] == pytest.approx([spread, spread])
        slow = figures.ablation_slowdowns(df).row(0, named=True)
        assert slow["slowdown"] == pytest.approx(1.1)
        assert slow["hi"] - slow["lo"] == pytest.approx(spread / 100)
        assert f"between-run range up to {spread:.1f} pp" in tables.ablation_effects_table(e)
        ((_, value),) = numbers.ablation_repeatability(e)
        assert f"between runs up to {spread:.1f} pp" in value
    ((_, value),) = numbers.ablation_repeatability(figures.ablation_effects(moved))
    assert value.startswith("Unsplit 0.0/0.0 | 20.0/20.0")


def test_the_quickscorer_head_to_head_on_expedia_filled():
    import polars as pl

    from treewalker_exp import tables

    lgb, xgb = (
        "expedia-filled/nt50_md4/lightgbm/sessions",
        "expedia-filled/nt50_md4/xgboost/sessions",
    )
    cells = {
        lgb: {"excluded": [{"method": "quickscorer", "variant": "-", "reason": "outputs differ"}]},
        xgb: {
            "excluded": [],
            "cell": {"baselines": {"quickscorer": {"effective": {"format": "xml"}}}},
        },
        "expedia/nt50_md4/xgboost/sessions": {"excluded": []},
    }
    row = {"arch": "intel", "mode": "serving"}
    df = pl.DataFrame(
        [
            row
            | {"cell": lgb, "method": "treewalker", "variant": "all-on", "latency_us": 10.0}
            | {"speedup": 1.0, "lo": 1.0, "hi": 1.0},
            row
            | {"cell": xgb, "method": "treewalker", "variant": "all-on", "latency_us": 10.0}
            | {"speedup": 1.0, "lo": 1.0, "hi": 1.0},
            row
            | {"cell": xgb, "method": "quickscorer", "variant": "-", "latency_us": 15.0}
            | {"speedup": 1.5, "lo": 1.4, "hi": 1.6},
        ]
    )
    rows = tables.quickscorer_filled(df, cells, "intel")
    assert [(r["framework"], r["input"], r["status"]) for r in rows] == [
        ("lightgbm", "direct", "excluded: outputs differ"),
        ("xgboost", "xml", "timed"),
    ]
    text = tables.quickscorer_filled_table(rows, "intel")
    assert "not a corrected value for the submitted Table 9, which is withdrawn" in text
    assert "Eligible: 2 cells (1 lightgbm, 1 xgboost); timed: 1 (0 lightgbm, 1 xgboost)" in text
    assert "| xgboost | 50 | 4 | xml | timed | 10.00 | 15.00 | 1.50x | [1.40, 1.60] |" in text
    assert "| lightgbm | 50 | 4 | direct | excluded: outputs differ | - | - | - | - |" in text


def test_the_scenario_table_matches_v1s_arithmetic():
    """v1's summarize_scenario.py arithmetic on the same counters,
    2,000 groups of 16 rows at T=500, L=8."""
    import polars as pl

    from treewalker_exp import tables

    cell = "scenario_credit/lightgbm/whatif-v1-k2-G16"
    d = results.design(cell)
    base = {"arch": "intel", "cell": cell, "mode": "serving", **d}
    fac = pl.DataFrame(
        [
            base
            | {"method": "treewalker", "variant": "all-on", "latency_us": 10.0}
            | {"speedup": 1.0, "lo": 1.0, "hi": 1.0},
            base
            | {"method": "treewalker_fullwalk", "variant": "-", "latency_us": 50.0}
            | {"speedup": 5.0, "lo": 4.9, "hi": 5.1},
        ]
    )
    w = {"cell": cell, "groups": 2000, "rows": 32000, "constant_steps": 1_000_000, **d}
    w |= {"varying_splits": 2_000_000, "precompute_row_evals": 3_000_000}
    work = pl.DataFrame(
        [
            w | {"variant": "all-on", "partition_row_evals": 0},
            w | {"variant": "disable_varying_precompute", "partition_row_evals": 48_000_000},
        ]
    )
    text = tables.scenario_table(fac, work, "intel")
    # v1: precompute ratio 0.172, trace ratio 0.523, predicted 0.500, d_v 3.00.
    assert (
        "| 2 | 16 | 10.00 | 50.00 | 5.00x | [4.90, 5.10] | 0.172 | 0.523 | 0.500 | 3.00 |" in text
    )
    paragraphs = [p for p in text.split("\n\n") if p and not p.startswith(("#", "|"))]
    assert paragraphs[0].startswith("Dataset:") and paragraphs[1].startswith("Trace ratio:")


def test_within_two_percent_counts_parity_cells_both_ways():
    from treewalker_exp import numbers

    assert numbers.within([0.98, 1.02, 1.0201, 0.979, 1.5, 0.999]) == 3
    # Just inside, at and just outside each boundary, as rounded doubles.
    assert [numbers.within([x]) for x in (0.9801, 0.98, 0.9799, 1.0199, 1.02, 1.0201)] == [
        1,
        1,
        0,
        1,
        1,
        0,
    ]
    assert abs(1.02 - 1) > 0.02  # why the boundary needs its tolerance


def test_the_filled_summary_counts_near_parity_on_the_timed_cells():
    from treewalker_exp import tables

    rows = [
        {"cell": f"c{i}", "framework": "xgboost", "T": 50, "L": i, "input": "xml"}
        | {"status": "timed", "tw_us": 1.0, "qs_us": x, "speedup": x, "lo": x, "hi": x}
        for i, x in enumerate((0.9799, 0.98, 1.01, 1.02, 1.0201))
    ] + [
        {"cell": "e", "framework": "lightgbm", "T": 50, "L": 2, "input": "direct"}
        | {"status": "excluded: outputs differ", "tw_us": None, "qs_us": None}
        | {"speedup": None, "lo": None, "hi": None}
    ]
    text = tables.quickscorer_filled_table(rows, "intel")
    assert "TreeWalker faster in 3 of the timed (within ±2%: 3)" in text


def test_a_missing_run_keeps_its_column_and_has_no_range():
    from treewalker_exp import numbers, tables

    # LightGBM in both runs; XGBoost only in the second (its first run lost the variant).
    df = ablation_runs({("r1", "lightgbm"): 0.2, ("r2", "lightgbm"): 0.3})
    df = df.vstack(ablation_runs({("r2", "xgboost"): 0.3}))
    # The first XGBoost run timed all-on but not the variant.
    first = ablation_runs({("r1", "xgboost"): 0.0})
    df = df.vstack(first.filter(first["variant"] == "all-on"))
    e = figures.ablation_effects(df)
    by_fw = {x["framework"]: x for x in figures.ablation_spread(e)}
    assert by_fw["xgboost"]["by_run"] == {"r2": pytest.approx(1.3)}
    assert by_fw["xgboost"]["between_runs_pp"] is None
    assert by_fw["lightgbm"]["between_runs_pp"] == pytest.approx(10.0)
    text = tables.ablation_effects_table(e)
    assert "| SUPPORT | Unsplit | xgboost | - | +30.0 | - |" in text
    assert "| SUPPORT | Unsplit | lightgbm | +20.0 | +30.0 | 10.0 |" in text
    ((_, value),) = numbers.ablation_repeatability(e)
    assert value == "Unsplit 20.0/- | 30.0/30.0 (between runs up to 10.0 pp)"
    # The waterfall's range uses only the frameworks both runs measured.
    slow = figures.ablation_slowdowns(df).row(0, named=True)
    assert (slow["lo"], slow["hi"]) == pytest.approx((1.2, 1.3))
    assert slow["slowdown"] == pytest.approx(1.25)  # the same population as the range
    # One run: no range, not a zero-length one.
    single = figures.ablation_slowdowns(ablation_runs({("r1", "lightgbm"): 0.1}))
    assert single.row(0, named=True)["lo"] is None
    # A variant missing from every run leaves no row; one run alone has no range.
    one = figures.ablation_effects(ablation_runs({("r1", "lightgbm"): 0.1}))
    assert "no row with two runs" in tables.ablation_effects_table(one)
    assert one.height == 1
