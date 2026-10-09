"""results and figures: cell designs, the sentinel's drift, and the TikZ heatmap."""

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


def test_drift_interpolates_between_sentinels_at_each_cells_position():
    def rec(cells_done, ticks):
        return {
            "role": "measure",
            "status": "ok",
            "cells_done": cells_done,
            "totals": {"k": [ticks, 10]},
        }

    warm_up = {"role": "warm-up", "status": "ok", "cells_done": 0, "totals": {}}
    run = {
        "sentinel": {"records": [warm_up, rec(0, 100), rec(2, 110), rec(4, 130)]},
        "invocations": [{"cells_done": [{"id": c} for c in "abcde"]}],
    }
    got = results.drift(run, key="k")
    assert got == pytest.approx({"a": 1.0, "b": 1.05, "c": 1.1, "d": 1.2, "e": 1.3})
    # Without two measured sentinels nothing is corrected.
    assert results.drift({"invocations": run["invocations"]}, key="k") == dict.fromkeys(
        "abcde", 1.0
    )


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
