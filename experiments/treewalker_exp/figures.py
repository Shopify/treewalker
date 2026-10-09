"""The paper's figures, regenerated from the final runs (`results`).

The same figures as the v1 scripts (`plot.py`, `gen_heatmap_tex.py` and the
convergence plot of `decomposition_validation.py`, at the neurips2026 tag), with
the same cells, axes and encodings, from the new suites:

- speedups are row-weighted ratios over the same groups, and the ribbons are their
  95% bootstrap intervals (v1 drew the range over repeats);
- latencies are p50 per group, with p5 to p95 as error bars;
- the Expedia size curve is the fixed-model cohort of sizes 4 to 32 (v1: Grid 4's
  group-size distributions);
- the ablation reads the research build's variants against its own all-on;
  monotonic scans act only without the precompute, so their effect is
  disable_monotonic+disable_varying_precompute over disable_varying_precompute; the
  node visits' "all disabled" is every switch but monotonic, which changes no node
  visit (only scan compares).

Figures are on Intel unless they compare machines, as in v1.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import polars as pl

from . import results

# ---------------------------------------------------------------------------
# Theme and palettes (v1's, grayscale-safe with redundant encoding)
# ---------------------------------------------------------------------------
DPI = 300
FW_COLORS = {"LightGBM": "#6E3A7A", "XGBoost": "#2A9D8F"}
FW_LINETYPES = {"LightGBM": "solid", "XGBoost": "dashed"}
FW_SHAPES = {"LightGBM": "o", "XGBoost": "s"}
DS_COLORS = {"FLCHAIN": "#6E3A7A", "SUPPORT": "#2A9D8F", "Expedia": "#E9C46A"}
DS_ORDER = ["FLCHAIN", "SUPPORT", "Expedia"]
DS_LINETYPES = {"FLCHAIN": "solid", "SUPPORT": "dashed", "Expedia": "dotted"}
DS_MARKERS = {"FLCHAIN": "o", "SUPPORT": "s", "Expedia": "^"}
METHOD_COLORS = {
    "TreeWalker": "#264653",
    "TreeWalker (full walk)": "#5A8A90",
    "lleaves": "#2A9D8F",
    "tl2cgen": "#8AB17D",
    "Native": "#E76F51",
    "QuickScorer": "#E9C46A",
}
METHOD_LABELS = {
    "treewalker": "TreeWalker",
    "treewalker_fullwalk": "TreeWalker (full walk)",
    "lightgbm_native": "Native",
    "xgboost_native": "Native",
    "tl2cgen": "tl2cgen",
    "lleaves": "lleaves",
    "quickscorer": "QuickScorer",
}
DATASET_LABELS = {"support": "SUPPORT", "flchain": "FLCHAIN", "expedia": "Expedia"}
FRAMEWORK_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}
HEATMAP_PALETTE = ["#EDEDED", "#E8D4C8", "#EDB880", "#A7B644", "#3D8B5E", "#2E5B6E", "#28272A"]

# The reference configuration and the ablation's anchors.
REF_T, REF_L, REF_G = 500, 8, 16
ANCHORS = [("FLCHAIN", "flchain", 4), ("SUPPORT", "support", 16), ("Expedia", "expedia", None)]
# v1's flags, then the ones v1 did not have, and their variants against the
# research build's all-on (monotonic against the precompute switched off).
FLAGS = [
    ("Precompute", "disable_varying_precompute", "all-on"),
    ("Unsplit", "disable_unsplit", "all-on"),
    ("Tree ordering", "|disable_tree_ordering", "all-on"),
    ("Prefix grouping", "|disable_prefix_grouping", "all-on"),
    ("Monotonic", "disable_monotonic+disable_varying_precompute", "disable_varying_precompute"),
    ("Bitset interning", "|disable_bitset_intern", "all-on"),
    ("Predicate sweep", "disable_predicate_sweep", "all-on"),
    ("Predicate dedup", "|disable_predicate_dedup", "all-on"),
    ("Exact sums", "disable_exact_sums", "all-on"),
]
ALL_DISABLED = (
    "disable_unsplit+disable_varying_precompute|"
    "disable_bitset_intern+disable_prefix_grouping+disable_tree_ordering"
)


def theme_paper(**overrides: Any) -> Any:
    from plotnine import element_line, element_rect, element_text, theme, theme_minimal

    base = theme_minimal() + theme(
        text=element_text(family="serif", size=10),
        axis_title=element_text(size=10),
        axis_text=element_text(size=8),
        strip_text=element_text(size=10, weight="bold"),
        legend_title=element_text(size=9),
        legend_text=element_text(size=8),
        legend_background=element_rect(fill="white", alpha=0.85, color="none", size=0),
        panel_grid_major=element_line(color="#E0E0E0", size=0.3),
        panel_grid_minor=element_line(color="#F0F0F0", size=0.2),
    )
    return base + theme(**overrides) if overrides else base


def _panel_label(labels: list[str]) -> dict[str, str]:
    return {lbl: f"({chr(97 + i)}) {lbl}" for i, lbl in enumerate(labels)}


def _save(p: Any, out: Path, name: str, width: float, height: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p.save(out / f"{name}.{ext}", width=width, height=height, dpi=DPI, verbose=False)
    print(f"  {name}", file=sys.stderr)


def _panels(pdf: Any, column: str = "dataset") -> Any:
    import pandas as pd

    pm = _panel_label(DS_ORDER)
    pdf["panel"] = pd.Categorical(
        pdf[column].map(DATASET_LABELS).map(pm), categories=[pm[k] for k in DS_ORDER], ordered=True
    )
    return pdf


def reference_cells(df: Any, horizon: int = REF_G) -> Any:
    """The v1 grid's cells: survival panels at ``horizon`` and Expedia's sessions,
    without seed replicates."""
    survival = (pl.col("family") == "panel") & (pl.col("horizon") == horizon)
    sessions = pl.col("family") == "sessions"
    return df.filter((survival | sessions) & pl.col("replicate").is_null())


def algorithmic(df: Any) -> Any:
    """TreeWalker's speedup over its own full walk, with its interval."""
    return results.rows(df, "treewalker_fullwalk")


def _speedup_lines(pdf: Any, x: str, xlab: str, out: Path, name: str, **scale: Any) -> None:
    from plotnine import (
        aes,
        facet_wrap,
        geom_hline,
        geom_line,
        geom_point,
        geom_ribbon,
        ggplot,
        labs,
        scale_color_manual,
        scale_fill_manual,
        scale_linetype_manual,
        scale_shape_manual,
        scale_x_log10,
    )

    pdf["fw"] = pdf["framework"].map(FRAMEWORK_LABELS)
    p = (
        ggplot(pdf, aes(x=x, y="speedup", color="fw", fill="fw"))
        + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.12, color=None)
        + geom_line(aes(linetype="fw"), size=1)
        + geom_point(aes(shape="fw"), size=2.5)
        + geom_hline(yintercept=1.0, linetype="dashed", color="gray", alpha=0.5)
        + facet_wrap("~panel", scales=scale.pop("facet_scales", "fixed"))
        + scale_x_log10(**scale)
        + scale_color_manual(values=FW_COLORS)
        + scale_fill_manual(values=FW_COLORS, guide=None)
        + scale_linetype_manual(values=FW_LINETYPES)
        + scale_shape_manual(values=FW_SHAPES)
        + labs(x=xlab, y="Speedup vs full walk", color="", linetype="", shape="")
        + theme_paper(figure_size=(5.5, 2.475), legend_position="top")
    )
    _save(p, out, name, 5.5, 2.475)


def fig_horizon_amortization(factorial: Any, out: Path) -> None:
    """Speedup over the full walk against group width at T=500, L=8: the survival
    panels by horizon, Expedia by cohort size."""
    a = algorithmic(factorial.filter(pl.col("arch") == "intel"))
    ref = (pl.col("T") == REF_T) & (pl.col("L") == REF_L) & pl.col("replicate").is_null()
    surv = a.filter(ref & (pl.col("family") == "panel"))
    cohort = a.filter(ref & (pl.col("family") == "cohort"))
    pdf = _panels(
        pl.concat([surv, cohort])
        .select("dataset", "framework", "G", "speedup", "lo", "hi")
        .to_pandas()
    )
    _speedup_lines(pdf, "G", "Group width (G)", out, "horizon_amortization", facet_scales="free_x")


def heatmap_cells(factorial: Any) -> dict[tuple[str, int, int], float]:
    """Speedup over the full walk per (dataset, T, L), the mean over frameworks,
    at horizon 16 and Expedia's sessions, on Intel."""
    a = reference_cells(algorithmic(factorial.filter(pl.col("arch") == "intel")))
    mean = a.group_by("dataset", "T", "L").agg(pl.col("speedup").mean())
    return {(r["dataset"], r["T"], r["L"]): r["speedup"] for r in mean.iter_rows(named=True)}


def fig_speedup_heatmap(factorial: Any, out: Path) -> None:
    import pandas as pd
    from plotnine import (
        aes,
        facet_wrap,
        geom_text,
        geom_tile,
        ggplot,
        labs,
        scale_color_identity,
        scale_fill_gradientn,
    )

    cells = heatmap_cells(factorial)
    pdf = pd.DataFrame(
        [
            {"dataset": d, "n_trees": t, "max_depth": lv, "speedup": v}
            for (d, t, lv), v in cells.items()
        ]
    )
    pdf = _panels(pdf)
    pdf["speedup_label"] = pdf["speedup"].map(lambda x: f"{x:.1f}")
    # Labels black or white for contrast, as in the TikZ version (v1 drew them black,
    # with speedups up to 4; the darkest tiles now need white).
    lo, hi = pdf["speedup"].min(), pdf["speedup"].max()
    pdf["text_color"] = [_text_color(_interp((v - lo) / (hi - lo or 1.0))) for v in pdf["speedup"]]
    pdf["n_trees_cat"] = pd.Categorical(
        pdf["n_trees"].astype(str), categories=["50", "500", "1000", "2000"], ordered=True
    )
    pdf["max_depth_cat"] = pd.Categorical(
        pdf["max_depth"].astype(str), categories=["2", "4", "8", "16"], ordered=True
    )
    p = (
        ggplot(pdf, aes(x="n_trees_cat", y="max_depth_cat", fill="speedup"))
        + geom_tile(color="white", size=0.5)
        + geom_text(aes(label="speedup_label", color="text_color"), size=8)
        + facet_wrap("~panel")
        + scale_fill_gradientn(colors=HEATMAP_PALETTE)
        + scale_color_identity()
        + labs(x="Number of trees (T)", y="Max depth (L)", fill="Speedup")
        + theme_paper(figure_size=(5.5, 2.2))
    )
    _save(p, out, "speedup_heatmap", 5.5, 2.2)
    (out / "speedup_heatmap.tex").write_text(heatmap_tex(cells))
    print("  speedup_heatmap.tex", file=sys.stderr)


def fig_method_comparison(factorial: Any, out: Path) -> None:
    """Latency per group at the reference cells, per method, dataset and framework."""
    import pandas as pd
    from plotnine import (
        aes,
        coord_flip,
        element_text,
        facet_grid,
        geom_bar,
        geom_errorbar,
        ggplot,
        labs,
        scale_fill_manual,
        scale_y_log10,
    )

    d = reference_cells(factorial.filter(pl.col("arch") == "intel"))
    d = d.filter((pl.col("T") == REF_T) & (pl.col("L") == REF_L) & (pl.col("mode") == "serving"))
    keep = (pl.col("method").is_in(list(METHOD_LABELS)) & (pl.col("variant") != "headline")) & (
        pl.col("method") != "quickscorer"
    )
    pdf = (
        d.filter(keep)
        .select("dataset", "framework", "method", "p5_us", "p50_us", "p95_us")
        .to_pandas()
    )
    pdf["method_label"] = pdf["method"].map(METHOD_LABELS)
    pdf["fw_cat"] = pd.Categorical(
        pdf["framework"].map(FRAMEWORK_LABELS), categories=["LightGBM", "XGBoost"], ordered=True
    )
    pdf["ds_cat"] = pd.Categorical(
        pdf["dataset"].map(DATASET_LABELS), categories=DS_ORDER, ordered=True
    )
    order = ["Native", "tl2cgen", "lleaves", "TreeWalker (full walk)", "TreeWalker"]
    pdf["method_label"] = pd.Categorical(pdf["method_label"], categories=order, ordered=True)
    p = (
        ggplot(pdf, aes(x="method_label", y="p50_us", fill="method_label"))
        + geom_bar(stat="identity")
        + geom_errorbar(aes(ymin="p5_us", ymax="p95_us"), width=0.3)
        + facet_grid("fw_cat~ds_cat")
        + scale_y_log10()
        + coord_flip()
        + scale_fill_manual(values=METHOD_COLORS, guide=None)
        + labs(x="", y="Latency (\u00b5s per group, p50, log scale)")
        + theme_paper(figure_size=(5.5, 2.4), axis_text_y=element_text(size=7))
    )
    # Bars start at zero on the log axis, as in v1: numpy warns on log10(0).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero", RuntimeWarning)
        _save(p, out, "method_comparison", 5.5, 2.4)


def fig_cross_platform(factorial: Any, out: Path) -> None:
    """Speedup over the full walk on Arm against Intel, over v1's 544 cells."""
    from plotnine import (
        aes,
        geom_abline,
        geom_point,
        ggplot,
        labs,
        scale_color_manual,
        scale_shape_manual,
        scale_x_log10,
        scale_y_log10,
    )

    a = algorithmic(factorial).filter(
        pl.col("family").is_in(["panel", "sessions"]) & pl.col("replicate").is_null()
    )
    wide = (
        a.select("arch", "cell", "dataset", "speedup")
        .pivot(on="arch", index=["cell", "dataset"], values="speedup")
        .drop_nulls()
    )
    pdf = wide.to_pandas()
    pdf["ds"] = pdf["dataset"].map(DATASET_LABELS)
    p = (
        ggplot(pdf, aes(x="intel", y="arm", color="ds", shape="ds"))
        + geom_point(alpha=0.45, size=1.8)
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="gray")
        + scale_x_log10()
        + scale_y_log10()
        + scale_color_manual(values=DS_COLORS)
        + scale_shape_manual(values=DS_MARKERS)
        + labs(x="Intel speedup", y="Arm speedup", color="", shape="")
        + theme_paper(figure_size=(3.3, 3.3), legend_position="top")
    )
    _save(p, out, "cross_platform", 3.3, 3.3)


def _anchor(df: Any, dataset: str, horizon: int | None) -> Any:
    sel = (pl.col("dataset") == dataset) & (pl.col("T") == REF_T) & (pl.col("L") == REF_L)
    sel &= pl.col("horizon").is_null() if horizon is None else pl.col("horizon") == horizon
    return df.filter(sel)


def ablation_slowdowns(ablation: Any, arch: str = "intel", mode: str = "serving") -> Any:
    """Each switch's slowdown at the ablation's anchors: the research build's time with
    the switch over its time without it, the mean over frameworks."""
    research = ablation.filter(
        (pl.col("arch") == arch)
        & (pl.col("method") == "treewalker_research")
        & (pl.col("mode") == mode)
    )
    out = []
    for label, ds, h in ANCHORS:
        a = _anchor(research, ds, h)
        t = {(r["framework"], r["variant"]): r["us_per_row"] for r in a.iter_rows(named=True)}
        for flag, variant, against in FLAGS:
            ratios = [
                t[(fw, variant)] / t[(fw, against)]
                for fw in ("lightgbm", "xgboost")
                if (fw, variant) in t and (fw, against) in t
            ]
            if ratios:
                out.append({"ds": label, "flag": flag, "slowdown": sum(ratios) / len(ratios)})
    return pl.DataFrame(out)


def fig_ablation_waterfall(ablation: Any, out: Path) -> None:
    import pandas as pd
    from plotnine import (
        aes,
        element_text,
        geom_hline,
        geom_line,
        geom_point,
        ggplot,
        labs,
        scale_color_manual,
        scale_linetype_manual,
        scale_shape_manual,
    )

    pdf = ablation_slowdowns(ablation).to_pandas()
    pdf["ds"] = pd.Categorical(pdf["ds"], categories=DS_ORDER, ordered=True)
    pdf["flag"] = pd.Categorical(pdf["flag"], categories=[f for f, _, _ in FLAGS], ordered=True)
    p = (
        ggplot(pdf, aes(x="flag", y="slowdown", color="ds", linetype="ds", shape="ds", group="ds"))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + geom_hline(yintercept=1.0, linetype="dashed", color="gray", alpha=0.4)
        + scale_color_manual(values=DS_COLORS)
        + scale_linetype_manual(values=DS_LINETYPES)
        + scale_shape_manual(values=DS_MARKERS)
        + labs(
            x="Disabled optimisation",
            y="Slowdown (disabled / baseline)",
            color="",
            linetype="",
            shape="",
        )
        + theme_paper(
            figure_size=(5.5, 2.475),
            axis_text_x=element_text(rotation=25, ha="right"),
            legend_position="top",
        )
    )
    _save(p, out, "ablation_waterfall", 5.5, 2.475)


def fig_ablation_nodes(work: Any, out: Path) -> None:
    """Node visits per group at the anchors (FLCHAIN and SUPPORT at horizon 16),
    all optimizations enabled and disabled, the mean over frameworks."""
    import pandas as pd
    from plotnine import (
        aes,
        element_text,
        facet_wrap,
        geom_bar,
        ggplot,
        labs,
        position_dodge,
        scale_fill_manual,
    )

    cols = {
        "constant_steps": "Const steps",
        "unsplit_skips": "Unsplit skips",
        "recursive_calls": "Recursive calls",
        "leaf_hits": "Leaf hits",
    }
    states = {
        "all-on": "(a) All optimizations enabled",
        ALL_DISABLED: "(b) All optimizations disabled",
    }
    rows = []
    for label, ds, h in [
        ("FLCHAIN", "flchain", 16),
        ("SUPPORT", "support", 16),
        ("Expedia", "expedia", None),
    ]:
        a = _anchor(work, ds, h)
        for variant, state in states.items():
            v = a.filter(pl.col("variant") == variant)
            for c, cl in cols.items():
                rows.append(
                    {
                        "ds": label,
                        "state_panel": state,
                        "category": cl,
                        "visits": float((v[c] / v["groups"]).mean()),
                    }
                )
    pdf = pd.DataFrame(rows)
    pdf["category"] = pd.Categorical(pdf["category"], categories=list(cols.values()), ordered=True)
    pdf["state_panel"] = pd.Categorical(
        pdf["state_panel"], categories=list(states.values()), ordered=True
    )
    pdf["ds"] = pd.Categorical(pdf["ds"], categories=DS_ORDER, ordered=True)
    p = (
        ggplot(pdf, aes(x="category", y="visits", fill="ds"))
        + geom_bar(stat="identity", position=position_dodge(width=0.8), width=0.7)
        + facet_wrap("~state_panel")
        + scale_fill_manual(values=DS_COLORS)
        + labs(x="", y="Node visits / group", fill="")
        + theme_paper(
            figure_size=(5.5, 2.475),
            legend_position="top",
            axis_text_x=element_text(rotation=30, ha="right"),
        )
    )
    _save(p, out, "ablation_nodes", 5.5, 2.475)


def fig_depth_sensitivity(factorial: Any, out: Path) -> None:
    a = reference_cells(algorithmic(factorial.filter(pl.col("arch") == "intel"))).filter(
        pl.col("T") == REF_T
    )
    pdf = _panels(a.select("dataset", "framework", "L", "speedup", "lo", "hi").to_pandas())
    _speedup_lines(
        pdf,
        "L",
        "Max tree depth (L)",
        out,
        "depth_sensitivity",
        breaks=[2, 4, 8, 16],
        labels=["2", "4", "8", "16"],
    )


def fig_ntrees_scaling(factorial: Any, out: Path) -> None:
    a = reference_cells(algorithmic(factorial.filter(pl.col("arch") == "intel"))).filter(
        pl.col("L") == REF_L
    )
    pdf = _panels(a.select("dataset", "framework", "T", "speedup", "lo", "hi").to_pandas())
    _speedup_lines(
        pdf,
        "T",
        "Number of trees (T)",
        out,
        "ntrees_scaling",
        breaks=[50, 500, 1000, 2000],
        labels=["50", "500", "1K", "2K"],
    )


def decomposition(work: Any) -> tuple[Any, dict[tuple[str, str], float]]:
    """The per-row work ratio W / (G T L) of TreeWalker's two evaluators at T=500,
    L=8 (Theorem 1's terms from the counters, per group: constant steps, varying
    splits, G T leaf writes, and the predicate work, partition row evaluations for
    trace and precompute row evaluations for the default), and the Corollary 1
    asymptotes: (d_v + 1) / L for trace, d_v from the largest G, and 1 / L."""
    evaluators = {"all-on": "precompute", "disable_varying_precompute": "trace"}
    w = work.filter(
        (pl.col("T") == REF_T)
        & (pl.col("L") == REF_L)
        & pl.col("variant").is_in(list(evaluators))
        & pl.col("family").is_in(["panel", "sessions"])
    ).with_columns(
        evaluator=pl.col("variant").replace_strict(evaluators),
        G=pl.col("rows") / pl.col("groups"),
        Q=pl.when(pl.col("variant") == "all-on")
        .then(pl.col("precompute_row_evals"))
        .otherwise(pl.col("partition_row_evals")),
    )
    w = w.with_columns(
        per_row_ratio=(
            (pl.col("constant_steps") + pl.col("varying_splits") + pl.col("Q")) / pl.col("groups")
            + pl.col("G") * pl.col("T")
        )
        / pl.col("G")
        / (pl.col("T") * pl.col("L"))
    )
    asym: dict[tuple[str, str], float] = {}
    for ds in ("support", "flchain"):
        trace = w.filter(
            (pl.col("dataset") == ds)
            & (pl.col("evaluator") == "trace")
            & (pl.col("framework") == "lightgbm")
        )
        top = trace.sort("G").row(-1, named=True)
        d_v = top["partition_row_evals"] / (top["G"] * top["T"] * top["groups"])
        asym[(ds, "trace")] = (d_v + 1) / REF_L
        asym[(ds, "precompute")] = 1 / REF_L
    return w, asym


def fig_asymptotic_convergence(work: Any, out: Path) -> None:
    import pandas as pd
    from plotnine import (
        aes,
        element_text,
        geom_hline,
        geom_line,
        geom_point,
        ggplot,
        labs,
        scale_color_manual,
        scale_linetype_manual,
        scale_shape_manual,
        scale_x_log10,
        scale_y_continuous,
        theme,
        theme_minimal,
    )

    w, asym = decomposition(work)
    ev_label = {"trace": "trace", "precompute": "precompute (default)"}
    df = w.filter(
        (pl.col("framework") == "lightgbm") & pl.col("dataset").is_in(["support", "flchain"])
    ).to_pandas()
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS)
    df["evaluator_label"] = df["evaluator"].map(ev_label)
    asym_df = pd.DataFrame(
        [
            {"dataset_label": DATASET_LABELS[ds], "evaluator_label": ev_label[ev], "yintercept": v}
            for (ds, ev), v in asym.items()
        ]
    )
    p = (
        ggplot(
            df,
            aes(
                x="G",
                y="per_row_ratio",
                color="dataset_label",
                linetype="evaluator_label",
                shape="evaluator_label",
            ),
        )
        + geom_hline(
            data=asym_df,
            mapping=aes(yintercept="yintercept", color="dataset_label", linetype="evaluator_label"),
            size=0.4,
            alpha=0.45,
            inherit_aes=False,
        )
        + geom_line(size=0.7)
        + geom_point(size=2.0)
        + scale_x_log10(breaks=[4, 16, 64, 128], limits=(3.5, 140))
        + scale_y_continuous(limits=(0, 0.75), breaks=[0, 0.25, 0.5, 0.75])
        + scale_color_manual(values={"SUPPORT": "#1e66f5", "FLCHAIN": "#df8e1d"}, name="Dataset")
        + scale_linetype_manual(
            values={"trace": "solid", "precompute (default)": "dashed"}, name="Evaluator"
        )
        + scale_shape_manual(values={"trace": "o", "precompute (default)": "s"}, name="Evaluator")
        + labs(x="Group size $G$ (log scale)", y="Per-row work ratio  $W / (G\\cdot T\\cdot L)$")
        + theme_minimal()
        + theme(
            figure_size=(5.5, 3.0),
            text=element_text(size=10),
            axis_title=element_text(size=10),
            legend_position="right",
            legend_title=element_text(size=9),
            legend_text=element_text(size=8),
        )
    )
    _save(p, out, "asymptotic_convergence", 5.5, 3.0)


# ---------------------------------------------------------------------------
# The heatmap as TikZ (v1's gen_heatmap_tex.py, with the new cells)
# ---------------------------------------------------------------------------
_TEX_PANELS = [("flchain", "(a)~FLCHAIN"), ("support", "(b)~SUPPORT"), ("expedia", "(c)~Expedia")]
_TEX_ROWS, _TEX_COLS = [2, 4, 8, 16], [50, 500, 1000, 2000]


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _interp(t: float) -> tuple[int, int, int]:
    """Linear RGB interpolation across the palette, as scale_fill_gradientn does."""
    t = max(0.0, min(1.0, t))
    n = len(HEATMAP_PALETTE) - 1
    pos = t * n
    lo = int(pos)
    if lo >= n:
        return _rgb(HEATMAP_PALETTE[-1])
    frac = pos - lo
    a, b = _rgb(HEATMAP_PALETTE[lo]), _rgb(HEATMAP_PALETTE[lo + 1])
    return (
        round(a[0] * (1 - frac) + b[0] * frac),
        round(a[1] * (1 - frac) + b[1] * frac),
        round(a[2] * (1 - frac) + b[2] * frac),
    )


def _text_color(rgb: tuple[int, int, int]) -> str:
    """The higher-WCAG-contrast of black or white against the cell."""

    def lin(c: int) -> float:
        x = c / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    y = 0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2])
    return "black" if (y + 0.05) / 0.05 >= 1.05 / (y + 0.05) else "white"


def heatmap_tex(cells: dict[tuple[str, int, int], float]) -> str:
    vals = list(cells.values())
    vmin, vmax = min(vals), max(vals)
    span = vmax - vmin if vmax > vmin else 1.0
    lines = [
        "% Auto-generated by treewalker-exp figures; do not hand-edit.",
        "% Cells are coloured by interpolating the heatmap's 7-stop palette;",
        "% text colour (black/white) is chosen per cell for contrast.",
        "",
        r"\begin{tikzpicture}[",
        r"  x=7.5mm, y=7mm,",
        r"  font=\footnotesize,",
        r"  cell/.style={",
        r"    rectangle, minimum width=7.5mm, minimum height=7mm,",
        r"    draw=white, line width=0.4pt,",
        r"    inner sep=0pt, anchor=center,",
        r"  },",
        r"  rowlabel/.style={anchor=east, font=\scriptsize, inner sep=2pt},",
        r"  collabel/.style={anchor=north, font=\scriptsize, inner sep=2pt},",
        r"  ptitle/.style={font=\small, anchor=south},",
        r"  axlab/.style={font=\scriptsize},",
        r"]",
        "",
        f"% Speedup range across all cells: [{vmin:.4f}, {vmax:.4f}]",
        "",
    ]
    shifts = [None, "3.7cm", "7.4cm"]
    for idx, (ds, title) in enumerate(_TEX_PANELS):
        lines.append(f"% ---------------- Panel {title} ----------------")
        ind = "  " if idx else ""
        if idx:
            lines.append(rf"\begin{{scope}}[xshift={shifts[idx]}]")
        lines.append(rf"{ind}\node[ptitle] at (1.5, 3.65) {{{title}}};")
        for r_idx, depth in enumerate(_TEX_ROWS):
            for c_idx, trees in enumerate(_TEX_COLS):
                val = cells[(ds, trees, depth)]
                rgb = _interp((val - vmin) / span)
                fill = f"{{rgb,255:red,{rgb[0]};green,{rgb[1]};blue,{rgb[2]}}}"
                lines.append(
                    rf"{ind}\node[cell, fill={fill}, text={_text_color(rgb)}] "
                    rf"at ({c_idx}, {r_idx}) {{{val:.2f}}};"
                )
        for c_idx, trees in enumerate(_TEX_COLS):
            lines.append(rf"{ind}\node[collabel] at ({c_idx}, -0.5) {{{trees}}};")
        if idx == 0:
            for r_idx, depth in enumerate(_TEX_ROWS):
                lines.append(rf"{ind}\node[rowlabel] at (-0.5, {r_idx}) {{{depth}}};")
            lines.append(rf"{ind}\node[rotate=90, axlab] at (-1.35, 1.5) " r"{Max depth ($L$)};")
        if idx:
            lines.append(r"\end{scope}")
        lines.append("")
    lines += [
        r"\node[axlab] at (6.43, -1.4) {Number of trees ($T$)};",
        "",
        "% ---------------- Colorbar ----------------",
    ]
    lines.append(r"\begin{scope}[xshift=11.1cm, yshift=-7mm, x=1cm, y=1cm]")
    lines.append(r"  \node[font=\scriptsize, anchor=south west] at (0, 3.65) {Speedup};")
    bands, bar_h, bar_w = 120, 3.5, 5
    for i in range(bands):
        rgb = _interp(i / (bands - 1))
        y = i * bar_h / bands
        fill = f"{{rgb,255:red,{rgb[0]};green,{rgb[1]};blue,{rgb[2]}}}"
        lines.append(
            rf"  \fill[fill={fill}] (0, {y:.4f}) rectangle ({bar_w}mm, {y + bar_h / bands:.4f});"
        )
    # v1's ticks were 1.0 to 5.0 in halves; a wider range steps by whole numbers.
    step = 0.5 if vmax - vmin <= 4.5 else 1.0
    ticks = [
        i * step for i in range(1, int(vmax / step) + 2) if vmin - 0.05 <= i * step <= vmax + 0.05
    ]
    for v in ticks:
        y = (v - vmin) / span * bar_h
        lines.append(
            rf"  \node[anchor=west, font=\scriptsize, inner sep=1pt] "
            rf"at ({bar_w + 0.5}mm, {y:.4f}) {{{v:.1f}}};"
        )
    lines += [
        rf"  \draw[line width=0.3pt] (0, 0) rectangle ({bar_w}mm, {bar_h});",
        r"\end{scope}",
        "",
        r"\end{tikzpicture}",
    ]
    return "\n".join(lines) + "\n"


def generate(runs: Path, out: Path) -> None:
    """Every figure, from the factorial and ablation runs."""
    factorial = results.speedups(runs, "factorial")
    ablation = results.speedups(runs, "ablation")
    work = results.work(runs, "ablation")
    fig_horizon_amortization(factorial, out)
    fig_speedup_heatmap(factorial, out)
    fig_method_comparison(factorial, out)
    fig_cross_platform(factorial, out)
    fig_ablation_waterfall(ablation, out)
    fig_ablation_nodes(work, out)
    fig_depth_sensitivity(factorial, out)
    fig_ntrees_scaling(factorial, out)
    fig_asymptotic_convergence(work, out)
