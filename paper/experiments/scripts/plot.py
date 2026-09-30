#!/usr/bin/env python3
"""Generate publication figures from factorial grid benchmark data.

All figures use plotnine with a shared theme and redundant encoding
(color + linetype/marker/alpha) for grayscale print.

Usage:
    uv run python3 paper/experiments/scripts/plot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from plotnine import (
    aes,
    coord_flip,
    element_line,
    element_rect,
    element_text,
    facet_grid,
    facet_wrap,
    geom_bar,
    geom_errorbar,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_tile,
    geom_text,
    ggplot,
    ggsave,
    labs,
    position_dodge,
    scale_alpha_manual,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_shape_manual,
    scale_x_continuous,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_minimal,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "paper" / "experiments" / "data"
FIG = ROOT / "paper" / "experiments" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DPI = 300

# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------
def theme_paper(**overrides):
    """Consistent publication theme for all figures."""
    base = (
        theme_minimal()
        + theme(
            text=element_text(family="serif", size=10),
            axis_title=element_text(size=10),
            axis_text=element_text(size=8),
            strip_text=element_text(size=10, weight="bold"),
            legend_title=element_text(size=9),
            legend_text=element_text(size=8),
            legend_background=element_rect(fill="white", alpha=0.85,
                                           color="none", size=0),
            panel_grid_major=element_line(color="#E0E0E0", size=0.3),
            panel_grid_minor=element_line(color="#F0F0F0", size=0.2),
        )
    )
    if overrides:
        base = base + theme(**overrides)
    return base


# ---------------------------------------------------------------------------
# Palettes — grayscale-safe with redundant encoding
# ---------------------------------------------------------------------------
FW_COLORS = {"LightGBM": "#6E3A7A", "XGBoost": "#2A9D8F"}
FW_LINETYPES = {"LightGBM": "solid", "XGBoost": "dashed"}
FW_SHAPES = {"LightGBM": "o", "XGBoost": "s"}

DS_COLORS = {"FLCHAIN": "#6E3A7A", "SUPPORT": "#2A9D8F", "Expedia": "#E9C46A"}
_DS_ORDER = ["FLCHAIN", "SUPPORT", "Expedia"]
DS_LINETYPES = {"FLCHAIN": "solid", "SUPPORT": "dashed", "Expedia": "dotted"}
DS_MARKERS = {"FLCHAIN": "o", "SUPPORT": "s", "Expedia": "^"}

METHOD_COLORS = {
    "TreeWalker": "#264653", "TreeWalker (full walk)": "#5A8A90",
    "lleaves": "#2A9D8F", "tl2cgen": "#8AB17D",
    "Native": "#E76F51", "QuickScorer": "#E9C46A",
}
# CSV naming <-> paper labels (one-to-one, post-rename on review-fixes branch):
#   method="treewalker"          -> paper label "TreeWalker"
#                                   (optimized product with partial evaluation)
#   method="treewalker_fullwalk" -> paper label "TreeWalker (full walk)"
#                                   (row-independent full-walk reference)
# Table 3 (paper) defines "TreeWalker (full walk)" as the row-independent
# baseline used to isolate partial evaluation from data-layout effects.
METHOD_LABELS = {
    "treewalker": "TreeWalker", "treewalker_fullwalk": "TreeWalker (full walk)",
    "lightgbm_native": "Native", "xgboost_native": "Native",
    "tl2cgen": "tl2cgen", "lleaves": "lleaves",
    "quickscorer": "QuickScorer",
    # Legacy CSV column names (pre-rename, for back-compat reading).
    "treewalker_full": "TreeWalker", "treewalker_baseline": "TreeWalker (full walk)",
}
DATASET_LABELS = {"support": "SUPPORT", "flchain": "FLCHAIN", "expedia": "Expedia"}
FRAMEWORK_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}


def _panel_label(labels: list[str]) -> dict[str, str]:
    return {lbl: f"({chr(97+i)}) {lbl}" for i, lbl in enumerate(labels)}


def _save(p, name: str, width: float = 8, height: float = 5) -> None:
    for ext in ("png", "pdf"):
        ggsave(p, FIG / f"{name}.{ext}", width=width, height=height, dpi=DPI)
    print(f"  {name}", file=sys.stderr)


def _cast_horizon(df: pl.DataFrame) -> pl.DataFrame:
    """Cast horizon column to Int64 (empty strings → null)."""
    if "horizon" in df.columns and df.schema["horizon"] == pl.Utf8:
        df = df.with_columns(pl.col("horizon").cast(pl.Int64, strict=False))
    return df

def load_grid1(arch: str = "intel") -> pl.DataFrame:
    return _cast_horizon(pl.read_csv(DATA / f"grid1_results_{arch}.csv"))

def load_grid3(arch: str = "intel") -> pl.DataFrame:
    return _cast_horizon(pl.read_csv(DATA / f"grid3_results_{arch}.csv"))

def load_grid4(arch: str = "intel") -> pl.DataFrame:
    return _cast_horizon(pl.read_csv(DATA / f"grid4_results_{arch}.csv"))


def _per_repeat_speedup(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    # Algorithmic speedup = slow_fullwalk / fast_treewalker  (>1 when TW wins).
    # Read both current ("treewalker_fullwalk"/"treewalker") and legacy
    # ("treewalker_baseline"/"treewalker_full") column values.
    methods = df["method"].unique().to_list()
    slow_name = "treewalker_fullwalk" if "treewalker_fullwalk" in methods else "treewalker_baseline"
    fast_name = "treewalker" if "treewalker" in methods else "treewalker_full"
    slow = df.filter(pl.col("method") == slow_name).rename({"median_us": "slow_us"})
    fast = df.filter(pl.col("method") == fast_name).rename({"median_us": "fast_us"})
    # New data has no repeat_id; old data does.
    join_on = [c for c in group_cols if c in slow.columns]
    if "repeat_id" in slow.columns:
        join_on = join_on + ["repeat_id"]
    joined = slow.select(join_on + ["slow_us"]).join(
        fast.select(join_on + ["fast_us"]), on=join_on
    ).with_columns((pl.col("slow_us") / pl.col("fast_us")).alias("speedup"))
    agg_cols = [c for c in group_cols if c in joined.columns]
    return joined.group_by(agg_cols).agg(
        pl.col("speedup").median().alias("speedup"),
        pl.col("speedup").min().alias("speedup_lo"),
        pl.col("speedup").max().alias("speedup_hi"),
    )


# ---------------------------------------------------------------------------
# Figure 1: Horizon amortization — (a)/(b), log x, ribbons
# ---------------------------------------------------------------------------
def fig_horizon_amortization() -> None:
    df = load_grid1()
    # Survival: speedup vs fixed group width G
    surv = df.filter(
        pl.col("dataset").is_in(["support", "flchain"])
        & (pl.col("n_trees") == 500) & (pl.col("max_depth") == 8)
        & pl.col("horizon").is_not_null()
        & pl.col("method").is_in(["treewalker", "treewalker_fullwalk", "treewalker_full", "treewalker_baseline"])
    )
    surv_merged = _per_repeat_speedup(surv, ["dataset", "framework", "horizon"])
    surv_pdf = surv_merged.to_pandas()
    surv_pdf["G"] = surv_pdf["horizon"]

    # Expedia: speedup vs mean group size from Grid 4 distribution data.
    cols = ["dataset", "framework", "G", "speedup", "speedup_lo", "speedup_hi"]
    df4 = load_grid4()
    if "mean_group_size" in df4.columns:
        exp4 = df4.filter(
            (pl.col("n_trees") == 500) & (pl.col("max_depth") == 8)
            & pl.col("method").is_in(["treewalker", "treewalker_fullwalk", "treewalker_full", "treewalker_baseline"])
        )
        if exp4.height > 0:
            config_cols = ["group_dist", "mean_group_size", "framework"]
            exp_merged = _per_repeat_speedup(exp4, config_cols)
            exp_pdf = exp_merged.to_pandas()
            exp_pdf["dataset"] = "expedia"
            exp_pdf["G"] = exp_pdf["mean_group_size"]
            pdf = pd.concat([surv_pdf[cols], exp_pdf[cols]], ignore_index=True)
        else:
            pdf = surv_pdf[cols].copy()
    else:
        pdf = surv_pdf[cols].copy()

    pm = _panel_label(["FLCHAIN", "SUPPORT", "Expedia"])
    pdf["panel"] = pdf["dataset"].map(DATASET_LABELS).map(pm)
    pdf["panel"] = pd.Categorical(pdf["panel"],
        categories=[pm[k] for k in ["FLCHAIN", "SUPPORT", "Expedia"]], ordered=True)
    pdf["fw"] = pdf["framework"].map(FRAMEWORK_LABELS)

    p = (
        ggplot(pdf, aes(x="G", y="speedup", color="fw", fill="fw"))
        + geom_ribbon(aes(ymin="speedup_lo", ymax="speedup_hi"), alpha=0.12, color=None)
        + geom_line(aes(linetype="fw"), size=1)
        + geom_point(aes(shape="fw"), size=2.5)
        + geom_hline(yintercept=1.0, linetype="dashed", color="gray", alpha=0.5)
        + facet_wrap("~panel", scales="free_x")
        + scale_x_log10()
        + scale_color_manual(values=FW_COLORS)
        + scale_fill_manual(values=FW_COLORS, guide=None)
        + scale_linetype_manual(values=FW_LINETYPES)
        + scale_shape_manual(values=FW_SHAPES)
        + labs(x="Group width (G)", y="Speedup vs full walk",
               color="", linetype="", shape="")
        + theme_paper(figure_size=(5.5, 2.475), legend_position="top")
    )
    _save(p, "horizon_amortization", width=5.5, height=2.475)


# ---------------------------------------------------------------------------
# Figure 2: Speedup heatmap — (a)/(b)/(c)
# ---------------------------------------------------------------------------
def fig_speedup_heatmap() -> None:
    df = load_grid1()
    surv = df.filter(
        pl.col("dataset").is_in(["support", "flchain"])
        & (pl.col("horizon") == 16)
        & pl.col("method").is_in(["treewalker", "treewalker_fullwalk", "treewalker_full", "treewalker_baseline"])
    )
    exp = df.filter(
        pl.col("dataset").eq("expedia")
        & pl.col("method").is_in(["treewalker", "treewalker_fullwalk", "treewalker_full", "treewalker_baseline"])
    )
    both = pl.concat([surv, exp])
    merged = _per_repeat_speedup(both, ["dataset", "framework", "n_trees", "max_depth"])
    avg = merged.group_by("dataset", "n_trees", "max_depth").agg(pl.col("speedup").mean())
    pdf = avg.to_pandas()
    pm = _panel_label(["FLCHAIN", "SUPPORT", "Expedia"])
    pdf["panel"] = pdf["dataset"].map(DATASET_LABELS).map(pm)
    pdf["panel"] = pd.Categorical(pdf["panel"],
        categories=[pm[k] for k in ["FLCHAIN", "SUPPORT", "Expedia"]], ordered=True)
    pdf["speedup_label"] = pdf["speedup"].apply(lambda x: f"{x:.1f}")
    pdf["n_trees_cat"] = pd.Categorical(pdf["n_trees"].astype(str),
        categories=["50", "500", "1000", "2000"], ordered=True)
    pdf["max_depth_cat"] = pd.Categorical(pdf["max_depth"].astype(str),
        categories=["2", "4", "8", "16"], ordered=True)

    from plotnine import scale_fill_gradientn
    p = (
        ggplot(pdf, aes(x="n_trees_cat", y="max_depth_cat", fill="speedup"))
        + geom_tile(color="white", size=0.5)
        + geom_text(aes(label="speedup_label"), size=8)
        + facet_wrap("~panel")
        + scale_fill_gradientn(colors=["#EDEDED", "#E8D4C8", "#EDB880", "#A7B644", "#3D8B5E", "#2E5B6E", "#28272A"])
        + labs(x="Number of trees (T)", y="Max depth (L)", fill="Speedup")
        + theme_paper(figure_size=(5.5, 2.2))
    )
    _save(p, "speedup_heatmap", width=5.5, height=2.2)


# ---------------------------------------------------------------------------
# Figure 3: Method comparison — (a)-(f), split by dataset × framework
# ---------------------------------------------------------------------------
def fig_method_comparison() -> None:
    df = load_grid1()
    surv = df.filter(
        pl.col("dataset").is_in(["support", "flchain"])
        & (pl.col("n_trees") == 500) & (pl.col("max_depth") == 8)
        & (pl.col("horizon") == 16)
    )
    exp = df.filter(
        pl.col("dataset").eq("expedia")
        & (pl.col("n_trees") == 500) & (pl.col("max_depth") == 8)
    )
    ref = pl.concat([surv, exp])
    agg = ref.group_by("dataset", "framework", "method").agg(
        pl.col("median_us").median().alias("latency_us"),
        pl.col("p5_us").median().alias("lat_lo"),
        pl.col("p95_us").median().alias("lat_hi"),
    )
    pdf = agg.to_pandas()
    pdf["method_label"] = pdf["method"].map(METHOD_LABELS)
    pdf["fw"] = pdf["framework"].map(FRAMEWORK_LABELS)
    pdf["ds"] = pdf["dataset"].map(DATASET_LABELS)
    pdf["ds_cat"] = pd.Categorical(pdf["ds"],
        categories=["FLCHAIN", "SUPPORT", "Expedia"], ordered=True)
    pdf["fw_cat"] = pd.Categorical(pdf["fw"],
        categories=["LightGBM", "XGBoost"], ordered=True)

    # Reversed so coord_flip lists TreeWalker at the top of each panel.
    method_order = ["Native", "tl2cgen", "lleaves", "TreeWalker (full walk)",
                    "TreeWalker"]
    pdf["method_label"] = pd.Categorical(pdf["method_label"],
        categories=[m for m in method_order if m in pdf["method_label"].unique()],
        ordered=True)
    pdf = pdf.dropna(subset=["method_label"])

    p = (
        ggplot(pdf, aes(x="method_label", y="latency_us", fill="method_label"))
        + geom_bar(stat="identity")
        + geom_errorbar(aes(ymin="lat_lo", ymax="lat_hi"), width=0.3)
        + facet_grid("fw_cat~ds_cat")
        + scale_y_log10(limits=(20, 1200), breaks=[30, 100, 300, 1000])
        + coord_flip()
        + scale_fill_manual(values=METHOD_COLORS, guide=None)
        + labs(x="", y="Latency (\u00b5s/obs, log scale)")
        + theme_paper(
            figure_size=(5.5, 2.4),
            axis_text_y=element_text(size=7),
        )
    )
    _save(p, "method_comparison", width=5.5, height=2.4)


# ---------------------------------------------------------------------------
# Figure 4: Cross-platform scatter — log-log, marker shapes
# ---------------------------------------------------------------------------
def fig_cross_platform() -> None:
    intel = load_grid1("intel")
    arm = load_grid1("arm")
    join_cols = ["dataset", "framework", "n_trees", "max_depth", "horizon"]

    def _speedups(d):
        d2 = d.with_columns(pl.col("horizon").fill_null(-1))
        return _per_repeat_speedup(d2, join_cols).drop("speedup_lo", "speedup_hi")

    si = _speedups(intel).rename({"speedup": "intel_speedup"})
    sa = _speedups(arm).rename({"speedup": "arm_speedup"})
    merged = si.join(sa, on=join_cols)
    pdf = merged.to_pandas()
    pdf["ds"] = pdf["dataset"].map(DATASET_LABELS)

    from plotnine import geom_abline
    p = (
        ggplot(pdf, aes(x="intel_speedup", y="arm_speedup", color="ds", shape="ds"))
        + geom_point(alpha=0.45, size=1.8)
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="gray")
        + scale_x_log10() + scale_y_log10()
        + scale_color_manual(values=DS_COLORS)
        + scale_shape_manual(values=DS_MARKERS)
        + labs(x="Intel speedup", y="ARM speedup", color="", shape="")
        + theme_paper(figure_size=(3.3, 3.3), legend_position="top")
    )
    _save(p, "cross_platform", width=3.3, height=3.3)


# ---------------------------------------------------------------------------
# Figure 5a: Ablation waterfall — cumulative speedup
# ---------------------------------------------------------------------------
def fig_ablation_waterfall() -> None:
    """Ablation: one-at-a-time cost of disabling each optimisation.

    New blocked-protocol Grid 3 has one-at-a-time ablations only.
    We show the slowdown ratio = (disabled_us / baseline_us) for each flag,
    which answers "how much does removing this optimisation hurt?"
    Ratio > 1 means the optimisation helps; ratio == 1 means no effect.
    """
    df3 = load_grid3()
    flags_order = [
        "disable_precompute", "disable_unsplit", "disable_tree_ordering",
        "disable_prefix_grouping", "disable_monotonic", "disable_bitset_intern",
    ]
    flag_labels = {
        "disable_precompute":     "Precompute",
        "disable_unsplit":        "Unsplit",
        "disable_tree_ordering":  "Tree ordering",
        "disable_prefix_grouping":"Prefix grouping",
        "disable_monotonic":      "Monotonic",
        "disable_bitset_intern":  "Bitset interning",
    }

    # Use anchors present in the new Grid 3 data.
    ds_cfgs = [
        ("FLCHAIN", "flchain", 500, 8, 4),
        ("SUPPORT", "support", 500, 8, 16),
        ("Expedia", "expedia", 500, 8, None),
    ]

    rows = []
    for label, ds, nt, md, h in ds_cfgs:
        anchor = df3.filter(
            (pl.col("dataset") == ds)
            & (pl.col("n_trees") == nt) & (pl.col("max_depth") == md)
        )
        if h is not None:
            anchor = anchor.filter(pl.col("horizon") == h)
        else:
            anchor = anchor.filter(pl.col("horizon").is_null())

        if len(anchor) == 0:
            continue

        # Baseline: all optimisations enabled (all flags == False)
        baseline = anchor
        for f in flags_order:
            baseline = baseline.filter(pl.col(f) == False)  # noqa: E712
        if len(baseline) == 0:
            continue
        base_us = float(baseline["median_us"].median())

        for flag in flags_order:
            # Row with only this one flag disabled
            filt = anchor.filter(pl.col(flag) == True)  # noqa: E712
            for f in flags_order:
                if f != flag:
                    filt = filt.filter(pl.col(f) == False)  # noqa: E712
            if len(filt) == 0:
                continue
            disabled_us = float(filt["median_us"].median())
            # slowdown > 1 means disabling this flag makes things slower
            slowdown = disabled_us / base_us if base_us > 0 else 1.0
            rows.append({
                "ds": label,
                "flag": flag_labels[flag],
                "slowdown": slowdown,
            })

    if not rows:
        print("  ablation_waterfall: no data found", file=sys.stderr)
        return

    pdf = pd.DataFrame(rows)
    pdf["ds"] = pd.Categorical(pdf["ds"],
        categories=["FLCHAIN", "SUPPORT", "Expedia"], ordered=True)
    all_flags = list(flag_labels.values())
    pdf["flag"] = pd.Categorical(pdf["flag"], categories=all_flags, ordered=True)

    p = (
        ggplot(pdf, aes(x="flag", y="slowdown", color="ds",
                        linetype="ds", shape="ds", group="ds"))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + geom_hline(yintercept=1.0, linetype="dashed", color="gray", alpha=0.4)
        + scale_color_manual(values=DS_COLORS)
        + scale_linetype_manual(values=DS_LINETYPES)
        + scale_shape_manual(values=DS_MARKERS)
        + labs(x="Disabled optimisation",
               y="Slowdown (disabled / baseline)",
               color="", linetype="", shape="")
        + theme_paper(
            figure_size=(5.5, 2.475),
            axis_text_x=element_text(rotation=25, ha="right"),
            legend_position="top",
        )
    )
    _save(p, "ablation_waterfall", width=5.5, height=2.475)


# ---------------------------------------------------------------------------
# Figure 5b: Ablation node visits — grouped bars, alpha for on/off
# ---------------------------------------------------------------------------
def load_grid3_stats(arch: str = "intel") -> pl.DataFrame:
    path = DATA / f"grid3_stats_{arch}.csv"
    if not path.exists():
        return pl.DataFrame()
    return _cast_horizon(pl.read_csv(path))

def fig_ablation_nodes() -> None:
    df3 = load_grid3_stats()
    if df3.height == 0:
        print("  ablation_nodes: SKIPPED (no grid3_stats CSV)", file=sys.stderr)
        return
    flags_order = [
        "disable_precompute", "disable_unsplit", "disable_tree_ordering",
        "disable_prefix_grouping", "disable_monotonic", "disable_bitset_intern",
    ]

    ds_panel = [
        ("flchain", 500, 8, 16, 1304, "FLCHAIN"),
        ("support", 500, 8, 16, 1774, "SUPPORT"),
        ("expedia", 500, 8, None, 10000, "Expedia"),
    ]

    cols = ["constant_steps", "unsplit_skips", "recursive_calls", "leaf_hits"]
    col_labels = {"constant_steps": "Const steps", "unsplit_skips": "Unsplit skips",
                  "recursive_calls": "Recursive calls", "leaf_hits": "Leaf hits"}

    rows = []
    for ds, nt, md, h, n_obs, label in ds_panel:
        a = df3.filter(
            (pl.col("dataset") == ds)
            & (pl.col("n_trees") == nt) & (pl.col("max_depth") == md)
        )
        if h is not None:
            a = a.filter(pl.col("horizon") == h)
        else:
            a = a.filter(pl.col("horizon").is_null())

        for state_name, state_val in [("Enabled", 0), ("Disabled", 1)]:
            filt = a
            for f in flags_order:
                filt = filt.filter(pl.col(f) == state_val)
            if filt.height == 0:
                continue
            for c in cols:
                val = filt[c].median()
                if val is None:
                    continue
                rows.append({
                    "ds": label,
                    "state": state_name,
                    "category": col_labels[c],
                    "visits": float(val) / n_obs,
                })
    pdf = pd.DataFrame(rows)
    pdf["category"] = pd.Categorical(pdf["category"],
        categories=["Const steps", "Unsplit skips", "Recursive calls", "Leaf hits"],
        ordered=True)
    # Facet by state for grayscale clarity
    pdf["state"] = pd.Categorical(pdf["state"],
        categories=["Enabled", "Disabled"], ordered=True)
    state_labels = {"Enabled": "(a) All optimizations enabled",
                    "Disabled": "(b) All optimizations disabled"}
    pdf["ds"] = pd.Categorical(pdf["ds"],
        categories=["FLCHAIN", "SUPPORT", "Expedia"], ordered=True)
    pdf["state_panel"] = pdf["state"].map(state_labels)
    pdf["state_panel"] = pd.Categorical(pdf["state_panel"],
        categories=[state_labels["Enabled"], state_labels["Disabled"]], ordered=True)

    p = (
        ggplot(pdf, aes(x="category", y="visits", fill="ds"))
        + geom_bar(stat="identity", position=position_dodge(width=0.8), width=0.7)
        + facet_wrap("~state_panel")
        + scale_fill_manual(values=DS_COLORS)
        + labs(x="", y="Node visits / obs", fill="")
        + theme_paper(
            figure_size=(5.5, 2.475),
            legend_position="top",
            axis_text_x=element_text(rotation=30, ha="right"),
        )
    )
    _save(p, "ablation_nodes", width=5.5, height=2.475)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Figure 7: Depth sensitivity — (a)/(b), log x, ribbons
# ---------------------------------------------------------------------------
def fig_depth_sensitivity() -> None:
    df = load_grid1()
    surv = df.filter(
        pl.col("dataset").is_in(["support", "flchain"])
        & (pl.col("n_trees") == 500) & (pl.col("horizon") == 16)
        & pl.col("method").is_in(["treewalker", "treewalker_fullwalk", "treewalker_full", "treewalker_baseline"])
    )
    exp = df.filter(
        pl.col("dataset").eq("expedia")
        & (pl.col("n_trees") == 500)
        & pl.col("method").is_in(["treewalker", "treewalker_fullwalk", "treewalker_full", "treewalker_baseline"])
    )
    both = pl.concat([surv, exp])
    merged = _per_repeat_speedup(both, ["dataset", "framework", "max_depth"])
    pdf = merged.to_pandas()
    pm = _panel_label(["FLCHAIN", "SUPPORT", "Expedia"])
    pdf["panel"] = pdf["dataset"].map(DATASET_LABELS).map(pm)
    pdf["panel"] = pd.Categorical(pdf["panel"],
        categories=[pm[k] for k in ["FLCHAIN", "SUPPORT", "Expedia"]], ordered=True)
    pdf["fw"] = pdf["framework"].map(FRAMEWORK_LABELS)

    p = (
        ggplot(pdf, aes(x="max_depth", y="speedup", color="fw", fill="fw"))
        + geom_ribbon(aes(ymin="speedup_lo", ymax="speedup_hi"), alpha=0.12, color=None)
        + geom_line(aes(linetype="fw"), size=1)
        + geom_point(aes(shape="fw"), size=2.5)
        + geom_hline(yintercept=1.0, linetype="dashed", color="gray", alpha=0.5)
        + facet_wrap("~panel")
        + scale_x_log10(breaks=[2, 4, 8, 16], labels=["2", "4", "8", "16"])
        + scale_color_manual(values=FW_COLORS)
        + scale_fill_manual(values=FW_COLORS, guide=None)
        + scale_linetype_manual(values=FW_LINETYPES)
        + scale_shape_manual(values=FW_SHAPES)
        + labs(x="Max tree depth (L)", y="Speedup vs full walk",
               color="", linetype="", shape="")
        + theme_paper(figure_size=(5.5, 2.475), legend_position="top")
    )
    _save(p, "depth_sensitivity", width=5.5, height=2.475)


# ---------------------------------------------------------------------------
# Figure 8: Tree count scaling — (a)/(b), log x, ribbons
# ---------------------------------------------------------------------------
def fig_ntrees_scaling() -> None:
    df = load_grid1()
    surv = df.filter(
        pl.col("dataset").is_in(["support", "flchain"])
        & (pl.col("max_depth") == 8) & (pl.col("horizon") == 16)
        & pl.col("method").is_in(["treewalker", "treewalker_fullwalk", "treewalker_full", "treewalker_baseline"])
    )
    exp = df.filter(
        pl.col("dataset").eq("expedia")
        & (pl.col("max_depth") == 8)
        & pl.col("method").is_in(["treewalker", "treewalker_fullwalk", "treewalker_full", "treewalker_baseline"])
    )
    both = pl.concat([surv, exp])
    merged = _per_repeat_speedup(both, ["dataset", "framework", "n_trees"])
    pdf = merged.to_pandas()
    pm = _panel_label(["FLCHAIN", "SUPPORT", "Expedia"])
    pdf["panel"] = pdf["dataset"].map(DATASET_LABELS).map(pm)
    pdf["panel"] = pd.Categorical(pdf["panel"],
        categories=[pm[k] for k in ["FLCHAIN", "SUPPORT", "Expedia"]], ordered=True)
    pdf["fw"] = pdf["framework"].map(FRAMEWORK_LABELS)

    p = (
        ggplot(pdf, aes(x="n_trees", y="speedup", color="fw", fill="fw"))
        + geom_ribbon(aes(ymin="speedup_lo", ymax="speedup_hi"), alpha=0.12, color=None)
        + geom_line(aes(linetype="fw"), size=1)
        + geom_point(aes(shape="fw"), size=2.5)
        + geom_hline(yintercept=1.0, linetype="dashed", color="gray", alpha=0.5)
        + facet_wrap("~panel")
        + scale_x_log10(breaks=[50, 500, 1000, 2000],
                        labels=["50", "500", "1K", "2K"])
        + scale_color_manual(values=FW_COLORS)
        + scale_fill_manual(values=FW_COLORS, guide=None)
        + scale_linetype_manual(values=FW_LINETYPES)
        + scale_shape_manual(values=FW_SHAPES)
        + labs(x="Number of trees (T)", y="Speedup vs full walk",
               color="", linetype="", shape="")
        + theme_paper(figure_size=(5.5, 2.475), legend_position="top")
    )
    _save(p, "ntrees_scaling", width=5.5, height=2.475)


def main() -> None:
    print("Generating figures...", file=sys.stderr)
    fig_horizon_amortization()
    fig_speedup_heatmap()
    fig_method_comparison()
    fig_cross_platform()
    fig_ablation_waterfall()
    fig_ablation_nodes()
    fig_depth_sensitivity()
    fig_ntrees_scaling()
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
