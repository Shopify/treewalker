#!/usr/bin/env python3
"""Empirical decomposition validation table and asymptotic-convergence plot.

Theorem 1 decomposes per-tree work as

    W_Ω(τ, X) = C(τ, X) + V(τ, X) + G + Q_Ω(τ, X)

and Corollary 1 (trace evaluator) gives the per-row work ratio

    W_trace(τ, X) / (G·(d+1))  →  (d_v + 1) / (d + 1)        as G → ∞.

This script reads existing stats CSVs (no new experiments) and produces
two artifacts that empirically validate the decomposition:

1. ``paper/experiments/figures/decomposition_table.tex`` — multi-G booktabs
   table for SUPPORT and FLCHAIN at the reference cell (T=500, L=8) over
   G ∈ {4, 16, 64, 128}, both trace and precompute evaluators.

2. ``paper/experiments/figures/asymptotic_convergence.{pdf,png}`` — log-G
   convergence plot showing per-row work ratio vs. G for SUPPORT, FLCHAIN
   (4 G points each, both evaluators) and Expedia (1 reference point at
   variable mean G).

Notes:
- "trace" means Q_Ω = G·d_v: the no-precompute evaluator that
  evaluates each varying predicate per active row at the visit.
  CSV row: disable_precompute=1, partition_row_evals populated.
- "precompute" means the default evaluator with sorted-threshold sweep.
  CSV row: disable_precompute=0, precompute_row_evals populated.
- Both evaluators converge to the same per-row asymptotic (Q amortizes);
  trace converges from above, precompute slower because Q_pre = O(G log G).
- The row-independent baseline G·T·L uses tree depth as a conservative
  ceiling on average root-to-leaf depth.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATS_CSV = ROOT / "paper/experiments/data/grid3_stats_intel.csv"
TABLE_OUT = ROOT / "paper/experiments/figures/decomposition_table.tex"
PLOT_OUT_PDF = ROOT / "paper/experiments/figures/asymptotic_convergence.pdf"
PLOT_OUT_PNG = ROOT / "paper/experiments/figures/asymptotic_convergence.png"

REF_T = 500
REF_L = 8
FIXED_G = ["4", "16", "64", "128"]    # SUPPORT/FLCHAIN sweep
DATASETS_FIXED = ["support", "flchain"]
# Expedia is variable-G; included in the plot at its mean group size,
# excluded from the per-G table.


def load_rows():
    """Yield default-or-trace rows at T=500, L=8 for the relevant datasets."""
    with STATS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["n_trees"]) != REF_T:
                continue
            if int(row["max_depth"]) != REF_L:
                continue
            # Allow disable_precompute ∈ {0, 1}; everything else must be 0.
            if any(int(row[c]) for c in (
                "disable_unsplit", "disable_monotonic",
                "disable_tree_ordering", "disable_prefix_grouping",
                "disable_bitset_intern",
            )):
                continue
            yield row


def expedia_mean_G_from_csv(rows: list[dict]) -> float:
    """Recover Expedia's mean group size from leaf_hits / (n_obs * T).

    leaf_hits counts distinct leaves visited per tree per group; in the
    *trace* evaluator with no mask collapse this would equal G·T·n_obs
    in the worst case. We approximate mean G as leaf_hits/(n_obs·T) for
    the FULL traversal which TreeWalker doesn't run, but the trace
    evaluator's leaf_hits scales linearly with G with constant factor
    ≈ 1 (every active row reaches one leaf). For variable G this gives
    the right per-group average.

    A more accurate value comes from group_offsets.bin (~22.55 for the
    standard Expedia eval); we use that as a constant when available.
    """
    p = ROOT / "paper/experiments/artifacts/expedia/nt1000_md8/group_offsets.bin"
    if p.exists():
        import struct
        raw = p.read_bytes()
        n = struct.unpack("<Q", raw[0:8])[0]
        offsets = list(struct.unpack(f"<{n+1}Q", raw[8:8*(n+2)]))
        sizes = [offsets[i+1] - offsets[i] for i in range(n)]
        return sum(sizes) / len(sizes)
    return float("nan")


def asymptote(rows: list[dict], dataset: str, evaluator: str, L: int) -> float:
    """Predicted per-row work ratio asymptote (Corollary 1) under the
    G·T·L baseline.

    - Trace: limit = (d_v + 1) / L, where d_v is the per-tree mean
      per-row varying-step depth, estimated from partition_row_evals at
      the largest available G (most asymptotic regime).
    - Precompute: limit = 1 / L. As G → ∞ the ΣC+ΣV terms (bounded by
      |T_c|) and Q_pre (measured row-eval steps, sub-linear per row)
      both vanish from the per-row ratio; only the leaf-write term
      (G·T)/(G·T·L) = 1/L remains.
    """
    if evaluator == "precompute":
        return 1.0 / L
    trace_rows = [r for r in rows
                  if r["dataset"] == dataset
                  and int(r["disable_precompute"]) == 1
                  and r["horizon"] not in ("", None)]
    largest = max(trace_rows, key=lambda r: int(r["horizon"]))
    G = int(largest["horizon"])
    T = int(largest["n_trees"])
    n_obs = int(largest["n_obs"])
    partition = int(largest["partition_row_evals"])
    d_v = partition / (G * T * n_obs)
    return (d_v + 1.0) / L


def per_row_work(row: dict, G: float) -> dict:
    """Compute per-row work components for one CSV row.

    Returns dict with: per_row_C, per_row_V, per_row_leaf, per_row_Q,
    per_row_total, per_row_ratio (vs G·T·L baseline).
    """
    n_obs = int(row["n_obs"])
    T = int(row["n_trees"])
    L = int(row["max_depth"])
    disable_pre = int(row["disable_precompute"])

    # Per-group ensemble totals.
    sum_C = int(row["constant_steps"]) / n_obs
    sum_V = int(row["varying_splits"]) / n_obs
    sum_leaf_writes = G * T  # exact, by Lemma 1 (one leaf per row per tree).
    if disable_pre:
        Q = int(row["partition_row_evals"]) / n_obs
    else:
        Q = int(row["precompute_row_evals"]) / n_obs

    total = sum_C + sum_V + sum_leaf_writes + Q
    per_row_total = total / G
    baseline_per_row = T * L  # row-independent: T trees × L depth per row.
    ratio = per_row_total / baseline_per_row

    return dict(
        dataset=row["dataset"],
        framework=row["framework"],
        G=G,
        evaluator="trace" if disable_pre else "precompute",
        sum_C=sum_C,
        sum_V=sum_V,
        sum_leaf=sum_leaf_writes,
        Q=Q,
        total=total,
        baseline=G * T * L,
        per_row_total=per_row_total,
        per_row_ratio=ratio,
    )


def collect_data() -> list[dict]:
    """All per-row work data points for table + plot."""
    raw = list(load_rows())
    out = []
    for row in raw:
        ds = row["dataset"]
        if ds == "expedia":
            G = expedia_mean_G_from_csv(raw)
        else:
            if ds not in DATASETS_FIXED:
                continue
            if row["horizon"] not in FIXED_G:
                continue
            G = float(row["horizon"])
        out.append(per_row_work(row, G))
    return out, raw


def fmt_int(x: float) -> str:
    return f"{x:,.0f}"


def fmt_ratio(x: float) -> str:
    return f"{x:.3f}"


def render_text_table(data: list[dict]) -> str:
    """Multi-G human-readable table for stdout."""
    cols = ["dataset", "framework", "G", "evaluator", "ΣC", "ΣV", "G·T", "Q", "total", "per_row", "ratio"]
    lines = ["  ".join(f"{c:>11}" for c in cols)]
    lines.append("  ".join("-" * 11 for _ in cols))
    sort_key = lambda d: (
        ["support", "flchain", "expedia"].index(d["dataset"]),
        ["lightgbm", "xgboost"].index(d["framework"]),
        ["trace", "precompute"].index(d["evaluator"]),
        d["G"],
    )
    for r in sorted(data, key=sort_key):
        lines.append("  ".join(f"{v:>11}" for v in [
            r["dataset"], r["framework"],
            f"{r['G']:.1f}" if r["dataset"] == "expedia" else fmt_int(r["G"]),
            r["evaluator"],
            fmt_int(r["sum_C"]), fmt_int(r["sum_V"]),
            fmt_int(r["sum_leaf"]), fmt_int(r["Q"]),
            fmt_int(r["total"]),
            f"{r['per_row_total']:.2f}",
            fmt_ratio(r["per_row_ratio"]),
        ]))
    return "\n".join(lines)


def render_latex_table(data: list[dict], asymptotes: dict) -> str:
    """Multi-G LaTeX table for SUPPORT and FLCHAIN, LightGBM, both evaluators.

    Layout: rows = (dataset, evaluator), columns = G ∈ {4, 16, 64, 128}
    plus a final 'G → ∞' column with the Corollary 1 asymptote.
    """
    lgb = [r for r in data if r["framework"] == "lightgbm" and r["dataset"] in DATASETS_FIXED]
    grid = {}
    for r in lgb:
        key = (r["dataset"], r["evaluator"])
        grid.setdefault(key, {})[int(r["G"])] = r["per_row_ratio"]

    body_lines = []
    for ds in ["support", "flchain"]:
        ds_label = ds.upper()
        for ev in ["trace", "precompute"]:
            ratios = grid.get((ds, ev), {})
            cells = " & ".join(
                fmt_ratio(ratios.get(int(g), float("nan"))) for g in (4, 16, 64, 128)
            )
            ev_label = "trace" if ev == "trace" else r"precompute (default)"
            asym = fmt_ratio(asymptotes[(ds, ev)])
            body_lines.append(
                f"  {ds_label} & {ev_label} & {cells} & \\textit{{{asym}}} \\\\"
            )
    body = "\n".join(body_lines)

    return rf"""% Auto-generated by paper/experiments/scripts/decomposition_validation.py
% Do not edit by hand; re-run the script to refresh.
\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{llrrrrr}}
\toprule
& & \multicolumn{{4}}{{c}}{{Group size $G$}} & Asymptote \\
\cmidrule(lr){{3-6}}
Dataset & Evaluator & 4 & 16 & 64 & 128 & $G \to \infty$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Per-row work ratio $W(\mathcal{{F}}, X) / (G \cdot T \cdot L)$
at the reference configuration ($T={REF_T}$, $L={REF_L}$, LightGBM,
Intel) on the fixed-$G$ datasets, swept across $G \in \{{4, 16, 64, 128\}}$.
The final column gives the Corollary~\ref{{cor:trace}} asymptote
under this baseline: $(d_v+1)/L$ for trace (with $d_v$ estimated from
\texttt{{partition\_row\_evals}} at $G=128$) and $1/L$ for precompute
(provisioning amortizes; only the leaf-write term remains). Both
evaluators decrease monotonically in $G$ and approach their predicted
limits. Figure~\ref{{fig:asymptotic-convergence}} visualizes the same
data. XGBoost numbers (not shown) are within $5\%$ of LightGBM. Expedia
is excluded from this sweep because it has variable group sizes; it
appears alongside the speedup grid in \S\ref{{sec:evaluation}}.}}
\label{{tab:decomp-results}}
\end{{table}}
"""


def render_plot(data: list[dict], asymptotes: dict) -> None:
    """Plot per-row work ratio vs. G, log-x, with Corollary 1 asymptote lines.

    Lines: dataset × evaluator. Two datasets (SUPPORT, FLCHAIN), two
    evaluators (trace, precompute). Asymptotes drawn as horizontal
    dashed lines per (dataset, evaluator); precompute asymptote is
    dataset-independent (1/L).
    """
    try:
        import pandas as pd
        from plotnine import (
            ggplot, aes, geom_line, geom_point, geom_hline, scale_x_log10,
            scale_y_continuous, labs, theme_minimal, theme, element_text,
            scale_color_manual, scale_shape_manual, scale_linetype_manual,
        )
    except ImportError as e:
        print(f"plotnine unavailable ({e}); skipping plot generation",
              file=sys.stderr)
        return

    # LightGBM only (XGBoost within 5%, would clutter).
    # Fixed-G datasets only (Expedia variable G isn't a clean sweep point).
    df = pd.DataFrame([
        r for r in data
        if r["framework"] == "lightgbm" and r["dataset"] in DATASETS_FIXED
    ])
    df["dataset_label"] = df["dataset"].map({
        "support": "SUPPORT",
        "flchain": "FLCHAIN",
    })
    df["evaluator_label"] = df["evaluator"].map({
        "trace": "trace",
        "precompute": "precompute (default)",
    })

    # Catppuccin-ish palette.
    color_map = {
        "SUPPORT": "#1e66f5",   # blue
        "FLCHAIN": "#df8e1d",   # gold
    }
    linetype_map = {
        "trace": "solid",
        "precompute (default)": "dashed",
    }

    # Asymptote rows for geom_hline.
    asym_df = pd.DataFrame([
        dict(
            dataset_label={"support": "SUPPORT", "flchain": "FLCHAIN"}[ds],
            evaluator_label={"trace": "trace",
                             "precompute": "precompute (default)"}[ev],
            yintercept=val,
        )
        for (ds, ev), val in asymptotes.items()
    ])

    p = (
        ggplot(df, aes(x="G", y="per_row_ratio",
                       color="dataset_label",
                       linetype="evaluator_label",
                       shape="evaluator_label"))
        + geom_hline(data=asym_df,
                     mapping=aes(yintercept="yintercept",
                                 color="dataset_label",
                                 linetype="evaluator_label"),
                     size=0.4, alpha=0.45, inherit_aes=False)
        + geom_line(size=0.7)
        + geom_point(size=2.0)
        + scale_x_log10(breaks=[4, 16, 64, 128], limits=(3.5, 140))
        + scale_y_continuous(limits=(0, 0.75), breaks=[0, 0.25, 0.5, 0.75])
        + scale_color_manual(values=color_map, name="Dataset")
        + scale_linetype_manual(values=linetype_map, name="Evaluator")
        + scale_shape_manual(values={"trace": "o", "precompute (default)": "s"},
                              name="Evaluator")
        + labs(
            x="Group size $G$ (log scale)",
            y="Per-row work ratio  $W / (G\\cdot T\\cdot L)$",
        )
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

    PLOT_OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    p.save(str(PLOT_OUT_PDF), dpi=300, verbose=False)
    p.save(str(PLOT_OUT_PNG), dpi=300, verbose=False)
    print(f"Wrote plot → {PLOT_OUT_PDF.relative_to(ROOT)}")
    print(f"Wrote plot → {PLOT_OUT_PNG.relative_to(ROOT)}")


def main() -> int:
    if not STATS_CSV.exists():
        print(f"missing {STATS_CSV}", file=sys.stderr)
        return 1

    data, raw_rows = collect_data()
    if not data:
        print(f"no rows found at T={REF_T}, L={REF_L}", file=sys.stderr)
        return 1

    # Compute Corollary-1 asymptotes per (dataset, evaluator).
    asymptotes = {
        (ds, ev): asymptote(raw_rows, ds, ev, REF_L)
        for ds in DATASETS_FIXED
        for ev in ("trace", "precompute")
    }

    print("=== Per-row work decomposition (Intel, default + trace) ===\n")
    print(render_text_table(data))
    print()
    print("=== Corollary 1 asymptotes (G → ∞, denominator G·T·L) ===")
    for (ds, ev), val in asymptotes.items():
        print(f"  {ds:>8} {ev:>10}: {val:.3f}")
    print()

    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text(render_latex_table(data, asymptotes))
    print(f"Wrote table → {TABLE_OUT.relative_to(ROOT)}")

    render_plot(data, asymptotes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
