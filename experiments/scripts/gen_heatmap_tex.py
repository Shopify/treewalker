#!/usr/bin/env python3
"""Generate experiments/figures/speedup_heatmap.tex from grid1 Intel data.

Mirrors the data logic of plot.py::fig_speedup_heatmap (line ~217) so the
TikZ figure in the paper stays in sync with the benchmark CSV.  Each cell
is coloured by interpolating the same 7-stop palette that plotnine uses,
labelled with two-decimal-place speedups, and given black or white text
based on background luminance for readability.

Usage:
    uv run python3 experiments/scripts/gen_heatmap_tex.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "figures" / "speedup_heatmap.tex"

# Palette derived from plot.py:251 but REVERSED so that low t (low speedup)
# maps to a light colour and high t (high speedup) maps to a dark colour.
# The caption reads "darker = higher speedup" — this ordering makes that true.
PALETTE = ["#EDEDED", "#E8D4C8", "#EDB880", "#A7B644",
           "#3D8B5E", "#2E5B6E", "#28272A"]

N_COLS = [50, 500, 1000, 2000]
N_ROWS = [2, 4, 8, 16]  # bottom-to-top
PANELS = [("flchain", "(a)~FLCHAIN"),
          ("support", "(b)~SUPPORT"),
          ("expedia", "(c)~Expedia")]

# -- Colour helpers --------------------------------------------------------

def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def interp(t: float) -> tuple[int, int, int]:
    """Linear RGB interpolation across PALETTE, matching scale_fill_gradientn."""
    t = max(0.0, min(1.0, t))
    n = len(PALETTE) - 1
    pos = t * n
    lo = int(pos)
    if lo >= n:
        return hex_to_rgb(PALETTE[-1])
    hi = lo + 1
    frac = pos - lo
    clo = hex_to_rgb(PALETTE[lo])
    chi = hex_to_rgb(PALETTE[hi])
    return tuple(int(round(clo[k] * (1 - frac) + chi[k] * frac)) for k in range(3))


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    def lin(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_color_for(rgb: tuple[int, int, int]) -> str:
    """Pick the higher-WCAG-contrast of black or white against the cell bg."""
    y = relative_luminance(rgb)
    contrast_white = 1.05 / (y + 0.05)
    contrast_black = (y + 0.05) / 0.05
    return "black" if contrast_black >= contrast_white else "white"


# -- Data ------------------------------------------------------------------

def load_speedups() -> dict[tuple[str, int, int], float]:
    df = pl.read_csv(DATA / "grid1_results_intel.csv")
    if df.schema.get("horizon") == pl.Utf8:
        df = df.with_columns(pl.col("horizon").cast(pl.Int64, strict=False))

    tw_methods = ["treewalker", "treewalker_fullwalk",
                  "treewalker_full", "treewalker_baseline"]
    surv = df.filter(
        pl.col("dataset").is_in(["support", "flchain"])
        & (pl.col("horizon") == 16)
        & pl.col("method").is_in(tw_methods)
    )
    exp = df.filter(
        pl.col("dataset").eq("expedia")
        & pl.col("method").is_in(tw_methods)
    )
    both = pl.concat([surv, exp])

    # CSV naming: treewalker_fullwalk = SLOW full-walk reference (paper:
    # TreeWalker (full walk)); treewalker = optimized TreeWalker (paper:
    # TreeWalker). Algorithmic speedup = slow / fast (>1 when TW wins).
    # Accept legacy names for back-compat.
    methods_present = both["method"].unique().to_list()
    slow_name = "treewalker_fullwalk" if "treewalker_fullwalk" in methods_present else "treewalker_baseline"
    fast_name = "treewalker" if "treewalker" in methods_present else "treewalker_full"
    slow = both.filter(pl.col("method") == slow_name).rename({"median_us": "slow_us"})
    fast = both.filter(pl.col("method") == fast_name).rename({"median_us": "fast_us"})
    # Match plot.py _per_repeat_speedup: join keys exclude horizon, which is
    # null for Expedia (null-null matches aren't treated as equal by polars).
    join_on = ["dataset", "framework", "n_trees", "max_depth"]
    joined = (slow.select(join_on + ["slow_us"])
                .join(fast.select(join_on + ["fast_us"]), on=join_on)
                .with_columns((pl.col("slow_us") / pl.col("fast_us")).alias("speedup")))
    # Median across repeats, mean across frameworks — matches plot.py.
    agg = joined.group_by(["dataset", "framework", "n_trees", "max_depth"]).agg(
        pl.col("speedup").median()
    )
    final = agg.group_by(["dataset", "n_trees", "max_depth"]).agg(
        pl.col("speedup").mean()
    )
    return {(r["dataset"], r["n_trees"], r["max_depth"]): r["speedup"]
            for r in final.iter_rows(named=True)}


# -- TeX emission ----------------------------------------------------------

def emit() -> str:
    cells = load_speedups()
    all_vals = list(cells.values())
    vmin, vmax = min(all_vals), max(all_vals)
    rng = vmax - vmin if vmax > vmin else 1.0

    lines: list[str] = []
    w = lines.append

    w("% Auto-generated by experiments/scripts/gen_heatmap_tex.py.")
    w("% Re-run after updating grid1_results_intel.csv; do not hand-edit.")
    w("% Cells are coloured by interpolating the 7-stop palette from plot.py;")
    w("% text colour (black/white) is chosen per-cell for readable contrast.")
    w("")
    w(r"\begin{tikzpicture}[")
    w(r"  x=7.5mm, y=7mm,")
    w(r"  font=\footnotesize,")
    w(r"  cell/.style={")
    w(r"    rectangle, minimum width=7.5mm, minimum height=7mm,")
    w(r"    draw=white, line width=0.4pt,")
    w(r"    inner sep=0pt, anchor=center,")
    w(r"  },")
    w(r"  rowlabel/.style={anchor=east, font=\scriptsize, inner sep=2pt},")
    w(r"  collabel/.style={anchor=north, font=\scriptsize, inner sep=2pt},")
    w(r"  ptitle/.style={font=\small, anchor=south},")
    w(r"  axlab/.style={font=\scriptsize},")
    w(r"]")
    w("")
    w(f"% Speedup range across all cells: [{vmin:.4f}, {vmax:.4f}]")
    w("")

    panel_xshifts = [None, "3.7cm", "7.4cm"]
    for idx, (ds, title) in enumerate(PANELS):
        w(f"% ---------------- Panel {title} ----------------")
        if idx > 0:
            w(rf"\begin{{scope}}[xshift={panel_xshifts[idx]}]")
            indent = "  "
        else:
            indent = ""
        w(rf"{indent}\node[ptitle] at (1.5, 3.65) {{{title}}};")
        for r_idx, L in enumerate(N_ROWS):          # r_idx=0 at bottom
            for c_idx, T in enumerate(N_COLS):
                val = cells[(ds, T, L)]
                t = (val - vmin) / rng
                rgb = interp(t)
                tcol = text_color_for(rgb)
                fill = f"{{rgb,255:red,{rgb[0]};green,{rgb[1]};blue,{rgb[2]}}}"
                w(rf"{indent}\node[cell, fill={fill}, text={tcol}] "
                  rf"at ({c_idx}, {r_idx}) {{{val:.2f}}};")
        # Column labels on every panel.
        for c_idx, T in enumerate(N_COLS):
            w(rf"{indent}\node[collabel] at ({c_idx}, -0.5) {{{T}}};")
        # Row labels + y-axis label on first panel only.
        if idx == 0:
            for r_idx, L in enumerate(N_ROWS):
                w(rf"{indent}\node[rowlabel] at (-0.5, {r_idx}) {{{L}}};")
            w(rf"{indent}\node[rotate=90, axlab] at (-1.35, 1.5) "
              r"{Max depth ($L$)};")
        if idx > 0:
            w(r"\end{scope}")
        w("")

    # Shared x-axis label — centred under all three panels.
    w(r"\node[axlab] at (6.43, -1.4) {Number of trees ($T$)};")
    w("")

    # -- Colorbar: continuous vertical gradient built from thin rectangles.
    # Colorbar height matches the cell grid (28 mm = 4 cells × 7 mm).
    w("% ---------------- Colorbar ----------------")
    w(r"\begin{scope}[xshift=11.1cm, yshift=-7mm, x=1cm, y=1cm]")
    w(r"  \node[font=\scriptsize, anchor=south west] at (0, 3.65) {Speedup};")
    # Build a smooth gradient with N bands.
    N_BANDS = 120
    bar_h_cm = 3.5    # total colorbar height in cm
    bar_w_mm = 5
    band_h_cm = bar_h_cm / N_BANDS
    w(f"  % Gradient: {N_BANDS} thin bands = smooth vertical fill.")
    for i in range(N_BANDS):
        t = i / (N_BANDS - 1)
        rgb = interp(t)
        y = i * band_h_cm
        fill = f"{{rgb,255:red,{rgb[0]};green,{rgb[1]};blue,{rgb[2]}}}"
        w(rf"  \fill[fill={fill}] (0, {y:.4f}) rectangle "
          rf"({bar_w_mm}mm, {y + band_h_cm:.4f});")
    # Tick labels along the bar at nice round values.
    w(r"  % Tick labels.")
    nice_ticks = [v for v in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                  if vmin - 0.05 <= v <= vmax + 0.05]
    for v in nice_ticks:
        t = (v - vmin) / rng
        y = t * bar_h_cm
        w(rf"  \node[anchor=west, font=\scriptsize, inner sep=1pt] "
          rf"at ({bar_w_mm + 0.5}mm, {y:.4f}) {{{v:.1f}}};")
    w(r"  % Outline.")
    w(rf"  \draw[line width=0.3pt] (0, 0) rectangle ({bar_w_mm}mm, {bar_h_cm});")
    w(r"\end{scope}")
    w("")
    w(r"\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def main() -> None:
    tex = emit()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"Wrote {OUT} ({len(tex)} bytes)")


if __name__ == "__main__":
    main()
