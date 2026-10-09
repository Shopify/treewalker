"""The paper's in-text numbers, regenerated from the final runs (`results`).

The same items as v1's paper_numbers.py (at the neurips2026 tag), in its order and
with its paper locations, computed from the new runs and printed as new values only
(the user's decision, 2026-10-08). The definitions follow the new estimands:

- an "algorithmic speedup" is the full walk's time over TreeWalker's, and a
  baseline's speedup its time over TreeWalker's, row-weighted over the same groups
  (v1: ratios of median latencies); LightGBM native is corrected for the sentinel's
  drift; XGBoost native is its headline (the faster mode where it shows two);
- a latency is the p50 per group;
- "v1's cells" are the 544 per machine v1's grid had: survival panels at horizons
  1 to 128 and Expedia's sessions, T in {50, 500, 1000, 2000}, L in {2, 4, 8, 16}.
  Counts the paper gives over its grid (QuickScorer's coverage, §6's boundary
  cases) are reported over v1's cells and over the whole factorial (1,134 cells);
- Table 3's trace-evaluator columns are dropped (`tables`); Table 11 compares the
  128-row chunks with whole groups at G = 256, 512 and 1024; App. A.6 reports the
  stopping rules and the intervals' half-widths instead of v1's CV estimate.
"""

from __future__ import annotations

import statistics as st
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl

from . import figures, results
from . import formats as fm

SURV, DS, FW = ("support", "flchain"), ("support", "flchain", "expedia"), ("lightgbm", "xgboost")
TS, LS = (50, 500, 1000, 2000), (2, 4, 8, 16)


def r(x: float, nd: int) -> str:
    """Rounded half up to ``nd`` places, as the paper prints."""
    q = Decimal(1).scaleb(-nd) if nd else Decimal(1)
    return str(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def rng(lo: float, hi: float, nd: int) -> str:
    return f"{r(lo, nd)}-{r(hi, nd)}"


def cid(ds: str, fw: str, t: int, depth: int, h: int | None = None) -> str:
    if ds == "expedia":
        return f"expedia/nt{t}_md{depth}/{fw}/sessions"
    return f"{ds}/nt{t}_md{depth}_h{h}/{fw}/panel"


def ref_cell(ds: str, fw: str, t: int = 500, depth: int = 8, h: int = 16) -> str:
    return cid(ds, fw, t, depth, None if ds == "expedia" else h)


class Grid:
    """Speedups by (machine, cell): per method, the row of `results.speedups`
    (XGBoost native's headline; its process-0 row as ``xgboost_native_p0``)."""

    def __init__(self, df: Any) -> None:
        self.rows: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        for row in df.filter(pl.col("mode") == "serving").iter_rows(named=True):
            m = row["method"]
            if m == "xgboost_native":
                m = "xgboost_native" if row["variant"] == "headline" else "xgboost_native_p0"
            elif m == "treewalker" and row["variant"] != "all-on":
                continue
            self.rows.setdefault((row["arch"], row["cell"]), {})[m] = row

    def __call__(self, arch: str, cell: str) -> dict[str, dict[str, Any]]:
        return self.rows[(arch, cell)]

    def cells(self, arch: str) -> dict[str, dict[str, dict[str, Any]]]:
        return {cell: m for (a, cell), m in self.rows.items() if a == arch}

    def v1(self, arch: str) -> dict[str, dict[str, dict[str, Any]]]:
        """v1's 544 cells on ``arch``."""
        out = {}
        for (a, cell), methods in self.rows.items():
            d = results.design(cell)
            if a == arch and d["family"] in ("panel", "sessions") and d["replicate"] is None:
                out[cell] = methods
        return out


def _qs_reason(reason: str) -> str:
    for key, label in (
        ("categorical", "categorical splits"),
        ("missing", "missing values"),
        ("leaves", "more than 128 leaves"),
        ("differ", "outputs differ (f32 thresholds)"),
    ):
        if key in reason:
            return label
    return reason


def alg(methods: dict[str, dict[str, Any]]) -> float:
    return float(methods["treewalker_fullwalk"]["speedup"])


def g_label(cell: str) -> str:
    g = results.design(cell)["G"]
    return "sessions" if g is None else str(g)


def collect(runs: Path) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []

    def say(where: str, what: str, value: str) -> None:
        out.append((where, what, value))

    factorial = results.speedups(runs, "factorial")
    grid = Grid(factorial)
    v1 = {a: grid.v1(a) for a in results.MACHINES}
    intel, arm = v1["intel"], v1["arm"]

    # --- Section 5.1 / 5.3: grid size and headline speedups ----------------------
    methods = {
        "treewalker",
        "treewalker_fullwalk",
        "lightgbm_native",
        "xgboost_native",
        "lleaves",
        "tl2cgen",
        "quickscorer",
    }
    for arch, cells in v1.items():
        pairs = sum(len(methods & set(m)) for m in cells.values())
        say("5.3", f"timed (cell, method) pairs in v1's cells, {arch}", str(pairs))
    say(
        "5.3",
        "v1's cells per machine; factorial cells per machine",
        f"{len(intel)}; {len(grid.rows) // 2}",
    )
    for arch, cells in v1.items():
        ref = [alg(cells[ref_cell(ds, fw)]) for ds in DS for fw in FW]
        say(
            "abstract/7",
            f"{arch} reference-configuration algorithmic speedup",
            rng(min(ref), max(ref), 1),
        )
        g128 = [alg(cells[cid(ds, fw, 500, 8, 128)]) for ds in SURV for fw in FW]
        say("abstract/5.5/7", f"{arch} survival speedup at G=128", rng(min(g128), max(g128), 1))
    heat = figures.heatmap_cells(factorial)
    say("5.3", "heatmap cells above 1x", f"{sum(v > 1 for v in heat.values())}/{len(heat)}")
    hk = max(heat, key=heat.get)  # type: ignore[arg-type]
    say("5.3", "heatmap maximum", f"{r(heat[hk], 1)} at {hk}")
    top = max(intel, key=lambda c: alg(intel[c]))
    say("5.3", "largest Intel algorithmic speedup", f"{r(alg(intel[top]), 1)} at {top}")
    say(
        "5.3",
        "Arm / Intel algorithmic speedup (median ratio)",
        r(st.median(alg(arm[c]) / alg(intel[c]) for c in intel), 2),
    )

    # --- Section 5.4: system comparison at the reference (Intel) -----------------
    s, sx = intel[ref_cell("support", "lightgbm")], intel[ref_cell("support", "xgboost")]
    say("5.4", "TreeWalker vs LightGBM native, SUPPORT", r(s["lightgbm_native"]["speedup"], 1))
    say("5.4", "TreeWalker vs XGBoost native, SUPPORT", r(sx["xgboost_native"]["speedup"], 1))
    say(
        "5.4",
        "data-layout effect (LightGBM native over the full walk)",
        r(s["lightgbm_native"]["speedup"] / alg(s), 1),
    )
    say("5.4", "partial-evaluation effect", r(alg(s), 1))
    for m, g in (("tl2cgen", 16), ("lleaves", 16), ("lleaves", 128)):
        v = [intel[cid(ds, "lightgbm", 500, 8, g)][m]["speedup"] for ds in SURV]
        say("5.4", f"{m} / TreeWalker, LightGBM survival, G={g}", rng(min(v), max(v), 1))
    c = intel[cid("support", "lightgbm", 2000, 8, 128)]
    say(
        "5.4",
        "LightGBM-native system speedup at SUPPORT T=2000 L=8 G=128",
        r(c["lightgbm_native"]["speedup"], 0),
    )
    say(
        "5.4",
        "  its latencies, p50 per group (ms)",
        f"{r(c['lightgbm_native']['p50_us'] / 1000, 0)} vs "
        f"{r(c['treewalker']['p50_us'] / 1000, 2)}",
    )
    lg = {
        cell: m["lightgbm_native"]["speedup"] for cell, m in intel.items() if "lightgbm_native" in m
    }
    top = max(lg, key=lg.get)  # type: ignore[arg-type]
    say("5.4", "largest LightGBM-native system speedup in v1's cells", f"{r(lg[top], 0)} at {top}")
    e = intel[ref_cell("expedia", "xgboost")]
    say(
        "5.4",
        "Expedia/XGBoost: TreeWalker vs native (p50 per group)",
        f"{r(e['xgboost_native']['speedup'], 1)} ({r(e['treewalker']['p50_us'], 0)} vs "
        f"{r(e['xgboost_native_p0']['p50_us'], 0)} us)",
    )
    say(
        "5.4",
        "Expedia/XGBoost: TreeWalker's time over tl2cgen's",
        f"{r(1 / e['tl2cgen']['speedup'], 2)} ({r(e['tl2cgen']['p50_us'], 0)} us)",
    )

    # --- QuickScorer (5.4, App. D) -----------------------------------------------
    # QuickScorer loads neither categorical splits (SUPPORT, FLCHAIN) nor missing
    # values (Expedia), so v1's cells hold almost none of its valid cells; the
    # factorial's are mostly credit's what-if models.
    reasons: Counter[str] = Counter()
    for m in fm.read_json(results.run_dir(runs, "factorial", "intel") / "cells.json").values():
        for e in m["excluded"]:
            if e["method"] == "quickscorer":
                reasons[f"{m['id'].split('/')[0]}: {_qs_reason(e['reason'])}"] += 1
    say(
        "5.4/D",
        "QuickScorer exclusions (Intel factorial, cells)",
        "; ".join(f"{k} {v}" for k, v in sorted(reasons.items())),
    )
    every = {a: grid.cells(a) for a in results.MACHINES}
    for arch, cells in v1.items():
        for scope, pool in (("v1's cells", cells), ("the factorial", every[arch])):
            qs = [cell for cell, m in pool.items() if "quickscorer" in m]
            fast = [cell for cell in qs if pool[cell]["quickscorer"]["speedup"] < 1]
            say(
                "5.4/D",
                f"QuickScorer coverage in {scope}, {arch}",
                f"{len(qs)} cells, faster in {len(fast)}",
            )
    qs_all = [cell for cell, m in every["intel"].items() if "quickscorer" in m]
    say(
        "5.4/D",
        "QuickScorer cells by depth (factorial, Intel)",
        str(dict(sorted(Counter(results.design(c)["L"] for c in qs_all).items()))),
    )
    adv = [1 / every["intel"][c]["quickscorer"]["speedup"] for c in qs_all]
    say("D", "largest QuickScorer advantage (factorial, Intel)", r(max(adv), 1) if adv else "none")
    for ds in DS:
        for lv in (2, 4):
            mi, ma = (
                intel[ref_cell(ds, "lightgbm", 500, lv)],
                arm[ref_cell(ds, "lightgbm", 500, lv)],
            )
            if "quickscorer" not in mi:
                say("D", f"Table 9 {ds} L={lv}", "QuickScorer cannot load the model")
                continue
            say(
                "D",
                f"Table 9 {ds} L={lv} (QS us, TW us p50 per group, TW/QS Intel, TW/QS Arm)",
                f"{r(mi['quickscorer']['p50_us'], 1)}, {r(mi['treewalker']['p50_us'], 1)}, "
                f"{r(1 / mi['quickscorer']['speedup'], 2)}, "
                f"{r(1 / ma['quickscorer']['speedup'], 2)}",
            )

    # --- Section 5.5: sensitivity (Intel, survival, T=500) ------------------------
    for lv in (2, 16):
        v = [alg(intel[cid(ds, fw, 500, lv, 16)]) for ds in SURV for fw in FW]
        say("5.5", f"survival speedup at L={lv}", rng(min(v), max(v), 1))
    v = [alg(intel[cid(ds, fw, 500, 8, 16)]) for ds in SURV for fw in FW]
    say("5.5", "survival speedup at T=500", rng(min(v), max(v), 1))
    cohort = results.rows(
        factorial.filter(pl.col("arch") == "intel"), "treewalker_fullwalk"
    ).filter((pl.col("family") == "cohort") & (pl.col("T") == 500) & (pl.col("L") == 8))
    mono = []
    for fw in FW:
        sp = cohort.filter(pl.col("framework") == fw).sort("G")["speedup"].to_list()
        mono.append(all(b > a for a, b in pairwise(sp)))
    say("5.5", "Expedia speedup rises with cohort size 4-32 (both frameworks)", str(all(mono)))

    # --- Section 6: boundary cases, in v1's cells and over the whole factorial -------
    def boundary(arch: str, cells: dict[str, dict[str, dict[str, Any]]], scope: str) -> None:
        tag = f"{arch}, {scope}"
        slow = [cell for cell, m in cells.items() if alg(m) < 1]
        if scope == "v1's cells":
            gs = sorted({g_label(c) for c in slow}, key=lambda x: (len(x), x))
            where = f"at G in {{{', '.join(gs)}}}"
        else:
            by = Counter((results.design(c)["family"], results.design(c)["G"]) for c in slow)
            where = ", ".join(f"{fam} G={g} {n}" for (fam, g), n in sorted(by.items(), key=str))
        say("6", f"cells slower than own full walk, {tag}", f"{len(slow)}/{len(cells)}: {where}")
        if slow:
            say("6", f"  worst case, {tag}", r(max(1 / alg(cells[c]) for c in slow), 2))
        lgv = [
            m["lightgbm_native"]["speedup"] for cell, m in cells.items() if "lightgbm_native" in m
        ]
        say(
            "6",
            f"LightGBM native beaten, {tag} (cells, minimum)",
            f"{sum(x > 1 for x in lgv)}/{len(lgv)}, {r(min(lgv), 2)}",
        )
        xg = {
            cell: m["xgboost_native"]["speedup"]
            for cell, m in cells.items()
            if "xgboost_native" in m
        }
        faster = [c for c, x in xg.items() if x < 1]
        wk = min(xg, key=xg.get)  # type: ignore[arg-type]
        d = results.design(wk)
        say(
            "6",
            f"XGBoost native faster, {tag} (cells, its best time over TreeWalker's, where)",
            f"{len(faster)}/{len(xg)}, {r(1 / xg[wk], 2)}, ({d['dataset']!r}, {d['T']}, {d['L']}, "
            f"{d['workload']}, G={d['G']})",
        )
        say(
            "6",
            f"  TreeWalker beats its own full walk in those cells, {tag}",
            str(all(alg(cells[c]) > 1 for c in faster)),
        )
        f = [cell for cell, m in cells.items() if "tl2cgen" in m and m["tl2cgen"]["speedup"] < 1]
        small = sum(1 for c in f if results.design(c)["G"] in (1, 2, 4))
        say("6", f"tl2cgen faster, {tag}", f"{len(f)}/{len(cells)}, {small} at G<=4")
        f = [cell for cell, m in cells.items() if "lleaves" in m and m["lleaves"]["speedup"] < 1]
        small = sum(1 for c in f if results.design(c)["G"] in (1, 2, 4, 8))
        n_lgb = sum(1 for c in cells if results.design(c)["framework"] == "lightgbm")
        say("6", f"lleaves faster, {tag}", f"{len(f)}/{n_lgb}, {small} at G<=8")

    for arch in results.MACHINES:
        boundary(arch, v1[arch], "v1's cells")
        boundary(arch, grid.cells(arch), "the factorial")

    # --- Section 4.4 / Table 1: work decomposition ---------------------------------
    w, asym = figures.decomposition(results.work(runs, "ablation"))
    val = {
        (row["dataset"], row["framework"], row["evaluator"], row["horizon"]): row["per_row_ratio"]
        for row in w.filter(pl.col("dataset").is_in(list(SURV))).iter_rows(named=True)
    }
    for ds in SURV:
        for ev in ("trace", "precompute"):
            say(
                "Table 1",
                f"{ds} {ev} (G=4,16,64,128)",
                " ".join(r(val[(ds, "lightgbm", ev, g)], 3) for g in (4, 16, 64, 128)),
            )
    gap = max(
        abs(val[(ds, "xgboost", ev, g)] / val[(ds, "lightgbm", ev, g)] - 1)
        for ds in SURV
        for ev in ("trace", "precompute")
        for g in (4, 16, 64, 128)
    )
    say(
        "Table 1",
        "largest XGBoost vs LightGBM gap",
        f"{r(100 * gap, 1)}% (within 10%: {gap < 0.10})",
    )
    for ds in SURV:
        tr = val[(ds, "lightgbm", "trace", 128)] / asym[(ds, "trace")] - 1
        say(
            "4.4",
            f"{ds} trace ratio at G=128 over its limit",
            f"+{r(100 * tr, 1)}% (within 5%: {tr < 0.05})",
        )
        pre = val[(ds, "lightgbm", "precompute", 128)] / 0.125 - 1
        say(
            "4.4",
            f"{ds} precompute ratio at G=128 over 1/L",
            f"+{r(100 * pre, 1)}% (within 12%: {pre < 0.12})",
        )

    # --- Section 5.7: ablation at Figure 9's anchors (Intel) -----------------------
    research = results.speedups(runs, "ablation").filter(
        (pl.col("arch") == "intel")
        & (pl.col("method") == "treewalker_research")
        & (pl.col("mode") == "serving")
    )

    def effect(ds: str, h: int | None, variant: str, against: str = "all-on") -> list[float]:
        a = figures._anchor(research, ds, h)
        t = {
            (row["framework"], row["variant"]): row["us_per_row"] for row in a.iter_rows(named=True)
        }
        return [100 * (t[(fw, variant)] / t[(fw, against)] - 1) for fw in FW if (fw, variant) in t]

    anchors = {"support": 16, "flchain": 4, "expedia": None}
    for flag in ("|disable_tree_ordering", "|disable_prefix_grouping"):
        for ds, h in anchors.items():
            v = effect(ds, h, flag)
            say(
                "5.7",
                f"{flag.strip('|')[8:]} effect, {ds} (% range over frameworks)",
                rng(min(v), max(v), 0),
            )
    single = {
        label: effect("support", 16, variant, against) for label, variant, against in figures.FLAGS
    }
    worst = max(single, key=lambda k: max(single[k]) if single[k] else 0)
    say(
        "5.7",
        "largest single-switch effect at the SUPPORT reference",
        f"{worst} +{r(max(single[worst]), 0)}% "
        f"(all within 3%: {all(max(v) < 3 for v in single.values() if v)})",
    )

    # --- Section 5.6 / Tables 3 and 10: the credit what-if --------------------------
    sel = (
        (pl.col("dataset") == "credit")
        & (pl.col("framework") == "lightgbm")
        & (pl.col("T") == 500)
        & (pl.col("L") == 8)
        & (pl.col("family") == "whatif")
        & pl.col("replicate").is_null()
    )
    sc = {
        arch: {
            (row["k"], row["G"]): row["speedup"]
            for row in results.rows(
                factorial.filter(sel & (pl.col("arch") == arch)), "treewalker_fullwalk"
            ).iter_rows(named=True)
        }
        for arch in results.MACHINES
    }
    for arch in ("arm", "intel"):
        say(
            "5.6",
            f"what-if cells faster than full walk, {arch}",
            f"{sum(v > 1 for v in sc[arch].values())}/{len(sc[arch])}",
        )
    for arch, kv in (("arm", 1), ("arm", 8), ("intel", 1), ("intel", 8)):
        v = [sc[arch][(kv, g)] for g in (4, 16, 64, 128)]
        say(
            "5.6",
            f"what-if speedup range, {arch}, k={kv}",
            f"{r(min(v), 2 if min(v) < 2 else 1)}-{r(max(v), 1)}",
        )
    work = results.work(runs, "factorial").filter(sel & (pl.col("variant") == "all-on"))
    pc: dict[tuple[int, int], float] = {}
    for row in work.iter_rows(named=True):
        g = row["rows"] / row["groups"]
        total = (row["constant_steps"] + row["varying_splits"] + row["precompute_row_evals"]) / row[
            "groups"
        ] + g * 500
        pc[(row["k"], row["G"])] = total / (g * 500 * 8)
    for kk in (1, 2, 4, 8):
        say(
            "Table 3",
            f"k={kk} at G=128: work ratio (default evaluator), speedup Arm, Intel",
            f"{r(pc[(kk, 128)], 3)} {r(sc['arm'][(kk, 128)], 1)} {r(sc['intel'][(kk, 128)], 1)}",
        )
    peak = all(sc[a][(k, 64)] > sc[a][(k, 128)] for a in ("arm", "intel") for k in (1, 2, 4))
    say("5.6", "for k<=4 speedup peaks at G=64 (both machines)", str(peak))
    say(
        "5.6",
        "work ratio (default evaluator), k=1, G=64 -> 128",
        f"{r(pc[(1, 64)], 3)} -> {r(pc[(1, 128)], 3)}",
    )

    # --- App. F / Table 11: 128-row chunks against whole groups ---------------------
    for arch in ("arm", "intel"):
        d = factorial.filter((pl.col("arch") == arch) & (pl.col("mode") == "serving"))
        ch = {
            row["cell"]: row["us_per_row"]
            for row in results.rows(d, "treewalker_chunked128").iter_rows(named=True)
        }
        tw = {
            row["cell"]: row["us_per_row"]
            for row in results.rows(d, "treewalker", "all-on").iter_rows(named=True)
        }
        by_g: dict[int, list[float]] = {}
        for cell, t in ch.items():
            by_g.setdefault(results.design(cell)["G"], []).append(t / tw[cell])
        say(
            "Table 11",
            f"128-row chunks over whole groups, per-row time, median over cells, {arch} "
            "(G=256 512 1024)",
            " ".join(r(st.median(by_g[g]), 3) for g in sorted(by_g)),
        )

    # --- App. A.6: measurement variability --------------------------------------------
    for arch in ("intel", "arm"):
        d = factorial.filter(pl.col("arch") == arch)
        stops = d.select("cell", "mode", "stop_reason").unique()
        counts = Counter(stops["stop_reason"].to_list())
        n = sum(counts.values())
        say(
            "A.6",
            f"(cell, mode) pairs stopped by precision / block cap / time budget, {arch}",
            " / ".join(
                f"{counts.get(k, 0)} ({r(100 * counts.get(k, 0) / n, 0)}%)"
                for k in ("precision", "block_cap", "time_budget")
            ),
        )
        hw = (
            d.filter(
                (pl.col("mode") == "serving")
                & pl.col("lo").is_not_null()
                & (pl.col("method") != "treewalker")
            )
            .with_columns(half=100 * (pl.col("hi") - pl.col("lo")) / 2 / pl.col("speedup"))["half"]
            .to_list()
        )
        q = sorted(hw)
        say(
            "A.6",
            f"interval half-width, % of the speedup (median, 95th percentile, max), {arch}",
            f"{r(st.median(q), 2)}, {r(q[int(0.95 * len(q))], 2)}, {r(q[-1], 1)}",
        )
    return out


def generate(runs: Path, out: Path) -> list[str]:
    lines = [f"[{where}] {what}: {value}" for where, what, value in collect(runs)]
    out.mkdir(parents=True, exist_ok=True)
    (out / "paper_numbers.txt").write_text("\n".join(lines) + "\n")
    return lines
