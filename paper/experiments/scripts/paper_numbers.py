#!/usr/bin/env python3
"""Recompute the paper's in-text numbers from the released result CSVs.

Every check prints the paper location, the value as printed in the paper, and
the value recomputed from paper/experiments/data/. A check passes if the
printed value equals the recomputed one rounded half up, or truncated to the
printed precision (the paper truncates in a few places). Numbers that need model
artifacts (f32/f64 precision audits, E1 zero-value statistics) are covered by
audit_f32.py, the correctness tests, and prepare_scenario.py instead.

    uv run python3 paper/experiments/scripts/paper_numbers.py
"""
from __future__ import annotations

import csv
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE))

RESULTS: list[tuple[str, str, str, str]] = []
ROUNDING = ROUND_HALF_UP


def r(x: float, nd: int) -> str:
    """Format to nd decimals using the current rounding mode."""
    q = Decimal(1).scaleb(-nd)
    out = str(Decimal(repr(float(x))).quantize(q, rounding=ROUNDING))
    return out[1:] if out.startswith("-") and Decimal(out) == 0 else out


def rng(lo: float, hi: float, nd: int) -> str:
    return f"{r(lo, nd)}-{r(hi, nd)}"


def check(where: str, what: str, paper: str, got: str) -> None:
    RESULTS.append((where, what, paper, got))


def load(name: str) -> list[dict[str, str]]:
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def grid1(arch: str) -> dict[tuple, dict[str, float]]:
    d: dict[tuple, dict[str, float]] = defaultdict(dict)
    for x in load(f"grid1_results_{arch}.csv"):
        d[(x["dataset"], x["framework"], int(x["n_trees"]), int(x["max_depth"]),
           x["horizon"])][x["method"]] = float(x["median_us"])
    return d


SURV, DS, FW = ("support", "flchain"), ("support", "flchain", "expedia"), ("lightgbm", "xgboost")


def ref_g(ds: str, g: str = "16") -> str:
    return "" if ds == "expedia" else g


def alg(v: dict[str, float]) -> float:
    return v["treewalker_fullwalk"] / v["treewalker"]


def collect() -> None:
    I, A = grid1("intel"), grid1("arm")

    # --- Section 5.1 / 5.3: grid size and headline speedups -------------------
    n_rows = len(load("grid1_results_intel.csv"))
    check("5.3", "measurement rows per architecture", "2588", str(n_rows))
    check("5.3", "grid cells per architecture", "544", str(len(I)))
    for arch, d, ref_s, g128_s in (("Intel", I, "2.5-3.2", "6.8-7.8"), ("Arm", A, "3.4-6.2", None)):
        ref = [alg(d[(ds, fw, 500, 8, ref_g(ds))]) for ds in DS for fw in FW]
        check("abstract/7", f"{arch} reference-configuration algorithmic speedup", ref_s, rng(min(ref), max(ref), 1))
        g128 = [alg(d[(ds, fw, 500, 8, "128")]) for ds in SURV for fw in FW]
        if g128_s:
            check("abstract/5.5", f"{arch} survival speedup at G=128", g128_s, rng(min(g128), max(g128), 1))
        else:
            check("7", "Arm survival speedup at G=128 (max)", "13.6", r(max(g128), 1))

    heat = {(ds, T, L): st.mean(alg(I[(ds, fw, T, L, ref_g(ds))]) for fw in FW)
            for ds in DS for T in (50, 500, 1000, 2000) for L in (2, 4, 8, 16)}
    check("5.3", "heatmap cells above 1x", "48/48", f"{sum(v > 1 for v in heat.values())}/{len(heat)}")
    k = max(heat, key=heat.get)
    check("5.3", "heatmap maximum (FLCHAIN, T=2000, L=2)", "4.0 at ('flchain', 2000, 2)", f"{r(heat[k], 1)} at {k}")
    k = max(I, key=lambda k: alg(I[k]))
    check("5.3", "largest Intel algorithmic speedup", "8.3 at ('flchain', 'xgboost', 1000, 4, '64')",
          f"{r(alg(I[k]), 1)} at {k}")
    check("5.3", "Arm / Intel algorithmic speedup (median ratio)", "1.4",
          r(st.median(alg(A[k]) / alg(I[k]) for k in I), 1))

    # --- Section 5.4: system comparison at the reference ----------------------
    s = I[("support", "lightgbm", 500, 8, "16")]
    sx = I[("support", "xgboost", 500, 8, "16")]
    check("5.4", "TreeWalker vs LightGBM native, SUPPORT", "7.1", r(s["lightgbm_native"] / s["treewalker"], 1))
    check("5.4", "TreeWalker vs XGBoost native, SUPPORT", "4.6", r(sx["xgboost_native"] / sx["treewalker"], 1))
    check("5.4", "data-layout effect (full walk vs LightGBM native)", "2.4",
          r(s["lightgbm_native"] / s["treewalker_fullwalk"], 1))
    check("5.4", "partial-evaluation effect", "2.9", r(alg(s), 1))
    for m, g, paper in (("tl2cgen", "16", "2.3-2.8"), ("lleaves", "16", "1.0-1.3"), ("lleaves", "128", "2.0-2.4")):
        v = [I[(ds, "lightgbm", 500, 8, g)][m] / I[(ds, "lightgbm", 500, 8, g)]["treewalker"] for ds in SURV]
        check("5.4", f"{m} / TreeWalker, LightGBM survival, G={g}", paper, rng(min(v), max(v), 1))
    c = I[("support", "lightgbm", 2000, 8, "128")]
    check("5.4", "largest LightGBM-native system speedup", "44", r(c["lightgbm_native"] / c["treewalker"], 0))
    check("5.4", "  its latencies (ms)", "29 vs 0.65",
          f"{r(c['lightgbm_native'] / 1000, 0)} vs {r(c['treewalker'] / 1000, 2)}")
    e = I[("expedia", "xgboost", 500, 8, "")]
    check("5.4", "Expedia/XGBoost: TreeWalker vs native", "1.9 (156 vs 291 us)",
          f"{r(e['xgboost_native'] / e['treewalker'], 1)} ({r(e['treewalker'], 0)} vs {r(e['xgboost_native'], 0)} us)")
    check("5.4", "Expedia/XGBoost: tl2cgen faster than TreeWalker", "1.09 (143 us)",
          f"{r(e['treewalker'] / e['tl2cgen'], 2)} ({r(e['tl2cgen'], 0)} us)")

    # --- QuickScorer (5.4, App. D) --------------------------------------------
    for arch, d, paper in (("Intel", I, "140 cells, faster in 108"), ("Arm", A, "140 cells, faster in 99")):
        q = [k for k, v in d.items() if "quickscorer" in v]
        f = [k for k in q if d[k]["quickscorer"] < d[k]["treewalker"]]
        check("5.4/D", f"QuickScorer coverage, {arch}", paper, f"{len(q)} cells, faster in {len(f)}")
    q = [k for k, v in I.items() if "quickscorer" in v]
    check("5.4/D", "QuickScorer cells by depth", "{2: 68, 4: 68, 8: 4}", str(dict(sorted(Counter(k[3] for k in q).items()))))
    refq = [I[(ds, "lightgbm", 500, L, ref_g(ds))] for ds in DS for L in (2, 4)]
    tq = [v["treewalker"] / v["quickscorer"] for v in refq]
    check("5.4", "QuickScorer advantage at the reference horizon, Intel", "1.1-2.5", rng(min(tq), max(tq), 1))
    check("D", "largest QuickScorer advantage, Intel", "16.6",
          r(max(I[k]["treewalker"] / I[k]["quickscorer"] for k in q), 1))
    t9 = {("support", 2): "6.8, 10.8, 1.57, 1.29", ("support", 4): "22.8, 25.3, 1.11, 0.59",
          ("flchain", 2): "5.2, 9.4, 1.82, 1.43", ("flchain", 4): "18.8, 21.0, 1.12, 0.55",
          ("expedia", 2): "10.5, 26.0, 2.46, 2.18", ("expedia", 4): "38.6, 65.1, 1.69, 0.96"}
    for ds in DS:
        for L in (2, 4):
            k = (ds, "lightgbm", 500, L, ref_g(ds))
            check("D", f"Table 9 {ds} L={L} (QS us, TW us, TW/QS Intel, TW/QS Arm)", t9[(ds, L)],
                  f"{r(I[k]['quickscorer'], 1)}, {r(I[k]['treewalker'], 1)}, "
                  f"{r(I[k]['treewalker'] / I[k]['quickscorer'], 2)}, {r(A[k]['treewalker'] / A[k]['quickscorer'], 2)}")

    # --- Section 5.5: sensitivity (Intel, survival, T=500) --------------------
    for L, paper in ((2, "2.9-4.5"), (16, "1.9-2.7")):
        v = [alg(I[(ds, fw, 500, L, "16")]) for ds in SURV for fw in FW]
        check("5.5", f"survival speedup at L={L}", paper, rng(min(v), max(v), 1))
    v = [alg(I[(ds, fw, 500, 8, "16")]) for ds in SURV for fw in FW]
    check("5.5", "survival speedup at T=500", "2.9-3.2", rng(min(v), max(v), 1))
    g4 = defaultdict(list)
    for x in load("grid4_results_intel.csv"):
        if x["n_trees"] == "500" and x["max_depth"] == "8":
            g4[(x["framework"], float(x["mean_group_size"]))].append(x)
    mono = []
    for fw in FW:
        pts = sorted((m, {x["method"]: float(x["median_us"]) for x in g4[(f, m)]}) for f, m in g4 if f == fw)
        sp = [p["treewalker_fullwalk"] / p["treewalker"] for _, p in pts]
        mono.append(all(b > a for a, b in zip(sp, sp[1:])))
    check("5.5", "Expedia speedup rises with mean group size (both frameworks)", "True", str(all(mono)))

    # --- Section 6: boundary cases ---------------------------------------------
    for arch, d, n_s, w_s, xn, xw in (("Intel", I, "64", "1.50", "38", "6.3"), ("Arm", A, "31", "1.74", "45", "8.9")):
        slow = [k for k, v in d.items() if alg(v) < 1]
        check("6", f"cells slower than own full walk, {arch}", f"{n_s}/544 at G in {{1, 2}}",
              f"{len(slow)}/544 at G in {{{', '.join(sorted({k[4] for k in slow}))}}}")
        check("6", f"  worst case, {arch}", w_s, r(max(1 / alg(d[k]) for k in slow), 2))
        lg = [d[k]["lightgbm_native"] / d[k]["treewalker"] for k in d if k[1] == "lightgbm"]
        check("6", f"LightGBM native beaten, {arch} (cells, minimum)", "272/272, " + ("2.03" if arch == "Intel" else "2.21"),
              f"{sum(x > 1 for x in lg)}/{len(lg)}, {r(min(lg), 2)}")
        xg = {k: d[k]["xgboost_native"] / d[k]["treewalker"] for k in d if k[1] == "xgboost"}
        wk = min(xg, key=xg.get)
        check("6", f"XGBoost native faster, {arch} (cells, worst, where)",
              f"{xn}/272, {xw}, ('expedia', 2000, 16)",
              f"{sum(x < 1 for x in xg.values())}/272, {r(1 / xg[wk], 1)}, {(wk[0], wk[2], wk[3])}")
        check("6", f"  TreeWalker beats its own full walk in those cells, {arch}", "True",
              str(all(alg(d[k]) > 1 for k, x in xg.items() if x < 1)))
    for arch, d, paper in (("Intel", I, "152/544, 114 at G<=4"), ("Arm", A, "91/544")):
        f = [k for k, v in d.items() if "tl2cgen" in v and v["tl2cgen"] < v["treewalker"]]
        got = f"{len(f)}/544" + (f", {sum(1 for k in f if k[4] in ('1', '2', '4'))} at G<=4" if arch == "Intel" else "")
        check("6", f"tl2cgen faster, {arch}", paper, got)
    for arch, d, paper in (("Intel", I, "83/272, 77 at G<=8"), ("Arm", A, "56/272")):
        f = [k for k, v in d.items() if "lleaves" in v and v["lleaves"] < v["treewalker"]]
        got = f"{len(f)}/272" + (f", {sum(1 for k in f if k[4] in ('1', '2', '4', '8'))} at G<=8" if arch == "Intel" else "")
        check("6", f"lleaves faster, {arch}", paper, got)

    # --- Section 4.4 / Table 1: work decomposition ------------------------------
    import decomposition_validation as dv
    data, raw = dv.collect_data()
    grid = {(x["dataset"], x["framework"], x["evaluator"], int(x["G"])): x["per_row_ratio"]
            for x in data if x["dataset"] in SURV}
    paper_t1 = {("support", "trace"): "0.674 0.432 0.339 0.309", ("support", "precompute"): "0.507 0.226 0.151 0.139",
                ("flchain", "trace"): "0.544 0.451 0.439 0.431", ("flchain", "precompute"): "0.420 0.213 0.153 0.140"}
    for (ds, ev), paper in paper_t1.items():
        check("Table 1", f"{ds} {ev} (G=4,16,64,128)", paper,
              " ".join(r(grid[(ds, "lightgbm", ev, g)], 3) for g in (4, 16, 64, 128)))
    gap = max(abs(grid[(ds, "xgboost", ev, g)] / grid[(ds, "lightgbm", ev, g)] - 1)
              for ds in SURV for ev in ("trace", "precompute") for g in (4, 16, 64, 128))
    check("Table 1", "largest XGBoost vs LightGBM gap (within 10%)", "True", str(gap < 0.10))
    for ds in SURV:
        tr = dv.asymptote([x for x in raw if x["framework"] == "lightgbm"], ds, "trace", 8)
        check("4.4", f"{ds} trace ratio at G=128 within 5% of its limit", "True",
              str(grid[(ds, "lightgbm", "trace", 128)] / tr - 1 < 0.05))
        check("4.4", f"{ds} precompute ratio at G=128 within 12% of 1/L", "True",
              str(grid[(ds, "lightgbm", "precompute", 128)] / 0.125 - 1 < 0.12))

    # --- Section 5.7: ablation at Figure 9's anchor cells (Intel) ---------------
    flags = ["disable_precompute", "disable_unsplit", "disable_monotonic", "disable_tree_ordering",
             "disable_prefix_grouping", "disable_bitset_intern"]
    g3 = load("grid3_results_intel.csv")

    def lat(ds, g, flag, fw):
        return next(float(x["median_us"]) for x in g3 if x["dataset"] == ds and x["framework"] == fw
                    and x["n_trees"] == "500" and x["max_depth"] == "8" and x["horizon"] == g
                    and all(x[f] == ("1" if f == flag else "0") for f in flags))

    anchors = {"support": "16", "flchain": "4", "expedia": ""}
    paper_ab = {("disable_tree_ordering", "support"): "1-2", ("disable_tree_ordering", "flchain"): "4-7",
                ("disable_tree_ordering", "expedia"): "3-3", ("disable_prefix_grouping", "support"): "0-2",
                ("disable_prefix_grouping", "flchain"): "4-5", ("disable_prefix_grouping", "expedia"): "1-1"}
    for (flag, ds), paper in paper_ab.items():
        v = [100 * (lat(ds, anchors[ds], flag, fw) / lat(ds, anchors[ds], None, fw) - 1) for fw in FW]
        check("5.7", f"{flag[8:]} effect, {ds} (% range over frameworks)", paper, rng(min(v), max(v), 0))
    ref_max = max(100 * (lat("support", "16", f, fw) / lat("support", "16", None, fw) - 1) for f in flags for fw in FW)
    check("5.7", "all single-flag effects at the SUPPORT reference within 3%", "True", str(ref_max < 3))

    # --- Section 5.6 / Tables 3 and 10: scenario analysis -----------------------
    sc = {}
    for arch in ("arm", "intel"):
        d = defaultdict(dict)
        for x in load(f"scenario_credit_results_{arch}.csv"):
            d[(int(x["k"]), int(x["G"]))][x["method"]] = float(x["median_us"])
        sc[arch] = {kg: alg(v) for kg, v in d.items()}
    for arch, paper in (("arm", "16/16"), ("intel", "16/16")):
        check("5.6", f"scenario cells faster than full walk, {arch}", paper,
              f"{sum(v > 1 for v in sc[arch].values())}/{len(sc[arch])}")
    for arch, k, paper in (("arm", 1, "3.8-21.7"), ("arm", 8, "1.85-11.0"), ("intel", 1, "2.0-11.9"), ("intel", 8, "1.07-6.1")):
        v = [sc[arch][(k, g)] for g in (4, 16, 64, 128)]
        lo = r(min(v), 2 if min(v) < 2 else 1)
        check("5.6", f"scenario speedup range, {arch}, k={k}", paper, f"{lo}-{r(max(v), 1)}")
    stats = [x for x in load("scenario_credit_stats_arm.csv") if x["G"] == "128"]
    exc, pc = [], {}
    for x in stats:
        n, k = int(x["n_obs"]), int(x["k"])
        if x["evaluator"] == "trace":
            w = (int(x["constant_steps"]) + int(x["varying_splits"]) + int(x["partition_row_evals"])) / n + 128 * 500
            ratio = w / (128 * 500 * 8)
            dvv = int(x["partition_row_evals"]) / (128 * 500 * n)
            pred = (dvv + 1) / 8
            exc.append(100 * (ratio / pred - 1))
            check("Table 3", f"k={k}: d_v, measured, predicted, Arm, Intel",
                  {1: "0.50 0.195 0.188 17.1 8.4", 2: "1.15 0.276 0.268 15.7 7.8",
                   4: "2.09 0.395 0.386 13.6 7.1", 8: "3.81 0.611 0.601 11.0 6.1"}[k],
                  f"{r(dvv, 2)} {r(ratio, 3)} {r(pred, 3)} {r(sc['arm'][(k, 128)], 1)} {r(sc['intel'][(k, 128)], 1)}")
    check("5.6", "measured work ratio above prediction at G=128 (%)", "1.8-3.9", rng(min(exc), max(exc), 1))
    peak = all(sc[a][(k, 64)] > sc[a][(k, 128)] for a in ("arm", "intel") for k in (1, 2, 4))
    check("5.6", "for k<=4 speedup peaks at G=64 (both architectures)", "True", str(peak))
    for x in load("scenario_credit_stats_arm.csv"):
        if x["evaluator"] == "precompute" and x["k"] == "1" and x["G"] in ("64", "128"):
            n, g = int(x["n_obs"]), int(x["G"])
            w = (int(x["constant_steps"]) + int(x["varying_splits"]) + int(x["precompute_row_evals"])) / n + g * 500
            pc[g] = w / (g * 500 * 8)
    check("5.6", "precompute ratio, k=1, G=64 -> 128", "0.140 -> 0.133", f"{r(pc[64], 3)} -> {r(pc[128], 3)}")

    # --- App. F / Table 11: chunked groups ---------------------------------------
    for arch, paper in (("arm", "0.987 0.980 0.960 0.932 | -0.72 -2.75 -5.58"),
                        ("intel", "0.977 0.962 0.942 0.917 | -1.54 -3.53 -6.09")):
        rows = sorted(load(f"chunked_g_results_{arch}.csv"), key=lambda x: int(x["logical_G"]))
        med = [float(x["tw_median_us_per_row"]) for x in rows]
        check("Table 11", f"per-row medians and deltas, {arch}", paper,
              " ".join(r(m, 3) for m in med) + " | " + " ".join(r(100 * (m / med[0] - 1), 2) for m in med[1:]))

    # --- App. A.6: measurement variability ---------------------------------------
    gen = np.random.default_rng(42)
    kn = {}
    for n in range(11, 22):
        i5, i95 = (min(math.floor(p * (n - 1) + 0.5), n - 1) for p in (0.05, 0.95))
        x = np.sort(gen.standard_normal((200_000, n)), axis=1)
        kn[n] = float((x[:, i95] - x[:, i5]).mean())
    check("A.6", "k_11, k_21", "2.65, 2.87", f"{r(kn[11], 2)}, {r(kn[21], 2)}")
    for arch, frac_s, med_s, abl_s, cap_s in (("intel", "92", "2.7", "18.7", "157/387"), ("arm", "93", "1.4", "14.2", "308/236")):
        g1 = load(f"grid1_results_{arch}.csv")
        cv = [100 * (float(x["p95_us"]) - float(x["p5_us"])) / (kn[int(x["iters"])] * float(x["median_us"])) for x in g1]
        check("A.6", f"estimated CV below 10%, {arch} (% of cell-method pairs)", frac_s, r(100 * np.mean(np.array(cv) < 10), 0))
        check("A.6", f"median estimated CV, {arch}", med_s, r(float(np.median(cv)), 1))
        g3r = load(f"grid3_results_{arch}.csv")
        cv3 = [100 * (float(x["p95_us"]) - float(x["p5_us"])) / (kn[int(x["iters"])] * float(x["median_us"])) for x in g3r]
        check("A.6", f"ablation-grid maximum estimated CV, {arch}", abl_s, r(max(cv3), 1))
        blocks = {}
        for x in g1:
            blocks[(x["dataset"], x["framework"], x["n_trees"], x["max_depth"], x["horizon"])] = int(x["iters"])
        n21 = sum(b == 21 for b in blocks.values())
        check("A.6", f"cells stopped early / at the 21-block cap, {arch}", cap_s, f"{len(blocks) - n21}/{n21}")
    top = max(max(100 * (float(x["p95_us"]) - float(x["p5_us"])) / (kn[int(x["iters"])] * float(x["median_us"]))
                  for x in load(f"grid1_results_{a}.csv")) for a in ("intel", "arm"))
    check("A.6", "largest factorial-grid estimate (up to)", "45", r(top, 0))



def main() -> int:
    global ROUNDING
    passes = []
    for mode in (ROUND_HALF_UP, ROUND_DOWN):
        ROUNDING = mode
        RESULTS.clear()
        collect()
        passes.append(list(RESULTS))
    bad = 0
    for (where, what, paper, got), (*_, got_down) in zip(*passes):
        if paper == got:
            status = "OK"
        elif paper == got_down:
            status = "OK (truncated)"
        else:
            status, bad = "MISMATCH", bad + 1
        print(f"{status:<15}[{where}] {what}: paper {paper}, data {got}")
    print(f"\n{len(passes[0])} checks, {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
