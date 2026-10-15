"""The paper's in-text numbers, regenerated from a campaign's runs (`results`).

The same items as v1's paper_numbers.py (at the neurips2026-submission tag), in its order and
with its paper locations, computed from the new runs and printed as new values only
(the user's decision, 2026-10-08). The definitions follow the camera-ready estimand
(`estimand`):

- a latency is the median over blocks of each block's median time per group
  (nearest rank, process 0); an "algorithmic speedup" is the full walk's latency
  over TreeWalker's, and a baseline's speedup its latency over TreeWalker's (v1:
  ratios of median latencies). XGBoost native is process 0, like every method; its
  faster mode across the extra processes is a sensitivity item, as is the
  row-weighted ratio. Each cell runs in a fresh process, so no method is
  corrected for drift; App. A.6 reports the sentinel's drift as a check;
- "v1's cells" are the 544 per machine v1's grid had: survival panels at horizons
  1 to 128 and Expedia's sessions, T in {50, 500, 1000, 2000}, L in {2, 4, 8, 16}.
  Counts the paper gives over its grid (QuickScorer's coverage, §6's boundary
  cases) are reported over v1's cells and over the whole factorial suite;
- Table 3's trace-evaluator columns come from scenario-v1's trace variant in the
  camera-ready campaign and are absent from the final run's (`tables`); Table 11 compares the
  128-row chunks with whole groups at G = 256, 512 and 1024; App. A.6 reports the
  stopping rules and the intervals' half-widths instead of v1's CV estimate;
- the campaign (`results.CAMPAIGNS`) picks the runs: the final run's factorial,
  ablation and what-if (whatif-v2) cells, or the camera-ready rerun's suite, its
  two ablation runs (effects with their spread between the runs) and E1's
  released scenario cells (scenario-v1).
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl

from . import analysis, figures, results, tables
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
    (XGBoost native's process-0 row; its faster-mode row as ``xgboost_native_fm``)."""

    def __init__(self, df: Any) -> None:
        self.rows: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        for row in df.filter(pl.col("mode") == "serving").iter_rows(named=True):
            m = row["method"]
            if m == "xgboost_native":
                m = "xgboost_native_fm" if row["variant"] == "faster-mode" else "xgboost_native"
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


within = results.within


def g_label(cell: str) -> str:
    g = results.design(cell)["G"]
    return "sessions" if g is None else str(g)


def ablation_repeatability(effects: Any) -> list[tuple[str, str]]:
    """Per machine and switch at the SUPPORT reference, each ablation run's effect per
    framework (%), and the largest between-run range over the frameworks and anchors
    (`figures.ablation_spread`, percentage points); one run has no range."""
    spread = figures.ablation_spread(effects)
    runs = sorted(set(effects["run"].to_list()))
    out = []
    for arch in sorted({x["arch"] for x in spread}):
        a = [x for x in spread if x["arch"] == arch]
        parts = []
        for flag, _, _ in figures.FLAGS:
            rows = sorted(
                (x for x in a if x["ds"] == "SUPPORT" and x["flag"] == flag),
                key=lambda x: x["framework"],
            )
            if not rows:
                continue
            by_fw = {x["framework"]: x["by_run"] for x in rows}
            per_run = [
                "/".join(
                    r(100 * (by_fw[fw][run] - 1), 1) if run in by_fw.get(fw, {}) else "-"
                    for fw in FW
                )
                for run in runs
            ]
            ranges = [
                x["between_runs_pp"]
                for x in a
                if x["flag"] == flag and x["between_runs_pp"] is not None
            ]
            parts.append(
                f"{flag} {' | '.join(per_run)}"
                + (f" (between runs up to {r(max(ranges), 1)} pp)" if ranges else "")
            )
        out.append(
            (
                f"switch effects at the SUPPORT reference, {arch}, % per run ({', '.join(runs)}; "
                "LightGBM/XGBoost, -: not measured; between-run range over every anchor and "
                "framework measured by two runs)",
                "; ".join(parts),
            )
        )
    return out


def retime_counts(runs: Path, c: results.Campaign) -> list[tuple[str, str]]:
    """Per machine, over the campaign's runs: the cells the re-time plans flagged,
    the (cell, mode) pairs re-timed, those whose attempt is reported, those with a
    refused attempt; a refusal is also a warning on stderr, for App. A.4."""
    import sys

    from . import retime

    out = []
    for arch in results.MACHINES:
        flagged: int | None = None
        rows = []
        for run in tables.campaign_runs(c):
            if not results.run_dir(runs, run, arch).exists():
                continue
            rows += retime.dispositions(runs, run, arch)
            n = retime.planned(runs.parent / "retime", run, arch)
            flagged = n if flagged is None else flagged + (n or 0)
        tries = [a for r in rows for a in r["attempts"]]
        refused = sum(any(a["status"] == "refused" for a in r["attempts"]) for r in rows)
        selected = sum(r["selected"] != r["parent"][0] for r in rows)
        if refused:
            print(
                f"warning: re-timing on {arch}: {refused} (cell, mode) pairs refused; see "
                f"retime_{arch}.md and disclose in App. A.4",
                file=sys.stderr,
            )
        out.append(
            (
                f"re-timing, {arch} (cells flagged; (cell, mode) re-timed; attempt reported; "
                "refused)",
                f"{'unknown' if flagged is None else flagged}; {len(rows)}; {selected}; {refused}"
                + ("" if tries else " (no re-time run)"),
            )
        )
    return out


def exact_sums_effect(ablation: Any) -> list[tuple[str, str]]:
    """The released default's exact-sums effect, which the README states: per
    machine, the research build's latency with exact sums off
    (``disable_exact_sums``) over its all-on, serving, T=500, L=8, the range over
    datasets and frameworks per G and for Expedia's sessions. Read from the final
    run's ablation suite (2026-10-09/10): the camera-ready ablation has no
    disable_exact_sums variant."""
    d = ablation.filter(
        (pl.col("method") == "treewalker_research")
        & (pl.col("mode") == "serving")
        & (pl.col("T") == 500)
        & (pl.col("L") == 8)
        & pl.col("variant").is_in(["all-on", "disable_exact_sums"])
    )
    t = {
        (row["arch"], row["cell"], row["variant"]): row["latency_us"]
        for row in d.iter_rows(named=True)
    }
    out = []
    for arch in sorted({k[0] for k in t}):
        by: dict[str, list[float]] = {}
        for (a, cell, v), lat in t.items():
            base = t.get((a, cell, "all-on"))
            if a != arch or v != "disable_exact_sums" or base is None:
                continue
            g = results.design(cell)["G"]
            by.setdefault("Expedia sessions" if g is None else f"G={g}", []).append(lat / base)
        keys = sorted(
            by, key=lambda k: (k.startswith("Expedia"), int(k[2:]) if k[2:].isdigit() else 0)
        )
        out.append(
            (
                f"exact sums off over the released default (on), research build, serving, "
                f"T=500, L=8, final run's ablation, {arch} (range over datasets and frameworks)",
                "; ".join(f"{k} {rng(min(by[k]), max(by[k]), 2)}x" for k in keys) or "none",
            )
        )
    return out


def unsplit(work: Any) -> list[tuple[str, str]]:
    """The unsplit counts at the ablation's anchors (T=500, L=8, both frameworks): U,
    the varying splits the default evaluator skips as unsplit (one side's rows
    empty), per group and as a share of its varying splits V, and the ablated path's
    extra calls: ``disable_unsplit`` recurses where the default skips, so its
    recursive calls exceed the default's B by E, and E - U are the calls beyond U
    (splits reached only below an unsplit node). Asserts U = V - B, the counters'
    identity, on every cell and variant of ``work`` (`results.work`)."""
    bad = work.filter(
        pl.col("unsplit_skips") != pl.col("varying_splits") - pl.col("recursive_calls")
    )
    if bad.height:
        raise ValueError(
            f"unsplit skips differ from varying splits minus recursive calls in {bad.height} "
            f"(cell, variant) pairs, e.g. {bad.row(0, named=True)['cell']}"
        )
    out = []
    for label, ds, h in figures.ANCHORS:
        a = figures._anchor(work, ds, h)
        t = {(r["framework"], r["variant"]): r for r in a.iter_rows(named=True)}
        for fw in FW:
            d, ab = t.get((fw, "all-on")), t.get((fw, "disable_unsplit"))
            if d is None or ab is None:
                continue
            u, v = d["unsplit_skips"], d["varying_splits"]
            extra = ab["recursive_calls"] - d["recursive_calls"]
            beyond = (extra - u) / u if u else float("nan")
            per = r(extra / d["groups"], 1)
            out.append(
                (
                    f"U at the {label} anchor, {fw} (per group, share of V; disable_unsplit's "
                    "extra calls E per group, share beyond U = (E - U) / U)",
                    f"{r(u / d['groups'], 1)} ({r(100 * u / v, 1)}% of V); E {per}, "
                    f"{r(100 * beyond, 2)}% beyond U",
                )
            )
    return out


def collect(runs: Path, campaign: results.Campaign | None = None) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    c = campaign or results.CAMPAIGNS["final"]

    def say(where: str, what: str, value: str) -> None:
        out.append((where, what, value))

    factorial = results.speedups(runs, c.factorial)
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
    say(
        "5.3",
        "heatmap cells above 1x",
        f"{sum(v > 1 for v in heat.values())}/{len(heat)}; "
        f"within ±2%: {within(list(heat.values()))}",
    )
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
    c_ = intel[cid("support", "lightgbm", 2000, 8, 128)]
    say(
        "5.4",
        "LightGBM-native system speedup at SUPPORT T=2000 L=8 G=128",
        r(c_["lightgbm_native"]["speedup"], 0),
    )
    say(
        "5.4",
        "  its latencies, median over blocks of the block medians per group (ms)",
        f"{r(c_['lightgbm_native']['latency_us'] / 1000, 0)} vs "
        f"{r(c_['treewalker']['latency_us'] / 1000, 2)}",
    )
    lg = {
        cell: m["lightgbm_native"]["speedup"] for cell, m in intel.items() if "lightgbm_native" in m
    }
    top = max(lg, key=lg.get)  # type: ignore[arg-type]
    say("5.4", "largest LightGBM-native system speedup in v1's cells", f"{r(lg[top], 0)} at {top}")
    e = intel[ref_cell("expedia", "xgboost")]
    say(
        "5.4",
        "Expedia/XGBoost: TreeWalker vs native (latency per group)",
        f"{r(e['xgboost_native']['speedup'], 1)} ({r(e['treewalker']['latency_us'], 0)} vs "
        f"{r(e['xgboost_native']['latency_us'], 0)} us)",
    )
    say(
        "5.4",
        "Expedia/XGBoost: TreeWalker's time over tl2cgen's",
        f"{r(1 / e['tl2cgen']['speedup'], 2)} ({r(e['tl2cgen']['latency_us'], 0)} us)",
    )
    # Sensitivity: XGBoost's faster mode over its extra processes, row-weighted.
    for ds in DS:
        x = intel[ref_cell(ds, "xgboost")].get("xgboost_native_fm")
        say(
            "5.4/A.2",
            f"sensitivity: TreeWalker vs XGBoost native's faster mode, {ds} reference "
            "(row-weighted)",
            "no mode analysis" if x is None else f"{r(x['speedup_rw'], 1)} ({x['note']})",
        )

    # --- QuickScorer (5.4, App. D) -----------------------------------------------
    # QuickScorer loads neither categorical splits (SUPPORT, FLCHAIN) nor missing
    # values (Expedia), so v1's cells hold almost none of its valid cells; the
    # factorial's are mostly credit's what-if models.
    reasons: Counter[str] = Counter()
    for m in fm.read_json(results.run_dir(runs, c.factorial, "intel") / "cells.json").values():
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
                f"{len(qs)} cells, faster in {len(fast)}; within ±2%: "
                f"{within([pool[c]['quickscorer']['speedup'] for c in qs])}",
            )
    qs_all = [cell for cell, m in every["intel"].items() if "quickscorer" in m]
    say(
        "5.4/D",
        "QuickScorer cells by depth (factorial, Intel)",
        str(dict(sorted(Counter(results.design(c)["L"] for c in qs_all).items()))),
    )
    adv = [1 / every["intel"][c]["quickscorer"]["speedup"] for c in qs_all]
    say("D", "largest QuickScorer advantage (factorial, Intel)", r(max(adv), 1) if adv else "none")
    # Table 9, the submitted head-to-head on the original datasets, is withdrawn: its
    # cells cannot be validated. Its replacement is the expedia-filled comparison.
    for ds in DS:
        for lv in (2, 4):
            say(
                "D",
                f"Table 9 {ds} L={lv}",
                "withdrawn; see the expedia-filled comparison (quickscorer_filled_<arch>.md)",
            )
    for arch in results.MACHINES:
        cells = fm.read_json(results.run_dir(runs, c.factorial, arch) / "cells.json")
        filled = tables.quickscorer_filled(factorial, cells, arch)
        qs_timed = [x for x in filled if x["status"] == "timed"]
        excluded = Counter(_qs_reason(x["status"]) for x in filled if x["status"] != "timed")
        say(
            "D",
            f"QuickScorer on expedia-filled, a new comparison on an encoded dataset, {arch} "
            "(eligible; timed LightGBM/XGBoost; TreeWalker faster of timed; speedup range; "
            "exclusions)",
            f"{len(filled)}; {sum(x['framework'] == 'lightgbm' for x in qs_timed)}/"
            f"{sum(x['framework'] == 'xgboost' for x in qs_timed)}; "
            f"{sum(x['speedup'] > 1 for x in qs_timed)} of {len(qs_timed)} (within ±2%: "
            f"{within([x['speedup'] for x in qs_timed])}); "
            + (
                rng(min(x["speedup"] for x in qs_timed), max(x["speedup"] for x in qs_timed), 2)
                if qs_timed
                else "none"
            )
            + "; "
            + (", ".join(f"{k} {v}" for k, v in sorted(excluded.items())) or "none"),
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
        near = within([alg(m) for m in cells.values()])
        say(
            "6",
            f"cells slower than own full walk, {tag}",
            f"{len(slow)}/{len(cells)}: {where}; within ±2%: {near}",
        )
        if slow:
            say("6", f"  worst case, {tag}", r(max(1 / alg(cells[c]) for c in slow), 2))
        lgv = [
            m["lightgbm_native"]["speedup"] for cell, m in cells.items() if "lightgbm_native" in m
        ]
        say(
            "6",
            f"LightGBM native beaten, {tag} (cells, minimum)",
            f"{sum(x > 1 for x in lgv)}/{len(lgv)}, {r(min(lgv), 2)}; within ±2%: {within(lgv)}",
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
            f"{d['workload']}, G={d['G']}); within ±2%: {within(list(xg.values()))}",
        )
        say(
            "6",
            f"  TreeWalker beats its own full walk in those cells, {tag}",
            str(all(alg(cells[c]) > 1 for c in faster)),
        )
        f = [cell for cell, m in cells.items() if "tl2cgen" in m and m["tl2cgen"]["speedup"] < 1]
        small = sum(1 for c in f if results.design(c)["G"] in (1, 2, 4))
        tl = [m["tl2cgen"]["speedup"] for m in cells.values() if "tl2cgen" in m]
        say(
            "6",
            f"tl2cgen faster, {tag}",
            f"{len(f)}/{len(cells)}, {small} at G<=4; within ±2%: {within(tl)}",
        )
        f = [cell for cell, m in cells.items() if "lleaves" in m and m["lleaves"]["speedup"] < 1]
        small = sum(1 for c in f if results.design(c)["G"] in (1, 2, 4, 8))
        n_lgb = sum(1 for c in cells if results.design(c)["framework"] == "lightgbm")
        ll = [m["lleaves"]["speedup"] for m in cells.values() if "lleaves" in m]
        say(
            "6",
            f"lleaves faster, {tag}",
            f"{len(f)}/{n_lgb}, {small} at G<=8; within ±2%: {within(ll)}",
        )

    for arch in results.MACHINES:
        boundary(arch, v1[arch], "v1's cells")
        boundary(arch, grid.cells(arch), "the factorial")

    # --- Section 4.4 / Table 1: work decomposition ---------------------------------
    w, asym = figures.decomposition(results.work(runs, c.ablation[0]))
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
    # Effects are ratios of latencies; with two ablation runs, each run's value and
    # their spread between the runs.
    research = [
        results.speedups(runs, run).filter(
            (pl.col("arch") == "intel")
            & (pl.col("method") == "treewalker_research")
            & (pl.col("mode") == "serving")
        )
        for run in c.ablation
    ]

    def effect(ds: str, h: int | None, variant: str, against: str = "all-on") -> list[float]:
        """The effect per framework and ablation run, in %."""
        out = []
        for run in research:
            a = figures._anchor(run, ds, h)
            t = {
                (row["framework"], row["variant"]): row["latency_us"]
                for row in a.iter_rows(named=True)
            }
            out += [
                100 * (t[(fw, variant)] / t[(fw, against)] - 1) for fw in FW if (fw, variant) in t
            ]
        return out

    over = "frameworks" if len(c.ablation) == 1 else "frameworks and both ablation runs"
    anchors = {"support": 16, "flchain": 4, "expedia": None}
    for flag in ("|disable_tree_ordering", "|disable_prefix_grouping"):
        for ds, h in anchors.items():
            v = effect(ds, h, flag)
            say(
                "5.7",
                f"{flag.strip('|')[8:]} effect, {ds} (% range over {over})",
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
    # Repeatability: each run's effect per framework and the range between the runs,
    # on every machine, never pooled with the framework spread.
    ablation = pl.concat(
        [results.speedups(runs, run) for run in c.ablation], how="diagonal_relaxed"
    )
    for line in ablation_repeatability(figures.ablation_effects(ablation)):
        say("5.7", *line)

    # --- README: the released default's exact-sums effect ---------------------------
    final = results.CAMPAIGNS["final"].ablation[0]
    if all(results.run_dir(runs, final, a).exists() for a in results.MACHINES):
        for line in exact_sums_effect(results.speedups(runs, final)):
            say("README", *line)

    # --- Section 5.7: unsplit, from the default evaluator's counters -----------------
    for line in unsplit(results.work(runs, c.ablation[0])):
        say("5.7", *line)

    # --- Section 5.6 / Tables 3 and 10: the credit what-if --------------------------
    # The final run's factorial holds whatif-v2 on credit's models; the camera-ready
    # campaign times E1's released scenario cells (scenario-v1) as a run of its own.
    whatif = factorial if c.scenario is None else results.speedups(runs, c.scenario)
    sel = (
        pl.col("dataset").is_in(["credit", "scenario_credit"])
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
                whatif.filter(sel & (pl.col("arch") == arch)), "treewalker_fullwalk"
            ).iter_rows(named=True)
        }
        for arch in results.MACHINES
    }
    for arch in ("arm", "intel"):
        say(
            "5.6",
            f"what-if cells faster than full walk, {arch}",
            f"{sum(v > 1 for v in sc[arch].values())}/{len(sc[arch])}; within ±2%: "
            f"{within(list(sc[arch].values()))}",
        )
    for arch, kv in (("arm", 1), ("arm", 8), ("intel", 1), ("intel", 8)):
        v = [sc[arch][(kv, g)] for g in (4, 16, 64, 128)]
        say(
            "5.6",
            f"what-if speedup range, {arch}, k={kv}",
            f"{r(min(v), 2 if min(v) < 2 else 1)}-{r(max(v), 1)}",
        )
    work = results.work(runs, c.scenario or c.factorial).filter(
        sel & (pl.col("variant") == "all-on")
    )
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
            row["cell"]: row["latency_us"]
            for row in results.rows(d, "treewalker_chunked128").iter_rows(named=True)
        }
        tw = {
            row["cell"]: row["latency_us"]
            for row in results.rows(d, "treewalker", "all-on").iter_rows(named=True)
        }
        by_g: dict[int, list[float]] = {}
        for cell, t in ch.items():
            by_g.setdefault(results.design(cell)["G"], []).append(t / tw[cell])
        say(
            "Table 11",
            f"128-row chunks over whole groups, latency, median over cells, {arch} "
            "(G=256 512 1024)",
            " ".join(r(st.median(by_g[g]), 3) for g in sorted(by_g)),
        )

    # --- App. A.4: re-timing dispositions ----------------------------------------------
    for line in retime_counts(runs, c):
        say("A.4", *line)

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
        # Sensitivity: the row-weighted ratio against the nested-median ratio.
        rw = (
            d.filter(
                (pl.col("mode") == "serving")
                & pl.col("speedup_rw").is_not_null()
                & (pl.col("method") != "treewalker")
                & (pl.col("variant") != "faster-mode")
            )
            .group_by("family")
            .agg(ratio=pl.col("speedup_rw") / pl.col("speedup"))
            .sort("family")
        )
        say(
            "A.6",
            f"sensitivity: row-weighted speedup over the latency ratio by family "
            f"(median, min-max), {arch}",
            "; ".join(
                f"{fam} {r(st.median(v), 3)} ({rng(min(v), max(v), 2)})"
                for fam, v in rw.iter_rows()
            ),
        )
    # The sentinel, a check on the machine: each method's largest drift from the
    # run's first measured sentinel, and how many sentinels exceed the run's flag.
    for arch in ("intel", "arm"):
        run = json.loads((results.run_dir(runs, c.factorial, arch) / "run.json").read_text())
        pct = float((run.get("run") or {}).get("sentinel_drift_pct", 3.0))
        table, _ = analysis.sentinel_table(run.get("sentinel") or {}, pct)
        worst = (
            table.filter(pl.col("drift_pct").is_not_null())
            .group_by("key")
            .agg(
                worst=pl.col("drift_pct").get(pl.col("drift_pct").abs().arg_max()),
                flagged=pl.col("flagged").sum(),
                n=pl.len(),
            )
            .sort("key")
        )
        say(
            "A.6",
            f"sentinel drift per method, largest (sentinels above {pct}%), {arch}",
            "; ".join(
                f"{k.split('/')[0]} {w:+.1f}% ({f}/{n})"
                for k, w, f, n in worst.select("key", "worst", "flagged", "n").rows()
            )
            or "no sentinel",
        )
    return out


def generate(runs: Path, out: Path, campaign: results.Campaign | None = None) -> list[str]:
    lines = [f"[{where}] {what}: {value}" for where, what, value in collect(runs, campaign)]
    out.mkdir(parents=True, exist_ok=True)
    (out / "paper_numbers.txt").write_text("\n".join(lines) + "\n")
    return lines
