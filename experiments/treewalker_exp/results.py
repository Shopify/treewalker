"""The final runs as tidy tables: what the paper's figures, tables and numbers read.

`speedups` has one row per machine, cell, method, variant and mode, with the cell's
design parsed from its ID and `analysis.estimates`' numbers:

- ``us_per_row``: mean time per row, row-weighted (total ticks over total rows);
  ``p5_us`` to ``p99_us``: quantiles of the per-group samples.
- ``speedup``: the method's time over TreeWalker's (all-on), so above 1 when
  TreeWalker is faster. In serving mode it is the paired ratio over the same
  groups and blocks, with ``lo`` and ``hi`` from the crossed entity and round
  bootstrap; in batch mode the ratio of us/row, without an interval.
- LightGBM native is divided by ``drift``, the sentinel's LightGBM drift at the
  cell's position in the run, interpolated between sentinels (the user's decision,
  2026-10-08). It assumes the sentinel cell's drift applies to every LightGBM
  cell; every other method's drift stays within 1.7%, so only LightGBM is
  corrected.
- XGBoost native has a ``headline`` variant beside its process-0 row: the faster
  mode where XGBoost shows two (`analysis.xgboost_faster_mode`), against
  TreeWalker over the same batches and weights, else process 0's time
  (``headline_source`` says which). The headline has no interval.

`work` holds TreeWalker's counters per cell and variant, summed over the cell's
groups: operation counts, the same on every machine.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import analysis

MACHINES = {"intel": "x86_64", "arm": "aarch64"}
BASELINES = {
    "lightgbm": ("lightgbm_native", "lleaves", "tl2cgen", "quickscorer"),
    "xgboost": ("xgboost_native", "tl2cgen", "quickscorer"),
}
_MODEL = re.compile(r"nt(\d+)_md(\d+)(?:_h(\d+))?(?:_r(\d+))?")


_DESIGN_SCHEMA = {
    "dataset": pl.String,
    "framework": pl.String,
    "workload": pl.String,
    "family": pl.String,
    "T": pl.Int64,
    "L": pl.Int64,
    "horizon": pl.Int64,
    "replicate": pl.Int64,
    "k": pl.Int64,
    "G": pl.Int64,
}


def run_dir(runs: Path, suite: str, arch: str) -> Path:
    return runs / f"{suite}-{MACHINES[arch]}"


def design(cell: str) -> dict[str, Any]:
    """A cell's design from its ID (dataset/model/framework/workload): trees ``T``,
    depth ``L``, the survival ``horizon``, the seed ``replicate``, the what-if's
    ``k``, and ``G``: the horizon, the cohort size or the what-if's scenarios (None
    for whole sessions). ``family`` is panel, panel-long (horizons of 256 or
    more), sessions, sessions-filled, cohort or whatif."""
    dataset, model, framework, workload = cell.split("/")
    m = _MODEL.fullmatch(model)
    if m is None:
        raise ValueError(f"unknown model name in {cell}")
    trees, depth, horizon, replicate = (int(x) if x else None for x in m.groups())
    k = None
    if workload == "panel":
        family, g = ("panel-long" if horizon and horizon >= 256 else "panel"), horizon
    elif workload == "sessions":
        family, g = "sessions", None
    elif workload.startswith("cohort"):
        family, g = "cohort", int(workload.rsplit("-n", 1)[1])
    elif workload.startswith("whatif"):
        k = int(re.search(r"-k(\d+)-", workload)[1])  # type: ignore[index]
        family, g = "whatif", int(workload.rsplit("-G", 1)[1])
    else:
        raise ValueError(f"unknown workload in {cell}")
    if dataset == "expedia-filled":
        family = "sessions-filled"
    return {
        "dataset": dataset,
        "framework": framework,
        "workload": workload,
        "family": family,
        "T": trees,
        "L": depth,
        "horizon": horizon,
        "replicate": replicate,
        "k": k,
        "G": g,
    }


def drift(run: dict[str, Any], key: str = "lightgbm_native/-/serving") -> dict[str, float]:
    """Each cell's sentinel drift for ``key``: its ticks per row against the first
    measured sentinel, interpolated linearly at the cell's position in the run (the
    order of ``invocations``' finished cells); 1.0 without a sentinel."""
    records = [
        r
        for r in (run.get("sentinel") or {}).get("records", [])
        if r["role"] == "measure" and r["status"] == "ok" and key in r["totals"]
    ]
    order = [c["id"] for inv in run.get("invocations", []) for c in inv.get("cells_done", [])]
    if len(records) < 2:
        return dict.fromkeys(order, 1.0)
    tpr = np.array([r["totals"][key][0] / r["totals"][key][1] for r in records])
    at = np.array([r["cells_done"] for r in records], dtype=float)
    factor = np.interp(np.arange(len(order)), at, tpr / tpr[0])
    return {cell: float(f) for cell, f in zip(order, factor, strict=True)}


@cache
def _speedups(run: Path, arch: str, n_boot: int) -> Any:
    d = analysis.load(run, analysis.SAMPLE_COLUMNS)
    est = analysis.estimates(d, n_boot=n_boot)
    us = 1e6 / est["hz"]
    lgb_drift = drift(d["run"])
    rows_per_group = dict(
        d["groups"]
        .filter(pl.col("timed"))
        .group_by(pl.col("cell").cast(pl.String))
        .agg(pl.col("rows").mean())
        .iter_rows()
    )
    out = []
    for cell, rows in est["by_cell"].items():
        m = d["manifests"][cell]
        base = {
            "arch": arch,
            "suite": d["run"]["suite"],
            "cell": cell,
            **design(cell),
            "mean_rows": rows_per_group.get(cell),
            "nodes": m["model"].get("nodes"),
        }
        for r in rows:
            f = lgb_drift.get(cell, 1.0) if r["method"] == "lightgbm_native" else 1.0
            stop = m["stop"].get(r["mode"], {})
            out.append(
                {
                    **base,
                    "method": r["method"],
                    "variant": r["variant"],
                    "mode": r["mode"],
                    "us_per_row": r["per_row"] / f,
                    "p5_us": r["p5"] / f,
                    "p50_us": r["p50"] / f,
                    "p95_us": r["p95"] / f,
                    "p99_us": r["p99"] / f if r["mode"] == "serving" else None,
                    "speedup": r["ratio"] / f,
                    "lo": None if r["lo"] is None else r["lo"] / f,
                    "hi": None if r["hi"] is None else r["hi"] / f,
                    "drift": f,
                    "headline_source": None,
                    "rounds": stop.get("rounds"),
                    "stop_reason": stop.get("reason"),
                }
            )
            if r["method"] == "xgboost_native":
                x = est["xgboost"].get((cell, r["mode"]))
                if x is not None and "tw" in x:
                    source = "faster mode" if x["outcome"] == "two modes" else "process 0"
                    head = {"us_per_row": x["faster"] * us, "speedup": x["faster"] / x["tw"]}
                else:
                    outcome = "no mode analysis" if x is None else x["outcome"]
                    source = f"process 0 ({outcome})"
                    head = {"us_per_row": out[-1]["us_per_row"], "speedup": out[-1]["speedup"]}
                out.append(
                    {
                        **out[-1],
                        **head,
                        "variant": "headline",
                        "lo": None,
                        "hi": None,
                        "headline_source": source,
                    }
                )
    return pl.DataFrame(out, infer_schema_length=None)


def speedups(runs: Path, suite: str, n_boot: int = 1000) -> Any:
    """`speedups` (module docstring) for ``suite`` on both machines."""
    return pl.concat(
        [_speedups(run_dir(runs, suite, arch), arch, n_boot) for arch in MACHINES],
        how="diagonal_relaxed",
    )


def rows(df: Any, method: str, variant: str = "-", mode: str = "serving") -> Any:
    """One method's rows of `speedups`, keyed by machine and cell."""
    return df.filter(
        (pl.col("method") == method) & (pl.col("variant") == variant) & (pl.col("mode") == mode)
    )


def baseline_rows(df: Any, mode: str = "serving") -> Any:
    """Every baseline's headline rows: XGBoost native's headline, the others' only
    rows, each only where it was timed (a method excluded from a cell has none)."""
    names = sorted({b for bs in BASELINES.values() for b in bs})
    native = (pl.col("method") == "xgboost_native") & (pl.col("variant") == "headline")
    other = (pl.col("method") != "xgboost_native") & pl.col("method").is_in(names)
    return df.filter((native | other) & (pl.col("mode") == mode))


def best_baseline(df: Any, mode: str = "serving") -> Any:
    """Per machine and cell, the fastest baseline (a descriptive minimum, which
    favours the baselines) and TreeWalker's speedup over it."""
    return (
        baseline_rows(df, mode)
        .sort("us_per_row")
        .group_by("arch", "cell", maintain_order=True)
        .first()
        .select("arch", "cell", best="method", best_speedup="speedup")
    )


@cache
def _work(run: Path) -> Any:
    counters = pl.read_parquet(run / "counters.parquet")
    groups = pl.read_parquet(run / "groups.parquet", columns=["cell", "group", "rows"])
    numeric = [c for c, t in counters.schema.items() if t.is_integer() and c != "group"]
    out = (
        counters.join(groups, on=["cell", "group"])
        .group_by(pl.col("cell").cast(pl.String), pl.col("variant").cast(pl.String))
        .agg(
            *(pl.col(c).sum() for c in numeric),
            groups=pl.len(),
            rows=pl.col("rows").sum(),
        )
    )
    return out.with_columns(
        pl.col("cell").map_elements(design, return_dtype=pl.Struct(_DESIGN_SCHEMA)).alias("design")
    ).unnest("design")


def work(runs: Path, suite: str, arch: str = "intel") -> Any:
    """TreeWalker's counters per cell and variant, summed over the cell's groups, with
    ``groups`` and ``rows`` counted (operation counts: any machine's run serves)."""
    return _work(run_dir(runs, suite, arch))
