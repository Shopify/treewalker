"""The runs as tidy tables: what the paper's figures, tables and numbers read.

`speedups` has one row per machine, cell, method, variant and mode, with the cell's
design parsed from its ID and `analysis.estimates`' numbers. The main estimand is
the camera-ready one (`estimand`), from process 0:

- ``latency_us``: the median over blocks (one (round, batch) pass each) of each
  block's nearest-rank median time per group, in us; ``p5_us`` and ``p95_us``:
  the same convention's 5th and 95th percentiles over blocks, descriptive
  dispersion, not an interval; ``blocks``.
- ``speedup``: the method's latency over TreeWalker's (all-on), so above 1 when
  TreeWalker is faster. In serving mode ``lo`` and ``hi`` are its 95% interval:
  entities and rounds resampled as crossed clusters, the two methods paired, the
  nested medians recomputed; batch mode has no interval.
- ``effect``: a research variant's latency over its ablation denominator
  (``effect_against``: the research build's all-on, or for a monotonic variant
  the same variant with monotonic on, `estimand.paper_against`), with
  ``effect_lo`` and ``effect_hi`` from the same paired draws in serving mode.
- The sensitivity columns: ``us_per_row``, the mean time per row, row-weighted
  (total ticks over total rows); ``speedup_rw``, the paired row-weighted ratio
  over the same groups and blocks, with ``lo_rw`` and ``hi_rw`` from the crossed
  bootstrap (batch: the ratio of us/row); ``p99_us``, the 99th percentile of the
  pooled per-group samples (serving).
- XGBoost native's main row is process 0's (variant ``-``), like every method's.
  Beside it a ``faster-mode`` sensitivity row: the faster mode where XGBoost shows
  two across its extra processes (`analysis.xgboost_faster_mode`), against
  TreeWalker over the same batches and weights, row-weighted, without an interval;
  ``note`` says the extra processes validate only their first 32 groups.

A campaign (`CAMPAIGNS`) names the runs the paper reads: the final run (2026-10)
or the camera-ready rerun. `work` holds TreeWalker's counters per cell and
variant, summed over the cell's groups: operation counts, the same on every machine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import polars as pl

from . import analysis
from . import formats as fm

MACHINES = {"intel": "x86_64", "arm": "aarch64"}
BASELINES = {
    "lightgbm": ("lightgbm_native", "lleaves", "tl2cgen", "quickscorer"),
    "xgboost": ("xgboost_native", "tl2cgen", "quickscorer"),
}
# The released scenario model (grids.toml's whatif-v1 workload: 500 trees, depth 8).
SCENARIO_MODEL = (500, 8)
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


@dataclass(frozen=True, slots=True)
class Campaign:
    """The run IDs (without the machine) a campaign's analysis reads: the factorial
    cells, the ablation runs (the camera-ready ablation runs twice), and E1's
    scenario cells, where they are a run of their own."""

    name: str
    factorial: str
    ablation: tuple[str, ...]
    scenario: str | None = None


# The run IDs follow infra/scripts/startup.sh: SUITE-ARCH, and SUITE-TAG-ARCH for
# a second run NAME@TAG of a suite.
CAMPAIGNS = {
    "final": Campaign("final", "factorial", ("ablation",)),
    "camera-ready": Campaign(
        "camera-ready",
        "camera-ready",
        ("camera-ready-ablation-1", "camera-ready-ablation-2"),
        "scenario-v1",
    ),
}


def campaign(runs: Path, name: str | None = None) -> Campaign:
    """``name``'s campaign; without one, the camera-ready rerun where its factorial
    runs exist, else the final run."""
    if name is not None:
        if name not in CAMPAIGNS:
            raise ValueError(f"unknown campaign {name!r}: one of {', '.join(CAMPAIGNS)}")
        return CAMPAIGNS[name]
    cr = CAMPAIGNS["camera-ready"]
    present = all(run_dir(runs, cr.factorial, arch).exists() for arch in MACHINES)
    return cr if present else CAMPAIGNS["final"]


# Ratios are rounded doubles: abs(1.02 - 1) is 0.020000000000000018, so the
# boundary takes a tolerance far below any measured difference.
_PARITY_TOL = 1e-9


def within(speedups: list[float], pct: float = 2.0) -> int:
    """The cells within ±``pct``% of parity, |speedup - 1| <= pct/100 inclusive,
    reported beside every "faster in N of M" count (v1's rule: the median ratio),
    on the same population."""
    return sum(abs(x - 1) <= pct / 100 + _PARITY_TOL for x in speedups)


def run_dir(runs: Path, suite: str, arch: str) -> Path:
    return runs / f"{suite}-{MACHINES[arch]}"


def design(cell: str) -> dict[str, Any]:
    """A cell's design from its ID (dataset/model/framework/workload): trees ``T``,
    depth ``L``, the survival ``horizon``, the seed ``replicate``, the what-if's
    ``k``, and ``G``: the horizon, the cohort size or the what-if's scenarios (None
    for whole sessions). ``family`` is panel, panel-long (horizons of 256 or
    more), sessions, sessions-filled, cohort or whatif. The released scenario cells
    (E1, ``scenario_credit/lightgbm/whatif-v1-kK-GG``) have no model part: their
    one model is `SCENARIO_MODEL`'s."""
    parts = cell.split("/")
    if len(parts) == 3 and parts[0] == "scenario_credit":
        dataset, framework, workload = parts
        model = "nt{}_md{}".format(*SCENARIO_MODEL)
    elif len(parts) == 4:
        dataset, model, framework, workload = parts
    else:
        raise ValueError(f"unknown cell ID {cell}")
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


@cache
def _speedups(run: Path, arch: str, n_boot: int) -> Any:
    d = analysis.load(run, analysis.SAMPLE_COLUMNS)
    est = analysis.estimates(d, n_boot=n_boot)
    us = 1e6 / est["hz"]
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
            stop = m["stop"].get(r["mode"], {})
            out.append(
                {
                    **base,
                    "method": r["method"],
                    "variant": r["variant"],
                    "mode": r["mode"],
                    "latency_us": r["latency"],
                    "p5_us": r["bp5"],
                    "p95_us": r["bp95"],
                    "blocks": r["blocks"],
                    "speedup": r["ratio"],
                    "lo": r["lo"],
                    "hi": r["hi"],
                    "effect": r["effect"],
                    "effect_lo": r["effect_lo"],
                    "effect_hi": r["effect_hi"],
                    "effect_against": r["effect_against"],
                    "us_per_row": r["per_row"],
                    "speedup_rw": r["ratio_rw"],
                    "lo_rw": r["lo_rw"],
                    "hi_rw": r["hi_rw"],
                    "p99_us": r["p99"] if r["mode"] == "serving" else None,
                    "note": None,
                    "rounds": stop.get("rounds"),
                    "stop_reason": stop.get("reason"),
                }
            )
            if r["method"] == "xgboost_native":
                x = est["xgboost"].get((cell, r["mode"]))
                if x is None or "tw" not in x:
                    continue
                source = "two modes" if x["outcome"] == "two modes" else "process 0"
                out.append(
                    {
                        **out[-1],
                        "variant": "faster-mode",
                        "latency_us": None,
                        "p5_us": None,
                        "p95_us": None,
                        "blocks": None,
                        "speedup": None,
                        "lo": None,
                        "hi": None,
                        "effect": None,
                        "effect_lo": None,
                        "effect_hi": None,
                        "effect_against": None,
                        "us_per_row": x["faster"] * us,
                        "speedup_rw": x["faster"] / x["tw"],
                        "lo_rw": None,
                        "hi_rw": None,
                        "p99_us": None,
                        "note": f"{source}; {analysis.FASTER_MODE_NOTE}",
                    }
                )
    return pl.DataFrame(out, infer_schema_length=None)


def speedups_one(run: Path, arch: str, n_boot: int = 1000) -> Any:
    """`speedups` of one run directory."""
    return _speedups(run, arch, n_boot)


def attempts(runs: Path, suite: str, arch: str) -> list[Path]:
    """The run ``suite`` (an ID without the machine) on ``arch`` and its re-time runs
    (`retime`: ``<suite>-retime-r<rounds>-<machine>``), in that order."""
    retimes = sorted(runs.glob(f"{suite}-retime-r*-{MACHINES[arch]}"))
    return [run_dir(runs, suite, arch), *retimes]


def longest(frames: list[Any]) -> Any:
    """Per (cell, mode), the rows of the attempt with the most rounds (the later on a
    tie): the paper reports the longer attempt, and every attempt stays in the data.
    ``attempt`` names the run directory reported, ``attempts`` counts them."""
    df = pl.concat(
        [f.with_columns(attempt_index=pl.lit(i)) for i, f in enumerate(frames)],
        how="diagonal_relaxed",
    )
    key = ["arch", "cell", "mode"]
    pick = df.group_by(key).agg(
        pl.col("attempt_index").sort_by("rounds", "attempt_index").last(),
        attempts=pl.col("attempt_index").n_unique(),
    )
    return df.join(pick, on=[*key, "attempt_index"]).drop("attempt_index")


def speedups(runs: Path, suite: str, n_boot: int = 1000) -> Any:
    """`speedups` (module docstring) for the run ID ``suite`` (without the machine)
    on both machines, each cell and mode from its longest attempt (`longest`), every
    re-time attempt checked against its parent (`retime.attempt_problems`);
    ``run`` is ``suite``, ``attempt`` the run directory reported. A refused attempt
    leaves its parent's rows reported, its cells marked in ``retime`` ("re-time
    refused: ...")."""
    from . import retime

    out = []
    for arch in MACHINES:
        parent, *retimes = attempts(runs, suite, arch)
        frames = [_speedups(parent, arch, n_boot).with_columns(attempt=pl.lit(parent.name))]
        marks: dict[str, str] = {}
        for attempt in retimes:
            problems = retime.attempt_problems(parent, attempt)
            if problems:
                for c in fm.read_json(attempt / "cells.json"):
                    marks[c] = f"re-time refused: {attempt.name}: {problems[0]}"
                continue
            frames.append(
                _speedups(attempt, arch, n_boot).with_columns(attempt=pl.lit(attempt.name))
            )
        df = longest(frames).with_columns(run=pl.lit(suite))
        out.append(
            df.with_columns(
                retime=pl.col("cell").replace_strict(marks, default=None, return_dtype=pl.String)
            )
        )
    return pl.concat(out, how="diagonal_relaxed")


def rows(df: Any, method: str, variant: str = "-", mode: str = "serving") -> Any:
    """One method's rows of `speedups`, keyed by machine and cell."""
    return df.filter(
        (pl.col("method") == method) & (pl.col("variant") == variant) & (pl.col("mode") == mode)
    )


def baseline_rows(df: Any, mode: str = "serving") -> Any:
    """Every baseline's main rows (process 0, so not XGBoost's faster-mode row), each
    only where it was timed (a method excluded from a cell has none)."""
    names = sorted({b for bs in BASELINES.values() for b in bs})
    return df.filter(
        pl.col("method").is_in(names) & (pl.col("variant") == "-") & (pl.col("mode") == mode)
    )


def best_baseline(df: Any, mode: str = "serving") -> Any:
    """Per machine and cell, the fastest baseline (a descriptive minimum, which
    favours the baselines) and TreeWalker's speedup over it."""
    return (
        baseline_rows(df, mode)
        .sort("latency_us")
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
