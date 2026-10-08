"""A first look at a run: per-group p50/p99, the row-weighted speedup, and one
bootstrap interval per comparison.

PR 4's ``figures`` and ``tables`` replace this with the full estimators. This
module checks the run's schema and the estimators on real data before any suite
is timed:

- Serving p50/p99 are quantiles of per-group samples (one call per group), in
  microseconds. Batch mode is amortized, never tail latency.
- The speedup of TreeWalker over a method is the ratio of their mean time per
  row, row-weighted, over the same groups: total ticks over total rows.
- The interval resamples entities and rounds as crossed clusters: every method
  times every entity once a round, so a round is shared, and methods stay
  matched on (block, group). The intervals are preliminary: PR 4 adds the
  group sample's design and, for XGBoost, processes as a level.
- XGBoost's speed has two modes per process on Intel (its AVX2 block walk runs
  only when a heap buffer is 32-byte aligned). Each XGBoost cell also times it
  in extra processes, and the mode can also switch within a process, so the
  modes are told apart inside each batch, by instructions per row
  (`xgboost_faster_mode`). The headline is the faster cluster of each batch
  holding both modes, row-weighted, against TreeWalker over the same batches,
  with the expected time over processes alongside; a cell with no second mode
  visible headlines process 0's time. The rest of the table is process 0.
"""

from pathlib import Path
from typing import Any

import numpy as np

from . import formats as fm


def load(run_dir: Path) -> dict[str, Any]:
    """A packed run (treewalker-exp pack): run.json, three tables and the manifests."""
    import polars as pl

    from . import pack

    pack.require_packed(run_dir)
    run = fm.read_json(run_dir / "run.json")
    samples = pl.read_parquet(run_dir / "samples.parquet")
    groups = pl.read_parquet(run_dir / "groups.parquet")
    hw = pl.read_parquet(run_dir / "hw.parquet")
    manifests = fm.read_json(run_dir / pack.CELLS)
    return {"run": run, "samples": samples, "groups": groups, "hw": hw, "manifests": manifests}


def bootstrap_ratio(
    ticks: np.ndarray,
    entity: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    rounds: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """The row-weighted ratio ``sum(ticks[:, 1]) / sum(ticks[:, 0])`` and its 95%
    percentile interval.

    ``ticks`` is (n, 2): two methods' samples of the same group in the same block,
    so both cover the same rows and the ratio of totals is row-weighted; ``entity``
    and ``rounds`` are per pair. Entities and rounds are resampled as crossed
    clusters: a round is shared by every entity (each method times all of them in
    one phase), so a round-wide shock is one observation, not one per entity.
    Without ``rounds``, each pair is its own round.
    """
    estimate = float(ticks[:, 1].sum() / ticks[:, 0].sum())
    rounds = np.arange(len(ticks)) if rounds is None else rounds
    ents, ei = np.unique(entity, return_inverse=True)
    rs, ri = np.unique(rounds, return_inverse=True)
    t0 = np.zeros((len(ents), len(rs)))
    t1 = np.zeros((len(ents), len(rs)))
    np.add.at(t0, (ei, ri), ticks[:, 0])
    np.add.at(t1, (ei, ri), ticks[:, 1])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        we = np.bincount(rng.integers(0, len(ents), len(ents)), minlength=len(ents))
        wr = np.bincount(rng.integers(0, len(rs), len(rs)), minlength=len(rs))
        draws[b] = (we @ t1 @ wr) / (we @ t0 @ wr)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return estimate, float(lo), float(hi)


def split_modes(values: np.ndarray, threshold: float, min_side: int = 2) -> np.ndarray:
    """Two modes or one, from log values: split the sorted values at their largest
    gap, between positions that leave at least ``min_side`` values a side, when
    that gap exceeds ``threshold`` (a log ratio). Returns 0 for the lower mode
    and 1 for the upper, per value."""
    n = len(values)
    labels = np.zeros(n, dtype=int)
    if n < 2 * min_side:
        return labels
    order = np.argsort(values)
    gaps = np.diff(values[order])[min_side - 1 : n - min_side]
    if len(gaps) == 0 or gaps.max() <= threshold:
        return labels
    j = int(gaps.argmax()) + min_side
    labels[order[j:]] = 1
    return labels


def xgboost_blocks(samples: Any, hw: Any, cell: str, mode: str) -> Any:
    """XGBoost's blocks in one cell and mode, every process: ticks, rows, batch,
    round and instructions (null where the block's counters were not read in
    full)."""
    import polars as pl

    key = ["process", "block"]
    sel = (
        (pl.col("cell") == cell) & (pl.col("method") == "xgboost_native") & (pl.col("mode") == mode)
    )
    blocks = (
        samples.filter(sel)
        .group_by(key)
        .agg(
            pl.col("ticks").sum(),
            pl.col("rows").sum(),
            pl.col("batch").first(),
            pl.col("repetition").first(),
        )
    )
    if "process" in hw.columns:
        ins = hw.filter(sel).select(
            *key,
            instructions=pl.when(pl.col("status") == "ok").then(pl.col("instructions")),
        )
        blocks = blocks.join(ins, on=key, how="left")
    else:
        blocks = blocks.with_columns(instructions=pl.lit(None, dtype=pl.UInt64))
    return blocks.sort(key)


# A gap in instructions per row above this, inside one batch, separates XGBoost's
# two code paths: instructions within a mode agree to about 0.1%, and preemption
# does not move them.
SPLIT = float(np.log(1.02))


def xgboost_faster_mode(blocks: Any, counters_required: bool = False) -> dict[str, Any] | None:
    """XGBoost's two modes, batch by batch, and its faster one (no interval until PR 4).

    Every process times the same batches, the same groups in the same calls, so
    the modes are compared only inside a batch, never across batches. In each
    batch the blocks with counters split at the largest gap in log instructions
    per row above 2% (`SPLIT`), one block a side: that batch is mixed. A cell needs
    two blocks on the minority sides over its batches, so one stray block is not
    a mode. In a mixed batch the faster cluster is the one with fewer ticks per
    row, its blocks pooled; the headline is that time, weighted by the batches'
    rows over the mixed batches, and TreeWalker is compared over the same batches
    and weights. The smaller of two observed means is at most the faster mode in
    expectation, so noise favours XGBoost; when the faster cluster differs
    between batches, each batch takes its own. That holds while each cluster is
    one code path. A block that switched mode mid-block lies between the levels
    (the shakedown's support/nt1000_md2_h8 has one in 120): it joins the side
    whose level it is nearer, so beside full blocks of both modes it adds at most
    a quarter of the gap to that batch's faster time when it ran mostly fast; in
    a batch with no full faster block it becomes the faster cluster itself. The
    result counts blocks more than 0.5% off their cluster's level in their batch;
    a mixed batch whose gap lies a point or more from the median gap (a cluster
    between the levels) is left out of the headline as unknown, and listed.

    Only batches holding both modes count (decided 2026-10-06): a batch holding
    one is unknown, and nothing links it to the others. A cell with no mixed
    batch reads "no second mode visible", with the span of the batch medians,
    flagged above 2% (content, or a mode every process shares, batch by batch,
    which does not show): a prompt to look at the cell, not a filter, since
    content alone fires it in most Expedia cells. Its headline is process 0's
    time, which is timed like TreeWalker (the user's decision, round 9: a child
    times one round after an untimed one, and adds no mode information there).
    Blocks without counters are left out and counted; with none at all there is
    no mode analysis: "unverifiable" where counters are required (Linux), else
    "no counters". The expected time over processes is the mean of each one's
    ticks per row: each process is one draw of the buffer's alignment, so each
    weighs the same."""
    if blocks.is_empty():
        return None
    import polars as pl

    per_process = blocks.group_by("process", maintain_order=True).agg(
        tpr=pl.col("ticks").sum() / pl.col("rows").sum()
    )
    read = blocks.filter(pl.col("instructions").is_not_null())
    p0 = blocks.filter(pl.col("process") == 0)
    common: dict[str, Any] = {
        "blocks": len(blocks),
        "unread_blocks": len(blocks) - len(read),
        "processes": len(per_process),
        "expected": float(per_process["tpr"].mean()),
        "process0": float(p0["ticks"].sum() / p0["rows"].sum()) if len(p0) else None,
    }
    if read.is_empty():
        return {**common, "outcome": "unverifiable" if counters_required else "no counters"}
    v = np.log((read["instructions"] / read["rows"]).to_numpy())
    batch = read["batch"].to_numpy()
    ticks = read["ticks"].to_numpy().astype(np.float64)
    rows = read["rows"].to_numpy().astype(np.float64)
    # Mode 1 is the higher-instruction cluster of its batch, 0 the lower or only one.
    cluster = np.zeros(len(read), dtype=int)
    members = {int(k): np.flatnonzero(batch == k) for k in np.unique(batch)}
    minority = 0
    for m in members.values():
        lab = split_modes(v[m], SPLIT, 1)
        cluster[m] = lab
        minority += int(min(lab.sum(), len(m) - lab.sum()))
    if minority < 2:
        cluster[:] = 0
    mixed = [k for k, m in members.items() if cluster[m].any()]
    # Blocks off their cluster's level in their batch by more than 0.5% (within a
    # mode, instructions agree to about 0.1%): a block that switched mid-block.
    off = np.zeros(len(read), dtype=bool)
    for m in members.values():
        for c in (0, 1):
            mc = m[cluster[m] == c]
            if len(mc):
                off[mc] = np.abs(v[mc] - np.median(v[mc])) > np.log(1.005)
    # A batch's rows: every block of it times the same groups.
    weight = {k: float(np.median(rows[m])) for k, m in members.items()}
    gap = {
        k: float(np.median(v[m][cluster[m] == 1]) - np.median(v[m][cluster[m] == 0]))
        for k, m in ((k, members[k]) for k in mixed)
    }
    # Mixed batches whose instruction gap lies a point or more from the median
    # gap: a cluster between the levels, such as a block that switched mid-block
    # or a first block off its level, reads as a mode there. They are left out
    # as unknown (selection on instructions, not on time).
    pct = {k: float(np.expm1(g) * 100) for k, g in gap.items()}
    centre = float(np.median(list(pct.values()))) if pct else 0.0
    odd = [k for k in mixed if abs(pct[k] - centre) >= 1.0]
    kept = [k for k in mixed if k not in odd]
    faster = np.full(len(read), None, dtype=object)
    fast_t, time_gap, lower_faster = [], [], []
    for k in kept:
        m = members[k]
        t = [ticks[m][cluster[m] == c].sum() / rows[m][cluster[m] == c].sum() for c in (0, 1)]
        fast = int(t[1] < t[0])
        faster[m] = cluster[m] == fast
        fast_t.append(t[fast])
        lower_faster.append(fast == 0)
        time_gap.append(abs(np.log(t[1] / t[0])))
    labels = read.select("process", "block", "batch", "repetition").with_columns(
        log_ipr=pl.Series(v),
        cluster=pl.Series(cluster),
        off_level=pl.Series(off),
        faster=pl.Series(faster.tolist(), dtype=pl.Boolean),
    )
    medians = [float(np.median(v[m])) for m in members.values()]
    common |= {"batches": len(members), "labels": labels, "off_level": int(off.sum())}
    if not mixed:
        span = float(np.expm1(np.ptp(medians)) * 100)
        return {
            **common,
            "outcome": "no second mode visible",
            "span_pct": span,
            "span_flag": span > 2.0,
            "faster": common["process0"],
            "weights": sorted(weight.items()),
        }
    split: dict[str, Any] = {
        "mixed_batches": len(kept),
        "mixed_share": len(kept) / len(members),
        "odd_gap_batches": odd,
    }
    if not kept:
        return {**common, **split, "outcome": "two modes, none identified"}
    w = np.array([weight[k] for k in kept])
    shares = (
        labels.filter(pl.col("faster").is_not_null())
        .group_by("process", maintain_order=True)
        .agg(pl.col("faster").cast(pl.Float64).mean().alias("share"))["share"]
    )
    return {
        **common,
        **split,
        "outcome": "two modes",
        "faster": float(w @ np.array(fast_t) / w.sum()),
        "weights": [(k, weight[k]) for k in kept],
        "gap_pct": float(np.expm1(np.median([gap[k] for k in kept])) * 100),
        "time_gap_pct": float(np.expm1(np.median(time_gap)) * 100),
        "lower_faster_share": float(np.mean(lower_faster)),
        "processes_all_faster": int((shares == 1).sum()),
        "processes_mixed": int(((shares > 0) & (shares < 1)).sum()),
    }


def batch_weighted(per_batch: Any, weights: list[tuple[int, float]]) -> float:
    """A method's ticks per row over the given batches, each weighted by its rows:
    ``per_batch`` holds each batch's pooled ticks and rows (columns batch, ticks,
    rows)."""
    import polars as pl

    w = per_batch.join(
        pl.DataFrame(weights, schema=["batch", "w"], orient="row").with_columns(
            pl.col("batch").cast(per_batch["batch"].dtype)
        ),
        on="batch",
    )
    return float((w["w"] * w["ticks"] / w["rows"]).sum() / w["w"].sum())


def xgboost_line(mode: str, r: dict[str, Any], us: float) -> str:
    """summarize's line for XGBoost's modes in one cell and mode."""

    n = r["processes"]
    expected = f"expected over processes {r['expected'] * us:.4f} us/row (the mean of {n})"
    unread = f", {r['unread_blocks']} of {r['blocks']} blocks without counters"
    unread = unread if r["unread_blocks"] else ""
    off = f"; {r.get('off_level')} blocks off their cluster's level by >0.5%"
    off = off if r.get("off_level") else ""
    odd = r.get("odd_gap_batches")
    off += (
        f"; FLAG: batches {odd} a point or more off the median gap, left out as unknown"
        if odd
        else ""
    )
    if r["outcome"] == "unverifiable":
        return (
            f"  XGBoost {mode}: modes unverifiable without XGBoost's counters (required on "
            f"Linux); {expected}"
        )
    if r["outcome"] == "no counters":
        return f"  XGBoost {mode}: no counters, no mode analysis; {expected}"
    head = f"  XGBoost {mode} by instructions per row, split above 2% inside a batch{unread}: "
    if r["outcome"] == "no second mode visible":
        return (
            f"{head}no second mode visible in {r['batches']} batches; process 0's time "
            f"{r['faster'] * us:.4f} us/row (timed like TreeWalker); {expected}; "
            f"batch medians span {r['span_pct']:.2f}%"
            + (
                " (FLAG: above 2%; content or a mode every process shares)"
                if r["span_flag"]
                else ""
            )
            + off
        )
    if r["outcome"] == "two modes, none identified":
        return (
            f"{head}two modes, but every mixed batch's gap is off the median gap: no "
            f"headline; process 0's time {r['process0'] * us:.4f} us/row; {expected}{off}"
        )
    return (
        f"{head}two modes in {r['mixed_batches']} of {r['batches']} batches "
        f"({r['mixed_share']:.0%}"
        + (", FLAG: under half" if r["mixed_share"] < 0.5 else "")
        + f"); faster {r['faster'] * us:.4f} us/row (each mixed batch's faster cluster, "
        f"row-weighted); gaps {r['gap_pct']:.1f}% in instructions, {r['time_gap_pct']:.1f}% "
        f"in time (medians over mixed batches); the lower-instruction cluster faster in "
        f"{r['lower_faster_share']:.0%} of them; processes {r['processes_all_faster']} all "
        f"faster, {r['processes_mixed']} mixed of {n}; {expected}{off}"
    )


def summarize(run_dir: Path, reference: str = "treewalker", n_boot: int = 1000) -> list[str]:
    import polars as pl

    d = load(run_dir)
    timer = d["run"]["timer"]["calibration"]
    hz = float(timer["hz"])
    s = d["samples"].with_columns(us=pl.col("ticks") / hz * 1e6)
    if "process" in s.columns:
        s = s.filter(pl.col("process") == 0)
    lines = [
        f"run {run_dir.name}: timer {timer['counter']} at {hz:.0f} Hz ({timer['source']}), "
        f"read-pair overhead p50 {timer['overhead']['p50']} ticks",
    ]
    lines += sentinel_lines(d["run"])
    for cell, m in d["manifests"].items():
        cs = s.filter(pl.col("cell") == cell)
        lines.append(f"\n{cell}: {m['groups']} groups, {m['rows']} rows")
        for e in m["excluded"]:
            lines.append(f"  excluded {e['method']}/{e['variant']}: {e['reason']}")
        sm = m.get("sample")
        if sm:
            exc = f"; {sm['exception']}" if sm.get("exception") else ""
            lines.append(
                f"  sample: fraction {sm['fraction']:.4f}, {m.get('timed_groups')} of "
                f"{sm['pool_groups']} groups, {sm['rows']:,} of {sm['pool_rows']:,} rows "
                f"(expected {sm['expected_rows']:,.0f}, {sm.get('overshoot', 0):+.1%}){exc}"
            )
        oracle = m["validation"]["oracle"]
        lines.append(f"  oracle: {oracle.get('status')} on {oracle.get('rows', 0)} rows")
        for mode, stop in m["stop"].items():
            lines.append(
                f"  {mode}: {stop['rounds']} rounds x {stop['blocks_per_round']} blocks, "
                f"stopped by {stop['reason']}"
            )
            for p in stop.get("probes", []):
                tried = ", ".join(f"{c['interface']} {c['ticks']}" for c in p["candidates"])
                lines.append(f"    probe {p['method']}: {tried} ticks -> {p['chosen']}")
            for method, share in conversion_share(stop.get("conversion", [])).items():
                lines.append(
                    f"    {method} input conversion (f64 to f32), amortized over the batch: "
                    f"{share:.2%} of its ticks"
                )
        xp = m.get("xgboost_processes") or {}
        if xp.get("skipped"):
            lines.append(f"  XGBoost extra processes: none ({xp['skipped']})")
        bad = [p for p in xp.get("processes", []) if p["status"] != "ok"]
        for p in bad:
            lines.append(f"  XGBoost process {p['process']} {p['mode']}: {p['status']}")
        fasters = {}
        for mode in m["stop"]:
            # On the Linux VMs (x86 and Arm) XGBoost's counters are required.
            required = (d["run"].get("host") or {}).get("os") == "linux"
            r = xgboost_faster_mode(
                xgboost_blocks(d["samples"], d["hw"], cell, mode), counters_required=required
            )
            if r is None:
                continue
            lines.append(xgboost_line(mode, r, 1e6 / hz))
            if "weights" in r:
                fasters[mode] = r
        lines.append(
            f"  {'method':<24}{'variant':<34}{'mode':<8}{'p50 us':>10}{'p99 us':>10}"
            f"{'us/row':>10}{'speedup':>9}  95% interval (preliminary)"
        )
        stats = (
            cs.group_by("method", "variant", "mode")
            .agg(
                p50=pl.col("us").quantile(0.5),
                p99=pl.col("us").quantile(0.99),
                per_row=pl.col("us").sum() / pl.col("rows").sum(),
            )
            .sort("mode", "per_row")
        )
        groups = d["groups"].filter(pl.col("cell") == cell).select("group", "entity")
        for row in stats.iter_rows(named=True):
            interval = ""
            ref = cs.filter(
                (pl.col("method") == reference)
                & (pl.col("variant") == "all-on")
                & (pl.col("mode") == row["mode"])
            )
            ratio = float("nan")
            if row["mode"] == "serving" and not ref.is_empty():
                this = cs.filter(
                    (pl.col("method") == row["method"])
                    & (pl.col("variant") == row["variant"])
                    & (pl.col("mode") == "serving")
                )
                paired = (
                    ref.select("block", "group", "repetition", t0="ticks")
                    .join(this.select("block", "group", t1="ticks"), on=["block", "group"])
                    .join(groups, on="group")
                )
                ticks = paired.select("t0", "t1").to_numpy().astype(np.float64)
                ratio, lo, hi = bootstrap_ratio(
                    ticks,
                    paired["entity"].to_numpy(),
                    n_boot,
                    rounds=paired["repetition"].to_numpy(),
                )
                interval = f"[{lo:.3f}, {hi:.3f}]"
            elif not ref.is_empty():
                ratio = row["per_row"] / float(ref["us"].sum() / ref["rows"].sum())
            p99 = f"{row['p99']:>10.2f}" if row["mode"] == "serving" else f"{'-':>10}"
            variant = "[process 0]" if row["method"] == "xgboost_native" else row["variant"]
            lines.append(
                f"  {row['method']:<24}{variant[:33]:<34}{row['mode']:<8}"
                f"{row['p50']:>10.2f}{p99}{row['per_row']:>10.4f}{ratio:>9.3f}  {interval}"
            )
        # XGBoost's faster mode against TreeWalker over the same batches and weights.
        for mode, r in fasters.items():
            ref = cs.filter(
                (pl.col("method") == reference)
                & (pl.col("variant") == "all-on")
                & (pl.col("mode") == mode)
            )
            if ref.is_empty():
                continue
            per_batch = ref.group_by("batch").agg(pl.col("ticks").sum(), pl.col("rows").sum())
            tw = batch_weighted(per_batch, r["weights"])
            label = (
                "[faster mode, all processes]"
                if r["outcome"] == "two modes"
                else "[no second mode visible]"
            )
            lines.append(
                f"  {'xgboost_native':<24}{label:<34}{mode:<8}"
                f"{'-':>10}{'-':>10}{r['faster'] / hz * 1e6:>10.4f}{r['faster'] / tw:>9.3f}"
                "  no interval until PR 4"
            )
    return lines


def sentinel_table(s: dict[str, Any], drift_pct: float) -> tuple[Any, int | None]:
    """The sentinel's attempts (``run.json``'s ``sentinel``): one row per attempt
    and ``method/variant/mode``, and one with a null key for an attempt that
    measured nothing. Ticks per row; against the run's first measured sentinel
    (the reference, which has none itself) the ratio and the drift, flagged beyond
    ``drift_pct`` either way (exactly ``drift_pct`` is within); and for a measure
    right after a warm-up, the warm-up's ticks per row over the measure's, minus
    1 (the first-load effect).
    Undefined values are null; ``status`` says why. Returns the table and the
    reference's index (None without one)."""
    import polars as pl

    def per_row(r: dict[str, Any] | None) -> dict[str, float | None]:
        if r is None or r["status"] != "ok":
            return {}
        return {k: t / n if n else None for k, (t, n) in r["totals"].items()}

    records = s.get("records") or []
    ref = next((r for r in records if r["status"] == "ok" and r["role"] == "measure"), None)
    ref_tpr = per_row(ref)
    out = []
    for i, r in enumerate(records):
        now = per_row(r)
        prev = records[i - 1] if i else None
        after_warm_up = r["role"] == "measure" and prev is not None and prev["role"] == "warm-up"
        warm = per_row(prev) if after_warm_up else {}
        for key in list(r["totals"]) or [None]:
            ticks, rows = r["totals"][key] if key is not None else (None, None)
            v = now.get(key) if key is not None else None
            base = ref_tpr.get(key) if r["role"] == "measure" and r is not ref else None
            ratio = v / base if v is not None and base else None
            w = warm.get(key) if key is not None else None
            out.append(
                {
                    "sentinel": r["index"],
                    "role": r["role"],
                    "status": r["status"],
                    "cells_done": r["cells_done"],
                    "unix_time": r["unix_time"],
                    "key": key,
                    "ticks": ticks,
                    "rows": rows,
                    "ticks_per_row": v,
                    "ratio": ratio,
                    "drift_pct": None if ratio is None else (ratio - 1) * 100,
                    # A small tolerance: a drift of exactly drift_pct is within.
                    "flagged": None if ratio is None else abs(ratio - 1) * 100 > drift_pct + 1e-9,
                    "first_load_pct": None if w is None or not v else (w / v - 1) * 100,
                }
            )
    schema = {
        "sentinel": pl.Int64,
        "role": pl.String,
        "status": pl.String,
        "cells_done": pl.Int64,
        "unix_time": pl.Int64,
        "key": pl.String,
        "ticks": pl.Int64,
        "rows": pl.Int64,
        "ticks_per_row": pl.Float64,
        "ratio": pl.Float64,
        "drift_pct": pl.Float64,
        "flagged": pl.Boolean,
        "first_load_pct": pl.Float64,
    }
    return pl.DataFrame(out, schema=schema), None if ref is None else ref["index"]


def sentinel_lines(run: dict[str, Any]) -> list[str]:
    """The sentinel's attempts, from ``run.json``: role, status, the drift flags
    and the first-load effect (`sentinel_table`)."""
    import polars as pl

    s = run.get("sentinel") or {}
    if s.get("skipped") and not s.get("records"):
        return [f"sentinel: none ({s['skipped']})"]
    if not s.get("records"):
        return []
    pct = float((run.get("run") or {}).get("sentinel_drift_pct", 3.0))
    t, ref = sentinel_table(s, pct)
    out = [f"sentinel {s['cell']}: reference {ref}, drift above {pct}%"]
    for (index,), g in t.group_by("sentinel", maintain_order=True):
        r = g.row(0, named=True)
        flagged = [
            f"{k} {d:+.1f}%"
            for k, d in g.filter(pl.col("flagged")).select("key", "drift_pct").rows()
        ]
        first = [
            f"{k} {v:+.1f}%"
            for k, v in g.filter(pl.col("first_load_pct").is_not_null())
            .select("key", "first_load_pct")
            .rows()
        ]
        out.append(
            f"  {index} {r['role']} after {r['cells_done']} cells: {r['status']}"
            + (f"; flagged {', '.join(flagged)}" if flagged else "")
            + (f"; first-load effect {', '.join(first)}" if first else "")
        )
    return out


def conversion_share(passes: list[dict[str, Any]]) -> dict[str, float]:
    """Each method's input-conversion share over its conversion passes: total
    conversion ticks over total ticks."""
    totals: dict[str, list[int]] = {}
    for p in passes:
        t = totals.setdefault(p["method"], [0, 0])
        t[0] += p["conversion_ticks"]
        t[1] += p["ticks"]
    return {m: c / t for m, (c, t) in totals.items() if t > 0}
