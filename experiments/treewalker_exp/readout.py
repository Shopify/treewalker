"""The validation deployment's readout: what the final run's settings rest on.

Each section checks one assumption of the new protocol on a finished run:

- round 0 against later rounds, per method: whether a method learns its inputs
  (branch predictors) over the rounds, and the 1% rule's verdict;
- the position profile: each method's first timed batch and warm-up batch
  (timed last) against its median batch, net of the batches' content;
- XGBoost's modes: the share of batches holding both, per cell and mode, the
  blocks per batch and cluster, and blocks off their cluster's level;
- counters on every XGBoost block, of every process (required on the VMs);
- XGBoost's children against process 0, on the same batch and cluster;
- the sentinel's drift and first-load effect;
- the interface probes' margins;
- the drawn group samples against their expected rows;
- the measured per-row rate, which `treewalker-exp budget --pilot-run` turns into
  the final run's estimate.
"""

from pathlib import Path
from typing import Any

import numpy as np

from . import analysis


def round_zero(samples: Any) -> Any:
    """Per cell, method, variant and mode (process 0): ticks per row in round 0
    over ticks per row in the later rounds."""
    import polars as pl

    s = samples.filter(pl.col("process") == 0) if "process" in samples.columns else samples
    per = s.group_by(
        "cell",
        "method",
        "variant",
        "mode",
        (pl.col("repetition") == 0).alias("first"),
        maintain_order=True,
    ).agg(tpr=pl.col("ticks").sum() / pl.col("rows").sum())
    first = per.filter(pl.col("first")).drop("first").rename({"tpr": "round0"})
    later = per.filter(~pl.col("first")).drop("first").rename({"tpr": "later"})
    return first.join(later, on=["cell", "method", "variant", "mode"]).with_columns(
        ratio=pl.col("round0") / pl.col("later")
    )


def position_profile(samples: Any) -> Any:
    """Per cell, method and mode (process 0): the first timed batch's and the
    warm-up batch's (timed last) ticks per row against the method's median batch,
    net of the batches' content: each batch's time is first divided by the
    median over methods of their time on that batch relative to their own mean."""
    import polars as pl

    s = samples.filter(pl.col("process") == 0) if "process" in samples.columns else samples
    s = s.filter(pl.col("variant").is_in(["-", "all-on"]))
    per = s.group_by("cell", "mode", "method", "batch", maintain_order=True).agg(
        t=pl.col("ticks").sum() / pl.col("rows").sum()
    )
    per = per.with_columns(rel=pl.col("t") / pl.col("t").mean().over("cell", "mode", "method"))
    content = per.group_by("cell", "mode", "batch", maintain_order=True).agg(
        c=pl.col("rel").median()
    )
    per = per.join(content, on=["cell", "mode", "batch"]).with_columns(
        net=pl.col("rel") / pl.col("c")
    )
    last = per.group_by("cell", "mode", maintain_order=True).agg(last=pl.col("batch").max())
    per = per.join(last, on=["cell", "mode"])
    med = per.group_by("cell", "mode", "method", maintain_order=True).agg(
        median=pl.col("net").median()
    )
    first = per.filter(pl.col("batch") == 0).select("cell", "mode", "method", first="net")
    warm = per.filter(pl.col("batch") == pl.col("last")).select(
        "cell", "mode", "method", warm="net"
    )
    return (
        med.join(first, on=["cell", "mode", "method"])
        .join(warm, on=["cell", "mode", "method"])
        .with_columns(
            first=pl.col("first") / pl.col("median"), warm=pl.col("warm") / pl.col("median")
        )
    )


def children(samples: Any, labels: Any) -> Any:
    """Per cell, mode and XGBoost child: its ticks per row over process 0's later
    rounds (a child's timed round follows a full untimed round), on the same
    batches and the same cluster of each batch (`analysis.xgboost_faster_mode`'s
    labels: a batch with one mode has one cluster). Process 0's time is weighted
    by the child's rows in each (batch, cluster); ``ratio`` is null, and
    ``matched`` 0, when process 0 shares no batch and cluster with the child."""
    import polars as pl

    x = samples.filter(pl.col("method") == "xgboost_native").join(
        labels.select("process", "block", "cluster"), on=["process", "block"]
    )
    key = ["cell", "mode", "batch", "cluster"]
    per = x.group_by(*key, "process", (pl.col("repetition") == 0).alias("first")).agg(
        pl.col("ticks").sum(), pl.col("rows").sum()
    )
    main = per.filter((pl.col("process") == 0) & ~pl.col("first")).select(
        *key, main=pl.col("ticks") / pl.col("rows")
    )
    kids = per.filter(pl.col("process") > 0).join(main, on=key, how="left")
    hit = pl.col("main").is_not_null()
    return (
        kids.group_by("cell", "mode", "process")
        .agg(
            ticks=pl.col("ticks").filter(hit).sum(),
            rows=pl.col("rows").filter(hit).sum(),
            main=(pl.col("main") * pl.col("rows")).filter(hit).sum(),
            matched=hit.sum(),
            groups=pl.len(),
        )
        .with_columns(ratio=pl.when(pl.col("matched") > 0).then(pl.col("ticks") / pl.col("main")))
        .sort("cell", "mode", "process")
    )


def process_zero_switches(labels: Any) -> bool:
    """Whether XGBoost's process 0 holds both modes inside any batch: its blocks
    split above `analysis.SPLIT` in log instructions per row, one block a side."""
    import polars as pl

    p0 = labels.filter(pl.col("process") == 0)
    return any(
        analysis.split_modes(g["log_ipr"].to_numpy(), analysis.SPLIT, 1).any()
        for _, g in p0.group_by("batch")
    )


def round_zero_rule(ratios: dict[str, list[float]], threshold: float = 0.01) -> list[str]:
    """The round-0 rule's verdict on one machine: a method is over it when its
    median round-0 ratio exceeds 1 by more than ``threshold``, slower or faster
    (strictly: exactly 1% is within). A method with no ratio, or no method at
    all, has no evidence and is never counted as holding. Round 0 is always a
    forward round of the method order."""
    over: list[str] = []
    none = [] if ratios else ["every method"]
    for method, q in ratios.items():
        if not q:
            none.append(method)
            continue
        med = float(np.median(q))
        if med > 1 + threshold or med < 1 - threshold:
            side = "round 0 slower" if med > 1 else "round 0 faster"
            over.append(f"{method} {med:.4f} ({side})")
    verdict = "FIRES" if over else "not decided" if none else "holds"
    return [
        f"round-0 rule at {threshold:.0%}: {verdict}"
        + (f"; over it: {', '.join(over)}" if over else "")
        + (f"; no evidence: {', '.join(none)}" if none else "")
        + " (round 0 is always a forward round)"
    ]


def xgboost_lines(cell: str, mode: str, r: dict[str, Any]) -> list[str]:
    """XGBoost's modes in one cell and mode: the outcome, counters, the blocks per
    batch and cluster, and the blocks off their cluster's level."""
    import polars as pl

    unread = f"{r['unread_blocks']} of {r['blocks']} blocks without counters"
    flag = " FLAG" if r["unread_blocks"] and r["outcome"] != "no counters" else ""
    if "labels" not in r:
        return [f"  {cell} {mode}: {r['outcome']}, {unread}{flag}"]
    odd = r.get("odd_gap_batches") or []
    if r["outcome"].startswith("two modes"):
        state = (
            f"{r['outcome']}, mixed {r['mixed_batches']} of {r['batches']} ({r['mixed_share']:.0%})"
        )
        flag += " FLAG" if r["mixed_share"] < 0.5 or odd else ""
        state += f", odd-gap batches left out as unknown {odd}" if odd else ""
    else:
        state = f"{r['outcome']}, batch medians span {r['span_pct']:.2f}%"
        flag += " FLAG" if r["span_flag"] else ""
    lab = r["labels"]
    per_batch = []
    for (k,), g in lab.group_by("batch", maintain_order=True):
        if k in odd:
            per_batch.append(f"{k}:odd")
        elif g["faster"].is_null().all():
            per_batch.append(f"{k}:{len(g)}")
        else:
            per_batch.append(f"{k}:{int(g['faster'].sum())}/{int((~g['faster']).sum())}")
    off = lab.filter(pl.col("off_level"))
    centre = lab.group_by("batch", "cluster").agg(c=pl.col("log_ipr").median())
    off = off.join(centre, on=["batch", "cluster"]).sort("process", "block")
    out = [
        f"  {cell} {mode}: {state}, {unread}, {len(off)} blocks off level{flag}",
        "    blocks per batch (batch:faster/slower, batch:blocks with one mode, or "
        "batch:odd for an odd-gap batch): "
        + " ".join(sorted(per_batch, key=lambda x: int(x.split(":")[0]))),
    ]
    if len(off):
        out.append(
            "    off their cluster's level by >0.5% (process/block/batch): "
            + ", ".join(
                f"{p}/{b}/{k} {np.expm1(v - c):+.1%}"
                for p, b, k, v, c in off.select("process", "block", "batch", "log_ipr", "c").rows()
            )
        )
    return out


def readout(run_dir: Path) -> list[str]:
    import polars as pl

    from . import budget as bg

    d = analysis.load(run_dir)
    s, hw = d["samples"], d["hw"]
    host = d["run"].get("host") or {}
    lines = [f"validation readout {run_dir.name}: {host.get('arch')} {host.get('cpu')}"]

    xgb_results = {}
    xgb = s.filter(pl.col("method") == "xgboost_native")
    for (cell, mode), _ in xgb.group_by("cell", "mode", maintain_order=True):
        blocks = analysis.xgboost_blocks(s, hw, cell, mode)
        r = analysis.xgboost_faster_mode(blocks, counters_required=host.get("os") == "linux")
        if r is not None:
            xgb_results[(cell, mode)] = (r, blocks)

    lines.append(
        "\nround 0 over later rounds, per method (median, range; cells off by more than 1%); "
        "XGBoost on process 0, without the cells where process 0 switches mode in a batch:"
    )
    r0 = round_zero(s)
    excluded = []
    keep = []
    for (cell, mode), (r, blocks) in xgb_results.items():
        p0 = blocks.filter(pl.col("process") == 0)
        if "labels" not in r:
            excluded.append(f"{cell} {mode} ({r['outcome']})")
        elif p0["instructions"].null_count():
            n = p0["instructions"].null_count()
            excluded.append(f"{cell} {mode} ({n} process-0 blocks without counters)")
        elif process_zero_switches(r["labels"]):
            excluded.append(f"{cell} {mode} (process 0 switches mode in a batch)")
        else:
            keep.append((cell, mode))
    xr = r0.filter(pl.col("method") == "xgboost_native").join(
        pl.DataFrame(keep, schema={"cell": pl.String, "mode": pl.String}, orient="row"),
        on=["cell", "mode"],
    )
    # Every method process 0 timed has an entry: one missing a whole period
    # (round 0 or the later rounds) in every cell has no ratio, so no evidence.
    p0 = s.filter(pl.col("process") == 0) if "process" in s.columns else s
    ratios: dict[str, list[float]] = {
        m: [] for m in p0["method"].unique(maintain_order=True).to_list()
    }
    for (method,), g in r0.group_by("method", maintain_order=True):
        ratios[method] = (xr if method == "xgboost_native" else g)["ratio"].to_list()
    for method, values in ratios.items():
        if not values:
            lines.append(f"  {method:<24} no evidence")
            continue
        a = np.array(values)
        off = int(((a > 1.01) | (a < 0.99)).sum())
        lines.append(
            f"  {method:<24} {np.median(a):.4f} [{a.min():.4f}, {a.max():.4f}] "
            f"over {len(a)}; {off} off"
        )
    if excluded:
        lines.append(f"  XGBoost cells excluded ({len(excluded)}): " + "; ".join(excluded))
    lines += ["  " + x for x in round_zero_rule(ratios)]

    lines.append(
        "\nposition profile, per method (median over cells), relative to the median method "
        "(an effect common to every method does not show): the first timed batch and the "
        "warm-up batch, timed last, against the median batch, net of content"
    )
    for (method,), g in position_profile(s).group_by("method", maintain_order=True):
        lines.append(
            f"  {method:<24} first {np.median(g['first'].to_numpy()):.4f}, "
            f"warm-up {np.median(g['warm'].to_numpy()):.4f} over {len(g)}"
        )

    lines.append("\nXGBoost's modes and counters, per cell and mode:")
    children_out = []
    for (cell, mode), (r, blocks) in xgb_results.items():
        lines += xgboost_lines(cell, mode, r)
        if blocks["process"].max() == 0:
            continue
        if "labels" not in r:
            children_out.append(f"  {cell} {mode}: not compared ({r['outcome']})")
            continue
        cs = s.filter((pl.col("cell") == cell) & (pl.col("mode") == mode))
        for row in children(cs, r["labels"]).iter_rows(named=True):
            ratio = (
                f"{row['ratio']:.3f} of process 0's later rounds, on {row['matched']} of "
                f"its {row['groups']} blocks"
                if row["ratio"] is not None
                else "- (no batch where process 0 shares this child's mode)"
            )
            children_out.append(f"  {cell} {mode} process {row['process']}: {ratio}")

    lines.append(
        "\nXGBoost's children over process 0's later rounds (a child's timed round follows "
        "a full untimed round), on the same batch and cluster:"
    )
    lines += children_out

    lines += ["", *analysis.sentinel_lines(d["run"])]

    lines.append("\nprobes with a margin under 2%:")
    for cell, m in d["manifests"].items():
        for mode, stop in m["stop"].items():
            for p in stop.get("probes", []):
                if p.get("margin_pct") is not None and p["margin_pct"] < 2:
                    lines.append(
                        f"  {cell} {mode} {p['method']}: {p['chosen']} by {p['margin_pct']:.2f}%"
                    )

    lines.append("\ngroup samples (drawn against expected rows):")
    for cell, m in d["manifests"].items():
        sm = m.get("sample")
        if sm:
            lines.append(
                f"  {cell}: {m.get('timed_groups')} of {sm['pool_groups']} groups, "
                f"{sm['rows']:,} rows against {sm['expected_rows']:,.0f} "
                f"({sm.get('overshoot', 0):+.1%})"
                + (f"; {sm['exception']}" if sm.get("exception") else "")
            )

    pilot = bg.Pilot.from_run(run_dir)
    lines += [
        f"\nmeasured rate: {pilot.secs_per_row_tree_level:.3e} s per timed row-tree-level, "
        f"{pilot.bytes_per_sample:.1f} bytes per sample",
        f"  final-run estimate: treewalker-exp --artifacts-dir <the pulled cache, with every "
        f"factorial cell.json> budget --suite factorial --pilot-run {run_dir} "
        f"--arch {host.get('arch')}",
    ]
    return lines
