"""Checks before reading any result.

Validity gates, which block the data (`gates`; ``check-run`` exits non-zero when
one fails):

- the run's tables are present (``stages`` from output schema 6);
- no failed cell: every cell of the manifest finished, and every failed attempt
  of a cell (``failed/``) or of a sentinel was followed by a successful attempt
  of the same cell or sentinel; the resolved failures are listed, not gated;
- every method validated on its timed groups, failing closed: per cell (and per
  sentinel cell), a passing nonempty stage oracle, every required method timed,
  each timed method and mode with a passing check of the interface it timed on
  the cell's timed rows (`validation_failures`), process 0's samples covering
  exactly the timed groups in every round of every timed method and mode
  (`sample_failures`), and tl2cgen's float64 checks recording the rows they
  attribute to its float64 aliasing defect, which are listed;
- every TreeWalker forest with the run's ``exact_sums``, which the paper's suites
  must set off: per cell and sentinel cell, the ``exact_sums`` record (every
  forest off when the setting is off; every forest as production reports when it
  is on). A run older than output schema 6 has no record, which is said, not
  gated, unless a setting is required.

Warnings, which only print (`warnings`): the sentinel (a method's median drift
above 2%, or two consecutive sentinels above 2%); the layout check (the median
over real-model cells of the 64-byte-aligned build's latency over the default
build's outside ±2%); G=1 at the T=500, L=8 survival anchors outside 0.85-1.15x
(the algorithmic speedup); and E1's Intel full-walk latency more than 10% from the
released ``scenario_credit_results_intel.csv``. Each response is fixed before
the data: investigate and report, keep every attempt, never rerun for a better
sample.
"""

from __future__ import annotations

import csv
import io
import statistics as st
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl

from . import analysis, estimand
from . import formats as fm

SENTINEL_PCT = 2.0
LAYOUT_PCT = 2.0
G1_RANGE = (0.85, 1.15)
E1_PCT = 10.0
G1_ANCHORS = [
    f"{ds}/nt500_md8_h1/{fw}/panel"
    for ds in ("support", "flchain")
    for fw in ("lightgbm", "xgboost")
]
E1_RELEASED = "neurips2026:paper/experiments/data/scenario_credit_results_intel.csv"


# Suites the paper's camera-ready numbers come from: every forest must have exact sums off.
PAPER_SUITES = {"camera-ready", "camera-ready-ablation", "scenario-v1"}
# The output schema from which runs record the exact_sums setting per forest and
# write the stages table.
EXACT_SUMS_SCHEMA = STAGES_SCHEMA = 6
TABLES = ("samples", "groups", "counters", "hw")


def timed_rows(m: dict[str, Any]) -> int:
    sample = m.get("sample")
    return int(sample["rows"]) if sample else int(m["rows"])


def validation_failures(m: dict[str, Any]) -> list[str]:
    """Why a cell's timed methods are not all validated on its timed groups, failing
    closed: the stage oracle must have passed on a nonempty set of rows; every
    required method must be timed; every timed method must name the interface it
    timed in each of the cell's modes, with a check of that interface on the
    cell's timed rows that passed and none that failed. A method excluded from the
    cell is not timed, and a candidate interface the probe did not choose is not
    timed either: their failed checks do not count (`diagnostics` lists them).
    Where every group is one row, a method times its one-row interface, which its
    loop candidate (the one-row call on every row) validated; batch-mode checks
    carry the validated candidate's name. tl2cgen's float64 checks need their
    record of the rows attributed to its aliasing defect."""
    out = []
    oracle = m.get("validation", {}).get("oracle") or {}
    if oracle.get("status") != "pass" or not oracle.get("rows"):
        out.append(f"stage oracle {oracle.get('status')!r} on {oracle.get('rows', 0)} rows")
    rows = timed_rows(m)
    excluded = {(e["method"], e["variant"]) for e in m.get("excluded", [])}
    checks = [
        c
        for c in m.get("validation", {}).get("checks", [])
        if (c["method"], c["variant"]) not in excluded
    ]
    methods = m.get("methods") or []
    if not methods:
        out.append("no timed method")
    timed = {meth["method"] for meth in methods}
    out += [
        f"required method {r} not timed" for r in m.get("required_methods", []) if r not in timed
    ]
    modes = set(m.get("stop") or {})
    for meth in methods:
        ifaces = meth.get("interfaces") or {}
        for mode in sorted(modes - set(ifaces)):
            out.append(f"{meth['method']}/{meth['variant']} {mode}: no timed interface")
        for mode, iface in ifaces.items():
            names = {iface}
            if iface == meth.get("interface_one_row"):
                names |= {f"{iface} loop", f"{iface} 1-row loop"}
            if mode == "batch":
                # Batch checks run the batch path (driver.rs, outputs) but are labelled
                # with the validated candidate's name.
                names |= set(meth.get("validated") or [])
            hits = [
                c
                for c in checks
                if (c["method"], c["variant"], c["mode"]) == (meth["method"], meth["variant"], mode)
                and c["interface"] in names
            ]
            what = f"{meth['method']}/{meth['variant']} {mode} {iface}"
            if not hits:
                out.append(f"{what}: no validation check")
            elif any(not c["passed"] for c in hits):
                out.append(f"{what}: a check of the timed interface failed")
            elif not any(c["rows"] == rows for c in hits):
                out.append(f"{what}: not validated on the {rows} timed rows")
            # The defect record is attributed from outputs: the timed interface's checks
            # that compared outputs need it; a call that failed has none to attribute.
            compared = [c for c in hits if c.get("against") != "call"]
            f64_tl2cgen = meth["method"] == "tl2cgen" and meth.get("precision") == "F64"
            if f64_tl2cgen and not all("defect" in c for c in compared):
                out.append(f"{what}: no record of the rows attributed to the defect")
    return out


def diagnostics(m: dict[str, Any]) -> list[str]:
    """Failed checks that do not gate: excluded methods' and unchosen candidates'."""
    excluded = {(e["method"], e["variant"]) for e in m.get("excluded", [])}
    chosen = {
        (meth["method"], meth["variant"], mode, iface)
        for meth in m.get("methods") or []
        for mode, iface in (meth.get("interfaces") or {}).items()
    }
    out = []
    for c in m.get("validation", {}).get("checks", []):
        if c["passed"]:
            continue
        key = (c["method"], c["variant"], c["mode"], c["interface"])
        why = "excluded" if key[:2] in excluded else "not chosen" if key not in chosen else None
        if why:
            out.append(f"{c['method']}/{c['variant']} {c['mode']} {c['interface']} ({why})")
    return out


def exact_sums_failures(m: dict[str, Any], setting: bool) -> list[str]:
    """Why a cell's TreeWalker forests disagree with the run's ``exact_sums``."""
    rec = m.get("exact_sums")
    if rec is None:
        return ["no exact_sums record"]
    if rec.get("setting") != setting:
        return [f"the cell records exact_sums = {rec.get('setting')}, the run {setting}"]
    forests = rec.get("forests") or {}
    if not forests:
        return ["no forest recorded"]
    if setting:
        base = forests.get("")
        if base is None:
            return ["no production forest recorded"]
        want = base["exact_sums"]
    else:
        want = False
    out = [
        f"forest {k or '(production)'} reports exact_sums = {f['exact_sums']}"
        for k, f in forests.items()
        if f["exact_sums"] != want
    ]
    oracle = m["validation"].get("oracle", {})
    if "exact_sums" in oracle and oracle["exact_sums"] != want:
        out.append(f"the oracle's forest reports exact_sums = {oracle['exact_sums']}")
    return out


def _unit(sentinel: bool) -> Any:
    """The key a cell's rows carry: its ID, or for a sentinel cell its attempt and ID
    (the packed sentinel tables' leading ``sentinel`` column)."""
    import polars as pl

    cell = pl.col("cell").cast(pl.String)
    if not sentinel:
        return cell
    return pl.format("sentinel {} {}", pl.col("sentinel").cast(pl.String).str.zfill(3), cell)


def sample_failures(
    run: Path, cells: dict[str, Any], sentinel: bool = False
) -> dict[str, list[str]]:
    """Per cell, how process 0's samples disagree with what the cell says it timed.

    Every timed method and mode must have samples in exactly the cell's recorded
    rounds (``stop[mode].rounds``). Serving: each round holds every timed group
    exactly once, by group ID, and no other, each sample with its group's row
    count. Batch, whose samples are calls of ``n_groups`` groups starting at
    ``group``: each round holds ``blocks_per_round`` blocks whose calls cover the
    timed groups' count and rows (the schema records no call's member groups, so
    their identities are not checked). Samples of any method the cell does not
    time, an excluded method's included, fail. ``sentinel`` reads the packed
    sentinel tables, whose cells are keyed ``sentinel NNN CELL``."""
    import polars as pl

    base = run / "sentinel" if sentinel else run
    unit = _unit(sentinel)
    groups = pl.read_parquet(base / "groups.parquet").select(
        unit=unit, group=pl.col("group"), g_rows=pl.col("rows"), timed=pl.col("timed")
    )
    timed = groups.filter(pl.col("timed"))
    n_timed = dict(timed.group_by("unit").agg(pl.len()).iter_rows())
    rows_timed = dict(timed.group_by("unit").agg(pl.col("g_rows").sum()).iter_rows())
    key = ["unit", "method", "variant", "mode"]
    s = (
        pl.scan_parquet(base / "samples.parquet")
        .filter(pl.col("process") == 0)
        .select(
            unit=unit,
            method=pl.col("method").cast(pl.String),
            variant=pl.col("variant").cast(pl.String),
            mode=pl.col("mode").cast(pl.String),
            repetition=pl.col("repetition").cast(pl.Int64),
            block=pl.col("block").cast(pl.Int64),
            group=pl.col("group").cast(pl.Int64),
            n_groups=pl.col("n_groups").cast(pl.Int64),
            rows=pl.col("rows").cast(pl.Int64),
        )
        .collect()
    )
    rounds = s.group_by(key).agg(
        rounds=pl.col("repetition").n_unique(), reps=pl.col("repetition").unique().sort()
    )
    srv = s.filter(pl.col("mode") == "serving").join(groups, on=["unit", "group"], how="left")
    serving_bad = srv.group_by([*key, "repetition"]).agg(
        n=pl.len(),
        distinct=pl.col("group").n_unique(),
        untimed=(~pl.col("timed").fill_null(False)).sum(),
        multi=(pl.col("n_groups") != 1).sum(),
        wrong_rows=(pl.col("rows") != pl.col("g_rows")).fill_null(True).sum(),
    )
    batch_bad = (
        s.filter(pl.col("mode") == "batch")
        .group_by([*key, "repetition"])
        .agg(
            groups=pl.col("n_groups").sum(),
            rows=pl.col("rows").sum(),
            blocks=pl.col("block").n_unique(),
        )
    )
    per_round: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for frame in (serving_bad, batch_bad):
        for row in frame.iter_rows(named=True):
            per_round.setdefault(
                (row["unit"], row["method"], row["variant"], row["mode"]), []
            ).append(row)
    seen = {
        (r["unit"], r["method"], r["variant"], r["mode"]): r for r in rounds.iter_rows(named=True)
    }
    out: dict[str, list[str]] = {}
    for name, m in cells.items():
        want = {
            (meth["method"], meth["variant"], mode)
            for meth in m.get("methods") or []
            for mode in (meth.get("interfaces") or {})
        }
        got = {k[1:] for k in seen if k[0] == name}
        bad = [f"{'/'.join(k)}: no samples" for k in sorted(want - got)]
        bad += [
            f"{'/'.join(k)}: samples of a method the cell does not time" for k in sorted(got - want)
        ]
        n = n_timed.get(name)
        if n is None:
            bad.append("no timed group in the groups table")
            out[name] = bad
            continue
        for k in sorted(want & got):
            what = "/".join(k)
            stop = (m.get("stop") or {}).get(k[2]) or {}
            if seen[(name, *k)]["rounds"] != stop.get("rounds"):
                bad.append(
                    f"{what}: {seen[(name, *k)]['rounds']} rounds sampled, "
                    f"{stop.get('rounds')} recorded"
                )
            for r in per_round.get((name, *k), []):
                rep = r["repetition"]
                if k[2] == "serving":
                    if r["n"] != n or r["distinct"] != n or r["untimed"] or r["multi"]:
                        bad.append(
                            f"{what} round {rep}: {r['n']} samples of {r['distinct']} groups "
                            f"({r['untimed']} untimed or unknown), not each of the {n} timed "
                            "groups once"
                        )
                    if r["wrong_rows"]:
                        bad.append(
                            f"{what} round {rep}: {r['wrong_rows']} samples whose rows differ "
                            "from their group's"
                        )
                else:
                    bpr = stop.get("blocks_per_round")
                    if (
                        r["groups"] != n
                        or r["rows"] != rows_timed[name]
                        or (bpr is not None and r["blocks"] != bpr)
                    ):
                        bad.append(
                            f"{what} round {rep}: calls cover {r['groups']} groups and "
                            f"{r['rows']} rows in {r['blocks']} blocks, not {n} groups, "
                            f"{rows_timed[name]} rows, {bpr} blocks"
                        )
        if bad:
            out[name] = bad
    return out


def _listed(lines: list[str], what: str, bad: dict[str, list[str]]) -> None:
    lines.append(f"  FAIL {what}: {len(bad)} cells")
    for cell, f in list(bad.items())[:20]:
        lines.append(f"       {cell}: {'; '.join(f)}")


def open_failures(run: Path, cells: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The run's failed cells (``failed/<cell>.json``), split into the open ones and
    those resolved by a later attempt of the same cell that finished (it is in
    cells.json). Every failed attempt stays in the data; only open ones gate."""
    open_: list[str] = []
    resolved: list[str] = []
    d = run / "failed"
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            cid = fm.read_json(p).get("id")
        except OSError, ValueError:
            cid = None
        (resolved if cid in cells else open_).append(p.name)
    return open_, resolved


def open_sentinel_failures(records: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    """Sentinel attempts not ok, split into the open ones and those retried with
    success later (a later record of the same index that is ok)."""
    open_: list[Any] = []
    retried: list[Any] = []
    for i, r in enumerate(records):
        if r.get("status") == "ok":
            continue
        later = any(
            q.get("index") == r.get("index") and q.get("status") == "ok" for q in records[i + 1 :]
        )
        (retried if later else open_).append(r)
    return open_, retried


def gates(run: Path, expect_exact_sums: bool | None = None) -> tuple[list[str], bool]:
    """The validity gates for one packed run: report lines and whether all pass.
    ``expect_exact_sums`` is the setting the run must have; by default off for the
    paper's suites (`PAPER_SUITES`) and any value elsewhere. Runs older than
    output schema 6 record no exact_sums setting and write no stages table, which
    is said, not gated; a run of schema 6 or later must have both."""
    lines, ok = [], True
    info = fm.read_json(run / "run.json")
    schema = int(info.get("output_schema") or 0)
    names = [*TABLES, *(("stages",) if schema >= STAGES_SCHEMA else ())]
    absent = [t for t in names if not (run / f"{t}.parquet").exists()]
    absent += [] if (run / "cells.json").exists() else ["cells.json"]
    if absent:
        lines.append(f"  FAIL tables missing: {', '.join(absent)}")
        return lines, False
    lines.append(f"  ok   tables present ({', '.join(names)}; output schema {schema})")
    cells = fm.read_json(run / "cells.json")
    failed, resolved = open_failures(run, cells)
    expected = (info.get("manifest") or {}).get("cells")
    missing = 0 if expected is None else expected - len(cells)
    sentinels = (info.get("sentinel") or {}).get("records") or []
    bad_sentinels, retried = open_sentinel_failures(sentinels)
    if failed or missing > 0 or bad_sentinels or not cells:
        ok = False
        lines.append(
            f"  FAIL failed cells: {len(failed)} in failed/ ({', '.join(failed[:5])}"
            f"{', ...' if len(failed) > 5 else ''}); {max(missing, 0)} of {expected} cells "
            f"missing from cells.json; {len(bad_sentinels)} sentinel attempts not ok"
        )
        for r in bad_sentinels[:10]:
            lines.append(f"       sentinel {r.get('index')} ({r.get('role')}): {r.get('status')}")
    else:
        lines.append(f"  ok   no failed cell ({len(cells)} cells, {len(sentinels)} sentinels ok)")
    if resolved or retried:
        lines.append(
            f"       resolved: {len(resolved)} failed cells finished on a later attempt "
            f"({', '.join(resolved[:5])}{', ...' if len(resolved) > 5 else ''}); "
            f"{len(retried)} failed sentinel attempts succeeded on a retry"
        )
    sentinel_cells: dict[str, Any] = {}
    packed = (
        fm.read_json(run / "sentinel" / "cells.json")
        if (run / "sentinel" / "cells.json").exists()
        else {}
    )
    for attempt, doc in packed.items():
        for cell, m in doc.items():
            sentinel_cells[f"sentinel {attempt} {cell}"] = m
    # Every successful sentinel attempt must be packed with its tables.
    missing_sentinel = [
        f"sentinel {int(r['index']):03d}: recorded ok, no packed manifest"
        for r in sentinels
        if r.get("status") == "ok" and f"{int(r['index']):03d}" not in packed
    ]
    if sentinel_cells:
        missing_sentinel += [
            f"sentinel/{t}.parquet missing"
            for t in names
            if not (run / "sentinel" / f"{t}.parquet").exists()
        ]
    bad = {
        cell: f for cell, m in {**cells, **sentinel_cells}.items() if (f := validation_failures(m))
    }
    bad_samples = sample_failures(run, cells)
    if sentinel_cells and not missing_sentinel:
        bad_samples |= sample_failures(run, sentinel_cells, sentinel=True)
    if bad or bad_samples or missing_sentinel:
        ok = False
        if missing_sentinel:
            lines.append(f"  FAIL sentinel evidence: {'; '.join(missing_sentinel[:10])}")
        if bad:
            _listed(lines, "validation", bad)
        if bad_samples:
            _listed(lines, "samples against the timed methods, rounds and groups", bad_samples)
    else:
        lines.append(
            "  ok   every timed method validated on its timed groups; process 0's samples hold "
            "the recorded rounds, each serving round every timed group once by ID with its "
            "rows, each batch round the timed groups' count and rows in its blocks; no other "
            f"method sampled ({len(cells)} cells, {len(sentinel_cells)} sentinel cells)"
        )
    notes = sum(len(diagnostics(m)) for m in cells.values())
    if notes:
        lines.append(f"       {notes} failed checks of excluded methods or unchosen candidates")
    attributed = [
        (cell, c["defect"])
        for cell, m in cells.items()
        for c in m["validation"]["checks"]
        if c["method"] == "tl2cgen" and c.get("defect", {}).get("affected_rows")
    ]
    by_cell: dict[str, dict[str, Any]] = {}
    for cell, d in attributed:
        prev = by_cell.get(cell)
        if prev is None or d["affected_rows"] > prev["affected_rows"]:
            by_cell[cell] = d
    lines.append(f"       tl2cgen's attributed rows (aliasing defect): {len(by_cell)} cells")
    for cell, d in sorted(by_cell.items()):
        lines.append(
            f"         {cell}: {d['affected_rows']} rows, {d['affected_differ']} differ, "
            f"largest error {d['affected_max_abs']:.3g}"
        )
    setting = (info.get("run") or {}).get("exact_sums")
    if expect_exact_sums is None and info.get("suite") in PAPER_SUITES:
        expect_exact_sums = False
    if setting is None and schema < EXACT_SUMS_SCHEMA and expect_exact_sums is None:
        lines.append(f"  --   exact sums: not recorded (output schema {schema} predates them)")
        return lines, ok
    if setting is None:
        ok = False
        lines.append(f"  FAIL exact sums: the run records no setting (output schema {schema})")
        return lines, ok
    if expect_exact_sums is not None and setting != expect_exact_sums:
        ok = False
        lines.append(
            f"  FAIL exact sums: the run sets {setting}; suite {info.get('suite')} needs "
            f"{expect_exact_sums}"
        )
    bad = {
        cell: f
        for cell, m in {**cells, **sentinel_cells}.items()
        if (f := exact_sums_failures(m, setting))
    }
    if bad:
        ok = False
        _listed(lines, f"exact sums ({setting})", bad)
    else:
        lines.append(f"  ok   every TreeWalker forest with exact_sums = {setting}")
    return lines, ok


def sentinel_warnings(info: dict[str, Any], pct: float = SENTINEL_PCT) -> list[str]:
    """A method whose median drift over the run's sentinels exceeds ``pct``, or whose
    drift exceeds it at two consecutive sentinels. Failed attempts are listed; with
    no measured drift, nothing is called stable."""
    records = (info.get("sentinel") or {}).get("records") or []
    failed = [r for r in records if r.get("status") != "ok"]
    out = [
        f"  WARN sentinel attempt {r.get('index')} ({r.get('role')}): {r.get('status')}"
        for r in failed
    ]
    table, _ = analysis.sentinel_table(info.get("sentinel") or {}, pct)
    drifts = (
        table.filter(pl.col("drift_pct").is_not_null()).sort("sentinel")
        if not table.is_empty() and "drift_pct" in table.columns
        else table.clear()
    )
    if drifts.is_empty():
        return [
            f"  WARN sentinel: no measured drift ({len(records)} attempts, {len(failed)} failed)",
            *out,
        ]
    for (key,), d in drifts.partition_by("key", as_dict=True, maintain_order=True).items():
        v = d["drift_pct"].to_list()
        med = st.median(v)
        above = [abs(x) > pct for x in v]
        pair = any(a and b for a, b in pairwise(above))
        if abs(med) > pct or pair:
            out.append(
                f"  WARN sentinel {key}: median drift {med:+.2f}%"
                + (f", two consecutive sentinels above {pct}%" if pair else "")
            )
    measured = drifts["sentinel"].n_unique()
    return out or [
        f"  ok   sentinel: no method drifts above {pct}% ({measured} measured sentinels)"
    ]


def latencies(run: Path, cells: list[str]) -> dict[tuple[str, str, str], float]:
    """Process 0's serving latency (`estimand`) per (cell, method, variant), in ticks,
    for ``cells`` of a packed run."""
    s = (
        pl.scan_parquet(run / "samples.parquet")
        .filter(pl.col("cell").cast(pl.String).is_in(cells) & (pl.col("mode") == "serving"))
        .filter(pl.col("process") == 0)
        .select(analysis.SAMPLE_COLUMNS)
        .collect()
    )
    if s.is_empty():
        return {}
    groups = pl.read_parquet(run / "groups.parquet")
    manifests = fm.read_json(run / "cells.json")
    est = estimand.block_estimates(s, groups, manifests, n_boot=1)
    return {(c, m, v): r["latency"] for (c, m, v, _), r in est.items()}


def layout_warnings(default: Path, aligned: Path, pct: float = LAYOUT_PCT) -> list[str]:
    """The 64-byte-aligned build's latency over the default build's, per method, the
    median over real-model cells (not the import fixtures)."""
    cells = [
        c
        for c in fm.read_json(default / "cells.json")
        if not c.startswith("fixtures/") and c in fm.read_json(aligned / "cells.json")
    ]
    a, b = latencies(default, cells), latencies(aligned, cells)
    out = []
    for method in sorted({(m, v) for _, m, v in a}):
        ratios = [
            b[(c, *method)] / a[(c, *method)]
            for c in cells
            if (c, *method) in a and (c, *method) in b
        ]
        if not ratios:
            continue
        med = st.median(ratios)
        flag = abs(med - 1) * 100 > pct
        out.append(
            f"  {'WARN' if flag else 'ok  '} layout check {'/'.join(method)}: median aligned / "
            f"default {med:.4f} over {len(ratios)} real-model cells (±{pct}%)"
        )
    return out


def g1_warnings(
    run: Path, cells: dict[str, Any], lo_hi: tuple[float, float] = G1_RANGE
) -> list[str]:
    """The algorithmic speedup at G=1 at the T=500, L=8 survival anchors."""
    present = [c for c in G1_ANCHORS if c in cells]
    if not present:
        return []
    lat = latencies(run, present)
    out = []
    for c in present:
        tw, fw = lat.get((c, "treewalker", "all-on")), lat.get((c, "treewalker_fullwalk", "-"))
        if tw is None or fw is None:
            continue
        sp = fw / tw
        flag = not lo_hi[0] <= sp <= lo_hi[1]
        out.append(
            f"  {'WARN' if flag else 'ok  '} G=1 anchor {c}: full walk / TreeWalker {sp:.3f} "
            f"({lo_hi[0]}-{lo_hi[1]})"
        )
    if not out:
        return [f"  WARN G=1 anchors: {len(present)} present, none with both latencies to compare"]
    return out


def released_e1(repo: Path, path: Path | None = None) -> dict[tuple[int, int], float] | None:
    """The released E1 full-walk median per (k, G) on Intel, in us; None if unreadable."""
    if path is not None:
        text = path.read_text()
    else:
        r = subprocess.run(
            ["git", "show", E1_RELEASED], cwd=repo, capture_output=True, text=True, check=False
        )
        if r.returncode != 0:
            return None
        text = r.stdout
    return {
        (int(row["k"]), int(row["G"])): float(row["median_us"])
        for row in csv.DictReader(io.StringIO(text))
        if row["method"] == "treewalker_fullwalk"
    }


def e1_warnings(
    run: Path, released: dict[tuple[int, int], float] | None, pct: float = E1_PCT
) -> list[str]:
    """E1's full-walk latency per cell against the released Intel numbers."""
    if released is None:
        return [f"  --   E1: the released numbers ({E1_RELEASED}) are not readable"]
    from . import results

    info = fm.read_json(run / "run.json")
    hz = float(info["timer"]["calibration"]["hz"])
    cells = [c for c in fm.read_json(run / "cells.json") if c.startswith("scenario_credit/")]
    lat = latencies(run, cells)
    out, compared = [], 0
    for c in cells:
        d = results.design(c)
        ticks = lat.get((c, "treewalker_fullwalk", "-"))
        want = released.get((d["k"], d["G"]))
        if ticks is None or want is None:
            continue
        compared += 1
        us = ticks / hz * 1e6
        dev = 100 * (us / want - 1)
        if abs(dev) > pct:
            out.append(f"  WARN E1 {c}: full walk {us:.2f} us vs released {want:.2f} ({dev:+.1f}%)")
    if not compared:
        return [f"  WARN E1: no cell compared ({len(cells)} scenario cells in the run)"]
    return out or [
        f"  ok   E1: every full-walk latency within {pct}% of the released numbers "
        f"({compared} cells compared)"
    ]


def check(
    runs: list[Path],
    repo: Path,
    released_path: Path | None = None,
    expect_exact_sums: bool | None = None,
) -> tuple[list[str], bool]:
    """Every gate and warning for ``runs`` (packed run directories): report lines and
    whether every gate passes. The layout check pairs layout-align64-ARCH with its
    sibling layout-default-ARCH; E1's check reads scenario-v1 runs on x86_64."""
    lines, ok = [], True
    released = None
    for run in runs:
        info = fm.read_json(run / "run.json")
        lines.append(f"{run.name} (suite {info.get('suite')}):")
        g, passed = gates(run, expect_exact_sums)
        lines += g
        ok &= passed
        lines += sentinel_warnings(info)
        cells = fm.read_json(run / "cells.json")
        lines += g1_warnings(run, cells)
        if run.name.startswith("layout-align64-"):
            default = run.with_name(run.name.replace("layout-align64-", "layout-default-", 1))
            if default.exists():
                lines += layout_warnings(default, run)
            else:
                lines.append(f"  --   layout check: no {default.name}")
        if info.get("suite") == "scenario-v1" and run.name.endswith("x86_64"):
            released = released if released is not None else released_e1(repo, released_path)
            lines += e1_warnings(run, released)
    lines.append("gates: " + ("all pass" if ok else "FAILED"))
    return lines, ok
