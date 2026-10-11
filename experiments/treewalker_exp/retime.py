"""Re-timing: a cell whose headline interval is wider than ±2% of its ratio is
timed once more with twice the rounds, on the parent run's VM.

The intervals tested are the paper's contrasts, each from the shared entity and
round draws of its two sides (`estimand`): every method's speedup over
TreeWalker, and every research variant's ablation effect over its own
denominator (the research build's all-on; for a monotonic variant, the same
variant with monotonic on). `flagged` lists every cell and mode where one of them
has a half-width above the threshold.

Native libraries and paths differ per machine, so re-timing is per machine:
`plan` checks the execution manifest the run executed against the run's own
record (`manifest_problems`), cuts it to the flagged cells with ``min_rounds =
max_rounds`` twice the cell's rounds (one manifest per round count, since run
settings are per run), and writes beside each manifest a script that runs it the
way the campaign does: ``treewalker-exp retime-run`` pins the runner to core 0,
requires the PMU and records the system facts. Each attempt runs into a run
directory of its own, ``<run>-retime-r<rounds>-<arch>``, so both attempts stay in
the data; `results.speedups` checks an attempt against its parent
(`attempt_problems`) and reports the attempt with more rounds.
"""

from __future__ import annotations

import copy
import shlex
from pathlib import Path
from typing import Any

import polars as pl

from . import formats as fm
from . import results

THRESHOLD_PCT = 2.0
# Run settings a re-time attempt changes; every other one must be its parent's.
ROUND_SETTINGS = ("min_rounds", "max_rounds")


def half_widths(df: Any) -> Any:
    """Every serving contrast's interval half-width, % of its ratio: each method's
    speedup over TreeWalker (``contrast`` treewalker/all-on) and each research
    variant's ablation effect (``contrast`` treewalker_research/<denominator>)."""
    serving = df.filter(pl.col("mode") == "serving")
    keep = ["arch", "cell", "mode", "method", "variant", "rounds"]
    speed = serving.filter(
        pl.col("lo").is_not_null()
        & ~((pl.col("method") == "treewalker") & (pl.col("variant") == "all-on"))
    ).select(
        *keep,
        contrast=pl.lit("treewalker/all-on"),
        half_pct=100 * (pl.col("hi") - pl.col("lo")) / 2 / pl.col("speedup"),
    )
    if "effect_lo" not in df.columns:
        return speed
    effect = serving.filter(pl.col("effect_lo").is_not_null()).select(
        *keep,
        contrast=pl.lit("treewalker_research/") + pl.col("effect_against"),
        half_pct=100 * (pl.col("effect_hi") - pl.col("effect_lo")) / 2 / pl.col("effect"),
    )
    return pl.concat([speed, effect], how="vertical_relaxed")


def flagged(df: Any, threshold_pct: float = THRESHOLD_PCT) -> Any:
    """Per (arch, cell, mode) above ``threshold_pct``: the widest contrast (method,
    variant, contrast, half-width), how many exceed it, and the attempt's rounds,
    sorted by half-width."""
    return (
        half_widths(df)
        .filter(pl.col("half_pct") > threshold_pct)
        .sort("half_pct", descending=True)
        .group_by("arch", "cell", "mode", maintain_order=True)
        .agg(
            pl.col("method").first(),
            pl.col("variant").first(),
            pl.col("contrast").first(),
            pl.col("half_pct").first(),
            pl.col("rounds").first(),
            pl.len().alias("comparisons"),
        )
    )


def _sha(doc: dict[str, Any] | None) -> str | None:
    return None if not doc else doc.get("sha256")


def manifest_problems(
    manifest: dict[str, Any], info: dict[str, Any], cells: dict[str, Any]
) -> list[str]:
    """How an execution manifest differs from what the run (``run.json``'s ``info``,
    ``cells.json``'s ``cells``) recorded executing: its suite, native libraries,
    tl2cgen runtime, artifacts, grids, source, run settings and cell keys."""
    out = []
    rec = info.get("manifest") or {}
    pairs = [
        ("suite", manifest.get("suite"), info.get("suite")),
        ("artifacts_dir", manifest.get("artifacts_dir"), rec.get("artifacts_dir")),
        ("grids_sha256", manifest.get("grids_sha256"), rec.get("grids_sha256")),
        (
            "source commit",
            (manifest.get("source") or {}).get("commit"),
            (rec.get("source") or {}).get("commit"),
        ),
        (
            "tl2cgen runtime",
            _sha(manifest.get("tl2cgen_runtime")),
            _sha(info.get("tl2cgen_runtime")),
        ),
    ]
    for lib in ("lightgbm", "xgboost"):
        pairs.append(
            (
                f"native {lib}",
                _sha((manifest.get("native") or {}).get(lib)),
                _sha((info.get("natives") or {}).get(lib)),
            )
        )
    run, ran = manifest.get("run") or {}, info.get("run") or {}
    schema = int(info.get("output_schema") or 0)
    # The run's record holds every setting as the runner resolved it: an omitted
    # one is written from it (`explicit_run`); one the record lacks is the runner's
    # to resolve, which a current run (schema 6 on) never leaves.
    for key in sorted(set(run) | set(ran)):
        if key in ROUND_SETTINGS or key not in run:
            continue
        if key not in ran:
            if schema >= 6:
                pairs.append((f"run setting {key}", run[key], "unrecorded"))
            continue
        pairs.append((f"run setting {key}", run.get(key), ran.get(key)))
    out = [f"{what}: manifest {a!r}, run {b!r}" for what, a, b in pairs if a != b]
    for c in manifest.get("cells", []):
        done = cells.get(c["id"])
        if done is not None and done.get("cell", {}).get("key") != c.get("key"):
            out.append(f"cell {c['id']}: the manifest's key differs from the run's")
    return out


def explicit_run(manifest: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """The manifest's run settings with every setting the parent run resolved written
    out, so the attempt runs the parent's settings whatever the manifest omits."""
    return {**(info.get("run") or {}), **(manifest.get("run") or {})}


def _get(doc: Any, path: str) -> Any:
    for part in path.split("."):
        doc = doc.get(part) if isinstance(doc, dict) else None
    return doc


# What an attempt's run.json must share with its parent's: the executed program
# and build, the protocol and the machine; `INSTANCE` adds the VM itself.
EXECUTED = (
    "suite",
    "source.commit",
    "binary_sha256",
    "host.arch",
    "host.rustc",
    "host.rustflags",
    "host.profile",
    "host.features",
    "host.pinned_cpus",
    "hardware_counters",
    "system_info.machine_type",
    "system_info.image",
    "system_info.zone",
    "system_info.turbo_mode",
    "system_info.pmu_level",
    "manifest.artifacts_dir",
    "manifest.grids_sha256",
    "tl2cgen_runtime.sha256",
    "natives.lightgbm.sha256",
    "natives.xgboost.sha256",
)
# A cell's identity: its key, the hashes of its model, inputs and oracle, and its
# compiled baselines' identities and hashes.
CELL_IDENTITY = ("key", "model_key", "content_key", "data_sha256", "files", "baselines")
# The VM: re-times run on the campaign's own VMs, so an attempt on another
# instance, or one whose instance is unrecorded on either side, is refused.
INSTANCE = ("system_info.instance_id", "system_info.hostname")


def attempt_problems(parent: Path, attempt: Path) -> list[str]:
    """How a re-time attempt differs from its parent run beyond its rounds.

    Any difference in `EXECUTED` (the source and binary that ran, the Rust
    toolchain and flags, pinned CPUs, the PMU, turbo mode, machine type, image and
    zone, every native library and the tl2cgen runtime), in the resolved run
    settings other than the rounds, or in a cell's `CELL_IDENTITY`; and an instance
    (`INSTANCE`) that differs or is unrecorded."""
    pi, ai = fm.read_json(parent / "run.json"), fm.read_json(attempt / "run.json")
    pc, ac = fm.read_json(parent / "cells.json"), fm.read_json(attempt / "cells.json")
    out = [
        f"{path}: attempt {_get(ai, path)!r}, parent {_get(pi, path)!r}"
        for path in EXECUTED
        if _get(ai, path) != _get(pi, path)
    ]
    pr, ar = pi.get("run") or {}, ai.get("run") or {}
    out += [
        f"run setting {k}: attempt {ar.get(k)!r}, parent {pr.get(k)!r}"
        for k in sorted(set(pr) | set(ar))
        if k not in ROUND_SETTINGS and ar.get(k) != pr.get(k)
    ]
    for c, m in ac.items():
        if c not in pc:
            out.append(f"cell {c}: not in the parent run")
            continue
        mine, theirs = m.get("cell") or {}, pc[c].get("cell") or {}
        out += [f"cell {c}: {f} differs" for f in CELL_IDENTITY if mine.get(f) != theirs.get(f)]
    for path in INSTANCE:
        a, b = _get(ai, path), _get(pi, path)
        if a is None or b is None:
            out.append(f"{path} unrecorded: not shown to be the parent's VM")
        elif a != b:
            out.append(f"{path}: attempt {a!r}, parent {b!r} (another VM)")
    return out


def manifests(
    manifest: dict[str, Any], cells: Any, run_base: str, threshold_pct: float
) -> dict[str, dict[str, Any]]:
    """The re-time manifests for one machine's ``cells`` (`flagged`'s rows), keyed
    by run base name (``<run_base>-retime-r<rounds>``): the run's own manifest cut to
    the cells, each group's rounds doubled. A cell's rounds are the most any of its
    flagged modes ran."""
    rounds: dict[str, int] = {}
    for row in cells.iter_rows(named=True):
        rounds[row["cell"]] = max(rounds.get(row["cell"], 0), int(row["rounds"]))
    by_id = {c["id"]: c for c in manifest["cells"]}
    missing = sorted(set(rounds) - set(by_id))
    if missing:
        raise ValueError(f"flagged cells not in the manifest: {', '.join(missing)}")
    out: dict[str, dict[str, Any]] = {}
    for n in sorted(set(rounds.values())):
        ids = sorted(c for c, r in rounds.items() if r == n)
        doc = copy.deepcopy(manifest)
        doc["cells"] = [by_id[c] for c in ids]
        doc["run"] = {**doc.get("run", {}), "min_rounds": 2 * n, "max_rounds": 2 * n}
        counts: dict[str, int] = {}
        for c in doc["cells"]:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        doc["counts"] = counts
        doc["retime"] = {"of": run_base, "threshold_pct": threshold_pct, "rounds": n, "cells": ids}
        out[f"{run_base}-retime-r{2 * n}"] = doc
    return out


def plan(
    runs: Path,
    run_base: str,
    arch: str,
    manifest_path: Path,
    out: Path,
    threshold_pct: float = THRESHOLD_PCT,
) -> list[str]:
    """List the flagged cells of ``run_base`` (a run ID without the machine) on
    ``arch`` and write its re-time manifests and scripts to ``out``
    (``<run_base>-retime-r<rounds>-<machine>.{json,sh}``). ``manifest_path`` is the
    execution manifest this machine's run executed; it must match the run's record."""
    machine = results.MACHINES[arch]
    run = results.run_dir(runs, run_base, arch)
    info, cells = fm.read_json(run / "run.json"), fm.read_json(run / "cells.json")
    manifest = fm.read_json(manifest_path)
    problems = manifest_problems(manifest, info, cells)
    if problems:
        raise ValueError(
            f"{manifest_path} is not the manifest {run.name} executed: " + "; ".join(problems)
        )
    f = flagged(results.speedups_one(run, arch), threshold_pct)
    lines = [
        f"{run.name}: {f.height} (cell, mode) pairs with an interval half-width above "
        f"{threshold_pct}% of the ratio"
    ]
    for row in f.iter_rows(named=True):
        lines.append(
            f"  {row['cell']} {row['mode']}: {row['half_pct']:.2f}% ({row['method']}/"
            f"{row['variant']} over {row['contrast']}, {row['comparisons']} contrasts), "
            f"{row['rounds']} rounds"
        )
    out.mkdir(parents=True, exist_ok=True)
    manifest = {**manifest, "run": explicit_run(manifest, info)}
    for rid, doc in manifests(manifest, f, run_base, threshold_pct).items():
        path = out / f"{rid}-{machine}.json"
        fm.write_json(path, doc)
        script = out / f"{rid}-{machine}.sh"
        script.write_text(
            "#!/bin/sh\n# Re-time "
            f"{len(doc['cells'])} cells of {run.name} with {doc['run']['min_rounds']} rounds, "
            "as the campaign runs: core 0, the PMU required, the system facts recorded.\n"
            "set -e\n"
            f"exec uv run --group baselines treewalker-exp retime-run {shlex.quote(str(path))} "
            f"--run-id {shlex.quote(f'{rid}-{machine}')} --parent {shlex.quote(run.name)} "
            "--core 0 --require-pmu "
            '${SYSTEM_INFO:+--system-info "$SYSTEM_INFO"}\n'
        )
        script.chmod(0o755)
        lines.append(
            f"  wrote {path} ({len(doc['cells'])} cells, {doc['run']['min_rounds']} rounds) "
            f"and {script.name}: SYSTEM_INFO=... {script}"
        )
    return lines


def dispositions(runs: Path, suite: str, arch: str) -> list[dict[str, Any]]:
    """Every (cell, mode) of the run ``suite`` on ``arch`` with more than one attempt:
    the parent and each attempt with its rounds and status (accepted, or refused
    with its first problem), and the attempt selected,
    as `results.speedups` selects it (the most rounds among the parent and the
    accepted attempts, the later on a tie)."""
    parent, *retimes = results.attempts(runs, suite, arch)
    if not retimes:
        return []

    def stops(run: Path) -> dict[tuple[str, str], int]:
        return {
            (cell, mode): int(stop["rounds"])
            for cell, m in fm.read_json(run / "cells.json").items()
            for mode, stop in (m.get("stop") or {}).items()
        }

    base = stops(parent)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for attempt in retimes:
        problems = attempt_problems(parent, attempt)
        status, note = ("refused", problems[0]) if problems else ("accepted", "")
        for key, n in stops(attempt).items():
            row = rows.setdefault(
                key,
                {"cell": key[0], "mode": key[1], "parent": (parent.name, base.get(key))}
                | {"attempts": []},
            )
            row["attempts"].append(
                {"run": attempt.name, "rounds": n, "status": status, "note": note}
            )
    for row in rows.values():
        best = row["parent"]
        for a in row["attempts"]:
            if a["status"] == "accepted" and (best[1] is None or a["rounds"] >= best[1]):
                best = (a["run"], a["rounds"])
        row["selected"] = best[0]
    return sorted(rows.values(), key=lambda r: (r["cell"], r["mode"]))


def planned(retime_dir: Path, suite: str, arch: str) -> int | None:
    """How many cells the re-time plans for ``suite`` on ``arch`` flagged, from the
    manifests in ``retime_dir`` (None when there are none to read)."""
    docs = sorted(retime_dir.glob(f"{suite}-retime-r*-{results.MACHINES[arch]}.json"))
    if not docs:
        return None
    return len({c for d in docs for c in (fm.read_json(d).get("retime") or {}).get("cells", [])})
