"""App. A.3's f32 audit: TreeWalker's and native XGBoost's raw margins against the
same f64 reference.

The reference is each cell's stage oracle (prepare's ``oracle.bin``): ``math.fsum``
of Treelite GTIL's per-tree outputs, then the staged finalization (``tree_sum /
divisor + base_score``, with the margin-scale base score Treelite parsed, rounded
at each step). Native XGBoost accumulates leaves in f32; ``predict(output_margin=True)``
gives its raw margins on the same rows, here. TreeWalker's are the ones the run
computed: each cell's ``stages`` table (output schema 6) holds its tree sum, raw
margin and output on the oracle's rows, with the run's ``exact_sums`` (off in the
camera-ready runs, so TreeWalker sums in f64 in its loaded tree order).

Per cell and machine the audit counts the rows where TreeWalker is closer to the
reference than native XGBoost, ties, or is farther, and separately the rows where
native equals the reference rounded to f32 (the best an f32 result can do). A run
without the stages table, or whose cell was prepared from another oracle than the
local one, contributes no TreeWalker comparison and says so.

v1's audit (audit_f32.py at the neurips2026-submission tag) summed the leaves itself in f32,
f64 and Kahan-compensated f64; this one compares the libraries' actual outputs.
One cell per XGBoost model is audited, the first in ID order, on the oracle's rows
(whole groups, up to 512 rows).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import formats as fm
from . import grids, pack, results


def cells(doc: dict[str, Any], suite: str = "factorial") -> list[Any]:
    """One XGBoost cell per model of ``suite``: the first in ID order."""
    first: dict[str, Any] = {}
    for c in sorted(grids.suite(doc, suite).cells, key=lambda c: c.id):
        if c.framework == "xgboost":
            first.setdefault(c.model.id, c)
    return list(first.values())


ROLES = ("model_native", "oracle", "test_data")


def files(cell: Any, artifacts: Path) -> list[Path]:
    """What the audit reads for ``cell``: its cell.json, and the native model, the
    oracle and the inputs it names (a panel's inputs are its model's test data)."""
    record = cell.dir(artifacts) / "cell.json"
    if not record.exists():
        return [record]
    named = fm.read_json(record)["files"]
    return [record] + [(cell.dir(artifacts) / named[role]["path"]).resolve() for role in ROLES]


def unusable(cell: Any, artifacts: Path) -> list[Path]:
    """``cell``'s files that are missing, or whose SHA-256 differs from the one its
    cell.json records (a local file from another prep)."""
    paths = files(cell, artifacts)
    if not paths[0].exists():
        return paths
    named = fm.read_json(paths[0])["files"]
    return [
        p
        for role, p in zip(ROLES, paths[1:], strict=True)
        if not p.exists() or fm.sha256_file(p) != named[role]["sha256"]
    ]


def _margins(cell: Any, artifacts: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The oracle's rows and raw margins, and native XGBoost's raw margins on them."""
    import xgboost as xgb

    _, model, oracle_path, data = files(cell, artifacts)
    oracle = fm.read_matrix(oracle_path)
    rows = oracle[:, 0].astype(np.int64)
    ref = np.asarray(oracle[:, 2], dtype=np.float64)
    X = np.asarray(fm.read_matrix(data)[rows])
    bst = xgb.Booster()
    bst.load_model(str(model))
    types = bst.feature_types
    dmat = xgb.DMatrix(
        X,
        feature_names=bst.feature_names,
        feature_types=types,
        enable_categorical="c" in (types or ()),
    )
    native = bst.predict(dmat, output_margin=True).astype(np.float64).ravel()
    return rows, ref, native


def compare(treewalker: np.ndarray, native: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Rows where TreeWalker's margin is closer to ``ref`` than native's, as close,
    or farther; and TreeWalker's own error."""
    d_tw, d_native = np.abs(treewalker - ref), np.abs(native - ref)
    scale = np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
    return {
        "closer": int((d_tw < d_native).sum()),
        "tie": int((d_tw == d_native).sum()),
        "farther": int((d_tw > d_native).sum()),
        "differ": int((treewalker != ref).sum()),
        "max_abs": float(d_tw.max()),
        "max_rel": float((d_tw / scale).max()),
    }


def audit_cell(
    cell: Any, artifacts: Path, treewalker: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Native XGBoost's raw margins against the oracle's, on the oracle's rows; with
    ``treewalker`` (machine: its ``(rows, raw_margin, exact_sums)`` from a run, or
    the reason there is none), TreeWalker's against the same reference."""
    rows, ref, native = _margins(cell, artifacts)
    err = np.abs(native - ref)
    scale = np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
    out: dict[str, Any] = {
        "cell": cell.id,
        "rows": len(rows),
        "differ": int((native != ref).sum()),
        "max_abs": float(err.max()),
        "median_abs": float(np.median(err)),
        "max_rel": float((err / scale).max()),
        # Native's f32 margin, widened, equals the reference rounded to f32.
        "agree_f32": int((native == ref.astype(np.float32).astype(np.float64)).sum()),
        "treewalker": {},
    }
    for arch, got in (treewalker or {}).items():
        if isinstance(got, str):
            out["treewalker"][arch] = {"missing": got}
            continue
        tw_rows, margin, exact = got
        if not np.array_equal(tw_rows, rows):
            out["treewalker"][arch] = {"missing": "its stages cover other rows than the oracle"}
            continue
        out["treewalker"][arch] = {**compare(margin, native, ref), "exact_sums": exact}
    return out


def run_stages(run: Path, chosen: list[Any], artifacts: Path) -> dict[str, Any]:
    """Per cell ID, TreeWalker's ``(rows, raw_margin, exact_sums)`` in ``run``, or why
    there are none: no run; no stages table, from an output schema before 6 or missing
    though required; the cell not run; or run from an oracle other than the local
    one."""
    import polars as pl

    if not (run / "cells.json").exists():
        return dict.fromkeys((c.id for c in chosen), f"no run {run.name}")
    if not (run / "stages.parquet").exists():
        schema = pack.output_schema(run)
        if schema is not None and schema >= pack.OPTIONAL["stages"]:
            why = f"{run.name} lacks its required stages table (output schema {schema})"
        else:
            why = f"{run.name} has no stages table (output schema {schema or 'unrecorded'})"
        return dict.fromkeys((c.id for c in chosen), why)
    manifests = fm.read_json(run / "cells.json")
    ids = [c.id for c in chosen]
    stages = (
        pl.read_parquet(run / "stages.parquet", columns=["cell", "row", "raw_margin"])
        .with_columns(pl.col("cell").cast(pl.String))
        .filter(pl.col("cell").is_in(ids))
        .sort("cell", "row", maintain_order=True)
    )
    by_cell = stages.partition_by("cell", as_dict=True)
    out: dict[str, Any] = {}
    for c in chosen:
        m = manifests.get(c.id)
        if m is None:
            out[c.id] = f"not in {run.name}"
            continue
        local = fm.read_json(c.dir(artifacts) / "cell.json")["files"]["oracle"]["sha256"]
        if m["cell"]["files"]["oracle"]["sha256"] != local:
            out[c.id] = f"{run.name} ran it from another oracle"
            continue
        t = by_cell.get((c.id,))
        if t is None:
            out[c.id] = f"no stages rows in {run.name}"
            continue
        out[c.id] = (
            t["row"].to_numpy().astype(np.int64),
            t["raw_margin"].to_numpy(),
            m["validation"]["oracle"].get("exact_sums"),
        )
    return out


def treewalker_checks(runs: Path, suite: str = "factorial") -> dict[str, dict[str, int]]:
    """Per machine, the cells whose oracle check passed, the rows checked, and the
    passing cells checked bit for bit (exact sums on) or within 1e-12 relative."""
    out = {}
    for arch in results.MACHINES:
        manifests = fm.read_json(results.run_dir(runs, suite, arch) / "cells.json")
        status = [m["validation"]["oracle"] for m in manifests.values()]
        passed = [s for s in status if s.get("status") == "pass"]
        out[arch] = {
            "cells": len(status),
            "pass": len(passed),
            "rows": sum(int(s.get("rows", 0)) for s in passed),
            "bitwise": sum(s.get("exact_sums", True) is not False for s in passed),
        }
    return out


def report(runs: Path, artifacts: Path, doc: dict[str, Any], suite: str = "factorial") -> list[str]:
    chosen = cells(doc, suite)
    present = [c for c in chosen if not unusable(c, artifacts)]
    stages = {
        arch: run_stages(results.run_dir(runs, suite, arch), present, artifacts)
        for arch in results.MACHINES
    }
    audited = [
        audit_cell(c, artifacts, {arch: stages[arch][c.id] for arch in stages}) for c in present
    ]
    tw = treewalker_checks(runs, suite)
    rows = sum(a["rows"] for a in audited)
    differ = sum(a["differ"] for a in audited)
    agree = sum(a["agree_f32"] for a in audited)
    lines = [
        "# f32 audit: raw margins against the same f64 reference",
        "",
        "Reference: each cell's stage oracle, math.fsum of Treelite GTIL's per-tree outputs "
        "and the staged finalization with the margin-scale base score.",
        "",
        f"TreeWalker's oracle check in the {suite} runs: "
        + "; ".join(
            f"{arch} {t['pass']} of {t['cells']} cells passed, {t['rows']:,} rows "
            f"({t['bitwise']} cells bit for bit, {t['pass'] - t['bitwise']} with exact sums "
            "off, within 1e-12 relative)"
            for arch, t in tw.items()
        )
        + ".",
        "",
        f"Native XGBoost (output_margin=True), one cell per model: {len(audited)} of "
        f"{len(chosen)} models audited ({len(chosen) - len(present)} without their artifacts, "
        "missing or from another prep), "
        f"{rows:,} rows; {differ:,} rows ({100 * differ / max(rows, 1):.1f}%) differ from the "
        f"reference; largest error {max((a['max_abs'] for a in audited), default=0):.3g} "
        f"(relative {max((a['max_rel'] for a in audited), default=0):.3g}). "
        f"Native equals the reference rounded to f32 on {agree:,} rows "
        f"({100 * agree / max(rows, 1):.1f}%).",
        "",
    ]
    for arch in results.MACHINES:
        done = [(a, a["treewalker"][arch]) for a in audited if "closer" in a["treewalker"][arch]]
        n = sum(a["rows"] for a, _ in done)
        tot = {k: sum(t[k] for _, t in done) for k in ("closer", "tie", "farther", "differ")}
        names = {True: "on", False: "off"}
        settings = sorted({names.get(t["exact_sums"], "unrecorded") for _, t in done})
        why = sorted({a["treewalker"][arch].get("missing") for a in audited} - {None})
        if not done:
            lines.append(f"TreeWalker against native XGBoost, {arch}: none ({'; '.join(why)}).")
        else:
            lines.append(
                f"TreeWalker against native XGBoost, {arch} (the run's stages, exact sums "
                f"{'/'.join(settings)}): {len(done)} cells, {n:,} rows; TreeWalker closer to "
                f"the reference on {tot['closer']:,} ({100 * tot['closer'] / max(n, 1):.2f}%), "
                f"tied on {tot['tie']:,}, farther on {tot['farther']:,}; TreeWalker differs "
                f"from the reference on {tot['differ']:,} rows, by at most "
                f"{max(t['max_abs'] for _, t in done):.3g} "
                f"(relative {max(t['max_rel'] for _, t in done):.3g})."
                + (f" Not compared: {'; '.join(why)}." if why else "")
            )
        lines.append("")
    lines += [
        "| cell | rows | differ | max abs | median abs | max rel | = f32(ref) |"
        + "".join(f" TreeWalker {arch} closer/tie/farther |" for arch in results.MACHINES),
        "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(results.MACHINES),
    ]
    for a in sorted(audited, key=lambda a: -a["max_abs"]):
        cols = []
        for arch in results.MACHINES:
            t = a["treewalker"][arch]
            cols.append(f"{t['closer']}/{t['tie']}/{t['farther']}" if "closer" in t else "-")
        lines.append(
            f"| {a['cell']} | {a['rows']} | {a['differ']} | {a['max_abs']:.3g} | "
            f"{a['median_abs']:.3g} | {a['max_rel']:.3g} | {a['agree_f32']} | "
            + " | ".join(cols)
            + " |"
        )
    return lines


def missing(artifacts: Path, doc: dict[str, Any], suite: str = "factorial") -> list[Path]:
    """The audit's files missing under ``artifacts`` or not matching their cell.json,
    relative to it."""
    root = artifacts.resolve()
    return [
        p.resolve().relative_to(root) for c in cells(doc, suite) for p in unusable(c, artifacts)
    ]
