"""App. A.3's f32 audit: native XGBoost's raw margins against the correctly rounded
reference, which TreeWalker matches exactly.

The reference is each cell's stage oracle (prepare's ``oracle.bin``): ``math.fsum``
of Treelite GTIL's per-tree outputs, then the staged finalization (``tree_sum /
divisor + base_score``, with the margin-scale base score Treelite parsed, rounded
at each step). TreeWalker sums leaves exactly and rounds once, and the runner
compared its raw margins with the oracle's bit for bit on every cell of the final
run (each manifest's ``validation.oracle``). Native XGBoost accumulates leaves in
f32; ``predict(output_margin=True)`` gives its raw margins on the same rows.

v1's audit (audit_f32.py at the neurips2026 tag) summed the leaves itself in f32,
f64 and Kahan-compensated f64; this one compares the libraries' actual outputs.
One cell per XGBoost model is audited, the first in ID order, on the oracle's rows
(whole groups, up to 512 rows).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import formats as fm
from . import grids, results


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


def audit_cell(cell: Any, artifacts: Path) -> dict[str, Any]:
    """Native XGBoost's raw margins against the oracle's, on the oracle's rows."""
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
    err = np.abs(native - ref)
    scale = np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
    return {
        "cell": cell.id,
        "rows": len(rows),
        "differ": int((native != ref).sum()),
        "max_abs": float(err.max()),
        "median_abs": float(np.median(err)),
        "max_rel": float((err / scale).max()),
    }


def treewalker_checks(runs: Path, suite: str = "factorial") -> dict[str, dict[str, int]]:
    """Per machine, the cells whose oracle check passed, failed, and the rows checked."""
    out = {}
    for arch in results.MACHINES:
        manifests = fm.read_json(results.run_dir(runs, suite, arch) / "cells.json")
        status = [m["validation"]["oracle"] for m in manifests.values()]
        out[arch] = {
            "cells": len(status),
            "pass": sum(s.get("status") == "pass" for s in status),
            "rows": sum(int(s.get("rows", 0)) for s in status if s.get("status") == "pass"),
        }
    return out


def report(runs: Path, artifacts: Path, doc: dict[str, Any], suite: str = "factorial") -> list[str]:
    chosen = cells(doc, suite)
    present = [c for c in chosen if not unusable(c, artifacts)]
    audited = [audit_cell(c, artifacts) for c in present]
    tw = treewalker_checks(runs, suite)
    rows = sum(a["rows"] for a in audited)
    differ = sum(a["differ"] for a in audited)
    lines = [
        "# f32 audit: raw margins against the correctly rounded reference",
        "",
        "Reference: each cell's stage oracle, math.fsum of Treelite GTIL's per-tree outputs "
        "and the staged finalization with the margin-scale base score.",
        "",
        "TreeWalker (the final run's oracle check, bit for bit): "
        + "; ".join(
            f"{arch} {t['pass']} of {t['cells']} cells, {t['rows']:,} rows"
            for arch, t in tw.items()
        )
        + ".",
        "",
        f"Native XGBoost (output_margin=True), one cell per model: {len(audited)} of "
        f"{len(chosen)} models audited ({len(chosen) - len(present)} without their artifacts, "
        "missing or from another prep), "
        f"{rows:,} rows; {differ:,} rows ({100 * differ / max(rows, 1):.1f}%) differ from the "
        f"reference; largest error {max((a['max_abs'] for a in audited), default=0):.3g} "
        f"(relative {max((a['max_rel'] for a in audited), default=0):.3g}).",
        "",
        "| cell | rows | differ | max abs | median abs | max rel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in sorted(audited, key=lambda a: -a["max_abs"]):
        lines.append(
            f"| {a['cell']} | {a['rows']} | {a['differ']} | {a['max_abs']:.3g} | "
            f"{a['median_abs']:.3g} | {a['max_rel']:.3g} |"
        )
    return lines


def missing(artifacts: Path, doc: dict[str, Any], suite: str = "factorial") -> list[Path]:
    """The audit's files missing under ``artifacts`` or not matching their cell.json,
    relative to it."""
    root = artifacts.resolve()
    return [
        p.resolve().relative_to(root) for c in cells(doc, suite) for p in unusable(c, artifacts)
    ]
