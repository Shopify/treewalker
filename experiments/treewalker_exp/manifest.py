"""The execution manifest: one suite resolved into every model, workload and cell.

The Rust runner executes the manifest and has no grid logic. Paths are
relative to the artifacts directory, and each entry carries the key that
``prepare`` checks, so a run can match models and data by hash across
machines.
"""

import subprocess
from pathlib import Path
from typing import Any

from . import formats as fm
from .grids import Suite
from .paths import Paths

SCHEMA = 1


def git_source(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def build(paths: Paths, suite: Suite) -> dict[str, Any]:
    art = paths.artifacts
    models: dict[str, dict[str, Any]] = {}
    workloads: dict[str, dict[str, Any]] = {}
    cells = []
    for cell in suite.cells:
        cell_dir = cell.dir(art)
        doc_path = cell_dir / "cell.json"
        entry: dict[str, Any] = {
            "id": cell.id,
            "dir": str(cell_dir.relative_to(art)),
            "dataset": cell.model.dataset,
            "framework": cell.framework,
            "grid_workload": cell.workload,
            "generator": cell.generator,
        }
        if not doc_path.exists():
            cells.append({**entry, "status": "missing"})
            continue
        doc = fm.read_cell(doc_path)
        model_dir = (cell_dir / doc["model"]["dir"]).resolve()
        model_id = str(model_dir.relative_to(art.resolve()))
        models.setdefault(model_id, {"id": model_id, "dir": model_id, "key": doc["model"]["key"]})
        data = doc["data_sha256"]
        wkey = fm.sha256_json(data)
        test_data = (cell_dir / doc["files"]["test_data"]["path"]).resolve()
        workloads.setdefault(
            wkey,
            {
                "key": wkey,
                "dir": str(test_data.parent.relative_to(art.resolve())),
                "generator": doc["generator"],
            },
        )
        cells.append(
            {
                **entry,
                "status": doc["status"],
                "key": doc["key"],
                "model": model_id,
                "workload": wkey,
                "rows": doc["grouping"]["rows"],
                "n_groups": doc["grouping"]["n_groups"],
            }
        )
    counts: dict[str, int] = {}
    for c in cells:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {
        "schema_version": SCHEMA,
        "suite": suite.name,
        "description": suite.description,
        "grids_sha256": fm.sha256_file(paths.grids),
        "source": git_source(paths.repo),
        "counts": counts,
        **({"ablation": suite.extra} if suite.extra else {}),
        "models": sorted(models.values(), key=lambda m: m["id"]),
        "workloads": sorted(workloads.values(), key=lambda w: w["dir"]),
        "cells": cells,
    }


def write(paths: Paths, suite: Suite) -> tuple[Path, dict[str, Any]]:
    doc = build(paths, suite)
    out = paths.manifests / f"{suite.name}.json"
    fm.write_json(out, doc)
    return out, doc
