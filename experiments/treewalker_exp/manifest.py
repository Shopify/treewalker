"""The execution manifest: one suite resolved into every model, workload and cell.

The Rust runner executes the manifest and has no grid logic. Cell paths are
relative to the artifacts directory, and each entry carries the key that
``prepare`` checks, so a run can match models and data by hash across
machines. Each cell also says what the runner times: its method set, its
research variants and its modes. The run
settings (grids.toml [run]), the native libraries from ``build-native`` and
tl2cgen's runtime library come with it.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import formats as fm
from . import grids
from .grids import Suite
from .paths import Paths

SCHEMA = 3


def git_source(repo: Path) -> dict[str, Any]:
    """The source commit. A git archive has no .git, so the VMs read
    experiments/SOURCE_COMMIT, which git archive fills in (export-subst)."""
    stamp = repo / "experiments" / "SOURCE_COMMIT"
    if stamp.exists() and not (text := stamp.read_text().strip()).startswith("$Format"):
        return {"commit": text, "dirty": False, "from": "experiments/SOURCE_COMMIT"}

    def git(*args: str) -> str:
        r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""

    return {
        "commit": git("rev-parse", "HEAD") or None,
        "dirty": bool(git("status", "--porcelain")),
        "from": "git",
    }


def files_intact(cell_dir: Path, files: dict[str, Any], hashes: dict[Path, str | None]) -> bool:
    """Whether every file a cell records still has its recorded hash. Files the
    cells share, such as a model or a workload's test data, are hashed once per
    manifest, through ``hashes``."""
    for rec in files.values():
        p = (cell_dir / rec["path"]).resolve()
        if p not in hashes:
            hashes[p] = fm.sha256_file(p) if p.is_file() else None
        if hashes[p] != rec["sha256"]:
            return False
    return True


def tl2cgen_runtime() -> dict[str, str] | None:
    """libtl2cgen from the installed tl2cgen package (the baselines group)."""
    spec = importlib.util.find_spec("tl2cgen")
    if spec is None or not spec.submodule_search_locations:
        return None
    lib_dir = Path(next(iter(spec.submodule_search_locations))) / "lib"
    name = {"darwin": "libtl2cgen.dylib", "win32": "tl2cgen.dll"}.get(sys.platform, "libtl2cgen.so")
    lib = lib_dir / name
    if not lib.exists():
        return None
    return {"path": str(lib), "sha256": fm.sha256_file(lib)}


def native_libs(paths: Paths) -> dict[str, Any]:
    """The native libraries ``build-native`` recorded, checked by hash."""
    from . import native

    out: dict[str, Any] = {}
    for lib, rec in native.records(paths).items():
        if Path(rec["path"]).exists() and fm.sha256_file(Path(rec["path"])) == rec["sha256"]:
            out[lib] = {
                **{k: rec[k] for k in ("path", "sha256", "version", "commit")},
                "build": {k: rec.get(k) for k in ("key", "identity", "cmake_args", "compiler")},
            }
    return out


# The sentinel times what the acceptance suite times on its cell.
SENTINEL_PLAN = {"methods": "factorial", "variants": [], "modes": ["serving"]}


def sentinel(
    paths: Paths,
    doc: dict[str, Any],
    run: dict[str, Any],
    hashes: dict[Path, str | None] | None = None,
) -> dict[str, Any] | None:
    """The sentinel cell's entry ([run] sentinel_cell), resolved against the
    artifacts like any cell, or its ID and why the runner cannot reach it."""
    cell_id = run.get("sentinel_cell", "")
    if not cell_id or not run.get("sentinel_every", 0):
        return None
    cell = grids.find_cell(doc, cell_id)
    if cell is None:
        return {"id": cell_id, "status": "skipped", "reason": "no such cell in grids.toml"}
    doc_path = cell.dir(paths.artifacts) / "cell.json"
    if not doc_path.exists():
        return {
            "id": cell_id,
            "status": "skipped",
            "reason": f"not prepared under {paths.artifacts}",
        }
    cdoc = fm.read_cell(doc_path)
    status = cdoc["status"]
    if status == "ready" and not files_intact(
        cell.dir(paths.artifacts), cdoc.get("files", {}), hashes or {}
    ):
        status = "stale"
    return {
        "id": cell_id,
        "dir": str(cell.dir(paths.artifacts).relative_to(paths.artifacts)),
        "status": status,
        "key": cdoc["key"],
        **({} if status == "ready" else {"reason": f"status {status}"}),
        **SENTINEL_PLAN,
    }


def build(
    paths: Paths,
    suite: Suite,
    grids_doc: dict[str, Any] | None = None,
    run_overrides: tuple[str, ...] = (),
) -> dict[str, Any]:
    art = paths.artifacts
    hashes: dict[Path, str | None] = {}
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
            **suite.plan_for(cell),
        }
        if not doc_path.exists():
            cells.append({**entry, "status": "missing"})
            continue
        doc = fm.read_cell(doc_path)
        # A ready cell stays ready only while its files are the ones it recorded:
        # rebuilding another cell can rewrite a file they share.
        status = doc["status"]
        if status == "ready" and not files_intact(cell_dir, doc["files"], hashes):
            status = "stale"
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
                "status": status,
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
    if grids_doc is None:
        grids_doc = grids.load(paths.grids)
    run = grids.run_config(grids_doc, run_overrides, suite.name)
    return {
        "schema_version": SCHEMA,
        "suite": suite.name,
        "description": suite.description,
        "artifacts_dir": str(art.resolve()),
        "grids_sha256": fm.sha256_file(paths.grids),
        "source": git_source(paths.repo),
        "run": run,
        "sentinel": sentinel(paths, grids_doc, run, hashes),
        "native": native_libs(paths),
        "tl2cgen_runtime": tl2cgen_runtime(),
        "counts": counts,
        **({"ablation": suite.extra} if suite.extra else {}),
        "models": sorted(models.values(), key=lambda m: m["id"]),
        "workloads": sorted(workloads.values(), key=lambda w: w["dir"]),
        "cells": cells,
    }


def write(
    paths: Paths, suite: Suite, run_overrides: tuple[str, ...] = ()
) -> tuple[Path, dict[str, Any]]:
    doc = build(paths, suite, run_overrides=run_overrides)
    out = paths.manifests / f"{suite.name}.json"
    fm.write_json(out, doc)
    return out, doc
