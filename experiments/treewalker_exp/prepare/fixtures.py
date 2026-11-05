"""Cells built from the import fixtures, for runner coverage their shape lacks."""

from pathlib import Path
from typing import Any

import numpy as np

from .. import formats as fm
from .. import train as tr
from .. import workloads as wl
from ..grids import Cell
from .cells import write_workload
from .context import Context
from .models import PREP_POLICY, SCHEMA, config_varying
from .store import put_bytes, put_matrix, put_npy, rel

# --- fixture cells ---------------------------------------------------------------------


# Cells built from the import fixtures for runner coverage their shape lacks.
# fallback_sums: identity.bin's model with leaves 1e300 and 1e-300, 2^-997 apart,
# so no i128 fixed point holds both and it sums in the wide accumulator
# (tests/import.rs builds the same model). wide_pieces: sigmoid_f64 over groups
# across the 1,024-row piece boundary.
WIDE_PIECES = [1023, 1024, 1025, 2049]


def _fallback_sums_model() -> Any:
    from treelite.model_builder import (
        Metadata,
        ModelBuilder,
        PostProcessorFunc,
        TreeAnnotation,
    )

    builder = ModelBuilder(
        threshold_type="float64",
        leaf_output_type="float64",
        metadata=Metadata(
            num_feature=2,
            task_type="kRegressor",
            average_tree_output=False,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=TreeAnnotation(num_tree=2, target_id=[0, 0], class_id=[0, 0]),
        postprocessor=PostProcessorFunc(name="identity"),
        base_scores=[3.0],
    )
    for tree, (left, right) in enumerate([(1e300, -0.5), (1e-300, -1.0)]):
        builder.start_tree()
        builder.start_node(0)
        builder.numerical_test(
            feature_id=tree,
            threshold=1.0,
            default_left=tree == 0,
            opname="<=",
            left_child_key=1,
            right_child_key=2,
        )
        builder.end_node()
        for node, val in [(1, left), (2, right)]:
            builder.start_node(node)
            builder.leaf(val)
            builder.end_node()
        builder.end_tree()
    return builder.commit()


def fixture_inputs(src: Path, name: str) -> tuple[bytes, np.ndarray, dict[str, Any], Any]:
    """A fixture cell's model bytes, data, walker config and group offsets (None
    for groups of the configured width)."""
    X = np.array(fm.read_matrix(src / "data.bin"))
    config = fm.read_walker_config(src / "walker_config.json")
    if name == "fallback_sums":
        return _fallback_sums_model().serialize_bytes(), X, config, None
    if name == "wide_pieces":
        rows = []
        for g, n in enumerate(WIDE_PIECES):
            r = X[np.arange(n) % X.shape[0]].copy()
            r[:, 1] = float(g)  # feature 1 is constant within a group
            rows.append(r)
        offsets = np.concatenate([[0], np.cumsum(WIDE_PIECES)]).astype(np.uint64)
        config = {**config, "max_group_width": max(WIDE_PIECES)}
        return (src / "sigmoid_f64.bin").read_bytes(), np.vstack(rows), config, offsets
    return (src / f"{name}.bin").read_bytes(), X, config, None


def ensure_fixture(ctx: Context, cell: Cell) -> dict[str, Any]:
    """An import fixture as a cell: its model, the fixtures' data (seven groups of
    128 rows, feature 1 constant within each) and Treelite GTIL references."""
    import treelite

    name = cell.model.fixture
    assert name is not None
    src = ctx.paths.repo / "tests" / "fixtures" / "import"
    art = ctx.paths.artifacts
    fw_dir, cell_dir = cell.model.dir(art) / cell.framework, cell.dir(art)
    model_bin = fw_dir / "model_treelite.bin"
    model_bytes, X, config, offsets = fixture_inputs(src, name)
    files = {"model_treelite_bin": model_bin}
    hashes = {"model_treelite_bin": put_bytes(model_bin, model_bytes)}
    if (src / f"{name}.json").exists():
        put_bytes(fw_dir / "model_treelite.json", (src / f"{name}.json").read_bytes())
    config["feature_names"] = ["x0", "x1"]
    n_groups = len(offsets) - 1 if offsets is not None else X.shape[0] // config["max_group_width"]
    meta: dict[str, Any] = {"generator": "fixture-v1", "grouping": {"kind": "fixture"}}
    w = wl.Workload(X, offsets, config, np.arange(n_groups, dtype=np.int64), meta)
    built = write_workload(w, cell_dir)
    files.update(built.files)
    hashes.update(built.hashes)
    tl = treelite.Model.deserialize(str(model_bin))
    ref_path = src / f"{name}_reference.bin"
    if ref_path.exists() and offsets is None and name != "fallback_sums":
        ref = np.array(fm.read_matrix(ref_path)).ravel()
    else:
        ref = tr.gtil(tl, X)
    files["predictions"] = cell_dir / "predictions.npy"
    hashes["predictions"] = put_npy(files["predictions"], ref)
    rows, oracle = tr.oracle(tl, X, w.group_offsets)
    files["oracle"] = cell_dir / "oracle.bin"
    hashes["oracle"] = put_matrix(files["oracle"], rows)
    identity = {
        "schema_version": SCHEMA,
        "id": cell.id,
        "workload": cell.workload,
        "generator": cell.generator,
        "dataset": "fixtures",
        "framework": cell.framework,
        "fixture": name,
        "data_sha256": dict(sorted(built.hashes.items())),
        "model_sha256": hashes["model_treelite_bin"],
    }
    key = fm.sha256_json({**identity, "prep_policy": PREP_POLICY})
    contract = wl.contracts(w)
    doc = {
        **identity,
        "prep_policy": PREP_POLICY,
        "key": key,
        "status": "ready" if contract["ok"] else "failed",
        "grouping": {**meta["grouping"], **wl.sizes(w.group_offsets)},
        "feature_order": config["feature_names"],
        "model": {
            "key": hashes["model_treelite_bin"],
            "dir": rel(fw_dir, cell_dir),
            "source": f"tests/fixtures/import ({name})",
        },
        "contracts": {k: v for k, v in contract.items() if k != "varying_columns"},
        "model_stats": ctx.model_stats(model_bin, config_varying(config)),
        "oracle": oracle,
        "files": {
            role: {"path": rel(files[role], cell_dir), "sha256": hashes[role]}
            for role in sorted(files)
        },
        "baselines": {},
    }
    fm.write_cell(cell_dir / "cell.json", doc)
    return doc
