"""Cells: a workload's files on one model, their references, contracts and cell.json."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import datasets as ds
from .. import formats as fm
from .. import train as tr
from .. import workloads as wl
from ..grids import SURVIVAL, Cell
from .context import RANKING, Context, TrainData
from .models import IDENTITY_EXTRAS, PREP_POLICY, SCHEMA, derived_inputs, ensure_model
from .store import files_match, put_matrix, put_npy, put_offsets, put_walker_config, rel

log = ds.log


# --- cells -----------------------------------------------------------------------------


@dataclass(slots=True)
class Built:
    """A workload on disk: where each file is, and its hash."""

    workload: wl.Workload
    files: dict[str, Path]  # role -> path
    hashes: dict[str, str]  # role -> sha256
    legacy_reference: Path | None = None  # whatif-v1's reference.bin


def write_workload(w: wl.Workload, data_dir: Path, offsets: bool = True) -> Built:
    files = {
        "test_data": data_dir / "test_data.bin",
        "walker_config": data_dir / "walker_config.json",
        "entities": data_dir / "entities.npy",
    }
    hashes = {
        "test_data": put_matrix(files["test_data"], w.X),
        "walker_config": put_walker_config(files["walker_config"], w.config),
        "entities": put_npy(files["entities"], w.entities),
    }
    if offsets and w.offsets is not None:
        files["group_offsets"] = data_dir / "group_offsets.bin"
        hashes["group_offsets"] = put_offsets(files["group_offsets"], w.offsets)
    return Built(w, files, hashes)


def build_workload(ctx: Context, cell: Cell, td: TrainData, mdoc: dict[str, Any]) -> Built:
    if cell.generator == "whatif-v1":
        return build_whatif_v1(ctx, cell, mdoc)
    w, out = make_workload(ctx, cell, td)
    return write_workload(w, out)


def make_workload(ctx: Context, cell: Cell, td: TrainData) -> tuple[wl.Workload, Path]:
    """A cell's workload, and the directory its files go to."""
    if cell.generator == "rows-v1":
        p = cell.param
        source, _ = make_workload(ctx, rows_source(cell), td)
        w = wl.single_rows(source, p["row_order"], p["rows"])
        w.meta["source"]["spec"] = wl.ROWS_SOURCE[cell.model.dataset]
        if "derived" in source.meta:
            w.meta["derived"] = source.meta["derived"]
        return w, cell.dir(ctx.paths.artifacts)
    w, out = _make_workload(ctx, cell, td)
    return derived_inputs(cell.model, w, td.cat), out


def rows_source(cell: Cell) -> Cell:
    """The cell whose rows a rows-v1 cell takes (workloads.ROWS_SOURCE)."""
    m = cell.model
    if m.dataset in SURVIVAL:
        return Cell(cell.workload, "panel-v2", m, cell.framework)
    if m.dataset in RANKING:
        return Cell(cell.workload, "ranking-sessions-v2", m, cell.framework)
    if m.dataset == "credit":
        params = tuple(wl.ROWS_WHATIF.items())
        return Cell(cell.workload, "whatif-v2", m, cell.framework, params)
    raise ValueError(f"rows-v1 has no source for {m.dataset}")


def _make_workload(ctx: Context, cell: Cell, td: TrainData) -> tuple[wl.Workload, Path]:
    art = ctx.paths.artifacts
    model_dir = cell.model.dir(art)
    p = cell.param
    match cell.generator:
        case "panel-v2":
            assert td.horizon is not None
            cov = ctx.survival(cell.model.dataset).covariates("test")
            w = wl.panel(cov, td.horizon["realized"])
            w.meta["horizon"] = td.horizon
            if "varying_share" in p:
                return layout_panel(ctx, cell, w), cell.dir(art)
            return w, model_dir
        case "ranking-sessions-v2":
            return ctx.sessions(cell.model.dataset), model_dir
        case "ranking-cohort-v2":
            sessions = ctx.sessions(cell.model.dataset)
            subsets = wl.ranking_cohort(
                sessions, p["min_candidates"], [p["size"]], max_groups=p.get("max_groups")
            )
            return subsets[p["size"]], art / cell.model.dataset / "workloads" / cell.workload_id
        case "whatif-v2":
            draws = ctx.whatif_draws(cell.model, td)
            # A replicate times its released cell's data: the features are the
            # released model's most-split ones, so that model must exist.
            released = cell.model.released
            if cell.model.replicate:
                ensure_model(ctx, released, cell.framework, ctx.train_data(released), False)
            native = released.dir(art) / cell.framework / tr.NATIVE_NAME[cell.framework]
            counts = tr.split_counts(cell.framework, native, len(td.names))
            features = wl.whatif_v2_features(draws, counts, p["k"])
            w = wl.whatif_v2(draws, features, p["k"], p["G"])
            if cell.model.replicate:
                w.meta["perturbation"]["ranked_by"] = f"{released.id}/{cell.framework}"
            if td.horizon:
                w.meta["horizon"] = td.horizon
            w.meta["perturbation"]["split_counts"] = {
                td.names[i]: int(counts[i]) for i in draws.pool
            }
            return w, cell.dir(art)
        case "whatif-grid-v1":
            draws = ctx.whatif_draws(cell.model, td)
            native = cell.model.dir(art) / cell.framework / tr.NATIVE_NAME[cell.framework]
            counts = tr.split_counts(cell.framework, native, len(td.names))
            w = wl.whatif_grid(draws, wl.whatif_v2_features(draws, counts, p["k"]), p["G"])
            if td.horizon:
                w.meta["horizon"] = td.horizon
            w.meta["grid"]["split_counts"] = {td.names[i]: int(counts[i]) for i in draws.pool}
            return w, cell.dir(art)
    raise ValueError(cell.generator)


def layout_panel(ctx: Context, cell: Cell, w: wl.Workload) -> wl.Workload:
    """The panel with extra constant covariates declared varying, so the share of
    the model's internal nodes on varying features comes closest to the cell's
    target."""
    fw_dir = cell.model.dir(ctx.paths.artifacts) / cell.framework
    counts = tr.internal_nodes(ctx.tl_model(fw_dir / "model_treelite.bin"))
    target = cell.param["varying_share"]
    base = list(w.config["varying_features"])
    added = wl.varying_for_share(counts, base, target)
    names = w.config["feature_names"]
    config = {**w.config, "varying_features": sorted(base + added)}
    layout = {
        "target_share": target,
        "base_share": wl.varying_share(counts, base),
        "varying_share": wl.varying_share(counts, config["varying_features"]),
        "declared_varying": [names[f] for f in added],
        "internal_nodes": int(counts.sum()),
        "rule": "the constant covariates whose internal nodes bring the share "
        "closest to the target, of every subset",
    }
    return wl.Workload(w.X, w.offsets, config, w.entities, {**w.meta, "layout": layout})


def build_whatif_v1(ctx: Context, cell: Cell, mdoc: dict[str, Any]) -> Built:
    """The released credit scenario cell, written exactly as before."""
    import lightgbm as lgb

    k, G = cell.param["k"], cell.param["G"]
    fw_dir = cell.model.dir(ctx.paths.artifacts) / "lightgbm"
    bst = lgb.Booster(model_file=str(fw_dir / tr.NATIVE_NAME["lightgbm"]))
    pool = wl.whatif_v1_ranked_pool(bst.feature_importance(importance_type="gain"))
    perturb = sorted(pool[:k])
    base_X = ctx.credit().test_X[: wl.N_WHATIF_BASE]
    X, offsets = wl.whatif_v1(base_X, perturb, G)
    any_zero = wl.any_zero_group_fraction(base_X, perturb)
    names = list(ds.CREDIT_FEATURES)
    config = {
        **wl.walker_config(names, G, perturb, [], []),
        # Provenance, not read by the engine.
        "scenario": {
            "k": k,
            "G": G,
            "perturbable_features": perturb,
            "perturbable_feature_names": [names[i] for i in perturb],
            "base_rows": int(offsets.shape[0] - 1),
            # Share of groups with at least one selected feature equal to 0;
            # such entries stay 0 under multiplicative shocks.
            "any_zero_group_fraction": any_zero,
            "perturbation": "multiplicative log-uniform [0.5, 2.0], seed 42",
        },
    }
    meta = {
        "generator": "whatif-v1",
        "grouping": {"kind": "whatif", "G": G},
        "k": {"requested": k, "selected": len(perturb)},
        "perturbation": {
            "features": perturb,
            "feature_names": [names[i] for i in perturb],
            "pool": [names[i] for i in wl.CREDIT_POOL],
            "ranking": "LightGBM gain importance",
            "shock": "multiplicative 2**Uniform(-1, 1)",
            "any_zero_group_fraction": any_zero,
        },
        "seeds": {"generator": wl.WHATIF_V1_SEED},
    }
    w = wl.Workload(X, offsets, config, np.arange(len(base_X), dtype=np.int64), meta)
    built = write_workload(w, cell.dir(ctx.paths.artifacts))
    built.legacy_reference = cell.dir(ctx.paths.artifacts) / "reference.bin"
    return built


def cell_identity(cell: Cell, mdoc: dict[str, Any], built: Built) -> dict[str, Any]:
    """What a cell's content depends on; the key adds PREP_POLICY."""
    return {
        "schema_version": SCHEMA,
        "id": cell.id,
        "workload": cell.workload,
        "generator": cell.generator,
        "dataset": cell.model.dataset,
        "framework": cell.framework,
        "n_trees": cell.model.n_trees,
        "max_depth": cell.model.max_depth,
        **({"num_leaves": cell.model.num_leaves} if cell.model.num_leaves is not None else {}),
        "params": cell.param,
        "model_key": mdoc["key"],
        "data_sha256": dict(sorted(built.hashes.items())),
    }


def ensure_cell(
    ctx: Context, cell: Cell, td: TrainData, mdoc: dict[str, Any], force: bool
) -> dict[str, Any]:
    art = ctx.paths.artifacts
    cell_dir = cell.dir(art)
    fw_dir = cell.model.dir(art) / cell.framework
    built = build_workload(ctx, cell, td, mdoc)
    identity = cell_identity(cell, mdoc, built)
    content_key = fm.sha256_json(identity)
    key = fm.sha256_json({**identity, "prep_policy": PREP_POLICY})
    doc_path = cell_dir / "cell.json"
    old: dict[str, Any] = {}
    if not force and doc_path.exists():
        old = fm.read_cell(doc_path)
        if not files_match(cell_dir, old["files"]):
            old = {}
        elif old.get("key") == key:
            return old
        elif old.get("content_key") != content_key:
            old = {}

    w = built.workload
    contract = wl.contracts(w)
    varying = set(w.config["varying_features"]) | set(w.config["mono_inc_features"])
    varying |= set(w.config["mono_dec_features"])
    tl = ctx.tl_model(fw_dir / "model_treelite.bin")
    predicates = tr.varying_predicates(tl, varying)
    limits = tr.model_limits(mdoc["structure"])
    warnings = tr.predicate_warning(predicates)

    files = dict(built.files)
    hashes = dict(built.hashes)
    reuse = {r: old["files"][r] for r in ("predictions", "reference") if r in old.get("files", {})}
    if (
        contract["ok"]
        and "predictions" in reuse
        and (built.legacy_reference is None or "reference" in reuse)
    ):
        # Only the rules changed: the references on disk are still current.
        for role, rec in reuse.items():
            files[role] = cell_dir / rec["path"]
            hashes[role] = rec["sha256"]
        status = "unsupported" if limits else "ready"
    elif contract["ok"]:
        X = np.asarray(w.X)
        if built.legacy_reference is not None:
            ref = tr.gtil(tl, X)
            files["reference"] = built.legacy_reference
            hashes["reference"] = put_matrix(built.legacy_reference, ref.reshape(-1, 1))
        native = fw_dir / tr.NATIVE_NAME[cell.framework]
        if cell.generator in ("panel-v2", "ranking-sessions-v2"):
            pred_path = fw_dir / "predictions.npy"  # where the released runner reads it
        else:
            pred_path = cell_dir / "predictions.npy"
        if cell.model.derive:
            preds = tr.gtil(tl, X.astype(np.float32) if tl.input_type == "float32" else X)
        elif built.legacy_reference is None or cell.framework != "lightgbm":
            preds = tr.Reference(cell.framework, native, td.names, td.cat).predict(X)
        else:
            preds = ref
        files["predictions"] = pred_path
        hashes["predictions"] = put_npy(pred_path, preds)
        status = "unsupported" if limits else "ready"
    else:
        status = "failed"
        for v in contract["violations"]:
            log(f"    CONTRACT: {cell.id}: {v}")
    oracle = None
    if contract["ok"]:
        rows, oracle = tr.oracle(tl, np.asarray(w.X), w.group_offsets)
        files["oracle"] = cell_dir / "oracle.bin"
        hashes["oracle"] = put_matrix(files["oracle"], rows)

    for role, rec in mdoc["files"].items():
        files[f"model_{role}"] = fw_dir / rec["path"]
        hashes[f"model_{role}"] = rec["sha256"]

    offsets = w.group_offsets
    cfg = w.config
    names = cfg["feature_names"]
    k = w.meta.get("k")
    if k is not None:
        k = {**k, "realized_varying_columns": len(contract["varying_columns"])}
    # Only the cells that have them: single rows, layout calibration, derived models,
    # what-if grids.
    extra: dict[str, Any] = {
        k_: w.meta[k_]
        for k_ in ("row_order", "source", "layout", "derived", "grid")
        if k_ in w.meta
    }
    if "derivation" in mdoc:
        extra["derivation"] = mdoc["derivation"]
    doc = {
        **identity,
        "prep_policy": PREP_POLICY,
        "content_key": content_key,
        "key": key,
        "status": status,
        "horizon": w.meta.get("horizon", td.horizon),
        "grouping": {**w.meta["grouping"], **wl.sizes(offsets)},
        "k": k,
        "feature_order": names,
        "features": {
            "varying": [names[i] for i in cfg["varying_features"]],
            "mono_inc": [names[i] for i in cfg["mono_inc_features"]],
            "mono_dec": [names[i] for i in cfg["mono_dec_features"]],
            "realized_varying": [names[i] for i in contract["varying_columns"]],
        },
        "perturbation": w.meta.get("perturbation"),
        **extra,
        "seeds": {"split": mdoc["split"]["seed"], **w.meta.get("seeds", {})},
        "split": mdoc["split"],
        "train": mdoc["train"],
        **{k: mdoc[k] for k in IDENTITY_EXTRAS if k in mdoc},
        "libraries": mdoc["libraries"],
        "model": {
            "key": mdoc["key"],
            "dir": rel(fw_dir, cell_dir),
            "structure": mdoc["structure"],
            "varying_predicates_upper_bound": predicates,
        },
        "contracts": {k_: v for k_, v in contract.items() if k_ != "varying_columns"},
        "model_stats": ctx.model_stats(fw_dir / "model_treelite.bin", varying),
        "limits": limits,
        "warnings": warnings,
        "oracle": oracle,
        "files": {
            role: {"path": rel(files[role], cell_dir), "sha256": hashes[role]}
            for role in sorted(files)
        },
        "baselines": old.get("baselines", {}),
    }
    fm.write_cell(doc_path, doc)
    return doc
