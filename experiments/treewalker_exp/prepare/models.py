"""Models: trained, derived from a trained one by editing its export, or reused and
brought up to the current rules; and the statistics the cells record about them."""

import resource
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import datasets as ds
from .. import formats as fm
from .. import train as tr
from .. import workloads as wl
from ..grids import Model
from .store import files_match, put_bytes

if TYPE_CHECKING:
    from .context import Context, TrainData


SCHEMA = 1
log = ds.log


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


# The version of the rules behind every derived record: a model's limits and
# JSON export, a cell's status, contract checks and preflight warnings. Cell
# keys include it and model.json records it, so a change to the rules
# re-evaluates every model and cell on resume; models are not retrained, and
# a cell's data and references are reused when only the rules changed.
# Policy 3 adds the stage oracle (oracle.bin) to every cell, and policy 4 the
# model statistics (model_stats).
PREP_POLICY = 4


# replicate: a seed replicate's entity sample; missing_values: expedia-filled's encoding.
IDENTITY_EXTRAS = ("replicate", "missing_values")


def model_identity(model: Model, fw: str, td: TrainData) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "dataset": model.dataset,
        "framework": fw,
        "source": td.source,
        "split": td.split,
        "n_trees": model.n_trees,
        "max_depth": model.max_depth,
        "horizon": td.horizon,
        "train": tr.params(fw, model.n_trees, model.max_depth),
        "categorical_features": [td.names[i] for i in td.cat],
        "feature_order": td.names,
        "libraries": tr.library_versions(fw),
        "train_data_sha256": td.sha256,
        **{k: td.identity[k] for k in IDENTITY_EXTRAS if k in td.identity},
    }


# Compiled baselines built from an older model would otherwise be picked up
# by the runner next to the new one.
STALE_BASELINES = ("tl2cgen.so", "tl2cgen.json", "lleaves.so", "lleaves.json")


def ensure_model(ctx: Context, model: Model, fw: str, td: TrainData, force: bool) -> dict[str, Any]:
    fw_dir = model.dir(ctx.paths.artifacts) / fw
    json = model.id in ctx.json_models
    identity = model_identity(model, fw, td)
    key = fm.sha256_json(identity)
    doc_path = fw_dir / "model.json"
    if not force and doc_path.exists():
        doc = fm.read_json(doc_path)
        if doc.get("key") == key and files_match(fw_dir, model_files(doc)):
            log(f"  {model.id}/{fw}: model matches its manifest, reused")
            return refresh_model(fw_dir, doc, json)
        log(f"  {model.id}/{fw}: manifest differs, retraining")
    fw_dir.mkdir(parents=True, exist_ok=True)
    for name in STALE_BASELINES:
        (fw_dir / name).unlink(missing_ok=True)

    native = fw_dir / tr.NATIVE_NAME[fw]
    log(f"  {model.id}/{fw}: training on {td.X.shape[0]:,} rows")
    t0 = time.perf_counter()
    tr.train(fw, td.X, td.y, td.names, td.cat, model.n_trees, model.max_depth, native)
    t1 = time.perf_counter()
    tl, json_bytes = tr.export_treelite(
        fw, native, fw_dir / "model_treelite.json", fw_dir / "model_treelite.bin", json
    )
    t2 = time.perf_counter()
    structure = tr.structure(tl)
    del tl
    files = {"native": native.name, "treelite_bin": "model_treelite.bin"}
    file_recs = {k: {"path": v, "sha256": fm.sha256_file(fw_dir / v)} for k, v in files.items()}
    doc = {
        **identity,
        "key": key,
        "prep_policy": PREP_POLICY,
        "files": file_recs,
        "structure": structure,
        "limits": tr.model_limits(structure),
        "treelite_json": json_record(fw_dir, json_bytes),
        "prep": {
            "train_seconds": round(t1 - t0, 3),
            "export_seconds": round(t2 - t1, 3),
            "peak_rss_bytes": peak_rss_bytes(),
            "bytes": sum((fw_dir / v).stat().st_size for v in files.values()),
        },
    }
    fm.write_json(doc_path, doc)
    log(
        f"    {structure['trees']} trees, {structure['total_nodes']:,} nodes "
        f"(largest tree {structure['max_tree_nodes']:,}), trained in {t1 - t0:.1f}s"
    )
    for v in doc["limits"]:
        log(f"    LIMIT: {v}")
    return doc


def model_files(doc: dict[str, Any]) -> dict[str, dict[str, str]]:
    """The files that make the model. The JSON dump is derived from the
    binary export and tracked on its own."""
    return {k: v for k, v in doc["files"].items() if k != "treelite_json"}


def json_record(fw_dir: Path, size: int | None) -> dict[str, Any]:
    if size is None:
        return {"status": "skipped: not a test model"}
    path = fw_dir / "model_treelite.json"
    if size > tr.MAX_JSON_BYTES:
        log(f"    JSON export skipped: {size} bytes, over the loader's limit")
        return {"bytes": size, "status": f"skipped: over the loader's limit of {tr.MAX_JSON_BYTES}"}
    return {"bytes": size, "status": "written", "path": path.name, "sha256": fm.sha256_file(path)}


def model_stats(tl: Any, varying: set[int]) -> dict[str, Any]:
    """What the kernel's layout policy weighs, from the model alone: the share
    of internal nodes on varying features and the range of events per walk step
    (train.walk_shares)."""
    return {
        "varying_share": wl.varying_share(tr.internal_nodes(tl), varying),
        "walk": tr.walk_shares(tl, varying),
    }


def config_varying(config: dict[str, Any]) -> set[int]:
    """Every feature a walker config lets vary: varying and monotonic."""
    keys = ("varying_features", "mono_inc_features", "mono_dec_features")
    return {int(f) for k in keys for f in config[k]}


# Models derived from a trained model by editing its Treelite export, for kernel
# paths no trained model reaches. They have no native model, so only TreeWalker
# times them, and their references come from Treelite GTIL.
POOLED_OFFSET = 200
WIDE_EVERY = 10
WIDE_SHIFT = -96
DERIVATIONS = {
    "pooled-categories": {
        "rule": "every category of every categorical split, and every categorical input, "
        "plus offset: the splits read the same rows, and every category is at least 32",
        "offset": POOLED_OFFSET,
    },
    "wide-sums": {
        "rule": "the leaves of one tree in every (0-based indices every - 1, 2 every - 1, "
        "...) times 2^shift: the leaf exponents span past the 126-bit exact-sum rule",
        "every": WIDE_EVERY,
        "shift": WIDE_SHIFT,
    },
}


def derive_model(bin_path: Path, derive: str) -> Any:
    """The Treelite model at ``bin_path`` with one of DERIVATIONS applied."""
    import treelite

    tl = treelite.Model.deserialize(str(bin_path))
    for i in range(tl.num_tree):
        acc = tl.get_tree_accessor(i)
        if derive == "pooled-categories":
            cats = acc.get_field("category_list")
            if len(cats):
                acc.set_field("category_list", (cats + POOLED_OFFSET).astype(cats.dtype))
        elif derive == "wide-sums":
            if i % WIDE_EVERY == WIDE_EVERY - 1:
                leaves = acc.get_field("leaf_value")
                acc.set_field("leaf_value", np.ldexp(leaves, WIDE_SHIFT).astype(leaves.dtype))
        else:
            raise ValueError(f"unknown derivation {derive}")
    return tl


def derived_inputs(model: Model, w: wl.Workload, cat: list[int]) -> wl.Workload:
    """A derived model's inputs: pooled categories shift the categorical columns."""
    if model.derive != "pooled-categories":
        return w
    if not cat:
        raise ValueError(f"{model.id}: no categorical features to pool")
    X = w.X.copy()
    X[:, cat] += POOLED_OFFSET
    meta = {**w.meta, "derived": {"model": model.derive, "shifted": cat}}
    return wl.Workload(X, w.offsets, w.config, w.entities, meta)


def ensure_derived(
    ctx: Context, model: Model, fw: str, trained: dict[str, Any], force: bool
) -> dict[str, Any]:
    """A derived model's export and model.json, rebuilt unless its key matches.
    The trained model's identity carries over, with the derivation added."""
    import treelite

    art = ctx.paths.artifacts
    src_dir = model.trained.dir(art) / fw
    fw_dir = model.dir(art) / fw
    derivation = {"name": model.derive, **DERIVATIONS[model.derive]}
    identity = {
        **{k: trained[k] for k in ("schema_version", "dataset", "framework", "source")},
        **{k: trained[k] for k in ("split", "n_trees", "max_depth", "horizon", "train")},
        **{k: trained[k] for k in ("categorical_features", "feature_order", "libraries")},
        "train_data_sha256": trained["train_data_sha256"],
        "derived_from": {"id": model.trained.id, "key": trained["key"]},
        "derivation": derivation,
        "treelite": treelite.__version__,
    }
    key = fm.sha256_json(identity)
    doc_path = fw_dir / "model.json"
    if not force and doc_path.exists():
        doc = fm.read_json(doc_path)
        if doc.get("key") == key and files_match(fw_dir, doc["files"]):
            log(f"  {model.id}/{fw}: derived model matches its manifest, reused")
            return doc
    log(f"  {model.id}/{fw}: deriving from {model.trained.id}/{fw}")
    tl = derive_model(src_dir / "model_treelite.bin", model.derive)
    put_bytes(fw_dir / "model_treelite.bin", tl.serialize_bytes())
    structure = tr.structure(tl)
    doc = {
        **identity,
        "key": key,
        "prep_policy": PREP_POLICY,
        "files": {
            "treelite_bin": {
                "path": "model_treelite.bin",
                "sha256": fm.sha256_file(fw_dir / "model_treelite.bin"),
            }
        },
        "structure": structure,
        "limits": tr.model_limits(structure),
        "treelite_json": json_record(fw_dir, None),
        "sum_headroom_bits": {
            "trained": tr.sum_headroom(ctx.tl_model(src_dir / "model_treelite.bin")),
            "derived": tr.sum_headroom(tl),
        },
    }
    fm.write_json(doc_path, doc)
    return doc


LEGACY_MODEL_FIELDS = ("treelite_json_bytes", "treelite_json_loadable")


def refresh_model(fw_dir: Path, doc: dict[str, Any], json: bool) -> dict[str, Any]:
    """Bring a reused model's derived records up to the current policy without
    retraining: the import limits, and the JSON dump, which is exported from
    the binary when the model is listed and deleted when it is not."""
    new = {
        **{k: v for k, v in doc.items() if k not in LEGACY_MODEL_FIELDS},
        "prep_policy": PREP_POLICY,
        "files": model_files(doc),
        "limits": tr.model_limits(doc["structure"]),
    }
    path = fw_dir / "model_treelite.json"
    rec = doc.get("treelite_json", {})
    current = rec.get("status") == "written" and path.exists()
    current = current and fm.sha256_file(path) == rec.get("sha256")
    if json and rec.get("status", "").startswith("skipped: over"):
        path.unlink(missing_ok=True)  # over the loader's limit: no dump, stale or not
    elif json and not current:
        import treelite

        log(f"    exporting the JSON dump from {fw_dir.name}/model_treelite.bin")
        tl = treelite.Model.deserialize(str(fw_dir / "model_treelite.bin"))
        dumped = tl.dump_as_json().encode()
        if len(dumped) <= tr.MAX_JSON_BYTES:
            fm.write_bytes(path, dumped)
        else:
            path.unlink(missing_ok=True)  # a stale dump must not outlive its record
        new["treelite_json"] = json_record(fw_dir, len(dumped))
    elif not json:
        path.unlink(missing_ok=True)
        new["treelite_json"] = json_record(fw_dir, None)
    if new != doc:
        fm.write_json(fw_dir / "model.json", new)
    return new
