"""Training, the Treelite export, reference predictions and parser preflight.

Training is seeded with 42 and its parameters are the released ones; changing
any of them changes every model.
"""

import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

FRAMEWORKS = ("lightgbm", "xgboost")
NATIVE_NAME = {"lightgbm": "model_native.txt", "xgboost": "model_native.json"}
LEARNING_RATE = 0.05
TRAIN_SEED = 42

# TreeWalker's import limits (src/parser/validation.rs, src/forest.rs).
MAX_TREE_NODES = 32767  # the light-child index is an i16
MAX_TOTAL_NODES = 32_000_000
MAX_PREDICATES = 65535  # u16 ids 0..=65534; u16::MAX is the sentinel
MAX_JSON_BYTES = 64 * 1024 * 1024  # the JSON loader's limit (src/parser/validation.rs)


def library_versions(framework: str) -> dict[str, str]:
    """The versions that determine a model's bytes."""
    return {name: _version(name) for name in ("numpy", "polars", "treelite", framework)}


def _version(dist: str) -> str:
    candidates = {"xgboost": ("xgboost", "xgboost-cpu")}.get(dist, (dist,))
    for name in candidates:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "missing"


def params(framework: str, n_trees: int, max_depth: int) -> dict[str, Any]:
    """Training parameters, as passed to the library, plus the round count."""
    if framework == "lightgbm":
        p: dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 2**max_depth - 1,
            "max_depth": max_depth,
            "learning_rate": LEARNING_RATE,
            "verbose": -1,
            "seed": TRAIN_SEED,
        }
    else:
        p = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": max_depth,
            "learning_rate": LEARNING_RATE,
            "tree_method": "hist",
            "seed": TRAIN_SEED,
        }
    return {"params": p, "num_boost_round": n_trees}


def xgb_feature_types(n_features: int, cat_indices: list[int]) -> list[str] | None:
    if not cat_indices:
        return None
    return ["c" if i in cat_indices else "q" for i in range(n_features)]


def train(
    framework: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    cat_indices: list[int],
    n_trees: int,
    max_depth: int,
    out: Path,
) -> None:
    """Train and save the native model to ``out``."""
    p = params(framework, n_trees, max_depth)
    if framework == "lightgbm":
        import lightgbm as lgb

        ds = lgb.Dataset(
            X,
            label=y,
            feature_name=feature_names,
            categorical_feature=[feature_names[i] for i in cat_indices] if cat_indices else "auto",
            free_raw_data=False,
        )
        lgb.train(p["params"], ds, num_boost_round=n_trees).save_model(str(out))
    else:
        import xgboost as xgb

        dtrain = xgb.DMatrix(
            X,
            label=y,
            feature_names=feature_names,
            feature_types=xgb_feature_types(len(feature_names), cat_indices),
            enable_categorical=bool(cat_indices),
        )
        xgb.train(p["params"], dtrain, num_boost_round=n_trees).save_model(str(out))


def load_treelite(framework: str, native: Path) -> Any:
    import treelite

    if framework == "lightgbm":
        return treelite.frontend.load_lightgbm_model(str(native))
    return treelite.frontend.load_xgboost_model(str(native))


def export_treelite(
    framework: str, native: Path, json_path: Path, bin_path: Path, json: bool = True
) -> tuple[Any, int | None]:
    """Write the binary export, which the runner and tl2cgen read, and the JSON
    dump only when ``json`` asks for it and it fits the JSON loader's limit.
    Returns the model and the dump's size, or None when it was not dumped."""
    from .formats import write_bytes

    tl = load_treelite(framework, native)
    size = None
    json_path.unlink(missing_ok=True)
    if json:
        dumped = tl.dump_as_json().encode()
        size = len(dumped)
        if size <= MAX_JSON_BYTES:
            write_bytes(json_path, dumped)
        del dumped
    write_bytes(bin_path, tl.serialize_bytes())
    return tl, size


class Reference:
    """Reference predictions for one model.

    LightGBM uses Treelite GTIL: thresholds are f64, so it matches TreeWalker
    exactly. XGBoost uses native predict: Treelite stores XGBoost thresholds
    as f32 and GTIL evaluates in f32, while TreeWalker reads the f64-promoted
    JSON thresholds, so the f32 model's own predict is the ground truth."""

    def __init__(self, framework: str, native: Path, feature_names: list[str], cat: list[int]):
        self.framework = framework
        self.feature_names = feature_names
        self.cat = cat
        if framework == "lightgbm":
            self.model = load_treelite(framework, native)
        else:
            import xgboost as xgb

            self.model = xgb.Booster()
            self.model.load_model(str(native))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.framework == "lightgbm":
            import treelite

            return treelite.gtil.predict(self.model, X).flatten().astype(np.float64)
        import xgboost as xgb

        dmat = xgb.DMatrix(
            X,
            feature_names=self.feature_names,
            feature_types=xgb_feature_types(len(self.feature_names), self.cat),
            enable_categorical=bool(self.cat),
        )
        return self.model.predict(dmat).astype(np.float64)


def gtil(tl_model: Any, X: np.ndarray) -> np.ndarray:
    import treelite

    return treelite.gtil.predict(tl_model, X).flatten().astype(np.float64)


# The stage oracle samples whole groups, in a seeded order, until it has this
# many rows or groups.
ORACLE_ROWS = 512
ORACLE_GROUPS = 64
ORACLE_SEED = 20261004


def oracle_groups(offsets: np.ndarray) -> list[int]:
    """Groups the oracle covers: a seeded sample, in ascending order."""
    sizes = np.diff(np.asarray(offsets, dtype=np.int64))
    order = np.random.default_rng(ORACLE_SEED).permutation(len(sizes))
    picked: list[int] = []
    rows = 0
    for g in order:
        if rows >= ORACLE_ROWS or len(picked) >= ORACLE_GROUPS:
            break
        picked.append(int(g))
        rows += int(sizes[g])
    return sorted(picked)


def oracle(tl_model: Any, X: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """An independent oracle for TreeWalker's stages on a sample of groups.

    ``tree_sum`` is ``math.fsum`` of Treelite GTIL's per-tree outputs, so it is the
    correctly rounded leaf sum; ``raw_margin`` is the staged finalization recomputed
    from it, ``tree_sum / divisor + base_score``, rounded at each step, with the
    divisor and the margin-scale base score Treelite parsed. Returns a matrix of
    (row, tree_sum, raw_margin) and the header the runner needs.
    """
    import math

    import treelite

    h = tl_model.get_header_accessor()
    divisor = float(tl_model.num_tree) if int(h.get_field("average_tree_output")[0]) else 1.0
    base_scores = np.asarray(h.get_field("base_scores"), dtype=np.float64)
    if base_scores.size != 1:
        raise ValueError(f"expected one base score, got {base_scores.size}")
    base = float(base_scores[0])
    groups = oracle_groups(offsets)
    rows = np.concatenate([np.arange(offsets[g], offsets[g + 1], dtype=np.int64) for g in groups])
    dtype = np.float32 if tl_model.input_type == "float32" else np.float64
    per_tree = treelite.gtil.predict_per_tree(tl_model, np.asarray(X[rows], dtype=dtype), nthread=1)
    per_tree = np.asarray(per_tree, dtype=np.float64).reshape(len(rows), tl_model.num_tree)
    out = np.empty((len(rows), 3), dtype=np.float64)
    for i, r in enumerate(rows):
        tree_sum = math.fsum(per_tree[i].tolist())
        out[i] = (float(r), tree_sum, tree_sum / divisor + base)
    header = {
        "groups": groups,
        "divisor": divisor,
        "base_score": base,
        "postprocessor": str(h.get_field("postprocessor")),
        "sigmoid_alpha": float(np.asarray(h.get_field("sigmoid_alpha"))[0]),
        "source": f"math.fsum of Treelite GTIL {treelite.__version__} predict_per_tree",
    }
    return out, header


def split_counts(framework: str, native: Path, n_features: int) -> np.ndarray:
    """How many splits each feature has in the model."""
    if framework == "lightgbm":
        import lightgbm as lgb

        counts = lgb.Booster(model_file=str(native)).feature_importance(importance_type="split")
        return np.asarray(counts, dtype=np.int64)
    import xgboost as xgb

    bst = xgb.Booster()
    bst.load_model(str(native))
    names = bst.feature_names or [f"f{i}" for i in range(n_features)]
    score = bst.get_score(importance_type="weight")
    return np.array([score.get(n, 0) for n in names], dtype=np.int64)


def structure(tl_model: Any) -> dict[str, Any]:
    """Per-model counts behind TreeWalker's import limits."""
    nodes = []
    for i in range(tl_model.num_tree):
        acc = tl_model.get_tree_accessor(i)
        nodes.append(int(acc.get_field("num_nodes")[0]))
    return {
        "trees": tl_model.num_tree,
        "total_nodes": int(sum(nodes)),
        "max_tree_nodes": int(max(nodes)),
    }


def varying_predicates(tl_model: Any, varying: set[int]) -> int:
    """An upper bound on the varying predicates the parser interns: numerical
    splits on varying features, deduplicated by (feature, threshold bits,
    default direction), plus every categorical split on one, since their
    category sets are not compared. The parser's own count can be lower, so
    this only warns; the runner's preflight loads the model and decides."""
    var = np.array(sorted(varying), dtype=np.int32)
    keys = []
    n_cat = 0
    for i in range(tl_model.num_tree):
        acc = tl_model.get_tree_accessor(i)
        split = acc.get_field("split_index")
        node_type = acc.get_field("node_type")
        on_varying = (acc.get_field("cleft") >= 0) & np.isin(split, var)
        cat = on_varying & (node_type == 2)
        n_cat += int(cat.sum())
        num = on_varying & ~cat
        k = np.zeros(int(num.sum()), dtype=[("f", "<i4"), ("t", "<u8"), ("d", "u1")])
        k["f"] = split[num]
        k["t"] = np.asarray(acc.get_field("threshold")[num], dtype=np.float64).view("<u8")
        k["d"] = acc.get_field("default_left")[num]
        keys.append(k)
    return (len(np.unique(np.concatenate(keys))) if keys else 0) + n_cat


def model_limits(s: dict[str, Any]) -> list[str]:
    """Violations of TreeWalker's per-model import limits; empty when it loads.
    The JSON size limit is not one: the runner loads the binary export."""
    out = []
    if s["max_tree_nodes"] > MAX_TREE_NODES:
        out.append(f"a tree has {s['max_tree_nodes']} nodes; limit {MAX_TREE_NODES}")
    if s["total_nodes"] > MAX_TOTAL_NODES:
        out.append(f"{s['total_nodes']} nodes; pool limit {MAX_TOTAL_NODES}")
    return out


def predicate_warning(upper_bound: int) -> list[str]:
    if upper_bound > MAX_PREDICATES:
        return [f"up to {upper_bound} varying predicates; the parser accepts {MAX_PREDICATES}"]
    return []
