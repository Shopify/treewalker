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


def export_treelite(framework: str, native: Path, json_path: Path, bin_path: Path) -> Any:
    from .formats import write_bytes

    tl = load_treelite(framework, native)
    write_bytes(json_path, tl.dump_as_json().encode())
    write_bytes(bin_path, tl.serialize_bytes())
    return tl


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
