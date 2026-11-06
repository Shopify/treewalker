# /// script
# requires-python = ">=3.14"
# dependencies = ["lightgbm==4.7.0", "treelite==4.7.2", "numpy==2.5.3"]
#
# [[tool.uv.index]]
# name = "pypi"
# url = "https://pypi.org/simple"
# ///
"""Small native text / Treelite checkpoint pairs; Rust tests need no Python.

uv run --locked --script tests/fixtures/import/generate_lightgbm.py

Trained models exercise real exports. Hand-written synthetic trees additionally
cover all decision bits, including combinations training need not produce.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import treelite

root = Path(__file__).parent / "lightgbm"
root.mkdir(exist_ok=True)
names = []


def save(name, text):
    # The training parameters and feature importances do not affect inference.
    text = text.split("end of trees")[0].rstrip() + "\n\nend of trees\n"
    path = root / f"{name}.txt"
    path.write_text(text)
    treelite.frontend.load_lightgbm_model(str(path)).serialize(root / f"{name}.bin")
    names.append(name)


rng = np.random.default_rng(42)
x = rng.normal(size=(96, 2))
x[::7, 0] = np.nan
y = np.nan_to_num(x[:, 0]) + 0.5 * x[:, 1]
common = dict(verbosity=-1, num_threads=1, seed=42, num_leaves=4,
              min_data_in_leaf=3, max_bin=16, learning_rate=0.2)
for name, objective, labels, extra in [
    ("regression", "regression", y, {}),
    ("binary", "binary", (y > 0).astype(float), dict(sigmoid=1.23456789)),
    ("xentropy", "xentropy", 1 / (1 + np.exp(-y)), {}),
    ("lambdarank", "lambdarank", np.clip((y + 2).astype(int), 0, 4), {}),
    ("rf", "regression", y, dict(boosting="rf", bagging_fraction=0.8,
                                 bagging_freq=1, feature_fraction=0.8)),
    ("single_leaf", "regression", np.ones(96), {}),
]:
    dataset = lgb.Dataset(x, label=labels, group=[12] * 8 if "rank" in name else None)
    model = lgb.train(dict(common, objective=objective, **extra), dataset, num_boost_round=3)
    text = model.model_to_string()
    if name == "regression":
        save("custom", "\n".join(line for line in text.splitlines()
                                 if not line.startswith("objective=")))
    else:
        save(name, text)

cx = np.column_stack([np.tile(np.arange(100), 8), rng.normal(size=800)])
cy = np.isin(cx[:, 0], [1, 31, 32, 40, 63, 64, 95, 99]).astype(float)
categorical = lgb.train(dict(common, objective="regression", cat_smooth=0,
                            cat_l2=0, min_data_per_group=1, max_cat_to_onehot=1),
                        lgb.Dataset(cx, label=cy, categorical_feature=[0]), num_boost_round=2)
save("categorical_trained", categorical.model_to_string())

header = "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nmax_feature_idx=1\nobjective=regression\n"
numeric = []
for i, decisions in enumerate(["0 2 4", "6 8 10"]):
    # Both directions of all three missing types. None/Zero deliberately
    # disagree with 0 <= threshold, which must override their default bits.
    numeric.append(f"""Tree={i}
num_leaves=4
num_cat=0
split_feature=0 1 0
threshold=0.5 -0.5 1
decision_type={decisions}
left_child=1 -1 -3
right_child=2 -2 -4
leaf_value=0.125 -0.25 0.5 -1
leaf_count=0 7 11 13
internal_count=31 7 24
""")
save("missing_types", header + "\n".join(numeric))


def words(categories):
    result = [0] * (max(categories) // 32 + 1)
    for c in categories:
        result[c // 32] |= 1 << (c % 32)
    return result


a, b = words([1, 31, 32, 40, 63, 64, 95]), words([0, 34, 65, 99])
save("category_words", header + f"""Tree=0
num_leaves=3
num_cat=2
split_feature=0 1
threshold=0 1
decision_type=7 11
left_child=1 -1
right_child=-3 -2
leaf_value=1 -2 4
leaf_count=3 5 7
internal_count=15 8
cat_boundaries=0 {len(a)} {len(a) + len(b)}
cat_threshold={' '.join(map(str, a + b))}
""")

(root / "manifest.json").write_text(json.dumps(dict(
    lightgbm=lgb.__version__, treelite=treelite.__version__, models=names), indent=2) + "\n")
