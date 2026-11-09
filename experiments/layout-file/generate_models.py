# /// script
# requires-python = ">=3.14"
# dependencies = ["lightgbm==4.7.0", "treelite==4.7.2", "numpy==2.5.3"]
#
# [[tool.uv.index]]
# name = "pypi"
# url = "https://pypi.org/simple"
# ///
"""Two public LightGBM models to measure loading with, each as native text, a
Treelite checkpoint and a grouping configuration:

uv run --locked --script experiments/layout-file/generate_models.py DIR
cargo run --release --manifest-path experiments/layout-file/Cargo.toml \\
    --example load_peak -- DIR/numeric.txt DIR/numeric.json

numeric has 800 trees of 255 leaves over 40 numeric features; categorical has 800
trees of 127 leaves over 20 features, 6 of them categorical with 200 to 5,000
categories. Training is deterministic, so every run writes the same models.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import treelite

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
common = dict(objective="regression", min_data_in_leaf=5, learning_rate=0.05, seed=1,
              deterministic=True, force_row_wise=True, num_threads=1, verbose=-1)


def save(name, booster):
    path = out / f"{name}.txt"
    booster.save_model(path)
    treelite.frontend.load_lightgbm_model(str(path)).serialize(out / f"{name}.bin")
    config = dict(n_features=booster.num_feature(), max_group_width=64,
                  varying_features=[0, 1, 2, 3], mono_inc_features=[], mono_dec_features=[])
    (out / f"{name}.json").write_text(json.dumps(config) + "\n")
    print(name, booster.num_trees(), "trees,", booster.num_feature(), "features")


n, f = 100_000, 40
x = rng.standard_normal((n, f))
y = x[:, :8] @ rng.standard_normal(8) + np.sin(3 * x[:, 8]) + 0.3 * rng.standard_normal(n)
params = dict(common, num_leaves=255)
save("numeric", lgb.train(params, lgb.Dataset(x, y), num_boost_round=800))

n, f = 200_000, 20
x = rng.standard_normal((n, f))
cards = [5000, 3000, 2000, 1000, 500, 200]
effects = []
for j, c in enumerate(cards):
    codes = np.minimum(rng.zipf(1.3, n) - 1, c - 1)
    x[:, j] = codes
    effects.append(rng.standard_normal(c)[codes])
y = np.sum(effects, axis=0) + x[:, 6] - x[:, 7] + 0.3 * rng.standard_normal(n)
params = dict(common, num_leaves=127, min_data_per_group=5, cat_smooth=1, cat_l2=1,
              max_cat_threshold=512, max_cat_to_onehot=4)
data = lgb.Dataset(x, y, categorical_feature=list(range(len(cards))),
                   params={"max_bin": 255, "max_cat_threshold": 512})
save("categorical", lgb.train(params, data, num_boost_round=800))
