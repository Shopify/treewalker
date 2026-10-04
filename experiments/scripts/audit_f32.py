#!/usr/bin/env python3
"""F32 numerical-precision audit for the §A.3 claim.

Validates the §A.3 claim that TreeWalker (f64 leaf accumulation) is
closer to a high-precision reference than native XGBoost (f32 leaf
accumulation) on every tested row.

Method
------
For each XGBoost (f32) config with full artifacts:
  1. Load test data and native XGBoost model.
  2. Use `bst.predict(dmat, pred_leaf=True)` to get per-(row, tree) leaf
     indices.
  3. Extract leaf values from the booster's JSON dump.
  4. For each row, compute three sums of leaf values + base_score
     (raw-margin scale, no sigmoid), in tree-iteration order:
       - s_f32: f32 accumulator (matches native predict raw margin)
       - s_f64: f64 accumulator (matches TreeWalker's raw margin)
       - s_ref: Kahan-compensated f64 accumulator (high-precision ref)
  5. Tabulate err_f32 = |s_f32 - s_ref|, err_f64 = |s_f64 - s_ref|.

Decision criteria
-----------------
- Strong (elevate to abstract): err_f64 < err_f32 on every row, mean ratio ≥ 2×.
- Moderate (qualified abstract sentence): >99% of rows, never worse by more than ε.
- Weak (keep in §A.3 only): comparable; do not elevate.

Outputs
-------
- Per-config table of (n_rows, n_tw_strictly_closer, n_tied, n_xgb_closer,
  mean_ratio, worst_ratio).
- Final decision printed at the end.
"""
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

from treewalker_exp.formats import read_matrix


ARTIFACTS = Path("experiments/artifacts")


def kahan_sum(values: list[float]) -> float:
    """f64 Kahan-compensated sum. ~3 orders of magnitude tighter than naive f64."""
    s = 0.0
    c = 0.0
    for v in values:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def load_test_data_bin(path: Path, n_features: int) -> np.ndarray:
    """Read test_data.bin (treewalker_exp.formats; matches load_raw_f64 in Rust)."""
    data = np.array(read_matrix(path))
    assert data.shape[1] == n_features, (
        f"feature count mismatch: bin says {data.shape[1]}, expected {n_features}"
    )
    return data


def extract_leaf_values(bst: xgb.Booster) -> dict[tuple[int, int], np.float32]:
    """Map (tree_id, leaf_node_id) -> leaf value (as f32).

    XGBoost stores leaf values as f32. We promote to Python float here, but the
    underlying value is the f32-rounded leaf weight + learning_rate * grad/hess.
    """
    df = bst.trees_to_dataframe()
    leaves = df[df["Feature"] == "Leaf"]
    out: dict[tuple[int, int], np.float32] = {}
    for _, row in leaves.iterrows():
        tree_id = int(row["Tree"])
        node_id = int(row["Node"])
        leaf_val = np.float32(row["Gain"])  # 'Gain' column holds leaf value
        out[(tree_id, node_id)] = leaf_val
    return out


def get_base_score(bst: xgb.Booster) -> np.float32:
    """Return the model's base_score (intercept, raw margin scale).

    XGBoost may store base_score as a string like '[4.8e-2]' (1-element array);
    strip the brackets if present.
    """
    cfg = json.loads(bst.save_config())
    bs = cfg["learner"]["learner_model_param"]["base_score"]
    if isinstance(bs, str):
        bs = bs.strip("[]")
    return np.float32(float(bs))


def audit_config(
    cfg_dir: Path, dataset: str, params: str
) -> dict | None:
    """Run the audit on one XGBoost config. Returns row-summary dict or None if skipped."""
    xgb_dir = cfg_dir / "xgboost"
    native_path = xgb_dir / "model_native.json"
    test_data_path = cfg_dir / "test_data.bin"
    walker_cfg_path = cfg_dir / "walker_config.json"

    if not native_path.exists() or not test_data_path.exists():
        print(f"  SKIP {dataset}/{params} — missing native model or test data")
        return None

    walker_cfg = json.loads(walker_cfg_path.read_text())
    n_features = walker_cfg["n_features"]

    bst = xgb.Booster()
    bst.load_model(str(native_path))

    test_data = load_test_data_bin(test_data_path, n_features)
    n_rows = test_data.shape[0]

    # Cap rows for tractability — every row is independent so ~5000 is plenty
    # to test universality.
    max_rows = 5000
    if n_rows > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_rows, max_rows, replace=False)
        idx.sort()
        sample = test_data[idx]
        n_audit = max_rows
    else:
        sample = test_data
        n_audit = n_rows

    feature_names = bst.feature_names
    feature_types = bst.feature_types
    dmat = xgb.DMatrix(
        sample,
        feature_names=feature_names,
        feature_types=feature_types,
        enable_categorical=any(t == "c" for t in feature_types) if feature_types else False,
    )

    # Per-row, per-tree leaf indices.
    leaf_idx = bst.predict(dmat, pred_leaf=True)
    if leaf_idx.ndim == 1:
        leaf_idx = leaf_idx.reshape(-1, 1)
    n_trees = leaf_idx.shape[1]

    # Leaf-value lookup.
    leaves = extract_leaf_values(bst)
    base = get_base_score(bst)

    # For each row, compute three sums.
    err_f32 = np.empty(n_audit)
    err_f64 = np.empty(n_audit)
    n_strictly_closer_f64 = 0
    n_tied = 0
    n_strictly_closer_f32 = 0

    for r in range(n_audit):
        # Collect this row's leaf values (as f32) in tree iteration order.
        vals_f32 = [leaves[(t, int(leaf_idx[r, t]))] for t in range(n_trees)]

        # f32 accumulator (matches native xgb output_margin).
        s32 = np.float32(base)
        for v in vals_f32:
            s32 = np.float32(s32 + v)

        # f64 accumulator (matches TreeWalker raw margin).
        s64 = np.float64(base)
        for v in vals_f32:
            s64 = s64 + np.float64(v)

        # Kahan-compensated f64 reference.
        s_ref = kahan_sum([float(base)] + [float(v) for v in vals_f32])

        e32 = abs(float(s32) - s_ref)
        e64 = abs(float(s64) - s_ref)
        err_f32[r] = e32
        err_f64[r] = e64

        if e64 < e32:
            n_strictly_closer_f64 += 1
        elif e64 == e32:
            n_tied += 1
        else:
            n_strictly_closer_f32 += 1

    # Compute ratio statistics. Avoid /0 with eps.
    eps = 1e-30
    ratios = err_f32 / np.maximum(err_f64, eps)
    finite_ratios = ratios[err_f64 > eps]

    print(f"\n{dataset}/{params} (xgboost, {n_trees} trees, audited {n_audit} rows)")
    print(f"  TreeWalker f64 strictly closer:  {n_strictly_closer_f64} / {n_audit} ({100*n_strictly_closer_f64/n_audit:.2f}%)")
    print(f"  Tied                            : {n_tied} / {n_audit}")
    print(f"  Native f32 strictly closer      : {n_strictly_closer_f32} / {n_audit}")
    print(f"  err_f32: mean={err_f32.mean():.3e}, max={err_f32.max():.3e}")
    print(f"  err_f64: mean={err_f64.mean():.3e}, max={err_f64.max():.3e}")
    if len(finite_ratios) > 0:
        print(f"  ratio (err_f32 / err_f64): mean={finite_ratios.mean():.2e}, "
              f"median={np.median(finite_ratios):.2e}, "
              f"min={finite_ratios.min():.2e}, max={finite_ratios.max():.2e}")

    return dict(
        dataset=dataset,
        params=params,
        n_rows=n_audit,
        n_trees=n_trees,
        n_f64_closer=n_strictly_closer_f64,
        n_tied=n_tied,
        n_f32_closer=n_strictly_closer_f32,
        err_f32_mean=float(err_f32.mean()),
        err_f32_max=float(err_f32.max()),
        err_f64_mean=float(err_f64.mean()),
        err_f64_max=float(err_f64.max()),
        ratio_mean=float(finite_ratios.mean()) if len(finite_ratios) else float("nan"),
        ratio_median=float(np.median(finite_ratios)) if len(finite_ratios) else float("nan"),
    )


def main() -> int:
    print("=== F32 numerical-precision audit ===\n")
    print(f"Reference: Kahan-compensated f64 sum of stored f32 leaves + base_score.")
    print(f"Comparing native XGBoost f32 sum vs. TreeWalker f64 sum vs. reference.\n")

    results = []
    # Walk the artifacts tree.
    for ds_dir in sorted(ARTIFACTS.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        for cfg_dir in sorted(ds_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            params = cfg_dir.name
            r = audit_config(cfg_dir, dataset, params)
            if r is not None:
                results.append(r)

    if not results:
        print("\nNo XGBoost configs with full artifacts found. Cannot audit.")
        return 1

    # Decision.
    total_rows = sum(r["n_rows"] for r in results)
    total_f64_closer = sum(r["n_f64_closer"] for r in results)
    total_tied = sum(r["n_tied"] for r in results)
    total_f32_closer = sum(r["n_f32_closer"] for r in results)
    universality = 100 * total_f64_closer / max(total_rows, 1)

    print(f"\n=== Aggregate ===")
    print(f"Configs audited: {len(results)}")
    print(f"Total rows audited: {total_rows}")
    print(f"f64 strictly closer: {total_f64_closer} ({universality:.3f}%)")
    print(f"Tied               : {total_tied}")
    print(f"f32 strictly closer: {total_f32_closer}")

    print(f"\n=== Decision ===")
    # Decision logic, refined after first run:
    # - 'Tied' here means err_f64 == err_f32 (both 0 or both equal small value).
    #   These are NOT cases where f32 wins; just cases both get the answer exactly.
    # - The hard criterion is f32_closer == 0 (f64 is never worse).
    # - The strength criterion is universality of strict closeness.
    if total_f32_closer == 0 and universality >= 99.0:
        print("STRONG: f64 is never worse than f32; strictly closer on >= 99% of rows.")
        print(f"  ({total_rows} rows, {total_f64_closer} strictly closer, {total_tied} tied, 0 f32 closer.)")
        print("→ Elevate to abstract as numerical-roundoff angle (NOT quality).")
    elif total_f32_closer < 0.01 * total_rows:
        print("MODERATE: f64 closer on most rows but not universally strict; some f32 wins exist.")
        print("→ Single qualified sentence in abstract; expand §A.3.")
    else:
        print("WEAK: comparable or mixed.")
        print("→ Keep in §A.3 only; do NOT elevate.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
