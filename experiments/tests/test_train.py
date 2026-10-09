from pathlib import Path

import numpy as np
import pytest
from treelite.model_builder import Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation

from treewalker_exp import train as tr


@pytest.mark.parametrize(
    ("nodes", "total", "violations"),
    [
        (32767, 32_000_000, 0),
        (32768, 32_000_000, 1),
        (32767, 32_000_001, 1),
        (32768, 32_000_001, 2),
    ],
)
def test_model_limits_boundaries(nodes, total, violations):
    s = {"trees": 1, "max_tree_nodes": nodes, "total_nodes": total}
    assert len(tr.model_limits(s)) == violations


def test_predicate_warning_boundary():
    # Rust interns ids 0..=65534, so 65,535 predicates still load.
    assert tr.predicate_warning(65535) == []
    assert tr.predicate_warning(65536) != []


def model(splits):
    """One stump per split: ("num", feature, threshold, default_left) or
    ("cat", feature, categories)."""
    b = ModelBuilder(
        threshold_type="float64",
        leaf_output_type="float64",
        metadata=Metadata(
            num_feature=3,
            task_type="kRegressor",
            average_tree_output=False,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=TreeAnnotation(
            num_tree=len(splits), target_id=[0] * len(splits), class_id=[0] * len(splits)
        ),
        postprocessor=PostProcessorFunc(name="identity"),
        base_scores=[0.0],
    )
    for split in splits:
        b.start_tree()
        b.start_node(0)
        if split[0] == "num":
            _, f, t, d = split
            b.numerical_test(
                feature_id=f,
                threshold=t,
                default_left=d,
                opname="<=",
                left_child_key=1,
                right_child_key=2,
            )
        else:
            _, f, cats = split
            b.categorical_test(
                feature_id=f,
                default_left=True,
                category_list=cats,
                category_list_right_child=False,
                left_child_key=1,
                right_child_key=2,
            )
        b.end_node()
        for node in (1, 2):
            b.start_node(node)
            b.leaf(float(node))
            b.end_node()
        b.end_tree()
    return b.commit()


def test_varying_predicates_dedup_and_categorical_upper_bound():
    m = model(
        [
            ("num", 0, 1.0, True),
            ("num", 0, 1.0, True),  # same predicate: interned once
            ("num", 0, 1.0, False),  # another default direction
            ("num", 0, -0.0, True),
            ("num", 0, 0.0, True),  # bitwise-distinct threshold
            ("cat", 1, [1, 2]),
            ("cat", 1, [1, 2]),  # a repeated categorical predicate, counted twice
            ("num", 2, 5.0, True),  # constant feature: not a varying predicate
        ]
    )
    assert tr.varying_predicates(m, {0, 1}) == 4 + 2
    assert tr.varying_predicates(m, {0}) == 4
    assert tr.varying_predicates(m, set()) == 0
    s = tr.structure(m)
    assert s == {"trees": 8, "total_nodes": 24, "max_tree_nodes": 3}


def test_export_skips_json_over_the_loader_limit(tmp_path, monkeypatch):
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 3)), rng.integers(0, 2, 200).astype(float)
    native = tmp_path / "model_native.txt"
    lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), 3).save_model(native)
    json_path, bin_path = tmp_path / "model_treelite.json", tmp_path / "model_treelite.bin"
    _, size = tr.export_treelite("lightgbm", native, json_path, bin_path)
    assert json_path.stat().st_size == size and bin_path.exists()
    monkeypatch.setattr(tr, "MAX_JSON_BYTES", size - 1)
    _, again = tr.export_treelite("lightgbm", native, json_path, bin_path)
    assert again == size and not json_path.exists() and bin_path.exists()
    monkeypatch.undo()
    _, none = tr.export_treelite("lightgbm", native, json_path, bin_path, json=False)
    assert none is None and not json_path.exists() and bin_path.exists()


FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "import"


@pytest.mark.parametrize("name", ["average_base", "sigmoid_f32", "random_forest", "identity"])
def test_oracle_stages_are_the_fsum_and_the_staged_finalization(name):
    import math

    import treelite

    from treewalker_exp import formats as fm

    tl = treelite.Model.deserialize(str(FIXTURES / f"{name}.bin"))
    X = np.array(fm.read_matrix(FIXTURES / "data.bin"))
    offsets = np.arange(0, X.shape[0] + 1, 128)
    rows, header = tr.oracle(tl, X, offsets)
    assert header["groups"] == sorted(header["groups"])
    assert len(rows) == 128 * len(header["groups"])
    dtype = np.float32 if tl.input_type == "float32" else np.float64
    for r, tree_sum, raw_margin in rows[:20]:
        leaves = treelite.gtil.predict_per_tree(tl, X[int(r) : int(r) + 1].astype(dtype)).ravel()
        assert tree_sum == math.fsum(float(v) for v in leaves)
        assert raw_margin == tree_sum / header["divisor"] + header["base_score"]
    averaging = name in ("average_base", "random_forest")
    assert header["divisor"] == (tl.num_tree if averaging else 1.0)


def test_oracle_rounds_once_where_tree_order_would_not():
    # fsum([1, 2^-53, 2^-53]) is 1 + 2^-52; adding in order gives 1.0.
    import math

    assert math.fsum([1.0, 2**-53, 2**-53]) == 1.0 + 2**-52
    assert (1.0 + 2**-53) + 2**-53 == 1.0
