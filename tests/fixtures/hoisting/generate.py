"""Regenerate tiny, synthetic Treelite models and independent GTIL references.

uv run --no-project --with treelite==4.7.0 --with numpy python tests/fixtures/hoisting/generate.py
On macOS, set DYLD_LIBRARY_PATH to the local libomp library directory.
Rust tests use the committed files and do not require Python or Treelite.
"""

import json
from pathlib import Path
import struct

import numpy as np
import treelite
from treelite.model_builder import Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation


def raw(path, values):
    values = np.asarray(values, dtype="<f8")
    path.write_bytes(struct.pack("<QQ", *values.shape) + values.tobytes())


root = Path(__file__).parent
width = 128
config = dict(n_features=3, max_group_width=width, varying_features=[0],
              mono_inc_features=[], mono_dec_features=[])
(root / "walker_config.json").write_text(json.dumps(config, indent=2) + "\n")
values = [-np.inf, -1.0, -0.0, 0.0, 1.0 - 2**-25, 1.0, 1.0 + 2**-25, np.inf, np.nan]
data = np.array([[values[r % len(values)], c, d]
                 for c in [0.0, 1.0, 40.0, np.nan]
                 for d in [0.0, 2.0, np.nan]
                 for r in range(width)], dtype=np.float64)
raw(root / "data.bin", data)

for name, dtype, categorical in [("numeric_f64", "float64", False),
                                 ("numeric_f32", "float32", False),
                                 ("categorical_f64", "float64", True)]:
    builder = ModelBuilder(
        threshold_type=dtype, leaf_output_type=dtype,
        metadata=Metadata(num_feature=3, task_type="kBinaryClf", average_tree_output=False,
                          num_target=1, num_class=[1], leaf_vector_shape=(1, 1)),
        tree_annotation=TreeAnnotation(num_tree=2, target_id=[0, 0], class_id=[0, 0]),
        postprocessor=PostProcessorFunc(name="sigmoid"), base_scores=[0.125])
    for tree in range(2):
        builder.start_tree()
        for node in range(15):
            builder.start_node(node)
            if node >= 7:
                builder.leaf([0.2, -0.3, 0.6, -0.2, 0.2, -0.3, 0.1, 0.4][node - 7] / (tree + 1))
            elif categorical and node in (1, 2):
                builder.categorical_test(feature_id=1, default_left=True, category_list=[1, 40],
                                         category_list_right_child=True,
                                         left_child_key=2*node+1, right_child_key=2*node+2)
            else:
                feature = 0 if node == 0 else 1 if node < 3 else 2
                builder.numerical_test(feature_id=feature, threshold=[1.0, 0.5, 1.5][feature],
                                       default_left=feature != 0, opname="<" if dtype == "float32" else "<=",
                                       left_child_key=2*node+1, right_child_key=2*node+2)
            builder.end_node()
        builder.end_tree()
    model = builder.commit()
    model.serialize(root / f"{name}.bin")
    (root / f"{name}.json").write_text(model.dump_as_json(pretty_print=True) + "\n")
    prediction = treelite.gtil.predict(model, data.astype(dtype), nthread=1).reshape(-1, 1)
    raw(root / f"{name}_reference.bin", prediction)
    print(f"{name}: {model.num_tree} trees, {len(data)} GTIL predictions (Treelite {treelite.__version__})")
