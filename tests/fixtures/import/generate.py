"""Independent fixtures; normal Rust tests need no Python.

DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib uv run --no-project --with treelite==4.7.0 --with numpy==2.2.6 --with scikit-learn==1.6.1 --with scipy==1.15.3 python tests/fixtures/import/generate.py
"""
import json
from pathlib import Path
import struct
import numpy as np
import treelite
import sklearn
from sklearn.ensemble import RandomForestRegressor
from treelite.model_builder import Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation

root = Path(__file__).parent
manifest = dict(treelite=treelite.__version__, numpy=np.__version__, sklearn=sklearn.__version__, models=[])
values = [-np.inf, -1.0, -0.5, -0.0, 0.0, np.nextafter(0.0, 1.0),
          np.nextafter(1.0, 0.0), 1.0 - 2**-25, 1.0, 1.0 + 2**-25, np.nextafter(1.0, 2.0),
          1.99, 2.0, 31.0, 32.0, 40.0, 63.0, 64.0, 8159.0, 8160.0, 2**32, np.inf, np.nan]
data = np.array([[values[i % len(values)], c] for c in [-1.0, -0.5, 0.0, 1.0, 40.0, 64.0, np.nan] for i in range(128)], dtype='<f8')
def raw(path, a):
    a = np.asarray(a, dtype='<f8')
    path.write_bytes(struct.pack('<QQ', *a.shape) + a.tobytes())
raw(root / 'data.bin', data)
(root / 'walker_config.json').write_text(json.dumps(dict(n_features=2, max_group_width=128, varying_features=[0], mono_inc_features=[], mono_dec_features=[]), indent=2)+'\n')

def save(name, model, dtype='float64', json_ok=True, inputs=data):
    model.serialize(root / f'{name}.bin')
    if json_ok:
        dumped = model.dump_as_json(pretty_print=True)
        json.loads(dumped)  # never repair invalid dumps
        (root / f'{name}.json').write_text(dumped+'\n')
    ref = treelite.gtil.predict(model, inputs.astype(dtype), nthread=1).reshape(-1, 1)
    raw(root / f'{name}_reference.bin', ref)
    manifest['models'].append(dict(name=name, input_dtype=dtype, json=json_ok,
        oracle='Treelite GTIL 4.7.0', atol=1e-7 if dtype=='float32' else 1e-14, rtol=1e-7 if dtype=='float32' else 1e-14))

def build(name, dtype='float64', task='kBinaryClf', post='sigmoid', alpha=1.0, average=False, base=0.125, cat=False, invert=False, threshold=1.0, op=None, ids=None, json_ok=True):
    builder = ModelBuilder(threshold_type=dtype, leaf_output_type=dtype,
        metadata=Metadata(num_feature=2, task_type=task, average_tree_output=average,
            num_target=1, num_class=[1], leaf_vector_shape=(1,1)),
        tree_annotation=TreeAnnotation(num_tree=2, target_id=ids or [0,0], class_id=[0,0]),
        postprocessor=PostProcessorFunc(name=post, sigmoid_alpha=alpha), base_scores=[base])
    for tree in range(2):
        builder.start_tree()
        builder.start_node(0)
        if cat:
            builder.categorical_test(feature_id=tree, default_left=tree==0,
                category_list=[1, 40] if tree==0 else [1, 2], category_list_right_child=invert,
                left_child_key=1, right_child_key=2)
        else:
            builder.numerical_test(feature_id=tree, threshold=threshold, default_left=tree==0,
                opname=op or ('<' if dtype=='float32' else '<='), left_child_key=1, right_child_key=2)
        builder.end_node()
        for node, val in [(1, 0.25*(tree+1)), (2, -0.5*(tree+1))]:
            builder.start_node(node); builder.leaf(val); builder.end_node()
        builder.end_tree()
    save(name, builder.commit(), dtype, json_ok)

build('sigmoid_f64')
build('sigmoid_f32', dtype='float32')
build('sigmoid_alpha', alpha=2.5)
build('identity', task='kRegressor', post='identity', base=3.0)
build('ranking', task='kLearningToRank', post='identity', base=-0.75)
build('average_base', task='kRegressor', post='identity', average=True, base=3.0)
build('categories_left', cat=True)
build('categories_right', cat=True, invert=True)
build('categories_f32', dtype='float32', cat=True, invert=True)
build('lt_zero', threshold=-0.0, op='<')
build('positive_infinity', threshold=np.inf, op='<', json_ok=False)
build('negative_infinity', threshold=-np.inf, op='<=', json_ok=False)

train_x = np.array([[0,0], [1,0], [2,0], [3,0], [0,1], [1,1], [2,1], [3,1]], dtype=np.float64)
train_y = np.array([0.0, 1.0, 2.0, 4.0, 1.0, 2.0, 3.0, 5.0])
rf = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=42, n_jobs=1).fit(train_x, train_y)
model = treelite.sklearn.import_model(rf)
save('random_forest', model)
# Native sklearn oracle on its finite input domain, with f32-rounded features.
raw(root / 'rf_native_data.bin', train_x)
raw(root / 'rf_native_reference.bin', rf.predict(train_x).reshape(-1,1))
# Real Treelite exports outside the supported scalar boundary.
for name, targets, classes, shape, task, post, leaves in [
    ('reject_multiclass', 1, [2], (1,2), 'kMultiClf', 'softmax', [0.25, 0.75]),
    ('reject_multitarget', 2, [1,1], (2,1), 'kRegressor', 'identity', [1.0, 2.0]),
    ('reject_vector', 1, [1], (1,1), 'kRegressor', 'identity', [1.0]),
    ('reject_postprocessor', 1, [1], (1,1), 'kRegressor', 'exponential', 1.0),
]:
    b = ModelBuilder(threshold_type='float64', leaf_output_type='float64',
        metadata=Metadata(num_feature=2, task_type=task, average_tree_output=False,
            num_target=targets, num_class=classes, leaf_vector_shape=shape),
        tree_annotation=TreeAnnotation(num_tree=1, target_id=[-1 if targets > 1 else 0],
            class_id=[-1 if max(classes) > 1 else 0]),
        postprocessor=PostProcessorFunc(name=post), base_scores=[0.0]*(targets*max(classes)))
    b.start_tree(); b.start_node(0); b.leaf(leaves); b.end_node(); b.end_tree()
    m = b.commit()
    m.serialize(root / f'{name}.bin')
    (root / f'{name}.json').write_text(m.dump_as_json(pretty_print=True)+'\n')

(root / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
