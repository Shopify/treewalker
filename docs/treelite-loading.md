# Loading Treelite models

TreeWalker imports a validated scalar subset of Treelite directly in Rust.
Python and `libtreelite` are needed only to export models or regenerate the
optional reference fixtures, not to load or execute models in the Rust library.

## Export and load

Binary checkpoints are recommended, especially for large models or infinite
thresholds. Treelite can emit invalid JSON for nonfinite values; TreeWalker
rejects those dumps rather than repairing them.

```python
import treelite  # fixture/reference version: 4.7.2
model = treelite.frontend.load_lightgbm_model("model.txt")
model.serialize("model.bin")
# Optional, for debugging: model.dump_as_json()
```

A Treelite JSON dump is different from native XGBoost JSON. Convert native
models with the appropriate Treelite frontend first.

```rust,no_run
use treewalker_gbdt::{Forest, LoadError};

fn main() -> Result<(), LoadError> {
    let forest = Forest::load("model.bin", "walker_config.json")?;
    // One predictor per worker; it owns the scratch space and is reused.
    let mut predictor = forest.predictor();
    // Row-major data in the model's original feature order: one group of one row.
    let data = vec![0.0; forest.config().n_features()];
    let mut output = [0.0];
    predictor.predict_group(&data, &mut output);
    Ok(())
}
```

For memory, streams, custom filenames, or programmatic grouping, select the
format explicitly:

```rust,no_run
use treewalker_gbdt::{Forest, LoadError, LoadOptions, ModelFormat, WalkerConfig};

fn load(bytes: &[u8]) -> Result<Forest, LoadError> {
    let config = WalkerConfig::builder(4)
        .max_group_width(14)
        .varying([2])
        .increasing([3])
        .decreasing([1])
        .build()?;
    Forest::from_bytes(bytes, ModelFormat::TreeliteBinaryV4, config, &LoadOptions::default())
}
```

`Forest::from_reader` accepts any `std::io::Read` with the same remaining
arguments, and `Forest::load_with` takes `LoadOptions` with the two paths.
Path-based loading recognizes exactly `.bin` and `.json`; other extensions
return an error pointing to explicit format selection.

Every loader returns `Result`. `LoadError` distinguishes `Io`,
`MalformedModel`, `MalformedConfig`, `Unsupported`, and `Limit`, implements
`Display` and `std::error::Error`, and is non-exhaustive. For example, an
exported multiclass model returns `LoadError::Unsupported` with its task or
output dimensions instead of producing a scalar prediction.

## Supported subset

| Property | Support |
|---|---|
| Binary | Little-endian Treelite v4 checkpoints, producer versions 4.0–4.7, no extensions |
| JSON | Strict Treelite scalar dump schema, as exercised with 4.7.2; no syntax repairs or unknown fields |
| Tasks | Scalar binary classification, regression, and ranking |
| Outputs | One target, one class/output slot, scalar leaves, nonempty forest |
| Tree assignments | Every tree contributes to the sole output; target and class IDs must both be 0 |
| Aggregation | Sum or average, with base score added afterwards |
| Postprocessors | Exactly `identity` or `sigmoid`; finite positive sigmoid alpha |
| Types | Matching float64 thresholds/leaves, or matching float32 thresholds/leaves |
| Numeric operators | float64 `<` and `<=`; float32 `<` |
| Categories | Nonnegative category IDs up to 8159; membership in either child direction |
| Features / grouped rows | 1–64 features; any positive maximum group width |

Multiclass/multiple targets, vector leaves (including singleton vectors), other
postprocessors/operators/type combinations, nonfinite leaves/base scores,
unknown versions and extensions are rejected. Framework identity alone is not
a compatibility guarantee: scalar random-forest regression is covered by an
independent sklearn fixture; vector-leaf random-forest classifiers are not.
Treelite's wildcard -1 assignments require vector leaves, even for a singleton
output shape, so they are outside this scalar-leaf subset.

For every row, all prediction entry points use the same finalization:

```text
tree_sum = sum(tree_i(row))
divisor  = tree_count if average_tree_output else 1
margin   = tree_sum / divisor + base_score
output   = margin                           # identity
output   = 1 / (1 + exp(-alpha * margin))    # sigmoid
```

The base score is stored on the forest, rather than distributed across leaves.
This can change the last few rounding bits from older TreeWalker versions.

Prediction computes `tree_sum` exactly: every leaf value is
an integer multiple of 2^-e for one scale e per model, the scaled leaves add as
128-bit integers, and the sum is rounded to float64 once. The result is the float64
nearest to the true sum and does not depend on tree order, node layout or group
width. This needs the scaled sums to fit in 126 bits; a model with a non-finite leaf
or leaf exponents too far apart adds in float64 in tree order instead, which
`Forest::exact_sums` reports. The research feature's `predict_full_walk` and the
`disable_exact_sums` ablation always add in float64 in tree order, so they can
differ from production prediction in the last bits.

Float64 numerical inputs compare in float64. `<` thresholds are normalized to
`<= next_down(threshold)`, including signed zero and positive infinity. Float64
`< -infinity` and all NaN thresholds are explicitly rejected because this
normalization cannot represent them exactly. Float64 `<= -infinity` is valid.
Float32 models round numerical inputs and thresholds to float32 before `<`;
categorical inputs are also rounded to float32. Leaves are promoted to float64
and summed as above, so native XGBoost bitwise parity is not promised. Use float32 input
to Treelite GTIL when testing this policy; GTIL with float64 input can select a
different branch near a float32 threshold.

NaN inputs take the declared missing branch. For categorical splits, other
out-of-range inputs take the non-membership branch, independently of missing
routing. In-range nonnegative fractional categories truncate toward zero,
following GTIL. Membership-right nodes are normalized by swapping their
children and missing routing; compact nodes remain 16 bytes.

## Grouping contract

The five-field wire schema is unchanged:

```json
{
  "n_features": 4,
  "max_group_width": 14,
  "varying_features": [1, 2, 3],
  "mono_inc_features": [3],
  "mono_dec_features": [1]
}
```

`WalkerConfig::from_json` and `from_file` validate through the same
`WalkerConfig::builder`. Indexes must be in range and unique within each list,
and the increasing and decreasing sets must be disjoint. Monotonic features are
varying whether or not `varying_features` lists them. The complement of the
varying set is constant, so the builder requires `varying(..)` or
`all_varying()`: an empty `varying([])` declares every feature constant.
All-varying and no-monotonic configurations are valid. The configuration's
fields are private, so a loaded forest's classification cannot change.

The caller preserves row order and the trained model's column order, and
ensures declared constant features agree within each group and declared
monotonic features have the promised order. Matching feature counts alone
cannot validate feature order. TreeWalker does not infer grouping from feature
names, training monotonic constraints, or split frequency, and does not scan
every input group to verify the caller's equality/monotonicity promises.
Application-specific requirements for what constitutes a complete group belong
to the producer/caller.

A `Predictor` takes row-major input with `n_features` values per row and writes
one output per row. `predict_group` predicts one group, `predict_groups` the
groups between consecutive offsets (starting at 0, ending at the row count,
strictly increasing), and `predict_fixed` consecutive groups of one width, the
last of which may be shorter. Every group has at most `max_group_width` rows;
empty input is a no-op. A call whose input length is not a multiple of
`n_features`, whose output length differs from the row count, or whose grouping
is invalid panics before writing any output.

## Bounded import

These limits bound input, representation, and decoding work; they are not a
promise about peak process RSS. Large legitimate forests should use binary.

| Resource | Limit |
|---|---:|
| Binary bytes consumed | 4 GiB |
| Individual binary field array | 64 MiB |
| JSON input | 64 MiB |
| Config JSON | 64 KiB |
| Attributes | 16 MiB |
| JSON nesting | 64 |
| Trees | 1,000,000 |
| Nodes per tree | 32,767 (i16 child indexes) |
| Tree depth | 256 edges |
| Total nodes | 32,000,000 |
| Pooled category bitsets | 256 MiB (within u32 offsets) |
| Category words per predicate | 255 (u8) |
| Referenced category-list entries per binary tree | 16,777,216 |
| Varying predicate IDs | 65,535; u16::MAX remains the sentinel |

The binary reader reuses per-tree buffers and never materializes whole-model
JSON. Declared lengths are checked before allocation or multiplication. Both
readers validate tree topology iteratively before layout: unique node IDs,
valid children, reachability, cycles and shared-child rejection. Optional
statistics may be absent; present field arrays must have consistent lengths.
Tree ordering may temporarily duplicate node/bitset pools; disabling it with
`LoadOptions::disable_tree_ordering` reduces peak import memory.

## Verification and regeneration

`cargo test --release` uses committed, small Treelite-generated binary/JSON
pairs, GTIL references, and a native sklearn random-forest reference. It needs
no Python, private models, network, or experiment artifacts. Tests cover all
workspace widths, numeric boundaries, missing and out-of-range categories,
aggregation order, loader variants, predictor call contracts, malformed topology,
truncated binary fields, size overflow declarations, and unsupported exports.

Optional fixture generation is documented in
[`tests/fixtures/import/generate.py`](../tests/fixtures/import/generate.py).
It pins Treelite 4.7.2, NumPy 2.5.3, sklearn 1.9.1, and SciPy 1.18.1, and
`generate.py.lock` pins their dependencies; the manifest records input dtype
and per-fixture absolute/relative tolerances.
Float64 fixtures use 1e-14; float32 fixtures allow 1e-7 for the different
accumulation precision. Existing artifact gates remain f64 1e-14 and native
XGBoost f32 1e-5.

The [implementation validation report](treelite-loading-validation.md) records
the checks run and the measured import and execution costs. These changes do
not update the archived paper's benchmark results.

Format references: [Treelite v4 serialization](https://treelite.readthedocs.io/en/latest/serialization/v4.html),
[postprocessors](https://treelite.readthedocs.io/en/latest/knobs/postprocessor.html),
and the [pinned 4.7.2 reference evaluator](https://github.com/dmlc/treelite/blob/4.7.2/src/gtil/predict.cc).
The `latest` documentation may describe a newer producer version.
