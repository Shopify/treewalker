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
| Categories | Any `u32` category ID; membership in either child direction |
| Features / grouped rows | 1 to `i32::MAX` features; any positive maximum group width |

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

The base score is stored on the forest and added once, after the sum and the
division, each step rounded to float64: the finalization is staged, in the same
order as Treelite GTIL's, so the two can differ only through `tree_sum`, which GTIL
adds in float64 in tree order. Version 1 (the `neurips2026-submission` tag) instead spread the base
score across the leaves: its parser added `base_score / tree_count` to every leaf
value, each addition rounded, and the margin was the float64 sum of those leaves.
The two agree up to rounding, so outputs can differ from version 1's in the last
few bits. Version 1 also read no `average_tree_output` and accepted only sigmoid
models.

Prediction computes `tree_sum` exactly for every model: every leaf value is an
integer multiple of 2^-e for one scale e per model, the scaled leaves add as
integers, and the sum is rounded to float64 once. The result is the float64
nearest to the true sum and does not depend on tree order, node layout, group
width or platform. When the scaled sums fit in 126 bits they add as 128-bit
integers; otherwise, for leaf exponents too far apart, they add in a wider
fixed-point accumulator of 29-bit digits in 64-bit slots, which covers any finite
float64 leaves with any tree count (73 digits at most) and is slower. A leaf adds
three digits without carries, whatever the span, and each row carries once when it
is rounded. A zero sum is +0.0. The research
feature's `predict_full_walk` sums the same way, so production matches it bit
for bit.

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
following GTIL; in range means below 2^32, after rounding to float32 for float32
models (GTIL treats float32 inputs above 2^24, and float64 inputs above
`u32::MAX`, as out of range). Membership-right nodes are normalized by swapping their
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
to the producer/caller. A broken constant promise changes outputs; a broken
monotonic one does not: prediction sorts each varying feature's rows when it
builds their row masks, and rows that arrive in order take the sort one pass.

A `Predictor` takes row-major input with `n_features` values per row and writes
one output per row. `predict_group` predicts one group, `predict_groups` the
groups between consecutive offsets (starting at 0, ending at the row count,
strictly increasing), and `predict_fixed` consecutive groups of one width, the
last of which may be shorter. Every group has at most `max_group_width` rows;
empty input is a no-op. A call whose input length is not a multiple of
`n_features`, whose output length differs from the row count, or whose grouping
is invalid panics before writing any output. A bad model or data input gives an error; a
violated calling contract, such as these, panics, as documented here.

A predictor allocates its scratch once, when it is created, and its calls never
allocate. The scratch holds the varying predicates' row masks, left and right, at
the width of the widest group `max_group_width` allows, two difference
arrays of one entry per row, and the walk queues, stops and mask arena of one chunk of trees.
The part that grows with the widest group G is linear in G and in the varying
predicates P: about G × (52 + (2P + 1) / 8) bytes (32 of the 52 are the difference
arrays' `i128` entries, 16 × limbs for models past the 126-bit rule), for example
1.2 MiB for G = 4,096 and P = 1,000. So `max_group_width` is a memory bound:
creating a predictor panics, naming it and the bytes needed, when that scratch
exceeds what memory can address or the allocator refuses it. Trees are
processed in chunks cut so that a chunk's working set (its scratch, the predicates'
masks, and its nodes and leaf values) fits one CPU's share of the L2 cache at its
group's mask width (a tree larger than that is a chunk on its own), so the scratch
scales with a chunk rather than with the model. The cache sizes come from `sysctl`
on macOS, sysfs on Linux and CPUID on other x86-64 systems; elsewhere a 256 KiB L2
is assumed. Creating the first predictor of a process also measures the caches'
load latencies, which set the walk's prefetch distance, in about a millisecond; a
model whose nodes exceed one CPU's share of the last cache has its own nodes'
latency measured instead, by a random walk over them when its first predictor is
created (a few milliseconds, allocating nothing), which reads whatever holds them,
including caches the system does not report;
`TREEWALKER_LATENCY_NS` (nanoseconds per cache level, then memory's, comma-separated)
replaces the measurement. Every group runs whole: groups of up to 1,024 rows take
masks of 32 to 1,024 bits, a power of two, and wider ones masks of their own width
in 128-bit units. A loaded model keeps the prediction layout,
16 bytes per node plus the bins and the leaf table; the parsed nodes are kept only
with the `research` feature.

## Bounded import

The caps on untrusted input are fields of `LoadOptions`, with these defaults; a
file past one fails with `LoadError::Limit`, and trusted inputs may raise them.
They bound input, representation, and decoding work; they are not a promise about
peak process RSS. Large legitimate forests should use binary.

| Cap (`LoadOptions` field) | Default | Protects against |
|---|---:|---|
| Binary bytes consumed (`max_binary_bytes`) | 4 GiB | unbounded reads |
| Individual binary field array (`max_binary_array_bytes`) | 64 MiB | allocations from declared lengths; it also bounds the trees (`target_id`), and each tree's nodes and category entries |
| JSON input (`max_json_bytes`) | 64 MiB | the materialized JSON document |
| Config JSON (`max_config_bytes`) | 64 KiB | oversized configurations |
| Attributes (`max_attributes_bytes`) | 16 MiB | the attributes document |
| JSON nesting (`max_json_nesting`) | 64 | parser recursion, in models, attributes and configurations |
| Categorical rank tables built at load (`max_rank_table_bytes`) | 256 MiB | amplification: a feature with C categories whose splits' heavy sets need W chains takes 4 × (W + C + 3) bytes and 8 per exception (a code's rank where it differs from an unlisted category's, at most one per listed category of each heavy set), linear in the lists, or, a constant one of more than 100,000 distinct heavy sets S whose bits fit, 4 × (C + 2) × ⌈S / 32⌉ bytes of bits, the codes times the sets (and beside it, any constant one whose bits take no more than 4 × (S + C + 3) + 8 per listed category, its chains' bound); and a varying one 4 × (C + 3) more and at most 8 per exception in a chain its varying splits test, for the masks its rows flip; the cap also bounds, per feature, the heavy sets built on the way (4 bytes per listed category of each split side) and their inclusion graph (8 bytes per inclusion between two listed sets or two complements, a bit per pair of a listed set and a complement), which a hostile file can make quadratic in its splits |

Nothing else has a fixed limit. What bounds the representation:

| Resource | Bound |
|---|---|
| Features | `i32::MAX`, Treelite's `num_feature`; feature roles take memory in proportion to the declared indices |
| Trees, total nodes | at most 2^28 − 2 nodes in all (268,435,454) on every target: with the layout's dummy node, a node pool below 4 GiB, which x86-64's walks address by 32-bit byte offsets; other targets use node indices and take the same bound, so that a model loads alike everywhere |
| Nodes per tree, depth | the tree's input and the node total; every pass over a tree is iterative |
| Category IDs | any `u32`: each categorical feature's splits share a codebook of the categories they name, and rank tables cover its codes, not the IDs |
| Category-list entries | the binary `category_list` array; segments must not overlap, so decoding is linear in it |
| Varying predicates | 2^29 − 1: the layout's 30-bit mask field indexes a mask per side of each |
| Features with splits | 2^27 bin slots: 4 per feature with numeric splits, and one per chain of a feature's categorical heavy sets, or per 32 of them in bits |
| Rank tables | `max_rank_table_bytes` |
| Prefix depth | any; chosen per group of trees at load, and a prefix stops at the first leaf or varying split |

The binary reader reuses per-tree buffers and never materializes whole-model
JSON. Declared lengths are checked before allocation or multiplication. Both
readers validate tree topology iteratively before layout: unique node IDs,
valid children, reachability, cycles and shared-child rejection. Optional
statistics may be absent; present field arrays must have consistent lengths.
Of a node's statistics, both readers take only the training weight that orders its
children heavy first, the data count if present, else the hessian sum, and reject it
if it is negative or not finite; the other statistics, gain among them, are not read.

## Load time and memory

Loading takes one decoding pass and one build pass, single-threaded. The decoder
reads each tree's fields once (in place from `Forest::from_bytes`, through one
reused buffer from a reader), validates it and puts its nodes in heavy-first order,
each in the 16-byte layout node it becomes, in input order: a leaf's value, or a
split's ID in a table of the forest's distinct splits with its heavy side and
training share. When the node count is known before the first tree, the nodes are
allocated once, where the loaded model keeps them: from the bytes of
`Forest::from_bytes`, from a JSON document, and from a model file, whose trees'
node counts `Forest::load` reads first (one or two positioned reads per tree,
about 0.5 µs each). From another reader the nodes are decoded in segments and
copied into place once, which holds them twice for a moment. Trees are then ordered
by their heavy paths, and the build rewrites each tree's nodes in place, in that
order, from the distinct splits' bins, codebooks and chains, and writes the leaf table:
tree order is the order of the trees' starts and leaf values, not of their nodes.
Peak memory is therefore the loaded model plus the split table (small for binned
models: thousands of splits against millions of nodes) and a reader's buffer of at
most 1 MiB or one tree's fields. For Expedia's 1,000-tree, depth-16 LightGBM model
(7.7 million nodes, a 636 MB file), `Forest::load` takes 0.27 s with a heap peak of
191 MB, against a loaded model of 186 MB. JSON keeps a
document in memory: simd-json's tape over the input, whose strings it borrows,
about five times the file at its peak; prefer binary for large models. Predictor
creation reads per-tree scratch needs computed at load, so it costs no pass over
the nodes.

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
