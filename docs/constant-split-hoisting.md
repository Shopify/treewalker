# Moving group-constant predicates above varying splits

The first prototype is implemented in `src/parser/hoist.rs`, independently of
the Treelite import improvements. The identities below preserve the prediction
function. An [exploratory benchmark sweep](hoisting-benchmarks.md) finds real
rewrite opportunities but no general latency improvement on the available
trained models. The pass remains disabled by default.

## Using the prototype

```rust
let forest = Forest::load_with_config(
    model_path,
    config_path,
    &ParseConfig { hoist_constants: true, ..Default::default() },
);
println!("{:?}", forest.hoist_stats());
```

It implements paired-child hoists and exact identical-subtree collapse, with
no growth and a limit of 16 bottom-up passes per tree. It supports numeric and
categorical predicates, including non-interned pooled categories. Weights for
swapped/merged paths are combined from their original subtrees; no hypothetical
joint counts are introduced. The normal layout, deduplication and prefix passes
run afterward. Unused nodes disappear during layout; unused bitset bytes may
remain in the pool.

The `hoist_bench` binary accepts model, configuration and input-data paths:

```bash
cargo run --release --manifest-path benchmarks/Cargo.toml --bin hoist_bench -- \
  tests/fixtures/hoisting/numeric_f64.bin \
  tests/fixtures/hoisting/walker_config.json \
  tests/fixtures/hoisting/data.bin
```

Replace these synthetic fixtures with any prepared benchmark cell, and use
`--group-offsets FILE` for variable groups. `--no-tree-ordering --prefix-depth 0`
isolates the transform; the default measures its interaction with the current
optimizations. `--min-blocks`, `--max-blocks` and `--warmup` control measurement.
The command fails before timing if predictions are nonfinite or differ from
the original full walk by more than 1e-14. The comparison is within TreeWalker's
f64 accumulation, including for f32-threshold models; it is not a native-XGBoost
parity tolerance. Independent GTIL references are exercised by the Rust tests.

JSON reports include work counters and raw model-pool sizes, which do not
include prediction workspace or all metadata. Load times are single samples
and susceptible to cache/order effects. Use a separate output directory for
experimental reports; do not overwrite the released paper CSVs.

For a serial sweep over prepared survival and ranking cells:

```bash
cargo build --release --manifest-path benchmarks/Cargo.toml --bin hoist_bench
python3 paper/experiments/scripts/hoist_sweep.py ARTIFACTS target/hoist-run
```

The output directory must be new. The runner retains raw JSON and stderr,
an incremental `summary.csv`, and a manifest with commands, platform and binary
hash. It prefers binary models, falls back to JSON, discovers group-offset
files, and translates legacy config names into local copies. The source
artifacts remain read-only. Both the default and the ordering/prefix ablation
run unless `--mode default` or `--mode isolated` selects one. Use repeatable
`--dataset`, `--framework` and `--cell` filters, for example
`--cell '*/nt500_md8_h16' --repeats 3`. The minimum is 11 timing blocks per run;
repetitions are separate process invocations. A failed correctness gate is
recorded in the manifest and stderr and makes the sweep exit unsuccessfully.

One-sided and deeper cofactor search remain future experiments. The next
decision should depend on opportunity counts and latency from real models.

## Recommendation

Prototype bounded constant-predicate hoisting during model lowering. Start
with identical constant predicates at the two children of a varying split;
then consider one-sided hoists only under explicit growth and work budgets.
Minimize measured grouped prediction work, with varying-root count as a
diagnostic. A lower root count by itself is not a useful acceptance criterion.

The grouping classification supplied by the application tells the transform
which predicates are constant within a group. Values of those predicates need
not be known at load time. Runtime evaluation selects the applicable residual
subtree once the group's actual feature values are available.

## Exact local rewrites

Write `P(X, Y)` for a node that selects X when predicate P is true, and Y
otherwise. C is constant within a group; V varies between rows. A, B, D and E
are arbitrary subtrees.

### Matched predicates in both branches

```text
V(C(A, B), C(D, E))  =  C(V(A, D), V(B, E))
```

Both trees have three internal nodes and the same four subtree occurrences.
When V splits the group's rows into two nonempty sets, the original tree
evaluates C twice. The transformed tree evaluates C once on the full group,
then evaluates just one of the two V nodes. If V does not split the group,
both forms evaluate C once. The transform does not inherently reduce the
number of active varying partitions.

It becomes more valuable when equal subtrees collapse:

```text
V(C(A, B), C(A, D))  =  C(A, V(B, D))
```

For groups where C is true, the transformed tree no longer visits V at all.
It also has fewer nodes. More generally, reduce `P(X, X)` to X using exact
structural equality, with identity including complete predicate semantics and
leaf value bits. Do not merge approximately equal leaf values.

### Constant predicate in only one branch

```text
V(C(A, B), D)  =  C(V(A, D), V(B, D))
```

This is exact, but duplicates D in an ordinary tree and adds one internal
node. If D is large, model growth can overwhelm any benefit. It also executes
C for groups whose original V outcome would have selected only D. A root
changing from varying to constant therefore does not guarantee less work.

Hoisting a deeper C can be expressed as:

```text
F = C(restrict(F, C=true), restrict(F, C=false))
```

Restriction replaces every occurrence of the exact predicate C by its chosen
branch and recursively simplifies equal branches. It is safe even when other
predicates are correlated with C; exploiting implications between different
thresholds is an additional optimization that needs its own proof, including
missing-value behavior. A bounded search should consider C predicates already
present near the varying root and stop before constructing an over-budget
candidate.

These are decision-diagram/cofactor identities. Variable ordering strongly
affects representation size, and forcing an order can cause exponential
growth. See [Bryant, Graph-Based Algorithms for Boolean Function Manipulation](https://www.cs.cmu.edu/~bryant/pubdir/ieeetc86.pdf).
Using shared subgraphs can reduce duplication, but does not remove the
worst-case ordering problem.

## Why TreeWalker needs a specific cost model

The current engine already shares work:

- `partial_eval` visits a constant node once per reached subtree and skips
  branches with empty row masks. It is not repeating every constant split
  independently for every row.
- The default path precomputes every unique varying predicate before walking
  the forest. Moving a V node downward usually retains that predicate in the
  precompute set. Claim a precompute saving only when all uses disappear or a
  separate change makes precompute conditional.
- `build_prefix_groups` can share initial constant heavy-path predicates
  between trees. Exposing useful matching prefixes is a plausible additional
  benefit, but one new constant root alone may not satisfy the configured
  prefix depth (currently two).
- `auto_order_trees` changes tree order and heavy-path DFS changes storage
  layout. Neither performs the logical predicate reordering described here.

A cost estimate should consider constant visits, active varying partitions,
recursive calls, leaf visits, shared-prefix hits, unique varying predicates,
and model size/cache effects. Estimate branch activity from representative
groups when available, including whether V actually partitions a group.
Training sample counts alone do not describe this grouped behavior.

TreeWalker encodes a contiguous tree with heavy-child fall-through and i16
local offsets. A decision DAG is not a drop-in replacement: sharing D in the
one-sided rule changes representation assumptions. Keep the first prototype
as ordinary trees with a hard size bound, followed by the existing layout and
predicate-deduplication passes.

## Proposed bounded experiment

1. Inventory each model: varying roots, first-varying depth, identical constant
   child predicates, opportunities for equal-subtree collapse, and potential
   new shared prefixes. Report trees with no constant predicate separately;
   there is no useful constant root to hoist into those trees.
2. Add an experimental opt-in pass before heavy-path layout. Apply exact
   paired-child hoists and equal-subtree reductions, with a bounded rewrite
   count and deterministic tie-breaking. Rebuild flags, indices, predicate
   IDs and prefix groups afterward.
3. Separately evaluate one-sided/deeper hoists with small tunable depth and
   node-growth caps. For an initial experiment, depth 2–3 and a 10–25% growth
   ceiling are candidate settings, not established defaults. Respect i16
   per-tree capacity and every other importer representation limit. Abort a
   candidate early rather than building a huge intermediate tree and then
   rejecting it.
4. Track affected training-count heuristics carefully: old marginal counts
   do not become the new joint counts after swapping predicates. Recompute
   layout weights using representative data or use an explicitly neutral
   fallback for synthesized nodes; do not describe guessed weights as exact.
5. Compare original versus rewritten per-tree outputs, keeping original tree
   order for that check. Include numerical thresholds/equality, f32 rounding,
   NaNs, both missing directions, categorical predicates, and identical-leaf
   reductions. Then compare ensemble outputs under the existing accumulation
   tolerances, with layout/prefix optimizations enabled and disabled.
6. Benchmark actual group widths and several group compositions: V splits
   both ways, V is effectively constant within a group, and C chooses both
   sides. Record latency, load time, bytes/nodes, precompute cost, and existing
   `PredictStats` counters. Compare with the current optimized baseline.

The matched-predicate transform uses the same total routing predicates on the
same rows. Preserve threshold precision/operator, category membership side,
and missing direction exactly. Do not rewrite `not(x < t)` as `x >= t` without
accounting for NaNs. In the compact representation, heavy-child orientation
is a layout choice and is not part of the logical predicate to match.

Keep the pass disabled by default unless representative results demonstrate a
benefit. In particular, count reductions with no latency gain do not justify
shipping it enabled.

## Other representations

A tree can also be expanded into a sum of leaf rules: each leaf value is
multiplied by the conjunction of decisions on its root-to-leaf path. Testing
the constant conditions first is exact and exposes constant guards even when
local hoists would duplicate large subtrees. But it can turn one tree into
many rules/trees and repeat shared tests. Floating-point accumulation order
also needs attention. This is a useful comparison or a basis for a factored
rule DAG, not the first change to this tree engine.

Do not insert irrelevant constant predicates merely to change the root
statistic. Trees whose outputs depend only on varying features should remain
so. The objective is to avoid repeated work and expose useful specialization,
while preserving the model's predictions.
