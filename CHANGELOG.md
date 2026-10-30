# Changelog

## 3.1.0

In 3.1, default x86-64 builds run the kernel at the CPU's instruction-set level. The
API is unchanged, and outputs are the same at every level. Outputs are bit-identical
to 3.0 except for models with a sigmoid link, which now computes its own `e^-m` rather
than calling the platform's libm.

### Changes

- On x86-64, a build that enables AVX2 and FMA (`-C target-cpu=x86-64-v3` or a CPU that
  has them) compiles the kernel for its own features, as 3.0 did. Other x86-64 builds,
  the default `x86-64` among them, also compile it for x86-64-v3 and for x86-64-v4 with
  VPOPCNTDQ, BITALG and BF16 (Sapphire Rapids and Zen 4 on), and each predictor runs
  the highest the CPU has; `TREEWALKER_ISA=base`, `v3` or `v4` asks for a lower one.
  Outputs are the same at every level. The kernel uses no BF16 instruction: requiring
  it keeps Ice Lake, Tiger Lake and Rocket Lake at x86-64-v3, which ran faster than
  x86-64-v4 on Ice Lake.
- The sigmoid computes `1 / (1 + e^-m)` with the `e^t` of Arm's optimized routines,
  which glibc's `exp` is, in a loop that vectorizes, and gives the same bits on every
  platform; 3.0's last bits came from the platform's libm. Against 3.0 with Apple's
  libm, 99.92% of outputs are unchanged and the rest move by at most 4 ulp; both are
  within 2 ulp of the correctly rounded sigmoid for `|m| <= 40`.
- A model of more than 2^28 − 2 nodes in all (268,435,454) fails to load with
  `LoadError::Limit` on every target, where 3.0 loaded up to 2^32 − 2. On x86-64 the
  walks now step from node to node by byte offset into the node pool, so that a
  node's address takes no shift, and the offsets are 32-bit: a pool below 4 GiB.
  AArch64 and the other targets keep node indices and take the same limit, so that a
  model loads, or fails to, alike everywhere. Only a model loaded with `LoadOptions`
  caps raised past their defaults can have that many nodes: a binary model takes at
  least 40 bytes per node.
- Debug builds check the predictor's contract that a group's constant features are
  equal in every row, on the features that constant splits read, and panic naming
  the feature and the row; a missing value equals a missing one. Release builds do
  not check it, as before.

### Fixes

- A `WalkerConfig` takes memory in proportion to the features it declares. 3.0 kept
  each role as a bitset up to its largest index, so a 120-byte configuration that
  declared feature 2,147,483,646 varying took about 768 MiB while it was built and
  256 MiB after.
- `WalkerConfig::from_json` and `from_file`, and their `_with` forms, return
  `LoadError::Limit` for a configuration nested deeper than
  `LoadOptions::max_json_nesting`, as they do for one past `max_config_bytes` and as
  the model loaders do for nesting. 3.0 returned `MalformedConfig`, with the limit's
  message inside its own.
- A model whose category lists take more than `u32::MAX` entries in all, or whose
  categorical features' codebooks do, fails to load with `LoadError::Limit`, as one
  past the other 32-bit index tables does. 3.0 converted the offsets past them to 32
  bits without a check, and so read the wrong lists. Only a model loaded with
  `LoadOptions` caps raised far past their defaults can have that many: the lists
  take at least 4 bytes of a binary model per entry.
- The JSON reader validates a node's statistics as the binary reader does: only the
  one it takes as the training weight, the data count if present, else the hessian
  sum, which must be finite and not negative, with the binary reader's error; `gain`
  is not read. 3.0 rejected a JSON model with any invalid `gain`, `sum_hess` or
  `data_count`, which the same model in binary loaded with.
- When the latency probe times L1 at 0 ns, as a clock too coarse to time it can, or
  `TREEWALKER_LATENCY_NS` gives latencies whose ratios overflow, the prefetch rule
  takes the assumed latency ratios. In 3.0 the infinite ratio overflowed the
  restart prefetch distance in `Forest::predictor`: a panic in builds with overflow
  checks, and no restart prefetch in others.
- A split whose children's training weights are finite but sum past `f64::MAX`
  gives its heavy child the share of their weight it has, which the research API's
  `Node::heavy` reports, rather than 0. The share guides layout choices only, so
  outputs are unchanged.
- Debug builds load a binary checkpoint whose node count, guessed from its trees'
  headers before decoding, is wrong, as release builds do, by copying the nodes.
  Only a crafted checkpoint can make the guess wrong and still decode; 3.0's debug
  builds panicked on it.

### Known gaps

- A model whose categorical splits take many distinct heavy sets that seldom nest,
  some 100,000 or more on one feature, can fail to load at the default limits, and
  loads slowly with them raised. 3.0 has the gap too: the chain cover
  (`src/load/layout/chains.rs`) and `max_rank_table_bytes` shipped in it. The cover
  matches over the sets' inclusion graph, which grows faster than the sets, and the
  rank tables take a slot per chain. The `categorical-scale` cell
  (`docs/categorical-scale.md`) reproduces it on public data: on Genoa
  (c3d-standard-90) its model, 600 LightGBM trees of 4,096 leaves on Expedia
  searches with 265,406 distinct sets on `srch_destination_id`, fails with
  `LoadError::Limit` on the inclusion graph in 1.3 s, and with the cap raised its
  190,335 chains load in 4.7 s at a peak of 1.5 GB.

## 3.0.0

3.0 replaces the prediction kernel. The new one walks several trees at once on
integer bins, resolves splits on varying features in separate passes, and sums every
model's leaves exactly. The serving API is unchanged except for the removal of
`Forest::exact_sums`, and the research API changes. Outputs are bit-identical to 2.0 for every model
that 2.0 summed exactly, and change only for models that 2.0 added in `f64` in tree
order.

### Migrating from 2.x

The loaders, `WalkerConfig`'s builder and readers, the three `Predictor` calls and
their panics, `Forest`'s `Send + Sync` and `Predictor`'s `Send` are unchanged.
`LoadOptions` loses its four optimization fields (`disable_tree_ordering`,
`disable_bitset_intern`, `prefix_depth`, `disable_predicate_dedup`): the library
makes those choices, and the research module's `LoadAblation` turns them off to
measure them. It gains the input caps instead, so a 2.x struct literal that set an
optimization field no longer compiles, and one that set none still does with
`..LoadOptions::default()`. `Forest::exact_sums()` is removed: every model sums
exactly, so it could only return true, and a 2.x call to it no longer compiles;
code that branched on it keeps its exact branch. Everything else that changed is
behind the `research` feature:

| 2.x | 3.0 |
|---|---|
| `forest.exact_sums()`, false for leaf exponents too far apart | removed: every model sums exactly |
| `Ablation::disable_exact_sums` | removed: every model sums exactly |
| `Ablation::disable_varying_precompute` | kept: each varying event scans its rows for its predicate's mask |
| `Ablation::disable_monotonic` | kept: those scans do not stop at the first row past a monotonic threshold |
| `Ablation::disable_unsplit` | kept: the sides of varying events that have no rows are passed on as items |
| `Ablation::disable_predicate_sweep` | kept |
| — | new `Ablation` fields: `lanes` (`Some(1..=12)`, for every walk), `disable_g1_constant`, `disable_width_dispatch`, `disable_restart_prefetch` |
| variant names built by the caller | `Ablation::tokens()`, `id()` (`all-on`, or the tokens joined by `+`), `from_id(id)` and `with(token)`; a token is a flag's name or `lanes=N` |
| `LoadOptions::{disable_tree_ordering, disable_bitset_intern, prefix_depth, disable_predicate_dedup}` | `LoadAblation`'s fields of the same names, with `prefix_depth: Option<usize>` (`None` for the rule's groups), passed to `Forest::load_ablated`, `from_bytes_ablated` or `from_reader_ablated` |
| `WorkCounters`, `COUNTERS_VERSION` 2: `constant_steps`, `varying_splits`, `unsplit_skips`, `recursive_calls`, `leaf_hits`, `leaf_adds`, `partition_row_evals`, `scan_row_evals`, `scan_compares`, `scan_missing_checks` | `COUNTERS_VERSION` 6: `groups`, `single_rows`, `bins_computed`, `scan_rows`, `scan_evals`, `prefix_steps`, `walk_steps`, `lane_slots`, `passes`, `varying_events`, `splits`, `empty_sides`, `leaf_events`, `runs_written`, `arena_slots`; `WorkCounters::fields()` lists every counter by name |
| `precompute_row_evals`, `precompute_sort_compares`, `precompute_sweep_compares`, `precompute_mask_writes` | the same meaning |
| — | `ResearchPredictor::take_phases()` returns the counted calls' time per phase as `Phases`: `bins`, `precompute`, `prefix`, `walk`, `resolve`, `leaf_pass`, `finish` |
| — | `Forest::layout_stats()` returns `LayoutStats`: the rules' choices on this machine (lanes per walk, chunk budget, prefetch distances, prefix groups) with their inputs (registers, cache sizes, probed latency, expected walk steps), the share of internal nodes on varying features, the bytes of the node pool, leaf table, categorical rank tables and varying categorical features' flip lists, the categorical chain slots and the heavy sets they cover, and the wide accumulator's slots |
| `predict_full_walk`, which added in `f64` in tree order | sums exactly, so production matches it bit for bit |
| `Node::varying_type()`, `Node::is_walkable()` | removed; `Node::is_constant()` remains, and flag bit 4 marks a varying split |
| `Node`: 16 bytes, `skip: i16` (-1 for a leaf), `varying_pred_id: u16`, `feature: u16`, `cat_n_words: u8` | 24 bytes, `skip: u32` (`u32::MAX` for a leaf), `varying_pred_id: u32` (`u32::MAX` for none), `feature: u32`, `heavy: u16` (the heavy child's share of the training weight); a categorical split with a category of 32 or more holds the offset of its category list instead of a bitset, and `cat_n_words` is gone |
| `Tree::bitset_start`, `Forest::bitset_bytes()` over the parsed bitsets | `Tree::category_start`; `bitset_bytes()` counts the parsed category lists |

`WorkCounters`, `Phases` and `LayoutStats` are `#[non_exhaustive]`.

### Limits

Representation limits are now what the types and memory allow, and the caps that
protect against hostile files are `LoadOptions` fields with the old values as
defaults (`max_binary_bytes` 4 GiB, `max_binary_array_bytes` 64 MiB,
`max_json_bytes` 64 MiB, `max_attributes_bytes` 16 MiB, `max_json_nesting` 64,
`max_config_bytes` 64 KiB, and `max_rank_table_bytes` 256 MiB for the
categorical rank tables built at load). `WalkerConfig::from_json_with` and
`from_file_with` read configurations within a `LoadOptions`' caps.

| Limit | 2.x | 3.0 |
|---|---:|---|
| Features | 64 | `i32::MAX`, Treelite's `num_feature`; feature roles are bitsets |
| Trees | 1,000,000 | the input; fewer than 2^32 − 1 nodes in all |
| Nodes in all | 32,000,000 | 2^32 − 2, the 32-bit node indices |
| Nodes per tree | 32,767 | the input and the total |
| Depth | 256 | the input; every pass over a tree is iterative |
| Category IDs | 8,159 | any `u32`: rank tables cover each feature's codebook of the categories its splits name |
| Category-list entries per binary tree | 16,777,216 | the `category_list` array; overlapping segments are rejected |
| Pooled bitset bytes | 256 MiB | none: no bitsets; the rank tables take `max_rank_table_bytes`, 256 MiB by default |
| Varying predicates | 65,535 | 2^29 − 1 |
| Prefix depth | 256 | any |

Every split is now one test, `rank <= k` on a slot: a categorical feature's splits
no longer read bitsets, inline or pooled, so the number and IDs of its categories
do not change how its splits are walked. A binary model whose category segments
overlap, which Treelite never writes, is rejected.

Before release, the 3.0 development branch named the cap `max_category_pool_bytes`,
for the pooled categorical bitsets; with the bitsets gone it is
`max_rank_table_bytes`, with the same default and role (code that set the old field
sets the new one), and `LayoutStats::category_bytes` is `rank_table_bytes`, beside
the new `chain_slots` and `chain_sets`. The bytes it bounds differ: a feature with
C categories and W chains takes 4 × (W + C + 3) bytes and 8 per exception (at most
one per listed category of each distinct heavy set), where its bitsets took a bit
per category for each pooled split, so a model near the old cap may land on
either side of the new one.

### The kernel

- At load, every split becomes `rank[slot] <= k`, and the nodes a 16-byte
  pre-oriented layout with the heavy child next. A slot is a chain of nested sets
  over a feature's values, a row's rank in it the first set that holds its value:
  a numeric feature's thresholds give four slots (missing sent left or right,
  plain or inverted), and a categorical feature's distinct heavy sets (missing and
  unknown categories being ordinary values) a minimum chain cover, by Hopcroft–Karp
  on their strict-inclusion graph, which an index of the sets by the categories
  they list builds. A category's rank differs from an unlisted one's only in the
  chains of sets that list it, so the rank tables hold an unlisted category's rank
  in each chain and each code's exceptions, and grow with the model's category
  lists rather than its categories times its chains; a fill restores the previous
  code's exceptions and writes the new one's. A group's constant features are
  binned once from its first row.
- The walk steps several trees at a time, each lane's state in registers, without
  data-dependent branches on constant splits. A leaf or a varying
  split ends a lane's item, and each step writes the lane's node at its item's
  position, so the walk keeps no event queue.
- A piece's varying predicates' masks come from a sweep of each feature's rows,
  sorted by value with 32-bit row indices (2.x: 16-bit), against its numeric
  thresholds in order; rows already in order take the sort one pass. Categorical
  predicates are rank tests on the feature's chains: each row's code is computed
  once per row and feature, every row starts in the mask of the first predicate at
  or past an unlisted category's rank, each of its code's exceptions moves it to its
  own (the masks each code flips are listed at load), and a running union turns
  those masks into every predicate's, instead of evaluating every predicate on each
  row's value.
- Resolve passes read the stops in item order, split each varying stop's rows with
  the precomputed mask of its heavy side (each predicate's mask is written with its
  complement), and a leaf pass adds each leaf to its runs of
  rows in two alternating difference arrays.
- A single row runs through the same walk, its varying splits stepped as
  constants, so it needs no masks and stops only at leaves. Its pass walks the
  trees' roots in place, and its leaf values are summed in a register from the walk's
  words. Leaf entries of a group's first pass carry all rows and are summed in
  registers too. Groups run at the
  narrowest mask width that holds them: 32 and 64 bits by value, 128 to 1,024 bits in an arena. 2.x
  chose one width per predictor from `max_group_width`. Groups wider than 1,024 rows
  run whole, on masks of a runtime width.
- Trees run in chunks whose working set fits one CPU's share of L2 at the group's
  mask width, so a predictor's scratch scales with a chunk, not the model. It is
  still allocated once per predictor, and calls never allocate.
- The kernel's constants come from the target, the machine and the forest, not
  from tuning (`src/rules/`): the lanes per walk, among the counts whose cursor
  advance is one instruction, are on AArch64 the most whose state fits the registers
  beside the loop's invariants and a step's temporaries, since a spilled lane
  lengthens its chain (9 for groups and 8 for single rows), and on x86-64 the fewest
  that fill the issue width over a step's chain with its node in L2, spilled values
  counted (8 and 8); the chunk budget is L2's share per
  CPU from `sysctl`, sysfs or CPUID; the restart prefetch distance covers the
  latency of the cache level holding the nodes, which a pointer chase measures once
  per process, or of the node pool itself, which a random walk over it measures once
  per model when it passes the last cache; and prefix groups and their depths maximize the walk steps saved net
  of the prefix work, from each split's training weights. Prefix groups are runs of
  consecutive trees, whose starts move with a vector add.
- Production keeps only the prediction layout; the parsed nodes stay with the
  `research` feature, for introspection and the full walk.
- NEON on AArch64, SSE2 on x86-64 (AVX2 when the build enables it), and plain words on
  other targets.

### Exact sums

Every model's leaves are added as exact fixed-point integers and rounded once, so a
row's `tree_sum` is the `f64` nearest to the true sum whatever the tree order,
layout, lanes, group width or platform. Leaves whose scaled sums fit in 126 bits add
as `i128`, as in 2.0. Leaves whose exponents span too wide a range, which 2.0 added
in `f64` in tree order, now add in a wide fixed-point accumulator of 29-bit digits
in 64-bit slots, which holds any finite `f64` leaves with any tree count. A leaf
adds three digits and carries nothing, so its cost does not depend on the span;
each row carries once when it is rounded. Those models' outputs change by their old
rounding error, and they predict 10–17% slower than ordinary models in groups of 16
rows and 11–16% slower for single rows, for spans of 137 to 1,968 bits (support, 500
trees, measured on the M4, 2026-10-10). A zero sum is +0.0. The base score, averaging and
the link function round as before.

### Measured gains

On an Apple M4 Pro (2026-10-08, rustc 1.97.1, `-C target-cpu=native`, fat LTO), total
time per group, median over rounds, against 2.x's kernel with `prefix_depth = 0`. 2.x's
default `prefix_depth = 2` is 0.4–2.5% faster than that on these cells.

| Cell | 2.x | 3.0 | Change |
|---|---:|---:|---:|
| SUPPORT LightGBM, 500 trees, depth 8, horizon 16 | 31,357 ns | 14,760 ns | −52.9% |
| Expedia LightGBM, 500 trees, depth 8 | 89,059 ns | 40,085 ns | −55.0% |
| Credit what-if, k = 4, G = 64 | 33,104 ns | 20,112 ns | −39.2% |
| FLCHAIN, 1,008-row groups | 149,397 ns | 108,492 ns | −27.4% |
| SUPPORT, single independent rows | 10,442 ns | 4,959 ns | −52.5% |
| Expedia, single independent rows | 10,223 ns | 4,586 ns | −55.1% |

Against 2.x's production defaults, the prototype of this kernel measured (same
machine, 2026-10-07 and 2026-10-08):
- Groups: −49.3% on SUPPORT, −38.8% on credit k4_G64, −53.3% on Expedia and −47.1%
  on FLCHAIN's 1,008-row groups.
- Single rows, 20,000 independent rows (a fixed-seed shuffle): −30.5% to −36.3% on
  the panels, −34.5% on credit and −43.9% to −46.2% on Expedia.
- Single rows in file order, where consecutive rows come from one subject and 2.x's
  branch predictor learns their paths: −3.8% to −9.4% on the panels, +6.9% on credit
  and −41.7% to −43.3% on Expedia.
- A predictor configured for groups and fed single rows costs about 2% over one with
  no varying features, against 21–43% in 2.x.

On Google Cloud's benchmark machines (Ubuntu 26.04, SMT off, rustc 1.97.1,
`-C target-cpu=native`), serving time against 2.x's production defaults on 90 cells:
the four datasets' models (SUPPORT, FLCHAIN, credit and Expedia) as single rows,
panels, sessions, what-ifs and what-if grids, 6 processes per side. The geometric mean
of 3.0's time over 2.x's was 0.476 on Emerald Rapids (c4-standard-32, 2026-10-16) and
0.564 on Axion (Neoverse V2, c4a-highmem-16, 2026-10-14); with 64-byte function
alignment, 0.479 and 0.562. No cell was slower at either alignment. By workload, over
both alignments:

| Workload | Cells | Emerald Rapids | Axion |
|---|---:|---:|---:|
| Single rows | 12 | 0.393 | 0.444 |
| Panels | 60 | 0.461 | 0.556 |
| Sessions | 4 | 0.538 | 0.665 |
| What-ifs | 6 | 0.560 | 0.666 |
| What-if grids | 8 | 0.689 | 0.715 |
| Per cell | | 0.327–0.981 | 0.386–0.960 |

Both machines ran release candidates. On x86-64 the candidate loaded each lane's next
item a step early, and its varying walks over 8-byte items held their nodes in 64 bits.
3.0.0 drops both: at 8 lanes the early load costs more instructions than it saves in
latency in the larger varying walks, and without it the 64-bit nodes lengthen the loop
on Emerald Rapids. It keeps the candidate's 64-bit cursors; 3.0.0's x86-64 walk was not
timed itself. On AArch64 the candidate differs from 3.0.0 only in a few stores per call
that cleared unused work counters.

## 2.0.0

2.0 predicts through a reusable `Predictor`, loads with `Result` everywhere, and
moves ablations, work counters and introspection behind a `research` feature. It
ships with the 1.x branch's validated Treelite loading and exact leaf sums, so it
also breaks the 1.x names those changes had replaced. Predictions change by
rounding; see "Predictions" for what is exact and what was measured.

### Migrating from 1.x

| 1.x | 2.0 |
|---|---|
| `Forest::load(model, config)`, which panicked | `Forest::load(model, config)?` |
| `Forest::load_with_config(model, config, &ParseConfig)` | `Forest::load_with(model, config, &LoadOptions)?` |
| `ParseConfig` | `LoadOptions`, with the same fields |
| `WalkerConfig::from_file(path)`, which panicked | `WalkerConfig::from_file(path)?`, or `WalkerConfig::from_json(bytes)?` |
| a `WalkerConfig` built from its public fields | `WalkerConfig::builder(n_features).max_group_width(w).varying(..).increasing(..).decreasing(..).build()?` |
| `config.n_features`, `config.max_group_width` | `config.n_features()`, `config.max_group_width()` |
| `config.varying_mask`, `is_mono_inc(f)`, `is_mono_dec(f)` | `is_varying(f)`, `is_increasing(f)`, `is_decreasing(f)` |
| `forest.config` | `forest.config()`, read-only |
| `forest.predict(&data, &mut out, start, end)` on a `&mut Forest` | `let mut p = forest.predictor();` once, then `p.predict_group(&data[start * nf..end * nf], &mut out[start..end])` |
| a loop of `predict` calls over groups | `p.predict_groups(&data, &offsets, &mut out)` or `p.predict_fixed(&data, width, &mut out)` |
| `predict_full(&data, &mut out, start, end)` | `forest.predict_full_walk(rows, out)`, with `research` |
| `config.ablation = AblationMode { .. }; predict_with_stats(..)` | `forest.research_predictor(Ablation { .. }).predict_group_counted(rows, out)`, with `research` |
| `AblationMode`, `PredictStats` | `research::Ablation`, `research::WorkCounters` |
| `nodes()`, `trees()`, `tree_nodes()`, `bitset_bytes()`, `threshold_type()` | the same, with `research` |
| `precompute_sweep`, `precompute_bruteforce` (`test-helpers`) | `forest.predicate_masks(rows, sweep)`, with `research` |
| feature `test-helpers` | feature `research` |
| `treewalker_gbdt::{config, forest, mask, parser, predict}::*` | the root exports, or `treewalker_gbdt::research::*` |
| `parser::parse_model` | `Forest::load`; the model keeps its own base score and averaging |
| `predict::MAX_GROUP_WIDTH` (128) | none: `max_group_width` takes any positive width |
| `mask::RowMask` for `u32`, `u64`, `u128` | private; masks are `u16` to `u64` and multi-word above 64 rows |
| `cat_test(node, category, default_left)` | private |

Names introduced on the way to 2.0 and never released also changed:
`Forest::try_load` is `load`, `try_load_with_config` is `load_with`,
`WalkerConfig::try_new` is the builder, and `try_from_file` and `try_from_json`
are `from_file` and `from_json`.

### Model and predictor

- `Forest` is a cheap handle on the immutable model: `Clone + Send + Sync`.
  Clone it to share a model between threads.
- `Forest::predictor()` creates a `Predictor`, which is `Send` and owns the
  scratch space for the widest group. Create one per worker and reuse it: calls
  never allocate.
- `predict_group(rows, out)` predicts one group, `predict_groups(data, offsets,
  out)` the groups between consecutive offsets, and `predict_fixed(data, width,
  out)` consecutive groups of one width, the last of which may be shorter.
- Every call checks its input before writing any output, and panics if a length,
  an offset or a group width is invalid. Empty input is a no-op. 1.x also
  panicked on bad ranges, and on a configuration changed after loading, which
  2.0 makes impossible.
- A group can have any number of rows up to the configured `max_group_width`.
  1.x capped groups at 128 rows.

### Configuration and loading

- `WalkerConfig::builder` requires `varying(..)` or `all_varying()`. 1.x treated
  every feature not listed as varying as constant, so a caller who forgot the list
  got wrong predictions for an ordinary matrix.
- `increasing(..)` and `decreasing(..)` imply varying, and a feature declared both
  is rejected. `walker_config.json` is validated by the same builder, so a
  monotonic feature missing from `varying_features` is now accepted as varying.
- `LoadError` is returned by every loader, derives `thiserror::Error`, and is
  `#[non_exhaustive]`, as is `ModelFormat`.

### Research feature

`research` replaces `test-helpers`. It adds:

- `forest.research_predictor(Ablation)`. Its flags are fixed when it is created.
  `predict_group` and `predict_groups` run the research timed build of the kernel
  and `predict_group_counted` the counted build; all use the flags. In 1.x,
  `predict()` ignored `config.ablation`, so ablations timed through it measured
  the production path.
- `predict_group_stages`, which returns each row's `tree_sum`, `raw_margin` and
  `output`.
- A new flag, `disable_exact_sums`: leaves add in `f64` in tree order.
- `WorkCounters` counting actual work, versioned by `COUNTERS_VERSION` (2).
  `leaf_adds` and the `scan_*` and `precompute_*` counters are new.
  `precompute_row_evals` now counts the brute force's predicate × row evaluations;
  the other counters keep their 1.x meaning.
- `predict_full_walk`, the introspection calls and `predicate_masks`.

Production prediction is compiled without ablation flags or counters, with or
without the feature.

### Predictions

A prediction has three stages, and only the first can be exact:

- `tree_sum`, the sum of the row's leaf values. When `Forest::exact_sums()` is
  true, the leaves are added as exact fixed-point integers and rounded once, so
  `tree_sum` is the `f64` nearest to the true sum, whatever the tree order,
  layout or group width. Otherwise, for a model with a non-finite leaf or leaf
  exponents too far apart, the leaves add in `f64` in tree order. The 12 models
  measured below all have exact sums.
- `raw_margin`, `tree_sum / divisor + base_score`, and the output, after the
  identity or the sigmoid, are ordinary `f64` operations, each rounded.

1.x added the leaves in `f64` in tree order and spread the base score across the
leaves. A sequential sum can be arbitrarily far from the exact one: leaves of
1e16, 1 and -1e16 sum to 1 exactly and to 0 in order, and after a sigmoid that
is 0.731 against 0.5. On the benchmark artifacts, 2.0's outputs differ from the
same model's sequential `f64` sums (`disable_exact_sums`, measured 2026-10-04)
by at most 2.2e-15 on the four survival cells at horizons 16 and 128 (FLCHAIN
and SUPPORT, LightGBM), 7.2e-16 on Expedia, and 1.1e-14 on FLCHAIN at horizon
1,024. The XGBoost models gave bit-identical outputs in all of these cells.

1.x also applied a sigmoid to every model and ignored averaging. 2.0 reads the
model's output metadata: identity or sigmoid output with its alpha, and sum or
average aggregation.

### Loading

The Treelite importer validates a scalar subset, returns errors instead of
panicking, and accepts binary checkpoints from Treelite 4.0 to 4.7 and Treelite
JSON. See `docs/treelite-loading.md`.
