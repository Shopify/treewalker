# Changelog

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
