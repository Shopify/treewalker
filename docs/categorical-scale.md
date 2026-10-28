# The categorical scale cell

`categorical-scale` is one LightGBM cell on Expedia's public competition data,
not a redistributed dataset. Its model is
`expedia-cat/nt600_md32_nl4096`: 600 trees, at most 4,096 leaves and depth 32.
`_nl4096` records the leaf limit independently of the depth; the existing
`nt<T>_md<L>` names, training keys and suites are unchanged.

## Recipe

The source is the same fingerprint-checked `experiments/data/expedia.parquet`
as the paper's Expedia cells. Searches with 2-128 candidates survive only if
each `datasets.EXPEDIA_SESSION` column has at most one observed value within
the search (nulls are ignored, as in the numeric split). All eligible searches
are split 80/20 by sorted `srch_id` with the paper's name-hashed dataset seed,
but without its 50,000-search cap.
Candidates retain source-file order within each search.

Features, in order, are the seven categorical columns `site_id`,
`visitor_location_country_id`, `srch_destination_id`, `srch_length_of_stay`,
`srch_adults_count`, `srch_children_count`, `srch_room_count`, then numeric
`srch_booking_window`, `srch_saturday_night_bool`, `random_bool`, `price_usd`.
Category IDs are the source integers, not dense encodings. The binary label
is `click_bool`; only `price_usd` varies, with no monotonic features. The
workload keeps the first 40,000 test searches with at least 16 candidates,
their first 16 rows, in ascending search order: 640,000 rows.
Its workload ID is `cohort16-n16-g40000`: the conditional `-g<N>` suffix
separates a capped pool from an uncapped cohort, whose ID stays unchanged.

`train.params` holds the scale recipe: the paper's LightGBM recipe (learning
rate 0.05, seed 42) with a 4,096-leaf limit, looser categorical regularization
(`cat_l2` 1, `min_data_per_group` 50) and deterministic row-wise training. It
is separate from the released numeric recipe. `model.json`
records those parameters, the feature declarations, split, data hash and
library versions; changing any of them invalidates the model's cache key.
The cell uses the ordinary binary Treelite export, reference predictions,
stage oracle, contracts and hashed files. No new artifact format is needed.

The reference preparation of this recipe (LightGBM 4.6.0, Treelite 4.7.2, NumPy
2.5.3, Polars 1.44.2) trained on 7,932,228 rows and produced 600 trees of 4,096
leaves: 2,457,000 internal nodes, 21% categorical splits and 45% splits on
`price_usd`. `srch_destination_id` had 271,683 categorical nodes, 265,406
distinct sets and 7,243 named categories; `visitor_location_country_id`,
`site_id` and `srch_length_of_stay` had 59,349, 14,705 and 7,722 distinct sets.
Its LightGBM text model had SHA-256
`cffb55c67b8196e5be7018b2e76b17aaf801f590ab1724cfd2c70b993c7f98b5` and its
Treelite binary
`48e31772bfa1a14fc9492f2cd27acd62b025946fc1f7ad924a83ec9d51ddff03`. These are
reference measurements, not assertions about a model trained with different
library versions.

## The default-cap gap

TreeWalker 3.1.0 rejected the full model at the default 256 MiB
`LoadOptions::max_rank_table_bytes`, with `LoadError::Limit` containing
`inclusion graph exceeds` and `LoadOptions::max_rank_table_bytes`. On a GCP
c3d-standard-90 the reference model was rejected in about 1.3 s; with a 4 GiB
cap it loaded in about 4.7 s, with 190,335 chains in 31 MB of rank tables.

Category bits close the gap (the changelog's Unreleased section): the 265,406
distinct sets of `srch_destination_id` take a bit each, and the model loads at
the default limits in about 1.0 s, with the same outputs. Its bits take 242 MB
of the 256 MiB cap. The cell records no load options, so the runner's loads and
`preflight` take the library defaults, and a regression of the fix fails there.
The runner still honours a `load_options.max_rank_table_bytes` that a cell's
`cell.json` and identity record. The suite times TreeWalker only, avoiding
compiled-baseline preparation of this large model; it is opt-in rather than an
addition to the paper's factorial.

For a loader change, check the full binary and its walker configuration with
`Forest::load` at default options as well as running the suite. A small-tree
smoke model need not reach the bits' threshold.

## Reproduce

Obtain Expedia through `fetch-expedia` as described in the README. The project
normally installs newer LightGBM; the explicit `--with` selects the reference
version without changing the project's lock. Treelite is recorded too.

```sh
uv run --with lightgbm==4.6.0 --with treelite==4.7.2 treewalker-exp prepare --suite categorical-scale
uv run treewalker-exp manifest --suite categorical-scale
uv run treewalker-exp preflight --suite categorical-scale
uv run treewalker-exp run --suite categorical-scale --run-id categorical-scale
```

Full preparation trains on roughly 7.93 million rows and writes a multi-million
node model plus the 640,000-row workload, predictions and oracle. On a
c3d-standard-90 the reference preparation took 11 minutes, 10 of them training,
and peaked at 8.5 GB; the two-tree smoke test is not a full-training time
estimate. The experiments test suite prepares and resumes a synthetic tiny
cell through the same code, checking the split, feature types, candidate
order, cohort cap, parameters and load options independently.
