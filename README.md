# TreeWalker

Code and data for **TreeWalker: Partial Evaluation for Grouped Tree-Ensemble
Inference** (Durmus Karatay and Richard Newman, NeurIPS 2026).

TreeWalker evaluates a trained tree ensemble on groups of rows that share most
feature values. It walks each tree once per group, partitions a row bitmask at
splits on varying features, and skips empty subtrees. This repository contains
the engine, the benchmark harness, the scripts that turn public datasets into
models and benchmark runs, the result CSVs behind every number in the paper,
and the cloud setup used for the measurements.

The released results use the engine preserved under the `neurips2026` tag.
The current library adds validated scalar Treelite loading. These changes are
separate from the archived paper measurements.

This repository serves two purposes: as a source for the library and as a
permanent archive of the code at time of submission. The latter can always be
found [under the neurips2026 tag](https://github.com/Shopify/treewalker/releases/tag/neurips2026).

## Layout

| Path | Contents |
|---|---|
| `src/` | inference library and treelite model parser |
| `experiments/benchmarks/` | separate unpublished crate: benchmark harness, `sweep_bench`, artifact correctness tests |
| `experiments/treewalker_exp/` | `treewalker-exp`: dataset preparation, workloads, training, compiled baselines |
| `experiments/grids.toml` | the benchmark suites, resolved by `treewalker-exp` into execution manifests |
| `experiments/scripts/` | figures, tables and the chunked-G run, on the released CSVs |
| `experiments/data/` | released result CSVs and machine descriptions |
| `infra/` | Terraform and VM startup script for the two GCE benchmark machines |

## Building and packaging the library

The root crate, `treewalker-gbdt`, contains the inference library. Rust imports
use `treewalker_gbdt`, for example `use treewalker_gbdt::Forest;`. These commands
build, test, and package it without the benchmark dependencies:

```bash
cargo build
cargo test --release
cargo package
```

`cargo test` also runs the research API's tests: the crate's dev-dependency on
itself turns the `research` feature on. Version 2.0 changed the API; the
[changelog](CHANGELOG.md) maps every 1.x call to its replacement.

The package includes the Rust library sources, loading documentation, README,
license, and small independent compatibility fixtures with their optional Python
generators. Benchmark sources, result data, experiment scripts, Terraform files,
and the repository's native CPU build settings are excluded.

The benchmark crate, `treewalker-bench`, is a workspace member with its own
manifest and the shared `Cargo.lock`. Build it explicitly; `--target-dir target`
preserves the binary path used by the experiment scripts:

```bash
cargo build --manifest-path experiments/benchmarks/Cargo.toml --target-dir target \
  --release --features external-bench
```

The `external-bench` feature enables the C FFI baselines and QuickScorer.
`quickscorer-bench` enables the legacy CLI's QuickScorer baseline.

## Using the library

Load a model once and predict through a `Predictor`, one per worker thread.
`Forest` is a cheap handle to the immutable model that can be cloned and
shared; a predictor owns the scratch space for the widest group, so its calls
never allocate.

```rust,no_run
use treewalker_gbdt::{Forest, LoadError};

fn main() -> Result<(), LoadError> {
    let forest = Forest::load("model.bin", "walker_config.json")?;
    let mut predictor = forest.predictor();
    // Two entities with 3 and 2 rows, row-major, in the model's trained column order.
    let data = vec![0.0; 5 * forest.config().n_features()];
    let mut out = vec![0.0; 5];
    predictor.predict_groups(&data, &[0, 3, 5], &mut out);
    Ok(())
}
```

`predict_group` predicts one group and `predict_fixed` consecutive groups of one
width. `Forest::from_reader` and `from_bytes` accept an explicit `ModelFormat`
and a `WalkerConfig`; build one in code with
`WalkerConfig::builder(n_features).max_group_width(w).varying([..]).build()?`.
The builder requires the varying features, so a forgotten list is an error
rather than every feature silently treated as constant.

The supported subset includes scalar regression, ranking and binary
classification, `identity`/`sigmoid`, sum/average aggregation, and scalar base
scores. Binary Treelite v4 is recommended; native XGBoost JSON must first be
converted through Treelite. Inputs have up to 64 features; groups can have any
number of rows.
See [Treelite loading](docs/treelite-loading.md) for export examples, the exact
output and precision policy, errors, limits, and the caller's grouping contract.

## Datasets

None of the datasets are redistributed here. `uv run treewalker-exp fetch`
downloads the public ones into `experiments/data/raw/` and checks each against
a pinned SHA-256; `prepare` fetches them on demand.

| Dataset | Source | Fetched by |
|---|---|---|
| SUPPORT | DeepSurv repository (`support_train_test.h5`) | `treewalker-exp fetch` |
| FLCHAIN | R `survival` package via Rdatasets | `treewalker-exp fetch` |
| Expedia (ICDM 2013) | Kaggle competition `expedia-personalized-sort` | `treewalker-exp fetch-expedia`, from the `data.zip` you download |
| UCI Default of Credit Card Clients | OpenML 42477, CC BY 4.0, DOI 10.24432/C55S3H | `treewalker-exp fetch` |

The Expedia data may not be redistributed. Accept the competition rules on
Kaggle, download `data.zip` (for example
`kaggle competitions download -c expedia-personalized-sort -f data.zip`), then
run `treewalker-exp fetch-expedia --train-csv data.zip`; extracting the zip needs Info-ZIP
`unzip`, because it uses Deflate64. It writes
`experiments/data/expedia.parquet` and checks its content fingerprint
against the file the paper used.

The factorial suite also prepares two kinds of derived models
(`experiments/grids.toml`), neither of which changes an existing model or cell:

- `expedia-filled`, the Expedia split with every missing value encoded before
  training as its column's training-split minimum minus max(1, |minimum|), the
  same constant in train and test (test values below it are kept). The constant
  stays below the minimum after the f32 conversion, so QuickScorer, which has no
  missing-value handling, computes the same function as every other method.
  Sessions workload, T in {50, 500, 1000, 2000}, L in {2, 4}, with Expedia's
  sessions, order and varying features.
- Seed replicates, `_r1` to `_r4` after a model's name. Training is
  deterministic, so each replicate trains with the released parameters on a
  seeded sample of exactly 80% of the training split's entities (patients,
  sessions, applicants) and is timed on the released cell's data. At three
  anchors: `support/nt500_md4_h16` (panel), `credit/nt500_md4` (what-if k4 G16,
  on the features the released model splits on most) and `expedia/nt500_md8`
  (sessions). A model's `model.json` records the sample (`replicate`) or the
  encoding (`missing_values`).

## Released results

All latencies are medians of per-block medians, in µs per observation (group);
`p5_us` and `p95_us` are over block medians and `iters` is the number of
blocks.

| File | Experiment |
|---|---|
| `grid1_results_{intel,arm}.csv` | factorial grid: 3 datasets × 2 frameworks × 4 T × 4 L × 8 G, all methods |
| `grid3_results_{intel,arm}.csv`, `grid3_stats_{intel,arm}.csv` | ablations (timing) and work counters |
| `grid4_results_{intel,arm}.csv` | Expedia group-size distributions |
| `scenario_credit_{results,stats}_{intel,arm}.csv` | scenario-analysis benchmark (§5.6, App. E); latencies are per scenario set of G variants |
| `chunked_g_results_{intel,arm}.csv` | groups wider than 128 rows (App. F), µs per row |
| `system_info_{intel,arm}.txt` | machine and toolchain for the scenario and chunked runs |

## Reproducing the tables and figures

This needs only Python 3.14 (`.python-version`) and [uv](https://docs.astral.sh/uv/), not the
benchmark machines. From the repository root:

```bash
uv sync
uv run python3 experiments/scripts/plot.py                    # Figures 3, 4, 6-9
uv run python3 experiments/scripts/gen_heatmap_tex.py         # Figure 2
uv run python3 experiments/scripts/decomposition_validation.py  # Table 1, Figure 5
uv run python3 experiments/scripts/summarize_scenario.py --arch arm   # Tables 3, 10 (also --arch intel)
uv run python3 experiments/scripts/paper_numbers.py           # every in-text number
```

Outputs go to `experiments/figures/`. `paper_numbers.py` prints each
number next to the value printed in the paper.

| Paper item | Source |
|---|---|
| Table 1, Figure 5 | `decomposition_validation.py` on `grid3_stats_intel.csv` |
| Figure 2 (heatmap) | `gen_heatmap_tex.py` on `grid1_results_intel.csv` |
| Figures 3, 4, 6-9 | `plot.py` on `grid1`, `grid3`, `grid4` |
| Tables 3 and 10 | `summarize_scenario.py`, `scenario_credit_*` |
| Table 11 | `chunked_g_results_*.csv` |
| Tables 9 and in-text counts (§5, §6, App. A.6, App. D) | `paper_numbers.py` |
| App. A.3 (f32 audit) | `audit_f32.py` (needs prepared artifacts) |

## Rerunning the benchmarks

The paper's measurements ran on two Google Compute Engine VMs in
`us-central1-a`: `c4-standard-32` (Intel Xeon Platinum 8581C) and
`c4a-highmem-16` (Google Axion, Neoverse V2), Ubuntu 24.04, SMT disabled,
performance governor, ASLR off, pinned to one core with `taskset -c 0`.
LightGBM 4.6.0 and XGBoost 3.2.0 were built from source with `-march=native`,
TreeWalker with `-C target-cpu=native` (`.cargo/config.toml`). The factorial
grid used Rust 1.94.1; the scenario and chunked runs used Rust 1.97.1, the
version now pinned in `rust-toolchain.toml`. `infra/scripts/startup.sh` is the
full machine recipe; it now builds the versions `uv.lock` installs, LightGBM
4.7.0 and XGBoost 3.4.1.

### On GCE with Terraform

```bash
cd infra
terraform init
terraform apply -var project=YOUR_PROJECT -var git_ref=REF              # factorial grid (bench_suite = "paper")
terraform apply -var project=YOUR_PROJECT -var git_ref=REF -var bench_suite=rebuttal   # scenario + chunked runs
```

The factorial grid needs `experiments/data/expedia.parquet` from
`treewalker-exp fetch-expedia`; Terraform uploads it for the trainer VM, which checks its
fingerprint before training. Terraform uploads `git archive` of `var.git_ref`
to a new bucket. The ref has no default: it must contain `experiments/` and
`treewalker-exp`, which the `neurips2026` tag predates. The Intel VM trains the models and shares them, so both machines
evaluate identical models. Results land in the bucket under `results/`. The
scenario and chunked runs take about 25 minutes per VM.

### By hand on a prepared Linux machine

```bash
uv sync --group baselines
uv run --group baselines treewalker-exp fetch-expedia --train-csv PATH/data.zip
uv run --group baselines treewalker-exp prepare --suite factorial
uv run --group baselines treewalker-exp compile-baselines --suite factorial   # tl2cgen, lleaves
uv run --group baselines treewalker-exp build-bench
cargo test --manifest-path experiments/benchmarks/Cargo.toml --target-dir target \
  --release --features research

taskset -c 0 ./target/release/sweep_bench experiments/artifacts --grid all \
  --output-dir experiments/data --warmup 3 --iters 21 --min-iters 11 \
  --max-time-secs 30 --lgb-lib /usr/local/lib/lib_lightgbm.so --xgb-lib PATH/libxgboost.so

uv run --group baselines treewalker-exp prepare --suite scenario-v1 --validate
uv run --group baselines python3 experiments/scripts/prepare_chunked.py --stage prepare
taskset -c 0 ./target/release/sweep_bench experiments/artifacts --grid scen \
  --output-dir experiments/data --warmup 3 --iters 21 --min-iters 11 --max-time-secs 10
taskset -c 0 uv run --group baselines python3 experiments/scripts/prepare_chunked.py --stage timing
```

tl2cgen comes from the locked `baselines` dependency group. It has no aarch64
Linux wheel, so there it builds from source with
`CXX=$PWD/infra/scripts/cxx-cstdint`. lleaves runs in
`experiments/compile.py`'s own script lock, with `llc` and `clang` from LLVM 20
on `PATH` or given by `--llc` and `--clang`. Every `uv run` passes
`--group baselines`, because `uv run` removes packages outside the groups it
syncs. `sweep_bench` loads the native LightGBM and XGBoost
libraries given by `--lgb-lib` and `--xgb-lib`.

`uv.lock`, `experiments/compile.py.lock` and
`tests/fixtures/import/generate.py.lock` resolve against PyPI through the
`pypi` index that the project and both scripts declare, which takes priority
over any index in a user's uv configuration. Relock with `uv lock`,
`uv lock --script experiments/compile.py` and
`uv lock --script tests/fixtures/import/generate.py`; `uv lock --check` (with
`--script` for a script) verifies a lock without changing it.

### What to expect from a rerun

A clean-room rerun (September 2026, fresh VMs of the same machine types in
`us-east4-a`, built from this repository) reproduced every work counter
exactly: all 932 ablation-grid rows per architecture and all 32
scenario-analysis rows match the released CSVs, because training is
deterministic (seed 42) and the engine does identical work. Timings vary with
the host. On Arm, algorithmic speedups moved by a median of +0.9%, and 91% of
the 544 grid cells were within 5% of the released values. On Intel, the whole
VM ran about 19% faster (median TreeWalker latency), and speedups moved by a
median of +3.4% (interquartile range -0.5% to +9.2%). Every qualitative result
held, but exact values move, so `paper_numbers.py` matches the paper exactly
only on the released CSVs.

## Correctness

A prediction has three stages: `tree_sum`, the sum of the row's leaf values;
`raw_margin`, `tree_sum / divisor + base_score`; and the output, after the
identity or the sigmoid. When `Forest::exact_sums()` is true, as it is for the
12 models of the cells measured below, the leaves are added as exact fixed-point
integers and rounded once: `tree_sum` is the `f64` nearest to the true sum, and tree
order, layout and every ablation except `disable_exact_sums` leave it unchanged.
The margin and the link are ordinary rounded `f64` operations.

LightGBM (f64) predictions match Treelite GTIL within 1e-13, and XGBoost (f32)
predictions match native XGBoost within 1e-5 (`experiments/benchmarks/tests/correctness.rs`).
GTIL and the full walk add leaves in `f64` in tree order, so they differ from the
exact sum by their rounding, which grows with the partial sums. Measured on
2026-10-04, the full walk's outputs differ by at most 2.2e-15 on the FLCHAIN and
SUPPORT LightGBM cells at horizons 16 and 128, 7.2e-16 on Expedia and 1.1e-14 on
FLCHAIN at horizon 1,024; the XGBoost models' outputs are bit-identical.

## Citation

```bibtex
@inproceedings{karatay2026treewalker,
  title     = {{TreeWalker}: Partial Evaluation for Grouped Tree-Ensemble Inference},
  author    = {Karatay, Durmus and Newman, Richard},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). This covers the code and the released result
CSVs; the datasets keep their own terms (see Datasets).
