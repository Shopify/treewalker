# TreeWalker

Code and data for **TreeWalker: Partial Evaluation for Grouped Tree-Ensemble
Inference** (Durmus Karatay and Richard Newman, NeurIPS 2026).

TreeWalker evaluates a trained tree ensemble on groups of rows that share most
feature values. It walks each tree once per group, partitions a row bitmask at
splits on varying features, and skips empty subtrees. This repository contains
the engine, the benchmark harness, the scripts that turn public datasets into
models and benchmark runs, the result CSVs behind every number in the paper,
and the cloud setup used for the measurements.

The engine sources (`src/predict.rs`, `src/forest.rs`, `src/parser/`,
`src/mask.rs`, `src/config.rs`) are identical to the revision that produced
the factorial-grid results; later changes only extend the benchmark harness
(scenario grid, `--validate`).

## Layout

| Path | Contents |
|---|---|
| `src/` | inference library and treelite model parser |
| `benchmarks/` | separate unpublished crate: benchmark harness, `sweep_bench`, artifact correctness tests |
| `paper/experiments/scripts/` | dataset preparation, baseline compilation, figures, tables |
| `paper/experiments/data/` | released result CSVs and machine descriptions |
| `infra/` | Terraform and VM startup script for the two GCE benchmark machines |

## Building and packaging the library

The root crate contains the inference library. These commands build, test, and
package it without the benchmark dependencies:

```bash
cargo build
cargo test --release
cargo package
```

The package includes the Rust library sources, README, and license. Benchmark
sources, result data, Python scripts, Terraform files, and the repository's
native CPU build settings are excluded.

The benchmark crate has its own manifest and lockfile. Build it explicitly;
`--target-dir target` preserves the binary path used by the experiment scripts:

```bash
cargo build --manifest-path benchmarks/Cargo.toml --target-dir target \
  --release --features external-bench
```

The `external-bench` feature enables the C FFI baselines and QuickScorer.
`quickscorer-bench` enables the legacy CLI's QuickScorer baseline.

## Datasets

The scripts download the public datasets; none are redistributed here.

| Dataset | Source | Fetched by |
|---|---|---|
| SUPPORT | DeepSurv repository (`support_train_test.h5`) | `prepare.py` |
| FLCHAIN | R `survival` package via Rdatasets | `prepare.py` |
| Expedia (ICDM 2013) | Kaggle competition `expedia-personalized-sort` | `fetch_expedia.py`, from the `data.zip` you download |
| UCI Default of Credit Card Clients | OpenML 42477, CC BY 4.0, DOI 10.24432/C55S3H | `prepare_scenario.py` |

The Expedia data may not be redistributed. Accept the competition rules on
Kaggle, download `data.zip` (for example
`kaggle competitions download -c expedia-personalized-sort -f data.zip`), then
run `fetch_expedia.py --train-csv data.zip`; extracting the zip needs Info-ZIP
`unzip`, because it uses Deflate64. It writes
`paper/experiments/data/expedia.parquet` and checks its content fingerprint
against the file the paper used.

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

This needs only Python (3.12+) and [uv](https://docs.astral.sh/uv/), not the
benchmark machines. From the repository root:

```bash
uv sync
uv run python3 paper/experiments/scripts/plot.py                    # Figures 3, 4, 6-9
uv run python3 paper/experiments/scripts/gen_heatmap_tex.py         # Figure 2
uv run python3 paper/experiments/scripts/decomposition_validation.py  # Table 1, Figure 5
uv run python3 paper/experiments/scripts/summarize_scenario.py --arch arm   # Tables 3, 10 (also --arch intel)
uv run python3 paper/experiments/scripts/paper_numbers.py           # every in-text number
```

Outputs go to `paper/experiments/figures/`. `paper_numbers.py` prints each
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
grid used Rust 1.94.1 (`rust-toolchain.toml`); the scenario and chunked runs
used Rust 1.97.1. `infra/scripts/startup.sh` is the full machine recipe.

### On GCE with Terraform

```bash
cd infra
terraform init
terraform apply -var project=YOUR_PROJECT              # factorial grid (bench_suite = "paper")
terraform apply -var project=YOUR_PROJECT -var bench_suite=rebuttal   # scenario + chunked runs
```

The factorial grid needs `paper/experiments/data/expedia.parquet` from
`fetch_expedia.py`; Terraform uploads it for the trainer VM, which checks its
fingerprint before training. Terraform uploads `git archive` of `var.git_ref`
(default `neurips2026`) to a new bucket. The Intel VM trains the models and shares them, so both machines
evaluate identical models. Results land in the bucket under `results/`. The
scenario and chunked runs take about 25 minutes per VM.

### By hand on a prepared Linux machine

```bash
uv sync
uv run python3 paper/experiments/scripts/fetch_expedia.py --train-csv PATH/data.zip
uv run python3 paper/experiments/scripts/prepare.py --grid all --prepare-groups --skip-compiled
uv run python3 paper/experiments/scripts/prepare.py --grid all --compile-only   # tl2cgen, lleaves, sweep_bench
cargo test --manifest-path benchmarks/Cargo.toml --target-dir target \
  --release --features test-helpers

taskset -c 0 ./target/release/sweep_bench paper/experiments/artifacts --grid all \
  --output-dir paper/experiments/data --warmup 3 --iters 21 --min-iters 11 \
  --max-time-secs 30 --lgb-lib /usr/local/lib/lib_lightgbm.so --xgb-lib PATH/libxgboost.so

uv run python3 paper/experiments/scripts/prepare_scenario.py
uv run python3 paper/experiments/scripts/prepare_chunked.py --stage prepare
taskset -c 0 ./target/release/sweep_bench paper/experiments/artifacts --grid scen \
  --output-dir paper/experiments/data --warmup 3 --iters 21 --min-iters 11 --max-time-secs 10
taskset -c 0 uv run python3 paper/experiments/scripts/prepare_chunked.py --stage timing
```

The compiled baselines are installed separately, as in `startup.sh`:
`uv pip install 'tl2cgen>=1.0.0'` and
`uv pip install 'lleaves @ git+https://github.com/siboehm/lleaves.git@v1.4.1'`,
with LLVM 20 for lleaves. `sweep_bench` loads the native LightGBM and XGBoost
libraries given by `--lgb-lib` and `--xgb-lib`.

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

LightGBM (f64) predictions match treelite GTIL within 1e-14; XGBoost (f32)
predictions match native XGBoost within 1e-5 (`benchmarks/tests/correctness.rs`).
Partial evaluation and the full walk agree within 1e-15.

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
