# TreeWalker

Code and data for **TreeWalker: Partial Evaluation for Grouped Tree-Ensemble
Inference** (Durmus Karatay and Richard Newman, NeurIPS 2026).

TreeWalker evaluates a trained tree ensemble on groups of rows that share most
feature values. It walks each tree once per group, partitions a row bitmask at
splits on varying features, and skips empty subtrees. This repository contains
the engine, the benchmark harness, the `treewalker-exp` package that turns
public datasets into models, runs the benchmark suites and computes the
paper's figures, tables and numbers, the raw measurements of the final run,
and the cloud setup used for them.

The measurements here are the 2.0 engine's final run (October 2026). The
paper as submitted, its engine, its result CSVs and the scripts that read them
are preserved under the `neurips2026-submission` tag; the camera-ready paper's code,
measurements and figure code are under `neurips2026-final`. Since 3.0 the library
has a new prediction kernel that sums every model's leaves exactly; the
measurements here predate it.

This repository serves two purposes: as a source for the library and as a
permanent archive of the code at time of submission. The latter can always be
found [under the neurips2026-submission tag](https://github.com/Shopify/treewalker/releases/tag/neurips2026-submission).

## Layout

| Path | Contents |
|---|---|
| `src/` | inference library: the public API (`forest.rs`, `predictor.rs`, `config.rs`, `error.rs`); loading (`load/`: the Treelite decoders in `parse/`, tree order, varying predicates, the prediction layout in `layout/`); prediction (`predict/`: precompute, walk, resolve, leaf pass, finish, row masks in `mask/`); exact sums (`exact/`); the kernel's rules and the machine model (`rules/`); and the `research` API (`research/`) |
| `tests/` | the library's integration tests: import fixtures and their GTIL references, recorded output vectors (`fixtures/vectors.bin`), property tests on generated models, and small cases for Miri |
| `experiments/benchmarks/` | separate unpublished crate: benchmark harness, `sweep_bench`, artifact correctness tests |
| `experiments/treewalker_exp/` | `treewalker-exp`: dataset preparation, workloads, training, compiled baselines |
| `experiments/grids.toml` | the benchmark suites, resolved by `treewalker-exp` into execution manifests |
| `experiments/data/runs/` | the final run's raw measurements: packed runs, Parquet in git-lfs |
| `experiments/figures/` | figures, tables and numbers `treewalker-exp` generates (git-ignored) |
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
itself turns the `research` feature on. Versions 2.0 and 3.0 changed the API; the
[changelog](CHANGELOG.md) maps every 1.x call and every 2.x research call to its
replacement.

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

The `external-bench` feature enables the C baselines and QuickScorer;
`quickscorer-bench` enables QuickScorer alone, and `pmu` the Linux hardware
counters. `cargo bench` runs small divan benchmarks of the library on the
committed fixtures.

### Instruction-set levels and CPU flags

A build without CPU flags runs the kernel at the CPU's level on x86-64. It compiles the
kernel for the SSE2 baseline, for x86-64-v3 and for x86-64-v4 with VPOPCNTDQ, BITALG
and BF16, and each predictor runs the highest level the CPU has, detected when the
predictor is made. `TREEWALKER_ISA=base`, `v3` or `v4` asks for a lower level, never a
higher one. Outputs are the same at every level. A build whose features include AVX2
and FMA, such as `-C target-cpu=x86-64-v3` or `-C target-cpu=native` on a CPU that has
them, compiles only its own level and does not read `TREEWALKER_ISA`, and so do builds
for other architectures. With the `research` feature, `ResearchPredictor::isa()` names
the level a predictor runs. A project that depends on `treewalker-gbdt` builds it with
its own flags; this repository's `.cargo/config.toml`, which the package excludes,
builds natively.

On the seven Google Cloud families of 3.1.0's measurements ([changelog](CHANGELOG.md)),
a default build took, against a `-C target-cpu=native` build, over the 90 cells at
both function alignments:

| | Emerald Rapids | Ice Lake | Milan | Genoa | Turin | Axion | Altra |
|---|---:|---:|---:|---:|---:|---:|---:|
| Level the default build runs | v4 | v3 | v3 | v4 | v4 | native | native |
| Time against native | 1.006 | 0.976 | 1.022 | 1.021 | 1.027 | 1.008 | 0.991 |

- On AMD, build with `-C target-cpu=native` for the best speed. The default build
  took 2.1 to 2.7% longer on Milan, Genoa and Turin, and 3.4 to 4.3% on Genoa's and
  Turin's single rows. Its dispatched code has LLVM's generic tuning, under which the
  walk loops differ from the native ones in instruction order and registers; why
  that costs on these cores is not established (the changelog's known gaps).
- On Intel, the default build is as fast as a native one: within 1% on Emerald
  Rapids, and 2.4% faster on Ice Lake, whose AVX-512 lacks BF16, so that it runs v3.
- On AArch64 every build compiles only its own level; the default one was within 1%
  of native on Neoverse V2 and N1.
- A native build runs only on CPUs with the build machine's features; a default one
  runs on any CPU of its architecture.

### Development checks

`experiments/scripts/checks.sh` runs the checks as concurrent lanes, each building in
its own target directory under `target/checks`, and writes every step's log and a
summary of their status, test counts and times to `target/checks/logs`:

```bash
PYTHON=.venv/bin/python experiments/scripts/checks.sh fast   # every commit
PYTHON=.venv/bin/python experiments/scripts/checks.sh full   # before a pull request or a release
```

The fast checks are `cargo fmt --check`, `cargo clippy --workspace --all-targets --
-D warnings` with and without the research features, `cargo test --workspace`, and the
Python tests, ruff and mypy: 44 s on an M4 Pro once built. The full checks, 4 to 5
minutes, add the release tests with the research API, the bench crate's tests, the
property tests at their full counts (`TW_PROPTEST_FULL=1`), cross clippy, Miri, the
spill checks and the bench crate's suite on the core artifacts (`TEST_ARTIFACTS_BASE`).

What guards what. The property tests (`tests/property.rs`) check production, each call
and the research predictor against an exact oracle and the full walk, bit for bit, on
random models in both formats with random layouts, and groups of 1 to 2,049 rows: the
widths around every mask boundary, more than 64 features, and category IDs anywhere in
`u32`. The artifact suite checks real XGBoost and LightGBM models against the frameworks'
own outputs, `tests/import.rs` the exported fixtures against them, and
`tests/vectors.rs` every output against recorded vectors, so that a release that changes
one fails. Unit tests localize the hard algorithms: exact rounding, the bins and
category chains, the precompute's sweep, the masks' runs and the rules. A bad model or
data input gives an error; a violated calling contract panics, as documented (see
[Treelite loading](docs/treelite-loading.md)). Each instruction set's paths run the same
tests natively on their machines.

The Linux-only code (the timestamp counter's calibration, `perf_event`) is checked by
cross clippy. Parquet's zstd is C, so without a Linux cross toolchain the check compiles
it with clang and the macOS SDK headers; nothing builds or links for Linux:

```bash
SDK=$(xcrun --show-sdk-path)
export CC_x86_64_unknown_linux_gnu=clang \
  CFLAGS_x86_64_unknown_linux_gnu="--target=x86_64-unknown-linux-gnu -isystem $SDK/usr/include" \
  CC_aarch64_unknown_linux_gnu=clang \
  CFLAGS_aarch64_unknown_linux_gnu="--target=aarch64-unknown-linux-gnu -isystem $SDK/usr/include -D__arm64__"
for t in "aarch64-apple-darwin apple-m4" "x86_64-unknown-linux-gnu emeraldrapids" \
         "x86_64-unknown-linux-gnu x86-64-v4" "x86_64-unknown-linux-gnu x86-64" \
         "aarch64-unknown-linux-gnu neoverse-v2"; do
  set -- $t
  for f in "" "--features treewalker-bench/external-bench,treewalker-bench/pmu" \
           "--features treewalker-bench/research"; do
    RUSTFLAGS="-C target-cpu=$2" cargo clippy --target $1 --workspace --all-targets $f -- -D warnings
  done
done
```

Miri checks the unsafe code on small production-shaped cases (`tests/miri.rs` and the
lib's `prefetching_walks` and `production_scratch_holds_a_leaf_per_row`, the latter in
buffers sized as production builds size them), each of which runs every kind of unsafe operation of its
path once, with a nightly toolchain (`rustup +nightly component add miri`). The x86-64
runs take the SSE2 and AVX2 masks, the RISC-V run the plain-word fallback; on AArch64,
Miri lacks a NEON intrinsic the masks use. The walks link nodes by byte offset on x86-64
and by index elsewhere (`src/load/layout`), so the x86-64 run also takes the layout's
tests of x86-64's encoding, which an AArch64 machine's native tests do not. An empty
`RUSTFLAGS` replaces the repository's `target-cpu=native`:

```bash
for cpu in x86-64 x86-64-v3; do
  RUSTFLAGS="-C target-cpu=$cpu" cargo +nightly miri test --target x86_64-unknown-linux-gnu --lib -- \
    prefetching_walks production_scratch
  RUSTFLAGS="-C target-cpu=$cpu" cargo +nightly miri test --target x86_64-unknown-linux-gnu --test miri
done
RUSTFLAGS="-C target-cpu=x86-64" cargo +nightly miri test --target x86_64-unknown-linux-gnu --lib -- \
  load::layout::tests::split_bounds load::layout::tests::heavy_right_splits \
  load::layout::tests::signed_zero load::layout::tests::nested_heavy
RUSTFLAGS= cargo +nightly miri test --target riscv64gc-unknown-linux-gnu --lib -- prefetching_walks \
  production_scratch
RUSTFLAGS= cargo +nightly miri test --target riscv64gc-unknown-linux-gnu --test miri
```

The walk's lane count comes from a model of its step (`src/rules/lanes.rs`): on AArch64
the most lanes that fit the registers, on x86-64 the fewest that reach the issue bound,
whose lanes beyond the registers spill. `experiments/scripts/walk_spills.py` builds
`examples/walk_asm.rs` for a target and CPU, emits its post-LTO assembly and counts each
walk loop's stack stores and loads. It fails if a loop stores to the stack (a spilled
lane value) although the lanes, a step's temporaries and the loop's invariants all fit
the registers, or stores nothing although they do not; `--lanes K` builds every walk at
K lanes to see where the registers run out:

```bash
python3 experiments/scripts/walk_spills.py                       # this machine
python3 experiments/scripts/walk_spills.py --target x86_64-unknown-linux-gnu --cpu emeraldrapids
```

`experiments/scripts/asm_identity.py` compares the production functions of two commits
on a target and CPU. It builds each commit's post-LTO assembly of `examples/walk_asm.rs`
(or, with `--bench`, of `sweep_bench`, whose timed regions it reports apart), demangles
and normalizes it, and lists the functions that differ or that one side lacks. A change
meant to leave the generated code alone, such as a refactoring, should show none; `--map`
renames the paths of code that moved between modules:

```bash
python3 experiments/scripts/asm_identity.py HEAD~1 HEAD --cpu apple-m4
python3 experiments/scripts/asm_identity.py HEAD~1 HEAD --target x86_64-unknown-linux-gnu --cpu emeraldrapids
```

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
converted through Treelite. Inputs have up to `i32::MAX` features, Treelite's own
limit; trees, nodes, depth and category IDs are bounded only by the input caps of
`LoadOptions`, memory and a node pool below 4 GiB (2^28 − 2 nodes in all, on every
target); groups can have any number of rows.
See [Treelite loading](docs/treelite-loading.md) for export examples, the exact
output and precision policy, errors, limits, prediction scratch, and the caller's
grouping contract.

### Research API

The `research` feature adds `treewalker_gbdt::research`, which the benchmark harness
uses to measure the kernel. Serving needs none of it.

- `forest.research_predictor(ablation)` runs the kernel with runtime ablations, each
  disabling one optimization or forcing one choice: `disable_predicate_sweep`,
  `disable_varying_precompute`, `disable_monotonic`, `disable_unsplit`,
  `disable_g1_constant`, `disable_width_dispatch`, `disable_restart_prefetch` and
  `lanes` (1 to 12, for every walk, instead of the rule's per walk). An ablation's
  tokens are its flags' names and `lanes=N`; `Ablation::id()` joins them with `+`
  (`all-on` for production), `from_id` parses an ID, and `with(token)` sets one
  field.
- `Forest::load_ablated`, `from_bytes_ablated` and `from_reader_ablated` take a
  `LoadAblation`, the load-time choices turned off or forced:
  `disable_tree_ordering`, `disable_bitset_intern`, `disable_predicate_dedup` and
  `prefix_depth` (every run of trees sharing that many constant splits, instead of
  the rule's groups; 0 for none) and `category_bits_above` (bits rather than chains
  for constant categorical features of more distinct heavy sets than that, instead
  of 100,000, and beside them for those whose bits take no more bytes than their
  chains can; 0 for every one, as far as the cap allows).
- `predict_group` and `predict_groups` are the timed build; `predict_group_counted`
  returns `WorkCounters` (`COUNTERS_VERSION` 6: groups, single rows, bins, precompute,
  event scans, prefix and walk steps, lane slots, passes, events, splits, runs and
  arena slots; `fields()` lists them) and adds its phase times to `take_phases()`
  (`Phases`: bins, precompute, prefix, walk, resolve, leaf pass and finish).
- `predict_group_stages` keeps each row's `tree_sum`, `raw_margin` and `output`.
- `forest.layout_stats()` gives what the kernel's rules chose for the model on this
  machine and their inputs, and the size of its layout: the lanes per walk and the
  registers they were counted from, the cache sizes and their source, the chunk
  budget, the expected walk steps per tree, the cache level holding the nodes with
  its probed latency and the prefetch distances, the prefix groups with the steps
  they save and cost, and the bytes of the node pool, leaf table, categorical
  rank tables (with the chain slots and the heavy sets they cover, and the bit slots
  and theirs) and the varying categorical features' flip lists.
- `forest.predict_full_walk(rows, out)` walks every tree for every row and sums
  exactly: an independent reference that production matches bit for bit.
- `nodes()`, `trees()`, `tree_nodes()`, `bitset_bytes()`, `threshold_type()` and
  `predicate_masks()` expose the parsed model.

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

The separate `categorical-scale` suite prepares a public-data model with very
large categorical features, which 3.1.0 failed to load at the default limits,
`expedia-cat/nt600_md32_nl4096/lightgbm/cohort16-n16-g40000`.
It does not change the paper's numeric Expedia cells or enter their factorial.
See [the recipe, load-cap expectation and reproduction commands](docs/categorical-scale.md).

The factorial suite also prepares two kinds of derived models
(`experiments/grids.toml`), neither of which changes an existing model or cell:

- `expedia-filled`, the Expedia split with every missing value encoded before
  training as its column's training-split minimum minus max(1, |minimum|), the
  same constant in train and test (test values below it are kept). The constant
  stays below the minimum after the f32 conversion, so QuickScorer, which has no
  missing-value handling, can take these models. That does not make it compute
  the same function: it compares thresholds in f32, and in the final run 5 of the
  16 cells' QuickScorer outputs failed validation against LightGBM's f64
  thresholds. Only cells whose QuickScorer outputs validate are timed.
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
- What-if grids (`whatif-grid-v1`, in the kernel suite), groups of more than 1,024
  rows, on the kernel's runtime-width masks, which no panel reaches (FLCHAIN realizes at most 1,008 steps,
  SUPPORT 104). Each of 64 seeded base entities over an ICE grid of its model's two
  most-split covariates, at 64 x 32 or 64 x 64 levels: training values at evenly
  spaced quantiles, so levels repeat where values tie. On `credit/nt500_md8` and
  `flchain/nt50_md4_h16`, groups of 2,048 and 4,096 rows; they validate the kernel
  and tune nothing.
- Scattered leaves (`kernel-whatif-credit`, whatif-v2 in the kernel suite): credit
  what-ifs at `credit/nt500_md8`, k = 1 and 8, G = 128, where most runs of a leaf's
  rows are one row long. They check the leaf pass on fragmented masks and tune
  nothing.

## Released results

`experiments/data/runs/` holds the camera-ready campaign (2026-10-11 to 13, exact
sums off) and the final run's factorial and ablation runs (2026-10-09 and 10,
exact sums on), one directory per run ID, packed
with `treewalker-exp pack`: `run.json` (configuration, host, timer, sentinel and
the order cells ran in), `cells.json` (each cell's manifest: its sample,
validation, stopping, probes and XGBoost's processes), the `samples`, `groups`,
`counters`, `hw` and `stages` tables, `timer.parquet`, and the sentinel's cells
under `sentinel/`. The tables are Parquet in git-lfs: run `git lfs pull` after
cloning. `stages` (per cell `stages.parquet`; packed as `stages.parquet` and
`sentinel/stages.parquet`) holds TreeWalker's tree sum, raw margin and output on
each row of the cell's stage oracle (columns suite, cell, group, row, tree_sum,
raw_margin, output), written untimed before timing; output schema 6, absent from
older runs.

| Run | Campaign | Contents |
|---|---|---|
| `camera-ready-{x86_64,aarch64}` | camera-ready | the paper's 712 cells: survival panels (SUPPORT and FLCHAIN, horizons 1-128; FLCHAIN also 256-1,024, as whole groups and as 128-row chunks), Expedia sessions, cohorts of 4-32 rows and filled missing values |
| `camera-ready-ablation-{1,2}-{x86_64,aarch64}` | camera-ready | the 36 ablation cells, run before and after `camera-ready` |
| `scenario-v1-{x86_64,aarch64}` | camera-ready | E1: the 16 released credit scenario cells |
| `acceptance-{x86_64,aarch64}` | camera-ready | 17 acceptance cells |
| `layout-{default,align64}-{x86_64,aarch64}` | camera-ready | the acceptance cells with default and 64-byte-aligned functions |
| `RUN-retime-rN-{x86_64,aarch64}` | camera-ready | re-time attempts: the cells and modes of `RUN` whose interval was wider than ±2%, timed again with N rounds on the same VM |
| `factorial-{x86_64,aarch64}` | final | 1,134 cells: survival panels (SUPPORT and FLCHAIN, horizons 1-128; FLCHAIN also 256-1,024), Expedia sessions, cohorts of 4-32 rows and filled missing values, the credit and reference what-ifs, and seed replicates |
| `ablation-{x86_64,aarch64}` | final | 36 cells with every switch and their combinations, in serving and batch mode |

`experiments/data/retime/` holds each re-time run's manifest and script. The
final run's acceptance and layout runs, which share their IDs with the
camera-ready ones, are in the history at `2562892`.

`experiments/data/precision/{intel,arm}/` holds App. A.3's measurements, taken
on the campaign's machine types (2026-10-13, the `precision` diagnostic):
`precision_order.json`, 2.0's `sweep_bench precision-order` over every
camera-ready LightGBM cell (tree ordering on against off with exact sums off,
and both against prepare's GTIL reference); `audit_f32.md`, `audit-f32` with
native XGBoost's margins from the library the campaign timed
(`xgboost_library.txt` records its hashes), beside the camera-ready runs'
TreeWalker stages; and both commands' logs.

## Reproducing the tables and figures

This needs Python 3.14 (`.python-version`), [uv](https://docs.astral.sh/uv/) and
git-lfs, not the benchmark machines. From the repository root:

```bash
git lfs pull
uv sync
uv run treewalker-exp figures   # Figures 2-9: PNG and PDF, and the heatmap as TikZ
uv run treewalker-exp tables    # Table 1 (TeX), Tables 3 and 10, App. A.5 and D, ablation runs (Markdown)
uv run treewalker-exp numbers   # every in-text number, with its paper location
uv run treewalker-exp summarize experiments/data/runs/factorial-x86_64   # one run, cell by cell
```

Outputs go to `experiments/figures/`. `figures`, `tables` and `numbers` read a
campaign's runs (`--campaign`): `final`, the run above, or `camera-ready`
(below); without the option, the camera-ready runs where they exist. The f32 audit also needs prepared models
and cells under `experiments/artifacts/`; `treewalker-exp audit-f32
--list-missing` lists the files it reads.

| Paper item | Command | Output |
|---|---|---|
| Figure 2 (heatmap) | `figures` | `speedup_heatmap.{pdf,png,tex}` |
| Figures 3, 4, 6-9 | `figures` | `horizon_amortization`, `method_comparison`, `cross_platform`, `ablation_waterfall`, `ablation_nodes`, `depth_sensitivity`, `ntrees_scaling` |
| Table 1, Figure 5 | `tables`, `figures` | `decomposition_table.tex`, `asymptotic_convergence` |
| Tables 3 and 10 | `tables` | `scenario_credit_{intel,arm}.md` |
| App. A.5 (tl2cgen's f64 aliasing defect) | `tables` | `tl2cgen_defect_{intel,arm}.md` |
| App. D (QuickScorer on `expedia-filled`, replacing the withdrawn Table 9) | `tables` | `quickscorer_filled_{intel,arm}.md` |
| §5.7 ablation effects per run and their between-run range | `tables`, `figures` | `ablation_effects.md`, `ablation_waterfall` |
| Table 11, in-text numbers (§4-§7, App. A.6, D, F) | `numbers` | `paper_numbers.txt` |
| App. A.3 (f32 audit) | `audit-f32` | `audit_f32.md` |

A method's latency in a cell is v1's estimand, completed for rounds: the median
over blocks (one round's pass over one batch) of each block's median time per
group, by v1's nearest-rank convention, from process 0; p5-p95 over blocks is
descriptive dispersion. A speedup is the ratio of two latencies, with a 95%
bootstrap interval that resamples entities and rounds as crossed clusters, pairs
the two methods and recomputes the nested medians. Sampled cells give every group
the same inclusion probability, which the analysis checks, so the medians are
unweighted. The row-weighted ratio of mean time per row and its interval are kept
as a sensitivity column (`speedup_rw`), and XGBoost native's faster mode across
its extra processes as a sensitivity row. Every cell runs in a process of its
own, so no method is corrected for drift. `numbers` keeps the v1 scripts' items,
and its module docstring says where an item changed.

## Rerunning the benchmarks

The final run ran on two Google Compute Engine VMs in `us-east4-a`:
`c4-standard-32` (Intel Xeon Platinum 8581C, at its all-core turbo) and
`c4a-highmem-16` (Google Axion, Neoverse V2), Ubuntu 26.04, SMT disabled, ASLR
off, hardware counters at the PMU's `STANDARD` level, pinned to one core with
`taskset -c 0`. LightGBM 4.7.0 and XGBoost 3.4.1, the versions `uv.lock`
installs, were built from source for the host, and TreeWalker with
`-C target-cpu=native` and Rust 1.97.1 (`rust-toolchain.toml`). Each
`run.json` records the rest (`host`, `system_info`).
`infra/scripts/startup.sh` is the full machine recipe. The paper's
measurements (the `neurips2026-submission` tag) ran on the same machine types in
`us-central1-a`.

### On GCE with Terraform

```bash
cd infra
terraform init
terraform apply -var project=YOUR_PROJECT -var git_ref=$(git rev-parse HEAD) -var 'suites=["acceptance"]' -var layout_check=true
terraform apply -var project=YOUR_PROJECT -var git_ref=$(git rev-parse HEAD)      # factorial and ablation
```

With `-var cache_bucket=NAME`, an existing bucket keeps the prepared models and each
machine type's compiled baselines across deployments, so a rerun retrains and
recompiles only what changed. Prep and `compile-baselines` reuse a cached file only
when its recorded key and hash match. Create the bucket once, outside Terraform, so
`terraform destroy` leaves it:
`gcloud storage buckets create gs://NAME --location US --uniform-bucket-level-access`.

The VMs run with the GCE PMU at `STANDARD` and `turbo_mode = "ALL_CORE_MAX"`. They boot
a pinned image, and every `apt` operation uses one Ubuntu archive snapshot
(`apt_snapshot`), so the compiler and tools, and with them the cached compiled
baselines, stay the same across deployments.
Suites with Expedia cells need `experiments/data/expedia.parquet` from
`treewalker-exp fetch-expedia`; Terraform uploads it for the trainer VM, which checks its
fingerprint before training. Terraform uploads `git archive` of `var.git_ref`
to a new bucket. `git_ref` is a full 40-character commit SHA, which Terraform
checks: the archive, the VMs and the model hand-over are named after it, so a
moving branch would keep the old source. It has no default: it must contain `experiments/` and
`treewalker-exp`, which the `neurips2026-submission` tag predates. The Intel VM trains the models and shares them, so both machines
evaluate identical models, checked by hash. Results land in the bucket under
`results/`.

### By hand on a prepared machine

```bash
uv sync --group baselines
uv run --group baselines treewalker-exp build-native            # LightGBM, XGBoost from pinned sources
uv run --group baselines treewalker-exp fetch-expedia --train-csv PATH/data.zip
uv run --group baselines treewalker-exp prepare --suite acceptance
uv run --group baselines treewalker-exp compile-baselines --suite acceptance   # tl2cgen, lleaves, QuickScorer XML
uv run --group baselines treewalker-exp preflight --suite acceptance
uv run --group baselines treewalker-exp run --suite acceptance --run-id acceptance
uv run --group baselines treewalker-exp pack experiments/data/runs/acceptance
uv run --group baselines treewalker-exp summarize experiments/data/runs/acceptance
uv run --group baselines treewalker-exp check-run experiments/data/runs/acceptance
```

Then the same with `--suite factorial` and `--suite ablation`. `run` resolves
the suite into an execution manifest, builds `sweep_bench` and runs it pinned to
core 0 with `taskset` where it exists; a rerun with the same run ID reuses the
finished cells whose manifests match. The runner writes `run.json` and, per
cell, `manifest.json` and the `samples`, `groups`, `counters`, `hw` and `stages`
Parquet tables. `pack` then writes one file per table and one `cells.json` for the run
(the sentinel's cells under `sentinel/`), checks every table against its cells,
and with `--remove` deletes only the finished cell directories it packed and
verified, keeping unfinished cells, failed sentinel attempts and the run's
`failed/`; from output schema 6 it requires `stages.parquet`. `summarize`,
`validation-readout` and `budget` read packed runs. `run --set KEY=VALUE` overrides a `grids.toml [run]` setting, after
the suite's own `[suites.NAME.run]` settings (the validation suite's).

The protocol, per cell (settings in `grids.toml [run]`):

- Each suite declares its modes: serving (one call per group) everywhere,
  batch (calls of up to 1,024 rows) in the ablation suite as well.
- Methods run one at a time. The timed groups are split into 12 batches; a
  round gives each method its own phase, a warm-up on the batch it times last
  and then a timed pass over every batch, the same groups in the same order for
  every method. Method order alternates across rounds, forward then reversed
  (A B B A). A mode times at least 3 rounds, then stops at the 1% precision
  target or its time budget.
- A cell whose pool exceeds `max_rows_per_round` rows times a seeded sample of
  its groups, the same for every method: every group has the same inclusion
  probability, the cap's share of the pool's rows (the cap is the expected rows),
  raised to keep at least `min_groups_per_round` groups, stratified by group size.
  The manifest records the seed, the fraction, the drawn rows and their overshoot,
  where the minimum binds, the strata with their weights and a hash of the group
  indices.
- Baselines get their faster interface per cell and mode: in its first phase,
  each method's candidates (its multi-row call, or a loop over its single-row
  call) run on the warm-up batch, and the faster is timed. One-row serving calls
  use the single-row interface. The manifest records each probe.
- QuickScorer is excluded up front from LightGBM models with categorical
  splits, and from cells with missing values in a feature the model may route
  otherwise than left (a conservative rule: QuickScorer sends every missing value
  left). Every method receives f64 row-major input and converts it inside its
  timed call; QuickScorer's f64-to-f32 copy is part of its samples, and is also
  timed alone, right after its timed passes, so `summarize` reports its share.
- On x86_64, each XGBoost cell also times XGBoost alone in 6 extra one-round
  processes (`process` in the samples), because its AVX2 block walk depends on
  a heap buffer's alignment. XGBoost's main number is process 0's, timed and
  paired like every method. As a sensitivity row, `summarize` tells the modes
  apart inside each batch by instructions per row and reports the faster one
  over the batches holding both, with the expected time across processes beside
  it; the extra processes validate only their first 32 groups.
- Each cell, and each sentinel, runs in a new process of the runner
  (`fresh_process`). In one long-lived process, LightGBM native ran up to 6.9%
  slower by the end of an earlier factorial, with the same instructions: a model
  loaded into a glibc heap that hundreds of models had churned. A new process, or
  jemalloc, showed none.
- A cell with a row order, the kernel suite's single-row cells, is timed in its
  prepared order, independent rows (a fixed-seed shuffle) or file order, the same
  rows in the same order for every method; every other cell's groups are timed in
  a seeded order.
- A sentinel, SUPPORT's acceptance cell, runs at the start of each run
  invocation (a warm-up, then a measure) and after every 25 measured cells,
  under `sentinel/`, its ticks and rows in `run.json`. It measures the machine;
  `summarize` flags drift beyond 2% from the run's first measured sentinel and
  shows the first-load effect, and `numbers` reports each method's largest
  drift.

`compile-baselines` runs as many compiler processes at once as there are CPUs
(`--jobs`, at least 4), largest models first, tl2cgen at 2 threads per model and
lleaves at its recipe's 4 chunks. The chunk count shapes lleaves's library, so it
is part of the recipe: on the 32-vCPU Intel VM the acceptance and shakedown runs
compiled lleaves in 8 chunks, the final run in 4. Running the largest models first
puts the biggest compiles side by side; their peak memory is checked on the
validation deployment. The artifact tests run on whatever was prepared:
`cargo test --manifest-path experiments/benchmarks/Cargo.toml --release
--features research`.

tl2cgen comes from the locked `baselines` dependency group. It has no aarch64
Linux wheel, so there it builds from source with
`CXX=$PWD/infra/scripts/cxx-cstdint`. lleaves runs in
`experiments/compile.py`'s own script lock and compiles through the LLVM that
llvmlite bundles; the system C compiler links it. Every `uv run` passes
`--group baselines`, because `uv run` removes packages outside the groups it
syncs.

`uv.lock`, `experiments/compile.py.lock` and
`tests/fixtures/import/generate.py.lock` resolve against PyPI through the
`pypi` index that the project and both scripts declare, which takes priority
over any index in a user's uv configuration. Relock with `uv lock`,
`uv lock --script experiments/compile.py` and
`uv lock --script tests/fixtures/import/generate.py`; `uv lock --check` (with
`--script` for a script) verifies a lock without changing it.

### The camera-ready rerun

The camera-ready reruns the paper's cells with the algorithm as reviewed, on the
2.0 engine (the `neurips2026-final` tag): every TreeWalker forest loads with exact
sums off (`exact_sums = false`), so each row's leaves add in f64 in tree order, as
Algorithms 1 and 2 describe, and production `predict` equals `predict_full_walk`
bit for bit. 2.0's default was on (`LoadOptions::exact_sums`: the leaves add as
exact fixed-point integers rounded once), and since 3.0 every model sums exactly
(`docs/treelite-loading.md`), so this runner refuses `exact_sums = false`: the
commands below run at the tag. In the final run's
ablation suites (`ablation-{x86_64,aarch64}`, measured 2026-10-09/10, `numbers`
item README), turning exact sums off (`disable_exact_sums`, research build, serving,
T=500, L=8) took 1.26-1.29x the default's latency at G=128 on Intel and 1.59-1.60x
on Arm, 1.25-1.28x and 1.40-1.45x at G=64, 1.10-1.15x and 1.13-1.15x at G=16,
1.02-1.06x and 0.99-1.02x at G=4, and 0.98-0.99x and 0.96-0.97x on Expedia's
sessions, where exact sums are the slower. Each mode times at least 4
rounds (`min_rounds = 4`), two full forward-and-reversed order cycles. The suites,
in `grids.toml`, each set both in their own run table, so `factorial` and
`ablation` keep their defaults:

- `camera-ready`: survival panels at horizons 1-128 and FLCHAIN's 256-1,024 (128-row
  chunks beside whole groups), Expedia's sessions, the fixed-model cohorts of 4-32
  rows and `expedia-filled`.
- `camera-ready-ablation`: the ablation's anchors and switches without the
  `disable_exact_sums` variant, run twice as two runs.
- `scenario-v1`: E1, the released credit scenario cells, with the trace evaluator
  for Table 3's d_v columns.

`infra/camera-ready.tfvars` holds the campaign: acceptance, the ablation, the
camera-ready suite, the ablation again, `scenario-v1` and the layout check, every
run with `--set exact_sums=false`:

```bash
cd infra
terraform apply -var project=YOUR_PROJECT -var git_ref=$(git rev-parse 'neurips2026-final^{commit}') -var 'suites=["acceptance"]' -var layout_check=true -var 'run_settings=["exact_sums=false"]'
terraform apply -var project=YOUR_PROJECT -var git_ref=$(git rev-parse 'neurips2026-final^{commit}') -var-file=camera-ready.tfvars
```

The runs land as `camera-ready-ARCH`, `camera-ready-ablation-1-ARCH`,
`camera-ready-ablation-2-ARCH` and `scenario-v1-ARCH`. By hand, the same suites
run with `treewalker-exp run --suite NAME --run-id NAME-ARCH --set exact_sums=false`
(the second ablation run as `camera-ready-ablation-2-ARCH`). Then, per run:

```bash
uv run treewalker-exp pack experiments/data/runs/RUN
uv run treewalker-exp check-run experiments/data/runs/RUN ...   # gates and warnings
uv run treewalker-exp retime camera-ready                       # wide intervals, twice the rounds (below)
uv run treewalker-exp figures --campaign camera-ready           # and tables, numbers
uv run treewalker-exp audit-f32 --suite camera-ready            # f32 margins against the oracle
cargo run --release --manifest-path experiments/benchmarks/Cargo.toml --features external-bench \
  --bin sweep_bench -- precision-order --manifest MANIFEST --output order.json   # offline: ordering on vs off
```

`audit-f32 --suite camera-ready` reads TreeWalker's f32 margins from each run's
packed `stages.parquet` and compares them, and native XGBoost's, with the same f64
oracle. `sweep_bench precision-order`, a 2.0 command (at the tag; under 3.0's exact
sums tree order cannot change a sum), runs offline over prepared LightGBM cells
(cell directories, or `--manifest` with `--cells` and `--artifacts-dir`): with
exact sums off it compares tree ordering on against off on the tree sums and raw
margins, and both orderings' outputs with the cell's GTIL reference.

`check-run` exits non-zero when a validity gate fails, and fails closed: the
run's tables missing, a failed cell or sentinel attempt that no later attempt of
the same cell (in the order of `run.json`'s invocations, each failure in
`failed/` recording its invocation), or of the sentinel's role, resolved (resolved failures stay in the
data and are listed), a cell whose stage oracle
did not pass, a required method not timed, a timed interface without a passing
check on the timed rows, process 0's samples not covering the timed groups in
every round, or a forest whose `exact_sums` differs from the run's, which the
camera-ready suites must set off (pass `--exact-sums off` for the campaign's
acceptance and layout runs); from output schema 7 (3.0) every forest sums exactly,
no forest's record is written, and the gate passes when the setting is on. It prints the warnings (sentinel drift, the layout
check, G=1 at the survival anchors, E1 against the released numbers).

Re-timing runs on each machine, from that machine's own execution manifest (the
native libraries differ), for every run the paper reads: `camera-ready`,
`camera-ready-ablation-1`, `camera-ready-ablation-2` and `scenario-v1`:

```bash
for run in camera-ready camera-ready-ablation-1 camera-ready-ablation-2 scenario-v1; do
  uv run --group baselines treewalker-exp retime $run   # this machine; --manifest <artifacts>/manifests/SUITE.json
done
SYSTEM_INFO=PATH experiments/data/retime/RUN-retime-rN-ARCH.sh   # each script it writes
```

`retime` lists every cell and mode where a method's speedup over TreeWalker, or a
research variant's ablation effect over its own denominator (the research build's
all-on; a monotonic variant's precompute-off counterpart), has a paired interval
wider than ±2% of the ratio. It checks the manifest against the run's record
(suite, libraries, artifacts, source, settings, cell keys), and writes manifests
that time those cells again with twice the rounds, with a script per manifest
that runs it as the campaign does (`treewalker-exp retime-run`: core 0, the PMU
required, the system facts recorded, and refused on any VM but the parent run's)
into a run of its own, `RUN-retime-rN-ARCH`.
The analysis checks each attempt against its parent and reports each cell's
longer attempt; both stay in the data.

A campaign VM that stops is restarted by hand (`automatic_restart` is off), and
`startup.sh` then picks up where it was. It records every run it launches in
`~bench/runs-claimed.txt`; on a restart a claimed run resumes by run ID while it
is raw, whether or not an invocation of the runner on it finished (a run none
finished, `partial`, is never packed), one already packed is skipped (never timed
again, no sentinel appended, not packed twice), and a packed directory that also
holds raw cells, or an incomplete pack, stops the boot with nothing deleted
(`treewalker-exp run-state` names the state). An existing directory this
deployment never claimed is refused, and `run` and `retime-run` refuse a packed
run directory themselves. A new boot moves the previous boot's `DONE` or `ERROR`
into the results' `MARKER_HISTORY` and out of the bucket, so `check_done` reads
`RUNNING` until the boot ends. It ends with `DONE` only if nothing failed: a run
with failed cells, a pack or re-time error, or a claimed run that is not complete
at the end (`treewalker-exp run-complete`: packed, every cell of its manifest
finished, no failed cell or sentinel attempt unresolved, whichever boot ran it)
is recorded, the later runs still go, the results are uploaded, and the boot ends
with `ERROR` naming each failure. Re-time runs are packed only after the re-time
phase has resumed them.

On the campaign's VMs re-timing happens on its own: with `retime = true`
(`infra/camera-ready.tfvars`), `startup.sh` packs the suites, runs `retime` for
each of the paper's runs, runs every script it writes, packs the re-time runs and
uploads them with the rest (`retime/` holds the manifests and scripts). By hand,
on the VM that ran the campaign (as the bench user, from the repository):

```bash
for run in camera-ready camera-ready-ablation-1 camera-ready-ablation-2 scenario-v1; do
  uv run --group baselines treewalker-exp retime $run --out experiments/data/retime
done
for script in experiments/data/retime/*-$(uname -m).sh; do
  SYSTEM_INFO=$HOME/system_info.json sh "$script"   # retime-run: taskset -c 0, --require-pmu
done
for run in experiments/data/runs/*-retime-*/; do
  uv run --group baselines treewalker-exp pack "$run" --remove
done
gcloud storage cp -r experiments/data/runs experiments/data/retime gs://BUCKET/results/MACHINE/
```

Re-time only on the VM that ran the campaign, and keep it up until the re-times
finish. The analysis accepts an attempt only if it ran the parent's program on the
parent's protocol and VM: the same executed source and `sweep_bench` binary (by
hash), Rust toolchain and flags, pinned CPUs, PMU, turbo mode, machine type, image,
zone and instance (its ID and host name, which the camera-ready runs record),
native libraries and tl2cgen runtime, resolved run settings other than the rounds,
and per cell the same key, model, input and oracle hashes and compiled baselines.
A refused attempt leaves the original reported, marked "re-time refused", and
`tables` lists every attempt and why it was refused (App. A.4).

### What to expect from a rerun

Work counters are deterministic: the final run's counts are identical on both
machines and to an earlier run's, and on the paper's models they equal the
paper's (Table 1's FLCHAIN row, for one). Timings vary with the host: between
that earlier run and the final one, on fresh VMs of the same types, TreeWalker's
median time per row moved by 1.3% on Intel and 0.3% on Arm. Each cell stops at a
1% precision target, or its block cap or time budget, and reports bootstrap
intervals; the sentinel measures the machine's drift over the run, which
`numbers` reports (App. A.6). Exact values move between reruns.

## Correctness

A prediction has three stages: `tree_sum`, the sum of the row's leaf values;
`raw_margin`, `tree_sum / divisor + base_score`; and the output, after the
identity or the sigmoid. The leaves of every model are added as exact fixed-point
integers and rounded once: `tree_sum` is the `f64` nearest to the true sum, and
tree order, layout, lanes, group width, chunking, platform and every ablation leave
it unchanged. Leaves whose scaled sums fit in 126 bits add as `i128`; others, with
exponents too far apart, add in a wider accumulator of 29-bit digits. The margin and
the link are ordinary rounded `f64` operations.

LightGBM (f64) predictions match Treelite GTIL within 1e-13, and XGBoost (f32)
predictions match native XGBoost within 1e-5 (`experiments/benchmarks/tests/correctness.rs`);
`tree_sum` and `raw_margin` match `math.fsum` of GTIL's per-tree outputs bit for bit.
GTIL adds leaves in `f64` in tree order, so it differs from the exact sum by its
rounding, which grows with the partial sums.

The library's tests check production bit for bit against `predict_full_walk`, which
walks every tree for every row on its own and sums exactly, and against output
vectors recorded for the import fixtures and for generated models
(`tests/vectors.rs`). Property tests generate Treelite models, configurations and
groups across every mask width and piece boundary and compare production and the
full walk with an independent evaluator and a big-integer sum (`tests/property.rs`;
`TW_PROPTEST_CASES` widens the sweep). Lane counts, width dispatch, single rows
with and without the constant walk, and chunk sizes are checked to give identical
outputs.

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

MIT, see [LICENSE](LICENSE). This covers the code and the released
measurements; the datasets keep their own terms (see Datasets).
