# Treelite loading validation — 2026-10-03

Implementation: `b2378db`, based directly on `dev` at `05d9316`.
The loader branch excludes the constant-hoisting experiment. Checks below were
rerun after separating the branches; historical performance measurements are
labeled separately. These are not updates to the archived paper results.

## Delivered behavior

The existing Rust readers now validate a scalar Treelite subset and return
`LoadError` from file, reader, and byte APIs. Binary import remains streaming;
Python and libtreelite are export/test tools only. The five-field grouping
schema is unchanged. Configuration validation, bounded tree/array decoding,
topology checks, and prediction dimension/mutation guards protect the optimized
traversal. Nodes remain 16 bytes.

Scalar classification, regression, and ranking support identity or sigmoid,
positive sigmoid alpha, summed or averaged tree output, and a base score added
after aggregation. Float32 comparisons retain input rounding and float64
accumulation. All prediction paths share output finalization. The historical
panic loaders remain wrappers; the legacy parser tuple explicitly refuses
output metadata it cannot represent.

Intentional rejections include multiclass/multitarget output, vector leaves
(including singleton vectors and wildcard output assignments), unsupported
operators/postprocessors/types, unknown binary versions/extensions, invalid
JSON, nonfinite leaves/base scores, NaN thresholds, and float64 `< -infinity`.
The supported version range is 4.0–4.7; committed independent fixtures exercise
4.7.0. This is not a claim of complete Treelite compatibility or exhaustive
testing of every producer release. See [the loading guide](treelite-loading.md)
for the exact APIs, precision policy, limits, and grouping responsibilities.

Categorical import also fixes membership-right and out-of-range behavior:
non-NaN values outside the category domain take the non-membership branch,
while NaN follows missing routing. Both directions and float32 categories have
independent GTIL references.

## Checks run

All commands below passed with Rust 1.94.1. Offline runs used dependencies
already present in the local Cargo cache.

| Check | Result |
|---|---|
| `cargo test --release --offline` | 14 unit, 11 import tests; 1 compiling doctest |
| `cargo fmt --check` | Passed |
| `cargo clippy --offline --all-targets -- -D warnings` | Passed |
| `cargo build --offline --release --manifest-path benchmarks/Cargo.toml` | Passed |
| Benchmark artifact `correctness` suite, `--features test-helpers` | 18 tests passed |
| Benchmark artifact `parse_test` suite | 6 tests passed |
| `cargo package --list --allow-dirty --offline` | Library tests, fixtures, and loading documentation included |
| `git diff --check` | Passed |

The committed fixtures include 13 supported models and four unsupported real
Treelite exports. Their optional generator pins Treelite 4.7.0, NumPy 2.2.6,
sklearn 1.6.1, and SciPy 1.15.3. GTIL supplies independent reference predictions,
with a native sklearn reference for random-forest regression and a
hand-computable average-plus-base case. Normal Rust tests need none of these
Python packages. Coverage includes both encodings, all prediction paths and
workspace widths, numeric/category boundaries, malformed topology and metadata,
every-byte truncation of a fixture, and 2,000 deterministic binary mutations.

Artifact checks used FLCHAIN and METABRIC `nt250_md8_h16` models from LightGBM
and XGBoost. Local configuration copies used the current five-field schema;
binary checkpoints were exported with Treelite 4.7.0. The original artifacts
were unchanged. The full experiment grid was not rerun. Existing artifact
tolerances remain 1e-14 for float64/native LightGBM and 1e-5 for native XGBoost.
`TEST_ARTIFACTS_BASE` selects the correctness-suite artifact root;
`TEST_ARTIFACTS` selects the parse-suite cell.

## Earlier execution and import measurements

These measurements predate branch separation: loader implementation `60a141e`
was compared with `722b6ab` on the experimental branch. They are retained as
historical evidence, not measurements of the corrected loader branch. The
`hoist_bench` driver belongs to those revisions and is not included here.
Performance has not been remeasured after removing the experiment.

Host: Apple M3 Pro, arm64 macOS. Both revisions used Rust 1.94.1, native CPU,
release fat LTO, and one codegen unit. The fixed workload was the FLCHAIN
LightGBM `nt250_md8_h16` binary: 250 trees, 58,138 nodes, 1,304 groups of 16 rows.
Hoisting was disabled for the reported method; tree ordering and prefix depth 2
were enabled.

The historical `hoist_bench` protocol ran 24 blocks across 12 batches, with three
warmup passes. Five serial process repetitions alternated revision order.
Reported values are medians across the five repetitions. Load time is the first
forest load in each process. Peak RSS comes from macOS `/usr/bin/time -l` and
covers the whole benchmark process, including data, both off/on forests and
workspace; it is not isolated loader memory.

| Metric | Baseline | New loader | Change |
|---|---:|---:|---:|
| Group prediction, separate executables | 14.000 µs | 14.958 µs | +6.8% |
| Load and lower model | 2.043 ms | 2.696 ms | +32.0% (+0.653 ms) |
| Whole-process peak RSS | 10,469,376 B | 11,059,200 B | +589,824 B |
| Compact node/category/tree pools | 933,208 B | 933,208 B | Unchanged |

Separate-executable group medians in run order:

- Baseline: 13.500, 13.583, 14.000, 14.541, 14.292 µs.
- New loader: 14.958, 14.708, 15.167, 14.875, 15.584 µs.

For reproduction, build `hoist_bench` at the two historical revisions with identical flags,
retain each executable, then run each on the same binary/config/data with
`--min-blocks 24 --max-blocks 24`. Use the `baseline` entry in its JSON output.
Run processes serially, alternate revision order, and keep compilation and
other CPU-heavy work outside the timing interval.

The separate-executable prediction slowdown was investigated with a disposable
driver linking both revisions into one process. It checked predictions before
timing, used the same `bench_blocked` harness, rotated method order per block,
and ran 48 blocks across 12 batches for five serial process repetitions.
The median of the five paired new/baseline ratios was 1.008 with default layout,
and 1.003 with tree ordering and prefix sharing disabled. An earlier paired
series gave 1.014 and 0.992 respectively. Predictions agreed exactly on this
workload. Earlier separate-executable series ranged from roughly neutral to
10% slower.

This establishes additional import cost and build/process-sensitive prediction
timings. The paired comparison does not prove unchanged deployment latency or
explain the separate-executable slowdown completely: linking both revisions
also changes compilation and memory layout. Retain the separate-executable
regression as a performance caveat and benchmark the actual serving build.
Validation was retained; two avoidable buffer costs were removed by keeping
scalar binary reads off the reusable array buffer and reusing topology scratch.
