# treewalker

## What this codebase does

`treewalker-gbdt`, the root crate in `src/`, is a Rust library. It loads a trained
gradient-boosted tree ensemble (Treelite binary or JSON checkpoint, LightGBM text) and
predicts groups of rows that share most feature values. It is published to crates.io.
Services embed the published crate: they load a saved layout file (`.twl`) with
`Forest::map_layout` or `Forest::load_layout`, or a source model with `Forest::load`,
and score with `Predictor::predict_groups`. The library has no network, HTTP, user or
auth surface. The rest of the repository is the NeurIPS 2026 paper's tooling, which no
service runs: `experiments/benchmarks/` (the unpublished `treewalker-bench` crate and
`sweep_bench`), `experiments/treewalker_exp/` (Python: datasets, training, compiled
baselines, figures), `experiments/layout-file/`, and `infra/` (Terraform and a startup
script for two GCE benchmark VMs).

## Auth shape

There is no authentication. These are the trust boundaries:

- **Import caps**: `LoadOptions` (`max_binary_bytes`, `max_binary_array_bytes`,
  `max_json_bytes`, `max_json_nesting`, `max_config_bytes`, `max_rank_table_bytes`)
  bound model input. Past a cap, a load returns `LoadError::Limit`.
- **Validation before unchecked code**: `load/parse/validation.rs` and
  `layout_file/validate.rs` check the invariants that the `unsafe` and `get_unchecked`
  kernels in `predict/` (`resolve.rs`, `walk.rs`, `precompute.rs`, `leaves.rs`,
  `mask/x86.rs`, `mask/neon.rs`) and `exact/` rely on. Bad input is a `LoadError`
  (`MalformedModel`, `MalformedConfig`, `Unsupported`, `Limit`, `Layout`, `LayoutMismatch`).
- **Layout file header and checksum**: the `TWLAYOUT` magic, schema version and target,
  then a non-cryptographic `Checksum` (`layout_file/codec.rs`) that detects corruption
  only. Neither authenticates a file (`docs/layout-file.md`); the embedding service must.
- **`unsafe fn Forest::map_layout`**: the caller keeps the mapped file immutable
  (`layout_file/storage.rs`).
- **Calling contract**: a `Predictor` call panics, before it writes output, on wrong
  input or output lengths or bad group offsets. Group equality is the caller's contract.

## Threat model

1. Memory unsafety: a model or layout file that passes validation but breaks an
   invariant that an unchecked kernel assumes (out-of-bounds access, uninitialized
   read, an ISA path that runs an instruction the CPU lacks). The library runs inside
   production services, so this matters most.
2. Wrong predictions from a file that loads (exact sums, precision, categorical chains):
   a service shows them to its users.
3. The release path: `release.yml` checks and packages a tag, and a maintainer runs
   `cargo publish` locally. No registry token is in the repository or CI.

The authors accept a crash, an allocation abort or wrong output from a deliberately
crafted model: whoever ships a model already controls its output. Report resource
exhaustion only when a file that a correct writer produces triggers it.

## Project-specific patterns to flag

- A `// SAFETY:` comment that cites a caller guarantee that no loader check establishes,
  for example the slot and mask headers in `predict/resolve.rs`.
- A count read from a file that sizes a buffer before it is checked against the
  remaining bytes or a `LoadOptions` cap (`load/parse/binary.rs`,
  `load/parse/lightgbm.rs`, `layout_file/codec.rs`).
- A new layout-file field that reaches the kernel with no check in
  `layout_file/validate.rs`.
- `TREEWALKER_ISA` (`predict/isa.rs`) may only lower the level that CPU detection
  reports. Flag any path where an environment value raises it.
- `.github/workflows/`: `permissions` wider than `contents: read`,
  `persist-credentials: true`, an action not pinned to a SHA, or a secret in a job that
  builds pull-request code.

## Known false-positives

- `experiments/**` and `infra/**` run only by hand, on the operator's own machine or
  benchmark VMs, with inputs that the operator chose. The authors accept these by
  design: `cell.json` names the library that `sweep_bench` loads, `driver/processes.rs`
  uses `/tmp` paths, `infra/scripts/startup.sh` runs the rustup and uv installers, and
  `asm_identity.py` builds a revision that the operator names. Report there only a path
  that reaches the library or the release.
- The `research` feature (`src/research/`, `ResearchPredictor`, ablations) is test and
  paper instrumentation; production builds do not enable it.
- `tests/fixtures/**` and `experiments/layout-file/tests/fixtures/**` hold committed
  models and layout files.
- `build.rs` hashes `src/` into `TREEWALKER_BUILD_ID`, a reproducibility tag, not a
  security control.
