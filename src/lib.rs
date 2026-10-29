//! `TreeWalker`: high-performance tree ensemble inference for grouped prediction tasks.
//!
//! # Problem structure
//!
//! Grouped prediction tasks expand each entity into `max_group_width` rows and predict
//! a per-row scalar output. This creates a specific data structure that tree inference
//! can exploit:
//!
//! - **Constant features** (~78% of splits) are identical across all rows of an
//!   entity. Tree splits on these features are evaluated once, not per-row.
//!   This is the single biggest win: it turns O(trees × rows × depth) into
//!   O(trees × depth) for the constant portion of each tree.
//!
//! - **Varying features** may differ per row. Splits on these require
//!   partitioning the row set. Each varying feature's rows are sorted once per
//!   group to build its splits' row masks; a **monotonic** feature (e.g., elapsed
//!   time increases, remaining time decreases) arrives sorted, which the sort
//!   finishes in one pass. ~88% of varying splits are monotonic.
//!
//! `TreeWalker` walks each tree once per group. At load, every feature's thresholds
//! become integer bins and the nodes a 16-byte layout; a group's constant features
//! are binned once, and the walk steps several trees at a time without
//! data-dependent branches. Splits on varying features are resolved afterwards
//! with row masks (32 and 64 bits by value, wider in an arena: up to 1,024 bits at a
//! width fixed at compile time, then at the group's width); a single row steps
//! through them as constants and needs no masks.
//!
//! # Usage
//!
//! Load a model once, then predict through a [`Predictor`], one per worker thread. A
//! [`Forest`] is a cheap handle to the immutable model: clone it to share it. A
//! predictor owns the scratch space for the widest group, so its calls never allocate.
//! The scratch is allocated when the predictor is created: the varying predicates'
//! masks at the widest width the configuration allows, and the walk queues, stops
//! and mask arena of one chunk of trees. Chunks are cut so that a chunk's working set
//! fits one CPU's share of the L2 cache at its group's mask width, so the scratch
//! scales with a chunk, not the model. The kernel's constants (lanes, chunk budget,
//! prefetch distance, prefix groups) are derived from the target, the caches and a
//! one-time latency probe when the first predictor is created, and from the forest
//! at load; nothing is tuned per model by the caller.
//!
//! ```no_run
//! use treewalker_gbdt::{Forest, LoadError};
//!
//! # fn main() -> Result<(), LoadError> {
//! let forest = Forest::load("model.bin", "walker_config.json")?;
//! let mut predictor = forest.predictor();
//!
//! // Two entities with 3 and 2 rows, row-major, in the model's trained column order.
//! let n_features = forest.config().n_features();
//! let data = vec![0.0; 5 * n_features];
//! let mut out = vec![0.0; 5];
//! predictor.predict_groups(&data, &[0, 3, 5], &mut out);
//!
//! // One group, or consecutive groups of a fixed width.
//! predictor.predict_group(&data[..3 * n_features], &mut out[..3]);
//! predictor.predict_fixed(&data, 3, &mut out); // groups of 3 and 2 rows
//! # Ok(())
//! # }
//! ```
//!
//! Without a `walker_config.json`, declare the feature roles with
//! [`WalkerConfig::builder`] and load with [`Forest::from_bytes`] or
//! [`Forest::from_reader`].
//!
//! Supports scalar regression, ranking and binary classification with identity or
//! sigmoid output, including averaging, base scores and positive sigmoid alpha.
//! Import errors are returned as [`LoadError`]. Inputs have 1 to `i32::MAX` features,
//! and groups any number of rows up to the configured maximum. Trees, nodes, depth,
//! varying splits and category IDs have no fixed limit: untrusted input is bounded by
//! the caps of [`LoadOptions`], which can be raised for trusted models. Group equality is a caller
//! contract, and production's outputs do not depend on monotonicity; loading validates the configuration, and each call checks its
//! dimensions and grouping before it writes any output. A row's leaf values are
//! summed exactly and rounded once, for every model, so its tree sum is the float64
//! nearest to the true sum whatever the tree order; averaging, the base score and the
//! link function then round as ordinary float64 operations. Float32
//! models round inputs for comparisons and promote leaves to float64. See the bundled
//! `docs/treelite-loading.md` for the full compatibility boundary, memory limits and
//! conversion examples.
//!
//! The `research` feature adds the `research` module: runtime ablations, work
//! counters and phase times, the stages of a prediction, a reference full walk and
//! model introspection. Serving needs none of it.

#![deny(clippy::undocumented_unsafe_blocks)]

// The crate follows the pipeline. The top-level files are the public API (`Forest`,
// `Predictor`, `WalkerConfig` and `LoadOptions`, `LoadError`) and the loaded model
// they share (`model`); each directory is a stage or a concern of its own:
//
// - `load`: a Treelite model decoded (`load::parse`), its trees ordered, its varying
//   predicates found and the prediction layout built (`load::layout`);
// - `predict`: the kernel, stage by stage (precompute, walk, resolve, leaf pass,
//   finish), over the row masks of `predict::mask`;
// - `exact`: exact sums of leaf values, in `i128` or a wide accumulator;
// - `rules`: the kernel's constants, derived from the forest and from the machine
//   (`rules::machine`);
// - `research`: the research API, behind the `research` feature.

mod config;
mod error;
mod exact;
mod forest;

mod load;
mod model;
mod predict;
mod predictor;
#[cfg(feature = "research")]
pub mod research;
mod rules;

pub use config::{LoadOptions, WalkerConfig, WalkerConfigBuilder};
pub use error::LoadError;
pub use forest::Forest;
pub use load::parse::ModelFormat;
pub use predictor::Predictor;

// The README's and the loading guide's Rust examples compile as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
#[cfg(doctest)]
#[doc = include_str!("../docs/treelite-loading.md")]
struct LoadingGuideDoctests;

const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    const fn send<T: Send>() {}
    send_sync::<Forest>();
    send::<Predictor>();
    #[cfg(feature = "research")]
    send::<research::ResearchPredictor>();
};
