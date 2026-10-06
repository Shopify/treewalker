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
//!   partitioning the row set. When a varying feature is **monotonic** across the
//!   panel (e.g., elapsed time increases, remaining time decreases), the
//!   partition reduces to a prefix or suffix scan — no per-row evaluation.
//!   ~88% of varying splits are monotonic.
//!
//! `TreeWalker` implements *recursive partial evaluation* that walks constant
//! splits once, then partitions rows only at varying splits using generic
//! bitmasks (u16/u32/u64, and multi-word masks for wider groups).
//!
//! # Usage
//!
//! Load a model once, then predict through a [`Predictor`], one per worker thread. A
//! [`Forest`] is a cheap handle to the immutable model: clone it to share it. A
//! predictor owns the scratch space for the widest group, so its calls never allocate.
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
//! Import errors are returned as [`LoadError`]. Inputs have 1–64 features and groups
//! any number of rows up to the configured maximum. Group equality and monotonicity are
//! caller contracts; loading validates the configuration, and each call checks its
//! dimensions and grouping before it writes any output. When [`Forest::exact_sums`]
//! is true, a row's leaf values are summed exactly and rounded once, so its tree sum
//! is the float64 nearest to the true sum whatever the tree order; averaging, the base
//! score and the link function then round as ordinary float64 operations. Float32
//! models round inputs for comparisons and promote leaves to float64. See the bundled
//! `docs/treelite-loading.md` for the full compatibility boundary, memory limits and
//! conversion examples.
//!
//! The `research` feature adds the `research` module: runtime ablations, work
//! counters, the stages of a prediction, a reference full walk and model
//! introspection. Serving needs none of it.

#![deny(clippy::undocumented_unsafe_blocks)]

mod config;
mod error;
mod exact;
mod forest;
mod mask;
mod parser;
mod predict;
#[cfg(feature = "research")]
pub mod research;

pub use config::{LoadOptions, WalkerConfig, WalkerConfigBuilder};
pub use error::LoadError;
pub use forest::Forest;
pub use parser::ModelFormat;
pub use predict::Predictor;

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
