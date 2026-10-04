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
//! ```no_run
//! use treewalker_gbdt::{Forest, LoadError};
//!
//! # fn main() -> Result<(), LoadError> {
//! let forest = Forest::try_load("model.bin", "walker_config.json")?;
//! let mut predictor = forest.predictor(); // reuse it: one per worker
//! let rows = vec![0.0; forest.config().n_features]; // one row, trained column order
//! let mut output = [0.0];
//! predictor.predict_group(&rows, &mut output);
//! # Ok(())
//! # }
//! ```
//!
//! Supports scalar regression, ranking and binary classification with identity or
//! sigmoid output, including averaging, base scores and positive sigmoid alpha.
//! Import errors are returned as [`LoadError`]. Inputs have 1–64 features and groups
//! any number of rows. Group equality and monotonicity are caller contracts; loading
//! validates the configuration, and prediction guards dimensions and structural
//! configuration mutations. Leaf values are summed exactly and rounded once, so a
//! prediction is the float64 nearest to the true sum, whatever the tree order (see
//! [`Forest::exact_sums`]). Float32 models round inputs for comparisons and promote
//! leaves to float64. See the bundled `docs/treelite-loading.md` for the
//! full compatibility boundary, memory limits and conversion examples.

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

pub use config::{ParseConfig, WalkerConfig};
pub use error::LoadError;
pub use forest::Forest;
pub use parser::ModelFormat;
pub use predict::Predictor;

const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    const fn send<T: Send>() {}
    send_sync::<Forest>();
    send::<Predictor>();
    #[cfg(feature = "research")]
    send::<research::ResearchPredictor>();
};
