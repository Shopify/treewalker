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
//! bitmasks (u32/u64/u128, supporting up to 128 rows per entity).
//!
//! # Usage
//!
//! ```no_run
//! use treewalker_gbdt::{Forest, LoadError};
//!
//! # fn main() -> Result<(), LoadError> {
//! let mut forest = Forest::try_load("model.bin", "walker_config.json")?;
//! let data = vec![0.0; forest.config.n_features]; // one row, trained column order
//! let mut output = [0.0];
//! forest.predict(&data, &mut output, 0, 1);
//! # Ok(())
//! # }
//! ```
//!
//! Supports scalar regression, ranking and binary classification with identity or
//! sigmoid output, including averaging, base scores and positive sigmoid alpha.
//! Import errors are returned as [`LoadError`]. Inputs have 1–64 features and groups
//! have 1–128 rows. Group equality and monotonicity are caller contracts; loading
//! validates the configuration, and prediction guards dimensions and structural
//! configuration mutations. Float32 models round inputs for comparisons and sum
//! promoted leaves in float64. See the bundled `docs/treelite-loading.md` for the
//! full compatibility boundary, memory limits and conversion examples.

mod error;
pub use error::LoadError;
pub mod config;
pub mod forest;
pub mod mask;
pub mod parser;
pub mod predict;

pub use config::{AblationMode, ParseConfig, WalkerConfig};
pub use forest::Forest;
pub use parser::ModelFormat;
pub use predict::PredictStats;
