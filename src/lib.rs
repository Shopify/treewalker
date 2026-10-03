//! `TreeWalker`: high-performance tree ensemble inference for grouped prediction tasks.
//!
//! # Problem structure
//!
//! Grouped prediction tasks expand each entity into `max_group_width` rows and predict
//! a per-row probability. This creates a specific data structure that tree inference
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
//! ```ignore
//! use treewalker::forest::Forest;
//!
//! // Accepts .json or .bin (treelite binary v4, preferred for large models)
//! let mut forest = Forest::load("model_treelite.bin", "walker_config.json");
//!
//! // data: row-major f64 array, rows = entities × max_group_width, cols = features
//! // results: one probability per row
//! let mut results = vec![0.0f64; n_rows];
//! let h = forest.config.max_group_width;
//! for obs in 0..n_entities {
//!     forest.predict(&data, &mut results, obs * h, (obs + 1) * h);
//! }
//! ```

pub mod config;
pub mod mask;
pub mod forest;
pub mod parser;
pub mod predict;

pub use config::{AblationMode, ParseConfig};
pub use predict::PredictStats;
