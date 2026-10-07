//! TreeWalker's benchmark runner.
//!
//! `treewalker-exp manifest` resolves a suite from `experiments/grids.toml` into an
//! execution manifest; `sweep_bench run` executes it, one cell at a time, into a run
//! directory of JSON and Parquet files. The runner has no grid logic: suites,
//! datasets, workloads and variants come from the manifest.
//!
//! - [`artifacts`] loads the manifest and each cell's `cell.json`, shared with the
//!   tests.
//! - [`driver`] validates, counts and times one cell; [`suites`] says which methods
//!   a cell's suite times; [`methods`] and [`external`] hold TreeWalker's builds and
//!   the baseline adapters.
//! - [`schedule`] plans the batches and the method order, [`timer`] reads the
//!   timestamp counter, [`pmu`] the hardware counters.
//! - [`output`] writes the run directory, [`sentinel`] times the drift cell.

pub mod artifacts;
pub mod data;
pub mod driver;
pub mod external;
pub mod methods;
pub mod output;
pub mod pmu;
pub mod run;
pub mod schedule;
pub mod sentinel;
pub mod suites;
pub mod system;
pub mod timer;

pub use data::{load_group_offsets, load_raw_f64, try_load_raw_f64};
