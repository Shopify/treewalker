//! Helpers for the property tests: a seeded generator of Treelite models and their
//! inputs, an evaluator independent of the library, and an exact sum; and for every
//! test, the import fixtures, groups of rows and bit-for-bit comparisons.
#![allow(dead_code, reason = "each test crate uses a part")]

pub mod bits;
pub mod fixtures;
pub mod groups;
pub mod json;
pub mod model;
pub mod oracle;
pub mod rng;

// The library's configuration, which `groups` builds.
use treewalker_gbdt::WalkerConfig;
