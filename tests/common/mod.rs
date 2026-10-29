//! Helpers for the property tests: a seeded generator of Treelite models and their
//! inputs, an evaluator independent of the library, and an exact sum.
#![allow(dead_code, reason = "each test crate uses a part")]

pub mod model;
pub mod oracle;
pub mod rng;
