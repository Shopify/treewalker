//! Row masks: bit `r` is set when row `r` of the group is active.
//!
//! `u32` and `u64` masks are carried by value, [`Wide`] masks of 128, 256, 512 and
//! 1,024 bits by reference into an arena, and [`rows`] masks of a runtime number of
//! 128-bit units likewise. Wide operations use plain words on every target for now;
//! the plain-word version is the reference the tests check the others against.

mod ops;
pub mod rows;
mod scalar;
#[cfg(test)]
mod tests;

use scalar as simd;

pub use ops::{Mask, Wide};
