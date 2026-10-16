//! Row masks: bit `r` is set when row `r` of the group is active.
//!
//! `u32` and `u64` masks are carried by value, [`Wide`] masks of 128, 256, 512 and
//! 1,024 bits by reference into an arena, and [`rows`] masks of a runtime number of
//! 128-bit units likewise. Wide operations use NEON on AArch64,
//! SSE2 on x86-64 (AVX2 when the build enables it) and plain words on other targets.
//! The plain-word version is the reference the tests check the others against.

#[cfg(target_arch = "aarch64")]
mod neon;
mod ops;
pub mod rows;
#[cfg(any(test, not(any(target_arch = "aarch64", target_arch = "x86_64"))))]
mod scalar;
#[cfg(test)]
mod tests;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
use neon as simd;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
use scalar as simd;
#[cfg(target_arch = "x86_64")]
use x86 as simd;

pub use ops::{Mask, Wide};
