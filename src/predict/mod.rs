//! The new kernel, stage by stage, as it lands.

#![expect(
    clippy::inline_always,
    reason = "keeps the production call path independent of the research builds"
)]

mod counters;
pub(crate) mod mask;
mod precompute;
#[cfg(test)]
mod tests;

pub(crate) use counters::Stats;
pub use counters::WorkCounters;
#[cfg(feature = "research")]
pub use counters::{COUNTERS_VERSION, Phases};
