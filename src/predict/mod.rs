//! The new kernel, stage by stage, as it lands.

#![expect(
    clippy::inline_always,
    reason = "keeps the production call path independent of the research builds"
)]

mod counters;
mod finish;
mod kernel;
pub(crate) mod mask;
mod passes;
mod precompute;
mod scratch;
#[cfg(test)]
mod tests;
mod walk;

pub(crate) use counters::Stats;
pub use counters::WorkCounters;
#[cfg(feature = "research")]
pub use counters::{COUNTERS_VERSION, Phases};

use crate::model::{Model, ThresholdType};
use crate::rules::Tuning;

pub(crate) use crate::predictor::{Groups, Predictor};
pub(crate) use kernel::Options;

/// A predictor's scratch, with the rules' choices for the model on this machine.
pub(crate) struct Workspace {
    kernel: scratch::Scratch,
    /// What the uncounted kernel (`STATS = false`) takes and never writes, kept here
    /// so that a call does not zero a block of counters it never reads.
    idle: Stats,
}

impl Workspace {
    pub(crate) fn new(model: &Model) -> Self {
        Self::with_tuning(model, Tuning::new(model))
    }

    fn with_tuning(model: &Model, tuning: Tuning) -> Self {
        Self {
            kernel: scratch::Scratch::new(model, tuning),
            idle: Stats::default(),
        }
    }
}

impl Predictor {
    /// Run a call's groups through the kernel. With `STATS`, the work and phase
    /// times are added to `stats`, which then must be given; without, the kernel
    /// takes the workspace's idle one, which it never writes. With `ABLATE`, `options`
    /// apply. The input must have passed [`Self::check`]. Without `finalize`, `out`
    /// receives the tree sums.
    pub(crate) fn kernel<const STATS: bool, const ABLATE: bool>(
        &mut self,
        data: &[f64],
        out: &mut [f64],
        groups: Groups<'_>,
        options: Options,
        stats: Option<&mut Stats>,
        finalize: bool,
    ) {
        let model: &Model = &self.model;
        let Workspace { kernel: s, idle } = &mut self.workspace;
        debug_assert!(STATS == stats.is_some(), "a counted call gives its stats");
        let stats = stats.unwrap_or(idle);
        match model.threshold_type {
            ThresholdType::F64 => model.kernel_groups::<false, STATS, ABLATE>(
                data, out, groups, s, options, stats, finalize,
            ),
            ThresholdType::F32 => model.kernel_groups::<true, STATS, ABLATE>(
                data, out, groups, s, options, stats, finalize,
            ),
        }
    }
}
