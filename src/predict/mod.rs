//! Prediction: the kernel behind [`Predictor`], stage by stage.
//!
//! A call checks its input once, borrows the model once and dispatches once on the
//! threshold type. Each group then runs whole ([`kernel`]), at the narrowest mask
//! width that holds it, with scratch allocated when the predictor was created
//! ([`scratch`]):
//!
//! 1. **Bins and prefix starts**: the group's first row is binned once, since its
//!    constant features are equal in every row, and each prefix group's shared
//!    splits are evaluated once to find where its trees start.
//! 2. **Precompute** ([`precompute`]): every varying predicate's left and right row
//!    masks.
//! 3. **Walk** ([`walk`]): per chunk of trees, lanes step constant splits without
//!    branches until each item stops at a leaf or a varying split.
//! 4. **Resolve** ([`resolve`]): leaves become leaf entries, and each varying stop's
//!    rows split by its heavy side's mask into the next pass's items. Walk and
//!    resolve passes alternate until no item is left ([`passes`]).
//! 5. **Leaf pass** ([`leaves`]): each leaf entry's value is added to its rows' runs
//!    in exact difference arrays.
//! 6. **Finish** ([`finish`]): each row's exact sum is rounded once, then averaged,
//!    offset by the base score and linked.
//!
//! A single row's bins hold every feature, so its walk steps through varying splits
//! as constants: one pass per chunk, without masks, items or difference arrays.
//! Row masks ([`mask`]) are `u32` or `u64` by value, and wider ones in an arena.
//! [`counters`] holds the work counters and phase times of the counted build, and
//! [`scan`] research's varying events that scan their rows instead of reading the
//! precomputed masks.
//!
//! The kernel is generic over `const STATS: bool` (work counters and phase times)
//! and `const ABLATE: bool` (runtime ablations). Its walks take their lanes and
//! prefetch distance as const parameters, chosen per walk by the rules
//! ([`crate::rules`]); production runs `(false, false)` with the rules' choices, and
//! the research predictor runs timed `(false, true)` and counted `(true, true)`
//! builds, which can force them.
//!
//! Compiling or using the research feature must not change production code. The
//! helpers both builds call are `#[inline(always)]`: the input check, the threshold
//! sweep and the finalization. The kernel's own functions take `ABLATE`, so research
//! never instantiates production's copies, which the research crate would otherwise
//! compile in another context and LTO could keep instead of production's. And
//! production's passes take a scan without data ([`scan::NoScan`]), so LLVM cannot
//! merge them with research's equivalent passes. The language does not guarantee the
//! result; by disassembly with rustc 1.97.1 and fat LTO on aarch64-apple-darwin
//! (apple-m4) and x86_64-unknown-linux-gnu (emeraldrapids), 2026-10-08, every kernel
//! function is identical with the feature off, compiled in and in use. Only the
//! model's drop glue differs, as it frees the research-only parsed nodes. A compiler
//! upgrade should repeat that check.

#![expect(
    clippy::inline_always,
    reason = "keeps the production call path independent of the research builds"
)]

mod counters;
mod finish;
mod isa;
mod kernel;
mod leaves;
pub(crate) mod mask;
mod passes;
mod precompute;
mod resolve;
mod scan;
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
use isa::Native;
pub(crate) use isa::{Level, level};

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
        macro_rules! at {
            ($x:ty) => {
                match model.threshold_type {
                    ThresholdType::F64 => model.kernel_groups_at::<$x, false, STATS, ABLATE>(
                        data, out, groups, s, options, stats, finalize,
                    ),
                    ThresholdType::F32 => model.kernel_groups_at::<$x, true, STATS, ABLATE>(
                        data, out, groups, s, options, stats, finalize,
                    ),
                }
            };
        }
        match self.level {
            Level::Native => at!(Native),
            #[cfg(dispatch)]
            Level::V3 => at!(isa::V3),
            #[cfg(dispatch)]
            Level::V4 => at!(isa::V4),
        }
    }
}
