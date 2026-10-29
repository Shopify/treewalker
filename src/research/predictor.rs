//! The research predictor: production's kernel with runtime ablations, timed, counted, or
//! keeping each stage of a prediction.

use super::ablation::valid_lanes;
use super::{Ablation, Phases, WorkCounters};
use crate::forest::Forest;
use crate::model::Model;
use crate::predict::{Groups, Predictor, Stats};
use crate::rules::MAX_LANES;

/// The three stages of a prediction, one value per row.
///
/// Production prediction returns `output`. Keeping the stages apart shows where a
/// value was rounded: adding the base score inside the sum would round differently.
/// For example `fsum([1, 2^-53]) + 2^-53` is 1.0, while `fsum([1, 2^-53, 2^-53])` is
/// 1.0000000000000002.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stages {
    /// The sum of the row's leaf values, correctly rounded.
    pub tree_sum: Vec<f64>,
    /// Production's staged finalization, rounded at each step:
    /// `tree_sum / divisor + base_score`. Averaging models divide by the tree count;
    /// others by 1.
    pub raw_margin: Vec<f64>,
    /// The link function applied to `raw_margin`: identity, or a sigmoid with the
    /// model's alpha.
    pub output: Vec<f64>,
}

/// A predictor that runs one variant of the predict path: production with the
/// runtime ablations of an [`Ablation`].
///
/// The ablation is fixed when the predictor is created, and every call runs with it.
/// [`Self::predict_group`] and [`Self::predict_groups`] are the timed build, without
/// counters; [`Self::predict_group_counted`] is the counted build, which also adds
/// its phase times to [`Self::take_phases`]. With the default ablation, the research
/// builds compute what production does, but are separate code.
#[derive(Debug)]
pub struct ResearchPredictor {
    inner: Predictor,
    variant: Ablation,
    phases: Phases,
}

impl Forest {
    /// Create a [`ResearchPredictor`] that runs with `ablation`.
    ///
    /// # Panics
    ///
    /// If `ablation.lanes` is not 1 to 12.
    #[must_use]
    pub fn research_predictor(&self, ablation: Ablation) -> ResearchPredictor {
        assert!(
            ablation.lanes.is_none_or(valid_lanes),
            "lanes must be 1 to {MAX_LANES}, got {:?}",
            ablation.lanes
        );
        ResearchPredictor {
            inner: self.predictor(),
            variant: ablation,
            phases: Phases::default(),
        }
    }
}

impl ResearchPredictor {
    /// The instruction-set level the kernel runs at: on x86-64 builds without AVX2 and
    /// FMA, `base`, `v3` or `v4`, the highest the CPU has unless `TREEWALKER_ISA` asks for
    /// a lower one; in other builds `native`, the build's own and only level.
    #[must_use]
    pub const fn isa(&self) -> &'static str {
        self.inner.level.name()
    }

    /// The runtime ablation every call runs with.
    #[must_use]
    pub const fn ablation(&self) -> Ablation {
        self.variant
    }

    /// Run checked groups with the ablation, counted with `STATS`.
    fn run<const STATS: bool>(
        &mut self,
        data: &[f64],
        out: &mut [f64],
        groups: Groups<'_>,
        stats: Option<&mut Stats>,
        finalize: bool,
    ) {
        let options = self.variant.options();
        self.inner
            .kernel::<STATS, true>(data, out, groups, options, stats, finalize);
    }

    /// Predict one group with the ablation: the research timed build, without
    /// counters. Same contract as [`Predictor::predict_group`].
    ///
    /// # Panics
    ///
    /// As [`Predictor::predict_group`].
    pub fn predict_group(&mut self, rows: &[f64], out: &mut [f64]) {
        self.inner.check(rows, out.len(), Groups::One);
        self.run::<false>(rows, out, Groups::One, None, true);
    }

    /// Predict consecutive groups of varying width with the ablation: the research
    /// timed build. Same contract as [`Predictor::predict_groups`].
    ///
    /// # Panics
    ///
    /// As [`Predictor::predict_groups`].
    pub fn predict_groups(&mut self, data: &[f64], offsets: &[usize], out: &mut [f64]) {
        let groups = Groups::Offsets(offsets);
        self.inner.check(data, out.len(), groups);
        self.run::<false>(data, out, groups, None, true);
    }

    /// Predict one group with the ablation and count its work: the research counted
    /// build. The outputs equal [`Self::predict_group`]'s, and its phase times are
    /// added to [`Self::take_phases`].
    ///
    /// # Panics
    ///
    /// As [`Predictor::predict_group`].
    pub fn predict_group_counted(&mut self, rows: &[f64], out: &mut [f64]) -> WorkCounters {
        self.inner.check(rows, out.len(), Groups::One);
        let mut stats = Stats::default();
        self.run::<true>(rows, out, Groups::One, Some(&mut stats), true);
        self.phases += stats.phases;
        stats.counters
    }

    /// The phase times of the counted calls since the last call, which it resets.
    pub fn take_phases(&mut self) -> Phases {
        std::mem::take(&mut self.phases)
    }

    /// Predict one group with the ablation, keeping every stage. Each of `stages`'
    /// vectors is resized to the row count; `stages.output` equals what
    /// [`Self::predict_group`] writes.
    ///
    /// # Panics
    ///
    /// As [`Predictor::predict_group`].
    pub fn predict_group_stages(&mut self, rows: &[f64], stages: &mut Stages) {
        let n = rows.len() / self.inner.model.config.n_features();
        self.inner.check(rows, n, Groups::One);
        stages.tree_sum.resize(n, 0.0);
        let mut tree_sum = std::mem::take(&mut stages.tree_sum);
        self.run::<false>(rows, &mut tree_sum, Groups::One, None, false);
        stages.tree_sum = tree_sum;
        let model: &Model = &self.inner.model;
        stages.raw_margin.clone_from(&stages.tree_sum);
        model.apply_margin(&mut stages.raw_margin);
        stages.output.clone_from(&stages.raw_margin);
        model.apply_link(&mut stages.output);
    }
}
