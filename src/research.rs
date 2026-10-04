//! Research API, behind the `research` feature: runtime ablations, work counters,
//! the stages of a prediction, the reference full walk and model introspection.
//!
//! None of it is needed to serve predictions. It exists to measure and check the
//! production path:
//!
//! ```
//! # use treewalker_gbdt::{Forest, ModelFormat, LoadOptions, WalkerConfig};
//! use treewalker_gbdt::research::{Ablation, Stages};
//! # let model = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/import/identity.bin")).unwrap();
//! # let config = WalkerConfig::builder(2).max_group_width(4).varying([0]).build().unwrap();
//! # let forest = Forest::from_bytes(&model, ModelFormat::TreeliteBinaryV4, config, &LoadOptions::default()).unwrap();
//! let rows = [0.5, 1.0, 1.5, 1.0]; // two rows of two features
//! let mut out = [0.0; 2];
//!
//! let mut r = forest.research_predictor(Ablation { disable_unsplit: true, ..Default::default() });
//! r.predict_group(&rows, &mut out); // timed: the ablation flags, no counters
//! let counters = r.predict_group_counted(&rows, &mut out); // the same flags, counted
//! assert_eq!(counters.unsplit_skips, 0);
//!
//! let mut stages = Stages::default();
//! r.predict_group_stages(&rows, &mut stages); // tree_sum, raw_margin, output per row
//! assert_eq!(stages.output, out);
//!
//! let mut reference = [0.0; 2];
//! forest.predict_full_walk(&rows, &mut reference); // every tree, row by row
//! ```

use crate::forest::{Forest, Model};
use crate::predict::{Groups, Predictor};

pub use crate::forest::{Node, ThresholdType, Tree};
pub use crate::predict::{Ablation, WorkCounters};

/// The three stages of a prediction, one value per row.
///
/// Production prediction returns `output`. Keeping the stages apart shows where a
/// value was rounded: adding the base score inside the sum would round differently.
/// For example `fsum([1, 2^-53]) + 2^-53` is 1.0, while `fsum([1, 2^-53, 2^-53])` is
/// 1.0000000000000002.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stages {
    /// The sum of the row's leaf values, correctly rounded when the model supports
    /// exact sums and the variant keeps them; otherwise added in `f64` in tree order.
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
/// runtime ablation flags of an [`Ablation`].
///
/// The flags are fixed when the predictor is created, and every call runs with them.
/// [`Self::predict_group`] is the timed build, without counters;
/// [`Self::predict_group_counted`] is the counted build. With every flag off, the
/// research builds compute what production does, but are separate code.
#[derive(Debug)]
pub struct ResearchPredictor {
    inner: Predictor,
    variant: Ablation,
}

impl Forest {
    /// Create a [`ResearchPredictor`] that runs with `ablation`.
    #[must_use]
    pub fn research_predictor(&self, ablation: Ablation) -> ResearchPredictor {
        ResearchPredictor {
            inner: self.predictor(),
            variant: ablation,
        }
    }

    /// Predict every row on its own: every tree is walked to its leaf for each row,
    /// leaves are added in `f64` in tree order, and the sum is finalized as production
    /// does. Rows need not form a group. A reference for the partial evaluation, not a
    /// serving path.
    ///
    /// # Panics
    ///
    /// If `rows.len()` is not a multiple of `n_features` or `out.len()` is not the row
    /// count.
    pub fn predict_full_walk(&self, rows: &[f64], out: &mut [f64]) {
        let nf = self.model.config.n_features();
        assert!(
            rows.len().is_multiple_of(nf),
            "input length {} is not a multiple of n_features {nf}",
            rows.len()
        );
        assert!(
            out.len() == rows.len() / nf,
            "output length {} differs from the row count {}",
            out.len(),
            rows.len() / nf
        );
        self.model.full_walk(rows, out);
    }

    /// All trees' nodes, in the optimized layout.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.model.nodes
    }

    /// Per-tree offsets into [`Self::nodes`].
    #[must_use]
    pub fn trees(&self) -> &[Tree] {
        &self.model.trees
    }

    /// One tree's nodes.
    #[must_use]
    pub fn tree_nodes(&self, tree: &Tree) -> &[Node] {
        &self.model.nodes[tree.node_start as usize..(tree.node_start + tree.node_count) as usize]
    }

    /// Bytes of categorical bitsets in the shared pool.
    #[must_use]
    pub fn bitset_bytes(&self) -> usize {
        self.model.bitsets.len()
    }

    /// Whether thresholds compare in `f32` (XGBoost) or `f64` (LightGBM).
    #[must_use]
    pub fn threshold_type(&self) -> ThresholdType {
        self.model.threshold_type
    }

    /// Left masks of every varying predicate over one group of at most 32 rows, from
    /// the production threshold sweep (`sweep = true`) or the brute force that
    /// [`Ablation::disable_predicate_sweep`] selects. Bit `r` is set when row `r` goes
    /// left.
    ///
    /// # Panics
    ///
    /// If `rows.len()` is not a multiple of `n_features` or holds more than 32 rows.
    #[must_use]
    pub fn predicate_masks(&self, rows: &[f64], sweep: bool) -> Vec<u32> {
        let nf = self.model.config.n_features();
        assert!(
            rows.len().is_multiple_of(nf) && rows.len() / nf <= 32,
            "predicate_masks takes up to 32 rows of n_features values"
        );
        self.model.predicate_masks(rows, sweep)
    }
}

impl ResearchPredictor {
    /// The runtime ablation flags every call runs with.
    #[must_use]
    pub const fn ablation(&self) -> Ablation {
        self.variant
    }

    /// Predict one group with the ablation flags: the research timed build, without
    /// counters. Same contract as [`Predictor::predict_group`].
    ///
    /// # Panics
    ///
    /// As [`Predictor::predict_group`].
    pub fn predict_group(&mut self, rows: &[f64], out: &mut [f64]) {
        self.inner
            .run::<false, true>(rows, out, Groups::One, self.variant, None);
    }

    /// Predict one group with the ablation flags and count its work: the research
    /// counted build. The outputs equal [`Self::predict_group`]'s.
    ///
    /// # Panics
    ///
    /// As [`Predictor::predict_group`].
    pub fn predict_group_counted(&mut self, rows: &[f64], out: &mut [f64]) -> WorkCounters {
        let mut counters = WorkCounters::default();
        self.inner
            .run::<true, true>(rows, out, Groups::One, self.variant, Some(&mut counters));
        counters
    }

    /// Predict one group with the ablation flags, keeping every stage. Each of
    /// `stages`' vectors is resized to the row count; `stages.output` equals what
    /// [`Self::predict_group`] writes.
    ///
    /// # Panics
    ///
    /// As [`Predictor::predict_group`].
    pub fn predict_group_stages(&mut self, rows: &[f64], stages: &mut Stages) {
        let n = rows.len() / self.inner.model.config.n_features();
        stages.tree_sum.resize(n, 0.0);
        self.inner.check(rows, &stages.tree_sum, Groups::One);
        self.inner.run_unchecked::<false, true>(
            rows,
            &mut stages.tree_sum,
            Groups::One,
            self.variant,
            None,
            false,
        );
        let model: &Model = &self.inner.model;
        stages.raw_margin.clone_from(&stages.tree_sum);
        model.apply_margin(&mut stages.raw_margin);
        stages.output.clone_from(&stages.raw_margin);
        model.apply_link(&mut stages.output);
    }
}

impl Model {
    /// Every tree walked for every row on its own, leaves added in `f64` in tree
    /// order, then finalized.
    fn full_walk(&self, rows: &[f64], out: &mut [f64]) {
        match self.threshold_type {
            ThresholdType::F64 => self.full_walk_inner::<false>(rows, out),
            ThresholdType::F32 => self.full_walk_inner::<true>(rows, out),
        }
    }

    fn full_walk_inner<const F32: bool>(&self, rows: &[f64], out: &mut [f64]) {
        let nf = self.config.n_features();
        let nodes = &self.nodes;
        out.fill(0.0);
        for tree in &self.trees {
            let base = tree.node_start as usize;
            for (res, features) in out.iter_mut().zip(rows.chunks_exact(nf)) {
                let mut local = 0usize;
                while !nodes[base + local].is_leaf() {
                    local = self.step::<F32>(nodes, base + local, local, features);
                }
                *res += nodes[base + local].value;
            }
        }
        self.apply_margin(out);
        self.apply_link(out);
    }

    fn predicate_masks(&self, rows: &[f64], sweep: bool) -> Vec<u32> {
        let n = rows.len() / self.config.n_features();
        let mut out = vec![0u32; self.varying_predicates.len()];
        let (mut column, mut order) = (vec![0.0; n], vec![(0.0, 0); n]);
        match (self.threshold_type, sweep) {
            (ThresholdType::F64, true) => {
                self.precompute_varying_masks::<false, u32>(
                    rows,
                    &mut column,
                    &mut order,
                    &mut out,
                );
            }
            (ThresholdType::F32, true) => {
                self.precompute_varying_masks::<true, u32>(rows, &mut column, &mut order, &mut out);
            }
            (ThresholdType::F64, false) => {
                self.precompute_bruteforce_generic::<false, u32>(rows, &mut out);
            }
            (ThresholdType::F32, false) => {
                self.precompute_bruteforce_generic::<true, u32>(rows, &mut out);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::Ablation;
    use crate::{Forest, LoadOptions, ModelFormat, WalkerConfig};

    fn forest() -> Forest {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/import/sigmoid_f64.bin"
        );
        let config = WalkerConfig::builder(2)
            .max_group_width(40)
            .varying([0])
            .build()
            .unwrap();
        Forest::from_bytes(
            &std::fs::read(path).unwrap(),
            ModelFormat::TreeliteBinaryV4,
            config,
            &LoadOptions::default(),
        )
        .unwrap()
    }

    /// Every combination of the five runtime flags.
    fn variants() -> impl Iterator<Item = Ablation> {
        (0..32u8).map(|bits| Ablation {
            disable_varying_precompute: bits & 1 != 0,
            disable_predicate_sweep: bits & 2 != 0,
            disable_unsplit: bits & 4 != 0,
            disable_monotonic: bits & 8 != 0,
            disable_exact_sums: bits & 16 != 0,
        })
    }

    #[test]
    fn timed_counted_and_staged_calls_receive_the_predictor_variant() {
        // Finding 0.1 timed predict() with the flags set on the side: the timed path
        // never saw them. Here every entry point must pass the predictor's own flags
        // down to the call path.
        let forest = forest();
        let rows: Vec<f64> = (0..33).flat_map(|r| [f64::from(r) / 8.0, 0.5]).collect();
        let mut out = vec![0.0; 33];
        for variant in variants() {
            let mut r = forest.research_predictor(variant);
            assert_eq!(r.ablation(), variant);
            assert_eq!(r.inner.last_variant(), None);
            r.predict_group(&rows, &mut out);
            assert_eq!(r.inner.last_variant(), Some(variant), "timed");
            r.inner.clear_last_variant();
            r.predict_group_counted(&rows, &mut out);
            assert_eq!(r.inner.last_variant(), Some(variant), "counted");
            r.inner.clear_last_variant();
            r.predict_group_stages(&rows, &mut super::Stages::default());
            assert_eq!(r.inner.last_variant(), Some(variant), "stages");
        }
        let mut p = forest.predictor();
        p.predict_group(&rows, &mut out);
        assert_eq!(p.last_variant(), Some(Ablation::default()), "production");
    }
}
