//! References for the kernel: the full walk of every tree for every row, and the
//! varying predicates' masks by the sweep or by brute force.

use super::{ThresholdType, WorkCounters};
use crate::exact::{self, Plan};
use crate::forest::Forest;
use crate::model::Model;

impl Forest {
    /// Predict every row on its own: every tree is walked to its leaf for each row,
    /// the leaf values are summed exactly and rounded once, and the sum is finalized
    /// as production does. Rows need not form a group. The tree sum is the `f64`
    /// nearest to the true sum, so production matches it bit for bit. A reference
    /// for the kernel, not a serving path; it allocates per call.
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

    /// Left masks of every varying predicate over one group of at most 32 rows, from
    /// the production threshold sweep (`sweep = true`) or the brute force that
    /// [`Ablation::disable_predicate_sweep`](super::Ablation::disable_predicate_sweep) selects. Bit `r` is set when row `r` goes
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

impl Model {
    /// Every tree walked for every row on its own, the leaves summed exactly with
    /// the model's [`Plan`] and rounded once, then finalized.
    fn full_walk(&self, rows: &[f64], out: &mut [f64]) {
        match self.plan {
            Plan::Fixed(e) => {
                let mut sums = vec![0i128; out.len()];
                self.walk_leaves(rows, |r, v| sums[r] += exact::to_fixed(v, e));
                for (o, &s) in out.iter_mut().zip(&sums) {
                    *o = exact::to_f64(s, e);
                }
            }
            Plan::Wide(wide) => {
                let n = wide.limbs();
                let mut sums = vec![0i64; out.len() * n];
                self.walk_leaves(rows, |r, v| wide.add(&mut sums[r * n..(r + 1) * n], v));
                for (o, s) in out.iter_mut().zip(sums.chunks_exact(n)) {
                    *o = wide.to_f64(s);
                }
            }
        }
        self.apply_margin(out);
        self.apply_link(out);
    }

    /// Call `leaf(row, value)` with the leaf each row reaches in each tree.
    fn walk_leaves(&self, rows: &[f64], leaf: impl FnMut(usize, f64)) {
        match self.threshold_type {
            ThresholdType::F64 => self.walk_leaves_inner::<false>(rows, leaf),
            ThresholdType::F32 => self.walk_leaves_inner::<true>(rows, leaf),
        }
    }

    /// [`Self::walk_leaves`] at the precision of `F32`.
    fn walk_leaves_inner<const F32: bool>(&self, rows: &[f64], mut leaf: impl FnMut(usize, f64)) {
        let nf = self.config.n_features();
        let nodes = &self.nodes;
        for tree in &self.trees {
            let base = tree.node_start as usize;
            for (r, features) in rows.chunks_exact(nf).enumerate() {
                let mut local = 0usize;
                while !nodes[base + local].is_leaf() {
                    local = self.step::<F32>(nodes, base + local, local, features);
                }
                leaf(r, nodes[base + local].value);
            }
        }
    }

    /// [`Forest::predicate_masks`]: the left masks of the sweep or the brute force,
    /// for up to 32 rows.
    fn predicate_masks(&self, rows: &[f64], sweep: bool) -> Vec<u32> {
        let n = rows.len() / self.config.n_features();
        let mut sides = vec![[0u32; 2]; self.varying_predicates.len()];
        let (mut order, mut codes) = (vec![[0; 2]; 2 * n], vec![0; n]);
        let counters = &mut WorkCounters::default();
        let (out, left, missing) = (&mut sides[..], &mut 0u32, &mut 0u32);
        match (self.threshold_type, sweep) {
            (ThresholdType::F64, true) => self.precompute_sweep::<false, u32, _, false>(
                rows,
                &mut order,
                &mut codes,
                out,
                &mut [0][..],
                missing,
                counters,
            ),
            (ThresholdType::F32, true) => self.precompute_sweep::<true, u32, _, false>(
                rows,
                &mut order,
                &mut codes,
                out,
                &mut [0][..],
                missing,
                counters,
            ),
            (ThresholdType::F64, false) => {
                self.precompute_brute_force::<false, u32, _, false>(rows, out, left, counters);
            }
            (ThresholdType::F32, false) => {
                self.precompute_brute_force::<true, u32, _, false>(rows, out, left, counters);
            }
        }
        sides.iter().map(|&[left, _]| left).collect()
    }
}
