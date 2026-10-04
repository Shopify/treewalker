//! The per-node kernel: prefix starts, the constant walk and recursive partial
//! evaluation, and the sorted-threshold sweep that precomputes varying masks.

use super::{EvalCtx, PredictStats};
use crate::config::AblationMode;
use crate::forest::{
    Forest, Node, PrefixGroup, SPLIT_MONO_DEC, SPLIT_MONO_INC, VaryingPredicate, threshold_go_left,
};
use crate::mask::RowMask;

impl Forest {
    // -----------------------------------------------------------------------
    // Prefix group evaluation (mask-agnostic)
    // -----------------------------------------------------------------------

    pub(super) fn precompute_prefix_starts<const F32: bool, const STATS: bool>(
        &self,
        const_features: &[f64],
        stats: &mut Option<PredictStats>,
        group: &PrefixGroup,
        k: usize,
        prefix_starts: &mut [u16],
    ) {
        let rep_base = group.node_base as usize;
        let nodes = &self.nodes;
        let mut bail_level = k;
        for j in 0..k {
            // SAFETY: prefix groups hold only trees whose first k heavy-path nodes are
            // constant splits, so rep_base + j with j < k lies in the representative tree.
            let node = unsafe { nodes.get_unchecked(rep_base + j) };
            if STATS && let Some(s) = stats {
                s.constant_steps += 1;
            }
            let go_left = self.eval_split::<F32>(node, const_features);
            if go_left != node.heavy_is_left() {
                bail_level = j;
                break;
            }
        }
        for &tree_idx in &group.trees {
            let start_idx = if bail_level == k {
                k as u16
            } else {
                let tree_base = self.trees[tree_idx as usize].node_start as usize;
                // SAFETY: as above, for this tree, with bail_level < k.
                unsafe { nodes.get_unchecked(tree_base + bail_level) }.skip as u16
            };
            prefix_starts[tree_idx as usize] = start_idx;
        }
    }

    // -----------------------------------------------------------------------
    // Node step / eval
    // -----------------------------------------------------------------------

    #[inline]
    pub(super) fn step<const F32: bool>(
        &self,
        nodes: &[Node],
        abs_idx: usize,
        local_idx: usize,
        features: &[f64],
    ) -> usize {
        let node = &nodes[abs_idx];
        if self.eval_split::<F32>(node, features) == node.heavy_is_left() {
            local_idx + 1
        } else {
            node.skip as usize
        }
    }

    #[inline]
    pub(super) fn eval_split<const F32: bool>(&self, node: &Node, features: &[f64]) -> bool {
        // SAFETY: the parser rejects split features >= n_features, and every caller
        // passes one row of n_features values.
        let val = unsafe { *features.get_unchecked(node.feature as usize) };
        if val.is_nan() {
            node.default_left()
        } else if node.is_categorical() {
            let val = if F32 { f64::from(val as f32) } else { val };
            val >= 0.0 && self.cat_test(node, val as i32)
        } else {
            threshold_go_left::<F32>(val, node.value)
        }
    }

    // -----------------------------------------------------------------------
    // Partial eval — ablation and stats only when STATS
    // -----------------------------------------------------------------------

    pub(super) fn partial_eval<const F32: bool, M: RowMask, const STATS: bool>(
        &self,
        ctx: &mut EvalCtx<M>,
        mut idx: usize,
        mut row_mask: M,
    ) {
        if row_mask.is_zero() {
            return;
        }
        let nodes = self.nodes.as_slice();
        let base = ctx.base;
        let ablation = if STATS {
            ctx.ablation
        } else {
            AblationMode::default()
        };
        let use_precompute = !ablation.disable_varying_precompute;
        let use_unsplit = !ablation.disable_unsplit;
        let use_mono = !ablation.disable_monotonic || use_precompute;

        loop {
            // Constant walk — single bit test per node.
            // SAFETY: base is a tree's node_start and idx a tree-local index reached by
            // fall-through or skip; validation keeps both inside the tree.
            while unsafe { nodes.get_unchecked(base + idx) }.is_walkable() {
                if STATS && let Some(ref mut s) = ctx.stats {
                    s.constant_steps += 1;
                }
                idx = self.step::<F32>(nodes, base + idx, idx, ctx.const_features);
            }

            // SAFETY: as above.
            let node = unsafe { nodes.get_unchecked(base + idx) };

            // Leaf (walked past all constant nodes, could be leaf or varying).
            if node.is_leaf() {
                if STATS && let Some(ref mut s) = ctx.stats {
                    s.leaf_hits += 1;
                }
                if let Some(e) = ctx.scale {
                    let x = crate::exact::to_fixed(node.value, e);
                    let diff = &mut *ctx.diff;
                    // SAFETY: diff has n + 1 entries, and runs lie within rows 0..n.
                    row_mask.for_each_run(|a, b| unsafe {
                        *diff.get_unchecked_mut(a) += x;
                        *diff.get_unchecked_mut(b) -= x;
                    });
                } else {
                    let val = node.value;
                    let results = &mut ctx.results[ctx.start..];
                    // SAFETY: set bits are rows of this piece, below n, and check_input
                    // asserts that results holds the piece's rows from start on.
                    row_mask.for_each_bit(|r| unsafe { *results.get_unchecked_mut(r) += val });
                }
                return;
            }

            // Varying split.
            if STATS && let Some(ref mut s) = ctx.stats {
                s.varying_splits += 1;
            }

            let (left_mask, right_mask) = if use_precompute {
                let pred_id = node.varying_pred_id as usize;
                // SAFETY: a varying node's varying_pred_id indexes varying_predicates,
                // and the mask table holds one mask per predicate.
                let pred_left_mask = unsafe { *ctx.pred_left_masks.get_unchecked(pred_id) };
                (row_mask & pred_left_mask, row_mask & !pred_left_mask)
            } else {
                let (rows, nf, feat) = (ctx.rows, ctx.n_features, node.feature as usize);
                let col = |r: usize| rows[r * nf + feat];
                let result = if node.is_categorical() {
                    self.partition_per_row::<F32, M>(node, rows, nf, row_mask)
                } else if use_mono {
                    match node.varying_type() {
                        SPLIT_MONO_INC => Self::partition_mono_inc::<F32, M>(node, col, row_mask),
                        SPLIT_MONO_DEC => Self::partition_mono_dec::<F32, M>(node, col, row_mask),
                        _ => Self::partition_non_mono::<F32, M>(node, col, row_mask),
                    }
                } else {
                    Self::partition_non_mono::<F32, M>(node, col, row_mask)
                };
                if STATS && let Some(ref mut s) = ctx.stats {
                    s.partition_row_evals += u64::from(row_mask.count_ones());
                }
                result
            };

            let (heavy_idx, light_idx) = (idx + 1, node.skip as usize);
            let (heavy_mask, light_mask) = if node.heavy_is_left() {
                (left_mask, right_mask)
            } else {
                (right_mask, left_mask)
            };

            // Unsplit optimization.
            if use_unsplit {
                if light_mask.is_zero() {
                    if STATS && let Some(ref mut s) = ctx.stats {
                        s.unsplit_skips += 1;
                    }
                    idx = heavy_idx;
                    continue;
                }
                if heavy_mask.is_zero() {
                    if STATS && let Some(ref mut s) = ctx.stats {
                        s.unsplit_skips += 1;
                    }
                    idx = light_idx;
                    continue;
                }
            }

            if STATS && let Some(ref mut s) = ctx.stats {
                s.recursive_calls += 1;
            }
            self.partial_eval::<F32, M, STATS>(ctx, light_idx, light_mask);
            idx = heavy_idx;
            row_mask = heavy_mask;
        }
    }

    // -----------------------------------------------------------------------
    // Predicate precompute — sorted-threshold sweep
    // -----------------------------------------------------------------------

    /// Left masks of every varying predicate for one piece: per feature, sort the
    /// rows by value once and sweep the feature's sorted thresholds.
    #[inline]
    pub(super) fn precompute_varying_masks<const F32: bool, M: RowMask>(
        &self,
        rows: &[f64],
        column: &mut [f64],
        order: &mut [(f64, u16)],
        out_left_masks: &mut [M],
    ) {
        debug_assert_eq!(out_left_masks.len(), self.varying_predicates.len());
        let nf = self.config.n_features;
        let column = &mut column[..rows.len() / nf];

        for range in &self.feature_ranges {
            let f = range.feature as usize;
            for (v, row) in column.iter_mut().zip(rows.chunks_exact(nf)) {
                *v = row[f];
            }

            if range.num_start < range.num_end {
                let mut nan_mask = M::ZERO;
                let mut n_non_nan = 0usize;
                for (r, &val) in column.iter().enumerate() {
                    if val.is_nan() {
                        nan_mask = nan_mask.set_bit(r);
                    } else {
                        order[n_non_nan] = (val, r as u16);
                        n_non_nan += 1;
                    }
                }

                let sorted = &mut order[..n_non_nan];
                if F32 {
                    sorted.sort_unstable_by(|a, b| (a.0 as f32).total_cmp(&(b.0 as f32)));
                } else {
                    sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                }

                let mut row_ptr = 0usize;
                let mut non_nan_left = M::ZERO;
                let preds =
                    &self.varying_predicates[range.num_start as usize..range.num_end as usize];
                let masks = &mut out_left_masks[range.num_start as usize..range.num_end as usize];

                for (out, pred) in masks.iter_mut().zip(preds.iter()) {
                    let (threshold, default_left) = match pred {
                        VaryingPredicate::Num {
                            threshold,
                            default_left,
                            ..
                        } => (threshold.0, *default_left),
                        _ => unreachable!(),
                    };
                    while row_ptr < n_non_nan {
                        // SAFETY: row_ptr < n_non_nan, the length of sorted.
                        let (val, r) = unsafe { *sorted.get_unchecked(row_ptr) };
                        if threshold_go_left::<F32>(val, threshold) {
                            non_nan_left = non_nan_left.set_bit(usize::from(r));
                            row_ptr += 1;
                        } else {
                            break;
                        }
                    }
                    *out = non_nan_left | if default_left { nan_mask } else { M::ZERO };
                }
            }

            let cat_preds =
                &self.varying_predicates[range.cat_start as usize..range.cat_end as usize];
            let cat_masks = &mut out_left_masks[range.cat_start as usize..range.cat_end as usize];
            for (out, pred) in cat_masks.iter_mut().zip(cat_preds.iter()) {
                let mut left_mask = M::ZERO;
                for (r, &val) in column.iter().enumerate() {
                    if pred.goes_left::<F32>(val, &self.bitsets) {
                        left_mask = left_mask.set_bit(r);
                    }
                }
                *out = left_mask;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::rows_from_columns;
    use crate::config::{AblationMode, WalkerConfig};
    use crate::forest::{FeatureRange, Forest, ThresholdType, VaryingPredicate};

    // --- Sweep tests ---

    fn sweep_forest(preds: Vec<VaryingPredicate>, ranges: Vec<FeatureRange>) -> Forest {
        Forest {
            output: crate::parser::Output {
                base_score: 0.0,
                divisor: 1.0,
                postprocessor: crate::parser::Postprocessor::Sigmoid(1.0),
            },
            compiled_config: WalkerConfig::try_new(64, 32, &(0..64).collect::<Vec<_>>(), &[], &[])
                .unwrap(),
            trees: Vec::new(),
            config: WalkerConfig {
                n_features: 64,
                max_group_width: 32,
                varying_mask: u128::from(u64::MAX),
                mono_inc_mask: 0,
                mono_dec_mask: 0,
                ablation: AblationMode::default(),
            },
            nodes: Vec::new(),
            bitsets: Vec::new(),
            varying_predicates: preds,
            feature_ranges: ranges,
            threshold_type: ThresholdType::F64,
            prefix_groups: Vec::new(),
            prefix_depth: 0,
            fixed_scale: None,
            workspace: None,
        }
    }

    fn num_pred(feature: u16, threshold: f64, default_left: bool) -> VaryingPredicate {
        VaryingPredicate::Num {
            feature,
            threshold: crate::forest::Threshold(threshold),
            default_left,
        }
    }

    pub(super) fn sweep<const F32: bool>(
        forest: &Forest,
        cols: &[[f64; 32]; 64],
        n_rows: usize,
    ) -> Vec<u32> {
        let rows = rows_from_columns(cols, n_rows, forest.config.n_features);
        let mut out = vec![0u32; forest.varying_predicates.len()];
        let (mut column, mut order) = (vec![0.0; n_rows], vec![(0.0, 0); n_rows]);
        forest.precompute_varying_masks::<F32, u32>(&rows, &mut column, &mut order, &mut out);
        out
    }

    fn run_sweep(forest: &Forest, cols: &[[f64; 32]; 64], n_rows: usize) -> Vec<u32> {
        sweep::<false>(forest, cols, n_rows)
    }

    #[test]
    fn test_sweep_all_nan_column() {
        let forest = sweep_forest(
            vec![num_pred(0, 5.0, true), num_pred(0, 10.0, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 2,
                cat_start: 2,
                cat_end: 2,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        for r in 0..4 {
            cols[0][r] = f64::NAN;
        }
        let masks = run_sweep(&forest, &cols, 4);
        assert_eq!(masks[0], 0b1111);
        assert_eq!(masks[1], 0b0000);
    }

    #[test]
    fn test_sweep_single_row() {
        let forest = sweep_forest(
            vec![num_pred(0, 3.0, false), num_pred(0, 7.0, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 2,
                cat_start: 2,
                cat_end: 2,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        cols[0][0] = 5.0;
        let masks = run_sweep(&forest, &cols, 1);
        assert_eq!(masks[0], 0b0);
        assert_eq!(masks[1], 0b1);
    }

    #[test]
    fn test_sweep_all_rows_identical() {
        let forest = sweep_forest(
            vec![num_pred(0, 3.0, false), num_pred(0, 7.0, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 2,
                cat_start: 2,
                cat_end: 2,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        for r in 0..8 {
            cols[0][r] = 5.0;
        }
        let masks = run_sweep(&forest, &cols, 8);
        assert_eq!(masks[0], 0);
        assert_eq!(masks[1], 0b1111_1111);
    }

    #[test]
    fn test_sweep_max_group_width() {
        let forest = sweep_forest(
            vec![num_pred(0, 16.0, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 1,
                cat_start: 1,
                cat_end: 1,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        for r in 0..32 {
            cols[0][r] = r as f64;
        }
        let masks = run_sweep(&forest, &cols, 32);
        let expected: u32 = (0..=16).fold(0u32, |m, r| m | (1 << r));
        assert_eq!(masks[0], expected);
    }

    #[test]
    fn test_sweep_mixed_nan_and_values() {
        let forest = sweep_forest(
            vec![num_pred(0, 2.0, true), num_pred(0, 4.0, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 2,
                cat_start: 2,
                cat_end: 2,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        cols[0][0] = f64::NAN;
        cols[0][1] = 1.0;
        cols[0][2] = f64::NAN;
        cols[0][3] = 5.0;
        cols[0][4] = 3.0;
        let masks = run_sweep(&forest, &cols, 5);
        assert_eq!(masks[0], 0b00111);
        assert_eq!(masks[1], 0b10010);
    }

    #[test]
    fn test_sweep_no_numerical_predicates() {
        let forest = sweep_forest(
            vec![VaryingPredicate::CatInline {
                feature: 0,
                default_left: false,
                word: 0b1010,
            }],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 0,
                cat_start: 0,
                cat_end: 1,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        cols[0][0] = 0.0;
        cols[0][1] = 1.0;
        cols[0][2] = 3.0;
        let masks = run_sweep(&forest, &cols, 3);
        assert_eq!(masks[0], 0b110);
    }

    #[test]
    fn test_sweep_f32_threshold_collisions() {
        let t1: f64 = 1.0 + f64::from(2.0f32.powi(-24));
        let t2: f64 = 1.0 + f64::from(2.0f32.powi(-25));
        assert_eq!(
            (t1 as f32).to_bits(),
            (t2 as f32).to_bits(),
            "precondition: same f32"
        );
        let forest = sweep_forest(
            vec![num_pred(0, t1, false), num_pred(0, t2, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 2,
                cat_start: 2,
                cat_end: 2,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        cols[0][0] = 1.0;
        cols[0][1] = 1.000_000_15;
        let out = sweep::<true>(&forest, &cols, 2);
        assert_eq!(out[0], out[1]);
    }
}
