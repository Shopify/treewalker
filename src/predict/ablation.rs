//! Ablation-only paths: brute-force mask precompute and the per-row partitions
//! used when precompute is disabled.

use super::WorkCounters;
use crate::forest::{Model, Node, threshold_go_left};
use crate::mask::RowMask;

impl Model {
    /// Brute-force O(P × n) precompute — evaluates each predicate against every row.
    /// Used as the ablation baseline when `disable_predicate_sweep` is set.
    pub(crate) fn precompute_bruteforce_generic<const F32: bool, M: RowMask, const STATS: bool>(
        &self,
        rows: &[f64],
        out_left_masks: &mut [M],
        counters: &mut WorkCounters,
    ) {
        debug_assert_eq!(out_left_masks.len(), self.varying_predicates.len());
        let nf = self.config.n_features();
        if STATS {
            let n_preds = self.varying_predicates.len() as u64;
            counters.precompute_row_evals += n_preds * (rows.len() / nf) as u64;
            counters.precompute_mask_writes += n_preds;
        }
        for (out, pred) in out_left_masks
            .iter_mut()
            .zip(self.varying_predicates.iter())
        {
            let f = pred.feature() as usize;
            let mut left_mask = M::ZERO;
            for (r, row) in rows.chunks_exact(nf).enumerate() {
                if pred.goes_left::<F32>(row[f], &self.bitsets) {
                    left_mask = left_mask.set_bit(r);
                }
            }
            *out = left_mask;
        }
    }

    // -----------------------------------------------------------------------
    // Partition functions — generic over M: RowMask. With STATS, each adds the rows
    // it visits, the missing-value checks and the threshold comparisons it makes.
    // -----------------------------------------------------------------------

    /// Ascending values: rows go left until the first that does not; then only
    /// missing values remain to route.
    #[inline]
    pub(super) fn partition_mono_inc<const F32: bool, M: RowMask, const STATS: bool>(
        node: &Node,
        col: impl Fn(usize) -> f64,
        row_mask: M,
        c: &mut WorkCounters,
    ) -> (M, M) {
        let thresh = node.value;
        let default_left = node.default_left();
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            if STATS {
                c.scan_row_evals += 1;
                c.scan_missing_checks += 1;
            }
            if col(r).is_nan() {
                if default_left {
                    left = left.set_bit(r);
                }
                m = m.clear_lowest();
                continue;
            }
            if STATS {
                c.scan_compares += 1;
            }
            if threshold_go_left::<F32>(col(r), thresh) {
                left = left.set_bit(r);
                m = m.clear_lowest();
            } else {
                if default_left {
                    m = m.clear_lowest();
                    while !m.is_zero() {
                        let r2 = m.trailing_zeros() as usize;
                        if STATS {
                            c.scan_row_evals += 1;
                            c.scan_missing_checks += 1;
                        }
                        if col(r2).is_nan() {
                            left = left.set_bit(r2);
                        }
                        m = m.clear_lowest();
                    }
                }
                break;
            }
        }
        (left, row_mask & !left)
    }

    /// Descending values: rows go right until the first that goes left; every later
    /// row goes left too, except missing values, which follow the default.
    #[inline]
    pub(super) fn partition_mono_dec<const F32: bool, M: RowMask, const STATS: bool>(
        node: &Node,
        col: impl Fn(usize) -> f64,
        row_mask: M,
        c: &mut WorkCounters,
    ) -> (M, M) {
        let thresh = node.value;
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            if STATS {
                c.scan_row_evals += 1;
                c.scan_missing_checks += 1;
            }
            if col(r).is_nan() {
                if node.default_left() {
                    left = left.set_bit(r);
                }
                m = m.clear_lowest();
                continue;
            }
            if STATS {
                c.scan_compares += 1;
            }
            if threshold_go_left::<F32>(col(r), thresh) {
                left = left.set_bit(r);
                m = m.clear_lowest();
                while !m.is_zero() {
                    let r2 = m.trailing_zeros() as usize;
                    if STATS {
                        c.scan_row_evals += 1;
                        c.scan_missing_checks += 1;
                    }
                    if col(r2).is_nan() {
                        if node.default_left() {
                            left = left.set_bit(r2);
                        }
                    } else {
                        left = left.set_bit(r2);
                    }
                    m = m.clear_lowest();
                }
                break;
            }
            m = m.clear_lowest();
        }
        (left, row_mask & !left)
    }

    /// Every row on its own.
    #[inline]
    pub(super) fn partition_non_mono<const F32: bool, M: RowMask, const STATS: bool>(
        node: &Node,
        col: impl Fn(usize) -> f64,
        row_mask: M,
        c: &mut WorkCounters,
    ) -> (M, M) {
        let thresh = node.value;
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            m = m.clear_lowest();
            if STATS {
                c.scan_row_evals += 1;
                c.scan_missing_checks += 1;
            }
            if col(r).is_nan() {
                if node.default_left() {
                    left = left.set_bit(r);
                }
                continue;
            }
            if STATS {
                c.scan_compares += 1;
            }
            if threshold_go_left::<F32>(col(r), thresh) {
                left = left.set_bit(r);
            }
        }
        (left, row_mask & !left)
    }

    /// Every row on its own, through the full split test (categorical splits). A
    /// category membership test counts as a comparison.
    #[inline]
    pub(super) fn partition_per_row<const F32: bool, M: RowMask, const STATS: bool>(
        &self,
        node: &Node,
        rows: &[f64],
        n_features: usize,
        row_mask: M,
        c: &mut WorkCounters,
    ) -> (M, M) {
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            m = m.clear_lowest();
            let row = &rows[r * n_features..(r + 1) * n_features];
            if STATS {
                c.scan_row_evals += 1;
                c.scan_missing_checks += 1;
                c.scan_compares += u64::from(!row[node.feature as usize].is_nan());
            }
            if self.eval_split::<F32>(node, row) {
                left = left.set_bit(r);
            }
        }
        (left, row_mask & !left)
    }
}

#[cfg(test)]
mod tests {
    use super::WorkCounters;
    use crate::forest::{Model, Node};

    fn num_node(threshold: f64, default_left: bool) -> Node {
        Node {
            value: threshold,
            skip: -1,
            varying_pred_id: u16::MAX,
            feature: 0,
            flags: u8::from(default_left),
            cat_n_words: 0,
        }
    }

    /// (left, right) masks and the scan counters of one partition of `col`.
    fn inc(col: &[f64], threshold: f64, default_left: bool) -> (u32, u32, [u64; 3]) {
        let node = num_node(threshold, default_left);
        let mut c = WorkCounters::default();
        let all = (1u32 << col.len()) - 1;
        let (l, r) = Model::partition_mono_inc::<false, u32, true>(&node, |i| col[i], all, &mut c);
        (
            l,
            r,
            [c.scan_row_evals, c.scan_compares, c.scan_missing_checks],
        )
    }

    fn dec(col: &[f64], threshold: f64, default_left: bool) -> (u32, u32, [u64; 3]) {
        let node = num_node(threshold, default_left);
        let mut c = WorkCounters::default();
        let all = (1u32 << col.len()) - 1;
        let (l, r) = Model::partition_mono_dec::<false, u32, true>(&node, |i| col[i], all, &mut c);
        (
            l,
            r,
            [c.scan_row_evals, c.scan_compares, c.scan_missing_checks],
        )
    }

    fn non_mono(col: &[f64], threshold: f64, default_left: bool) -> (u32, u32, [u64; 3]) {
        let node = num_node(threshold, default_left);
        let mut c = WorkCounters::default();
        let all = (1u32 << col.len()) - 1;
        let (l, r) = Model::partition_non_mono::<false, u32, true>(&node, |i| col[i], all, &mut c);
        (
            l,
            r,
            [c.scan_row_evals, c.scan_compares, c.scan_missing_checks],
        )
    }

    #[test]
    fn test_partition_mono_inc_nan_after_threshold() {
        // After the first row that goes right, the scan stops comparing but still
        // checks every remaining row for a missing value.
        let (left, right, work) = inc(&[1.0, 2.0, 5.0, f64::NAN, 8.0], 3.0, true);
        assert_eq!((left, right), (0b01011, 0b10100));
        assert_eq!(work, [5, 3, 5]);
    }

    #[test]
    fn test_partition_mono_inc_nan_no_default_left() {
        // Missing values go right with everything after the threshold: no suffix scan.
        let (left, right, work) = inc(&[1.0, 2.0, 5.0, f64::NAN, 8.0], 3.0, false);
        assert_eq!((left, right), (0b00011, 0b11100));
        assert_eq!(work, [3, 3, 3]);
    }

    #[test]
    fn test_partition_mono_inc_nan_before_threshold() {
        let (left, right, work) = inc(&[f64::NAN, 1.0, 2.0, 5.0], 3.0, true);
        assert_eq!((left, right), (0b0111, 0b1000));
        assert_eq!(work, [4, 3, 4]);
    }

    #[test]
    fn mono_dec_checks_a_suffix_for_missing_values() {
        let (left, right, work) = dec(&[8.0, 5.0, 2.0, f64::NAN, 1.0], 3.0, false);
        assert_eq!((left, right), (0b10100, 0b01011));
        assert_eq!(work, [5, 3, 5]);
    }

    #[test]
    fn scans_compare_fewer_rows_than_per_row_partitions() {
        // The counters that disable_monotonic changes: the same split, the same
        // masks, fewer comparisons.
        let col: Vec<f64> = (0..20).map(f64::from).collect();
        let (inc_left, _, inc_work) = inc(&col, 4.5, false);
        let (non_left, _, non_work) = non_mono(&col, 4.5, false);
        assert_eq!(inc_left, non_left);
        assert_eq!(inc_work, [6, 6, 6]);
        assert_eq!(non_work, [20, 20, 20]);
    }

    #[test]
    fn scans_count_nothing_without_stats() {
        let node = num_node(3.0, true);
        let mut c = WorkCounters::default();
        let col = [1.0, 5.0];
        Model::partition_mono_inc::<false, u32, false>(&node, |i| col[i], 0b11, &mut c);
        Model::partition_non_mono::<false, u32, false>(&node, |i| col[i], 0b11, &mut c);
        assert_eq!(c, WorkCounters::default());
    }
}
