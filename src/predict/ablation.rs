//! Ablation-only paths: brute-force mask precompute and the per-row partitions
//! used when precompute is disabled.

use crate::forest::{Model, Node, threshold_go_left};
use crate::mask::RowMask;

impl Model {
    /// Brute-force O(P × n) precompute — evaluates each predicate against every row.
    /// Used as the ablation baseline when `disable_predicate_sweep` is set.
    pub(crate) fn precompute_bruteforce_generic<const F32: bool, M: RowMask>(
        &self,
        rows: &[f64],
        out_left_masks: &mut [M],
    ) {
        debug_assert_eq!(out_left_masks.len(), self.varying_predicates.len());
        let nf = self.config.n_features();
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
    // Partition functions — generic over M: RowMask
    // -----------------------------------------------------------------------

    #[inline]
    pub(super) fn partition_mono_inc<const F32: bool, M: RowMask>(
        node: &Node,
        col: impl Fn(usize) -> f64,
        row_mask: M,
    ) -> (M, M) {
        let thresh = node.value;
        let default_left = node.default_left();
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            if col(r).is_nan() {
                if default_left {
                    left = left.set_bit(r);
                }
                m = m.clear_lowest();
            } else if threshold_go_left::<F32>(col(r), thresh) {
                left = left.set_bit(r);
                m = m.clear_lowest();
            } else {
                if default_left {
                    m = m.clear_lowest();
                    while !m.is_zero() {
                        let r2 = m.trailing_zeros() as usize;
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

    #[inline]
    pub(super) fn partition_mono_dec<const F32: bool, M: RowMask>(
        node: &Node,
        col: impl Fn(usize) -> f64,
        row_mask: M,
    ) -> (M, M) {
        let thresh = node.value;
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            if col(r).is_nan() {
                if node.default_left() {
                    left = left.set_bit(r);
                }
                m = m.clear_lowest();
            } else if threshold_go_left::<F32>(col(r), thresh) {
                left = left.set_bit(r);
                m = m.clear_lowest();
                while !m.is_zero() {
                    let r2 = m.trailing_zeros() as usize;
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
            } else {
                m = m.clear_lowest();
            }
        }
        (left, row_mask & !left)
    }

    #[inline]
    pub(super) fn partition_non_mono<const F32: bool, M: RowMask>(
        node: &Node,
        col: impl Fn(usize) -> f64,
        row_mask: M,
    ) -> (M, M) {
        let thresh = node.value;
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            m = m.clear_lowest();
            if col(r).is_nan() {
                if node.default_left() {
                    left = left.set_bit(r);
                }
            } else if threshold_go_left::<F32>(col(r), thresh) {
                left = left.set_bit(r);
            }
        }
        (left, row_mask & !left)
    }

    #[inline]
    pub(super) fn partition_per_row<const F32: bool, M: RowMask>(
        &self,
        node: &Node,
        rows: &[f64],
        n_features: usize,
        row_mask: M,
    ) -> (M, M) {
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            m = m.clear_lowest();
            if self.eval_split::<F32>(node, &rows[r * n_features..(r + 1) * n_features]) {
                left = left.set_bit(r);
            }
        }
        (left, row_mask & !left)
    }
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn test_partition_mono_inc_nan_after_threshold() {
        let node = num_node(3.0, true);
        let col = &[
            1.0,
            2.0,
            5.0,
            f64::NAN,
            8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ][..];
        let (left, right) = Model::partition_mono_inc::<false, u32>(&node, |r| col[r], 0b11111u32);
        assert_eq!(left, 0b01011);
        assert_eq!(right, 0b10100);
    }

    #[test]
    fn test_partition_mono_inc_nan_no_default_left() {
        let node = num_node(3.0, false);
        let col = &[
            1.0,
            2.0,
            5.0,
            f64::NAN,
            8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ][..];
        let (left, right) = Model::partition_mono_inc::<false, u32>(&node, |r| col[r], 0b11111u32);
        assert_eq!(left, 0b00011);
        assert_eq!(right, 0b11100);
    }

    #[test]
    fn test_partition_mono_inc_nan_before_threshold() {
        let node = num_node(3.0, true);
        let col = &[
            f64::NAN,
            1.0,
            2.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ][..];
        let (left, right) = Model::partition_mono_inc::<false, u32>(&node, |r| col[r], 0b1111u32);
        assert_eq!(left, 0b0111);
        assert_eq!(right, 0b1000);
    }
}
