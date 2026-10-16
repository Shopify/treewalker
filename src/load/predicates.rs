//! The varying predicates: each varying split's predicate, deduplicated or not, sorted by
//! feature, kind and threshold, and the per-feature ranges the precompute reads.

use super::parse::decoded::Decoded;
use crate::LoadError;
use crate::model::{FeatureRange, NONE, VaryingPredicate};

/// The varying predicate of each varying node, by its split or by its position.
pub(crate) enum PredicateIds {
    /// Identical varying splits share a predicate: each split's ID, [`NONE`] for
    /// constant splits.
    BySplit(Vec<u32>),
    /// Every varying node has its own, in node order (an ablation).
    ByNode(Vec<u32>),
}

/// The varying predicates of the decoded forest with its trees in `order`, sorted by
/// [`sort_varying_predicates`], and the predicate of each varying node.
///
/// With `dedup`, two nodes with the same (feature, threshold or categories,
/// default_left) share one predicate ID, so the precompute evaluates each distinct
/// predicate once per row rather than each node. Thresholds compare by their bits
/// (so -0.0 and 0.0 differ), and listed categories by their pool offset, which
/// interning makes equal for equal lists. IDs number predicates in the order they
/// first appear in the ordered nodes before the sort.
pub(crate) fn varying_predicates(
    decoded: &Decoded,
    order: &[u32],
    dedup: bool,
) -> Result<(Vec<VaryingPredicate>, PredicateIds), LoadError> {
    let splits = &decoded.splits.all;
    let mut predicates = Vec::new();
    if dedup {
        let mut ids = vec![NONE; splits.len()];
        for &t in order {
            for &s in decoded.firsts(t as usize) {
                let split = &splits[s as usize];
                if split.is_varying() && ids[s as usize] == NONE {
                    ids[s as usize] = predicate_id(predicates.len())?;
                    predicates.push(VaryingPredicate::of_split(split));
                }
            }
        }
        let old_to_new = sort_varying_predicates(&mut predicates);
        for id in ids.iter_mut().filter(|id| **id != NONE) {
            *id = old_to_new[*id as usize];
        }
        Ok((predicates, PredicateIds::BySplit(ids)))
    } else {
        for &t in order {
            for node in decoded.nodes(t as usize).iter() {
                if !node.is_leaf() && decoded.split(node).is_varying() {
                    predicate_id(predicates.len())?;
                    predicates.push(VaryingPredicate::of_split(decoded.split(node)));
                }
            }
        }
        let old_to_new = sort_varying_predicates(&mut predicates);
        Ok((predicates, PredicateIds::ByNode(old_to_new)))
    }
}

/// The ID of the predicate after `n` others: below the layout's limit.
fn predicate_id(n: usize) -> Result<u32, LoadError> {
    if n >= crate::load::layout::MAX_PREDICATES {
        return Err(LoadError::Limit(format!(
            "more than {} varying predicates: the layout's mask field, which holds a \
             mask per side of each, is {} bits",
            crate::load::layout::MAX_PREDICATES,
            crate::load::layout::MASK_BITS
        )));
    }
    Ok(n as u32)
}

// ---------------------------------------------------------------------------
// Predicate sorting and feature-range index
// ---------------------------------------------------------------------------

/// Sort key ordinal: numerical predicates sort before categoricals within each
/// feature so the sweep kernel can process the contiguous numerical range first.
const fn pred_kind_ordinal(pred: &VaryingPredicate) -> u8 {
    match pred {
        VaryingPredicate::Num { .. } => 0,
        VaryingPredicate::CatInline { .. } | VaryingPredicate::CatPool { .. } => 1,
    }
}

/// Sort `predicates` by `(feature, kind, threshold)` and return each one's new
/// position by its old one.
///
/// After sorting, all predicates for the same feature are contiguous, with
/// numerical predicates (sorted by ascending threshold) before categoricals.
/// This order is what the sweep of [`crate::model::Model::precompute_sweep`] merges with.
pub(crate) fn sort_varying_predicates(predicates: &mut Vec<VaryingPredicate>) -> Vec<u32> {
    // Build sort permutation (indices into the original vec).
    let mut sorted_indices: Vec<usize> = (0..predicates.len()).collect();
    sorted_indices.sort_unstable_by(|&a, &b| {
        let pa = &predicates[a];
        let pb = &predicates[b];
        pa.feature()
            .cmp(&pb.feature())
            .then_with(|| pred_kind_ordinal(pa).cmp(&pred_kind_ordinal(pb)))
            .then_with(|| {
                // For numerical predicates: sort by threshold ascending.
                // For categoricals: arbitrary stable order (by raw bits).
                match (pa, pb) {
                    (
                        VaryingPredicate::Num { threshold: ta, .. },
                        VaryingPredicate::Num { threshold: tb, .. },
                    ) => ta.0.total_cmp(&tb.0),
                    _ => std::cmp::Ordering::Equal,
                }
            })
    });

    // Build old→new mapping.
    let mut old_to_new = vec![0u32; predicates.len()];
    for (new_pos, &old_pos) in sorted_indices.iter().enumerate() {
        old_to_new[old_pos] = new_pos as u32;
    }

    // Reorder predicates vec.
    let reordered: Vec<VaryingPredicate> = sorted_indices.iter().map(|&i| predicates[i]).collect();
    *predicates = reordered;
    old_to_new
}

/// Each varying feature's ranges of numeric and categorical predicates, from
/// `predicates` sorted by [`sort_varying_predicates`].
///
/// Index loops rather than `chunk_by` and `partition_point`, whose collected form
/// compiles to different load code (checked against the post-LTO assembly).
pub(crate) fn build_feature_ranges(predicates: &[VaryingPredicate]) -> Vec<FeatureRange> {
    if predicates.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut i = 0;
    while i < predicates.len() {
        let feature = predicates[i].feature();
        let group_start = i;

        // Scan numerical predicates (kind ordinal 0).
        let num_start = i;
        while i < predicates.len()
            && predicates[i].feature() == feature
            && pred_kind_ordinal(&predicates[i]) == 0
        {
            i += 1;
        }
        let num_end = i;

        // Scan categorical predicates (kind ordinal 1).
        let cat_start = i;
        while i < predicates.len()
            && predicates[i].feature() == feature
            && pred_kind_ordinal(&predicates[i]) == 1
        {
            i += 1;
        }
        let cat_end = i;

        debug_assert!(
            i > group_start,
            "empty feature group at index {group_start}"
        );

        ranges.push(FeatureRange {
            feature,
            num: num_start as u32..num_end as u32,
            cat: cat_start as u32..cat_end as u32,
        });
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicate_ids_stop_at_the_layout_limit() {
        let max = crate::load::layout::MAX_PREDICATES;
        assert_eq!(predicate_id(max - 1).unwrap() as usize, max - 1);
        assert!(matches!(predicate_id(max), Err(LoadError::Limit(_))));
        // The limit leaves the sentinel unassigned.
        assert!(max < NONE as usize);
    }
}
