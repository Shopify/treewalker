//! The boundary between decoded fields and the unchecked prediction kernel.
//!
//! Check before exposing Model: config/precision, bin/code domains and exception
//! extents; dummy/node/link/mask/leaf fields; disjoint full binary trees (not DAGs)
//! and exact per-tree scratch needs; constant prefixes; predicate coverage;
//! fixed/wide accumulator bounds. The MaskPlan, with its numeric/chain/flip ranges,
//! is not stored: the builder's own, from the checked fields, is the model's. Chunks
//! and scratch are rebuilt by the predictor, not trusted from disk. Keep this list in
//! sync with every new unchecked index in predict/ and exact/.
//! Ownership sorts and per-split owner searches take O(R log R) work; the
//! ownership/traversal tables require O(R) scratch, and the MaskPlan at most two
//! flips per exception and chain of its word.

use super::codec::invalid;
use crate::exact::Plan;
use crate::load::layout::{
    ChainPoint, FLAG_LEAF, LINK_STRIDE, LayoutNode, MASK_FIELD, MASK_SHIFT, MAX_NODES,
    MAX_PREDICATES, MaskPlan, node_index, node_link,
};
use crate::model::{Model, Postprocessor, ThresholdType, VaryingPredicate};
use std::io;

pub(crate) fn check(ok: bool, message: &str) -> io::Result<()> {
    if ok { Ok(()) } else { Err(invalid(message)) }
}

pub(crate) fn span(start: usize, len: usize, total: usize, message: &str) -> io::Result<()> {
    check(start <= total && len <= total - start, message)
}

/// Each builder table is a partition, including the held-code slots after bins.
/// Empty ranges own no cells and may share a boundary.
pub(crate) fn partition(
    ranges: &mut [std::ops::Range<usize>],
    total: usize,
    message: &str,
) -> io::Result<()> {
    ranges.sort_unstable_by_key(|r| (r.start, r.end));
    let mut end = 0;
    for range in ranges.iter().filter(|r| !r.is_empty()) {
        check(range.start == end, message)?;
        end = range.end;
    }
    check(end == total, message)
}

/// Check `m`, decoded without its mask plan, and build that plan.
pub(crate) fn model(m: &Model) -> io::Result<MaskPlan> {
    m.config.validate_layout()?;
    check(
        m.output.base_score.is_finite() && m.output.divisor.is_finite() && m.output.divisor > 0.0,
        "output scale",
    )?;
    if let Postprocessor::Sigmoid(alpha) = m.output.postprocessor {
        check(alpha.is_finite() && alpha > 0.0, "sigmoid scale")?;
    }
    check(
        [
            m.steps.leaf,
            m.steps.stop,
            m.prefix.mean_depth,
            m.prefix.saved,
            m.prefix.cost,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0),
        "rule inputs",
    )?;
    check(
        m.rules
            .items
            .leaf_short
            .iter()
            .all(|v| (0.0..=1.0).contains(v)),
        "item length shares",
    )?;
    let f32 = m.threshold_type == ThresholdType::F32;
    let l = &m.layout;
    let b = &l.bins;
    let bins = b.validate_layout(&m.config, f32)?;
    check(
        !l.nodes.is_empty() && l.nodes.len() <= MAX_NODES,
        "node count",
    )?;
    let dummy = l.nodes.len() - 1;
    check(
        l.nodes[dummy].w0 == 0 && l.nodes[dummy].w1 == u64::from(node_link(dummy)),
        "dummy self-loop",
    )?;
    let np = m.varying_predicates.len();
    check(
        np <= MAX_PREDICATES && m.bounds.len() == np,
        "predicate count/bounds",
    )?;
    for p in &m.varying_predicates {
        check(
            (p.feature() as usize) < m.config.n_features()
                && m.config.is_varying(p.feature() as usize),
            "predicate feature",
        )?;
        if let VaryingPredicate::Num { threshold, .. } = p {
            check(!threshold.0.is_nan(), "NaN threshold")?;
        }
    }
    let kind = |p: &VaryingPredicate| usize::from(!matches!(p, VaryingPredicate::Num { .. }));
    // Unlike numeric threshold order, feature/kind order is required for safety:
    // build_feature_ranges partitions each feature into numeric then categorical
    // ranges, which must not expose ChainPoint::NONE to the categorical builder.
    for pair in m.varying_predicates.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        check(
            (a.feature(), kind(a)) <= (b.feature(), kind(b)),
            "predicate ordering",
        )?;
    }
    check(
        m.bounds == crate::predict::sweep_bounds(&m.varying_predicates, m.threshold_type),
        "sweep bound values",
    )?;
    let nleaves = match &m.plan {
        Plan::Fixed(e) => {
            check((-1023..=1074).contains(e), "fixed exponent")?;
            check(l.leaves.is_empty(), "fixed plan has a Wide leaf table")?;
            0
        }
        Plan::Wide(wide) => {
            wide.validate_layout(&l.leaves)?;
            l.leaves.len()
        }
    };
    check(
        m.tree_roots.len() == m.trees.len() && m.needs.len() == m.trees.len(),
        "tree metadata lengths",
    )?;
    let mut intervals = Vec::with_capacity(m.trees.len());
    // Check ownership before visiting any tree's nodes: repeating a valid large
    // extent must not multiply validation work by the number of tree records.
    for (tree, &root) in m.trees.iter().zip(&m.tree_roots) {
        let start = tree.node_start as usize;
        let count = tree.node_count as usize;
        span(start, count, dummy, "tree node extent")?;
        check(count > 0 && root == node_link(start), "tree root")?;
        intervals.push(start..start + count);
    }
    partition(&mut intervals, dummy, "tree node coverage/overlap")?;
    drop(intervals);
    let mut pending = Vec::new();
    let mut points = vec![ChainPoint::NONE; np];
    let mut seen = vec![false; np];
    let mut max_fixed = 0u128;
    for (tree, need) in m.trees.iter().zip(&m.needs) {
        let start = tree.node_start as usize;
        let count = tree.node_count as usize;
        let end = start + count;
        pending.clear();
        pending.push((start, end));
        let (mut leaves, mut varying) = (0, 0);
        // Each subtree has one parent: passes terminate, and these counts bound
        // every unchecked scratch write. An in-range DAG is not sufficient.
        while let Some((at, end)) = pending.pop() {
            let node = l.nodes[at];
            // A Fixed leaf's value occupies the mask and sign bits. Its flag
            // must be checked before interpreting any split-only fields.
            if node.w1 & FLAG_LEAF != 0 {
                check(at + 1 == end, "leaf subtree extent")?;
                match &m.plan {
                    Plan::Fixed(_) => {
                        let fixed = LayoutNode::fixed_value(node.w1);
                        check(
                            LayoutNode::fixed(fixed) == node
                                && node.w1 & 0x80 == 0
                                && node.w1 >> 63 == (node.w1 >> 62) & 1,
                            "fixed leaf fields",
                        )?;
                        max_fixed = max_fixed.max(fixed.validate_layout()?.unsigned_abs());
                    }
                    Plan::Wide(_) => {
                        let mask = (node.w1 >> MASK_SHIFT) & MASK_FIELD;
                        check(
                            node.w1 >> 63 == 0
                                && mask == 0
                                && node.w0 >> 32 == 0
                                && node.child() == node.payload()
                                && (node.payload() as usize) < nleaves,
                            "wide leaf fields",
                        )?;
                    }
                }
                leaves += 1;
            } else {
                let mask = ((node.w1 >> MASK_SHIFT) & MASK_FIELD) as usize;
                check(
                    node.w1 >> 63 == 0 && mask <= 2 * np && node.slot() < b.n_slots(),
                    "node mask/slot",
                )?;
                if let Some((id, point)) = bins.split(node, &m.varying_predicates)? {
                    seen[id] = true;
                    if point != ChainPoint::NONE {
                        // Same tree/split order as the builder: it retains the last
                        // occurrence's orientation for each categorical predicate.
                        points[id] = point;
                    }
                }
                let child = node_index(node.child());
                check(
                    node.child().is_multiple_of(LINK_STRIDE) && at + 1 < child && child < end,
                    "split children/preorder",
                )?;
                check(
                    b.has_shifts() || (node.w0 >> 32).trailing_zeros() >= 5,
                    "shift dispatch",
                )?;
                varying += usize::from(mask != 0);
                pending.push((child, end));
                pending.push((at + 1, child));
            }
        }
        let slots = if varying == 0 {
            0
        } else {
            leaves.min(varying + 1)
        };
        check(
            need.leaves == leaves && need.slots == slots,
            "tree scratch needs",
        )?;
    }
    check(
        max_fixed
            .checked_mul(m.trees.len() as u128)
            .is_some_and(|v| v < 1u128 << 125),
        "fixed sum overflow",
    )?;
    let mut previous_end = 0;
    for g in &m.prefix_groups {
        let (first, end, depth) = (g.first as usize, g.end as usize, g.depth as usize);
        check(
            first >= previous_end && first < end && end <= m.trees.len(),
            "prefix tree range",
        )?;
        check(g.node_base == m.trees[first].node_start, "prefix base")?;
        let base = g.node_base as usize;
        for tree in &m.trees[first..end] {
            check(depth < tree.node_count as usize, "prefix depth")?;
            for j in 0..depth {
                let n = l.nodes[tree.node_start as usize + j];
                check(
                    n.w1 & (FLAG_LEAF | MASK_FIELD << MASK_SHIFT) == 0
                        && n.w0 == l.nodes[base + j].w0,
                    "prefix constant split",
                )?;
            }
        }
        previous_end = end;
    }
    check(seen.iter().all(|&used| used), "node predicate coverage")?;
    drop(seen);
    let ranges = crate::load::predicates::build_feature_ranges(&m.varying_predicates);
    // The file stores no MaskPlan to check or to follow: the builder's, from the
    // checked bins, predicates and chain points, is the model's. Disjoint feature
    // tables and at most 32 shift fields per slot bound the builder's work, and the
    // file's own codes and exceptions its flips.
    let mask_bytes = b.rank_table_bytes() as u64 + MaskPlan::max_flip_bytes(b, &ranges);
    let masks =
        MaskPlan::build(b, &ranges, &points, mask_bytes).map_err(|_| invalid("mask plan size"))?;
    // Bound the scratch arithmetic before a predictor can run it. Chunks normally
    // make scratch much smaller, but one large tree can occupy a chunk alone.
    // This deliberately loose upper bound covers all queue/arena/mask carriers,
    // wide sums and padding. It is immaterial on 64-bit targets under the format
    // caps, and rejects potentially wrapping products on smaller address spaces.
    let extent = l.nodes.len() as u128 + np as u128 + b.n_slots() as u128 + 64;
    let scratch_bound = extent * (m.config.max_group_width() as u128 + 128) * 128;
    check(
        scratch_bound <= isize::MAX as u128,
        "layout scratch address-space limit",
    )?;
    Ok(masks)
}

impl crate::exact::FixedLeaf {
    /// Check the packed form before the kernel's shift can discard high bits.
    pub(crate) fn validate_layout(self) -> io::Result<i128> {
        if self.0 == 0 {
            return Ok(0);
        }
        let shift = (self.0 & 255) as u32;
        let odd = self.0 >> 8;
        check(
            shift < 125 && odd & 1 != 0 && odd.unsigned_abs() < 1 << 53,
            "fixed leaf encoding",
        )?;
        check(
            u128::from(odd.unsigned_abs()) < 1u128 << (125 - shift),
            "fixed leaf overflow",
        )?;
        Ok(i128::from(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout_file::testsupport as support;
    use crate::{Forest, WalkerConfig};

    #[test]
    fn packed_leaves_validate_before_decoding() {
        use crate::exact::FixedLeaf;
        for (value, scale) in [(0.0, 0), (1.5, 20), (-0.5, 20), (2.0f64.powi(124), 0)] {
            let leaf = FixedLeaf::new(value, scale);
            assert_eq!(
                leaf.validate_layout().unwrap(),
                crate::exact::to_fixed(value, scale)
            );
        }
        for bits in [
            1,
            (1 << 8) | 0x7d,
            (1 << 8) | 128,
            (1 << 8) | 255,
            (2 << 8) | 1,
            1i64 << 62,
            (3 << 8) | 0x7c,
        ] {
            assert!(FixedLeaf(bits).validate_layout().is_err(), "{bits}");
        }
    }

    fn saved_corruption(mut forest: Forest, mutate: impl FnOnce(&mut Model), reason: &str) {
        model(&forest.model).unwrap();
        mutate(std::sync::Arc::get_mut(&mut forest.model).unwrap());
        support::rejects(&forest, reason);
    }

    fn set_mask(node: &mut crate::load::layout::LayoutNode, mask: u64) {
        node.w1 = (node.w1 & !(MASK_FIELD << MASK_SHIFT)) | mask << MASK_SHIFT;
    }

    #[test]
    fn in_range_node_masks_must_match_slot_owners() {
        saved_corruption(
            support::numeric(2, true),
            |m| {
                let at = node_index(m.tree_roots[0]);
                set_mask(&mut m.layout.nodes[at], 0);
            },
            "mask ownership",
        );
        saved_corruption(
            support::numeric(2, true),
            |m| {
                let at = node_index(m.tree_roots[0]);
                let mask = ((m.layout.nodes[at].w1 >> MASK_SHIFT) & MASK_FIELD) as usize;
                set_mask(
                    &mut m.layout.nodes[at],
                    (2 * (1 - (mask - 1) / 2) + 1) as u64,
                );
            },
            "predicate feature",
        );
    }

    fn categories() -> Forest {
        support::forest(
            1,
            true,
            &[
                support::category_tree(&[1], (3, 1)),
                support::category_tree(&[2], (3, 1)),
            ],
        )
    }

    #[test]
    fn the_mask_plan_is_rebuilt_not_stored() {
        let mut forest = categories();
        let built = forest.model.masks.clone();
        assert!(!built.flips.is_empty());
        assert_eq!(model(&forest.model).unwrap(), built);
        // Whatever plan the saved forest holds, a loaded one has the builder's.
        std::sync::Arc::get_mut(&mut forest.model).unwrap().masks = MaskPlan::default();
        let file = support::File::new();
        forest.save_layout_impl(&file.0).unwrap();
        let loaded = Forest::load_layout_impl(&file.0).unwrap();
        assert_eq!(loaded.model.masks, built);
        #[cfg(all(unix, target_pointer_width = "64", not(miri)))]
        {
            // SAFETY: the test owns the file and does not write it while it is mapped.
            let mapped = unsafe { Forest::map_layout_impl(&file.0) }.unwrap();
            assert_eq!(mapped.model.masks, built);
        }
    }

    #[test]
    fn the_mask_plans_flips_are_bounded_by_the_codes_and_exceptions() {
        let flips = |forest: &Forest| {
            let m = &forest.model;
            let ranges = crate::load::predicates::build_feature_ranges(&m.varying_predicates);
            (
                4 * (m.masks.flip_bounds.len() + m.masks.flips.len()) as u64,
                MaskPlan::max_flip_bytes(&m.layout.bins, &ranges),
            )
        };
        // Four codes, no category, missing and two categories, and four exceptions, a
        // split's category and missing in each of the two chains: five bounds and four
        // flips, within five bounds and two flips for each exception and each of a
        // word's 32 shifts.
        assert_eq!(flips(&categories()), (4 * (5 + 4), 4 * (5 + 2 * 4 * 32)));
        // Without varying categorical predicates there is nothing to bound.
        assert_eq!(flips(&support::numeric(2, true)), (0, 0));
    }

    #[test]
    fn wide_leaves_reject_shift_bits_as_well_as_slots() {
        let tree = crate::load::json::stump(
            r#""node_type": "numerical_test_node", "comparison_op": "<=", "threshold": 0.5"#,
            true,
            (3, 1),
        )
        .replace(
            "\"leaf_value\": 1.0",
            &format!("\"leaf_value\": {:e}", f64::from_bits(1)),
        )
        .replace(
            "\"leaf_value\": 2.0",
            &format!("\"leaf_value\": {:e}", f64::MAX),
        );
        for bit in [32, 37] {
            saved_corruption(
                support::forest(1, true, std::slice::from_ref(&tree)),
                |m| {
                    assert!(matches!(m.plan, Plan::Wide(_)));
                    m.layout
                        .nodes
                        .iter_mut()
                        .find(|n| n.w1 & FLAG_LEAF != 0)
                        .unwrap()
                        .w0 |= 1 << bit;
                },
                "wide leaf fields",
            );
        }
    }

    #[test]
    fn predicate_feature_and_kind_ordering_is_required_for_mask_ranges() {
        saved_corruption(
            support::numeric(2, true),
            |m| m.varying_predicates.swap(0, 1),
            "predicate ordering",
        );
        let num = crate::load::json::stump(
            r#""node_type": "numerical_test_node", "comparison_op": "<=", "threshold": 0.5"#,
            true,
            (3, 1),
        );
        saved_corruption(
            support::forest(1, true, &[num, support::category_tree(&[1, 2], (3, 1))]),
            |m| m.varying_predicates.swap(0, 1),
            "predicate ordering",
        );
    }

    #[test]
    fn checksummed_inline_fixed_leaves_must_be_canonical() {
        type Corrupt = fn(&mut LayoutNode);
        let changes: &[(Corrupt, &str)] = &[
            (|n| n.w0 = 1, "fixed leaf fields"),
            (|n| n.w1 |= 0x80, "fixed leaf fields"),
            (|n| n.w1 ^= 1 << 63, "fixed leaf fields"),
            (|n| n.w1 = (n.w1 & !0x7f) | 0x7d, "fixed leaf encoding"),
        ];
        for &(change, reason) in changes {
            saved_corruption(
                support::numeric(1, true),
                |m| {
                    let leaf = m
                        .layout
                        .nodes
                        .iter_mut()
                        .find(|n| n.w1 & FLAG_LEAF != 0)
                        .unwrap();
                    change(leaf);
                },
                reason,
            );
        }
    }

    #[test]
    fn inline_fixed_values_keep_the_forest_sum_bound() {
        saved_corruption(
            support::numeric(2, true),
            |m| {
                let leaf = m
                    .layout
                    .nodes
                    .iter_mut()
                    .find(|n| n.w1 & FLAG_LEAF != 0)
                    .unwrap();
                *leaf = LayoutNode::fixed(crate::exact::FixedLeaf::new(2.0f64.powi(124), 0));
            },
            "fixed sum overflow",
        );
    }

    #[test]
    fn tree_aliases_are_rejected_before_any_node_traversal() {
        let mut forest = support::numeric(1, false);
        let m = std::sync::Arc::get_mut(&mut forest.model).unwrap();
        m.trees.push(m.trees[0]);
        m.tree_roots.push(m.tree_roots[0]);
        m.needs.push(m.needs[0]);
        // Even an invalid first node must not precede the ownership rejection.
        m.layout.nodes[0].w1 |= 1 << 63;
        support::rejects(&forest, "tree node coverage/overlap");
    }

    #[test]
    fn rejects_corrupt_models_without_relying_on_file_checksum() {
        let corruptions: &[fn(&mut Model)] = &[
            |m| m.output.divisor = 0.0,
            |m| m.steps.leaf = f64::NAN,
            |m| {
                m.config = WalkerConfig::builder(1)
                    .max_group_width(1)
                    .all_varying()
                    .build()
                    .unwrap();
            },
            |m| m.layout.nodes[0].w1 = u64::from(u32::MAX),
            |m| m.layout.nodes[0].w0 = u64::MAX,
            |m| m.layout.nodes[0].w1 |= 1 << 63,
            |m| {
                let n = m.layout.nodes.len();
                m.layout.nodes[n - 1].w1 = 0;
            },
            |m| m.layout.bins.defaults[0] = 0,
            |m| m.layout.bins.offsets.clear(),
            |m| m.layout.bins.offsets[0] = u32::MAX,
            |m| m.needs[0].leaves = 0,
            |m| m.tree_roots[0] = u32::MAX,
            |m| m.trees[0].node_count = u32::MAX,
            |m| {
                m.prefix_groups.push(crate::model::PrefixGroup {
                    first: 0,
                    end: u32::MAX,
                    depth: 1,
                    node_base: 0,
                });
            },
            |m| m.bounds.clear(),
            |m| m.plan = Plan::Fixed(i32::MAX),
        ];
        for (i, corrupt) in corruptions.iter().enumerate() {
            let mut forest = support::fixture();
            model(&forest.model).unwrap();
            let m = std::sync::Arc::get_mut(&mut forest.model).unwrap();
            corrupt(m);
            assert!(model(m).is_err(), "corruption {i}");
        }
    }
}
