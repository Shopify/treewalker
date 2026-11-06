//! The boundary between decoded fields and the unchecked prediction kernel.
//!
//! Check before exposing Model: config/precision, bin/code domains and exception
//! extents; dummy/node/link/mask/leaf fields; disjoint full binary trees (not DAGs)
//! and exact per-tree scratch needs; constant prefixes; all numeric/chain/flip
//! ranges and predicate coverage; fixed/wide accumulator bounds. Research scans
//! additionally require parsed nodes and category lists. Chunks and scratch are
//! rebuilt by the predictor, not trusted from disk. Keep this list in sync with
//! every new unchecked index in predict/ and exact/.

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

/// Linear record budgets, not a cap that can reject a writer's long chains.
/// Small comparisons use at most 64R merge visits; the batched fallback at most
/// 32E field visits, 64E toggles and 2P queries; the index D + E + 2P and the
/// defaults P. R counts E, D and P, so 192R covers them all.
pub(crate) struct Work {
    remaining: u64,
    merge_remaining: u64,
}
impl Work {
    pub(crate) const fn new(records: usize) -> Self {
        Self {
            remaining: (records as u64).saturating_add(1).saturating_mul(192),
            merge_remaining: (records as u64).saturating_add(1).saturating_mul(64),
        }
    }
    pub(crate) fn spend(&mut self, records: usize) -> io::Result<()> {
        self.remaining = self
            .remaining
            .checked_sub(records as u64)
            .ok_or_else(|| invalid("layout validation work limit"))?;
        Ok(())
    }

    /// Once repeated slice merges would exceed a linear total, use the batched
    /// fallback instead of rejecting a valid model or rescanning a shared slot.
    pub(crate) fn merge(&mut self, records: usize) -> io::Result<bool> {
        if records as u64 > self.merge_remaining {
            return Ok(false);
        }
        self.merge_remaining -= records as u64;
        self.spend(records)?;
        Ok(true)
    }
}

#[cfg(any(test, feature = "research"))]
fn category_starts(categories: &[u32]) -> io::Result<Vec<u32>> {
    let mut starts = Vec::new();
    let mut at = 0;
    while at < categories.len() {
        let len = categories[at] as usize;
        span(at + 1, len, categories.len(), "category list extent")?;
        check(
            categories[at + 1..at + 1 + len]
                .windows(2)
                .all(|w| w[0] < w[1]),
            "sorted category list",
        )?;
        starts.push(at as u32);
        at += len + 1;
    }
    Ok(starts)
}

pub(crate) fn model(m: &Model) -> io::Result<()> {
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
    #[cfg(any(test, feature = "research"))]
    let category_starts = category_starts(&m.categories)?;
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
        // Production frees these lists after building the bins. CatPool offsets
        // are read only by research scans and brute-force masks.
        #[cfg(any(test, feature = "research"))]
        if let VaryingPredicate::CatPool { offset, .. } = p {
            check(
                category_starts.binary_search(offset).is_ok(),
                "category list offset",
            )?;
        }
    }
    let kind = |p: &VaryingPredicate| usize::from(!matches!(p, VaryingPredicate::Num { .. }));
    for pair in m.varying_predicates.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        check(
            (a.feature(), kind(a)) <= (b.feature(), kind(b)),
            "predicate ordering",
        )?;
        if let (
            VaryingPredicate::Num { threshold: a, .. },
            VaryingPredicate::Num { threshold: b, .. },
        ) = (a, b)
            && pair[0].feature() == pair[1].feature()
        {
            check(a.0.total_cmp(&b.0).is_le(), "numeric predicate ordering")?;
        }
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
    let mut sides = vec![[None; 2]; np];
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
                        let previous = &mut sides[id][usize::from(point.invert)];
                        check(
                            previous.is_none_or(|p| p == point),
                            "predicate side encoding",
                        )?;
                        *previous = Some(point);
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
    check(
        m.masks
            .features
            .windows(2)
            .all(|f| f[0].feature < f[1].feature),
        "unique mask features",
    )?;
    let mut work = Work::new(l.nodes.len() + np + bins.records());
    let pairs: Vec<_> = sides
        .into_iter()
        .filter_map(|encodings| {
            if let [Some(a), Some(b)] = encodings {
                Some([a, b])
            } else {
                None
            }
        })
        .collect();
    bins.check_sides(&pairs, &mut work)?;
    drop(pairs);
    let ranges = crate::load::predicates::build_feature_ranges(&m.varying_predicates);
    // Disjoint feature tables and at most 32 shift fields per slot bound the
    // builder's work. Never traverse untrusted MaskPlan ranges before equality.
    // Match the builder's cap accounting exactly. A corrupt stored plan must
    // not make its reconstruction allocate more flips than the file supplied.
    let mask_bytes = b.rank_table_bytes() as u64
        + 4 * (m.masks.flip_bounds.len() as u64 + m.masks.flips.len() as u64);
    let expected = MaskPlan::build(b, &ranges, &points, mask_bytes)
        .map_err(|_| invalid("mask plan correspondence"))?;
    check(m.masks == expected, "mask plan correspondence")?;
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
    #[cfg(any(test, feature = "research"))]
    {
        check(m.nodes.len() == dummy, "research nodes length")?;
        let mut extent = crate::exact::Extent::default();
        for (node, layout) in m.nodes.iter().zip(l.nodes.as_slice()) {
            check(
                node.is_leaf() == (layout.w1 & FLAG_LEAF != 0),
                "research leaf/split agreement",
            )?;
            if node.is_leaf() {
                check(node.value.is_finite(), "research leaf value")?;
                extent.add(node.value);
            }
        }
        // Reconstructing the builder's plan from finite leaves proves that its
        // research conversions cannot shift out of range or lose integer bits.
        check(
            crate::exact::Plan::of(extent, m.trees.len()) == m.plan,
            "research exact plan",
        )?;
        for tree in &m.trees {
            for (i, node) in m.nodes
                [tree.node_start as usize..(tree.node_start + tree.node_count) as usize]
                .iter()
                .enumerate()
            {
                if node.is_leaf() {
                    let layout = l.nodes[tree.node_start as usize + i];
                    check(
                        match &m.plan {
                            Plan::Fixed(e) => {
                                LayoutNode::fixed(crate::exact::FixedLeaf::new(node.value, *e))
                                    == layout
                            }
                            Plan::Wide(wide) => {
                                wide.leaf(node.value) == l.leaves[layout.payload() as usize]
                            }
                        },
                        "research leaf encoding",
                    )?;
                } else {
                    let encoded = l.nodes[tree.node_start as usize + i];
                    let mask = ((encoded.w1 >> MASK_SHIFT) & MASK_FIELD) as usize;
                    check(
                        node.skip as usize
                            == node_index(encoded.child()) - tree.node_start as usize
                            && node.feature as usize == bins.feature(encoded.slot())?
                            && node.is_constant() == (mask == 0),
                        "research split agreement",
                    )?;
                    if mask == 0 {
                        check(
                            node.varying_pred_id == crate::model::NONE,
                            "research constant predicate",
                        )?;
                    } else {
                        let id = (mask - 1) / 2;
                        check(
                            node.varying_pred_id as usize == id
                                && node.heavy_is_left() == (mask & 1 != 0)
                                && VaryingPredicate::of(node) == m.varying_predicates[id],
                            "research varying predicate",
                        )?;
                    }
                    if node.is_categorical() && !node.inline_cat() {
                        check(
                            node.value >= 0.0
                                && node.value <= f64::from(u32::MAX)
                                && node.value.fract() == 0.0,
                            "research category offset",
                        )?;
                        check(
                            category_starts.binary_search(&(node.value as u32)).is_ok(),
                            "research category offset",
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
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
    fn in_range_node_masks_must_match_slots_predicates_and_sides() {
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
        saved_corruption(
            support::numeric(1, true),
            |m| {
                let at = node_index(m.tree_roots[0]);
                set_mask(&mut m.layout.nodes[at], 2);
            },
            "numeric predicate encoding",
        );
        let trees: Vec<_> = [0.5, 1.5].map(|t| crate::load::json::stump(&format!(r#""node_type": "numerical_test_node", "comparison_op": "<=", "threshold": {t}"#), true, (3, 1))).into();
        saved_corruption(
            support::forest(1, true, &trees),
            |m| {
                let at = node_index(m.tree_roots[0]);
                let mask = (m.layout.nodes[at].w1 >> MASK_SHIFT) & MASK_FIELD;
                set_mask(&mut m.layout.nodes[at], if mask == 1 { 3 } else { 1 });
            },
            "numeric predicate encoding",
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
    fn in_range_mask_plan_mutations_are_not_just_bounds_checked() {
        saved_corruption(
            categories(),
            |m| m.masks.points.swap(0, 1),
            "mask plan correspondence",
        );
        saved_corruption(
            categories(),
            |m| {
                m.masks.chains[0].start =
                    (m.masks.chains[0].start + 1) % (m.masks.chains[0].len + 1);
            },
            "mask plan correspondence",
        );
        saved_corruption(
            categories(),
            |m| m.masks.flips[0] = 1 - m.masks.flips[0],
            "mask plan correspondence",
        );
        saved_corruption(
            categories(),
            |m| {
                m.masks.features.push(m.masks.features[0].clone());
            },
            "unique mask features",
        );
    }

    #[test]
    fn mask_rebuild_is_capped_at_stored_flip_bytes() {
        let mut forest = categories();
        let m = std::sync::Arc::get_mut(&mut forest.model).unwrap();
        let points: Vec<_> = m
            .tree_roots
            .iter()
            .map(|&root| {
                let n = m.layout.nodes[node_index(root)];
                ChainPoint {
                    slot: n.slot() as u32,
                    shift: ((n.w0 >> 32) & 31) as u32,
                    bound: n.payload(),
                    invert: ((n.w1 >> MASK_SHIFT) & MASK_FIELD) & 1 == 0,
                }
            })
            .collect();
        let ranges = crate::load::predicates::build_feature_ranges(&m.varying_predicates);
        let b = &m.layout.bins;
        assert!(!m.masks.flips.is_empty());
        m.masks.flips.clear();
        let cap = b.rank_table_bytes() as u64 + 4 * m.masks.flip_bounds.len() as u64;
        assert!(MaskPlan::build(b, &ranges, &points, cap).is_err());
        support::rejects(&forest, "mask plan correspondence");
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
    fn earlier_opposite_side_encoding_must_agree_with_the_retained_point() {
        let forest = support::forest(
            1,
            true,
            &[
                support::category_tree(&[1, 2], (3, 1)),
                support::category_tree(&[1, 2], (1, 3)),
                support::category_tree(&[3], (3, 1)),
            ],
        );
        saved_corruption(
            forest,
            |m| {
                let nodes: Vec<_> = m.tree_roots.iter().map(|&root| node_index(root)).collect();
                let id = |at: usize| {
                    (((m.layout.nodes[at].w1 >> MASK_SHIFT) & MASK_FIELD) as usize - 1) / 2
                };
                let repeated = (0..2)
                    .find(|&p| nodes.iter().filter(|&&at| id(at) == p).count() == 2)
                    .unwrap();
                let earlier = *nodes.iter().find(|&&at| id(at) == repeated).unwrap();
                let other = *nodes.iter().find(|&&at| id(at) != repeated).unwrap();
                m.layout.nodes[earlier].w0 = m.layout.nodes[other].w0;
            },
            "categorical predicate sides",
        );
    }

    #[test]
    fn research_leaf_agreement_and_exact_representability_are_checked() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            saved_corruption(
                support::numeric(1, true),
                |m| {
                    m.nodes.iter_mut().find(|n| n.is_leaf()).unwrap().value = value;
                },
                "research leaf value",
            );
        }
        saved_corruption(
            support::numeric(1, true),
            |m| {
                m.nodes.iter_mut().find(|n| n.is_leaf()).unwrap().value = f64::MAX;
            },
            "research exact plan",
        );
        saved_corruption(
            support::numeric(1, true),
            |m| {
                m.nodes.iter_mut().find(|n| n.is_leaf()).unwrap().value = 3.0;
            },
            "research leaf encoding",
        );
        saved_corruption(
            support::numeric(1, true),
            |m| {
                m.nodes.iter_mut().find(|n| !n.is_leaf()).unwrap().skip = crate::model::NONE;
            },
            "research leaf/split agreement",
        );
        saved_corruption(
            support::numeric(1, true),
            |m| {
                m.nodes.iter_mut().find(|n| n.is_leaf()).unwrap().skip = 0;
            },
            "research leaf/split agreement",
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
    fn categorical_comparison_work_is_explicitly_bounded() {
        let mut work = Work::new(0);
        assert!(work.merge(64).unwrap());
        assert!(!work.merge(1).unwrap());
        work.spend(192 - 64).unwrap();
        assert!(work.spend(1).is_err());
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
            |m| m.nodes[0].skip = u32::MAX - 1,
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
