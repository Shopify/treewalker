//! Experimental, size-nonincreasing predicate reordering before tree layout.
//!
//! `V(C(A,B), C(D,E)) -> C(V(A,D), V(B,E))`, followed by `P(A,A) -> A`.
//! C must be the same group-constant predicate in both children. Predicate
//! identity includes missing routing, precision-normalized threshold bits and
//! category contents. No comparisons are inverted and no leaf values change.

use rustc_hash::FxHashMap;

use super::common::TempNode;
use crate::forest::{FLAG_CATEGORICAL, FLAG_INLINE_CAT, FLAG_WALKABLE};

/// Aggregate load-time counters for the experimental hoisting pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HoistStats {
    pub trees_examined: usize,
    pub trees_changed: usize,
    pub swaps: usize,
    pub collapsed_splits: usize,
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub varying_roots_before: usize,
    pub varying_roots_after: usize,
    /// Trees that changed on the last allowed pass; more rewrites may remain.
    pub pass_limit_hits: usize,
}

// Bound import work independently of tree depth. All walks are iterative.
const MAX_PASSES: usize = 16;

#[derive(Hash, PartialEq, Eq)]
enum ValueKey {
    Bits(u64),
    Categories(Vec<u8>),
}

#[derive(Hash, PartialEq, Eq)]
enum SubtreeKey {
    Leaf(u64),
    Split {
        feature: u16,
        flags: u8,
        value: ValueKey,
        left: usize,
        right: usize,
    },
}

const fn is_constant(node: &TempNode) -> bool {
    node.left >= 0 && node.flags & FLAG_WALKABLE != 0
}

const fn is_varying(node: &TempNode) -> bool {
    node.left >= 0 && !is_constant(node)
}

fn category_bytes<'a>(node: &TempNode, bitsets: &'a [u8]) -> &'a [u8] {
    let start = node.value as usize;
    &bitsets[start..start + usize::from(node.cat_n_words) * 4]
}

const fn is_pool_category(node: &TempNode) -> bool {
    node.flags & FLAG_CATEGORICAL != 0 && node.flags & FLAG_INLINE_CAT == 0
}

fn same_predicate(a: &TempNode, b: &TempNode, bitsets: &[u8]) -> bool {
    a.feature == b.feature
        && a.flags == b.flags
        && a.cat_n_words == b.cat_n_words
        && if is_pool_category(a) {
            // Compare contents, even when bitset interning is disabled.
            category_bytes(a, bitsets) == category_bytes(b, bitsets)
        } else {
            a.value.to_bits() == b.value.to_bits()
        }
}

fn intern(
    node: &TempNode,
    ids: &[usize],
    bitsets: &[u8],
    keys: &mut FxHashMap<SubtreeKey, usize>,
) -> usize {
    let key = if node.left < 0 {
        SubtreeKey::Leaf(node.value.to_bits())
    } else {
        SubtreeKey::Split {
            feature: node.feature,
            flags: node.flags,
            value: if is_pool_category(node) {
                ValueKey::Categories(category_bytes(node, bitsets).to_vec())
            } else {
                ValueKey::Bits(node.value.to_bits())
            },
            left: ids[node.left as usize],
            right: ids[node.right as usize],
        }
    };
    let next = keys.len();
    *keys.entry(key).or_insert(next)
}

// Reverse preorder is a valid children-before-parents ordering for a tree.
fn visit(nodes: &[TempNode], root: usize, order: &mut Vec<usize>, stack: &mut Vec<usize>) {
    order.clear();
    stack.clear();
    stack.push(root);
    while let Some(i) = stack.pop() {
        order.push(i);
        if nodes[i].left >= 0 {
            stack.push(nodes[i].left as usize);
            stack.push(nodes[i].right as usize);
        }
    }
}

// Equal subtrees may have different training weights. Combine corresponding
// counts/hessians before retaining one copy, rather than retaining one path's
// statistics for both paths. Weights remain heuristics when the source omitted
// them; they never affect prediction semantics or structural equality.
fn collapse(nodes: &mut [TempNode], i: usize, ids: &mut [usize], pairs: &mut Vec<(usize, usize)>) {
    let old = nodes[i];
    let left = old.left as usize;
    pairs.clear();
    pairs.push((left, old.right as usize));
    while let Some((a, b)) = pairs.pop() {
        nodes[a].weight += nodes[b].weight;
        if nodes[a].left >= 0 {
            pairs.push((nodes[a].left as usize, nodes[b].left as usize));
            pairs.push((nodes[a].right as usize, nodes[b].right as usize));
        }
    }
    nodes[i] = nodes[left];
    nodes[i].weight = old.weight;
    ids[i] = ids[left];
}

pub(super) fn hoist(nodes: &mut [TempNode], root: usize, bitsets: &[u8], stats: &mut HoistStats) {
    let mut order = Vec::with_capacity(nodes.len());
    let mut stack = Vec::new();
    let mut pairs = Vec::new();
    let mut ids = vec![0; nodes.len()];
    let mut keys = FxHashMap::default();
    visit(nodes, root, &mut order, &mut stack);
    stats.trees_examined += 1;
    stats.nodes_before += order.len();
    stats.varying_roots_before += usize::from(is_varying(&nodes[root]));
    let mut tree_changed = false;

    for pass in 0..MAX_PASSES {
        keys.clear();
        let mut changed = false;
        for &i in order.iter().rev() {
            let parent = nodes[i];
            if parent.left >= 0 {
                let l = parent.left as usize;
                let r = parent.right as usize;
                let left = nodes[l];
                let right = nodes[r];
                if is_varying(&parent)
                    && is_constant(&left)
                    && is_constant(&right)
                    && same_predicate(&left, &right, bitsets)
                {
                    nodes[l] = TempNode {
                        left: left.left,
                        right: right.left,
                        weight: nodes[left.left as usize].weight
                            + nodes[right.left as usize].weight,
                        ..parent
                    };
                    nodes[r] = TempNode {
                        left: left.right,
                        right: right.right,
                        weight: nodes[left.right as usize].weight
                            + nodes[right.right as usize].weight,
                        ..parent
                    };
                    nodes[i] = TempNode {
                        left: parent.left,
                        right: parent.right,
                        weight: parent.weight,
                        ..left
                    };
                    for child in [l, r] {
                        if ids[nodes[child].left as usize] == ids[nodes[child].right as usize] {
                            collapse(nodes, child, &mut ids, &mut pairs);
                            stats.collapsed_splits += 1;
                        } else {
                            ids[child] = intern(&nodes[child], &ids, bitsets, &mut keys);
                        }
                    }
                    stats.swaps += 1;
                    changed = true;
                }
                if ids[nodes[i].left as usize] == ids[nodes[i].right as usize] {
                    collapse(nodes, i, &mut ids, &mut pairs);
                    stats.collapsed_splits += 1;
                    changed = true;
                    continue;
                }
            }
            ids[i] = intern(&nodes[i], &ids, bitsets, &mut keys);
        }
        if !changed {
            break;
        }
        tree_changed = true;
        visit(nodes, root, &mut order, &mut stack);
        if pass + 1 == MAX_PASSES {
            stats.pass_limit_hits += 1;
        }
    }
    stats.trees_changed += usize::from(tree_changed);
    stats.nodes_after += order.len();
    stats.varying_roots_after += usize::from(is_varying(&nodes[root]));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forest::{FLAG_DEFAULT_LEFT, SPLIT_CONSTANT, SPLIT_NON_MONO};
    use crate::parser::common::build_flags;

    fn leaf(value: f64) -> TempNode {
        TempNode {
            value,
            left: -1,
            right: -1,
            feature: 0,
            flags: 0,
            cat_n_words: 0,
            weight: 1.0,
        }
    }

    fn split(feature: u16, left: i16, right: i16) -> TempNode {
        TempNode {
            value: 0.0,
            left,
            right,
            feature,
            flags: build_flags(
                false,
                false,
                false,
                if feature == 0 {
                    SPLIT_NON_MONO
                } else {
                    SPLIT_CONSTANT
                },
            ),
            cat_n_words: 0,
            weight: 1.0,
        }
    }

    fn paired() -> Vec<TempNode> {
        vec![
            split(0, 1, 2),
            split(1, 3, 4),
            split(1, 5, 6),
            leaf(1.0),
            leaf(2.0),
            leaf(3.0),
            leaf(4.0),
        ]
    }

    // Independent interpreter of the input tree, without layout/partial eval.
    fn evaluate(nodes: &[TempNode], mut i: usize, row: &[f64; 3], f32_inputs: bool) -> u64 {
        while nodes[i].left >= 0 {
            let n = nodes[i];
            let x = row[n.feature as usize];
            let left = if x.is_nan() {
                n.flags & FLAG_DEFAULT_LEFT != 0
            } else if f32_inputs {
                (x as f32) < (n.value as f32)
            } else {
                x <= n.value
            };
            i = if left { n.left } else { n.right } as usize;
        }
        nodes[i].value.to_bits()
    }

    #[test]
    fn paired_hoist_preserves_leaf_identity_and_combines_weights() {
        let original = paired();
        let mut nodes = original.clone();
        let mut stats = HoistStats::default();
        hoist(&mut nodes, 0, &[], &mut stats);
        assert_eq!(stats.swaps, 1);
        assert_eq!(stats.nodes_before, stats.nodes_after);
        assert_eq!(
            (stats.varying_roots_before, stats.varying_roots_after),
            (1, 0)
        );
        assert_eq!(nodes[0].feature, 1);
        assert_eq!(nodes[1].weight.to_bits(), 2.0f64.to_bits());
        assert_eq!(nodes[2].weight.to_bits(), 2.0f64.to_bits());
        for x in [
            -1.0,
            -0.0,
            0.0,
            f64::from_bits(1),
            1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            for c in [-1.0, 0.0, 1.0, f64::NAN] {
                for f32_inputs in [false, true] {
                    assert_eq!(
                        evaluate(&original, 0, &[x, c, 0.0], f32_inputs),
                        evaluate(&nodes, 0, &[x, c, 0.0], f32_inputs)
                    );
                }
            }
        }
    }

    #[test]
    fn shared_leaf_eliminates_varying_split() {
        let mut nodes = paired();
        nodes[5].value = nodes[3].value;
        let mut stats = HoistStats::default();
        hoist(&mut nodes, 0, &[], &mut stats);
        assert_eq!(stats.swaps, 1);
        assert_eq!(stats.collapsed_splits, 1);
        assert_eq!(stats.nodes_after, 5);
        assert!(nodes[nodes[0].left as usize].left < 0);
    }

    #[test]
    fn differing_missing_directions_thresholds_and_signed_zero_do_not_match() {
        for differing in ["missing", "threshold", "zero", "feature"] {
            let mut nodes = paired();
            match differing {
                "missing" => nodes[2].flags |= FLAG_DEFAULT_LEFT,
                "threshold" => nodes[2].value = 1.0,
                "zero" => nodes[2].value = -0.0,
                _ => nodes[2].feature = 2,
            }
            let mut stats = HoistStats::default();
            hoist(&mut nodes, 0, &[], &mut stats);
            assert_eq!(stats.swaps, 0, "{differing}");
            assert_eq!(stats.trees_changed, 0);
        }
        let mut nodes = vec![split(0, 1, 2), leaf(0.0), leaf(-0.0)];
        let mut stats = HoistStats::default();
        hoist(&mut nodes, 0, &[], &mut stats);
        assert_eq!(stats.collapsed_splits, 0);
    }

    #[test]
    fn categorical_matching_uses_contents_even_without_interning() {
        let mut nodes = paired();
        let mut bitsets = vec![0u8; 16];
        bitsets[0] = 1;
        bitsets[8] = 1;
        for (i, offset) in [(1, 0), (2, 8)] {
            nodes[i].flags = build_flags(true, true, false, SPLIT_CONSTANT);
            nodes[i].value = f64::from(offset);
            nodes[i].cat_n_words = 2;
        }
        let mut different = nodes.clone();
        let mut stats = HoistStats::default();
        hoist(&mut nodes, 0, &bitsets, &mut stats);
        assert_eq!(stats.swaps, 1);
        bitsets[8] = 2;
        let mut stats = HoistStats::default();
        hoist(&mut different, 0, &bitsets, &mut stats);
        assert_eq!(stats.swaps, 0);
    }

    #[test]
    fn nonzero_root_and_identical_subtrees_collapse_without_recursion() {
        let mut nodes = vec![
            leaf(99.0),
            split(0, 2, 3),
            split(1, 4, 5),
            split(1, 6, 7),
            leaf(1.0),
            leaf(2.0),
            leaf(1.0),
            leaf(2.0),
        ];
        let mut stats = HoistStats::default();
        hoist(&mut nodes, 1, &[], &mut stats);
        assert_eq!(stats.nodes_after, 3);
        assert_eq!(nodes[1].feature, 1);
        assert_eq!(nodes[0].value.to_bits(), 99.0f64.to_bits());
        let mut second = HoistStats::default();
        hoist(&mut nodes, 1, &[], &mut second);
        assert_eq!(second.trees_changed, 0);
    }

    #[test]
    fn randomized_trees_preserve_every_reached_leaf() {
        fn next(state: &mut u64) -> u64 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            *state >> 32
        }
        let mut seed = 42;
        let mut total_swaps = 0;
        let mut total_collapses = 0;
        for _ in 0..256 {
            let mut original = vec![leaf(0.0); 63];
            for (i, node) in original.iter_mut().enumerate() {
                *node = if i < 31 {
                    let mut n = split(
                        (next(&mut seed) % 3) as u16,
                        (2 * i + 1) as i16,
                        (2 * i + 2) as i16,
                    );
                    n.value = (next(&mut seed) % 3) as f64 - 1.0;
                    if next(&mut seed) & 1 != 0 {
                        n.flags |= FLAG_DEFAULT_LEFT;
                    }
                    n
                } else {
                    leaf((next(&mut seed) % 4) as f64 - 2.0)
                };
            }
            let mut nodes = original.clone();
            let mut stats = HoistStats::default();
            hoist(&mut nodes, 0, &[], &mut stats);
            total_swaps += stats.swaps;
            total_collapses += stats.collapsed_splits;
            assert!(stats.nodes_after <= stats.nodes_before);
            for x in [
                f64::NEG_INFINITY,
                -1.0,
                -0.0,
                0.0,
                f64::from_bits(1),
                1.0,
                f64::INFINITY,
                f64::NAN,
            ] {
                for c in [-1.0, 0.0, 1.0, f64::NAN] {
                    for d in [-1.0, 1.0, f64::NAN] {
                        for f32_inputs in [false, true] {
                            let row = [x, c, d];
                            assert_eq!(
                                evaluate(&original, 0, &row, f32_inputs),
                                evaluate(&nodes, 0, &row, f32_inputs)
                            );
                        }
                    }
                }
            }
        }
        assert!(total_swaps > 0 && total_collapses > 0);
    }

    #[test]
    fn deep_hoisting_stops_at_pass_budget_and_remains_exact() {
        let mut original = vec![split(0, 0, 0)];
        let mut roots = [0; 2];
        for (side, root) in roots.iter_mut().enumerate() {
            let mut child = original.len() as i16;
            original.push(leaf(10.0 + side as f64));
            for depth in 0..32 {
                let other = original.len() as i16;
                original.push(leaf(100.0f64.mul_add(side as f64, f64::from(depth))));
                let mut node = split(1, child, other);
                node.value = f64::from(depth);
                child = original.len() as i16;
                original.push(node);
            }
            *root = child;
        }
        original[0].left = roots[0];
        original[0].right = roots[1];
        let mut nodes = original.clone();
        let mut stats = HoistStats::default();
        hoist(&mut nodes, 0, &[], &mut stats);
        assert_eq!(stats.pass_limit_hits, 1);
        assert_eq!(stats.swaps, MAX_PASSES);
        assert_eq!(stats.nodes_after, stats.nodes_before);
        for c in [-1.0, 0.0, 15.0, 32.0, f64::NAN] {
            for x in [-1.0, 1.0, f64::NAN] {
                let row = [x, c, 0.0];
                assert_eq!(
                    evaluate(&original, 0, &row, false),
                    evaluate(&nodes, 0, &row, false)
                );
            }
        }
    }
}
