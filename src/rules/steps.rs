//! Expected walk steps per tree, from the training weights, which the prefetch distance
//! and the prefix groups' cost model read.

/// Expected lane steps per tree, over the trees, for a row distributed like the
/// training data: to the leaf, and to the first stop of a walk that stops at varying
/// splits. A step reads a node, so a leaf at depth d takes d + 1.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Steps {
    pub(crate) leaf: f64,
    pub(crate) stop: f64,
}

/// How the items of the walk without varying stops divide into short ones, which
/// the prefetch rate reads ([`super::prefetch::rate`]), for a row distributed like
/// the training data: an item is a tree, from its start to the leaf.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ItemLengths {
    /// Share of the trees whose leaf is 1, 2 and 3 steps from the item's start (the
    /// root, or where a row leaves the tree's prefix).
    pub(crate) leaf_short: [f64; SHORT],
}

/// Item lengths [`ItemLengths`] counts one by one: shorter than the longest prefetch rate
/// ([`super::prefetch::RATES`]), the only ones a rate can leave without a prefetch.
pub(crate) const SHORT: usize = 3;

/// [`Steps`] of a forest, from each split's heavy share, added tree by tree with
/// each tree's nodes in heavy-first preorder, so one forward pass carries each
/// node's probability of being reached to its children.
#[derive(Default)]
pub(crate) struct StepSums {
    leaf: f64,
    stop: f64,
    to_leaf: Vec<f64>,
    to_stop: Vec<f64>,
    /// Probability mass of the trees' leaves 1, 2, 3 and more steps from their item's
    /// start.
    leaf_ends: [f64; SHORT + 1],
    /// A light child's steps from its item's start on reaching it.
    to_depth: Vec<u32>,
}

impl StepSums {
    /// Add a tree of `n` nodes, each in order a leaf (`None`), or a split that goes
    /// heavy with probability `p`, to node `i + 1`, else to node `light`, and that
    /// a walk stopping at varying splits passes if `constant`.
    ///
    /// A split's heavy child is the next node and has no other parent, so its
    /// probabilities are carried in registers rather than stored and reloaded. No
    /// branch on the kind: a leaf, and a varying split for the stops, add nothing
    /// (`r × 0`: the probabilities are non-negative, so the sums are unchanged) and
    /// add to the entry past the tree's nodes. The sums are the ones of adding each
    /// node's probability, node by node, to its children.
    ///
    /// Each node also carries its steps from its item's start in the walk without
    /// varying stops, and a leaf adds its probability to the item lengths, capped at
    /// [`SHORT`] + 1. Items start at the root, or with a prefix of `prefix` splits
    /// (the tree's first `prefix` nodes, its heavy path), where a row leaves it: the
    /// light child of a prefix split or the node past the prefix.
    #[inline]
    pub(crate) fn tree(
        &mut self,
        n: usize,
        prefix: usize,
        node: impl Fn(usize) -> Option<(f64, usize, bool)>,
    ) {
        self.to_leaf.clear();
        self.to_leaf.resize(n + 1, 0.0);
        self.to_stop.clear();
        self.to_stop.resize(n + 1, 0.0);
        self.to_depth.clear();
        self.to_depth.resize(n + 1, 1);
        let (to_leaf, to_stop) = (&mut self.to_leaf[..=n], &mut self.to_stop[..=n]);
        let to_depth = &mut self.to_depth[..=n];
        let (mut rl, mut rs) = (1.0f64, 1.0f64);
        let mut depth = 1u32;
        for i in 0..n {
            self.leaf += rl;
            self.stop += rs;
            let split = node(i);
            let (p, light, constant) = split.unwrap_or((0.0, n, false));
            let light = light.min(n);
            let (ps, qs) = if constant { (p, 1.0 - p) } else { (0.0, 0.0) };
            to_leaf[light] = rl.mul_add(1.0 - p, to_leaf[light]);
            to_stop[light] = rs.mul_add(qs, to_stop[light]);
            let heavy = split.is_some();
            // A leaf ends its item.
            let ends = (depth.min(SHORT as u32 + 1) - 1) as usize;
            self.leaf_ends[ends] += if heavy { 0.0 } else { rl };
            let child_depth = if i < prefix { 1 } else { depth + 1 };
            to_depth[light] = child_depth;
            // The next node: this split's heavy child, reached only from it, or a
            // light child whose parent added to it before.
            let next = (i + 1).min(n);
            rl = std::hint::select_unpredictable(heavy, rl * p, to_leaf[next]);
            rs = std::hint::select_unpredictable(heavy, rs * ps, to_stop[next]);
            depth = std::hint::select_unpredictable(heavy, child_depth, to_depth[next]);
        }
    }

    /// The steps per tree over `trees` trees.
    pub(crate) fn steps(&self, trees: usize) -> Steps {
        let n = trees.max(1) as f64;
        Steps {
            leaf: self.leaf / n,
            stop: self.stop / n,
        }
    }

    /// The item lengths.
    pub(crate) fn items(&self) -> ItemLengths {
        let total: f64 = self.leaf_ends.iter().sum();
        ItemLengths {
            leaf_short: std::array::from_fn(|j| {
                if total > 0.0 {
                    self.leaf_ends[j] / total
                } else {
                    0.0
                }
            }),
        }
    }
}

/// Expected lane slots of a group of `width` rows, from the training weights, which
/// the packing rule reads ([`super::pack`]): each tree's nodes that a row of the group
/// reaches, and the round in which the lanes run out of items.
///
/// The rows of a group share the constant features, so they take a constant split's
/// side together, and they take a varying split's sides apart, each like the training
/// data, independently. A node is then reached by the group with the probability of
/// its constant path `c` and by at least one row of its varying path `v`: `c (1 − (1 −
/// v)^width)`, which a step of the group's walk reads once. The trees' last round leaves
/// up to `k − 1` lanes on the dummy for about a tree's steps. Over the categorical
/// cell's cuts of 1 to 20 trees at 16 rows this gave 308 to 2,226 lane slots where the
/// counted walk took 392 to 2,267, within 8% from 2 trees: the independence counts
/// about 30% more steps than the rows take, and the tail fewer.
#[derive(Default)]
pub(crate) struct GroupSums {
    /// The expected steps of the group's walk and of a row's, over the trees so far.
    pub(crate) group: f64,
    leaf: f64,
    trees: usize,
    /// The light children's path probabilities, constant and varying, of the splits
    /// whose heavy subtree the pass is in.
    pending: Vec<(f64, f64)>,
}

impl GroupSums {
    /// Add a tree for groups of `width` rows: its `nodes` in heavy-first preorder, each
    /// a leaf (`None`) or a split that goes heavy with probability `p` and is constant
    /// with `constant`. A split's heavy child is the next node, and the node after a
    /// leaf is the light child of the deepest split whose heavy subtree it ends, so a
    /// stack of the splits' light children carries the probabilities.
    pub(crate) fn tree(&mut self, width: usize, nodes: impl Iterator<Item = Option<(f64, bool)>>) {
        let width = i32::try_from(width).unwrap_or(i32::MAX);
        self.trees += 1;
        self.pending.clear();
        let (mut c, mut v) = (1.0f64, 1.0f64);
        for node in nodes {
            // `1 − (1 − v)^width` is `v` for a lone row or a constant path (`v`, at
            // most 1, is 1).
            let reach = if width == 1 || v >= 1.0 {
                v
            } else {
                1.0 - (1.0 - v).powi(width)
            };
            self.group = c.mul_add(reach, self.group);
            self.leaf = c.mul_add(v, self.leaf);
            match node {
                Some((p, true)) => {
                    self.pending.push((c * (1.0 - p), v));
                    c *= p;
                }
                Some((p, false)) => {
                    self.pending.push((c, v * (1.0 - p)));
                    v *= p;
                }
                None => (c, v) = self.pending.pop().unwrap_or((0.0, 0.0)),
            }
        }
    }

    /// The lane slots of a group on `lanes` lanes.
    pub(crate) fn lane_slots(&self, lanes: usize) -> f64 {
        let tail = lanes.saturating_sub(1) as f64 * self.leaf / self.trees.max(1) as f64;
        self.group + tail
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// A random tree of at most `depth` levels in heavy-first preorder, appended to
    /// `nodes`, with each node's probability of being reached by a group of `width`
    /// rows added to `group`: on its constant path, `c`, by any row on its varying
    /// one, `v`.
    fn random_tree(
        rng: &mut Rng,
        depth: u32,
        (c, v): (f64, f64),
        width: usize,
        nodes: &mut Vec<Option<(f64, bool)>>,
        group: &mut f64,
    ) {
        *group = c.mul_add(1.0 - (1.0 - v).powi(i32::try_from(width).unwrap()), *group);
        if depth == 0 || rng.chance(0.25) {
            nodes.push(None);
            return;
        }
        let (p, constant) = (rng.unit(), rng.chance(0.5));
        nodes.push(Some((p, constant)));
        let (heavy, light) = if constant {
            ((c * p, v), (c * (1.0 - p), v))
        } else {
            ((c, v * p), (c, v * (1.0 - p)))
        };
        random_tree(rng, depth - 1, heavy, width, nodes, group);
        random_tree(rng, depth - 1, light, width, nodes, group);
    }

    /// [`StepSums`] of one tree of `nodes` with a prefix of `prefix` splits.
    fn sums(nodes: &[Option<(f64, usize, bool)>], prefix: usize) -> StepSums {
        let mut sums = StepSums::default();
        sums.tree(nodes.len(), prefix, |i| nodes[i]);
        sums
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn a_single_leaf_takes_one_step() {
        let s = sums(&[None], 0);
        assert_eq!(
            s.steps(1),
            Steps {
                leaf: 1.0,
                stop: 1.0
            }
        );
        let short = s.items().leaf_short;
        assert!(close(short[0], 1.0) && close(short[1], 0.0) && close(short[2], 0.0));
    }

    #[test]
    fn steps_weigh_each_leaf_by_its_probability() {
        // Root 0.75 heavy to a split of 0.5, light to a leaf at depth 1: leaves at
        // depth 2 with 0.375 each and at depth 1 with 0.25, 3 × 0.75 + 2 × 0.25.
        let nodes = [
            Some((0.75, 4, true)),
            Some((0.5, 3, true)),
            None,
            None,
            None,
        ];
        let s = sums(&nodes, 0).steps(1);
        assert!(close(s.leaf, 2.75) && close(s.stop, 2.75), "{s:?}");
    }

    #[test]
    fn a_walk_with_stops_ends_at_the_first_varying_split() {
        // A varying root: 1 step to the stop, 2 to a leaf.
        let s = sums(&[Some((0.75, 2, false)), None, None], 0).steps(1);
        assert!(close(s.leaf, 2.0) && close(s.stop, 1.0), "{s:?}");
        // A constant root, heavy to a varying split: the stop at 2 steps with 0.75, the
        // light leaf at 2 with 0.25.
        let nodes = [
            Some((0.75, 4, true)),
            Some((0.5, 3, false)),
            None,
            None,
            None,
        ];
        let s = sums(&nodes, 0).steps(1);
        assert!(close(s.leaf, 2.75) && close(s.stop, 2.0), "{s:?}");
    }

    #[test]
    fn steps_per_tree_average_over_the_trees() {
        let mut s = StepSums::default();
        s.tree(1, 0, |_| None);
        let nodes = [
            Some((0.75, 4, true)),
            Some((0.5, 3, true)),
            None,
            None,
            None,
        ];
        s.tree(nodes.len(), 0, |i| nodes[i]);
        let steps = s.steps(2);
        assert!(
            close(steps.leaf, 1.875) && close(steps.stop, 1.875),
            "{steps:?}"
        );
        // Of the 2 trees' mass, the lone leaf's 1 is 1 step from the root, 0.25 is 2
        // steps and 0.75 is 3.
        let short = s.items().leaf_short;
        assert!(close(short[0], 0.5) && close(short[1], 0.125) && close(short[2], 0.375));
    }

    #[test]
    fn an_item_starts_where_a_row_leaves_the_prefix() {
        let nodes = [
            Some((0.75, 4, true)),
            Some((0.5, 3, true)),
            None,
            None,
            None,
        ];
        // Without a prefix the light leaf is 2 steps from the root, the others 3.
        let short = sums(&nodes, 0).items().leaf_short;
        assert!(close(short[0], 0.0) && close(short[1], 0.25) && close(short[2], 0.75));
        // With the root as the prefix, a row leaves it by the light child, a leaf 1
        // step from the item's start, or by the node past it, whose leaves are 2.
        let short = sums(&nodes, 1).items().leaf_short;
        assert!(close(short[0], 0.25) && close(short[1], 0.75) && close(short[2], 0.0));
        // The prefix changes the item lengths, not the steps.
        assert_eq!(sums(&nodes, 1).steps(1), sums(&nodes, 0).steps(1));
    }

    #[test]
    fn a_group_takes_a_constant_split_together_and_a_varying_one_apart() {
        // A constant split: the group's rows take one side, 2 steps as for a row.
        let mut sums = GroupSums::default();
        sums.tree(4, [Some((0.75, true)), None, None].into_iter());
        assert!((sums.group - 2.0).abs() < 1e-12);
        // A varying one: the heavy leaf unless all 4 rows go light, the light one unless
        // all go heavy.
        let mut sums = GroupSums::default();
        sums.tree(4, [Some((0.75, false)), None, None].into_iter());
        let group = 1.0 + (1.0 - 0.25f64.powi(4)) + (1.0 - 0.75f64.powi(4));
        assert!((sums.group - group).abs() < 1e-12);
        // The last round's idle lanes: 7 for a tree's 2 steps.
        assert!((sums.lane_slots(8) - group - 14.0).abs() < 1e-12);
    }

    #[test]
    fn group_steps_follow_each_nodes_paths() {
        let mut rng = Rng::new(41);
        for width in [1, 2, 16] {
            let mut sums = GroupSums::default();
            let (mut expected, mut before) = (0.0, 0.0);
            for _ in 0..50 {
                let mut nodes = Vec::new();
                random_tree(&mut rng, 8, (1.0, 1.0), width, &mut nodes, &mut expected);
                sums.tree(width, nodes.into_iter());
                assert!((sums.group - expected).abs() <= 1e-9 * expected);
                // Steps only add, so a pass that stops once they pass a bound decides
                // as the whole pass would.
                assert!(sums.group >= before);
                before = sums.group;
            }
            assert!(sums.lane_slots(8) >= sums.group);
        }
    }
}
