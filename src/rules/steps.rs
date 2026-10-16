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
