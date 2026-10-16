//! Prefix groups: runs of trees sharing their first heavy-path constant splits, which a
//! group evaluates once, and how deep each group goes.

use std::cmp::Ordering;

use crate::model::PrefixGroup;

/// How the load forms prefix groups: by [`prefix_groups`]' rule, or, for the research
/// ablations, every run of trees sharing exactly `k` heavy-path constant splits (0
/// for none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Prefix {
    Rule,
    Depth(usize),
}

/// What the prefix groups save per call, in lane steps, as the rule counts them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PrefixStats {
    pub(crate) groups: usize,
    pub(crate) trees: usize,
    /// The deepest group's depth, and the mean over grouped trees.
    pub(crate) max_depth: usize,
    pub(crate) mean_depth: f64,
    /// Expected walk steps the grouped trees skip per call.
    pub(crate) saved: f64,
    /// Expected cost of the prefix work per call, in walk steps.
    pub(crate) cost: f64,
}

/// The cost model of prefix starts, in lane steps of the walk (a node read, a bin
/// read, a compare and a select):
///
/// - evaluating one of a group's prefix nodes is one step: the same reads and
///   compare, outside the lanes;
/// - a group also costs [`GROUP_STEPS`] for its loop and the branch on where its
///   prefix bails, which mispredicts as often as rows leave the prefix;
/// - moving a tree's start past a prefix its rows all follow is a vector add over
///   the group's consecutive starts, a fraction of an instruction per tree: free;
/// - moving it to the light child where the prefix bails reads that node's word 1,
///   one of a step's two reads: [`BAIL_STEPS`].
///
/// A group of `n` trees whose rows reach `e` of its `k` prefix nodes on average,
/// and leave it before its end with probability `b`, skips `n × e` steps of the
/// walks (a tree that bails at node j skips j + 1 steps, as many as the group
/// evaluated) for `e + GROUP_STEPS + n × b × BAIL_STEPS`.
const GROUP_STEPS: f64 = 1.0;
const BAIL_STEPS: f64 = 0.5;

/// A heavy-path token: what makes two trees' splits at one depth the same split.
pub(crate) type Token = (u32, u8, u64);

/// A node of a tree's heavy path: its token if it can be part of a prefix (a
/// constant split), and its probability of going heavy.
#[derive(Clone, Copy)]
pub(crate) struct PathNode {
    pub(crate) token: Option<Token>,
    pub(crate) heavy: f64,
}

/// The trees' heavy paths: node `depth` of tree `t`'s heavy path is its node
/// `depth` in heavy-first preorder, if the tree has that many nodes.
pub(crate) trait Paths {
    fn len(&self) -> usize;
    fn node(&self, t: usize, depth: usize) -> Option<PathNode>;
}

/// Compare two trees' heavy paths of constant splits, lexicographically from depth
/// `from`; a path that ends sorts first.
pub(crate) fn compare_paths(
    paths: &impl Paths,
    first: usize,
    second: usize,
    from: usize,
) -> Ordering {
    let at = |t: usize, depth: usize| paths.node(t, depth).and_then(|n| n.token);
    let mut depth = from;
    loop {
        match (at(first, depth), at(second, depth)) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) => match x.cmp(&y) {
                Ordering::Equal => depth += 1,
                order => return order,
            },
        }
    }
}

/// A node of the trie of heavy paths: the sorted trees `lo..hi` share their first
/// `depth` constant splits.
struct Trie {
    lo: usize,
    hi: usize,
    depth: usize,
    /// Children: runs of at least two trees sharing the next split too.
    children: Vec<usize>,
    /// Expected prefix nodes evaluated, the probability of passing the whole prefix,
    /// and the best value of the subtree with its decision.
    reached: f64,
    passes: f64,
    best: f64,
    group: bool,
    merged: Vec<bool>,
}

/// Prefix groups of a forest and what they save.
///
/// With [`Prefix::Rule`], each tree's depth maximizes the steps saved net of the
/// prefix work under the cost model above, over the trie of the trees' heavy paths of
/// constant splits: at each trie node, the trees that share its splits either form a
/// group at its depth, each child subtree joining it or keeping its own best, or
/// leave the choice to the children. The training weights give each split's
/// probability of going heavy, as the mean over the group's trees of that split's
/// heavy share.
///
/// A group is a run of consecutive trees, so its starts are moved with a vector
/// operation; tree ordering sorts trees by their heavy paths, which makes every
/// set of trees sharing a prefix one run. Without tree ordering, groups split into
/// runs of consecutive trees.
/// `starts` holds each tree's root.
pub(crate) fn prefix_groups(
    paths: &impl Paths,
    starts: &[u32],
    prefix: Prefix,
) -> (Vec<PrefixGroup>, PrefixStats) {
    if prefix == Prefix::Depth(0) || paths.len() < 2 {
        return (Vec::new(), PrefixStats::default());
    }
    let mut order: Vec<u32> = (0..paths.len() as u32).collect();
    order.sort_by(|&a, &b| compare_paths(paths, a as usize, b as usize, 0));
    let node_at = |t: u32, d: usize| paths.node(t as usize, d);
    let tok = |t: u32, d: usize| node_at(t, d).and_then(|n| n.token);

    // The trie, in breadth-first order, so that children follow their parents.
    let mut trie = vec![Trie {
        lo: 0,
        hi: order.len(),
        depth: 0,
        children: Vec::new(),
        reached: 0.0,
        passes: 1.0,
        best: 0.0,
        group: false,
        merged: Vec::new(),
    }];
    let mut at = 0;
    while at < trie.len() {
        let (lo, hi, depth) = (trie[at].lo, trie[at].hi, trie[at].depth);
        if prefix == Prefix::Depth(depth) {
            at += 1;
            continue;
        }
        let mut i = lo;
        while i < hi {
            let Some(t) = tok(order[i], depth) else {
                i += 1;
                continue;
            };
            let mut j = i + 1;
            while j < hi && tok(order[j], depth) == Some(t) {
                j += 1;
            }
            if j - i >= 2 {
                // The split's probability of going heavy, over the run's trees.
                let p = order[i..j]
                    .iter()
                    .map(|&t| node_at(t, depth).map_or(0.5, |n| n.heavy))
                    .sum::<f64>()
                    / (j - i) as f64;
                let parent = &trie[at];
                let (reached, passes) = (parent.reached + parent.passes, parent.passes * p);
                trie.push(Trie {
                    lo: i,
                    hi: j,
                    depth: depth + 1,
                    children: Vec::new(),
                    reached,
                    passes,
                    best: 0.0,
                    group: false,
                    merged: Vec::new(),
                });
                let child = trie.len() - 1;
                trie[at].children.push(child);
            }
            i = j;
        }
        at += 1;
    }

    // Best values, children first.
    for v in (1..trie.len()).rev() {
        let node = &trie[v];
        let n = node.hi - node.lo;
        let in_children: usize = node.children.iter().map(|&c| trie[c].hi - trie[c].lo).sum();
        let separate: f64 = node.children.iter().map(|&c| trie[c].best).sum();
        let (e, b) = (node.reached, 1.0 - node.passes);
        let per_tree = b.mul_add(-BAIL_STEPS, e);
        let fixed = e + GROUP_STEPS;
        let rule = prefix == Prefix::Rule;
        let mut merged = Vec::with_capacity(node.children.len());
        let mut joined = n - in_children;
        let mut value = ((n - in_children) as f64).mul_add(per_tree, -fixed);
        for &c in &node.children {
            let size = trie[c].hi - trie[c].lo;
            let join = size as f64 * per_tree;
            // A fixed depth groups every tree at that depth: children are deeper.
            let merge = !rule || join >= trie[c].best;
            merged.push(merge);
            if merge {
                joined += size;
                value += join;
            } else {
                value += trie[c].best;
            }
        }
        let fixed_depth = prefix == Prefix::Depth(node.depth);
        let group = joined >= 2 && if rule { value > separate } else { fixed_depth };
        let node = &mut trie[v];
        node.group = group;
        node.best = if group { value } else { separate };
        node.merged = merged;
    }

    // Groups, top down: a grouped node takes its trees but those of children that
    // keep their own choice.
    let mut groups = Vec::new();
    let mut stats = PrefixStats::default();
    let mut stack: Vec<usize> = trie[0].children.clone();
    let mut members: Vec<u32> = Vec::new();
    while let Some(v) = stack.pop() {
        let node = &trie[v];
        if !node.group {
            stack.extend(&node.children);
            continue;
        }
        members.clear();
        let mut cursor = node.lo;
        for (&c, &merge) in node.children.iter().zip(&node.merged) {
            let child = &trie[c];
            if !merge {
                members.extend(&order[cursor..child.lo]);
                cursor = child.hi;
                stack.push(c);
            }
        }
        members.extend(&order[cursor..node.hi]);
        members.sort_unstable();
        let (e, b) = (node.reached, 1.0 - node.passes);
        for run in members.chunk_by(|a, b| b == &(a + 1)) {
            let n = run.len();
            let saved = n as f64 * e;
            let cost = (n as f64 * b).mul_add(BAIL_STEPS, e + GROUP_STEPS);
            if n < 2 || (prefix == Prefix::Rule && saved <= cost) {
                continue;
            }
            groups.push(PrefixGroup {
                node_base: starts[run[0] as usize],
                depth: node.depth as u32,
                first: run[0],
                end: run[n - 1] + 1,
            });
            stats.groups += 1;
            stats.trees += n;
            stats.max_depth = stats.max_depth.max(node.depth);
            stats.mean_depth += (n * node.depth) as f64;
            stats.saved += saved;
            stats.cost += cost;
        }
    }
    groups.sort_unstable_by_key(|g| g.first);
    if stats.trees > 0 {
        stats.mean_depth /= stats.trees as f64;
    }
    (groups, stats)
}
