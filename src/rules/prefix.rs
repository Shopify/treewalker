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
/// constant split), and its heavy share, in 65,535ths of the rows reaching it, which
/// the sorts that compare tokens do not read.
#[derive(Clone, Copy)]
pub(crate) struct PathNode {
    pub(crate) token: Option<Token>,
    pub(crate) heavy: u16,
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
    /// The trees' span in emission order, which need not be trie order.
    first: usize,
    end: usize,
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

/// The best merge choices, charging each run between cut children separately.
fn rule_choice(node: &Trie, trie: &[Trie], per_tree: f64, fixed: f64) -> (f64, Vec<bool>) {
    // The loader sorts ended paths last; compare_paths sorts them first. Use tree
    // spans when contiguous, and trie positions for the unordered ablation.
    let ordered = node.end - node.first == node.hi - node.lo
        && node.children.iter().all(|&c| {
            let child = &trie[c];
            child.end - child.first == child.hi - child.lo
        });
    let bounds = |n: &Trie| {
        if ordered {
            (n.first, n.end)
        } else {
            (n.lo, n.hi)
        }
    };
    let mut children: Vec<_> = node.children.iter().enumerate().collect();
    children.sort_unstable_by_key(|&(_, &c)| bounds(&trie[c]).0);
    let (mut cursor, end) = bounds(node);
    // Emitted: inside a paid run. Dropped: no group until the next cut child.
    let (mut emitted, mut dropped) = (-fixed, 0.0);
    let mut trace = Vec::with_capacity(children.len());
    for &(_, &c) in &children {
        let child = &trie[c];
        let (lo, hi) = bounds(child);
        emitted = ((lo - cursor) as f64).mul_add(per_tree, emitted);
        let cut_emitted = emitted > dropped;
        let cut = emitted.max(dropped) + child.best;
        let join = ((hi - lo) as f64).mul_add(per_tree, emitted);
        let merge = join > cut - fixed;
        trace.push((merge, cut_emitted));
        emitted = if merge { join } else { cut - fixed };
        dropped = cut;
        cursor = hi;
    }
    emitted = ((end - cursor) as f64).mul_add(per_tree, emitted);
    let mut in_run = emitted > dropped;
    let value = emitted.max(dropped);
    let mut merged = vec![false; node.children.len()];
    for (&(i, _), &(merge, cut_emitted)) in children.iter().zip(&trace).rev() {
        if in_run && merge {
            merged[i] = true;
        } else {
            in_run = cut_emitted;
        }
    }
    (value, merged)
}

/// Prefix groups of a forest and what they save.
///
/// With [`Prefix::Rule`], a dynamic program maximizes the steps saved net of the
/// prefix work under the cost model above, over the trie of the trees' heavy paths
/// of constant splits. At each trie node, each child subtree either joins a run at
/// the node's depth or keeps its own best and breaks the run. The program charges
/// each emitted run's prefix work separately, drops unprofitable runs, and compares
/// the plan with leaving all children separate.
/// The training weights give each split's probability of going heavy, as the mean
/// over the group's trees of that split's heavy share.
///
/// A group is a run of consecutive trees, so its starts are moved with a vector
/// operation; tree ordering sorts trees by their heavy paths, which makes every
/// set of trees sharing a prefix one run. Without tree ordering, groups split into
/// runs of consecutive trees; the rule's values assume tree ordering.
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
        first: 0,
        end: order.len(),
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
                    .map(|&t| {
                        node_at(t, depth).map_or(0.5, |n| f64::from(n.heavy) / f64::from(u16::MAX))
                    })
                    .sum::<f64>()
                    / (j - i) as f64;
                let parent = &trie[at];
                let (reached, passes) = (parent.reached + parent.passes, parent.passes * p);
                trie.push(Trie {
                    lo: i,
                    hi: j,
                    first: *order[i..j].iter().min().unwrap() as usize,
                    end: *order[i..j].iter().max().unwrap() as usize + 1,
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
        let separate: f64 = node.children.iter().map(|&c| trie[c].best).sum();
        let (e, b) = (node.reached, 1.0 - node.passes);
        let per_tree = b.mul_add(-BAIL_STEPS, e);
        let fixed = e + GROUP_STEPS;
        let rule = prefix == Prefix::Rule;
        let (value, merged) = if rule {
            rule_choice(node, &trie, per_tree, fixed)
        } else {
            // A fixed depth groups every tree at that depth: children are deeper.
            (
                (n as f64).mul_add(per_tree, -fixed),
                vec![true; node.children.len()],
            )
        };
        let fixed_depth = prefix == Prefix::Depth(node.depth);
        let group = n >= 2 && if rule { value > separate } else { fixed_depth };
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

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPaths {
        trees: Vec<Vec<u32>>,
        heavy: u16,
    }

    impl Paths for TestPaths {
        fn len(&self) -> usize {
            self.trees.len()
        }

        fn node(&self, t: usize, depth: usize) -> Option<PathNode> {
            self.trees[t].get(depth).map(|&split| PathNode {
                token: Some((split, 0, 0)),
                heavy: self.heavy,
            })
        }
    }

    fn plan(paths: &TestPaths, prefix: Prefix) -> (Vec<(u32, u32, u32)>, f64) {
        let starts: Vec<_> = (0..paths.len() as u32).collect();
        let (groups, stats) = prefix_groups(paths, &starts, prefix);
        (
            groups.iter().map(|g| (g.first, g.end, g.depth)).collect(),
            stats.saved - stats.cost,
        )
    }

    #[test]
    fn cut_child_charges_each_remaining_run() {
        let paths = TestPaths {
            trees: [0, 1, 2, 2, 2, 2, 2, 3, 4]
                .map(|second| vec![0, second])
                .to_vec(),
            heavy: 52_428,
        };
        let (groups, net) = plan(&paths, Prefix::Rule);
        let (_, shallow) = plan(&paths, Prefix::Depth(1));
        assert!(net + 1e-12 >= shallow, "rule {net}, depth 1 {shallow}");
        assert!((net - 6.1).abs() < 1e-12, "{net}");
        assert_eq!(groups, [(0, 9, 1)]);
    }

    /// Enumerate every merge/cut subset, scoring the actual emitted runs rather
    /// than keeping the two states of the rule's dynamic program.
    fn brute(paths: &TestPaths, members: &[usize], depth: usize) -> f64 {
        let mut by_token = std::collections::BTreeMap::<_, Vec<_>>::new();
        for &t in members {
            if let Some(node) = paths.node(t, depth) {
                by_token.entry(node.token.unwrap()).or_default().push(t);
            }
        }
        let children: Vec<_> = by_token
            .into_values()
            .filter(|trees| trees.len() >= 2)
            .collect();
        let best: Vec<_> = children
            .iter()
            .map(|trees| brute(paths, trees, depth + 1))
            .collect();
        if depth == 0 {
            return best.iter().sum();
        }
        let p = f64::from(paths.heavy) / f64::from(u16::MAX);
        let mut reached = 0.0;
        let mut passes = 1.0;
        for _ in 0..depth {
            reached += passes;
            passes *= p;
        }
        let per_tree = (1.0 - passes).mul_add(-BAIL_STEPS, reached);
        let fixed = reached + GROUP_STEPS;
        let mut best_value: f64 = 0.0;
        for cut in 0..1usize << children.len() {
            let mut remaining = members.to_vec();
            let mut value = 0.0;
            for (i, trees) in children.iter().enumerate() {
                if cut & (1 << i) != 0 {
                    value += best[i];
                    remaining.retain(|t| !trees.contains(t));
                }
            }
            for run in remaining.chunk_by(|a, b| *b == a + 1) {
                value += (run.len() as f64).mul_add(per_tree, -fixed).max(0.0);
            }
            best_value = best_value.max(value);
        }
        best_value
    }

    #[test]
    fn small_ordered_forests_match_all_merge_choices_and_beat_fixed_depths() {
        // Paths ending before their siblings sort last in the loader, rather than
        // first as compare_paths sorts them. Exercise both orders.
        let shapes = [
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1],
            vec![0, 2],
            vec![0],
            vec![1, 0],
            vec![1],
        ];
        let mut choices = [0usize; 9];
        loop {
            for heavy in [32_768, 52_428, u16::MAX] {
                let mut paths = TestPaths {
                    trees: choices.iter().map(|&i| shapes[i].clone()).collect(),
                    heavy,
                };
                for lexicographic in [false, true] {
                    if lexicographic {
                        paths.trees.sort();
                    }
                    let (_, net) = plan(&paths, Prefix::Rule);
                    for depth in 1..=3 {
                        let (_, fixed) = plan(&paths, Prefix::Depth(depth));
                        assert!(
                            net + 1e-10 >= fixed,
                            "{:?}, heavy {heavy}, rule {net}, depth {depth}: {fixed}",
                            paths.trees
                        );
                    }
                    let members: Vec<_> = (0..paths.len()).collect();
                    let best = brute(&paths, &members, 0);
                    assert!(
                        (net - best).abs() < 1e-10,
                        "{:?}, heavy {heavy}, rule {net}, brute {best}",
                        paths.trees
                    );
                }
            }
            let Some(i) = choices.iter().rposition(|&i| i + 1 < shapes.len()) else {
                break;
            };
            let next = choices[i] + 1;
            choices[i..].fill(next);
        }
    }
}
