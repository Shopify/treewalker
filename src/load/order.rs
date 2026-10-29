//! Tree order: the decoded trees sorted by their heavy paths, and the category lists
//! renumbered in that order.

use rustc_hash::FxHashMap as HashMap;

use super::parse::decoded::{Decoded, Packed};
use crate::model::FLAG_HEAVY_IS_LEFT;
use crate::rules::{PathNode, Paths, Token, compare_paths};

/// Heavy-path tokens kept per tree for the sort, which the sort compares in place;
/// ties compare the rest of the paths from the decoded nodes.
const ORDER_PATH_DEPTH: usize = 8;

/// The token of a path shorter than [`ORDER_PATH_DEPTH`] past its end: above every
/// split's, as no split has feature `u32::MAX`.
const EMPTY_PATH_TOKEN: Token = (u32::MAX, u8::MAX, u64::MAX);

/// A tree's first [`ORDER_PATH_DEPTH`] heavy-path nodes, ordered lexicographically
/// from the root.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TreePathSummary {
    /// Each node's split identity (feature, flags and threshold bits, or the category
    /// word's or list's), so that trees with the same splits at the same depths sort
    /// next to each other.
    tokens: [Token; ORDER_PATH_DEPTH],
    len: u8,
}

/// The order of the decoded trees: by a lexicographic sort on their heavy paths, or
/// as decoded without `ordering` or when the sort keeps them so.
///
/// Trees with the same first splits sort next to each other, so that every set of
/// trees sharing a constant prefix is a run, as prefix groups need, and consecutive
/// walks start alike. O(T log T).
pub(crate) fn tree_order(decoded: &Decoded, ordering: bool) -> Vec<u32> {
    let n = decoded.trees.len() as u32;
    if !ordering || n < 2 {
        return (0..n).collect();
    }
    let mut indexed: Vec<(TreePathSummary, u32)> = (0..n)
        .map(|t| (tree_path_summary(decoded, t as usize), t))
        .collect();

    // Ties on the first ORDER_PATH_DEPTH tokens compare the rest of the heavy paths'
    // constant splits, so that trees sharing a constant prefix of any depth are
    // adjacent, which prefix groups need.
    let paths = HeavyPaths {
        decoded,
        order: None,
    };
    indexed.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| compare_paths(&paths, a.1 as usize, b.1 as usize, ORDER_PATH_DEPTH))
    });
    indexed.into_iter().map(|(_, t)| t).collect()
}

/// Tree `t`'s first [`ORDER_PATH_DEPTH`] heavy-path splits.
fn tree_path_summary(decoded: &Decoded, t: usize) -> TreePathSummary {
    let mut tokens = [EMPTY_PATH_TOKEN; ORDER_PATH_DEPTH];
    let mut len = 0usize;
    for node in decoded.nodes(t).iter().take(ORDER_PATH_DEPTH) {
        if node.is_leaf() {
            break;
        }
        tokens[len] = token(decoded, node);
        len += 1; // heavy child is always idx + 1
    }
    TreePathSummary {
        tokens,
        len: len as u8,
    }
}

/// A split node's (feature, flags, value bits), with its parsed node's flags.
#[inline]
fn token(decoded: &Decoded, node: Packed) -> Token {
    let split = decoded.split(node);
    let side = if node.heavy_left() {
        FLAG_HEAVY_IS_LEFT
    } else {
        0
    };
    (split.feature, split.flags | side, split.value.to_bits())
}

/// The decoded trees' heavy paths, in input order or in `order`.
pub(crate) struct HeavyPaths<'a> {
    pub(crate) decoded: &'a Decoded,
    pub(crate) order: Option<&'a [u32]>,
}

impl Paths for HeavyPaths<'_> {
    fn len(&self) -> usize {
        self.decoded.trees.len()
    }

    #[inline]
    fn node(&self, t: usize, depth: usize) -> Option<PathNode> {
        let t = self.order.map_or(t, |o| o[t] as usize);
        let node = self.decoded.nodes(t).get(depth)?;
        Some(if node.is_leaf() {
            PathNode {
                token: None,
                heavy: 0,
            }
        } else {
            PathNode {
                token: (!self.decoded.split(node).is_varying()).then(|| token(self.decoded, node)),
                heavy: node.heavy(),
            }
        })
    }
}

/// Renumber the decoded pool's lists in tree `order`, as a pool rebuilt tree by tree
/// would hold them: each list where a node first names it. Listed splits' values
/// take the new offsets, and each tree the start of its lists. Returns the pool and
/// the starts by ordered tree. Trees in input order keep the decoded pool.
pub(crate) fn repack_categories(decoded: &mut Decoded, order: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let identity = order.iter().enumerate().all(|(i, &t)| t as usize == i);
    if identity {
        let starts = decoded.trees.iter().map(|t| t.category_start).collect();
        return (std::mem::take(&mut decoded.categories), starts);
    }
    let old = std::mem::take(&mut decoded.categories);
    let mut pool = Vec::with_capacity(old.len());
    let mut starts = Vec::with_capacity(order.len());
    let mut remap: HashMap<u32, u32> = HashMap::default();
    let mut moved = vec![false; decoded.splits.all.len()];
    for &t in order {
        starts.push(pool.len() as u32);
        for i in 0..decoded.firsts(t as usize).len() {
            let s = decoded.firsts(t as usize)[i] as usize;
            let split = &mut decoded.splits.all[s];
            if !split.listed() || moved[s] {
                continue;
            }
            let at = split.value as u32;
            let new = *remap.entry(at).or_insert_with(|| {
                let list = &old[at as usize..=at as usize + old[at as usize] as usize];
                let offset = pool.len() as u32;
                pool.extend_from_slice(list);
                offset
            });
            split.value = f64::from(new);
            moved[s] = true;
        }
    }
    (pool, starts)
}
