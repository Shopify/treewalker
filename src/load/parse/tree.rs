//! A tree as either decoder reads it, and its validation and heavy-first order.
//!
//! Both the JSON and binary parsers convert framework-specific tree representations
//! into `TempNode` arrays, then call [`order_tree`], which validates each tree and
//! adds it to the decoded forest in heavy-first preorder.

use std::hint::select_unpredictable;

use rustc_hash::FxHashMap as HashMap;

use super::decoded::{Decoded, Packed};
use crate::LoadError;
use crate::config::WalkerConfig;
use crate::model::{
    FLAG_CATEGORICAL, FLAG_DEFAULT_LEFT, FLAG_INLINE_CAT, FLAG_VARYING, NONE, ThresholdType,
};

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Intermediate node representation before heavy-path reordering.
///
/// Both parsers build a `Vec<TempNode>` per tree (preserving original tree
/// structure with left/right child indices), then [`order_tree`] adds it to the
/// decoded forest in heavy-first preorder.
pub struct TempNode {
    pub value: f64,
    /// Children's indices in the tree; [`NONE`] for a leaf.
    pub left: u32,
    pub right: u32,
    pub feature: u32,
    pub flags: u8,
    pub weight: f64,
}

impl TempNode {
    /// A leaf with `value` and training `weight`.
    pub const fn leaf(value: f64, weight: f64) -> Self {
        Self {
            value,
            left: NONE,
            right: NONE,
            feature: 0,
            flags: 0,
            weight,
        }
    }
}

/// Mutable context threaded through tree parsing.
///
/// Holds the decoded forest, which the parsers fill one tree at a time, and
/// model-level configuration.
pub struct ParseContext<'a> {
    pub decoded: &'a mut Decoded,
    pub list_intern: Option<&'a mut HashMap<Vec<u32>, usize>>,
    pub config: &'a WalkerConfig,
    pub threshold_type: ThresholdType,
    /// A split's categories, sorted and distinct.
    pub list: Vec<u32>,
}

/// Reusable scratch buffers for [`order_tree`]. Allocated once, cleared per tree.
#[derive(Default)]
pub struct OrderScratch {
    /// Each node's position in heavy-first preorder, and whether its heavy child is
    /// its left one; one more entry, which leaves write.
    position: Vec<u32>,
    /// The tree's packed nodes, by position.
    packed: Vec<Packed>,
    heavy_left: Vec<bool>,
    /// Subtree sizes and parents, with an entry for leaves' missing children.
    size: Vec<u32>,
    parent: Vec<u32>,
    /// The splits' and the leaves' indices.
    splits: Vec<u32>,
    leaves: Vec<u32>,
    visit_order: Vec<usize>,
    stack: Vec<usize>,
    validation_seen: Vec<bool>,
    validation_stack: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Threshold normalization
// ---------------------------------------------------------------------------

/// Largest IEEE 754 float strictly less than `x`.
///
/// Used to normalize LightGBM's `<` comparisons to `<=`:
/// for all finite IEEE 754 floats, `val < threshold` ⟺ `val <= next_down(threshold)`.
/// This lets the runtime use a single `<=` comparison for all F64 models.
/// Not applied to F32 models — XGBoost's `<` is handled directly via `(val as f32) < (thr as f32)`.
pub fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1); // smallest negative subnormal
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

// ---------------------------------------------------------------------------
// Feature classification
// ---------------------------------------------------------------------------

/// Build node flags from split properties.
#[expect(
    clippy::fn_params_excessive_bools,
    reason = "one parameter per flag bit"
)]
pub const fn build_flags(default_left: bool, is_cat: bool, inline: bool, varying: bool) -> u8 {
    (default_left as u8 * FLAG_DEFAULT_LEFT)
        | (is_cat as u8 * FLAG_CATEGORICAL)
        | (inline as u8 * FLAG_INLINE_CAT)
        | (varying as u8 * FLAG_VARYING)
}

// ---------------------------------------------------------------------------
// Categorical encoding
// ---------------------------------------------------------------------------

/// Encode a split's categories: inline as a u32 word when every category is below
/// 32, otherwise as a list in `ctx.categories`, its length then its categories,
/// sorted and distinct. Lists are interned unless `disable_bitset_intern`.
///
/// Returns `(value, is_inline)`: the word, or the list's offset, as f64 (exact
/// below 2^53). The list pool grows with the input's entries only.
pub fn encode_categories(
    categories: impl IntoIterator<Item = u32>,
    ctx: &mut ParseContext<'_>,
) -> Result<(f64, bool), LoadError> {
    let list = &mut ctx.list;
    list.clear();
    list.extend(categories);
    list.sort_unstable();
    list.dedup();
    if list.last().is_none_or(|&c| c < 32) {
        let word = list.iter().fold(0u32, |w, &c| w | 1 << c);
        return Ok((f64::from(word), true));
    }
    let interned = ctx
        .list_intern
        .as_ref()
        .and_then(|intern| intern.get(list.as_slice()).copied());
    let offset = if let Some(offset) = interned {
        offset
    } else {
        let categories = &mut ctx.decoded.categories;
        let offset = categories.len();
        categories
            .try_reserve(list.len() + 1)
            .map_err(|e| LoadError::Limit(e.to_string()))?;
        categories.push(list.len() as u32);
        categories.extend_from_slice(list);
        if let Some(intern) = ctx.list_intern.as_mut() {
            intern.insert(list.clone(), offset);
        }
        offset
    };
    Ok((offset as f64, false))
}

// ---------------------------------------------------------------------------
// Heavy-first order
// ---------------------------------------------------------------------------

/// Validate a tree, put its nodes in heavy-first preorder and add it to the decoded
/// forest.
///
/// In heavy-first preorder the heavier child (by training `weight`, the left one on
/// ties) follows its parent, so walks that go heavy read the next node; the light
/// child follows the heavy child's subtree.
pub fn order_tree(
    temp: &[TempNode],
    root_idx: usize,
    category_start: u32,
    ctx: &mut ParseContext<'_>,
    scratch: &mut OrderScratch,
) -> Result<(), LoadError> {
    if !(root_idx == 0 && forward_positions(temp, scratch)) {
        validate_tree(temp, root_idx, scratch)?;
        search_positions(temp, root_idx, scratch);
    }
    let decoded = &mut *ctx.decoded;
    check_node_total(decoded.node_count(), temp.len())?;
    let (splits, extent) = decoded.parts();
    // Packed here by position, where the scattered writes stay in cache, then
    // appended in order.
    let nodes = &mut scratch.packed;
    nodes.clear();
    nodes.resize(temp.len(), Packed::leaf(0.0));
    let position = &scratch.position;
    // Leaves and splits apart, so that neither loop branches on the kind.
    for &i in &scratch.leaves {
        let value = temp[i as usize].value;
        nodes[position[i as usize] as usize] = Packed::leaf(value);
        extent.add(value);
    }
    let mut varying = 0;
    for &i in &scratch.splits {
        let tn = &temp[i as usize];
        let hil = scratch.heavy_left[i as usize];
        let (heavy, light) = if hil {
            (tn.left, tn.right)
        } else {
            (tn.right, tn.left)
        };
        let share = heavy_share(temp[heavy as usize].weight, temp[light as usize].weight);
        let id = splits.intern(tn.value, tn.feature, tn.flags);
        if tn.flags & FLAG_CATEGORICAL != 0 {
            splits.heavy_side(id, hil);
        }
        varying += u32::from(tn.flags & FLAG_VARYING != 0);
        nodes[position[i as usize] as usize] = Packed::split(id, share, hil);
    }
    decoded.push_tree(
        &scratch.packed,
        category_start,
        scratch.leaves.len() as u32,
        varying,
    )
}

/// Positions when every child's index exceeds its parent's and the root is node 0,
/// as in Treelite's own trees: subtree sizes from the last node back, then
/// positions from the root forward, without a stack or data-dependent branches,
/// and the splits and leaves listed apart. Returns false, for [`validate_tree`] and
/// [`search_positions`], unless the nodes form one binary tree that way.
///
/// Such nodes form a tree exactly when every node has zero or two distinct
/// children, there are `n − 1` child links, and each child has one parent: links
/// only go forward, so the links reach every node but the root, and following
/// parents from any node ends at the root. A child claimed by two splits keeps the
/// first one as its parent, which the second then fails to find.
///
/// Where a node's child is the next node, as in a depth-first numbering, the loops
/// carry the child's size or position in a register rather than through memory.
#[expect(
    clippy::needless_bitwise_bool,
    reason = "every node takes the same path: no branch on its kind"
)]
fn forward_positions(temp: &[TempNode], s: &mut OrderScratch) -> bool {
    let n = temp.len();
    s.size.clear();
    s.size.resize(n + 1, 0);
    s.parent.clear();
    s.parent.resize(n + 1, NONE);
    // Slices of known length, so that indices clamped to `n` need no bounds checks.
    let (size, parent) = (&mut s.size[..=n], &mut s.parent[..=n]);
    let (mut ok, mut next) = (true, 0u32);
    for (i, t) in temp.iter().enumerate().rev() {
        let (l, r) = (t.left as usize, t.right as usize);
        let leaf = t.left == NONE && t.right == NONE;
        ok &= leaf | ((i < l) & (l < n) & (i < r) & (r < n) & (l != r));
        let (l, r) = (l.min(n), r.min(n));
        let size_l = select_unpredictable(l == i + 1, next, size[l]);
        let size_r = select_unpredictable(r == i + 1, next, size[r]);
        next = 1u32.wrapping_add(size_l).wrapping_add(size_r);
        size[i] = next;
        parent[l] = i as u32;
        parent[r] = i as u32;
    }
    if !ok {
        return false;
    }
    s.position.clear();
    s.position.resize(n + 1, 0);
    s.heavy_left.clear();
    s.heavy_left.resize(n, false);
    s.splits.clear();
    s.splits.resize(n, 0);
    s.leaves.clear();
    s.leaves.resize(n, 0);
    let (position, heavy_left) = (&mut s.position[..=n], &mut s.heavy_left[..n]);
    let (splits, leaves) = (&mut s.splits[..n], &mut s.leaves[..n]);
    let (mut at, mut n_splits, mut n_leaves) = (0u32, 0usize, 0usize);
    for (i, t) in temp.iter().enumerate() {
        let split = t.left != NONE;
        let (l, r) = if split {
            ((t.left as usize).min(n), (t.right as usize).min(n))
        } else {
            (n, n)
        };
        ok &= !split | ((parent[l] == i as u32) & (parent[r] == i as u32));
        splits[n_splits.min(n - 1)] = i as u32;
        leaves[n_leaves.min(n - 1)] = i as u32;
        n_splits += usize::from(split);
        n_leaves += usize::from(!split);
        let hil = split & (temp[l.min(n - 1)].weight >= temp[r.min(n - 1)].weight);
        let heavy = select_unpredictable(hil, l, r);
        let light = select_unpredictable(hil, r, l);
        let (at_heavy, at_light) = (at + 1, at + 1 + size[heavy]);
        position[i] = at;
        position[heavy] = at_heavy;
        position[light] = at_light;
        heavy_left[i] = hil;
        let stored = position[(i + 1).min(n)];
        at = select_unpredictable(
            heavy == i + 1,
            at_heavy,
            select_unpredictable(light == i + 1, at_light, stored),
        );
    }
    s.splits.truncate(n_splits);
    s.leaves.truncate(n_leaves);
    ok && 2 * n_splits + 1 == n
}

/// Positions by a depth-first search from `root_idx`, heavy child first, for trees
/// [`validate_tree`] accepted.
fn search_positions(temp: &[TempNode], root_idx: usize, s: &mut OrderScratch) {
    let total = temp.len();
    s.visit_order.clear();
    s.heavy_left.clear();
    s.heavy_left.resize(total, false);
    s.position.clear();
    s.position.resize(total, 0);
    s.stack.clear();
    s.stack.push(root_idx);

    while let Some(old_idx) = s.stack.pop() {
        s.visit_order.push(old_idx);
        let tn = &temp[old_idx];
        if tn.left == NONE {
            continue;
        }
        let (l, r) = (tn.left as usize, tn.right as usize);
        if temp[l].weight >= temp[r].weight {
            s.heavy_left[old_idx] = true;
            s.stack.push(r);
            s.stack.push(l);
        } else {
            s.stack.push(l);
            s.stack.push(r);
        }
    }
    s.splits.clear();
    s.leaves.clear();
    for (new_idx, &old_idx) in s.visit_order.iter().enumerate() {
        s.position[old_idx] = new_idx as u32;
        if temp[old_idx].left == NONE {
            s.leaves.push(old_idx as u32);
        } else {
            s.splits.push(old_idx as u32);
        }
    }
}

/// The heavy child's share of its parent's training weight, `heavy / (heavy +
/// light)`, in 1/65535ths; one half when neither child has weight.
fn heavy_share(heavy: f64, light: f64) -> u16 {
    let total = heavy + light;
    let share = if total > 0.0 { heavy / total } else { 0.5 };
    (share.clamp(0.0, 1.0) * f64::from(u16::MAX)).round() as u16
}

/// Node indices are 32-bit, and the layout appends a dummy node whose heavy child
/// index, one past it, must fit too: `existing` nodes and `new` ones stay below
/// `u32::MAX`.
pub(crate) fn check_node_total(existing: usize, new: usize) -> Result<(), LoadError> {
    if existing
        .checked_add(new)
        .is_none_or(|n| n >= u32::MAX as usize)
    {
        return Err(LoadError::Limit(format!(
            "more than {} nodes: node indices are 32-bit",
            u32::MAX - 1
        )));
    }
    Ok(())
}

/// Validate topology iteratively before the layout or the full walk sees it.
fn validate_tree(
    nodes: &[TempNode],
    root: usize,
    scratch: &mut OrderScratch,
) -> Result<(), LoadError> {
    // Children are u32 indices with NONE for a leaf; the node pool's bound in
    // order_tree keeps every tree below it.
    if nodes.is_empty() {
        return Err(LoadError::MalformedModel("a tree has no nodes".into()));
    }
    let visited = &mut scratch.validation_seen;
    visited.clear();
    visited.resize(nodes.len(), false);
    let stack = &mut scratch.validation_stack;
    stack.clear();
    stack.push(root);
    let mut count = 0;
    while let Some(i) = stack.pop() {
        if i >= nodes.len() {
            return Err(LoadError::MalformedModel(format!(
                "child/root index {i} out of range"
            )));
        }
        if visited[i] {
            return Err(LoadError::MalformedModel(format!(
                "node {i}: cycle or shared child"
            )));
        }
        visited[i] = true;
        count += 1;
        let n = &nodes[i];
        if n.left == NONE && n.right == NONE {
            continue;
        }
        if n.left == NONE || n.right == NONE {
            return Err(LoadError::MalformedModel(format!(
                "node {i}: invalid children {}, {}",
                n.left, n.right
            )));
        }
        stack.push(n.right as usize);
        stack.push(n.left as usize);
    }
    if count != nodes.len() {
        return Err(LoadError::MalformedModel(format!(
            "{} unreachable nodes",
            nodes.len() - count
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The node total's bound, without building 4 billion nodes.
    #[test]
    fn node_totals_stop_below_u32_max() {
        let max = u32::MAX as usize - 1;
        assert!(check_node_total(max - 10, 10).is_ok());
        assert!(check_node_total(32_000_001, 1).is_ok());
        assert!(matches!(check_node_total(max, 1), Err(LoadError::Limit(_))));
        assert!(matches!(
            check_node_total(usize::MAX, 1),
            Err(LoadError::Limit(_))
        ));
    }
}
