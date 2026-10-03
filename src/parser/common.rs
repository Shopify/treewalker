//! Shared types and logic for model parsing.
//!
//! Both the JSON and binary parsers convert framework-specific tree representations
//! into `TempNode` arrays, then call `reorder_and_emit` to produce the optimized
//! in-memory layout with heavy-path fall-through encoding.

use rustc_hash::FxHashMap as HashMap;

use crate::config::WalkerConfig;
use crate::forest::{
    FLAG_CATEGORICAL, FLAG_DEFAULT_LEFT, FLAG_HEAVY_IS_LEFT, FLAG_INLINE_CAT,
    FLAG_VARYING_TYPE_SHIFT, FLAG_WALKABLE, Node, ThresholdType, Tree, SPLIT_CONSTANT, SPLIT_MONO_DEC,
    SPLIT_MONO_INC, SPLIT_NON_MONO,
};

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Intermediate node representation before heavy-path reordering.
///
/// Both parsers build a `Vec<TempNode>` per tree (preserving original tree
/// structure with left/right child indices), then `reorder_and_emit` produces
/// the optimized `Node` layout in the global pool.
#[derive(Clone, Copy)]
pub struct TempNode {
    pub value: f64,
    pub left: i16,
    pub right: i16,
    pub feature: u16,
    pub flags: u8,
    pub cat_n_words: u8,
    pub weight: f64,
}

/// Mutable context threaded through tree parsing.
///
/// Holds references to the global pools (nodes, bitsets) and model-level
/// configuration. Both parsers populate this incrementally, one tree at a time.
pub struct ParseContext<'a> {
    pub hoist_constants: bool,
    pub hoist_stats: &'a mut super::HoistStats,
    pub nodes: &'a mut Vec<Node>,
    pub bitsets: &'a mut Vec<u8>,
    pub bitset_intern: Option<&'a mut HashMap<Vec<u32>, usize>>,
    pub config: &'a WalkerConfig,
    pub leaf_bias: f64,
    pub threshold_type: ThresholdType,
}

/// Reusable scratch buffers for `reorder_and_emit`. Allocated once, cleared per tree.
pub struct ReorderScratch {
    pub visit_order: Vec<usize>,
    pub heavy_is_left: Vec<bool>,
    pub old_to_new: Vec<i16>,
    pub stack: Vec<usize>,
}

impl ReorderScratch {
    pub const fn new() -> Self {
        Self {
            visit_order: Vec::new(),
            heavy_is_left: Vec::new(),
            old_to_new: Vec::new(),
            stack: Vec::new(),
        }
    }
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

/// Classify a feature into its varying type for the node flags field.
pub const fn classify_feature(config: &WalkerConfig, feat: usize) -> u8 {
    if config.is_mono_inc(feat) {
        SPLIT_MONO_INC
    } else if config.is_mono_dec(feat) {
        SPLIT_MONO_DEC
    } else if config.is_varying(feat) {
        SPLIT_NON_MONO
    } else {
        SPLIT_CONSTANT
    }
}

/// Build node flags from split properties.
/// Sets `FLAG_WALKABLE` for constant internal nodes (varying_type == 0).
pub fn build_flags(default_left: bool, is_cat: bool, inline: bool, varying_type: u8) -> u8 {
    let walkable = if varying_type == 0 { FLAG_WALKABLE } else { 0 };
    (u8::from(default_left) * FLAG_DEFAULT_LEFT)
        | (u8::from(is_cat) * FLAG_CATEGORICAL)
        | (u8::from(inline) * FLAG_INLINE_CAT)
        | (varying_type << FLAG_VARYING_TYPE_SHIFT)
        | walkable
}

// ---------------------------------------------------------------------------
// Categorical bitset encoding
// ---------------------------------------------------------------------------

/// Encode a category list into either an inline u32 word or a pool bitset.
///
/// Returns `(value, is_inline, cat_n_words)`:
/// - Inline: `value` is the raw u32 word cast to f64, `cat_n_words` = 0.
/// - Pool: `value` is the byte offset into `ctx.bitsets`, `cat_n_words` = number of u32 words.
pub fn encode_categories(
    categories: &[u32],
    invert: bool,
    ctx: &mut ParseContext<'_>,
) -> (f64, bool, u8) {
    let max_cat = categories.iter().copied().max().unwrap_or(0);
    let n_words_needed = (max_cat / 32 + 1) as usize;

    if n_words_needed == 1 {
        let mut word = 0u32;
        for &cat in categories {
            word |= 1u32 << cat;
        }
        if invert {
            word = !word;
        }
        (f64::from(word), true, 0u8)
    } else {
        let mut words = vec![0u32; n_words_needed];
        for &cat in categories {
            words[(cat / 32) as usize] |= 1u32 << (cat % 32);
        }
        if invert {
            for w in &mut words {
                *w = !*w;
            }
        }

        let offset = if let Some(intern) = ctx.bitset_intern.as_mut() {
            if let Some(&existing) = intern.get(&words) {
                existing
            } else {
                let offset = ctx.bitsets.len();
                for word in &words {
                    ctx.bitsets.extend_from_slice(&word.to_le_bytes());
                }
                intern.insert(words, offset);
                offset
            }
        } else {
            let offset = ctx.bitsets.len();
            for word in &words {
                ctx.bitsets.extend_from_slice(&word.to_le_bytes());
            }
            offset
        };
        assert!(u8::try_from(n_words_needed).is_ok());
        (offset as f64, false, n_words_needed as u8)
    }
}

// ---------------------------------------------------------------------------
// Heavy-path DFS reorder + emit
// ---------------------------------------------------------------------------

/// Reorder a tree's TempNodes by heavy-path DFS and emit to the global pools.
///
/// Steps:
/// 1. Heavy-path DFS: visit nodes heavy-child-first so the heaviest subtree is
///    always at the next sequential index (fall-through). The heavy child is
///    chosen by training sample count (`weight`).
/// 2. Emit reordered nodes with fall-through encoding: the heavy child is always
///    at `idx + 1`, the light child is stored in `Node.skip`.
/// 3. Append to the global `nodes` pool.
pub fn reorder_and_emit(
    temp: &mut [TempNode],
    root_idx: usize,
    bitset_start: u32,
    ctx: &mut ParseContext<'_>,
    scratch: &mut ReorderScratch,
) -> Tree {
    if ctx.hoist_constants {
        super::hoist::hoist(temp, root_idx, ctx.bitsets, ctx.hoist_stats);
    }
    let total = temp.len();

    scratch.visit_order.clear();
    scratch.heavy_is_left.clear();
    scratch.heavy_is_left.resize(total, false);
    scratch.old_to_new.clear();
    scratch.old_to_new.resize(total, 0);
    scratch.stack.clear();
    scratch.stack.push(root_idx);

    while let Some(old_idx) = scratch.stack.pop() {
        scratch.visit_order.push(old_idx);
        let tn = &temp[old_idx];
        if tn.left == -1 {
            continue;
        }
        let (l, r) = (tn.left as usize, tn.right as usize);
        if temp[l].weight >= temp[r].weight {
            scratch.heavy_is_left[old_idx] = true;
            scratch.stack.push(r);
            scratch.stack.push(l);
        } else {
            scratch.stack.push(l);
            scratch.stack.push(r);
        }
    }

    for (new_idx, &old_idx) in scratch.visit_order.iter().enumerate() {
        scratch.old_to_new[old_idx] = new_idx as i16;
    }

    // Emit reordered nodes with fall-through encoding.
    let node_start = ctx.nodes.len() as u32;

    for &old_idx in &scratch.visit_order {
        let tn = &temp[old_idx];
        if tn.left == -1 {
            ctx.nodes.push(Node {
                value: tn.value,
                skip: -1,
                varying_pred_id: u16::MAX,
                feature: 0,
                flags: 0,
                cat_n_words: 0,
            });
        } else {
            let hil = scratch.heavy_is_left[old_idx];
            let skip = if hil {
                scratch.old_to_new[tn.right as usize]
            } else {
                scratch.old_to_new[tn.left as usize]
            };
            ctx.nodes.push(Node {
                value: tn.value,
                skip,
                varying_pred_id: u16::MAX,
                feature: tn.feature,
                flags: tn.flags | (u8::from(hil) * FLAG_HEAVY_IS_LEFT),
                cat_n_words: tn.cat_n_words,
            });
        }
    }

    Tree {
        node_start,
        node_count: scratch.visit_order.len() as u32,
        bitset_start,
    }
}
