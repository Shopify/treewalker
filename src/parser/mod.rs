//! Model parsing: treelite → optimized in-memory forest.
//!
//! # Parser pipeline
//!
//! 1. **Read model** — JSON (`dump_as_json()`) or binary (`serialize_bytes()`).
//! 2. **Per-tree heavy-path DFS reorder** — place the most-visited (heaviest)
//!    child at `idx + 1` so the constant walk follows sequential memory. The
//!    light child is reached via the `skip` field. This is the "fall-through"
//!    encoding: ~86% of constant-walk steps access the next 16 bytes in memory.
//! 3. **Cross-tree ordering** — permute trees so adjacent trees have similar
//!    root/early-path structure, improving branch predictor accuracy across
//!    tree transitions. Then **physically repack** both nodes and bitsets in
//!    the new tree order for contiguous memory layout.
//! 4. **Varying predicate deduplication** — extract and deduplicate all varying
//!    split predicates across the forest. Write back `varying_pred_id` into each node
//!    so the precompute pass evaluates ~5K unique predicates instead of ~50K nodes.
//! 5. **Constant-prefix grouping** — group trees sharing the same first K
//!    constant heavy-path splits. At predict time, shared splits are evaluated
//!    once per observation; each tree resumes `partial_eval` at the continuation
//!    index (K if all heavy, or the tree-specific `skip` if a light bail occurs).
//!
//! # Treelite normalization (Python side)
//!
//! The Python pipeline uses treelite to normalize framework-specific models:
//! ```text
//! treelite.frontend.load_lightgbm_model("model.txt").dump_as_json()
//! treelite.frontend.load_xgboost_model("model.json").dump_as_json()
//! ```
//! This handles framework quirks: LightGBM uses `<=` with f64 thresholds,
//! XGBoost uses `<` with f32 thresholds. The parser detects `threshold_type`
//! from the header and adjusts comparison semantics accordingly:
//! - F64 (`<=`): thresholds stored as-is. `<` converted to `<=` via `next_down`.
//! - F32 (`<`): thresholds stored as-is. Runtime comparison uses `(val as f32) < (thr as f32)`.

pub(crate) mod common;
mod binary;
mod json;

use rustc_hash::FxHashMap as HashMap;
use std::path::Path;

use crate::config::{ParseConfig, WalkerConfig};
use crate::forest::{
    FeatureRange, Node, PrefixGroup, ThresholdType, Tree, VaryingPredicate,
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Parse a treelite model file into global node/bitset pools.
///
/// Auto-detects format by file extension: `.bin` for treelite binary v4,
/// `.json` (or anything else) for treelite JSON.
///
/// Returns `(trees, nodes, bitsets, threshold_type)` where all trees share the contiguous pools.
/// Each tree's nodes are reordered by heavy-path DFS for cache-optimal traversal.
/// Trees are auto-ordered so trees with similar early-path behavior are adjacent.
pub fn parse_model(
    path: impl AsRef<Path>,
    config: &WalkerConfig,
    parse_config: &ParseConfig,
) -> (Vec<Tree>, Vec<Node>, Vec<u8>, ThresholdType) {
    let path = path.as_ref();

    let (mut trees, mut nodes, mut bitsets, threshold_type) = match path
        .extension()
        .and_then(|e| e.to_str())
    {
        Some("bin") => binary::parse(path, config, parse_config),
        _ => json::parse(path, config, parse_config),
    };

    if !parse_config.disable_tree_ordering {
        (trees, nodes, bitsets) = auto_order_trees(trees, nodes, bitsets);
    }

    (trees, nodes, bitsets, threshold_type)
}

// ---------------------------------------------------------------------------
// Varying predicate building
// ---------------------------------------------------------------------------

/// Build deduplicated varying predicates and write `varying_pred_id` back into each varying node.
///
/// Scans all nodes, extracts a `VaryingPredicate` from each non-constant internal node,
/// and deduplicates via HashMap. Two nodes with the same (feature, threshold/bitset,
/// default_left) share one predicate ID. This is critical for precompute efficiency:
/// the precompute pass evaluates each unique predicate once per observation, not
/// each node occurrence.
///
/// Uses `f64::to_bits()` for threshold hashing (handles -0.0 and NaN correctly).
/// `CatPool` dedup relies on bitset interning: content-identical bitsets share the
/// same pool offset, so offset equality implies content equality.
pub(crate) fn build_varying_predicates(nodes: &mut [Node]) -> Vec<VaryingPredicate> {
    let n_varying_nodes = nodes
        .iter()
        .filter(|n| !n.is_leaf() && !n.is_constant())
        .count();
    let mut pred_to_id: HashMap<VaryingPredicate, u16> =
        HashMap::with_capacity_and_hasher(n_varying_nodes / 2, rustc_hash::FxBuildHasher);
    let mut predicates = Vec::with_capacity(n_varying_nodes / 2);

    for node in nodes.iter_mut() {
        if node.is_leaf() || node.is_constant() {
            continue;
        }

        let default_left = node.default_left();
        let feature = node.feature;

        let pred = if node.is_categorical() {
            if node.inline_cat() {
                VaryingPredicate::CatInline {
                    feature,
                    default_left,
                    word: node.value as u32,
                }
            } else {
                VaryingPredicate::CatPool {
                    feature,
                    default_left,
                    offset: node.value as u32,
                    n_words: node.cat_n_words,
                }
            }
        } else {
            VaryingPredicate::Num {
                feature,
                threshold: node.value,
                default_left,
            }
        };

        let pred_id = if let Some(&id) = pred_to_id.get(&pred) {
            id
        } else {
            let new_id = predicates.len();
            assert!(
                new_id < u16::MAX as usize,
                "too many unique varying predicates: {new_id}"
            );
            let new_id = new_id as u16;
            pred_to_id.insert(pred, new_id);
            predicates.push(pred);
            new_id
        };
        node.varying_pred_id = pred_id;
    }

    predicates
}

/// Build varying predicates WITHOUT deduplication (ablation baseline).
///
/// Every varying node gets its own unique predicate ID. This inflates the
/// predicates vec from ~5K (deduplicated) to ~50K (one per varying node),
/// measuring the contribution of predicate deduplication to precompute cost.
pub(crate) fn build_varying_predicates_no_dedup(nodes: &mut [Node]) -> Vec<VaryingPredicate> {
    let mut predicates = Vec::new();

    for node in nodes.iter_mut() {
        if node.is_leaf() || node.is_constant() {
            continue;
        }

        let default_left = node.default_left();
        let feature = node.feature;

        let pred = if node.is_categorical() {
            if node.inline_cat() {
                VaryingPredicate::CatInline { feature, default_left, word: node.value as u32 }
            } else {
                VaryingPredicate::CatPool {
                    feature, default_left,
                    offset: node.value as u32, n_words: node.cat_n_words,
                }
            }
        } else {
            VaryingPredicate::Num { feature, threshold: node.value, default_left }
        };

        let new_id = predicates.len();
        assert!(
            new_id < u16::MAX as usize,
            "too many varying predicates without dedup ({new_id}): model too large for \
             disable_predicate_dedup (u16 limit). Use dedup or a smaller model.",
        );
        node.varying_pred_id = new_id as u16;
        predicates.push(pred);
    }

    predicates
}

// ---------------------------------------------------------------------------
// Predicate sorting and feature-range index
// ---------------------------------------------------------------------------

/// Sort key ordinal: numerical predicates sort before categoricals within each
/// feature so the sweep kernel can process the contiguous numerical range first.
const fn pred_kind_ordinal(pred: &VaryingPredicate) -> u8 {
    match pred {
        VaryingPredicate::Num { .. } => 0,
        VaryingPredicate::CatInline { .. } | VaryingPredicate::CatPool { .. } => 1,
    }
}

/// Sort `varying_predicates` by `(feature, kind, threshold)` and remap every
/// node's `varying_pred_id` to match the new positions.
///
/// After sorting, all predicates for the same feature are contiguous, with
/// numerical predicates (sorted by ascending threshold) before categoricals.
/// This layout enables the two-pointer sweep in `precompute_varying_masks`.
pub(crate) fn sort_varying_predicates(
    predicates: &mut Vec<VaryingPredicate>,
    nodes: &mut [Node],
) {
    if predicates.is_empty() {
        return;
    }

    // Build sort permutation (indices into the original vec).
    let mut sorted_indices: Vec<usize> = (0..predicates.len()).collect();
    sorted_indices.sort_unstable_by(|&a, &b| {
        let pa = &predicates[a];
        let pb = &predicates[b];
        pa.feature().cmp(&pb.feature())
            .then_with(|| pred_kind_ordinal(pa).cmp(&pred_kind_ordinal(pb)))
            .then_with(|| {
                // For numerical predicates: sort by threshold ascending.
                // For categoricals: arbitrary stable order (by raw bits).
                match (pa, pb) {
                    (
                        VaryingPredicate::Num { threshold: ta, .. },
                        VaryingPredicate::Num { threshold: tb, .. },
                    ) => ta.total_cmp(tb),
                    _ => std::cmp::Ordering::Equal,
                }
            })
    });

    // Build old→new mapping.
    let mut old_to_new = vec![0u16; predicates.len()];
    for (new_pos, &old_pos) in sorted_indices.iter().enumerate() {
        old_to_new[old_pos] = new_pos as u16;
    }

    // Reorder predicates vec.
    let reordered: Vec<VaryingPredicate> =
        sorted_indices.iter().map(|&i| predicates[i]).collect();
    *predicates = reordered;

    // Remap all nodes' varying_pred_id.
    for node in nodes.iter_mut() {
        if node.varying_pred_id != u16::MAX {
            node.varying_pred_id = old_to_new[node.varying_pred_id as usize];
        }
    }
}

/// Build the per-feature range index from sorted `varying_predicates`.
///
/// Each entry gives the contiguous `[num_start..num_end)` and `[cat_start..cat_end)`
/// ranges for one varying feature. Assumes predicates are sorted by
/// `(feature, kind, threshold)` via [`sort_varying_predicates`].
pub(crate) fn build_feature_ranges(predicates: &[VaryingPredicate]) -> Vec<FeatureRange> {
    if predicates.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut i = 0;
    while i < predicates.len() {
        let feature = predicates[i].feature();
        let group_start = i;

        // Scan numerical predicates (kind ordinal 0).
        let num_start = i;
        while i < predicates.len()
            && predicates[i].feature() == feature
            && pred_kind_ordinal(&predicates[i]) == 0
        {
            i += 1;
        }
        let num_end = i;

        // Scan categorical predicates (kind ordinal 1).
        let cat_start = i;
        while i < predicates.len()
            && predicates[i].feature() == feature
            && pred_kind_ordinal(&predicates[i]) == 1
        {
            i += 1;
        }
        let cat_end = i;

        debug_assert!(i > group_start, "empty feature group at index {group_start}");

        ranges.push(FeatureRange {
            feature,
            num_start: num_start as u32,
            num_end: num_end as u32,
            cat_start: cat_start as u32,
            cat_end: cat_end as u32,
        });
    }

    ranges
}

// ---------------------------------------------------------------------------
// Constant-prefix grouping
// ---------------------------------------------------------------------------

/// Group trees by their first `k` constant heavy-path node identities.
///
/// Returns `(groups, ungrouped)` where each group has ≥2 trees sharing an
/// exact prefix, and ungrouped contains all remaining tree indices.
pub(crate) fn build_prefix_groups(
    trees: &[Tree],
    nodes: &[Node],
    _config: &WalkerConfig,
    k: usize,
) -> (Vec<PrefixGroup>, Vec<u32>) {
    type LevelKey = (u64, u16, u8, u8);

    if k == 0 {
        return (Vec::new(), (0..trees.len() as u32).collect());
    }

    let mut group_map: HashMap<Vec<LevelKey>, Vec<u32>> = HashMap::default();
    let mut ungrouped: Vec<u32> = Vec::new();

    for (tree_idx, tree) in trees.iter().enumerate() {
        let base = tree.node_start as usize;
        let mut key = Vec::with_capacity(k);
        let mut valid = true;

        for d in 0..k {
            let node = &nodes[base + d];
            if node.is_leaf() || !node.is_constant() {
                valid = false;
                break;
            }
            key.push((node.value.to_bits(), node.feature, node.flags, node.cat_n_words));
        }

        if valid {
            group_map.entry(key).or_default().push(tree_idx as u32);
        } else {
            ungrouped.push(tree_idx as u32);
        }
    }

    let mut groups: Vec<PrefixGroup> = Vec::new();
    for (_, tree_list) in group_map {
        if tree_list.len() < 2 {
            ungrouped.extend(tree_list);
        } else {
            let representative_base = trees[tree_list[0] as usize].node_start;
            let mut trees_sorted = tree_list;
            trees_sorted.sort_unstable_by_key(|&idx| trees[idx as usize].node_start);
            groups.push(PrefixGroup {
                node_base: representative_base,
                trees: trees_sorted,
            });
        }
    }

    groups.sort_unstable_by_key(|g| g.node_base);
    ungrouped.sort_unstable();

    (groups, ungrouped)
}

// ---------------------------------------------------------------------------
// Tree ordering: greedy nearest-neighbor chaining
// ---------------------------------------------------------------------------

const ORDER_PATH_DEPTH: usize = 8;

/// Per-node summary for tree ordering. Captures the split predicate identity
/// (feature + threshold bits + flags) so trees with identical predicates at the
/// same depth sort adjacent. This improves branch prediction warmth, prefix
/// grouping, and mask cache locality (same predicate → same varying_pred_id).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TreePathToken {
    feature: u16,
    flags: u8,
    /// Threshold bits (f64::to_bits) — identifies the split predicate.
    /// Trees sharing (feature, threshold_bits, flags) at the same depth
    /// will use the same varying_pred_id at predict time.
    threshold_bits: u64,
}

const EMPTY_PATH_TOKEN: TreePathToken = TreePathToken {
    feature: u16::MAX,
    flags: u8::MAX,
    threshold_bits: u64::MAX,
};

/// Summary of a tree's first ORDER_PATH_DEPTH heavy-path nodes.
/// Implements Ord for lexicographic sort — primary key is depth 0, then depth 1, etc.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TreePathSummary {
    tokens: [TreePathToken; ORDER_PATH_DEPTH],
    len: u8,
}

/// Order trees by lexicographic sort on their heavy-path summary.
///
/// Trees with identical early splits (feature, threshold, flags) sort adjacent,
/// maximizing prefix group sizes and branch prediction warmth. O(T log T)
/// instead of the previous O(T²) greedy nearest-neighbor.
fn auto_order_trees(
    trees: Vec<Tree>,
    nodes: Vec<Node>,
    bitsets: Vec<u8>,
) -> (Vec<Tree>, Vec<Node>, Vec<u8>) {
    if trees.len() < 2 {
        return (trees, nodes, bitsets);
    }

    // Build (summary, original_index) pairs and sort by summary.
    let mut indexed: Vec<(TreePathSummary, usize)> = trees
        .iter()
        .enumerate()
        .map(|(i, tree)| (tree_path_summary(tree, &nodes), i))
        .collect();

    indexed.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let order: Vec<usize> = indexed.iter().map(|&(_, idx)| idx).collect();

    // Check if the sort actually changed the order.
    let is_identity = order.iter().enumerate().all(|(i, &orig)| orig == i);
    if is_identity {
        return (trees, nodes, bitsets);
    }

    repack_pools_in_tree_order(&order, &trees, &nodes, &bitsets)
}

/// Reorder nodes and bitsets so trees appear contiguous in the given order.
///
/// Builds new pools in sorted order and drops the old ones. Peak memory is
/// 2× the nodes pool during the copy. For models where this matters (>100MB),
/// the tree ordering can be disabled with `ParseConfig::disable_tree_ordering`.
fn repack_pools_in_tree_order(
    order: &[usize],
    trees: &[Tree],
    nodes: &[Node],
    bitsets: &[u8],
) -> (Vec<Tree>, Vec<Node>, Vec<u8>) {
    let mut new_trees = Vec::with_capacity(trees.len());
    let mut new_nodes = Vec::with_capacity(nodes.len());
    let mut new_bitsets = Vec::with_capacity(bitsets.len());
    let mut offset_remap: HashMap<usize, usize> = HashMap::default();

    for &idx in order {
        let tree = &trees[idx];
        let node_start = new_nodes.len() as u32;
        let bitset_start = new_bitsets.len() as u32;
        let old_start = tree.node_start as usize;
        let old_end = old_start + tree.node_count as usize;

        for &node in &nodes[old_start..old_end] {
            if node.is_categorical() && !node.inline_cat() && !node.is_leaf() {
                let old_offset = node.value as usize;
                let n_bytes = node.cat_n_words as usize * 4;
                let new_offset = *offset_remap.entry(old_offset).or_insert_with(|| {
                    let mapped = new_bitsets.len();
                    new_bitsets.extend_from_slice(&bitsets[old_offset..old_offset + n_bytes]);
                    mapped
                });
                let mut remapped = node;
                remapped.value = new_offset as f64;
                new_nodes.push(remapped);
            } else {
                new_nodes.push(node);
            }
        }

        new_trees.push(Tree {
            node_start,
            node_count: tree.node_count,
            bitset_start,
        });
    }

    (new_trees, new_nodes, new_bitsets)
}

fn tree_path_summary(tree: &Tree, nodes: &[Node]) -> TreePathSummary {
    let base = tree.node_start as usize;
    let mut idx = 0usize;
    let mut tokens = [EMPTY_PATH_TOKEN; ORDER_PATH_DEPTH];
    let mut len = 0usize;

    while len < ORDER_PATH_DEPTH {
        let node = &nodes[base + idx];
        if node.is_leaf() {
            break;
        }
        tokens[len] = TreePathToken {
            feature: node.feature,
            flags: node.flags,
            threshold_bits: node.value.to_bits(),
        };
        len += 1;
        idx += 1; // heavy child is always idx + 1
    }

    TreePathSummary {
        tokens,
        len: len as u8,
    }
}

