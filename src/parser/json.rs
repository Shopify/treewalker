//! JSON model parser: treelite `dump_as_json()` format.
//!
//! Uses simd-json's `OwnedValue` DOM for simple field access with SIMD-accelerated
//! parsing. The treelite JSON format is small enough that DOM overhead is negligible.
//!
//! **Note**: this path reads the entire file into a `String`, applies two
//! `replace()` calls to patch treelite's invalid NaN output, then converts to
//! bytes for `simd-json`. The `.bin` (binary v4) format avoids this overhead
//! and is the intended production path. JSON is fallback/debug.

use rustc_hash::FxHashMap as HashMap;
use std::path::Path;

use simd_json::OwnedValue;
use simd_json::prelude::*;

use crate::config::{ParseConfig, WalkerConfig};
use crate::forest::{Node, ThresholdType, Tree};

use super::common::{
    ParseContext, ReorderScratch, TempNode, build_flags, classify_feature, encode_categories,
    next_down, reorder_and_emit,
};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse a treelite JSON model file using simd-json DOM.
pub fn parse(
    path: &Path,
    config: &WalkerConfig,
    parse_config: &ParseConfig,
    hoist_stats: &mut super::HoistStats,
) -> (Vec<Tree>, Vec<Node>, Vec<u8>, ThresholdType) {
    let raw = std::fs::read_to_string(path).expect("failed to read model JSON file");
    // Treelite emits invalid JSON for NaN values: `"threshold": ,` or `"leaf_value": ,`.
    let content = raw.replace("\": ,", "\": null,").replace("\": }", "\": null}");
    let mut bytes = content.into_bytes();

    let root: OwnedValue =
        simd_json::to_owned_value(&mut bytes).expect("failed to parse treelite JSON structure");

    // --- Extract threshold_type ---
    let threshold_type = match root.get("threshold_type").and_then(OwnedValue::as_str) {
        Some("float32") => ThresholdType::F32,
        _ => ThresholdType::F64,
    };

    // --- Validate postprocessor ---
    if let Some(pp) = root.get("postprocessor").and_then(OwnedValue::as_str) {
        assert!(
            pp.contains("sigmoid"),
            "unsupported postprocessor: '{pp}'. Only binary-logistic (sigmoid) models supported."
        );
    } else {
        panic!("missing or non-string 'postprocessor' field. Only binary-logistic (sigmoid) models supported.");
    }

    // --- Extract base_scores ---
    let base_score = root
        .get("base_scores")
        .and_then(OwnedValue::as_array)
        .and_then(|arr| arr.first())
        .and_then(OwnedValue::as_f64)
        .unwrap_or(0.0);

    // --- Extract trees array ---
    let tree_array = root
        .get("trees")
        .and_then(OwnedValue::as_array)
        .expect("missing 'trees' array in treelite JSON");

    assert!(
        !tree_array.is_empty(),
        "Unknown JSON model format. Expected treelite format (top-level 'trees' array). \
         Use `treelite.frontend.load_lightgbm_model()` or `load_xgboost_model()` then \
         `model.dump_as_json()` to convert.",
    );

    let n_trees = tree_array.len();
    let leaf_bias = base_score / n_trees as f64;

    let mut trees = Vec::with_capacity(n_trees);
    let mut nodes = Vec::new();
    let mut bitsets = Vec::new();
    let mut bitset_intern: HashMap<Vec<u32>, usize> = HashMap::default();

    let mut ctx = ParseContext {
        hoist_constants: parse_config.hoist_constants,
        hoist_stats,
        nodes: &mut nodes,
        bitsets: &mut bitsets,
        bitset_intern: if parse_config.disable_bitset_intern {
            None
        } else {
            Some(&mut bitset_intern)
        },
        config,
        leaf_bias,
        threshold_type,
    };

    let mut scratch = ReorderScratch::new();
    for tl_tree in tree_array {
        trees.push(process_tree(tl_tree, &mut ctx, &mut scratch));
    }

    (trees, nodes, bitsets, threshold_type)
}

// ---------------------------------------------------------------------------
// Tree processing: OwnedValue tree object -> TempNode -> reorder_and_emit
// ---------------------------------------------------------------------------

fn process_tree(tree: &OwnedValue, ctx: &mut ParseContext<'_>, scratch: &mut ReorderScratch) -> Tree {
    let nodes_arr = tree
        .get("nodes")
        .and_then(OwnedValue::as_array)
        .expect("tree missing 'nodes' array");

    let total = nodes_arr.len();
    assert!(
        i16::try_from(total).is_ok(),
        "tree has {total} nodes, exceeds i16 capacity"
    );

    // Build node_id -> array_index mapping.
    let mut id_to_idx: HashMap<u64, usize> = HashMap::with_capacity_and_hasher(total, rustc_hash::FxBuildHasher);
    for (idx, node) in nodes_arr.iter().enumerate() {
        let node_id = node
            .get("node_id")
            .and_then(OwnedValue::as_u64)
            .expect("node missing node_id");
        id_to_idx.insert(node_id, idx);
    }

    let bitset_start = ctx.bitsets.len() as u32;
    let mut temp = Vec::with_capacity(total);

    for node in nodes_arr {
        let node_type_str = node.get("node_type").and_then(OwnedValue::as_str);
        let leaf_value = node.get("leaf_value").and_then(OwnedValue::as_f64);

        // Leaf node: has leaf_value, no node_type
        if leaf_value.is_some() && node_type_str.is_none() {
            let weight = node
                .get("data_count")
                .and_then(OwnedValue::as_f64)
                .or_else(|| node.get("sum_hess").and_then(OwnedValue::as_f64))
                .unwrap_or(1.0);
            temp.push(TempNode {
                value: leaf_value.unwrap_or(0.0) + ctx.leaf_bias,
                left: -1,
                right: -1,
                feature: 0,
                flags: 0,
                cat_n_words: 0,
                weight,
            });
            continue;
        }

        // Internal node
        let split_feature_raw = node
            .get("split_feature_id")
            .and_then(OwnedValue::as_u64)
            .expect("missing split_feature_id");
        assert!(
            split_feature_raw < ctx.config.n_features as u64,
            "split_feature_id {split_feature_raw} >= n_features {}",
            ctx.config.n_features,
        );
        let split_feature = split_feature_raw as u16;

        let default_left = node
            .get("default_left")
            .and_then(OwnedValue::as_bool)
            .unwrap_or(false);
        let left_id = node
            .get("left_child")
            .and_then(OwnedValue::as_u64)
            .expect("missing left_child");
        let right_id = node
            .get("right_child")
            .and_then(OwnedValue::as_u64)
            .expect("missing right_child");
        let left_idx = id_to_idx[&left_id] as i16;
        let right_idx = id_to_idx[&right_id] as i16;
        let weight = node
            .get("data_count")
            .and_then(OwnedValue::as_f64)
            .or_else(|| node.get("sum_hess").and_then(OwnedValue::as_f64))
            .unwrap_or(1.0);

        let is_cat = node_type_str == Some("categorical_test_node");

        let (value, inline, n_words) = if is_cat {
            let categories: Vec<u32> = node
                .get("category_list")
                .and_then(OwnedValue::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|x| x as u32))
                        .collect()
                })
                .unwrap_or_default();
            let invert = node
                .get("category_list_right_child")
                .and_then(OwnedValue::as_bool)
                .unwrap_or(false);
            encode_categories(&categories, invert, ctx)
        } else {
            let raw_threshold = node
                .get("threshold")
                .and_then(OwnedValue::as_f64)
                .unwrap_or(f64::NAN);
            let comparison_op = node
                .get("comparison_op")
                .and_then(OwnedValue::as_str)
                .unwrap_or("<=");

            let threshold = if raw_threshold.is_nan() {
                if default_left {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            } else {
                raw_threshold
            };

            let adjusted = match (comparison_op, ctx.threshold_type) {
                ("<", ThresholdType::F64) => next_down(threshold),
                ("<=", ThresholdType::F64) | ("<", ThresholdType::F32) => threshold,
                ("<=", ThresholdType::F32) => panic!(
                    "F32 model with <= comparison not supported (XGBoost always uses <)"
                ),
                _ => panic!("unsupported comparison_op: {comparison_op}"),
            };
            (adjusted, false, 0u8)
        };

        let varying_type = classify_feature(ctx.config, split_feature as usize);
        let flags = build_flags(default_left, is_cat, inline, varying_type);

        temp.push(TempNode {
            value,
            left: left_idx,
            right: right_idx,
            feature: split_feature,
            flags,
            cat_n_words: n_words,
            weight,
        });
    }

    let root_id = tree
        .get("root_id")
        .and_then(OwnedValue::as_u64)
        .unwrap_or(0);
    let root_idx = id_to_idx[&root_id];

    reorder_and_emit(&mut temp, root_idx, bitset_start, ctx, scratch)
}
