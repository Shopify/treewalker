//! Treelite v4 binary format reader — streaming with reusable buffers.
//!
//! Reads from a `BufReader` sequentially. Per-tree field arrays are read into
//! reusable buffers that are cleared and refilled for each tree. After the first
//! tree, no further allocations occur for field data — only the global `nodes`
//! and `bitsets` pools grow.
//!
//! Format spec: <https://treelite.readthedocs.io/en/latest/serialization/v4.html>

use rustc_hash::FxHashMap as HashMap;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::config::{ParseConfig, WalkerConfig};
use crate::forest::{Node, ThresholdType, Tree};

use super::common::{
    ParseContext, ReorderScratch, TempNode, build_flags, classify_feature, encode_categories,
    next_down, reorder_and_emit,
};

// ---------------------------------------------------------------------------
// Reusable per-tree field buffers
// ---------------------------------------------------------------------------

/// Pre-allocated buffers for per-tree field arrays. Cleared and reused across trees.
/// After the first tree (~240 nodes), no further allocations occur.
struct TreeBufs {
    node_type: Vec<i8>,
    cleft: Vec<i32>,
    cright: Vec<i32>,
    split_index: Vec<i32>,
    default_left: Vec<bool>,
    leaf_value: Vec<f64>,
    threshold: Vec<f64>,
    cmp: Vec<i8>,
    cat_right_child: Vec<bool>,
    category_list: Vec<u32>,
    cat_list_begin: Vec<u64>,
    cat_list_end: Vec<u64>,
    data_count: Vec<u64>,
    data_count_present: Vec<bool>,
    sum_hess: Vec<f64>,
    sum_hess_present: Vec<bool>,
    temp_nodes: Vec<TempNode>,
    // Shared byte buffer for raw reads (avoids per-read_array allocation)
    raw: Vec<u8>,
}

impl TreeBufs {
    const fn new() -> Self {
        Self {
            node_type: Vec::new(),
            cleft: Vec::new(),
            cright: Vec::new(),
            split_index: Vec::new(),
            default_left: Vec::new(),
            leaf_value: Vec::new(),
            threshold: Vec::new(),
            cmp: Vec::new(),
            cat_right_child: Vec::new(),
            category_list: Vec::new(),
            cat_list_begin: Vec::new(),
            cat_list_end: Vec::new(),
            data_count: Vec::new(),
            data_count_present: Vec::new(),
            sum_hess: Vec::new(),
            sum_hess_present: Vec::new(),
            temp_nodes: Vec::new(),
            raw: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Buffered read helpers — read into reusable Vec
// ---------------------------------------------------------------------------

fn read_exact(r: &mut impl Read, buf: &mut [u8]) {
    r.read_exact(buf)
        .unwrap_or_else(|e| panic!("unexpected EOF reading binary model: {e}"));
}

fn read_u8(r: &mut impl Read) -> u8 {
    let mut b = [0u8; 1];
    read_exact(r, &mut b);
    b[0]
}

fn read_bool(r: &mut impl Read) -> bool { read_u8(r) != 0 }

fn read_i32(r: &mut impl Read) -> i32 {
    let mut b = [0u8; 4];
    read_exact(r, &mut b);
    i32::from_le_bytes(b)
}

fn read_f32(r: &mut impl Read) -> f32 {
    let mut b = [0u8; 4];
    read_exact(r, &mut b);
    f32::from_le_bytes(b)
}

fn read_u64(r: &mut impl Read) -> u64 {
    let mut b = [0u8; 8];
    read_exact(r, &mut b);
    u64::from_le_bytes(b)
}

/// Read a length-prefixed array into a reusable raw byte buffer, then decode into `out`.
fn read_into_i8(r: &mut impl Read, raw: &mut Vec<u8>, out: &mut Vec<i8>) {
    let len = read_u64(r) as usize;
    raw.resize(len, 0);
    read_exact(r, raw);
    out.clear();
    out.extend(raw.iter().map(|&b| b as i8));
}

fn read_into_bool(r: &mut impl Read, raw: &mut Vec<u8>, out: &mut Vec<bool>) {
    let len = read_u64(r) as usize;
    raw.resize(len, 0);
    read_exact(r, raw);
    out.clear();
    out.extend(raw.iter().map(|&b| b != 0));
}

fn read_into_i32(r: &mut impl Read, raw: &mut Vec<u8>, out: &mut Vec<i32>) {
    let len = read_u64(r) as usize;
    raw.resize(len * 4, 0);
    read_exact(r, raw);
    out.clear();
    out.extend(raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())));
}

fn read_into_u32(r: &mut impl Read, raw: &mut Vec<u8>, out: &mut Vec<u32>) {
    let len = read_u64(r) as usize;
    raw.resize(len * 4, 0);
    read_exact(r, raw);
    out.clear();
    out.extend(raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
}

fn read_into_u64(r: &mut impl Read, raw: &mut Vec<u8>, out: &mut Vec<u64>) {
    let len = read_u64(r) as usize;
    raw.resize(len * 8, 0);
    read_exact(r, raw);
    out.clear();
    out.extend(raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())));
}

fn read_into_f64(r: &mut impl Read, raw: &mut Vec<u8>, out: &mut Vec<f64>) {
    let len = read_u64(r) as usize;
    raw.resize(len * 8, 0);
    read_exact(r, raw);
    out.clear();
    out.extend(raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())));
}

fn read_into_f64_from_f32(r: &mut impl Read, raw: &mut Vec<u8>, out: &mut Vec<f64>) {
    let len = read_u64(r) as usize;
    raw.resize(len * 4, 0);
    read_exact(r, raw);
    out.clear();
    out.extend(raw.chunks_exact(4).map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap()))));
}

/// Skip a length-prefixed array.
fn skip_array(r: &mut impl Read, elem_size: usize, raw: &mut Vec<u8>) {
    let len = read_u64(r) as usize;
    raw.resize(len * elem_size, 0);
    read_exact(r, raw);
}

/// Read a length-prefixed string (allocates — only used for header, not per-tree).
fn read_string(r: &mut impl Read) -> String {
    let len = read_u64(r) as usize;
    let mut buf = vec![0u8; len];
    read_exact(r, &mut buf);
    if buf.last() == Some(&0) { buf.pop(); }
    String::from_utf8(buf).expect("invalid UTF-8 in binary model string")
}

// Temporary Vec for header arrays that we read once and discard.
fn skip_header_array(r: &mut impl Read, elem_size: usize) {
    let len = read_u64(r) as usize;
    let mut buf = vec![0u8; len * elem_size];
    read_exact(r, &mut buf);
}

fn read_header_f64(r: &mut impl Read) -> Vec<f64> {
    let len = read_u64(r) as usize;
    let mut buf = vec![0u8; len * 8];
    read_exact(r, &mut buf);
    buf.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn parse(
    path: &Path,
    config: &WalkerConfig,
    parse_config: &ParseConfig,
    hoist_stats: &mut super::HoistStats,
) -> (Vec<Tree>, Vec<Node>, Vec<u8>, ThresholdType) {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("failed to open binary model {}: {e}", path.display()));
    let mut r = BufReader::with_capacity(128 * 1024, file);

    // -- Header --
    let major = read_i32(&mut r);
    assert!(major == 4, "Expected treelite binary v4, got v{major}");
    let _minor = read_i32(&mut r);
    let _patch = read_i32(&mut r);

    let thr_type_raw = read_u8(&mut r);
    let _leaf_type_raw = read_u8(&mut r);

    let threshold_type = match thr_type_raw {
        2 => ThresholdType::F32,
        3 => ThresholdType::F64,
        _ => panic!("unsupported threshold_type: {thr_type_raw}"),
    };

    let num_tree = read_u64(&mut r) as usize;
    let _num_feature = read_i32(&mut r);
    let _task_type = read_u8(&mut r);
    let _average_tree_output = read_bool(&mut r);

    let _num_target = read_i32(&mut r);
    skip_header_array(&mut r, 4); // num_class
    skip_header_array(&mut r, 4); // leaf_vector_shape
    skip_header_array(&mut r, 4); // target_id
    skip_header_array(&mut r, 4); // class_id

    let postprocessor = read_string(&mut r);
    assert!(
        postprocessor.contains("sigmoid"),
        "unsupported postprocessor: '{postprocessor}'. Only binary-logistic (sigmoid) models supported."
    );
    let _sigmoid_alpha = read_f32(&mut r);
    let _ratio_c = read_f32(&mut r);

    let base_scores = read_header_f64(&mut r);
    let _attributes = read_string(&mut r);
    let _num_opt_field_per_model = read_i32(&mut r);

    let base_score = if base_scores.is_empty() { 0.0 } else { base_scores[0] };
    let leaf_bias = base_score / num_tree as f64;

    // -- Per-tree parsing with reusable buffers --
    let mut trees = Vec::with_capacity(num_tree);
    let mut nodes: Vec<Node> = Vec::new();
    let mut bitsets: Vec<u8> = Vec::new();
    let mut bitset_intern: HashMap<Vec<u32>, usize> = HashMap::default();
    let mut bufs = TreeBufs::new();
    let mut scratch = ReorderScratch::new();

    {
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

        for _ in 0..num_tree {
            trees.push(read_tree(&mut r, &mut ctx, &mut bufs, &mut scratch));
        }
    }

    (trees, nodes, bitsets, threshold_type)
}

// ---------------------------------------------------------------------------
// Per-tree reading — streaming with reusable buffers
// ---------------------------------------------------------------------------

fn read_tree(
    r: &mut impl Read,
    ctx: &mut ParseContext<'_>,
    b: &mut TreeBufs,
    scratch: &mut ReorderScratch,
) -> Tree {
    let num_nodes = read_i32(r) as usize;
    let _has_categorical_split = read_bool(r);

    assert!(
        i16::try_from(num_nodes).is_ok(),
        "tree has {num_nodes} nodes, exceeds i16 capacity"
    );

    // Read field arrays into reusable buffers.
    read_into_i8(r, &mut b.raw, &mut b.node_type);
    read_into_i32(r, &mut b.raw, &mut b.cleft);
    read_into_i32(r, &mut b.raw, &mut b.cright);
    read_into_i32(r, &mut b.raw, &mut b.split_index);
    read_into_bool(r, &mut b.raw, &mut b.default_left);

    match ctx.threshold_type {
        ThresholdType::F64 => {
            read_into_f64(r, &mut b.raw, &mut b.leaf_value);
            read_into_f64(r, &mut b.raw, &mut b.threshold);
        }
        ThresholdType::F32 => {
            read_into_f64_from_f32(r, &mut b.raw, &mut b.leaf_value);
            read_into_f64_from_f32(r, &mut b.raw, &mut b.threshold);
        }
    }

    read_into_i8(r, &mut b.raw, &mut b.cmp);
    read_into_bool(r, &mut b.raw, &mut b.cat_right_child);

    // Leaf vectors (skip)
    match ctx.threshold_type {
        ThresholdType::F64 => skip_array(r, 8, &mut b.raw),
        ThresholdType::F32 => skip_array(r, 4, &mut b.raw),
    }
    skip_array(r, 8, &mut b.raw); // leaf_vector_begin
    skip_array(r, 8, &mut b.raw); // leaf_vector_end

    read_into_u32(r, &mut b.raw, &mut b.category_list);
    read_into_u64(r, &mut b.raw, &mut b.cat_list_begin);
    read_into_u64(r, &mut b.raw, &mut b.cat_list_end);

    read_into_u64(r, &mut b.raw, &mut b.data_count);
    read_into_bool(r, &mut b.raw, &mut b.data_count_present);
    read_into_f64(r, &mut b.raw, &mut b.sum_hess);
    read_into_bool(r, &mut b.raw, &mut b.sum_hess_present);
    skip_array(r, 8, &mut b.raw); // gain
    skip_array(r, 1, &mut b.raw); // gain_present

    let _num_opt_field_per_tree = read_i32(r);
    let _num_opt_field_per_node = read_i32(r);

    // -- Build TempNodes into reusable buffer --
    b.temp_nodes.clear();
    let bitset_start = ctx.bitsets.len() as u32;

    for i in 0..num_nodes {
        let nt = b.node_type[i];

        let weight = if i < b.data_count_present.len() && b.data_count_present[i] {
            b.data_count[i] as f64
        } else if i < b.sum_hess_present.len() && b.sum_hess_present[i] {
            b.sum_hess[i]
        } else {
            1.0
        };

        assert!(
            nt == 0 || nt == 1 || nt == 2,
            "unknown node_type {nt} at node {i} (expected 0=leaf, 1=numerical, 2=categorical)",
        );

        if nt == 0 {
            b.temp_nodes.push(TempNode {
                value: b.leaf_value[i] + ctx.leaf_bias,
                left: -1, right: -1, feature: 0, flags: 0, cat_n_words: 0, weight,
            });
            continue;
        }

        let split_feature = b.split_index[i] as u16;
        assert!(
            (split_feature as usize) < ctx.config.n_features,
            "split_index {} >= n_features {}", split_feature, ctx.config.n_features,
        );
        let dl = b.default_left[i];
        let left_idx = b.cleft[i] as i16;
        let right_idx = b.cright[i] as i16;
        let is_cat = nt == 2;

        let (value, inline, n_words) = if is_cat {
            let begin = b.cat_list_begin[i] as usize;
            let end = b.cat_list_end[i] as usize;
            let categories: Vec<u32> = b.category_list[begin..end].to_vec();
            let invert = b.cat_right_child[i];
            encode_categories(&categories, invert, ctx)
        } else {
            let raw_threshold = b.threshold[i];
            let comparison_op = b.cmp[i];

            let thr = if raw_threshold.is_nan() {
                if dl { f64::NEG_INFINITY } else { f64::INFINITY }
            } else {
                raw_threshold
            };

            let adjusted = match (comparison_op, ctx.threshold_type) {
                (2, ThresholdType::F64) => next_down(thr),
                (3, ThresholdType::F64) | (2, ThresholdType::F32) => thr,
                (3, ThresholdType::F32) => panic!("F32 model with LE comparison not supported"),
                _ => panic!("unsupported comparison_op: {comparison_op}"),
            };
            (adjusted, false, 0u8)
        };

        let varying_type = classify_feature(ctx.config, split_feature as usize);
        let flags = build_flags(dl, is_cat, inline, varying_type);

        b.temp_nodes.push(TempNode {
            value, left: left_idx, right: right_idx,
            feature: split_feature, flags, cat_n_words: n_words, weight,
        });
    }

    reorder_and_emit(&mut b.temp_nodes, 0, bitset_start, ctx, scratch)
}
