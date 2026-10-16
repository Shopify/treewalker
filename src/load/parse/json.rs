//! Strict Treelite JSON dump reader, over simd-json's tape: strings borrow the input,
//! and node objects are read in one pass. Binary is preferred for large models.

use super::decoded::Decoded;
use super::tree::{
    OrderScratch, ParseContext, TempNode, build_flags, encode_categories, order_tree,
};
use super::validation::{self, Metadata, ParsedModel};
use crate::config::Choices;
use crate::{LoadError, LoadOptions, WalkerConfig};
use rustc_hash::FxHashMap;
use simd_json::prelude::*;
use simd_json::tape::{Array, Value};
use std::io::Read;

/// The value of `key` in object `v`: the last one, as a map built from the object
/// would hold.
fn get<'t, 'i>(v: Value<'t, 'i>, key: &str) -> Option<Value<'t, 'i>> {
    v.as_object()?
        .iter()
        .filter(|&(k, _)| k == key)
        .last()
        .map(|(_, v)| v)
}

/// The value of `key` in object `v`, which must have it.
fn field<'t, 'i>(v: Value<'t, 'i>, key: &str) -> Result<Value<'t, 'i>, LoadError> {
    get(v, key).ok_or_else(|| LoadError::MalformedModel(format!("missing {key}")))
}

fn malformed(key: &str) -> LoadError {
    LoadError::MalformedModel(format!("invalid {key}"))
}

fn string<'i>(v: Value<'_, 'i>, key: &str) -> Result<&'i str, LoadError> {
    field(v, key)?.into_string().ok_or_else(|| malformed(key))
}

fn integer(v: Value<'_, '_>, key: &str) -> Result<i64, LoadError> {
    field(v, key)?.as_i64().ok_or_else(|| malformed(key))
}

/// A JSON number as `f64`, however simd-json typed it.
fn numeric(v: Value<'_, '_>) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|x| x as f64))
        .or_else(|| v.as_u64().map(|x| x as f64))
}

fn number(v: Value<'_, '_>, key: &str) -> Result<f64, LoadError> {
    numeric(field(v, key)?).ok_or_else(|| malformed(key))
}

fn boolean(v: Value<'_, '_>, key: &str) -> Result<bool, LoadError> {
    field(v, key)?.as_bool().ok_or_else(|| malformed(key))
}

fn array<'t, 'i>(v: Value<'t, 'i>, key: &str) -> Result<Array<'t, 'i>, LoadError> {
    field(v, key)?.as_array().ok_or_else(|| malformed(key))
}

fn ints(v: Value<'_, '_>, key: &str) -> Result<Vec<i32>, LoadError> {
    array(v, key)?
        .iter()
        .map(|x| {
            x.as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| malformed(key))
        })
        .collect()
}

/// Reject an object with a field outside `fields`.
fn known(v: Value<'_, '_>, fields: &[&str]) -> Result<(), LoadError> {
    let obj = v.as_object().ok_or_else(|| malformed("object"))?;
    for (k, _) in &obj {
        if !fields.contains(&k) {
            return Err(LoadError::Unsupported(format!("unknown JSON field {k:?}")));
        }
    }
    Ok(())
}

/// Reject an object that turns on one of the extension fields `names`.
fn extensions(v: Value<'_, '_>, names: &[&str]) -> Result<(), LoadError> {
    for &name in names {
        if get(v, name).is_some() && integer(v, name)? != 0 {
            return Err(LoadError::Unsupported(format!(
                "{name}: extensions are unsupported"
            )));
        }
    }
    Ok(())
}

/// Decode a JSON dump from `reader`, its nesting bounded before simd-json parses it.
pub(super) fn parse(
    reader: impl Read,
    config: &WalkerConfig,
    options: &LoadOptions,
    choices: &Choices,
) -> Result<ParsedModel, LoadError> {
    let mut bytes = Vec::new();
    let max = options.max_json_bytes;
    reader.take(max.saturating_add(1)).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err(LoadError::Limit(format!(
            "JSON input exceeds {max} bytes (LoadOptions::max_json_bytes); use binary for large models"
        )));
    }
    validation::validate_json_depth(&bytes, options.max_json_nesting)?;
    // The tape is the document as one flat array of tokens whose strings borrow the
    // input: no allocation per object or key.
    let tape = simd_json::to_tape(&mut bytes)
        .map_err(|e| LoadError::MalformedModel(format!("JSON: {e}")))?;
    let root = tape.as_value();
    known(
        root,
        &[
            "threshold_type",
            "leaf_output_type",
            "num_feature",
            "task_type",
            "average_tree_output",
            "num_target",
            "num_class",
            "leaf_vector_shape",
            "target_id",
            "class_id",
            "postprocessor",
            "sigmoid_alpha",
            "ratio_c",
            "base_scores",
            "attributes",
            "trees",
            "num_opt_field_per_model",
        ],
    )?;
    extensions(root, &["num_opt_field_per_model"])?;
    let threshold_type = validation::threshold_type(
        string(root, "threshold_type")?,
        string(root, "leaf_output_type")?,
    )?;
    let trees_json = array(root, "trees")?;
    let _ratio_c = number(root, "ratio_c")?;
    validation::attributes(string(root, "attributes")?, options)?;
    let meta = Metadata {
        num_tree: trees_json.len(),
        num_feature: integer(root, "num_feature")?,
        task_type: string(root, "task_type")?.into(),
        average: boolean(root, "average_tree_output")?,
        num_target: integer(root, "num_target")?,
        num_class: ints(root, "num_class")?,
        leaf_shape: ints(root, "leaf_vector_shape")?,
        target_id: ints(root, "target_id")?,
        class_id: ints(root, "class_id")?,
        postprocessor: string(root, "postprocessor")?.into(),
        sigmoid_alpha: f64::from(number(root, "sigmoid_alpha")? as f32),
        base_scores: array(root, "base_scores")?
            .iter()
            .map(|v| numeric(v).ok_or_else(|| malformed("base_scores")))
            .collect::<Result<_, _>>()?,
    };
    let output = meta.validate(config)?;
    // The node count, from the trees' node arrays, so that the nodes are allocated
    // once; a malformed tree fails below.
    let nodes = trees_json
        .iter()
        .map(|tree| array(tree, "nodes").ok().map(|nodes| nodes.len()))
        .try_fold(0usize, |total, n| total.checked_add(n?));
    let mut decoded = Decoded::new(nodes)?;
    let mut intern = FxHashMap::default();
    let mut ctx = ParseContext {
        decoded: &mut decoded,
        list_intern: choices.bitset_intern.then_some(&mut intern),
        config,
        threshold_type,
        list: Vec::new(),
    };
    let mut scratch = OrderScratch::default();
    let mut tree_scratch = TreeScratch::default();
    for (i, tree) in trees_json.iter().enumerate() {
        process_tree(tree, &mut ctx, &mut scratch, &mut tree_scratch)
            .map_err(|e| e.at(&format!("tree {i}")))?;
    }
    decoded.seal()?;
    Ok(ParsedModel {
        decoded,
        threshold_type,
        output,
    })
}

/// Per-tree scratch, reused.
#[derive(Default)]
struct TreeScratch<'t, 'i> {
    nodes: Vec<NodeFields<'t, 'i>>,
    ids: FxHashMap<i64, u32>,
    temp: Vec<TempNode>,
    categories: Vec<u32>,
}

/// Decode one tree object and add it to the decoded forest in heavy-first preorder.
fn process_tree<'t, 'i>(
    tree: Value<'t, 'i>,
    ctx: &mut ParseContext<'_>,
    scratch: &mut OrderScratch,
    t: &mut TreeScratch<'t, 'i>,
) -> Result<(), LoadError> {
    known(
        tree,
        &[
            "num_nodes",
            "has_categorical_split",
            "nodes",
            "root_id",
            "num_opt_field_per_tree",
            "num_opt_field_per_node",
        ],
    )?;
    extensions(tree, &["num_opt_field_per_tree", "num_opt_field_per_node"])?;
    let nodes = array(tree, "nodes")?;
    let n = nodes.len();
    if n == 0 {
        return Err(malformed("num_nodes: a tree has no nodes"));
    }
    if integer(tree, "num_nodes")? != n as i64 {
        return Err(malformed("num_nodes disagrees with nodes length"));
    }
    t.nodes.clear();
    t.ids.clear();
    for (i, node) in nodes.iter().enumerate() {
        let fields = NodeFields::of(node)?;
        let id = NodeFields::integer(fields.node_id, "node_id")?;
        if id < 0 || t.ids.insert(id, i as u32).is_some() {
            return Err(malformed(&format!("duplicate or negative node_id {id}")));
        }
        t.nodes.push(fields);
    }
    let root_id = if get(tree, "root_id").is_some() {
        integer(tree, "root_id")?
    } else {
        0
    };
    let root = *t
        .ids
        .get(&root_id)
        .ok_or_else(|| malformed(&format!("root_id {root_id} is absent")))? as usize;
    t.temp.clear();
    let category_start = ctx.decoded.categories.len() as u32;
    let mut has_cat = false;
    for node in &t.nodes {
        let id = NodeFields::integer(node.node_id, "node_id")?;
        has_cat |= node.node_type.and_then(Value::into_string) == Some("categorical_test_node");
        let decoded = decode_node(node, &t.ids, ctx, &mut t.categories)
            .map_err(|e| e.at(&format!("node {id}")))?;
        t.temp.push(decoded);
    }
    if boolean(tree, "has_categorical_split")? != has_cat {
        return Err(malformed("has_categorical_split"));
    }
    order_tree(&t.temp, root, category_start, ctx, scratch)
}

/// A node object's fields, found in one pass over its keys: the last value of each,
/// as a map built from the object would hold, and the first key that is not a
/// node field, and the first that is not a leaf's.
#[derive(Clone, Copy, Default)]
struct NodeFields<'t, 'i> {
    node_id: Option<Value<'t, 'i>>,
    leaf_value: Option<Value<'t, 'i>>,
    split_feature_id: Option<Value<'t, 'i>>,
    default_left: Option<Value<'t, 'i>>,
    node_type: Option<Value<'t, 'i>>,
    comparison_op: Option<Value<'t, 'i>>,
    threshold: Option<Value<'t, 'i>>,
    left_child: Option<Value<'t, 'i>>,
    right_child: Option<Value<'t, 'i>>,
    category_list: Option<Value<'t, 'i>>,
    category_list_right_child: Option<Value<'t, 'i>>,
    data_count: Option<Value<'t, 'i>>,
    sum_hess: Option<Value<'t, 'i>>,
    gain: Option<Value<'t, 'i>>,
    unknown: Option<&'i str>,
    not_leaf: Option<&'i str>,
}

impl<'t, 'i> NodeFields<'t, 'i> {
    fn of(node: Value<'t, 'i>) -> Result<Self, LoadError> {
        let obj = node.as_object().ok_or_else(|| malformed("object"))?;
        let mut f = Self::default();
        for (k, v) in &obj {
            let (slot, leaf) = match k {
                "node_id" => (&mut f.node_id, true),
                "leaf_value" => (&mut f.leaf_value, true),
                "split_feature_id" => (&mut f.split_feature_id, false),
                "default_left" => (&mut f.default_left, false),
                "node_type" => (&mut f.node_type, false),
                "comparison_op" => (&mut f.comparison_op, false),
                "threshold" => (&mut f.threshold, false),
                "left_child" => (&mut f.left_child, false),
                "right_child" => (&mut f.right_child, false),
                "category_list" => (&mut f.category_list, false),
                "category_list_right_child" => (&mut f.category_list_right_child, false),
                "data_count" => (&mut f.data_count, true),
                "sum_hess" => (&mut f.sum_hess, true),
                "gain" => (&mut f.gain, true),
                _ => {
                    f.unknown = f.unknown.or(Some(k));
                    f.not_leaf = f.not_leaf.or(Some(k));
                    continue;
                }
            };
            *slot = Some(v);
            if !leaf {
                f.not_leaf = f.not_leaf.or(Some(k));
            }
        }
        Ok(f)
    }

    fn field(value: Option<Value<'t, 'i>>, key: &str) -> Result<Value<'t, 'i>, LoadError> {
        value.ok_or_else(|| LoadError::MalformedModel(format!("missing {key}")))
    }
    fn integer(value: Option<Value<'t, 'i>>, key: &str) -> Result<i64, LoadError> {
        Self::field(value, key)?
            .as_i64()
            .ok_or_else(|| malformed(key))
    }
    fn number(value: Option<Value<'t, 'i>>, key: &str) -> Result<f64, LoadError> {
        numeric(Self::field(value, key)?).ok_or_else(|| malformed(key))
    }
    fn boolean(value: Option<Value<'t, 'i>>, key: &str) -> Result<bool, LoadError> {
        Self::field(value, key)?
            .as_bool()
            .ok_or_else(|| malformed(key))
    }
    fn string(value: Option<Value<'t, 'i>>, key: &str) -> Result<&'i str, LoadError> {
        Self::field(value, key)?
            .into_string()
            .ok_or_else(|| malformed(key))
    }
}

/// A node's fields as a [`TempNode`], validated, its children numbered by `ids`.
fn decode_node(
    node: &NodeFields<'_, '_>,
    ids: &FxHashMap<i64, u32>,
    ctx: &mut ParseContext<'_>,
    categories: &mut Vec<u32>,
) -> Result<TempNode, LoadError> {
    if let Some(k) = node.unknown {
        return Err(LoadError::Unsupported(format!("unknown JSON field {k:?}")));
    }
    let mut weight = 1.0;
    for (name, value) in [
        ("gain", node.gain),
        ("sum_hess", node.sum_hess),
        ("data_count", node.data_count),
    ] {
        if value.is_some() {
            let v = NodeFields::number(value, name)?;
            if !v.is_finite() || (name != "gain" && v < 0.0) {
                return Err(malformed(name));
            }
            if name != "gain" {
                weight = v;
            }
        }
    }
    if let Some(value) = node.leaf_value {
        if let Some(k) = node.not_leaf {
            return Err(LoadError::Unsupported(format!("unknown JSON field {k:?}")));
        }
        if value.as_array().is_some() {
            return Err(LoadError::Unsupported("vector leaf_value".into()));
        }
        return Ok(TempNode::leaf(
            validation::leaf(
                numeric(value).ok_or_else(|| malformed("leaf_value"))?,
                ctx.threshold_type,
            )?,
            weight,
        ));
    }
    let kind = NodeFields::string(node.node_type, "node_type")?;
    let cat = match kind {
        "numerical_test_node" => false,
        "categorical_test_node" => true,
        _ => return Err(LoadError::Unsupported(format!("node_type={kind:?}"))),
    };
    let incompatible = if cat {
        [
            ("threshold", node.threshold),
            ("comparison_op", node.comparison_op),
        ]
    } else {
        [
            ("category_list", node.category_list),
            ("category_list_right_child", node.category_list_right_child),
        ]
    };
    for (field, value) in incompatible {
        if value.is_some() {
            return Err(malformed(&format!("{kind} has incompatible field {field}")));
        }
    }
    let feature = NodeFields::integer(node.split_feature_id, "split_feature_id")?;
    if feature < 0 || feature as usize >= ctx.config.n_features() {
        return Err(malformed(&format!(
            "split_feature_id {feature} out of range"
        )));
    }
    let mut dl = NodeFields::boolean(node.default_left, "default_left")?;
    let child = |value, key| -> Result<u32, LoadError> {
        let id = NodeFields::integer(value, key)?;
        ids.get(&id)
            .copied()
            .ok_or_else(|| malformed(&format!("{key} {id} is absent")))
    };
    let (mut left, mut right) = (
        child(node.left_child, "left_child")?,
        child(node.right_child, "right_child")?,
    );
    let (value, inline) = if cat {
        let list = NodeFields::field(node.category_list, "category_list")?
            .as_array()
            .ok_or_else(|| malformed("category_list"))?;
        categories.clear();
        for v in &list {
            categories.push(
                v.as_u64()
                    .and_then(|x| u32::try_from(x).ok())
                    .ok_or_else(|| malformed("category_list value"))?,
            );
        }
        if NodeFields::boolean(node.category_list_right_child, "category_list_right_child")? {
            std::mem::swap(&mut left, &mut right);
            dl = !dl;
        }
        encode_categories(categories.iter().copied(), ctx)?
    } else {
        (
            validation::threshold(
                NodeFields::number(node.threshold, "threshold")?,
                NodeFields::string(node.comparison_op, "comparison_op")?,
                ctx.threshold_type,
            )?,
            false,
        )
    };
    Ok(TempNode {
        value,
        left,
        right,
        feature: feature as u32,
        flags: build_flags(dl, cat, inline, ctx.config.is_varying(feature as usize)),
        weight,
    })
}
