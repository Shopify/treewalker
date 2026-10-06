//! Strict Treelite JSON dump reader. Binary is preferred for large models.
use super::common::{
    ParseContext, ReorderScratch, TempNode, build_flags, classify_feature, encode_categories,
    reorder_and_emit,
};
use super::validation::{self, MAX_JSON_BYTES, Metadata, ParsedModel};
use crate::forest::Tree;
use crate::{LoadError, LoadOptions, WalkerConfig};
use rustc_hash::FxHashMap;
use simd_json::{OwnedValue as Value, prelude::*};
use std::io::Read;

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value, LoadError> {
    v.get(key)
        .ok_or_else(|| LoadError::MalformedModel(format!("missing {key}")))
}
fn malformed(key: &str) -> LoadError {
    LoadError::MalformedModel(format!("invalid {key}"))
}
fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str, LoadError> {
    field(v, key)?.as_str().ok_or_else(|| malformed(key))
}
fn integer(v: &Value, key: &str) -> Result<i64, LoadError> {
    field(v, key)?.as_i64().ok_or_else(|| malformed(key))
}
fn numeric(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|x| x as f64))
        .or_else(|| v.as_u64().map(|x| x as f64))
}
fn number(v: &Value, key: &str) -> Result<f64, LoadError> {
    numeric(field(v, key)?).ok_or_else(|| malformed(key))
}
fn boolean(v: &Value, key: &str) -> Result<bool, LoadError> {
    field(v, key)?.as_bool().ok_or_else(|| malformed(key))
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>, LoadError> {
    field(v, key)?.as_array().ok_or_else(|| malformed(key))
}
fn ints(v: &Value, key: &str) -> Result<Vec<i32>, LoadError> {
    array(v, key)?
        .iter()
        .map(|x| {
            x.as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| malformed(key))
        })
        .collect()
}
fn known(v: &Value, fields: &[&str]) -> Result<(), LoadError> {
    let obj = v.as_object().ok_or_else(|| malformed("object"))?;
    for k in obj.keys() {
        if !fields.contains(&k.as_str()) {
            return Err(LoadError::Unsupported(format!("unknown JSON field {k:?}")));
        }
    }
    Ok(())
}
fn extensions(v: &Value, names: &[&str]) -> Result<(), LoadError> {
    for &name in names {
        if v.get(name).is_some() && integer(v, name)? != 0 {
            return Err(LoadError::Unsupported(format!(
                "{name}: extensions are unsupported"
            )));
        }
    }
    Ok(())
}

pub(super) fn parse(
    reader: impl Read,
    config: &WalkerConfig,
    options: &LoadOptions,
) -> Result<ParsedModel, LoadError> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_JSON_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_JSON_BYTES {
        return Err(LoadError::Limit(
            "JSON input exceeds 64 MiB; use binary for large models".into(),
        ));
    }
    validation::validate_json_depth(&bytes)?;
    let root = simd_json::to_owned_value(&mut bytes)
        .map_err(|e| LoadError::MalformedModel(format!("JSON: {e}")))?;
    known(
        &root,
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
    extensions(&root, &["num_opt_field_per_model"])?;
    let threshold_type = validation::threshold_type(
        string(&root, "threshold_type")?,
        string(&root, "leaf_output_type")?,
    )?;
    let trees_json = array(&root, "trees")?;
    let _ratio_c = number(&root, "ratio_c")?;
    validation::attributes(string(&root, "attributes")?)?;
    let meta = Metadata {
        num_tree: trees_json.len(),
        num_feature: integer(&root, "num_feature")?,
        task_type: string(&root, "task_type")?.into(),
        average: boolean(&root, "average_tree_output")?,
        num_target: integer(&root, "num_target")?,
        num_class: ints(&root, "num_class")?,
        leaf_shape: ints(&root, "leaf_vector_shape")?,
        target_id: ints(&root, "target_id")?,
        class_id: ints(&root, "class_id")?,
        postprocessor: string(&root, "postprocessor")?.into(),
        sigmoid_alpha: f64::from(number(&root, "sigmoid_alpha")? as f32),
        base_scores: array(&root, "base_scores")?
            .iter()
            .map(|v| numeric(v).ok_or_else(|| malformed("base_scores")))
            .collect::<Result<_, _>>()?,
    };
    let output = meta.validate(config)?;
    let (mut trees, mut nodes, mut bitsets) = (Vec::new(), Vec::new(), Vec::new());
    let mut intern = FxHashMap::default();
    let mut ctx = ParseContext {
        nodes: &mut nodes,
        bitsets: &mut bitsets,
        bitset_intern: if options.disable_bitset_intern {
            None
        } else {
            Some(&mut intern)
        },
        config,
        threshold_type,
    };
    let mut scratch = ReorderScratch::new();
    for (i, tree) in trees_json.iter().enumerate() {
        trees.push(
            process_tree(tree, &mut ctx, &mut scratch).map_err(|e| e.at(&format!("tree {i}")))?,
        );
    }
    Ok(ParsedModel {
        trees,
        nodes,
        bitsets,
        threshold_type,
        output,
    })
}

fn process_tree(
    tree: &Value,
    ctx: &mut ParseContext<'_>,
    scratch: &mut ReorderScratch,
) -> Result<Tree, LoadError> {
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
    if n == 0 || n > i16::MAX as usize {
        return Err(LoadError::Limit(format!(
            "num_nodes={n}; require 1..=32767"
        )));
    }
    if integer(tree, "num_nodes")? != n as i64 {
        return Err(malformed("num_nodes disagrees with nodes length"));
    }
    let mut ids = FxHashMap::default();
    for (i, node) in nodes.iter().enumerate() {
        let id = integer(node, "node_id")?;
        if id < 0 || ids.insert(id, i as i16).is_some() {
            return Err(malformed(&format!("duplicate or negative node_id {id}")));
        }
    }
    let root_id = if tree.get("root_id").is_some() {
        integer(tree, "root_id")?
    } else {
        0
    };
    let root = *ids
        .get(&root_id)
        .ok_or_else(|| malformed(&format!("root_id {root_id} is absent")))? as usize;
    let mut temp = Vec::with_capacity(n);
    let bitset_start = ctx.bitsets.len() as u32;
    let mut has_cat = false;
    for node in nodes {
        let id = integer(node, "node_id")?;
        has_cat |= node.get("node_type").and_then(Value::as_str) == Some("categorical_test_node");
        temp.push(decode_node(node, &ids, ctx).map_err(|e| e.at(&format!("node {id}")))?);
    }
    if boolean(tree, "has_categorical_split")? != has_cat {
        return Err(malformed("has_categorical_split"));
    }
    reorder_and_emit(&temp, root, bitset_start, ctx, scratch)
}

fn decode_node(
    node: &Value,
    ids: &FxHashMap<i64, i16>,
    ctx: &mut ParseContext<'_>,
) -> Result<TempNode, LoadError> {
    known(
        node,
        &[
            "node_id",
            "leaf_value",
            "split_feature_id",
            "default_left",
            "node_type",
            "comparison_op",
            "threshold",
            "left_child",
            "right_child",
            "category_list",
            "category_list_right_child",
            "data_count",
            "sum_hess",
            "gain",
        ],
    )?;
    let mut weight = 1.0;
    for name in ["gain", "sum_hess", "data_count"] {
        if node.get(name).is_some() {
            let v = number(node, name)?;
            if !v.is_finite() || (name != "gain" && v < 0.0) {
                return Err(malformed(name));
            }
            if name != "gain" {
                weight = v;
            }
        }
    }
    if let Some(value) = node.get("leaf_value") {
        known(
            node,
            &["node_id", "leaf_value", "data_count", "sum_hess", "gain"],
        )?;
        if value.as_array().is_some() {
            return Err(LoadError::Unsupported("vector leaf_value".into()));
        }
        return Ok(TempNode {
            value: validation::leaf(
                numeric(value).ok_or_else(|| malformed("leaf_value"))?,
                ctx.threshold_type,
            )?,
            left: -1,
            right: -1,
            feature: 0,
            flags: 0,
            cat_n_words: 0,
            weight,
        });
    }
    let kind = string(node, "node_type")?;
    let cat = match kind {
        "numerical_test_node" => false,
        "categorical_test_node" => true,
        _ => return Err(LoadError::Unsupported(format!("node_type={kind:?}"))),
    };
    let incompatible = if cat {
        ["threshold", "comparison_op"]
    } else {
        ["category_list", "category_list_right_child"]
    };
    for field in incompatible {
        if node.get(field).is_some() {
            return Err(malformed(&format!("{kind} has incompatible field {field}")));
        }
    }
    let feature = integer(node, "split_feature_id")?;
    if feature < 0 || feature as usize >= ctx.config.n_features() {
        return Err(malformed(&format!(
            "split_feature_id {feature} out of range"
        )));
    }
    let mut dl = boolean(node, "default_left")?;
    let child = |key| -> Result<i16, LoadError> {
        let id = integer(node, key)?;
        ids.get(&id)
            .copied()
            .ok_or_else(|| malformed(&format!("{key} {id} is absent")))
    };
    let (mut left, mut right) = (child("left_child")?, child("right_child")?);
    let (value, inline, words) = if cat {
        let categories = array(node, "category_list")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|x| u32::try_from(x).ok())
                    .ok_or_else(|| malformed("category_list value"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if boolean(node, "category_list_right_child")? {
            std::mem::swap(&mut left, &mut right);
            dl = !dl;
        }
        encode_categories(&categories, ctx)?
    } else {
        (
            validation::threshold(
                number(node, "threshold")?,
                string(node, "comparison_op")?,
                ctx.threshold_type,
            )?,
            false,
            0,
        )
    };
    Ok(TempNode {
        value,
        left,
        right,
        feature: feature as u16,
        flags: build_flags(
            dl,
            cat,
            inline,
            classify_feature(ctx.config, feature as usize),
        ),
        cat_n_words: words,
        weight,
    })
}
