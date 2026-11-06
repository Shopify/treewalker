//! Native LightGBM text, normalized like Treelite 4.7.2's LightGBM frontend.
//! NaN at None and Zero missing-type splits compares as zero, ignoring the default
//! bit; LightGBM routes Zero-type NaN and zeros by that bit.

use std::collections::HashMap;
use std::io::Read;
use std::str::FromStr;

use super::decoded::Decoded;
use super::tree::{
    ListIntern, OrderScratch, ParseContext, TempNode, build_flags, check_node_total,
    encode_categories, order_tree,
};
use super::validation::{self, Metadata, ParsedModel};
use crate::config::Choices;
use crate::model::ThresholdType;
use crate::{LoadError, LoadOptions, WalkerConfig};

type Fields<'a> = HashMap<&'a str, &'a str>;

fn malformed(message: impl Into<String>) -> LoadError {
    LoadError::MalformedModel(message.into())
}

fn field<'a>(fields: &Fields<'a>, key: &str) -> Result<&'a str, LoadError> {
    fields
        .get(key)
        .copied()
        .ok_or_else(|| malformed(format!("missing {key}")))
}

fn number<T: FromStr>(value: &str, key: &str) -> Result<T, LoadError> {
    value
        .parse()
        .map_err(|_| malformed(format!("invalid {key}: {value:?}")))
}

fn scalar<T: FromStr>(fields: &Fields<'_>, key: &str) -> Result<T, LoadError> {
    number(field(fields, key)?, key)
}

/// Apply the binary representation's field bounds before allocating text arrays.
fn cap(n: usize, size: usize, options: &LoadOptions) -> Result<(), LoadError> {
    if (n as u64)
        .checked_mul(size as u64)
        .is_none_or(|bytes| bytes > options.max_binary_array_bytes)
    {
        return Err(LoadError::Limit(format!(
            "LightGBM array exceeds {} bytes (LoadOptions::max_binary_array_bytes)",
            options.max_binary_array_bytes
        )));
    }
    Ok(())
}

fn array<T: FromStr>(
    fields: &Fields<'_>,
    key: &str,
    n: usize,
    options: &LoadOptions,
) -> Result<Vec<T>, LoadError> {
    let text = field(fields, key)?;
    // Do not allocate from a declared count without its corresponding input.
    if text.split_whitespace().count() != n {
        return Err(malformed(format!("{key}: expected {n} entries")));
    }
    cap(n, size_of::<T>(), options)?;
    text.split_whitespace().map(|v| number(v, key)).collect()
}

pub(super) fn parse(
    reader: impl Read,
    config: &WalkerConfig,
    options: &LoadOptions,
    choices: &Choices,
) -> Result<ParsedModel, LoadError> {
    let bytes = validation::document(reader, options.max_json_bytes, "LightGBM text")?;
    let text = std::str::from_utf8(&bytes).map_err(|e| malformed(format!("LightGBM text: {e}")))?;
    let mut lines = text.lines();
    if lines.next().map(str::trim_end) != Some("tree") {
        return Err(malformed("LightGBM text must start with tree"));
    }
    let mut global = Fields::default();
    let mut trees: Vec<Fields<'_>> = Vec::new();
    for line in lines {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        if line == "end of trees" {
            break;
        }
        let (key, value) = match line.split_once('=') {
            Some(pair) => pair,
            None if line == "average_output" && trees.is_empty() => (line, ""),
            None => return Err(malformed(format!("expected key=value: {line:?}"))),
        };
        if key == "Tree" {
            let id: usize = number(value, "Tree")?;
            if id != trees.len() {
                return Err(malformed("Tree indices must be consecutive from zero"));
            }
            cap(trees.len() + 1, 4, options)?;
            trees.push(Fields::default());
        } else {
            let fields = trees.last_mut().unwrap_or(&mut global);
            if fields.insert(key, value).is_some() {
                return Err(malformed(format!("duplicate {key}")));
            }
        }
    }
    let max_feature: i32 = scalar(&global, "max_feature_idx")?;
    let num_feature = i64::from(max_feature) + 1;
    let num_class: i32 = scalar(&global, "num_class")?;
    if num_class > 1 {
        return Err(LoadError::Unsupported("LightGBM num_class > 1".into()));
    }
    if num_class != 1 || max_feature < 0 {
        return Err(malformed("invalid num_class or max_feature_idx"));
    }
    if global.contains_key("num_tree_per_iteration")
        && scalar::<i32>(&global, "num_tree_per_iteration")? != 1
    {
        return Err(LoadError::Unsupported(
            "LightGBM num_tree_per_iteration must be 1".into(),
        ));
    }
    let (postprocessor, sigmoid_alpha) = objective(global.get("objective").copied())?;
    let output = Metadata {
        num_tree: trees.len(),
        num_feature,
        task_type: "kRegressor".into(),
        average: global.contains_key("average_output"),
        num_target: 1,
        num_class: vec![1],
        leaf_shape: vec![1, 1],
        target_id: vec![0; trees.len()],
        class_id: vec![0; trees.len()],
        postprocessor: postprocessor.into(),
        sigmoid_alpha,
        base_scores: vec![0.0],
    }
    .validate(config)?;
    let mut total = 0usize;
    for tree in &trees {
        let leaves: usize = scalar(tree, "num_leaves")?;
        let n = leaves
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| malformed("invalid num_leaves"))?;
        if field(tree, "leaf_value")?.len() < n {
            return Err(malformed("leaf_value text is too short for num_leaves"));
        }
        cap(n, 8, options)?;
        check_node_total(total, n)?;
        total += n;
    }
    let mut decoded = Decoded::new(Some(total))?;
    let mut intern = ListIntern::default();
    let mut ctx = ParseContext {
        decoded: &mut decoded,
        list_intern: choices.bitset_intern.then_some(&mut intern),
        config,
        threshold_type: ThresholdType::F64,
        list: Vec::new(),
    };
    let mut scratch = OrderScratch::default();
    let mut category_entries = 0;
    for (i, tree) in trees.iter().enumerate() {
        process_tree(tree, &mut ctx, &mut scratch, &mut category_entries, options)
            .map_err(|e| e.at(&format!("tree {i}")))?;
    }
    decoded.seal()?;
    Ok(ParsedModel {
        decoded,
        threshold_type: ThresholdType::F64,
        output,
    })
}

/// The names LightGBM saves, with Treelite's output transforms.
fn objective(text: Option<&str>) -> Result<(&'static str, f64), LoadError> {
    let Some(text) = text else {
        return Ok(("identity", 1.0));
    };
    let mut tokens = text.split_whitespace();
    let name = tokens.next().ok_or_else(|| malformed("empty objective"))?;
    match name {
        "regression" | "regression_l1" | "mape" | "huber" | "fair" | "quantile" => {
            if tokens.any(|p| p == "sqrt") {
                return Err(LoadError::Unsupported("LightGBM sqrt postprocessor".into()));
            }
            Ok(("identity", 1.0))
        }
        "binary" => {
            for param in tokens {
                if let Some(value) = param.strip_prefix("sigmoid:") {
                    let alpha: f32 = number(value, "sigmoid")?;
                    return Ok(("sigmoid", f64::from(alpha)));
                }
            }
            Err(malformed("binary objective needs a sigmoid parameter"))
        }
        "cross_entropy" => Ok(("sigmoid", 1.0)),
        "lambdarank" | "rank_xendcg" => Ok(("identity", 1.0)),
        _ => Err(LoadError::Unsupported(format!(
            "LightGBM objective {name:?}"
        ))),
    }
}

/// `category_entries` counts the model's categories so far: a bitset word of
/// eleven bytes expands to 32 of them, so the text cap alone does not bound them.
fn process_tree(
    fields: &Fields<'_>,
    ctx: &mut ParseContext<'_>,
    scratch: &mut OrderScratch,
    category_entries: &mut usize,
    options: &LoadOptions,
) -> Result<(), LoadError> {
    if fields.contains_key("is_linear") && scalar::<i32>(fields, "is_linear")? != 0 {
        return Err(LoadError::Unsupported("LightGBM linear trees".into()));
    }
    let leaves: usize = scalar(fields, "num_leaves")?;
    let splits = leaves - 1; // checked before allocating the decoded forest
    let num_cat: usize = scalar(fields, "num_cat")?;
    if num_cat > splits {
        return Err(malformed("num_cat exceeds split count"));
    }
    let values: Vec<f64> = array(fields, "leaf_value", leaves, options)?;
    let leaf_counts: Vec<i32> = array(fields, "leaf_count", leaves, options)?;
    let category_start = ctx.decoded.categories.len() as u32;
    if splits == 0 {
        let leaf = TempNode::leaf(
            validation::leaf(values[0], ThresholdType::F64)?,
            validation::weight(f64::from(leaf_counts[0]))?,
        );
        return order_tree(&[leaf], 0, category_start, ctx, scratch);
    }
    let internal_counts: Vec<i32> = array(fields, "internal_count", splits, options)?;
    let features: Vec<i32> = array(fields, "split_feature", splits, options)?;
    let thresholds: Vec<f64> = array(fields, "threshold", splits, options)?;
    let decisions: Vec<i8> = array(fields, "decision_type", splits, options)?;
    let left: Vec<i32> = array(fields, "left_child", splits, options)?;
    let right: Vec<i32> = array(fields, "right_child", splits, options)?;
    let (boundaries, words): (Vec<usize>, Vec<u32>) = if num_cat > 0 {
        let boundaries: Vec<usize> = array(fields, "cat_boundaries", num_cat + 1, options)?;
        if boundaries[0] != 0 || !boundaries.is_sorted() {
            return Err(malformed("invalid cat_boundaries"));
        }
        let words = array(fields, "cat_threshold", boundaries[num_cat], options)?;
        (boundaries, words)
    } else {
        (Vec::new(), Vec::new())
    };
    let mut temp = Vec::new();
    let child = |old: i32| -> Result<usize, LoadError> {
        let index = if old < 0 {
            splits + (!old as u32) as usize
        } else {
            old as usize
        };
        if (old >= 0 && index >= splits) || index >= splits + leaves {
            return Err(malformed(format!("child index {old} out of range")));
        }
        Ok(index)
    };
    // The builder records nodes in right-first preorder, not LightGBM's split
    // order. Match it before encoding categories and interning splits as well.
    let mut order = Vec::new();
    let mut positions = vec![u32::MAX; splits + leaves];
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        if positions[i] != u32::MAX {
            return Err(malformed("cycle or shared child"));
        }
        positions[i] = order.len() as u32;
        order.push(i);
        if i < splits {
            stack.push(child(left[i])?);
            stack.push(child(right[i])?);
        }
    }
    if order.len() != splits + leaves {
        return Err(malformed("unreachable nodes"));
    }
    let mut used = vec![false; num_cat];
    for i in order {
        if i >= splits {
            let leaf = i - splits;
            temp.push(TempNode::leaf(
                validation::leaf(values[leaf], ThresholdType::F64)?,
                validation::weight(f64::from(leaf_counts[leaf]))?,
            ));
            continue;
        }
        let feature = features[i];
        if feature < 0 || feature as usize >= ctx.config.n_features() {
            return Err(malformed(format!("split_feature {feature} out of range")));
        }
        let cat = decisions[i] & 1 != 0;
        let (value, inline, default_left) = if cat {
            let t = thresholds[i];
            // Treelite casts the threshold to int, truncating toward zero.
            if !t.is_finite() || t < 0.0 || t.trunc() >= num_cat as f64 {
                return Err(malformed("categorical threshold index out of range"));
            }
            let index = t as usize;
            if std::mem::replace(&mut used[index], true) {
                return Err(malformed("shared categorical threshold index"));
            }
            let range = &words[boundaries[index]..boundaries[index + 1]];
            // A JSON document holds at most one category per two bytes.
            let entries = range.iter().map(|w| w.count_ones() as usize).sum::<usize>();
            *category_entries = category_entries
                .checked_add(entries)
                .filter(|&n| n as u64 <= options.max_json_bytes / 2)
                .ok_or_else(|| {
                    LoadError::Limit(format!(
                        "LightGBM categories exceed {} entries (LoadOptions::max_json_bytes / 2)",
                        options.max_json_bytes / 2
                    ))
                })?;
            if range.len() as u64 > (u64::from(u32::MAX) + 1) / 32 {
                return Err(malformed("category ID exceeds u32"));
            }
            let categories = range.iter().enumerate().flat_map(|(word, &bits)| {
                (0..32).filter_map(move |bit| {
                    (bits & (1 << bit) != 0).then_some((word * 32 + bit) as u32)
                })
            });
            let (value, inline) = encode_categories(categories, ctx)?;
            (value, inline, false)
        } else {
            let value = validation::threshold(thresholds[i], "<=", ThresholdType::F64)?;
            let default_left = if (decisions[i] >> 2) & 3 == 2 {
                decisions[i] & 2 != 0
            } else {
                0.0 <= value
            };
            (value, false, default_left)
        };
        temp.push(TempNode {
            value,
            left: positions[child(left[i])?],
            right: positions[child(right[i])?],
            feature: feature as u32,
            flags: build_flags(
                default_left,
                cat,
                inline,
                ctx.config.is_varying(feature as usize),
            ),
            weight: validation::weight(f64::from(internal_counts[i]))?,
        });
    }
    order_tree(&temp, 0, category_start, ctx, scratch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_arrays_fail_before_the_cap() {
        let mut fields = Fields::default();
        let options = LoadOptions {
            max_binary_array_bytes: 0,
            ..LoadOptions::default()
        };
        for text in [None, Some("0")] {
            if let Some(text) = text {
                fields.insert("threshold", text);
            }
            assert!(matches!(
                array::<f64>(&fields, "threshold", usize::MAX, &options),
                Err(LoadError::MalformedModel(_))
            ));
        }
        assert!(matches!(
            array::<f64>(&fields, "threshold", 1, &options),
            Err(LoadError::Limit(_))
        ));
    }

    fn same(a: &ParsedModel, b: &ParsedModel) {
        assert_eq!(a.threshold_type, b.threshold_type);
        assert_eq!(a.output.base_score.to_bits(), b.output.base_score.to_bits());
        assert_eq!(a.output.divisor.to_bits(), b.output.divisor.to_bits());
        assert_eq!(a.output.postprocessor, b.output.postprocessor);
        let (a, b) = (&a.decoded, &b.decoded);
        assert_eq!(a.categories, b.categories);
        assert_eq!(a.node_count(), b.node_count());
        assert_eq!(a.trees.len(), b.trees.len());
        for (i, (ta, tb)) in a.trees.iter().zip(&b.trees).enumerate() {
            let fields = |t: &super::super::decoded::DecodedTree| {
                (t.start, t.count, t.category_start, t.leaves, t.varying)
            };
            assert_eq!(fields(ta), fields(tb));
            assert_eq!(a.firsts(i), b.firsts(i));
            assert_eq!(
                a.nodes(i).iter().collect::<Vec<_>>(),
                b.nodes(i).iter().collect::<Vec<_>>()
            );
        }
        let splits = |d: &Decoded| {
            d.splits
                .all
                .iter()
                .map(|s| (s.value.to_bits(), s.feature, s.flags, s.sides))
                .collect::<Vec<_>>()
        };
        assert_eq!(splits(a), splits(b));
    }

    #[test]
    fn native_and_treelite_decoded_models_are_identical() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/import/lightgbm");
        for entry in std::fs::read_dir(&root).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|ext| ext != "txt") {
                continue;
            }
            let text = std::fs::read(&path).unwrap();
            let binary = std::fs::read(path.with_extension("bin")).unwrap();
            for varying in [false, true] {
                let config = WalkerConfig::builder(2)
                    .max_group_width(128)
                    .varying(if varying { vec![0, 1] } else { vec![] })
                    .build()
                    .unwrap();
                for bitset_intern in [false, true] {
                    let choices = Choices {
                        bitset_intern,
                        ..Choices::default()
                    };
                    let options = LoadOptions::default();
                    let a = parse(text.as_slice(), &config, &options, &choices).unwrap();
                    let b = super::super::binary::parse_bytes(&binary, &config, &options, &choices)
                        .unwrap();
                    same(&a, &b);
                }
            }
        }
    }
}
