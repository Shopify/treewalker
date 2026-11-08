//! Native LightGBM text, normalized like Treelite 4.7.2's LightGBM frontend.
//! NaN at None and Zero missing-type splits compares as zero, ignoring the default
//! bit; LightGBM routes Zero-type NaN and zeros by that bit.

use std::collections::HashMap;
use std::io::{BufRead, Take};
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
    let expected = || malformed(format!("{key}: expected {n} entries"));
    // A wrong count is malformed rather than over the cap, and rather than an
    // invalid entry; only those errors count the entries separately.
    let or_count = |error| {
        if text.split_ascii_whitespace().count() == n {
            error
        } else {
            expected()
        }
    };
    cap(n, size_of::<T>(), options).map_err(or_count)?;
    // Do not allocate from a declared count without its corresponding input: an
    // entry and its separator take two bytes, but the last.
    let mut values = Vec::with_capacity(n.min(text.len() / 2 + 1));
    for entry in text.split_ascii_whitespace() {
        if values.len() == n {
            return Err(expected());
        }
        values.push(number(entry, key).map_err(or_count)?);
    }
    if values.len() != n {
        return Err(expected());
    }
    Ok(values)
}

/// The text's lines, read under the shared text/JSON input cap and cut into
/// sections, the header and then one per tree, so that one tree's text is held at a
/// time.
struct Sections<R> {
    input: Take<R>,
    max: u64,
    read: u64,
    line: Vec<u8>,
    /// The `Tree=` line that ended the last section.
    tree: String,
}

impl<R: BufRead> Sections<R> {
    fn new(input: R, max: u64) -> Self {
        Self {
            input: input.take(max.saturating_add(1)),
            max,
            read: 0,
            line: Vec::new(),
            tree: String::new(),
        }
    }

    fn count(&mut self, bytes: u64) -> Result<(), LoadError> {
        self.read += bytes;
        if self.read > self.max {
            return Err(validation::too_long("LightGBM text", self.max));
        }
        Ok(())
    }

    /// The next line into `self.line`, counted; false at the input's end.
    fn read(&mut self) -> Result<bool, LoadError> {
        self.line.clear();
        let n = self.input.read_until(b'\n', &mut self.line)?;
        self.count(n as u64)?;
        Ok(n > 0)
    }

    /// The next section into `text`: the line that ended the last one, then each
    /// line before the next `Tree=` line. Whether a tree follows; a section also
    /// ends at `end of trees` and at the input's end.
    fn next(&mut self, text: &mut String) -> Result<bool, LoadError> {
        text.clear();
        text.push_str(&self.tree);
        while self.read()? {
            let line = utf8(&self.line, self.read)?;
            let trimmed = line.trim_end();
            if trimmed == "end of trees" {
                return Ok(false);
            }
            if trimmed.starts_with("Tree=") {
                self.tree.clear();
                self.tree.push_str(line);
                return Ok(true);
            }
            text.push_str(line);
        }
        Ok(false)
    }

    /// Read the rest of the input, which no tree is decoded from, under the cap and
    /// as UTF-8, as every line before it.
    fn finish(mut self) -> Result<(), LoadError> {
        while self.read()? {
            utf8(&self.line, self.read)?;
        }
        Ok(())
    }
}

/// A line's text; a character never spans lines. `end` is the line's end in the
/// input, to say where an invalid byte is.
fn utf8(line: &[u8], end: u64) -> Result<&str, LoadError> {
    std::str::from_utf8(line).map_err(|e| {
        let at = end - line.len() as u64 + e.valid_up_to() as u64;
        malformed(format!("LightGBM text: invalid UTF-8 at byte offset {at}"))
    })
}

/// A section's `key=value` lines; the header may also say `average_output`.
fn parse_fields<'a>(
    lines: impl Iterator<Item = &'a str>,
    header: bool,
) -> Result<Fields<'a>, LoadError> {
    let mut fields = Fields::default();
    for line in lines {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        let (key, value) = match line.split_once('=') {
            Some(pair) => pair,
            None if header && line == "average_output" => (line, ""),
            None => return Err(malformed(format!("expected key=value: {line:?}"))),
        };
        if fields.insert(key, value).is_some() {
            return Err(malformed(format!("duplicate {key}")));
        }
    }
    Ok(fields)
}

/// Decode a LightGBM text model one tree at a time, into one vector of `nodes`
/// nodes if they were counted first. The header is checked before the first tree,
/// and the forest's tree count after the last.
pub(super) fn parse(
    reader: impl BufRead,
    nodes: Option<usize>,
    config: &WalkerConfig,
    options: &LoadOptions,
    choices: &Choices,
) -> Result<ParsedModel, LoadError> {
    let mut sections = Sections::new(reader, options.max_json_bytes);
    // The header's text is freed before the first tree.
    let (mut more, metadata) = {
        let mut header = String::new();
        let more = sections.next(&mut header)?;
        let mut lines = header.lines();
        if lines.next().map(str::trim_end) != Some("tree") {
            return Err(malformed("LightGBM text must start with tree"));
        }
        let global = parse_fields(lines, true)?;
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
        let average = global.contains_key("average_output");
        let metadata = move |num_tree| Metadata {
            num_tree,
            num_feature,
            task_type: "kRegressor".into(),
            average,
            num_target: 1,
            num_class: vec![1],
            leaf_shape: vec![1, 1],
            target_id: vec![0; num_tree],
            class_id: vec![0; num_tree],
            postprocessor: postprocessor.into(),
            sigmoid_alpha,
            base_scores: vec![0.0],
        };
        (more, metadata)
    };
    // The checks that do not depend on the tree count, before the first tree; one
    // tree stands in for the count, which is checked after the last.
    metadata(1).validate(config)?;
    let mut decoded = Decoded::new(nodes)?;
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
    let mut trees = 0;
    let mut text = String::new();
    while more {
        let at = |e: LoadError| e.at(&format!("tree {trees}"));
        more = sections.next(&mut text).map_err(at)?;
        let mut lines = text.lines();
        let id = lines
            .next()
            .and_then(|line| line.trim_end().strip_prefix("Tree="));
        if number::<usize>(id.unwrap_or_default(), "Tree")? != trees {
            return Err(malformed("Tree indices must be consecutive from zero"));
        }
        cap(trees + 1, 4, options)?;
        process_tree(
            lines,
            &mut ctx,
            &mut scratch,
            &mut category_entries,
            options,
        )
        .map_err(at)?;
        trees += 1;
    }
    sections.finish()?;
    let output = metadata(trees).validate(config)?;
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

/// A word's set bits, lowest first: a word costs its categories, not its 32 bits.
struct SetBits(u32);

impl Iterator for SetBits {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        (self.0 != 0).then(|| {
            let bit = self.0.trailing_zeros();
            self.0 &= self.0 - 1;
            bit
        })
    }
}

/// A tree's leaves and nodes, checked against its leaf text, which must back them
/// before anything is allocated for them.
fn tree_size(fields: &Fields<'_>, options: &LoadOptions) -> Result<(usize, usize), LoadError> {
    let leaves: usize = scalar(fields, "num_leaves")?;
    let nodes = leaves
        .checked_mul(2)
        .and_then(|n| n.checked_sub(1))
        .ok_or_else(|| malformed("invalid num_leaves"))?;
    if field(fields, "leaf_value")?.len() < nodes {
        return Err(malformed("leaf_value text is too short for num_leaves"));
    }
    cap(nodes, 8, options)?;
    Ok((leaves, nodes))
}

/// The forest's nodes, by each tree's [`tree_size`], if the whole input is within
/// the cap and UTF-8; `None` if anything fails, for the decoder to report.
pub(super) fn count_nodes(reader: impl BufRead, options: &LoadOptions) -> Option<usize> {
    let mut sections = Sections::new(reader, options.max_json_bytes);
    let mut text = String::new();
    let mut more = sections.next(&mut text).ok()?;
    let mut total = 0usize;
    while more {
        more = sections.next(&mut text).ok()?;
        let fields = parse_fields(text.lines().skip(1), false).ok()?;
        let (_, nodes) = tree_size(&fields, options).ok()?;
        check_node_total(total, nodes).ok()?;
        total += nodes;
    }
    sections.finish().ok()?;
    Some(total)
}

/// Decode a tree from its section's lines after `Tree=`. `category_entries` counts
/// the model's categories: a bitset word of eleven bytes expands to 32 of them, so
/// the text cap alone does not bound them.
fn process_tree(
    lines: std::str::Lines<'_>,
    ctx: &mut ParseContext<'_>,
    scratch: &mut OrderScratch,
    category_entries: &mut usize,
    options: &LoadOptions,
) -> Result<(), LoadError> {
    let fields = &parse_fields(lines, false)?;
    let (leaves, n) = tree_size(fields, options)?;
    check_node_total(ctx.decoded.node_count(), n)?;
    if fields.contains_key("is_linear") && scalar::<i32>(fields, "is_linear")? != 0 {
        return Err(LoadError::Unsupported("LightGBM linear trees".into()));
    }
    let splits = leaves - 1; // num_leaves is positive, checked above
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
            let categories = range
                .iter()
                .enumerate()
                .flat_map(|(word, &bits)| SetBits(bits).map(move |bit| (word * 32) as u32 + bit));
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

    #[test]
    fn a_wrong_count_is_malformed_before_an_invalid_entry() {
        let mut fields = Fields::default();
        fields.insert("threshold", "0.5 x");
        let options = LoadOptions::default();
        let message = |n| match array::<f64>(&fields, "threshold", n, &options) {
            Err(LoadError::MalformedModel(message)) => message,
            other => panic!("{other:?}"),
        };
        for n in [1, 3] {
            assert!(message(n).contains(&format!("expected {n} entries")));
        }
        assert!(message(2).contains("invalid threshold"));
        // Any ASCII whitespace separates entries.
        fields.insert("threshold", " 0.5\t1 \x0c2 ");
        let values = array::<f64>(&fields, "threshold", 3, &options).unwrap();
        assert_eq!(values, [0.5, 1.0, 2.0]);
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

    /// The LightGBM text fixtures.
    fn fixtures() -> impl Iterator<Item = std::path::PathBuf> {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/import/lightgbm");
        std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "txt"))
    }

    #[test]
    fn native_and_treelite_decoded_models_are_identical() {
        for path in fixtures() {
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
                    let a = parse(text.as_slice(), None, &config, &options, &choices).unwrap();
                    let b = super::super::binary::parse_bytes(&binary, &config, &options, &choices)
                        .unwrap();
                    same(&a, &b);
                }
            }
        }
    }

    /// Lines that cross the reader's buffer, down to a byte at a time, decode as
    /// from memory.
    #[test]
    fn text_read_in_pieces_decodes_as_from_memory() {
        let config = WalkerConfig::builder(2)
            .max_group_width(128)
            .varying([0, 1])
            .build()
            .unwrap();
        let (options, choices) = (LoadOptions::default(), Choices::default());
        for path in fixtures() {
            let text = std::fs::read(&path).unwrap();
            let whole = parse(text.as_slice(), None, &config, &options, &choices).unwrap();
            for capacity in [1, 2, 3, 64] {
                let reader = std::io::BufReader::with_capacity(capacity, text.as_slice());
                same(
                    &whole,
                    &parse(reader, None, &config, &options, &choices).unwrap(),
                );
            }
            // Characters of several bytes, which a buffer can split, in feature names
            // and in the categories LightGBM saves after the trees.
            let text = String::from_utf8(text).unwrap();
            let key = "\nfeature_names=";
            let named = text.find(key).map_or_else(
                || text.replacen("tree\n", "tree\nfeature_names=größe_€\n", 1),
                |at| {
                    let (head, tail) = text.split_at(at + key.len());
                    format!("{head}größe_€ {tail}")
                },
            ) + "\npandas_categorical:[[\"prix_€\", \"größe\"]]\n";
            for capacity in [1, 2, 3] {
                let reader = std::io::BufReader::with_capacity(capacity, named.as_bytes());
                same(
                    &whole,
                    &parse(reader, None, &config, &options, &choices).unwrap(),
                );
            }
        }
    }

    /// The nodes counted before decoding are those decoded, and decoding into them
    /// decodes the same model. Nothing is counted for a tree whose leaf text does
    /// not back its count, nor for input past the cap or not UTF-8 after the trees.
    #[test]
    fn nodes_are_counted_as_decoded() {
        let config = WalkerConfig::builder(2)
            .max_group_width(128)
            .varying([0, 1])
            .build()
            .unwrap();
        let (options, choices) = (LoadOptions::default(), Choices::default());
        for path in fixtures() {
            let text = std::fs::read(&path).unwrap();
            let whole = parse(text.as_slice(), None, &config, &options, &choices).unwrap();
            let nodes = count_nodes(text.as_slice(), &options);
            assert_eq!(nodes, Some(whole.decoded.node_count()), "{path:?}");
            same(
                &whole,
                &parse(text.as_slice(), nodes, &config, &options, &choices).unwrap(),
            );
            let capped = LoadOptions {
                max_json_bytes: text.len() as u64,
                ..LoadOptions::default()
            };
            assert_eq!(count_nodes(text.as_slice(), &capped), nodes);
            let longer = [text.as_slice(), b"\n"].concat();
            assert_eq!(count_nodes(longer.as_slice(), &capped), None);
            let invalid = [text.as_slice(), b"\xff\n"].concat();
            assert_eq!(count_nodes(invalid.as_slice(), &options), None);
        }
        let unbacked = b"tree\nTree=0\nnum_leaves=2\nleaf_value=0\n";
        assert_eq!(count_nodes(&unbacked[..], &options), None);
        let backed = b"tree\nTree=0\nnum_leaves=2\nleaf_value=0 0\n";
        assert_eq!(count_nodes(&backed[..], &options), Some(3));
    }

    /// A read that fails after the trees fails the load as an I/O error, as one before
    /// them does.
    #[test]
    fn read_errors_after_the_trees_fail_the_load() {
        struct Failing<'a>(&'a [u8]);
        impl std::io::Read for Failing<'_> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.0.is_empty() {
                    return Err(std::io::Error::other("disconnected"));
                }
                self.0.read(buf)
            }
        }
        let text = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/import/lightgbm/missing_types.txt"),
        )
        .unwrap();
        assert!(text.ends_with(b"end of trees\n"));
        let config = WalkerConfig::builder(2)
            .max_group_width(1)
            .varying([])
            .build()
            .unwrap();
        let reader = std::io::BufReader::new(Failing(&text));
        let result = parse(
            reader,
            None,
            &config,
            &LoadOptions::default(),
            &Choices::default(),
        );
        assert!(matches!(result, Err(LoadError::Io(_))));
    }
}
