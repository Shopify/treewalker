//! Streaming Treelite v4 reader with reusable, length-checked per-tree buffers.
use super::common::{
    ParseContext, ReorderScratch, TempNode, build_flags, classify_feature, encode_categories,
    reorder_and_emit,
};
use super::validation::{self, MAX_TREES, Metadata, ParsedModel};
use crate::forest::{ThresholdType, Tree};
use crate::{LoadError, LoadOptions, WalkerConfig};
use rustc_hash::FxHashMap;
use std::io::{BufReader, Read};

trait Element: Copy {
    const SIZE: usize;
    /// Decode one value from exactly `SIZE` bytes.
    fn decode(bytes: &[u8]) -> Self;
    /// Append the values of `bytes`, whose length is a multiple of `SIZE`.
    fn extend_from(out: &mut Vec<Self>, bytes: &[u8]);
}
macro_rules! element {
    ($($t:ty),*) => { $(impl Element for $t {
        const SIZE: usize = size_of::<Self>();
        fn decode(bytes: &[u8]) -> Self { Self::from_le_bytes(bytes.try_into().unwrap()) }
        fn extend_from(out: &mut Vec<Self>, bytes: &[u8]) {
            let (values, rest) = bytes.as_chunks::<{ size_of::<$t>() }>();
            debug_assert!(rest.is_empty());
            out.extend(values.iter().map(|&b| Self::from_le_bytes(b)));
        }
    })* };
}
element!(u8, i8, i32, u32, u64, f32, f64);

#[derive(Clone, Copy)]
enum Length {
    Exact(usize),
    Optional(usize),
    Max(usize),
}
struct Reader<R> {
    input: R,
    raw: Vec<u8>,
    consumed: u64,
}
impl<R: Read> Reader<R> {
    fn account(&mut self, n: usize, field: &str) -> Result<(), LoadError> {
        self.consumed = self
            .consumed
            .checked_add(n as u64)
            .filter(|&n| n <= 4 * 1024 * 1024 * 1024)
            .ok_or_else(|| LoadError::Limit("binary input exceeds 4 GiB".into()))?;
        if n > 64 * 1024 * 1024 {
            return Err(LoadError::Limit(format!("{field}: array exceeds 64 MiB")));
        }
        Ok(())
    }
    fn bytes(&mut self, n: usize, field: &str) -> Result<(), LoadError> {
        self.account(n, field)?;
        self.raw
            .try_reserve(n.saturating_sub(self.raw.len()))
            .map_err(|e| LoadError::Limit(e.to_string()))?;
        self.raw.resize(n, 0);
        read_exact(&mut self.input, &mut self.raw, field)
    }
    fn scalar<T: Element>(&mut self, field: &str) -> Result<T, LoadError> {
        self.account(T::SIZE, field)?;
        // Keep the array buffer's length intact across scalar/length reads.
        // All supported Element types fit in eight bytes.
        let mut buf = [0; 8];
        let bytes = &mut buf[..T::SIZE];
        read_exact(&mut self.input, bytes, field)?;
        Ok(T::decode(bytes))
    }
    fn boolean(&mut self, field: &str) -> Result<bool, LoadError> {
        match self.scalar::<u8>(field)? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(LoadError::MalformedModel(format!(
                "{field}: invalid bool {v}"
            ))),
        }
    }
    fn array<T: Element>(
        &mut self,
        out: &mut Vec<T>,
        length: Length,
        field: &str,
    ) -> Result<(), LoadError> {
        let declared = self.scalar::<u64>(field)?;
        let n = usize::try_from(declared)
            .map_err(|_| LoadError::Limit(format!("{field}: length {declared}")))?;
        let valid = match length {
            Length::Exact(x) => n == x,
            Length::Optional(x) => n == 0 || n == x,
            Length::Max(x) => n <= x,
        };
        if !valid {
            return Err(LoadError::MalformedModel(format!(
                "{field}: invalid declared length {declared}"
            )));
        }
        let bytes = n
            .checked_mul(T::SIZE)
            .ok_or_else(|| LoadError::Limit(format!("{field}: byte length overflow")))?;
        self.bytes(bytes, field)?;
        out.clear();
        out.try_reserve(n)
            .map_err(|e| LoadError::Limit(e.to_string()))?;
        T::extend_from(out, &self.raw);
        Ok(())
    }
    fn bools(&mut self, out: &mut Vec<u8>, length: Length, field: &str) -> Result<(), LoadError> {
        self.array(out, length, field)?;
        if out.iter().any(|&v| v > 1) {
            return Err(LoadError::MalformedModel(format!("{field}: invalid bool")));
        }
        Ok(())
    }
    fn values(
        &mut self,
        out: &mut Vec<f64>,
        n: usize,
        kind: ThresholdType,
        field: &str,
    ) -> Result<(), LoadError> {
        if kind == ThresholdType::F64 {
            self.array(out, Length::Exact(n), field)
        } else {
            let len = self.scalar::<u64>(field)?;
            if len != n as u64 {
                return Err(LoadError::MalformedModel(format!(
                    "{field}: length {len}, expected {n}"
                )));
            }
            self.bytes(n * 4, field)?;
            out.clear();
            let (values, _) = self.raw.as_chunks::<4>();
            out.extend(values.iter().map(|&b| f64::from(f32::from_le_bytes(b))));
            Ok(())
        }
    }
    fn string(&mut self, max: usize, field: &str) -> Result<String, LoadError> {
        let mut s = Vec::new();
        self.array(&mut s, Length::Max(max), field)?;
        if s.last() == Some(&0) {
            s.pop();
        }
        String::from_utf8(s).map_err(|e| LoadError::MalformedModel(format!("{field}: {e}")))
    }
    fn extension(&mut self, field: &str) -> Result<(), LoadError> {
        let n = self.scalar::<i32>(field)?;
        if n != 0 {
            return Err(LoadError::Unsupported(format!(
                "{field}={n}; extensions are unsupported"
            )));
        }
        Ok(())
    }
}

fn read_exact(reader: &mut impl Read, bytes: &mut [u8], field: &str) -> Result<(), LoadError> {
    reader.read_exact(bytes).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            LoadError::MalformedModel(format!("{field}: truncated binary field"))
        } else {
            LoadError::Io(e)
        }
    })
}

#[derive(Default)]
struct TreeBufs {
    node_type: Vec<i8>,
    cleft: Vec<i32>,
    cright: Vec<i32>,
    split_index: Vec<i32>,
    default_left: Vec<u8>,
    leaf_value: Vec<f64>,
    threshold: Vec<f64>,
    cmp: Vec<i8>,
    cat_right: Vec<u8>,
    leaf_begin: Vec<u64>,
    leaf_end: Vec<u64>,
    categories: Vec<u32>,
    cat_begin: Vec<u64>,
    cat_end: Vec<u64>,
    data_count: Vec<u64>,
    data_present: Vec<u8>,
    sum_hess: Vec<f64>,
    hess_present: Vec<u8>,
    gain: Vec<f64>,
    gain_present: Vec<u8>,
    temp: Vec<TempNode>,
}

pub(super) fn parse(
    reader: impl Read,
    config: &WalkerConfig,
    options: &LoadOptions,
) -> Result<ParsedModel, LoadError> {
    let mut r = Reader {
        input: BufReader::with_capacity(128 * 1024, reader),
        raw: Vec::new(),
        consumed: 0,
    };
    let major = r.scalar::<i32>("major_ver")?;
    let minor = r.scalar::<i32>("minor_ver")?;
    let patch = r.scalar::<i32>("patch_ver")?;
    if major != 4 || !(0..=7).contains(&minor) || patch < 0 {
        return Err(LoadError::Unsupported(format!(
            "Treelite version {major}.{minor}.{patch}; supported 4.0 through 4.7"
        )));
    }
    let thr = r.scalar::<u8>("threshold_type")?;
    let leaf = r.scalar::<u8>("leaf_output_type")?;
    let threshold_type = match (thr, leaf) {
        (2, 2) => ThresholdType::F32,
        (3, 3) => ThresholdType::F64,
        _ => {
            return Err(LoadError::Unsupported(format!(
                "threshold_type={thr}, leaf_output_type={leaf}"
            )));
        }
    };
    let count = r.scalar::<u64>("num_tree")?;
    if count == 0 || count > MAX_TREES as u64 {
        return Err(LoadError::Limit(format!(
            "num_tree={count}; require 1..={MAX_TREES}"
        )));
    }
    let num_tree = count as usize;
    let num_feature = i64::from(r.scalar::<i32>("num_feature")?);
    let task = r.scalar::<u8>("task_type")?;
    let task_type = match task {
        0 => "kBinaryClf",
        1 => "kRegressor",
        3 => "kLearningToRank",
        _ => return Err(LoadError::Unsupported(format!("task_type={task}"))),
    }
    .into();
    let average = r.boolean("average_tree_output")?;
    let num_target = i64::from(r.scalar::<i32>("num_target")?);
    if num_target != 1 {
        return Err(LoadError::Unsupported(format!(
            "num_target={num_target}; require scalar output"
        )));
    }
    let (mut num_class, mut leaf_shape, mut target_id, mut class_id) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    r.array(&mut num_class, Length::Exact(1), "num_class")?;
    r.array(&mut leaf_shape, Length::Exact(2), "leaf_vector_shape")?;
    r.array(&mut target_id, Length::Exact(num_tree), "target_id")?;
    r.array(&mut class_id, Length::Exact(num_tree), "class_id")?;
    let postprocessor = r.string(64, "postprocessor")?;
    let sigmoid_alpha = f64::from(r.scalar::<f32>("sigmoid_alpha")?);
    let _ratio_c = r.scalar::<f32>("ratio_c")?;
    let mut base_scores = Vec::new();
    r.array(&mut base_scores, Length::Exact(1), "base_scores")?;
    validation::attributes(&r.string(16 * 1024 * 1024, "attributes")?)?;
    r.extension("num_opt_field_per_model")?;
    let meta = Metadata {
        num_tree,
        num_feature,
        task_type,
        average,
        num_target,
        num_class,
        leaf_shape,
        target_id,
        class_id,
        postprocessor,
        sigmoid_alpha,
        base_scores,
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
    let mut b = TreeBufs::default();
    let mut scratch = ReorderScratch::new();
    for i in 0..num_tree {
        trees.push(
            read_tree(&mut r, &mut ctx, &mut b, &mut scratch)
                .map_err(|e| e.at(&format!("tree {i}")))?,
        );
    }
    let mut trailing = [0];
    if r.input.read(&mut trailing)? != 0 {
        return Err(LoadError::Unsupported(
            "trailing bytes after final tree".into(),
        ));
    }
    Ok(ParsedModel {
        trees,
        nodes,
        bitsets,
        threshold_type,
        output,
    })
}

fn read_tree<R: Read>(
    r: &mut Reader<R>,
    ctx: &mut ParseContext<'_>,
    b: &mut TreeBufs,
    scratch: &mut ReorderScratch,
) -> Result<Tree, LoadError> {
    let count = r.scalar::<i32>("num_nodes")?;
    if !(1..=i32::from(i16::MAX)).contains(&count) {
        return Err(LoadError::Limit(format!(
            "num_nodes={count}; require 1..=32767"
        )));
    }
    let n = count as usize;
    let has_cat = r.boolean("has_categorical_split")?;
    r.array(&mut b.node_type, Length::Exact(n), "node_type")?;
    r.array(&mut b.cleft, Length::Exact(n), "cleft")?;
    r.array(&mut b.cright, Length::Exact(n), "cright")?;
    r.array(&mut b.split_index, Length::Exact(n), "split_index")?;
    r.bools(&mut b.default_left, Length::Exact(n), "default_left")?;
    r.values(&mut b.leaf_value, n, ctx.threshold_type, "leaf_value")?;
    r.values(&mut b.threshold, n, ctx.threshold_type, "threshold")?;
    r.array(&mut b.cmp, Length::Exact(n), "cmp")?;
    r.bools(
        &mut b.cat_right,
        Length::Exact(n),
        "category_list_right_child",
    )?;
    // Scalar models have no vector payload; reject before allocating any.
    let vector_len = r.scalar::<u64>("leaf_vector")?;
    if vector_len != 0 {
        return Err(LoadError::Unsupported(format!(
            "leaf_vector length {vector_len}; vector leaves are unsupported"
        )));
    }
    r.array(&mut b.leaf_begin, Length::Optional(n), "leaf_vector_begin")?;
    r.array(&mut b.leaf_end, Length::Optional(n), "leaf_vector_end")?;
    if b.leaf_begin.len() != b.leaf_end.len()
        || b.leaf_begin.iter().chain(&b.leaf_end).any(|&x| x != 0)
    {
        return Err(LoadError::MalformedModel(
            "leaf_vector offsets must be zero for scalar leaves".into(),
        ));
    }
    r.array(
        &mut b.categories,
        Length::Max(16 * 1024 * 1024),
        "category_list",
    )?;
    r.array(&mut b.cat_begin, Length::Exact(n), "category_list_begin")?;
    r.array(&mut b.cat_end, Length::Exact(n), "category_list_end")?;
    r.array(&mut b.data_count, Length::Optional(n), "data_count")?;
    r.bools(
        &mut b.data_present,
        Length::Exact(b.data_count.len()),
        "data_count_present",
    )?;
    r.array(&mut b.sum_hess, Length::Optional(n), "sum_hess")?;
    r.bools(
        &mut b.hess_present,
        Length::Exact(b.sum_hess.len()),
        "sum_hess_present",
    )?;
    r.array(&mut b.gain, Length::Optional(n), "gain")?;
    r.bools(
        &mut b.gain_present,
        Length::Exact(b.gain.len()),
        "gain_present",
    )?;
    r.extension("num_opt_field_per_tree")?;
    r.extension("num_opt_field_per_node")?;
    if has_cat != b.node_type.contains(&2) {
        return Err(LoadError::MalformedModel(
            "has_categorical_split disagrees with node_type".into(),
        ));
    }
    // Bound total decoding work even if several valid segments overlap.
    let mut category_items = 0u64;
    for (&begin, &end) in b.cat_begin.iter().zip(&b.cat_end) {
        if begin > end || end > b.categories.len() as u64 {
            return Err(LoadError::MalformedModel(format!(
                "invalid category segment {begin}..{end}"
            )));
        }
        category_items += end - begin;
        if category_items > 16 * 1024 * 1024 {
            return Err(LoadError::Limit(
                "more than 16 million referenced category entries in one tree".into(),
            ));
        }
    }
    let bitset_start = ctx.bitsets.len() as u32;
    b.temp.clear();
    for i in 0..n {
        let node = decode_node(i, n, ctx, b).map_err(|e| e.at(&format!("node {i}")))?;
        b.temp.push(node);
    }
    reorder_and_emit(&b.temp, 0, bitset_start, ctx, scratch)
}

fn decode_node(
    i: usize,
    n: usize,
    ctx: &mut ParseContext<'_>,
    b: &TreeBufs,
) -> Result<TempNode, LoadError> {
    let kind = b.node_type[i];
    if !(0..=2).contains(&kind) {
        return Err(LoadError::Unsupported(format!("node_type={kind}")));
    }
    let weight = if b.data_present.get(i) == Some(&1) {
        b.data_count[i] as f64
    } else if b.hess_present.get(i) == Some(&1) {
        b.sum_hess[i]
    } else {
        1.0
    };
    if !weight.is_finite() || weight < 0.0 {
        return Err(LoadError::MalformedModel(format!(
            "invalid node statistic {weight}"
        )));
    }
    let (begin, end) = (b.cat_begin[i], b.cat_end[i]);
    if begin > end || end > b.categories.len() as u64 || (kind != 2 && begin != end) {
        return Err(LoadError::MalformedModel(format!(
            "category segment {begin}..{end} is invalid"
        )));
    }
    if kind == 0 {
        if b.cleft[i] != -1 || b.cright[i] != -1 || b.split_index[i] != -1 {
            return Err(LoadError::MalformedModel(
                "leaf has children or split_index".into(),
            ));
        }
        return Ok(TempNode {
            value: validation::leaf(b.leaf_value[i], ctx.threshold_type)?,
            left: -1,
            right: -1,
            feature: 0,
            flags: 0,
            cat_n_words: 0,
            weight,
        });
    }
    let feature = b.split_index[i];
    if feature < 0 || feature as usize >= ctx.config.n_features() {
        return Err(LoadError::MalformedModel(format!(
            "split_index={feature} out of range"
        )));
    }
    let (mut left, mut right) = (b.cleft[i], b.cright[i]);
    if left < 0 || right < 0 || left as usize >= n || right as usize >= n {
        return Err(LoadError::MalformedModel(format!(
            "invalid children {left}, {right}"
        )));
    }
    let mut dl = b.default_left[i] != 0;
    let (value, inline, words) = if kind == 2 {
        // Normalize membership-right into membership-left, including missing routing.
        if b.cat_right[i] != 0 {
            std::mem::swap(&mut left, &mut right);
            dl = !dl;
        }
        encode_categories(&b.categories[begin as usize..end as usize], ctx)?
    } else {
        let op = match b.cmp[i] {
            2 => "<",
            3 => "<=",
            x => return Err(LoadError::Unsupported(format!("comparison_op={x}"))),
        };
        (
            validation::threshold(b.threshold[i], op, ctx.threshold_type)?,
            false,
            0,
        )
    };
    Ok(TempNode {
        value,
        left: left as i16,
        right: right as i16,
        feature: feature as u16,
        flags: build_flags(
            dl,
            kind == 2,
            inline,
            classify_feature(ctx.config, feature as usize),
        ),
        cat_n_words: words,
        weight,
    })
}
