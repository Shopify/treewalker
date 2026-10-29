//! Treelite v4 binary reader: streaming from any reader through reusable, length-checked
//! field buffers, or straight from the bytes of [`crate::Forest::from_bytes`].
use super::decoded::Decoded;
use super::tree::{
    OrderScratch, ParseContext, TempNode, build_flags, encode_categories, order_tree,
};
use super::validation::{self, Metadata, ParsedModel};
use crate::config::Choices;
use crate::model::ThresholdType;
use crate::{LoadError, LoadOptions, WalkerConfig};
use rustc_hash::FxHashMap;
use std::io::Read;

/// A little-endian scalar of the checkpoint.
trait Element: Copy {
    const SIZE: usize;
    /// Decode one value from exactly `SIZE` bytes.
    fn decode(bytes: &[u8]) -> Self;
}
macro_rules! element {
    ($($t:ty),*) => { $(impl Element for $t {
        const SIZE: usize = size_of::<Self>();
        fn decode(bytes: &[u8]) -> Self { Self::from_le_bytes(bytes.try_into().unwrap()) }
    })* };
}
element!(u8, i8, i32, u32, u64, f32, f64);

/// The element count an array field must have.
#[derive(Clone, Copy)]
enum Length {
    Exact(usize),
    /// This many or none.
    Optional(usize),
    /// At most this many.
    Max(usize),
}

/// A tree's array fields, each a buffer of the source.
#[derive(Clone, Copy)]
#[repr(usize)]
enum Field {
    NodeType,
    Cleft,
    Cright,
    SplitIndex,
    DefaultLeft,
    LeafValue,
    Threshold,
    Cmp,
    CatRight,
    LeafBegin,
    LeafEnd,
    Categories,
    CatBegin,
    CatEnd,
    DataCount,
    DataPresent,
    SumHess,
    HessPresent,
    Gain,
    GainPresent,
    /// Model-level arrays, decoded as soon as they are read.
    Meta,
}
const FIELDS: usize = Field::Meta as usize + 1;

/// Where the binary input comes from: the bytes of each field, read into a buffer
/// per field from a reader, or borrowed from the input in memory.
trait Source {
    /// Consume `n` bytes into `field`'s buffer.
    fn take(&mut self, n: usize, field: Field, name: &str) -> Result<(), LoadError>;
    /// The bytes last taken into `field`.
    fn get(&self, field: Field) -> &[u8];
    /// Start a tree: earlier fields are no longer read.
    fn tree(&mut self) {}
    /// Whether input is left after the last tree.
    fn trailing(&mut self) -> Result<bool, LoadError>;
}

/// A [`Source`] over a reader, read in large blocks into one buffer that the
/// current tree's fields borrow, as [`Bytes`] borrows its input: the input is copied
/// once, by the reads, and read in few calls.
///
/// A field that runs past the buffered bytes moves the current tree's bytes to the
/// buffer's front, growing the buffer if the tree needs it, and reads on.
struct Stream<R> {
    input: R,
    buf: Vec<u8>,
    /// The next unread byte, the end of the buffered bytes, and where the current
    /// tree's fields begin.
    at: usize,
    filled: usize,
    tree: usize,
    spans: [(usize, usize); FIELDS],
}

/// The buffer's size once grown: bytes read at once, at most, for trees that fit.
const STREAM_BLOCK: usize = 1 << 20;

impl<R: Read> Stream<R> {
    const fn new(input: R) -> Self {
        Self {
            input,
            buf: Vec::new(),
            at: 0,
            filled: 0,
            tree: 0,
            spans: [(0, 0); FIELDS],
        }
    }

    /// Buffer at least `n` bytes past `at`; false at the input's end.
    fn fill(&mut self, n: usize) -> Result<bool, LoadError> {
        if self.filled - self.at >= n {
            return Ok(true);
        }
        // Keep only the current tree's bytes, at the front.
        let keep = self.tree;
        if keep > 0 {
            self.buf.copy_within(keep..self.filled, 0);
            for span in &mut self.spans {
                *span = (span.0.saturating_sub(keep), span.1.saturating_sub(keep));
            }
            self.at -= keep;
            self.filled -= keep;
            self.tree = 0;
        }
        // The buffer doubles from 64 KiB to a block, so small models stay small, and
        // grows past it only for a larger tree.
        let need = self.at + n;
        let len = need.max((2 * self.buf.len()).clamp(64 << 10, STREAM_BLOCK));
        if self.buf.len() < len {
            self.buf
                .try_reserve(len - self.buf.len())
                .map_err(|e| LoadError::Limit(e.to_string()))?;
            self.buf.resize(len, 0);
        }
        while self.filled < need {
            match self.input.read(&mut self.buf[self.filled..]) {
                Ok(0) => return Ok(false),
                Ok(k) => self.filled += k,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(LoadError::Io(e)),
            }
        }
        Ok(true)
    }
}

impl<R: Read> Source for Stream<R> {
    fn take(&mut self, n: usize, field: Field, name: &str) -> Result<(), LoadError> {
        if !self.fill(n)? {
            return Err(LoadError::MalformedModel(format!(
                "{name}: truncated binary field"
            )));
        }
        self.spans[field as usize] = (self.at, self.at + n);
        self.at += n;
        Ok(())
    }

    #[inline]
    fn get(&self, field: Field) -> &[u8] {
        let (a, b) = self.spans[field as usize];
        &self.buf[a..b]
    }

    fn tree(&mut self) {
        self.tree = self.at;
    }

    fn trailing(&mut self) -> Result<bool, LoadError> {
        self.tree = self.at;
        self.fill(1)
    }
}

/// A [`Source`] over the input's bytes, which fields borrow.
struct Bytes<'a> {
    input: &'a [u8],
    at: usize,
    spans: [(usize, usize); FIELDS],
}

impl Source for Bytes<'_> {
    fn take(&mut self, n: usize, field: Field, name: &str) -> Result<(), LoadError> {
        if self.input.len() - self.at < n {
            return Err(LoadError::MalformedModel(format!(
                "{name}: truncated binary field"
            )));
        }
        self.spans[field as usize] = (self.at, self.at + n);
        self.at += n;
        Ok(())
    }

    #[inline]
    fn get(&self, field: Field) -> &[u8] {
        let (a, b) = self.spans[field as usize];
        &self.input[a..b]
    }

    fn trailing(&mut self) -> Result<bool, LoadError> {
        Ok(self.at < self.input.len())
    }
}

/// A [`Source`] with the input caps of [`LoadOptions`].
struct Reader<S> {
    source: S,
    consumed: u64,
    /// [`LoadOptions::max_binary_bytes`] and [`LoadOptions::max_binary_array_bytes`].
    max_bytes: u64,
    max_array_bytes: u64,
}
impl<S: Source> Reader<S> {
    fn account(&mut self, n: usize, field: &str) -> Result<(), LoadError> {
        let max = self.max_bytes;
        self.consumed = self
            .consumed
            .checked_add(n as u64)
            .filter(|&n| n <= max)
            .ok_or_else(|| {
                LoadError::Limit(format!(
                    "binary input exceeds {max} bytes (LoadOptions::max_binary_bytes)"
                ))
            })?;
        if n as u64 > self.max_array_bytes {
            return Err(LoadError::Limit(format!(
                "{field}: array exceeds {} bytes (LoadOptions::max_binary_array_bytes)",
                self.max_array_bytes
            )));
        }
        Ok(())
    }
    fn bytes(&mut self, n: usize, into: Field, field: &str) -> Result<&[u8], LoadError> {
        self.account(n, field)?;
        self.source.take(n, into, field)?;
        Ok(self.source.get(into))
    }
    fn scalar<T: Element>(&mut self, field: &str) -> Result<T, LoadError> {
        Ok(T::decode(self.bytes(T::SIZE, Field::Meta, field)?))
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
    /// An array of `T` into `into`; returns its length.
    fn array<T: Element>(
        &mut self,
        into: Field,
        length: Length,
        field: &str,
    ) -> Result<usize, LoadError> {
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
        self.bytes(bytes, into, field)?;
        Ok(n)
    }
    /// A model-level array.
    fn vec<T: Element>(&mut self, length: Length, field: &str) -> Result<Vec<T>, LoadError> {
        self.array::<T>(Field::Meta, length, field)?;
        Ok(self
            .source
            .get(Field::Meta)
            .chunks_exact(T::SIZE)
            .map(T::decode)
            .collect())
    }
    fn bools(&mut self, into: Field, length: Length, field: &str) -> Result<usize, LoadError> {
        let n = self.array::<u8>(into, length, field)?;
        if self.source.get(into).iter().any(|&v| v > 1) {
            return Err(LoadError::MalformedModel(format!("{field}: invalid bool")));
        }
        Ok(n)
    }
    fn values(
        &mut self,
        into: Field,
        n: usize,
        kind: ThresholdType,
        field: &str,
    ) -> Result<(), LoadError> {
        if kind == ThresholdType::F64 {
            self.array::<f64>(into, Length::Exact(n), field)?;
        } else {
            let len = self.scalar::<u64>(field)?;
            if len != n as u64 {
                return Err(LoadError::MalformedModel(format!(
                    "{field}: length {len}, expected {n}"
                )));
            }
            self.bytes(n * 4, into, field)?;
        }
        Ok(())
    }
    fn string(&mut self, max: usize, field: &str) -> Result<String, LoadError> {
        let mut s = self.vec::<u8>(Length::Max(max), field)?;
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
    #[inline]
    fn field(&self, field: Field) -> &[u8] {
        self.source.get(field)
    }
}

/// Per-tree scratch.
#[derive(Default)]
struct TreeBufs {
    /// The nonempty category segments, to check that none overlap.
    segments: Vec<(u64, u64)>,
    temp: Vec<TempNode>,
    /// The categorical splits' indices.
    categorical: Vec<u32>,
}

/// Decode a checkpoint from `reader`, streaming each tree's fields through one
/// buffer; `nodes` is its node count, if known in advance.
pub(super) fn parse(
    reader: impl Read,
    nodes: Option<usize>,
    config: &WalkerConfig,
    options: &LoadOptions,
    choices: &Choices,
) -> Result<ParsedModel, LoadError> {
    parse_source(Stream::new(reader), nodes, config, options, choices)
}

/// [`parse`] of the input in memory, which the fields borrow.
pub(super) fn parse_bytes(
    input: &[u8],
    config: &WalkerConfig,
    options: &LoadOptions,
    choices: &Choices,
) -> Result<ParsedModel, LoadError> {
    let source = Bytes {
        input,
        at: 0,
        spans: [(0, 0); FIELDS],
    };
    let nodes = count_nodes(input, options.max_binary_bytes);
    parse_source(source, nodes, config, options, choices)
}

/// Decode a checkpoint from `source`: the header and the model's metadata, then
/// every tree, validated against `config` and the caps of `options`.
fn parse_source(
    source: impl Source,
    nodes: Option<usize>,
    config: &WalkerConfig,
    options: &LoadOptions,
    choices: &Choices,
) -> Result<ParsedModel, LoadError> {
    let mut r = Reader {
        source,
        consumed: 0,
        max_bytes: options.max_binary_bytes,
        max_array_bytes: options.max_binary_array_bytes,
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
    // The tree count is bounded by the `target_id` array, 4 bytes per tree.
    let count = r.scalar::<u64>("num_tree")?;
    let num_tree = usize::try_from(count)
        .map_err(|_| LoadError::Limit(format!("num_tree={count} exceeds the address space")))?;
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
    let num_class = r.vec(Length::Exact(1), "num_class")?;
    let leaf_shape = r.vec(Length::Exact(2), "leaf_vector_shape")?;
    let target_id = r.vec(Length::Exact(num_tree), "target_id")?;
    let class_id = r.vec(Length::Exact(num_tree), "class_id")?;
    let postprocessor = r.string(64, "postprocessor")?;
    let sigmoid_alpha = f64::from(r.scalar::<f32>("sigmoid_alpha")?);
    let _ratio_c = r.scalar::<f32>("ratio_c")?;
    let base_scores = r.vec(Length::Exact(1), "base_scores")?;
    let max_attributes = usize::try_from(options.max_attributes_bytes).unwrap_or(usize::MAX);
    validation::attributes(&r.string(max_attributes, "attributes")?, options)?;
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
    let mut decoded = Decoded::new(nodes)?;
    let mut intern = FxHashMap::default();
    let mut ctx = ParseContext {
        decoded: &mut decoded,
        list_intern: choices.bitset_intern.then_some(&mut intern),
        config,
        threshold_type,
        list: Vec::new(),
    };
    let mut b = TreeBufs::default();
    let mut scratch = OrderScratch::default();
    for i in 0..num_tree {
        read_tree(&mut r, &mut ctx, &mut b, &mut scratch)
            .map_err(|e| e.at(&format!("tree {i}")))?;
    }
    if r.source.trailing()? {
        return Err(LoadError::Unsupported(
            "trailing bytes after final tree".into(),
        ));
    }
    decoded.seal()?;
    Ok(ParsedModel {
        decoded,
        threshold_type,
        output,
    })
}

/// Input that can be read at any offset, for [`count_nodes`].
pub(super) trait At {
    /// Fill `buf` from `offset`; false if the input ends first.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> bool;
    /// The input's length in bytes.
    fn length(&self) -> Option<u64>;
}

impl At for [u8] {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> bool {
        let at = usize::try_from(offset).unwrap_or(usize::MAX);
        let bytes = self.get(at..).and_then(|rest| rest.get(..buf.len()));
        bytes.is_some_and(|bytes| {
            buf.copy_from_slice(bytes);
            true
        })
    }

    fn length(&self) -> Option<u64> {
        Some(self.len() as u64)
    }
}

#[cfg(unix)]
impl At for std::fs::File {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> bool {
        // Positioned reads leave the file's cursor where the decoder reads from.
        std::os::unix::fs::FileExt::read_exact_at(self, buf, offset).is_ok()
    }

    fn length(&self) -> Option<u64> {
        self.metadata().ok().map(|m| m.len())
    }
}

/// The little-endian integer of `n` bytes at `at`.
fn integer_at(input: &(impl At + ?Sized), at: u64, n: usize) -> Option<u64> {
    let mut le = [0u8; 8];
    input
        .read_at(at, &mut le[..n])
        .then(|| u64::from_le_bytes(le))
}

/// The entry counts of a tree's arrays that are not one per node: its leaf-vector
/// offsets, category list, data counts, hessian sums and gains.
#[derive(Clone, Copy)]
struct Lengths {
    leaf_vector: u64,
    categories: u64,
    data_count: u64,
    sum_hess: u64,
    gain: u64,
}

impl Lengths {
    /// A tree's bytes in [`read_tree`]'s field order, for `n` nodes of `value`-byte
    /// values; `None` on overflow.
    fn bytes(self, n: u64, value: u64) -> Option<u64> {
        // An array: its length, then its entries.
        let array = |len: u64, size: u64| len.checked_mul(size)?.checked_add(8);
        let per_node = [1, 4, 4, 4, 1, value, value, 1, 1, 8, 8]
            .into_iter()
            .try_fold(0u64, |sum, size| sum.checked_add(array(n, size)?))?;
        let optional = [
            array(self.leaf_vector, 8)?,
            array(self.leaf_vector, 8)?,
            array(self.categories, 4)?,
            array(self.data_count, 8)?,
            array(self.data_count, 1)?,
            array(self.sum_hess, 8)?,
            array(self.sum_hess, 1)?,
            array(self.gain, 8)?,
            array(self.gain, 1)?,
        ]
        .into_iter()
        .try_fold(per_node, u64::checked_add)?;
        // num_nodes and has_categorical_split; leaf_vector, empty; the two
        // extension counts.
        optional.checked_add(4 + 1 + 8 + 4 + 4)
    }
}

/// The node count of the checkpoint in `input`, from its trees' node counts, so
/// that the decoder allocates the nodes once; `None` if the input ends early, or
/// passes `max_bytes`, before the last tree.
///
/// It reads the header's and each tree's lengths, in [`parse_source`]'s and
/// [`read_tree`]'s field order, and skips the fields' contents. Reading all of a
/// tree's lengths takes 6 reads, each a positioned read from a file, but trees
/// usually have the optional fields of the tree before: so each tree is first read
/// with one read, of its node count and its first array's length, which must match,
/// and one more for its category list's length if it has categorical splits, and
/// these guesses hold if the last tree ends where the input does; otherwise every
/// length is read. Nothing else is validated: the decoder checks every field, and a
/// wrong count only costs it a copy of the nodes.
pub(super) fn count_nodes(input: &(impl At + ?Sized), max_bytes: u64) -> Option<usize> {
    count_trees(input, max_bytes, true).or_else(|| count_trees(input, max_bytes, false))
}

/// [`count_nodes`], guessing each tree's optional fields from the tree before with
/// `guess`.
fn count_trees(input: &(impl At + ?Sized), max_bytes: u64, guess: bool) -> Option<usize> {
    let length = input.length()?;
    let (start, trees, value) = trees_at(input)?;
    let (mut at, mut total, mut last) = (start, 0usize, None::<Lengths>);
    for _ in 0..trees {
        if at > max_bytes.min(length) {
            return None;
        }
        // num_nodes, has_categorical_split and node_type's length.
        let mut head = [0u8; 13];
        if !input.read_at(at, &mut head) {
            return None;
        }
        let n = i32::from_le_bytes(head[..4].try_into().expect("4 bytes"));
        let n = u64::try_from(n).unwrap_or(0);
        let categorical = head[4] != 0;
        let first = u64::from_le_bytes(head[5..].try_into().expect("8 bytes"));
        let guessed = last.filter(|_| guess && first == n).and_then(|l| {
            let present = |len: u64| n * u64::from(len > 0);
            let leaf_vector = present(l.leaf_vector);
            let categories = if categorical {
                integer_at(input, categories_at(at, n, value, leaf_vector)?, 8)?
            } else {
                0
            };
            Some(Lengths {
                leaf_vector,
                categories,
                data_count: present(l.data_count),
                sum_hess: present(l.sum_hess),
                gain: present(l.gain),
            })
        });
        let lengths = match guessed {
            Some(lengths) => lengths,
            None => read_lengths(input, at, n, value)?,
        };
        last = Some(lengths);
        total = total.checked_add(usize::try_from(n).ok()?)?;
        at = at.checked_add(lengths.bytes(n, value)?)?;
    }
    (at == length).then_some(total)
}

/// Where the trees of the checkpoint in `input` begin, their count and their values'
/// bytes, from the header's lengths in [`parse_source`]'s order.
fn trees_at(input: &(impl At + ?Sized)) -> Option<(u64, u64, u64)> {
    let value = if integer_at(input, 12, 1)? == 3 { 8 } else { 4 };
    let trees = integer_at(input, 14, 8)?;
    // Versions, types, num_tree, num_feature, task_type, average_tree_output and
    // num_target.
    let mut at = 12 + 2 + 8 + 4 + 1 + 1 + 4;
    // num_class, leaf_vector_shape, target_id, class_id and postprocessor; then
    // sigmoid_alpha and ratio_c; base_scores; attributes.
    for (size, after) in [(4, 0), (4, 0), (4, 0), (4, 0), (1, 8), (8, 0), (1, 0)] {
        let len = integer_at(input, at, 8)?;
        at = len.checked_mul(size)?.checked_add(at + 8 + after)?;
    }
    // num_opt_field_per_model.
    Some((at + 4, trees, value))
}

/// The offset after an array of `len` entries of `size` bytes at `at`.
fn skip(at: u64, len: u64, size: u64) -> Option<u64> {
    at.checked_add(len.checked_mul(size)?.checked_add(8)?)
}

/// The offset of the leaf-vector offsets of the tree at `at`, of `n` nodes of
/// `value`-byte values: after num_nodes and has_categorical_split; node_type, cleft,
/// cright, split_index, default_left, leaf_value, threshold, cmp and
/// category_list_right_child; and leaf_vector, empty.
fn offsets_at(at: u64, n: u64, value: u64) -> Option<u64> {
    let mut at = at.checked_add(4 + 1)?;
    for size in [1, 4, 4, 4, 1, value, value, 1, 1] {
        at = skip(at, n, size)?;
    }
    at.checked_add(8)
}

/// The offset of the category list of the tree at `at`, after its `leaf_vector`
/// leaf-vector offsets.
fn categories_at(at: u64, n: u64, value: u64, leaf_vector: u64) -> Option<u64> {
    skip(
        skip(offsets_at(at, n, value)?, leaf_vector, 8)?,
        leaf_vector,
        8,
    )
}

/// The lengths of the tree at `at`, of `n` nodes of `value`-byte values, each read.
fn read_lengths(input: &(impl At + ?Sized), at: u64, n: u64, value: u64) -> Option<Lengths> {
    let leaf_vector = integer_at(input, offsets_at(at, n, value)?, 8)?;
    let mut at = categories_at(at, n, value, leaf_vector)?;
    let categories = integer_at(input, at, 8)?;
    at = skip(skip(skip(at, categories, 4)?, n, 8)?, n, 8)?;
    // data_count, sum_hess and gain, each with its presence flags.
    let mut optional = [0; 3];
    for len in &mut optional {
        *len = integer_at(input, at, 8)?;
        at = skip(skip(at, *len, 8)?, *len, 1)?;
    }
    let [data_count, sum_hess, gain] = optional;
    Some(Lengths {
        leaf_vector,
        categories,
        data_count,
        sum_hess,
        gain,
    })
}

/// Read one tree's fields, decode its nodes and add it to the decoded forest in
/// heavy-first preorder.
fn read_tree(
    r: &mut Reader<impl Source>,
    ctx: &mut ParseContext<'_>,
    b: &mut TreeBufs,
    scratch: &mut OrderScratch,
) -> Result<(), LoadError> {
    r.source.tree();
    let count = r.scalar::<i32>("num_nodes")?;
    if count < 1 {
        return Err(LoadError::MalformedModel(format!(
            "num_nodes={count}; a tree has at least one node"
        )));
    }
    let n = count as usize;
    let has_cat = r.boolean("has_categorical_split")?;
    r.array::<i8>(Field::NodeType, Length::Exact(n), "node_type")?;
    r.array::<i32>(Field::Cleft, Length::Exact(n), "cleft")?;
    r.array::<i32>(Field::Cright, Length::Exact(n), "cright")?;
    r.array::<i32>(Field::SplitIndex, Length::Exact(n), "split_index")?;
    r.bools(Field::DefaultLeft, Length::Exact(n), "default_left")?;
    r.values(Field::LeafValue, n, ctx.threshold_type, "leaf_value")?;
    r.values(Field::Threshold, n, ctx.threshold_type, "threshold")?;
    r.array::<i8>(Field::Cmp, Length::Exact(n), "cmp")?;
    r.bools(
        Field::CatRight,
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
    let begins = r.array::<u64>(Field::LeafBegin, Length::Optional(n), "leaf_vector_begin")?;
    let ends = r.array::<u64>(Field::LeafEnd, Length::Optional(n), "leaf_vector_end")?;
    if begins != ends
        || r.field(Field::LeafBegin)
            .iter()
            .chain(r.field(Field::LeafEnd))
            .any(|&x| x != 0)
    {
        return Err(LoadError::MalformedModel(
            "leaf_vector offsets must be zero for scalar leaves".into(),
        ));
    }
    let n_categories =
        r.array::<u32>(Field::Categories, Length::Max(usize::MAX), "category_list")?;
    r.array::<u64>(Field::CatBegin, Length::Exact(n), "category_list_begin")?;
    r.array::<u64>(Field::CatEnd, Length::Exact(n), "category_list_end")?;
    let counts = r.array::<u64>(Field::DataCount, Length::Optional(n), "data_count")?;
    r.bools(
        Field::DataPresent,
        Length::Exact(counts),
        "data_count_present",
    )?;
    let hessians = r.array::<f64>(Field::SumHess, Length::Optional(n), "sum_hess")?;
    r.bools(
        Field::HessPresent,
        Length::Exact(hessians),
        "sum_hess_present",
    )?;
    let gains = r.array::<f64>(Field::Gain, Length::Optional(n), "gain")?;
    r.bools(Field::GainPresent, Length::Exact(gains), "gain_present")?;
    r.extension("num_opt_field_per_tree")?;
    r.extension("num_opt_field_per_node")?;
    let f = TreeFields::new(r, ctx.threshold_type);
    if has_cat != f.node_type.contains(&2) {
        return Err(LoadError::MalformedModel(
            "has_categorical_split disagrees with node_type".into(),
        ));
    }
    // Treelite appends each split's segment to the list, so segments never overlap.
    // Rejecting overlaps bounds the decoding work by the list's length; otherwise
    // every node could name the whole list.
    b.segments.clear();
    for (begin, end) in f.cat_begin.iter().zip(f.cat_end) {
        let (begin, end) = (u64::from_le_bytes(*begin), u64::from_le_bytes(*end));
        if begin > end || end > n_categories as u64 {
            return Err(LoadError::MalformedModel(format!(
                "invalid category segment {begin}..{end}"
            )));
        }
        if begin < end {
            b.segments.push((begin, end));
        }
    }
    if !b.segments.is_sorted() {
        b.segments.sort_unstable();
    }
    if let Some(w) = b.segments.windows(2).find(|w| w[1].0 < w[0].1) {
        return Err(LoadError::MalformedModel(format!(
            "category segments {}..{} and {}..{} overlap",
            w[0].0, w[0].1, w[1].0, w[1].1
        )));
    }
    let category_start = ctx.decoded.categories.len() as u32;
    b.temp.clear();
    b.temp
        .try_reserve(n)
        .map_err(|e| LoadError::Limit(e.to_string()))?;
    if !decode_fast(&f, ctx, &mut b.temp, &mut b.categorical)? {
        b.temp.clear();
        for i in 0..n {
            let node = decode_node(i, n, ctx, &f).map_err(|e| e.at(&format!("node {i}")))?;
            b.temp.push(node);
        }
    }
    order_tree(&b.temp, 0, category_start, ctx, scratch)
}

/// A tree's leaf values or thresholds, as stored.
#[derive(Clone, Copy)]
enum Values<'a> {
    F32(&'a [[u8; 4]]),
    F64(&'a [[u8; 8]]),
}

impl Values<'_> {
    #[inline]
    fn get(self, i: usize) -> f64 {
        match self {
            Self::F32(v) => f64::from(f32::from_le_bytes(v[i])),
            Self::F64(v) => f64::from_le_bytes(v[i]),
        }
    }
}

/// A tree's fields, as little-endian values.
struct TreeFields<'a> {
    node_type: &'a [u8],
    cleft: &'a [[u8; 4]],
    cright: &'a [[u8; 4]],
    split_index: &'a [[u8; 4]],
    default_left: &'a [u8],
    leaf_value: Values<'a>,
    threshold: Values<'a>,
    cmp: &'a [u8],
    cat_right: &'a [u8],
    categories: &'a [[u8; 4]],
    cat_begin: &'a [[u8; 8]],
    cat_end: &'a [[u8; 8]],
    data_count: &'a [[u8; 8]],
    data_present: &'a [u8],
    sum_hess: &'a [[u8; 8]],
    hess_present: &'a [u8],
}

impl<'a> TreeFields<'a> {
    fn new(r: &'a Reader<impl Source>, kind: ThresholdType) -> Self {
        fn words<const N: usize>(bytes: &[u8]) -> &[[u8; N]] {
            let (words, rest) = bytes.as_chunks::<N>();
            debug_assert!(rest.is_empty());
            words
        }
        let values = |field| {
            let bytes = r.field(field);
            if kind == ThresholdType::F64 {
                Values::F64(words(bytes))
            } else {
                Values::F32(words(bytes))
            }
        };
        Self {
            node_type: r.field(Field::NodeType),
            cleft: words(r.field(Field::Cleft)),
            cright: words(r.field(Field::Cright)),
            split_index: words(r.field(Field::SplitIndex)),
            default_left: r.field(Field::DefaultLeft),
            leaf_value: values(Field::LeafValue),
            threshold: values(Field::Threshold),
            cmp: r.field(Field::Cmp),
            cat_right: r.field(Field::CatRight),
            categories: words(r.field(Field::Categories)),
            cat_begin: words(r.field(Field::CatBegin)),
            cat_end: words(r.field(Field::CatEnd)),
            data_count: words(r.field(Field::DataCount)),
            data_present: r.field(Field::DataPresent),
            sum_hess: words(r.field(Field::SumHess)),
            hess_present: r.field(Field::HessPresent),
        }
    }
}

/// Decode a tree as [`decode_node`] would: leaves and numeric splits by the same
/// steps and selects rather than branches on their kind, then the categorical
/// splits, in order, by [`decode_node`]. Returns false if a leaf or numeric split
/// is invalid, leaving [`decode_node`] to report the tree's first error.
#[expect(
    clippy::needless_bitwise_bool,
    reason = "every node takes the same path: no branch on its kind"
)]
fn decode_fast(
    f: &TreeFields<'_>,
    ctx: &mut ParseContext<'_>,
    temp: &mut Vec<TempNode>,
    categorical: &mut Vec<u32>,
) -> Result<bool, LoadError> {
    let n = f.node_type.len();
    let n_features = ctx.config.n_features();
    let f64_model = ctx.threshold_type == ThresholdType::F64;
    let categories = f.categories.len() as u64;
    let mut ok = true;
    categorical.clear();
    for i in 0..n {
        let kind = f.node_type[i] as i8;
        let leaf = kind == 0;
        let cat = kind == 2;
        ok &= (0..=2).contains(&kind);
        if cat {
            categorical.push(i as u32);
        }
        let counted = f.data_present.get(i) == Some(&1);
        let hessian = f.hess_present.get(i) == Some(&1);
        let count = f
            .data_count
            .get(i)
            .map_or(0.0, |&c| u64::from_le_bytes(c) as f64);
        let hess = f.sum_hess.get(i).map_or(0.0, |&h| f64::from_le_bytes(h));
        let weight = if counted {
            count
        } else if hessian {
            hess
        } else {
            1.0
        };
        ok &= weight.is_finite() & (weight >= 0.0);
        let (begin, end) = (
            u64::from_le_bytes(f.cat_begin[i]),
            u64::from_le_bytes(f.cat_end[i]),
        );
        ok &= cat | ((begin == end) & (end <= categories));
        let (feature, left, right) = (
            i32::from_le_bytes(f.split_index[i]),
            i32::from_le_bytes(f.cleft[i]),
            i32::from_le_bytes(f.cright[i]),
        );
        let leaf_value = f.leaf_value.get(i);
        let threshold = f.threshold.get(i);
        let less = f.cmp[i] == 2;
        let op_ok = less | (f64_model & (f.cmp[i] == 3));
        let split_ok = (feature >= 0)
            & ((feature as usize) < n_features)
            & (left >= 0)
            & (right >= 0)
            & ((left as usize) < n)
            & ((right as usize) < n)
            & op_ok
            & !threshold.is_nan()
            & !(f64_model & less & (threshold == f64::NEG_INFINITY));
        let leaf_ok = (left == -1) & (right == -1) & (feature == -1) & leaf_value.is_finite();
        ok &= if leaf { leaf_ok } else { cat | split_ok };
        // The threshold as `validation::threshold` normalizes it: float64 `<` becomes
        // `<=` the next float down; float32 thresholds arrive rounded.
        let value = if f64_model && less {
            super::tree::next_down(threshold)
        } else {
            threshold
        };
        let flags = build_flags(
            f.default_left[i] != 0,
            false,
            false,
            ctx.config.is_varying(feature as usize),
        );
        temp.push(if leaf {
            TempNode::leaf(leaf_value, weight)
        } else {
            TempNode {
                value,
                left: left as u32,
                right: right as u32,
                feature: feature as u32,
                flags,
                weight,
            }
        });
    }
    if !ok {
        return Ok(false);
    }
    for &i in categorical.iter() {
        let i = i as usize;
        temp[i] = decode_node(i, n, ctx, f).map_err(|e| e.at(&format!("node {i}")))?;
    }
    Ok(true)
}

/// Node `i` of a tree of `n` nodes, validated: what [`decode_fast`] falls back on, and
/// the source of every error a tree can report.
fn decode_node(
    i: usize,
    n: usize,
    ctx: &mut ParseContext<'_>,
    b: &TreeFields<'_>,
) -> Result<TempNode, LoadError> {
    let kind = b.node_type[i] as i8;
    if !(0..=2).contains(&kind) {
        return Err(LoadError::Unsupported(format!("node_type={kind}")));
    }
    let weight = if b.data_present.get(i) == Some(&1) {
        u64::from_le_bytes(b.data_count[i]) as f64
    } else if b.hess_present.get(i) == Some(&1) {
        f64::from_le_bytes(b.sum_hess[i])
    } else {
        1.0
    };
    if !weight.is_finite() || weight < 0.0 {
        return Err(LoadError::MalformedModel(format!(
            "invalid node statistic {weight}"
        )));
    }
    let (begin, end) = (
        u64::from_le_bytes(b.cat_begin[i]),
        u64::from_le_bytes(b.cat_end[i]),
    );
    if begin > end || end > b.categories.len() as u64 || (kind != 2 && begin != end) {
        return Err(LoadError::MalformedModel(format!(
            "category segment {begin}..{end} is invalid"
        )));
    }
    let (split_index, cleft, cright) = (
        i32::from_le_bytes(b.split_index[i]),
        i32::from_le_bytes(b.cleft[i]),
        i32::from_le_bytes(b.cright[i]),
    );
    if kind == 0 {
        if cleft != -1 || cright != -1 || split_index != -1 {
            return Err(LoadError::MalformedModel(
                "leaf has children or split_index".into(),
            ));
        }
        return Ok(TempNode::leaf(
            validation::leaf(b.leaf_value.get(i), ctx.threshold_type)?,
            weight,
        ));
    }
    let feature = split_index;
    if feature < 0 || feature as usize >= ctx.config.n_features() {
        return Err(LoadError::MalformedModel(format!(
            "split_index={feature} out of range"
        )));
    }
    let (mut left, mut right) = (cleft, cright);
    if left < 0 || right < 0 || left as usize >= n || right as usize >= n {
        return Err(LoadError::MalformedModel(format!(
            "invalid children {left}, {right}"
        )));
    }
    let mut dl = b.default_left[i] != 0;
    let (value, inline) = if kind == 2 {
        // Normalize membership-right into membership-left, including missing routing.
        if b.cat_right[i] != 0 {
            std::mem::swap(&mut left, &mut right);
            dl = !dl;
        }
        let list = &b.categories[begin as usize..end as usize];
        encode_categories(list.iter().map(|&c| u32::from_le_bytes(c)), ctx)?
    } else {
        let op = match b.cmp[i] as i8 {
            2 => "<",
            3 => "<=",
            x => return Err(LoadError::Unsupported(format!("comparison_op={x}"))),
        };
        (
            validation::threshold(b.threshold.get(i), op, ctx.threshold_type)?,
            false,
        )
    };
    Ok(TempNode {
        value,
        left: left as u32,
        right: right as u32,
        feature: feature as u32,
        flags: build_flags(
            dl,
            kind == 2,
            inline,
            ctx.config.is_varying(feature as usize),
        ),
        weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The binary fixtures that decode, as bytes.
    fn fixtures() -> Vec<(std::path::PathBuf, Vec<u8>)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/import");
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let stem = path.file_stem().unwrap().to_str().unwrap();
            let binary = path.extension().is_some_and(|e| e == "bin");
            if binary && !stem.ends_with("_reference") && stem != "data" {
                let bytes = std::fs::read(&path).unwrap();
                out.push((path, bytes));
            }
        }
        assert!(out.len() >= 10);
        out
    }

    /// The node count from the trees' headers is the decoder's, from the bytes and
    /// from the file, and no count comes from a truncated input.
    #[test]
    fn count_nodes_matches_the_decoder() {
        let config = WalkerConfig::builder(2)
            .max_group_width(16)
            .varying([0])
            .build()
            .unwrap();
        let (options, choices) = (LoadOptions::default(), Choices::default());
        let mut decoded = 0;
        for (path, bytes) in fixtures() {
            let counted = count_nodes(bytes.as_slice(), u64::MAX);
            assert_eq!(count_trees(bytes.as_slice(), u64::MAX, false), counted);
            #[cfg(unix)]
            assert_eq!(
                count_nodes(&std::fs::File::open(&path).unwrap(), u64::MAX),
                counted
            );
            assert_eq!(count_nodes(&bytes[..bytes.len() - 1], u64::MAX), None);
            assert_eq!(count_nodes(bytes.as_slice(), 64), None);
            if let Ok(parsed) = parse_bytes(&bytes, &config, &options, &choices) {
                let nodes = parsed.decoded.trees.iter().map(|t| t.count as usize).sum();
                assert_eq!(counted, Some(nodes), "{}", path.display());
                decoded += 1;
            }
        }
        assert!(decoded >= 10);
    }
}
