//! Forest data structures: Node, Tree, VaryingPredicate, Forest.
//!
//! # Memory layout
//!
//! All data lives in three contiguous allocations on [`Forest`]:
//! - `nodes: Vec<Node>` — all trees' nodes in one buffer (~52 MB for 3.3M nodes).
//! - `bitsets: Vec<u8>` — all categorical split bitsets, packed (~25 MB after dedup).
//! - `trees: Vec<Tree>` — per-tree metadata, 12 bytes each (~11 KB for 938 trees).
//!
//! **Why one pool instead of per-tree Vecs?** A naive parser creates 938 separate
//! `Vec<Node>` allocations scattered across the heap. The allocator places them
//! wherever there's space, causing fragmentation. During prediction, jumping between
//! trees means jumping between unrelated heap regions — destroying spatial locality.
//! A single contiguous pool means tree N's nodes are adjacent to tree N+1's nodes
//! in physical memory, and the HW prefetcher can stride across them.
//!
//! # Node layout (16 bytes, fall-through)
//!
//! ```text
//! Offset  Size  Field             Purpose
//! 0       8     value: f64        Overloaded: threshold / bitset offset / leaf output
//! 8       2     skip: i16         Light-child index (-1 = leaf)
//! 10      2     varying_pred_id   Index into Forest.varying_predicates (u16::MAX = constant/leaf)
//! 12      2     feature: u16      Split feature index
//! 14      1     flags: u8         Packed: default_left|categorical|inline_cat|heavy_is_left|varying_type
//! 15      1     cat_n_words       Bitset pool word count (0 for non-cat or inline)
//! ```
//!
//! Exactly 16 bytes = 4 nodes per cache line. The `#[repr(C)]` layout guarantee
//! ensures no padding. The `left` child pointer is eliminated entirely — it's
//! always `idx + 1` (the fall-through convention).

use std::path::Path;

use crate::config::{ParseConfig, WalkerConfig};
use crate::parser;
use crate::{LoadError, ModelFormat};
use std::io::{Cursor, Read};

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A 16-byte decision tree node.
///
/// The `value` field is overloaded by node type:
/// - **Numerical split**: f64 threshold. `val <= value` → go left.
/// - **Categorical split (inline)**: u32 bitset word stored as f64.
/// - **Categorical split (pool)**: byte offset into `Forest.bitsets` stored as f64.
/// - **Leaf**: f64 scalar leaf output.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub value: f64,
    /// Light-child tree-local index, or `-1` for leaf nodes.
    /// The heavy child is always at `idx + 1` (implicit fall-through).
    pub skip: i16,
    /// Index into `Forest.varying_predicates` for varying nodes, or `u16::MAX` for constant/leaf.
    pub varying_pred_id: u16,
    pub feature: u16,
    /// Bit 0: `default_left` — go left when feature is NaN.
    /// Bit 1: `categorical` — use bitset test instead of threshold.
    /// Bit 2: `inline_cat` — bitset word is in `value`, not the pool.
    /// Bit 3: `heavy_is_left` — the fall-through child is the left split side.
    /// Bits 4-5: varying type — `00`=constant, `01`=`mono_inc`, `10`=`mono_dec`, `11`=non-mono varying.
    pub flags: u8,
    /// Number of u32 words in the bitset pool (0 for non-cat or inline cat).
    pub cat_n_words: u8,
}

const _: () = assert!(std::mem::size_of::<Node>() == 16);

// Flag bit constants for Node.flags bitfield.
pub(crate) const FLAG_DEFAULT_LEFT: u8 = 0x01;
pub(crate) const FLAG_CATEGORICAL: u8 = 0x02;
pub(crate) const FLAG_INLINE_CAT: u8 = 0x04;
pub(crate) const FLAG_HEAVY_IS_LEFT: u8 = 0x08;
pub(crate) const FLAG_VARYING_TYPE_MASK: u8 = 0x30;
pub(crate) const FLAG_VARYING_TYPE_SHIFT: u8 = 4;
/// Bit 6: set for constant internal nodes (not leaf, not varying).
/// Enables single-instruction constant walk loop condition.
pub(crate) const FLAG_WALKABLE: u8 = 0x40;

impl Node {
    #[inline]
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.skip == -1
    }

    #[inline]
    #[must_use]
    pub const fn default_left(&self) -> bool {
        self.flags & FLAG_DEFAULT_LEFT != 0
    }

    #[inline]
    #[must_use]
    pub const fn is_categorical(&self) -> bool {
        self.flags & FLAG_CATEGORICAL != 0
    }

    #[inline]
    #[must_use]
    pub const fn inline_cat(&self) -> bool {
        self.flags & FLAG_INLINE_CAT != 0
    }

    #[inline]
    #[must_use]
    pub const fn heavy_is_left(&self) -> bool {
        self.flags & FLAG_HEAVY_IS_LEFT != 0
    }

    /// Varying type: `0`=constant, `1`=`mono_inc`, `2`=`mono_dec`, `3`=non-mono varying.
    #[inline]
    #[must_use]
    pub const fn varying_type(&self) -> u8 {
        (self.flags >> FLAG_VARYING_TYPE_SHIFT) & 3
    }

    /// True if this is a constant-feature split (walk through without row partition).
    #[inline]
    #[must_use]
    pub const fn is_constant(&self) -> bool {
        self.flags & FLAG_VARYING_TYPE_MASK == 0
    }

    /// True if this node should be walked through in the constant walk:
    /// it's a constant-feature internal node (not leaf, not varying).
    /// Single bit test — compiles to one `tbnz` instruction.
    #[inline]
    #[must_use]
    pub const fn is_walkable(&self) -> bool {
        self.flags & FLAG_WALKABLE != 0
    }
}

pub const SPLIT_CONSTANT: u8 = 0;
pub const SPLIT_MONO_INC: u8 = 1;
pub const SPLIT_MONO_DEC: u8 = 2;
pub const SPLIT_NON_MONO: u8 = 3;

// ---------------------------------------------------------------------------
// Tree / Forest
// ---------------------------------------------------------------------------

/// Canonicalized varying split predicate used for per-observation mask precomputation.
///
/// Each variant captures exactly the fields needed to evaluate the split for any
/// row's feature value. Predicates are deduplicated across all trees: if tree 5
/// and tree 200 both split on "feature 14 ≤ 3.5 with default_left=true", they
/// share the same `VaryingPredicate` and the same `varying_pred_id`. This means the
/// precompute pass evaluates ~5K unique predicates instead of ~50K total varying nodes.
#[derive(Clone, Copy)]
pub(crate) enum VaryingPredicate {
    Num {
        feature: u16,
        threshold: f64,
        default_left: bool,
    },
    CatInline {
        feature: u16,
        default_left: bool,
        word: u32,
    },
    CatPool {
        feature: u16,
        default_left: bool,
        offset: u32,
        n_words: u8,
    },
}

// Manual Hash/Eq using f64::to_bits() so VaryingPredicate can serve as its own HashMap key,
// eliminating the need for a separate key type.
impl std::hash::Hash for VaryingPredicate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match *self {
            Self::Num {
                feature,
                threshold,
                default_left,
            } => {
                feature.hash(state);
                threshold.to_bits().hash(state);
                default_left.hash(state);
            }
            Self::CatInline {
                feature,
                default_left,
                word,
            } => {
                feature.hash(state);
                default_left.hash(state);
                word.hash(state);
            }
            Self::CatPool {
                feature,
                default_left,
                offset,
                n_words,
            } => {
                feature.hash(state);
                default_left.hash(state);
                offset.hash(state);
                n_words.hash(state);
            }
        }
    }
}

impl PartialEq for VaryingPredicate {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (
                Self::Num {
                    feature: f1,
                    threshold: t1,
                    default_left: d1,
                },
                Self::Num {
                    feature: f2,
                    threshold: t2,
                    default_left: d2,
                },
            ) => f1 == f2 && t1.to_bits() == t2.to_bits() && d1 == d2,
            (
                Self::CatInline {
                    feature: f1,
                    default_left: d1,
                    word: w1,
                },
                Self::CatInline {
                    feature: f2,
                    default_left: d2,
                    word: w2,
                },
            ) => f1 == f2 && d1 == d2 && w1 == w2,
            (
                Self::CatPool {
                    feature: f1,
                    default_left: d1,
                    offset: o1,
                    n_words: n1,
                },
                Self::CatPool {
                    feature: f2,
                    default_left: d2,
                    offset: o2,
                    n_words: n2,
                },
            ) => f1 == f2 && d1 == d2 && o1 == o2 && n1 == n2,
            _ => false,
        }
    }
}

impl Eq for VaryingPredicate {}

impl VaryingPredicate {
    /// Feature index this predicate splits on.
    #[inline]
    pub(crate) const fn feature(&self) -> u16 {
        match *self {
            Self::Num { feature, .. }
            | Self::CatInline { feature, .. }
            | Self::CatPool { feature, .. } => feature,
        }
    }

    /// Evaluate whether `val` goes left for this predicate.
    #[inline]
    pub(crate) fn goes_left<const F32: bool>(&self, val: f64, bitsets: &[u8]) -> bool {
        match *self {
            Self::Num {
                threshold,
                default_left,
                ..
            } => {
                if val.is_nan() {
                    default_left
                } else {
                    threshold_go_left::<F32>(val, threshold)
                }
            }
            Self::CatInline {
                default_left, word, ..
            } => {
                if val.is_nan() {
                    default_left
                } else {
                    let val = if F32 { f64::from(val as f32) } else { val };
                    let cat = val as i32;
                    if val < 0.0 || !(0..32).contains(&cat) {
                        return false;
                    }
                    (word >> cat as u32) & 1 != 0
                }
            }
            Self::CatPool {
                default_left,
                offset,
                n_words,
                ..
            } => {
                if val.is_nan() {
                    default_left
                } else {
                    let val = if F32 { f64::from(val as f32) } else { val };
                    let cat = val as i32;
                    if val < 0.0 || cat < 0 {
                        return false;
                    }
                    let word_idx = (cat / 32) as usize;
                    if word_idx >= n_words as usize {
                        return false;
                    }
                    let byte_offset = offset as usize + word_idx * 4;
                    let word = unsafe {
                        bitsets
                            .as_ptr()
                            .add(byte_offset)
                            .cast::<u32>()
                            .read_unaligned()
                    };
                    (word >> (cat as u32 % 32)) & 1 != 0
                }
            }
        }
    }
}

/// Per-tree metadata: offsets into the global node and bitset pools.
#[derive(Clone, Copy, Debug)]
pub struct Tree {
    /// Index of this tree's root in `Forest.nodes`.
    pub node_start: u32,
    /// Total nodes (internal + leaves) in this tree.
    pub node_count: u32,
    /// Byte offset where this tree's bitset region begins in `Forest.bitsets`.
    ///
    /// After repacking with interned bitsets, trees sharing identical bitsets may
    /// point into earlier regions. The hot path uses per-node absolute offsets
    /// (stored in `Node.value`), not this field. Retained for diagnostics and
    /// memory accounting.
    pub bitset_start: u32,
}

/// The full model: all trees sharing contiguous node and bitset pools.
/// Controls numerical threshold comparison precision.
/// Monomorphized at parse time — zero-cost dispatch on the hot path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThresholdType {
    /// LightGBM: thresholds are native f64, compare `val <= threshold` in f64.
    F64,
    /// XGBoost: thresholds are f32-promoted-to-f64. Compare `(val as f32) < (threshold as f32)`
    /// to match the original framework's split decisions.
    F32,
}

/// Numerical "go left" test, const-generic over threshold precision.
/// The `const F32: bool` parameter is resolved at compile time — zero branches.
///
/// - `F32=false` (LightGBM): `val <= threshold` (thresholds adjusted by `next_down` at parse time)
/// - `F32=true` (XGBoost): `(val as f32) < (threshold as f32)` (f32 comparison, thresholds as-is)
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn threshold_go_left<const F32: bool>(val: f64, threshold: f64) -> bool {
    if F32 {
        (val as f32) < (threshold as f32)
    } else {
        val <= threshold
    }
}

/// Per-feature index into the sorted `varying_predicates` vec.
///
/// Constructed at parse time by [`parser::build_feature_ranges`] after predicates
/// are sorted by `(kind, feature, threshold)`. Each varying feature gets a
/// contiguous range of numerical predicates followed by categorical ones.
/// The sweep kernel in [`predict::precompute_varying_masks`] iterates
/// `feature_ranges` to process one feature at a time.
#[derive(Clone, Debug)]
pub(crate) struct FeatureRange {
    pub feature: u16,
    /// First numerical predicate index (inclusive).
    pub num_start: u32,
    /// One past last numerical predicate index.
    pub num_end: u32,
    /// First categorical predicate index (inclusive).
    pub cat_start: u32,
    /// One past last categorical predicate index.
    pub cat_end: u32,
}

/// A group of trees sharing the same first K constant heavy-path splits.
/// At predict time, the shared splits are evaluated once; all trees in the group
/// then resume `partial_eval` at the continuation index.
pub(crate) struct PrefixGroup {
    /// Node pool offset of the representative tree (used to read shared splits).
    pub node_base: u32,
    /// Tree indices (into `Forest.trees`) in this group.
    pub trees: Vec<u32>,
}

pub struct Forest {
    pub(crate) output: parser::Output,
    pub(crate) compiled_config: WalkerConfig,
    pub(crate) trees: Vec<Tree>,
    pub config: WalkerConfig,
    pub(crate) nodes: Vec<Node>,
    pub(crate) bitsets: Vec<u8>,
    pub(crate) varying_predicates: Vec<VaryingPredicate>,
    /// Per-feature ranges into `varying_predicates` (sorted by feature, then kind).
    pub(crate) feature_ranges: Vec<FeatureRange>,
    pub(crate) threshold_type: ThresholdType,
    /// Groups of trees sharing constant heavy-path prefixes. Empty = disabled.
    pub(crate) prefix_groups: Vec<PrefixGroup>,
    /// Depth of the shared prefix (K). 0 when prefix grouping is disabled.
    pub(crate) prefix_depth: usize,
    /// Fixed-point scale for exact leaf sums ([`crate::exact::scale`]); `None`
    /// when the leaf values cannot be summed exactly and prediction adds in `f64`.
    pub(crate) fixed_scale: Option<i32>,
    /// Reusable scratch space for partial evaluation. Lazily initialized on first
    /// `predict` call. Private — callers never touch this.
    pub(crate) workspace: Option<crate::predict::Workspace>,
}

impl std::fmt::Debug for Forest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Forest")
            .field("trees", &self.trees.len())
            .field("nodes", &self.nodes.len())
            .field("bitsets_bytes", &self.bitsets.len())
            .field("varying_predicates", &self.varying_predicates.len())
            .field("threshold_type", &self.threshold_type)
            .finish_non_exhaustive()
    }
}

impl Forest {
    /// Compatibility wrapper; panics on I/O, malformed or unsupported models.
    /// Prefer [`Self::try_load`].
    pub fn load(model_path: impl AsRef<Path>, config_path: impl AsRef<Path>) -> Self {
        Self::try_load(model_path, config_path).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Compatibility wrapper; panics on load errors. Prefer [`Self::try_load_with_config`].
    pub fn load_with_config(
        model_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        parse_config: &ParseConfig,
    ) -> Self {
        Self::try_load_with_config(model_path, config_path, parse_config)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Load a supported Treelite `.bin` or `.json` model and grouping configuration.
    pub fn try_load(
        model_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
    ) -> Result<Self, LoadError> {
        Self::try_load_with_config(model_path, config_path, &ParseConfig::default())
    }

    /// Load model/configuration files with explicit parse-time optimization options.
    pub fn try_load_with_config(
        model_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        parse_config: &ParseConfig,
    ) -> Result<Self, LoadError> {
        let config = WalkerConfig::try_from_file(config_path)?;
        let format = ModelFormat::from_path(model_path.as_ref())?;
        Self::from_reader(
            std::fs::File::open(model_path)?,
            format,
            config,
            parse_config,
        )
    }

    /// Load a model from memory with explicit format and validated grouping metadata.
    pub fn from_bytes(
        bytes: &[u8],
        format: ModelFormat,
        config: WalkerConfig,
        parse_config: &ParseConfig,
    ) -> Result<Self, LoadError> {
        Self::from_reader(Cursor::new(bytes), format, config, parse_config)
    }

    /// Load a model from a reader. Binary decoding streams with reusable tree buffers;
    /// JSON decoding materializes a bounded document. Structural validation precedes
    /// optimized layout, and unsupported prediction semantics return an error.
    pub fn from_reader(
        reader: impl Read,
        format: ModelFormat,
        config: WalkerConfig,
        parse_config: &ParseConfig,
    ) -> Result<Self, LoadError> {
        let parser::ParsedModel {
            trees,
            mut nodes,
            bitsets,
            threshold_type,
            output,
        } = parser::parse_reader(reader, format, &config, parse_config)?;
        let mut varying_predicates = if parse_config.disable_predicate_dedup {
            parser::build_varying_predicates_no_dedup(&mut nodes)?
        } else {
            parser::build_varying_predicates(&mut nodes)?
        };
        parser::sort_varying_predicates(&mut varying_predicates, &mut nodes);
        let feature_ranges = parser::build_feature_ranges(&varying_predicates);
        let (prefix_groups, prefix_depth) = if parse_config.prefix_depth > 0 {
            let (groups, _ungrouped) =
                parser::build_prefix_groups(&trees, &nodes, &config, parse_config.prefix_depth);
            (groups, parse_config.prefix_depth)
        } else {
            (Vec::new(), 0)
        };

        let fixed_scale = crate::exact::scale(
            nodes.iter().filter(|n| n.is_leaf()).map(|n| n.value),
            trees.len(),
        );

        #[cfg(target_os = "linux")]
        unsafe {
            libc::malloc_trim(0);
        }

        Ok(Self {
            compiled_config: config.clone(),
            output,
            trees,
            config,
            nodes,
            bitsets,
            varying_predicates,
            feature_ranges,
            threshold_type,
            prefix_groups,
            prefix_depth,
            fixed_scale,
            workspace: None,
        })
    }

    #[inline]
    #[must_use]
    pub fn tree_nodes(&self, tree: &Tree) -> &[Node] {
        &self.nodes[tree.node_start as usize..(tree.node_start + tree.node_count) as usize]
    }

    #[inline]
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    #[inline]
    #[must_use]
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    #[inline]
    #[must_use]
    pub const fn bitset_bytes(&self) -> usize {
        self.bitsets.len()
    }

    #[inline]
    #[must_use]
    pub const fn threshold_type(&self) -> ThresholdType {
        self.threshold_type
    }

    /// Whether `predict` sums leaf values exactly and rounds once, so predictions are
    /// the `f64` nearest to the true sum whatever the tree order. False only if a leaf
    /// is not finite or the leaf exponents span too wide a range for a 126-bit fixed
    /// point; prediction then adds in `f64` in tree order.
    #[inline]
    #[must_use]
    pub const fn exact_sums(&self) -> bool {
        self.fixed_scale.is_some()
    }

    /// Test if `category` belongs to a categorical node's split set.
    ///
    /// For inline nodes the bitset word is in `node.value`. For pool nodes
    /// the packed u32 words start at `node.value` (byte offset) in `self.bitsets`.
    ///
    /// Out-of-range categories return false (non-membership). The parser normalizes
    /// membership-right splits by swapping children and missing routing. The last
    /// argument is retained for source compatibility; only NaN uses missing routing.
    #[inline]
    #[must_use]
    pub const fn cat_test(&self, node: &Node, category: i32, _default_left: bool) -> bool {
        if category < 0 {
            return false;
        }
        if node.inline_cat() {
            if category >= 32 {
                return false;
            }
            (node.value as u32 >> category) & 1 != 0
        } else {
            let word_idx = (category / 32) as usize;
            if word_idx >= node.cat_n_words as usize {
                return false;
            }
            let offset = node.value as usize + word_idx * 4;
            // SAFETY: offset + 4 is within bitsets — guaranteed by the parser which
            // writes exactly cat_n_words × 4 bytes at the stored offset.
            let word = unsafe {
                self.bitsets
                    .as_ptr()
                    .add(offset)
                    .cast::<u32>()
                    .read_unaligned()
            };
            (word >> (category as u32 % 32)) & 1 != 0
        }
    }
}
