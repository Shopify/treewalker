//! The loaded model ([`Model`]) and the types it is described in: the parsed node
//! ([`Node`]) and tree ([`Tree`]), the varying predicates ([`VaryingPredicate`]), the
//! prefix groups ([`PrefixGroup`]) and the output transform ([`Output`]).
//!
//! # Memory
//!
//! Loading decodes each tree into packed nodes ([`crate::load::parse::decoded`]) and
//! writes the prediction layout from them ([`crate::load::layout`]). Production keeps
//! the layout, the categorical splits' lists, the trees and the varying predicates.
//! With the `research` feature, the layout's build also writes each node as a parsed
//! [`Node`], kept for introspection and the full walk.
//!
//! # Node layout (24 bytes, fall-through)
//!
//! ```text
//! Offset  Size  Field             Purpose
//! 0       8     value: f64        Overloaded: threshold / category word or list / leaf output
//! 8       4     skip: u32         Light-child index (u32::MAX = leaf)
//! 12      4     varying_pred_id   Index into the varying predicates (u32::MAX = constant/leaf)
//! 16      4     feature: u32      Split feature index
//! 20      1     flags: u8         Packed: default_left|categorical|inline_cat|heavy_is_left|varying
//! 22      2     heavy: u16        The heavy child's share of the training weight, in 1/65535ths
//! ```
//!
//! The heavy child is always at `idx + 1` (the fall-through convention), so the node
//! stores only the light child. A categorical split whose categories are all below
//! 32 holds their bits in `value`; any other holds the offset of its list in the
//! model's category lists: the list's length, then its categories, sorted and
//! distinct.

use std::ops::Range;

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// The sentinel of [`Node::skip`] for a leaf and of [`Node::varying_pred_id`] for a
/// leaf or a constant split.
pub(crate) const NONE: u32 = u32::MAX;

/// A decision tree node, as parsed: what the research API's introspection and full
/// walk read. Production predicts from the layout instead.
///
/// The `value` field is overloaded by node type:
/// - **Numerical split**: f64 threshold. `val <= value` → go left.
/// - **Categorical split (inline)**: the u32 bitset of categories 0-31, as f64.
/// - **Categorical split (listed)**: the offset of its list in the model's category
///   lists, as f64.
/// - **Leaf**: f64 scalar leaf output.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Node {
    /// The threshold, category word or list offset, or leaf value; see above.
    pub value: f64,
    /// Light-child tree-local index, or `u32::MAX` for leaf nodes.
    /// The heavy child is always at `idx + 1` (implicit fall-through).
    pub skip: u32,
    /// Index into the model's varying predicates for varying nodes, or `u32::MAX` for
    /// constant splits and leaves.
    pub varying_pred_id: u32,
    /// The split's feature; 0 for a leaf.
    pub feature: u32,
    /// Bit 0: `default_left` — go left when feature is NaN.
    /// Bit 1: `categorical` — use a category test instead of a threshold.
    /// Bit 2: `inline_cat` — the categories' bits are in `value`, not a list.
    /// Bit 3: `heavy_is_left` — the fall-through child is the left split side.
    /// Bit 4: `varying` — the split feature varies within a group.
    pub flags: u8,
    /// The heavy child's share of the split's training weight (Treelite's
    /// `data_count`, else `sum_hess`; one half without either), in 1/65535ths: the
    /// probability that a row like the training data goes heavy. 0 for a leaf.
    pub heavy: u16,
}

const _: () = assert!(std::mem::size_of::<Node>() == 24);

// The bits of [`Node::flags`], which the decoded splits' flags share.
pub(crate) const FLAG_DEFAULT_LEFT: u8 = 0x01;
pub(crate) const FLAG_CATEGORICAL: u8 = 0x02;
pub(crate) const FLAG_INLINE_CAT: u8 = 0x04;
pub(crate) const FLAG_HEAVY_IS_LEFT: u8 = 0x08;
pub(crate) const FLAG_VARYING: u8 = 0x10;

#[cfg_attr(
    not(feature = "research"),
    expect(dead_code, reason = "read through the research introspection")
)]
impl Node {
    /// Whether the node is a leaf.
    #[inline]
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.skip == NONE
    }

    /// Whether a missing value goes left.
    #[inline]
    #[must_use]
    pub const fn default_left(&self) -> bool {
        self.flags & FLAG_DEFAULT_LEFT != 0
    }

    /// Whether the split tests categories rather than a threshold.
    #[inline]
    #[must_use]
    pub const fn is_categorical(&self) -> bool {
        self.flags & FLAG_CATEGORICAL != 0
    }

    /// Whether a categorical split's categories are the bits of `value` rather than
    /// a list.
    #[inline]
    #[must_use]
    pub const fn inline_cat(&self) -> bool {
        self.flags & FLAG_INLINE_CAT != 0
    }

    /// Whether the heavy child, at `idx + 1`, is the split's left side.
    #[inline]
    #[must_use]
    pub const fn heavy_is_left(&self) -> bool {
        self.flags & FLAG_HEAVY_IS_LEFT != 0
    }

    /// The probability that a row goes to the heavy child, from [`Self::heavy`].
    #[inline]
    #[must_use]
    pub fn heavy_probability(&self) -> f64 {
        f64::from(self.heavy) / f64::from(u16::MAX)
    }

    /// True for a leaf or a split on a constant feature.
    #[inline]
    #[must_use]
    pub const fn is_constant(&self) -> bool {
        self.flags & FLAG_VARYING == 0
    }
}

// ---------------------------------------------------------------------------
// Varying predicates
// ---------------------------------------------------------------------------

/// A varying split's test, whose row masks the precompute builds for each group.
///
/// Each variant holds exactly what evaluating the split on a row's value needs.
/// Identical splits share one predicate across the forest (unless the research
/// ablation turns that off): if tree 5 and tree 200 both split on "feature 14 ≤ 3.5,
/// missing left", they share one `VaryingPredicate` and one ID, so the precompute
/// evaluates each distinct predicate once per row however many nodes test it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum VaryingPredicate {
    Num {
        feature: u32,
        threshold: Threshold,
        default_left: bool,
    },
    CatInline {
        feature: u32,
        default_left: bool,
        word: u32,
    },
    /// Categories listed at `offset` in the model's category lists; content-identical lists
    /// share an offset when lists are interned.
    CatPool {
        feature: u32,
        default_left: bool,
        offset: u32,
    },
}

/// A split threshold compared and hashed by its bits, so `-0.0` and `0.0` are distinct
/// predicates and [`VaryingPredicate`] can derive `Eq` and `Hash` to key the dedup map.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Threshold(pub f64);

impl PartialEq for Threshold {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for Threshold {}

impl std::hash::Hash for Threshold {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl VaryingPredicate {
    /// The predicate of a split node.
    #[cfg(any(test, feature = "research"))]
    #[inline]
    pub(crate) const fn of(node: &Node) -> Self {
        Self::new(node.feature, node.flags, node.value)
    }

    /// The predicate of a decoded split.
    #[inline]
    pub(crate) const fn of_split(split: &crate::load::parse::decoded::Split) -> Self {
        Self::new(split.feature, split.flags, split.value)
    }

    /// The predicate on `feature` with a parsed node's `flags` and `value`.
    #[inline]
    const fn new(feature: u32, flags: u8, value: f64) -> Self {
        let default_left = flags & FLAG_DEFAULT_LEFT != 0;
        if flags & FLAG_CATEGORICAL == 0 {
            Self::Num {
                feature,
                threshold: Threshold(value),
                default_left,
            }
        } else if flags & FLAG_INLINE_CAT != 0 {
            Self::CatInline {
                feature,
                default_left,
                word: value as u32,
            }
        } else {
            Self::CatPool {
                feature,
                default_left,
                offset: value as u32,
            }
        }
    }

    /// Feature index this predicate splits on.
    #[inline]
    pub(crate) const fn feature(&self) -> u32 {
        match *self {
            Self::Num { feature, .. }
            | Self::CatInline { feature, .. }
            | Self::CatPool { feature, .. } => feature,
        }
    }

    /// Evaluate whether `val` goes left for this predicate, with the parsed
    /// category lists `categories`.
    #[inline]
    pub(crate) fn goes_left<const F32: bool>(&self, val: f64, categories: &[u32]) -> bool {
        match *self {
            Self::Num {
                threshold: Threshold(threshold),
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
                    category(val, F32).is_some_and(|c| c < 32 && (word >> c) & 1 != 0)
                }
            }
            Self::CatPool {
                default_left,
                offset,
                ..
            } => {
                if val.is_nan() {
                    default_left
                } else {
                    category(val, F32).is_some_and(|c| {
                        category_list(categories, offset).binary_search(&c).is_ok()
                    })
                }
            }
        }
    }
}

/// The category of a non-missing input value: float32 models round it first, and a
/// fractional value truncates. `None` for values outside `0 ..= u32::MAX`, which are in
/// no category.
#[inline]
pub(crate) fn category(val: f64, f32: bool) -> Option<u32> {
    if f32 {
        // The rounded value itself: 2^32 is exact in f32, so the range test and the
        // truncation are those of the value widened back.
        let v = val as f32;
        return (0.0..4_294_967_296.0).contains(&v).then_some(v as u32);
    }
    (0.0..4_294_967_296.0).contains(&val).then_some(val as u32)
}

/// The sorted categories of the list at `offset` in the category lists `categories`.
#[inline]
pub(crate) fn category_list(categories: &[u32], offset: u32) -> &[u32] {
    let at = offset as usize;
    &categories[at + 1..=at + categories[at] as usize]
}

// ---------------------------------------------------------------------------
// Trees and the model
// ---------------------------------------------------------------------------

/// Per-tree metadata: offsets into the parsed nodes and the category lists.
#[derive(Clone, Copy, Debug)]
pub struct Tree {
    /// Index of this tree's root in the parsed nodes and in the layout.
    #[cfg_attr(
        not(feature = "research"),
        expect(dead_code, reason = "read through the research introspection")
    )]
    pub node_start: u32,
    /// Total nodes (internal + leaves) in this tree.
    #[cfg_attr(
        not(feature = "research"),
        expect(dead_code, reason = "read through the research introspection")
    )]
    pub node_count: u32,
    /// Offset where this tree's category lists begin in the category lists.
    ///
    /// After repacking with interned lists, trees sharing identical lists may point
    /// into earlier regions. Nodes hold absolute offsets (in `Node.value`), not this
    /// field. Retained for diagnostics and memory accounting.
    #[cfg_attr(
        not(feature = "research"),
        expect(dead_code, reason = "read through the research introspection")
    )]
    pub category_start: u32,
}

/// The precision numeric splits compare in, from the model's threshold type. The
/// kernel is monomorphized on it, so the hot path never branches on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThresholdType {
    /// LightGBM: thresholds are native f64, compare `val <= threshold` in f64.
    F64,
    /// XGBoost: thresholds are f32-promoted-to-f64. Compare `(val as f32) < (threshold as f32)`
    /// to match the original framework's split decisions.
    F32,
}

/// Whether a numeric split sends `val` left, at the precision of `F32`:
///
/// - `F32 = false` (LightGBM): `val <= threshold`, with `<` thresholds made `<=` by
///   `next_down` at load;
/// - `F32 = true` (XGBoost): `(val as f32) < (threshold as f32)`, thresholds as they are.
#[inline]
pub(crate) fn threshold_go_left<const F32: bool>(val: f64, threshold: f64) -> bool {
    if F32 {
        (val as f32) < (threshold as f32)
    } else {
        val <= threshold
    }
}

/// A varying feature's predicates, as ranges of the sorted predicates.
///
/// Built by [`crate::load::predicates::build_feature_ranges`] from the predicates
/// sorted by feature, kind and threshold: each varying feature has a contiguous
/// range of numeric predicates followed by categorical ones, from which the load
/// builds the precompute's [`crate::load::layout::MaskPlan`]. In declaration order
/// (`repr(C)`), which keeps the load's code as it was with five `u32` fields.
#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct FeatureRange {
    pub feature: u32,
    /// Its numeric predicates, in increasing threshold order.
    pub num: Range<u32>,
    /// Its categorical predicates.
    pub cat: Range<u32>,
}

/// A run of consecutive trees sharing their first `depth` heavy-path constant
/// splits, chosen by [`crate::rules::prefix_groups`]. At predict time the shared
/// splits are evaluated once per group, and every tree of the group starts its walk
/// where they lead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrefixGroup {
    /// The first tree's root, whose splits the group evaluates.
    pub node_base: u32,
    pub depth: u32,
    /// The trees `first..end`.
    pub first: u32,
    pub end: u32,
}

/// The link function applied to a row's margin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Postprocessor {
    Identity,
    /// `1 / (1 + exp(-alpha × margin))`, with the model's positive `alpha`.
    Sigmoid(f64),
}

/// How a row's tree sum becomes its output: `sum / divisor + base_score`, rounded
/// at each step, then the link function.
#[derive(Clone, Copy, Debug)]
pub struct Output {
    pub base_score: f64,
    /// The tree count for averaging models (random forests), else 1.
    pub divisor: f64,
    pub postprocessor: Postprocessor,
}
