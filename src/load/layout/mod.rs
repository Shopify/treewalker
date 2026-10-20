//! The prediction layout, built at load from the decoded forest: integer bins per
//! slot, 16-byte pre-oriented nodes and the leaf-value table.
//!
//! Every split is one test, `bin[slot] <= k`: heavy when it holds. A slot is a chain
//! of nested sets over its feature's values, and a row's bin in it is its rank, the
//! first set of the chain that holds the row's value; a split names a slot and the
//! position k of its heavy set in the chain.
//!
//! # Nodes
//!
//! The trees follow each other in input order, where they were decoded, each tree's
//! nodes in heavy-first preorder, as the research API's parsed nodes are numbered
//! too, so tree starts and prefix groups index both; one dummy node follows the last
//! tree. Tree order is the order of the tree starts, so a chunk's trees need not be
//! adjacent: a group reaches each node at most once, through one path per tree. The
//! leaf values are in tree order. A node is two words:
//!
//! ```text
//! Word 0:
//!   0-31   a split's k, pre-oriented; a leaf's leaf-value index
//!   32-33  unused
//!   34-63  bin slot
//! Word 1:
//!   0-31   absolute light child; for a leaf, its leaf-value index
//!   32     leaf
//!   33-62  mask field: 0 for the all-rows mask, which leaves read, and for
//!          constant splits; a split on varying predicate i reads its heavy side's
//!          mask, 2i + 1 when that is the left side and 2i + 2 when it is the right
//!   63     unused
//! ```
//!
//! What bounds each field: node indices and leaf indices are 32-bit, so a model has
//! fewer than 2^32 - 1 nodes with the dummy; k is below the node count or a chain's
//! length; slots below 2^30 allow 2^28 features with numeric splits; mask fields
//! below 2^30 allow 2^29 - 1 varying predicates, each with a mask per side. Each
//! field is extracted with one shift, mask or 32-bit compare, as narrower fields
//! were: a 32-bit compare on word 0 ignores the slot.
//!
//! The heavy child is the next node. A varying node carries the same constant
//! encoding, for single rows and prefixes. The precompute writes both sides' masks
//! of every varying predicate, so resolve splits a stop's rows with the mask of its
//! heavy side and never flips it. A leaf reads the all-rows mask as its heavy side,
//! so splitting its rows changes nothing. A walk that stops at varying splits stops
//! where word 1 has a set bit in 32-62: the leaf flag or a mask field. That mask,
//! like the leaf flag alone, is one AArch64 logical immediate; a mask of word 1's top
//! bits would become a shift and a compare.
//!
//! # Numeric slots
//!
//! Slot 0 holds `u32::MAX` for the dummy, which reads k = 0 and loops on itself;
//! other slots belong to the features that splits use, in feature order. A feature
//! with numeric splits takes four: its bin with missing values sent left, sent
//! right, and both inverted (`u32::MAX - bin`). A value's bin is the number of the
//! feature's sorted distinct thresholds below it (at or below it after rounding for
//! float32 models), so `x <= t_k` (`x < t_k` for float32) holds exactly when
//! `bin <= k`: the thresholds' sublevel sets are a chain. A split whose heavy child
//! is on the right reads an inverted slot against `u32::MAX - 1 - k`.
//!
//! # Categorical slots
//!
//! Each categorical feature has a codebook: the sorted distinct categories its
//! splits name. Its domain is their codes and two more: code 0 is no category, for
//! values in no list (negative, fractional past a listed category, out of range or
//! unlisted), code 1 is missing, and codebook entry j is code j + 2. Codes depend
//! only on the codebook's size, so any `u32` category works. A value's code is its
//! category's rank in the codebook, read from a table of 65 codes when every
//! category is below 64, and otherwise found by a binary search.
//!
//! A split sends left its listed codes and, when its default is left, missing: its
//! list. Its heavy set is the list or, by its heavy side, the complement. The
//! feature's distinct heavy sets are covered by the fewest chains ([`chains`]),
//! each a slot after the feature's numeric ones, and a split tests the position of
//! its heavy set in its chain. Missing and unknown values are ordinary codes, so no
//! slot depends on a default direction.
//!
//! No category is in no list, so a chain's listed sets precede its complements,
//! and no category's rank in it, the chain's default, is the number of listed sets.
//! Another code's rank differs from the default only in chains with a set that
//! lists the code, so the rank tables hold, per feature, the defaults and per code
//! its exceptions, `(slot, rank)` where its rank differs: at most one per listed
//! code of each heavy set, so the tables grow with the model's lists, not with its
//! codes times its chains. A bins buffer holds the defaults and, after the slots,
//! each categorical feature's held code, whose exceptions its chain slots hold; a
//! fill restores the held code's exceptions and writes the new code's. The tables
//! take at most [`crate::LoadOptions::max_rank_table_bytes`] in all.

mod build;
mod chains;
mod masks;
#[cfg(test)]
mod tests;

use std::hint::select_unpredictable;
use std::ops::Range;

use crate::LoadError;
use crate::exact::WideLeaf;
use crate::model::category;

pub(crate) use build::{Built, Input};
pub(crate) use masks::{ChainPoint, MaskChain, MaskFeature, MaskPlan};

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

/// A 16-byte node; see the module documentation for its fields.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayoutNode {
    pub(crate) w0: u64,
    pub(crate) w1: u64,
}

const _: () = assert!(size_of::<LayoutNode>() == 16 && align_of::<LayoutNode>() == 16);

/// Word 0: the payload and the slot.
pub(crate) const PAYLOAD_MASK: u64 = 0xFFFF_FFFF;
pub(crate) const SLOT_SHIFT: u32 = 34;
const SLOT_BITS: u32 = 30;
/// Word 1: the light child, the leaf flag and the mask field.
pub(crate) const CHILD_MASK: u64 = 0xFFFF_FFFF;
pub(crate) const FLAG_LEAF: u64 = 1 << 32;
pub(crate) const MASK_SHIFT: u32 = 33;
pub(crate) const MASK_BITS: u32 = 30;
pub(crate) const MASK_FIELD: u64 = (1 << MASK_BITS) - 1;
/// The bits of word 1 that stop a walk at varying splits: the leaf flag and the
/// mask field. Bit 63 stays clear: a test of word 1's top 32 bits would compile to
/// a shift and a compare rather than one AArch64 logical immediate.
pub(crate) const STOPS_VARYING: u64 = FLAG_LEAF | MASK_FIELD << MASK_SHIFT;
/// Varying predicates a model may have: predicate i takes the fields 2i + 1 and
/// 2i + 2, both below 2^30.
pub(crate) const MAX_PREDICATES: usize = ((1 << MASK_BITS) - 1) / 2;
/// Bin slots a model may have.
const MAX_SLOTS: usize = 1 << SLOT_BITS;

const _: () = assert!(SLOT_SHIFT + SLOT_BITS == u64::BITS);
const _: () = assert!(FLAG_LEAF == 1 << 32 && MASK_SHIFT == 33);
const _: () = assert!(MASK_SHIFT + MASK_BITS == 63);
const _: () = assert!(2 * MAX_PREDICATES as u64 <= MASK_FIELD);

impl LayoutNode {
    /// The dummy at `index`: a numeric split that reads `u32::MAX` from slot 0 against
    /// k = 0 and so always takes its light child, itself.
    const fn dummy(index: u32) -> Self {
        Self {
            w0: 0,
            w1: index as u64,
        }
    }

    #[inline]
    pub(crate) const fn child(self) -> u32 {
        (self.w1 & CHILD_MASK) as u32
    }

    /// The mask field: 0 for leaves and constant splits, else the index of the
    /// heavy side's mask, `2i + 1` (left) or `2i + 2` (right) for predicate `i`.
    #[cfg(test)]
    #[inline]
    pub(crate) const fn mask_field(self) -> usize {
        ((self.w1 >> MASK_SHIFT) & MASK_FIELD) as usize
    }

    #[inline]
    pub(crate) const fn slot(self) -> usize {
        (self.w0 >> SLOT_SHIFT) as usize
    }

    #[cfg(test)]
    #[inline]
    pub(crate) const fn is_leaf(self) -> bool {
        self.w1 & FLAG_LEAF != 0
    }

    #[cfg(test)]
    #[inline]
    pub(crate) const fn is_varying(self) -> bool {
        !self.is_leaf() && self.mask_field() != 0
    }

    /// Word 0's low 32 bits: a split's k or a leaf's leaf-value index.
    #[inline]
    pub(crate) const fn payload(self) -> u32 {
        (self.w0 & PAYLOAD_MASK) as u32
    }

    /// Whether a split goes to its heavy child, reading `bins`. Varying nodes answer
    /// through their constant encoding.
    #[inline]
    pub(crate) fn goes_heavy(self, bins: &[u32]) -> bool {
        bins[self.slot()] <= self.payload()
    }
}

// ---------------------------------------------------------------------------
// Bins
// ---------------------------------------------------------------------------

/// Categorical codes: no category, missing, then the codebook's entries.
pub(crate) const NO_CATEGORY: u32 = 0;
pub(crate) const MISSING: u32 = 1;
pub(crate) const FIRST_CATEGORY: u32 = 2;
/// Categories below this take their codes from a table, as a 64-bit set of them
/// would fit one machine word; the table costs 260 bytes per feature.
const SMALL_CATEGORIES: u32 = u64::BITS;

/// How a feature's input value becomes a categorical code: its codebook, at
/// `start..start + len` in [`BinMap::books`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct CategoryCodes {
    start: u32,
    len: u32,
    /// When every category is below 64, the offset in the books of a table of 65
    /// codes: category c's at c, and no category's at 64 for the others. A lookup
    /// then takes a compare and a load instead of a search of the codebook.
    small: Option<u32>,
}

impl CategoryCodes {
    /// The feature's codebook in `books`.
    #[inline]
    fn book(self, books: &[u32]) -> &[u32] {
        &books[self.start as usize..(self.start + self.len) as usize]
    }

    /// The feature's codes: its categories', no category and missing.
    pub(crate) const fn domain(self) -> usize {
        self.len as usize + FIRST_CATEGORY as usize
    }

    /// The codes of the codebook `book`, which is appended to `books` with its
    /// table if it has one.
    fn new(books: &mut Vec<u32>, book: &[u32]) -> Result<Self, LoadError> {
        let at = |books: &Vec<u32>| {
            u32::try_from(books.len())
                .map_err(|_| LoadError::Limit("codebooks exceed u32 entries".into()))
        };
        let start = at(books)?;
        books.extend_from_slice(book);
        let small = if book.iter().all(|&c| c < SMALL_CATEGORIES) {
            let table = at(books)?;
            let mut codes = [NO_CATEGORY; SMALL_CATEGORIES as usize + 1];
            for (j, &c) in book.iter().enumerate() {
                codes[c as usize] = j as u32 + FIRST_CATEGORY;
            }
            books.extend_from_slice(&codes);
            Some(table)
        } else {
            None
        };
        Ok(Self {
            start,
            len: book.len() as u32,
            small,
        })
    }

    /// The code of category `c`: its codebook entry's, or no category.
    #[inline]
    fn of(self, books: &[u32], c: u32) -> u32 {
        if let Some(table) = self.small {
            return books[table as usize + c.min(SMALL_CATEGORIES) as usize];
        }
        let book = self.book(books);
        let (mut lo, mut len) = (0, book.len());
        while len > 0 {
            let half = len / 2;
            // SAFETY: the search keeps lo + len <= book.len(), and half < len.
            let below = unsafe { *book.get_unchecked(lo + half) } < c;
            lo = select_unpredictable(below, lo + half + 1, lo);
            len = select_unpredictable(below, len - half - 1, half);
        }
        let found = lo < book.len() && book[lo] == c;
        select_unpredictable(found, lo as u32 + FIRST_CATEGORY, NO_CATEGORY)
    }

    /// The code of `x`: missing, or its category's. Float32 models round first;
    /// fractional categories truncate.
    #[inline]
    pub(crate) fn code<const F32: bool>(self, books: &[u32], x: f64) -> u32 {
        if x.is_nan() {
            return MISSING;
        }
        category(x, F32).map_or(NO_CATEGORY, |c| self.of(books, c))
    }
}

/// A categorical feature's slots: how its values become codes, and where each code's
/// ranks in its chains differ from no category's, its exceptions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CategoryBins {
    pub(crate) codes: CategoryCodes,
    /// Its first chain's slot.
    pub(crate) slot: usize,
    pub(crate) chains: usize,
    /// Its codes' exception bounds, `domain + 1` entries from here in
    /// [`BinMap::offsets`].
    pub(crate) offsets: usize,
    /// Its index among the categorical features: a bins buffer holds, at
    /// `n_slots + held`, the code whose exceptions its slots hold.
    pub(crate) held: usize,
}

/// The bins one feature fills.
#[derive(Clone, Debug)]
struct FeatureBins {
    feature: usize,
    /// Its thresholds in [`BinMap::thresholds`] and their first slot; empty without
    /// numeric splits.
    thresholds: Range<usize>,
    slot: usize,
    /// Its categorical slots, if it has categorical splits.
    categories: Option<CategoryBins>,
}

/// What a row's features become: which slots each feature fills, and how.
#[derive(Clone, Debug)]
pub(crate) struct BinMap {
    f32: bool,
    n_features: usize,
    n_slots: usize,
    thresholds: Vec<f64>,
    /// Every categorical feature's codebook.
    pub(crate) books: Vec<u32>,
    /// Every slot's default bin: no category's ranks in the chain slots, `u32::MAX`
    /// in the dummy's and 0 in the numeric slots.
    pub(crate) defaults: Vec<u32>,
    /// Every categorical feature's codes' exceptions, `(slot, rank)` in increasing
    /// slot order, code `e`'s at `offsets[e]..offsets[e + 1]` from its feature's
    /// offsets.
    pub(crate) offsets: Vec<u32>,
    pub(crate) exceptions: Vec<[u32; 2]>,
    /// The categorical features.
    n_categorical: usize,
    /// The constant and the varying features with splits, each in feature order.
    constant: Vec<FeatureBins>,
    varying: Vec<FeatureBins>,
}

/// The dummy's slot.
const DUMMY_SLOT: usize = 0;

impl BinMap {
    /// Slots in a bins buffer.
    #[inline]
    pub(crate) const fn n_slots(&self) -> usize {
        self.n_slots
    }

    /// A bins buffer: the defaults, and after the slots each categorical feature's
    /// held code, no category, which has no exceptions.
    pub(crate) fn new_bins(&self) -> Vec<u32> {
        let mut bins = self.defaults.clone();
        bins.resize(self.n_slots + self.n_categorical, NO_CATEGORY);
        bins
    }

    /// Code `code`'s exceptions in the chains of `c`.
    #[inline]
    pub(crate) fn exceptions(&self, c: CategoryBins, code: u32) -> &[[u32; 2]] {
        let at = c.offsets + code as usize;
        &self.exceptions[self.offsets[at] as usize..self.offsets[at + 1] as usize]
    }

    /// Bytes of the rank tables: the defaults of the chain slots, the codes'
    /// exception bounds and the exceptions.
    pub(crate) fn rank_table_bytes(&self) -> usize {
        size_of::<u32>() * (self.chain_slots() + self.offsets.len())
            + size_of_val(self.exceptions.as_slice())
    }

    /// Code `code`'s ranks in the chains of `c`.
    #[cfg(test)]
    pub(crate) fn ranks_of(&self, c: CategoryBins, code: u32) -> Vec<u32> {
        let mut ranks = self.defaults[c.slot..c.slot + c.chains].to_vec();
        for &[s, r] in self.exceptions(c, code) {
            ranks[s as usize - c.slot] = r;
        }
        ranks
    }

    /// The categorical features' chain slots.
    pub(crate) fn chain_slots(&self) -> usize {
        self.constant
            .iter()
            .chain(&self.varying)
            .filter_map(|b| b.categories)
            .map(|c| c.chains)
            .sum()
    }

    /// A feature's bins, if it has splits.
    #[cfg(test)]
    fn feature(&self, feature: usize) -> Option<&FeatureBins> {
        [&self.constant, &self.varying]
            .into_iter()
            .find_map(|list| {
                list.binary_search_by_key(&feature, |b| b.feature)
                    .ok()
                    .map(|i| &list[i])
            })
    }

    /// The constant features that splits read, whose bins come from a group's first
    /// row.
    #[cfg(debug_assertions)]
    pub(crate) fn constant_features(&self) -> impl Iterator<Item = usize> + '_ {
        self.constant.iter().map(|b| b.feature)
    }

    /// A varying feature's categorical slots, if it has categorical splits.
    pub(crate) fn varying_categories(&self, feature: usize) -> Option<CategoryBins> {
        let i = self
            .varying
            .binary_search_by_key(&feature, |b| b.feature)
            .ok()?;
        self.varying[i].categories
    }

    /// A feature's sorted distinct thresholds.
    #[cfg(test)]
    pub(crate) fn feature_thresholds(&self, feature: usize) -> &[f64] {
        self.feature(feature)
            .map_or(&[], |b| &self.thresholds[b.thresholds.clone()])
    }

    /// A feature's categorical slots.
    #[cfg(test)]
    pub(crate) fn feature_categories(&self, feature: usize) -> Option<CategoryBins> {
        self.feature(feature).and_then(|b| b.categories)
    }

    /// A feature's codebook; empty without categorical splits.
    #[cfg(test)]
    pub(crate) fn feature_book(&self, feature: usize) -> &[u32] {
        self.feature_categories(feature)
            .map_or(&[], |c| c.codes.book(&self.books))
    }

    /// A feature's first numeric slot, if it has numeric splits.
    #[cfg(test)]
    pub(crate) fn numeric_slot(&self, feature: usize) -> Option<usize> {
        self.feature(feature)
            .filter(|b| !b.thresholds.is_empty())
            .map(|b| b.slot)
    }

    /// [`Self::fill_with`] at the model's precision.
    #[cfg(test)]
    pub(crate) fn fill(&self, row: &[f64], bins: &mut [u32], all: bool) {
        if self.f32 {
            self.fill_with::<true>(row, bins, all);
        } else {
            self.fill_with::<false>(row, bins, all);
        }
    }

    /// Fill `bins`, a buffer of [`Self::new_bins`], from `row`, for a float32 model
    /// with `F32`, else a float64 model: the constant features' slots, and with `all`
    /// the varying features' too, as a single row needs. Slots of features without
    /// splits are left as they are. A categorical feature's chain slots hold the
    /// defaults but for its held code's exceptions, so it restores those and writes
    /// its code's: work in the exceptions, not the chains.
    #[inline]
    pub(crate) fn fill_with<const F32: bool>(&self, row: &[f64], bins: &mut [u32], all: bool) {
        debug_assert_eq!(F32, self.f32);
        assert!(row.len() >= self.n_features && bins.len() >= self.n_slots + self.n_categorical);
        let varying: &[FeatureBins] = if all { &self.varying } else { &[] };
        for b in self.constant.iter().chain(varying) {
            let x = row[b.feature];
            let thresholds = &self.thresholds[b.thresholds.clone()];
            if !thresholds.is_empty() {
                let (left, right) = if x.is_nan() {
                    (0, thresholds.len() as u32)
                } else {
                    let bin = count_below::<F32>(thresholds, x) as u32;
                    (bin, bin)
                };
                let s = b.slot;
                bins[s] = left;
                bins[s + 1] = right;
                bins[s + 2] = u32::MAX - left;
                bins[s + 3] = u32::MAX - right;
            }
            if let Some(c) = b.categories {
                let code = c.codes.code::<F32>(&self.books, x);
                let held = self.n_slots + c.held;
                for &[s, _] in self.exceptions(c, bins[held]) {
                    bins[s as usize] = self.defaults[s as usize];
                }
                for &[s, rank] in self.exceptions(c, code) {
                    bins[s as usize] = rank;
                }
                bins[held] = code;
            }
        }
        bins[DUMMY_SLOT] = u32::MAX;
    }
}

/// The number of `thresholds` below `x` (at or below `x` rounded to float32).
#[inline]
fn count_below<const F32: bool>(thresholds: &[f64], x: f64) -> usize {
    let (mut lo, mut len) = (0, thresholds.len());
    let x = if F32 { f64::from(x as f32) } else { x };
    while len > 0 {
        let half = len / 2;
        let t = thresholds[lo + half];
        let below = if F32 { t <= x } else { t < x };
        lo = select_unpredictable(below, lo + half + 1, lo);
        len = select_unpredictable(below, len - half - 1, half);
    }
    lo
}

// ---------------------------------------------------------------------------
// Leaf values
// ---------------------------------------------------------------------------

/// Leaf values by leaf-value index.
#[derive(Clone, Debug)]
pub(crate) enum LeafValues {
    /// Values times 2^e for the model's [`crate::exact::Plan::Fixed`] scale e.
    Fixed(Vec<i128>),
    /// The values in the digits of the model's [`crate::exact::Plan::Wide`] accumulator.
    Wide(Vec<WideLeaf>),
}

impl LeafValues {
    #[cfg(test)]
    pub(crate) const fn len(&self) -> usize {
        match self {
            Self::Fixed(values) => values.len(),
            Self::Wide(values) => values.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// The kernel's view of a model: its nodes, the bins their splits read and the leaf
/// values.
#[derive(Clone, Debug)]
pub(crate) struct Layout {
    /// Every tree's nodes, in input order, then the dummy.
    pub(crate) nodes: Vec<LayoutNode>,
    pub(crate) bins: BinMap,
    pub(crate) leaves: LeafValues,
    /// Share of internal nodes that split on varying features.
    #[cfg_attr(
        not(any(test, feature = "research")),
        expect(dead_code, reason = "for the research module's model statistics")
    )]
    pub(crate) varying_share: f64,
    /// The categorical features' distinct heavy sets, which their chains hold.
    #[cfg_attr(
        not(any(test, feature = "research")),
        expect(dead_code, reason = "for the research module's model statistics")
    )]
    pub(crate) chain_sets: usize,
}

/// What a tree's passes need: an item per leaf and its arena slots, at most one
/// split per varying node and one fewer than its leaves.
///
/// A chunk's nodes and leaf values are not part of its working set: each node is
/// reached by at most one item of a group, since a tree has one path to it, and each
/// leaf value is read once, so they stream through once per group, whereas every
/// pass rewrites and rereads the scratch and rereads the masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Need {
    pub(crate) leaves: usize,
    pub(crate) slots: usize,
}

impl Layout {
    /// The dummy node's index.
    #[inline]
    pub(crate) const fn dummy(&self) -> u32 {
        (self.nodes.len() - 1) as u32
    }

    /// The node after node `index` for a row with `bins`.
    #[cfg(test)]
    pub(crate) fn next(&self, index: u32, bins: &[u32]) -> u32 {
        let node = self.nodes[index as usize];
        if node.goes_heavy(bins) {
            index + 1
        } else {
            node.child()
        }
    }

    /// A layout without trees or features, for tests that build a model by hand.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self {
            nodes: vec![LayoutNode::dummy(0)],
            bins: BinMap {
                f32: false,
                n_features: 0,
                n_slots: 1,
                thresholds: Vec::new(),
                books: Vec::new(),
                defaults: vec![u32::MAX],
                offsets: Vec::new(),
                exceptions: Vec::new(),
                n_categorical: 0,
                constant: Vec::new(),
                varying: Vec::new(),
            },
            leaves: LeafValues::Wide(Vec::new()),
            varying_share: 0.0,
            chain_sets: 0,
        }
    }
}
