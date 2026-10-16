//! The decoded forest: what the decoders hand to the build.
//!
//! Each tree's nodes are in heavy-first preorder, as the layout keeps them, each a
//! [`Packed`] word: a leaf is its value and a split names its entry in the forest's
//! split table, so a split's threshold or categories, feature and flags are stored
//! once however many nodes share them. The trees stay in the input's order, and
//! each node sits in the 16-byte layout node it becomes, which the build rewrites in
//! place once the bins and predicate IDs are known: the decoded nodes are the
//! layout's own, and the model is never held twice.

use crate::exact::Extent;
use crate::load::layout::LayoutNode;
use crate::model::{FLAG_CATEGORICAL, FLAG_DEFAULT_LEFT, FLAG_INLINE_CAT, FLAG_VARYING};

/// The tag of a packed split: a quiet NaN's exponent and top fraction bit. Leaf values
/// are finite, so they never carry it.
const SPLIT_TAG: u64 = 0x7FF8 << 48;
const HEAVY_SHIFT: u32 = 32;
const HEAVY_LEFT: u64 = 1 << 48;

/// A decoded node: a leaf's value, or a split's ID in the [`Splits`] table with its
/// heavy child's training share ([`crate::model::Node::heavy`]) and side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Packed(u64);

impl Packed {
    /// A leaf of finite `value`.
    #[inline]
    pub(crate) fn leaf(value: f64) -> Self {
        debug_assert!(value.is_finite());
        Self(value.to_bits())
    }

    #[inline]
    pub(crate) const fn split(id: u32, heavy: u16, heavy_left: bool) -> Self {
        Self(
            SPLIT_TAG
                | (heavy as u64) << HEAVY_SHIFT
                | if heavy_left { HEAVY_LEFT } else { 0 }
                | id as u64,
        )
    }

    /// The node a provisional layout node holds.
    #[inline]
    pub(crate) const fn of(node: LayoutNode) -> Self {
        Self(node.w0)
    }

    /// The provisional layout node that holds it, until the build rewrites it.
    #[inline]
    pub(crate) const fn node(self) -> LayoutNode {
        LayoutNode { w0: self.0, w1: 0 }
    }

    #[inline]
    pub(crate) const fn is_leaf(self) -> bool {
        self.0 & SPLIT_TAG != SPLIT_TAG
    }

    /// A leaf's value.
    #[inline]
    pub(crate) const fn value(self) -> f64 {
        f64::from_bits(self.0)
    }

    /// A split's ID.
    #[inline]
    pub(crate) const fn id(self) -> u32 {
        self.0 as u32
    }

    #[inline]
    pub(crate) const fn heavy(self) -> u16 {
        (self.0 >> HEAVY_SHIFT) as u16
    }

    #[inline]
    pub(crate) const fn heavy_left(self) -> bool {
        self.0 & HEAVY_LEFT != 0
    }

    /// The probability that a row goes heavy, as [`crate::model::Node::heavy_probability`].
    #[inline]
    pub(crate) fn heavy_probability(self) -> f64 {
        f64::from(self.heavy()) / f64::from(u16::MAX)
    }
}

/// A split as decoded: the parsed node's value (threshold, inline word or list
/// offset), feature and flags, without the heavy side.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Split {
    pub(crate) value: f64,
    pub(crate) feature: u32,
    pub(crate) flags: u8,
    /// For a categorical split, the heavy sides its nodes take: bit 0 left, bit 1
    /// right, so that the build makes chains of the heavy sets the nodes test only.
    pub(crate) sides: u8,
}

impl Split {
    #[inline]
    pub(crate) const fn default_left(&self) -> bool {
        self.flags & FLAG_DEFAULT_LEFT != 0
    }

    #[inline]
    pub(crate) const fn is_categorical(&self) -> bool {
        self.flags & FLAG_CATEGORICAL != 0
    }

    #[inline]
    pub(crate) const fn inline_cat(&self) -> bool {
        self.flags & FLAG_INLINE_CAT != 0
    }

    #[inline]
    pub(crate) const fn is_varying(&self) -> bool {
        self.flags & FLAG_VARYING != 0
    }

    /// A categorical split whose categories are a list in the pool.
    #[inline]
    pub(crate) const fn listed(&self) -> bool {
        self.is_categorical() && !self.inline_cat()
    }
}

/// The forest's distinct splits, by (value bits, feature, flags).
///
/// The lookup is an open-addressing table of split IDs, at most half full, probed
/// linearly from a multiplicative hash of the key: models have few distinct splits
/// against their nodes (thousands against millions for binned GBDTs), so the table
/// stays in cache and a lookup is a hash, a load and a compare.
#[derive(Default)]
pub(crate) struct Splits {
    pub(crate) all: Vec<Split>,
    /// `EMPTY` or an index in `all`.
    table: Vec<u32>,
}

const EMPTY: u32 = u32::MAX;

impl Splits {
    /// The ID of a split, added if new. IDs fit 32 bits: there are fewer splits than
    /// nodes.
    #[inline]
    pub(crate) fn intern(&mut self, value: f64, feature: u32, flags: u8) -> u32 {
        if 2 * (self.all.len() + 1) > self.table.len() {
            self.grow();
        }
        let mask = self.table.len() - 1;
        let bits = value.to_bits();
        let mut at = Self::slot(bits, feature, flags, mask);
        loop {
            let id = self.table[at];
            if id == EMPTY {
                let id = self.all.len() as u32;
                self.table[at] = id;
                self.all.push(Split {
                    value,
                    feature,
                    flags,
                    sides: 0,
                });
                return id;
            }
            let s = &self.all[id as usize];
            if s.value.to_bits() == bits && s.feature == feature && s.flags == flags {
                return id;
            }
            at = (at + 1) & mask;
        }
    }

    /// The first slot of a key: the top bits of a multiplicative hash, which depend
    /// on every bit of the key. (Low bits would not: a product's low bits see only
    /// the factors' low bits, where round thresholds are all zero.)
    #[inline]
    fn slot(bits: u64, feature: u32, flags: u8, mask: usize) -> usize {
        const K: u64 = 0x9E37_79B9_7F4A_7C15;
        let key = u64::from(feature) << 8 | u64::from(flags);
        let h = (bits ^ key.rotate_left(40)).wrapping_mul(K);
        let h = (h ^ h >> 29).wrapping_mul(K);
        (h >> (u64::BITS - mask.count_ones())) as usize
    }

    /// Double the table, at least 64 slots.
    #[cold]
    fn grow(&mut self) {
        let len = (2 * self.table.len()).max(64);
        self.table = vec![EMPTY; len];
        let mask = len - 1;
        for (id, s) in self.all.iter().enumerate() {
            let mut at = Self::slot(s.value.to_bits(), s.feature, s.flags, mask);
            while self.table[at] != EMPTY {
                at = (at + 1) & mask;
            }
            self.table[at] = id as u32;
        }
    }

    /// Record that a node of categorical split `id` has its heavy child on the left
    /// with `heavy_left`, else on the right.
    #[inline]
    pub(crate) fn heavy_side(&mut self, id: u32, heavy_left: bool) {
        self.all[id as usize].sides |= 1 << u8::from(!heavy_left);
    }

    /// Drop the lookup table once every tree is decoded.
    pub(crate) fn seal(&mut self) {
        self.table = Vec::new();
    }
}

/// A decoded tree, in input order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DecodedTree {
    /// Its root's index in the node store, which the layout keeps.
    pub(crate) start: u32,
    pub(crate) count: u32,
    /// Where the tree's lists begin in the decoded pool.
    pub(crate) category_start: u32,
    pub(crate) leaves: u32,
    pub(crate) varying: u32,
    /// The tree's [`Decoded::firsts`].
    firsts: (u32, u32),
}

/// A tree's decoded nodes, in heavy-first preorder: each a [`Packed`] in word 0 of
/// the layout node it becomes.
#[derive(Clone, Copy)]
pub(crate) struct TreeNodes<'a>(&'a [LayoutNode]);

impl<'a> TreeNodes<'a> {
    /// The decoded nodes `nodes`.
    #[inline]
    pub(crate) const fn new(nodes: &'a [LayoutNode]) -> Self {
        Self(nodes)
    }

    #[inline]
    pub(crate) const fn len(self) -> usize {
        self.0.len()
    }

    /// Node `i`, if the tree has it.
    #[inline]
    pub(crate) fn get(self, i: usize) -> Option<Packed> {
        self.0.get(i).map(|n| Packed(n.w0))
    }

    /// Node `i`.
    #[inline]
    pub(crate) fn at(self, i: usize) -> Packed {
        Packed(self.0[i].w0)
    }

    #[inline]
    pub(crate) fn iter(self) -> impl ExactSizeIterator<Item = Packed> + 'a {
        self.0.iter().map(|n| Packed(n.w0))
    }
}

/// Nodes in the store's first segment and in its largest, when the node count is not
/// known in advance: segments are allocated whole, so the store grows without
/// reallocating, and a tree never spans two. Each segment doubles the last, so
/// small models take little and large ones few segments.
const FIRST_SEGMENT: usize = 1 << 14;
const SEGMENT_NODES: usize = 1 << 20;

/// The decoded nodes, in segments of whole trees, each node in the 16 bytes of the
/// layout node it becomes.
///
/// When the decoder knows the forest's node count before the first tree, the first
/// segment holds exactly every node and the layout's dummy, so the nodes are
/// decoded where the layout keeps them and allocated once. Otherwise
/// [`Self::seal`] copies the segments into one such vector. A large vector is
/// never grown or shrunk by `realloc`: on macOS, the memory of a large block that
/// `realloc` resized stays with the process once the block is freed, so a process
/// that loaded models repeatedly grew by about a node array per load (124 MB per
/// load of a 7.75M-node forest, measured with the vector grown by doubling and
/// with it reserved large and then shrunk).
#[derive(Default)]
struct Store {
    segments: Vec<Vec<LayoutNode>>,
    /// The nodes in all segments.
    len: usize,
    /// The node count the decoder announced.
    expected: Option<usize>,
}

impl Store {
    /// A store for `nodes` nodes, if they are known and node indices can hold them.
    fn new(nodes: Option<usize>) -> Result<Self, crate::LoadError> {
        let nodes = nodes.filter(|&n| n < u32::MAX as usize);
        let mut store = Self {
            expected: nodes,
            ..Self::default()
        };
        if let Some(n) = nodes {
            store.segment(n.saturating_add(1))?;
        }
        Ok(store)
    }

    /// A new segment of `n` nodes.
    fn segment(&mut self, n: usize) -> Result<(), crate::LoadError> {
        let mut segment = Vec::new();
        segment
            .try_reserve_exact(n)
            .map_err(|e| crate::LoadError::Limit(e.to_string()))?;
        self.segments.push(segment);
        Ok(())
    }

    /// Append a tree's `nodes`, in a segment with room for them.
    fn append(&mut self, nodes: &[Packed]) -> Result<(), crate::LoadError> {
        let n = nodes.len();
        let fits = self
            .segments
            .last()
            .is_some_and(|s| s.capacity() - s.len() >= n);
        if !fits {
            let last = self
                .segments
                .last()
                .map_or(FIRST_SEGMENT / 2, Vec::capacity);
            self.segment(n.max((2 * last).clamp(FIRST_SEGMENT, SEGMENT_NODES)))?;
        }
        let segment = self.segments.last_mut().expect("a segment");
        segment.extend(nodes.iter().map(|node| node.node()));
        self.len += n;
        Ok(())
    }

    /// The nodes in one vector with room for exactly one more, the layout's dummy.
    fn seal(&mut self) -> Result<(), crate::LoadError> {
        let exact = self.segments.len() == 1 && self.segments[0].capacity() == self.len + 1;
        // A decoder that counts the nodes in advance counts the trees it decodes.
        debug_assert!(self.expected.is_none_or(|n| n == self.len));
        if exact {
            return Ok(());
        }
        let mut all = Vec::new();
        all.try_reserve_exact(self.len + 1)
            .map_err(|e| crate::LoadError::Limit(e.to_string()))?;
        // Each segment freed once copied.
        for segment in std::mem::take(&mut self.segments) {
            all.extend_from_slice(&segment);
        }
        self.segments.push(all);
        Ok(())
    }

    /// Every node, once sealed.
    fn all(&self) -> &[LayoutNode] {
        debug_assert!(self.segments.len() <= 1);
        self.segments.first().map_or(&[], Vec::as_slice)
    }
}

/// The decoded forest.
#[derive(Default)]
pub(crate) struct Decoded {
    /// Every tree's nodes, in input order, which the layout keeps: the build
    /// rewrites them in place.
    store: Store,
    pub(crate) trees: Vec<DecodedTree>,
    pub(crate) splits: Splits,
    /// The categorical splits' lists: each its length, then its sorted categories.
    pub(crate) categories: Vec<u32>,
    /// Per tree, its varying and listed splits' IDs in the order they first appear.
    firsts: Vec<u32>,
    /// The tree that last listed each split in `firsts`, plus one.
    stamps: Vec<u32>,
    pub(crate) extent: Extent,
}

impl Decoded {
    /// A forest to decode, of `nodes` nodes if the decoder knows them in advance, so
    /// that they are allocated once, where the layout keeps them. A wrong count
    /// costs a copy, not an error.
    pub(crate) fn new(nodes: Option<usize>) -> Result<Self, crate::LoadError> {
        Ok(Self {
            store: Store::new(nodes)?,
            ..Self::default()
        })
    }

    /// The nodes decoded so far.
    #[inline]
    pub(crate) const fn node_count(&self) -> usize {
        self.store.len
    }

    /// The split table, to intern the next tree's splits in, and the leaves' extent.
    pub(crate) const fn parts(&mut self) -> (&mut Splits, &mut Extent) {
        (&mut self.splits, &mut self.extent)
    }

    /// Append a tree's `nodes`, in heavy-first preorder, of which `leaves` leaves
    /// and `varying` varying splits.
    pub(crate) fn push_tree(
        &mut self,
        nodes: &[Packed],
        category_start: u32,
        leaves: u32,
        varying: u32,
    ) -> Result<(), crate::LoadError> {
        let (at, n) = (self.store.len, nodes.len());
        self.store.append(nodes)?;
        // Each node's split, or for a leaf one past the last, whose stamp is the
        // current tree's: no branch on the node's kind.
        let t = self.trees.len() as u32 + 1;
        let none = self.splits.all.len();
        self.stamps.resize(none + 1, 0);
        self.stamps[none] = t;
        let start = self.firsts.len() as u32;
        for node in nodes {
            let id = if node.is_leaf() {
                none
            } else {
                node.id() as usize
            };
            if self.stamps[id] != t {
                self.stamps[id] = t;
                let split = &self.splits.all[id];
                if split.is_varying() || split.listed() {
                    self.firsts.push(id as u32);
                }
            }
        }
        self.trees.push(DecodedTree {
            start: at as u32,
            count: n as u32,
            category_start,
            leaves,
            varying,
            firsts: (start, self.firsts.len() as u32),
        });
        Ok(())
    }

    /// Drop what only decoding needs, and put the nodes in one vector with room for
    /// the layout's dummy.
    pub(crate) fn seal(&mut self) -> Result<(), crate::LoadError> {
        self.splits.seal();
        self.stamps = Vec::new();
        self.store.seal()
    }

    /// Tree `t`'s nodes, in heavy-first preorder.
    #[inline]
    pub(crate) fn nodes(&self, t: usize) -> TreeNodes<'_> {
        let tree = &self.trees[t];
        let at = tree.start as usize;
        TreeNodes(&self.store.all()[at..at + tree.count as usize])
    }

    /// Tree `t`'s varying and listed splits, in the order they first appear in it.
    #[inline]
    pub(crate) fn firsts(&self, t: usize) -> &[u32] {
        let (a, b) = self.trees[t].firsts;
        &self.firsts[a as usize..b as usize]
    }

    /// The split of a split node.
    #[inline]
    pub(crate) fn split(&self, node: Packed) -> &Split {
        &self.splits.all[node.id() as usize]
    }

    /// The node store, for the layout to rewrite in place.
    pub(crate) fn take_nodes(&mut self) -> Vec<LayoutNode> {
        std::mem::take(&mut self.store)
            .segments
            .pop()
            .unwrap_or_default()
    }
}
