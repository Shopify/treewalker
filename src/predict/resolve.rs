//! The resolve pass, and how items carry their rows ([`Carrier`]).
//!
//! Resolve reads a walk pass's stops in item order, turns leaves into leaf entries and
//! splits each varying stop's rows with its heavy side's mask into the next pass's
//! items. The precompute writes both sides' masks of each predicate, so the split
//! needs no flip. By value, leaves read the all-rows mask as their heavy side, so one
//! branch-free loop treats both kinds alike. An arena carrier's split reads and writes
//! its slots, which a leaf must not pay:
//!
//! - A fixed-width handle's split is a few dozen instructions, so a test per item for
//!   leaves would mispredict on the mix of leaves and varying stops in item order at
//!   about the split's own cost (about 13 cycles, on up to half the items). A first
//!   loop writes the leaf entries and lists the varying stops, both without branches,
//!   and a second splits the listed stops.
//! - A runtime-width slot's split loops over the 16-byte units of its rows' span (up
//!   to the group's, at least nine) and stores two masks of them. Back to back, as a
//!   listed loop would run them, those stores bound the loop; in one loop over the
//!   items, the leaves between splits, and the mispredicted tests among them, run
//!   while the stores drain. So slots keep one loop that skips leaves with a branch.
//!   On the M4, with splits over every unit, a listed loop's resolve was even at 16
//!   units (2,048 rows) and 17–20% slower at 32 (4,096 rows); with fixed-width
//!   handles (at most 8 units) it fell 15–37% on groups of 104–128 rows, and on the
//!   256–1,008-row panels, whose 50-tree models resolve in under 0.5 µs, it moved
//!   within ±8% of that.

use std::hint::select_unpredictable;

use super::isa::{DISPATCH, Isa, Native};
use super::mask::rows::{self, Span, Unit};
use super::mask::{Mask, Wide};
use super::scan::Scan;
use super::walk::Item;
use crate::exact::RunSink;
use crate::load::layout::{FLAG_LEAF, LINK_STRIDE, MASK_FIELD, MASK_SHIFT};

/// How an item carries its rows: the mask itself when it fits a register, or a
/// handle into the group's mask arena.
///
/// Masks are counted in `u` units of `M` each: one for fixed-width carriers, and the
/// group's [`rows::units`] for [`Slot`], whose `M` is the runtime-width [`Unit`].
pub(super) trait Carrier<M: Mask>: Copy {
    /// Whether the carrier is a handle into an arena.
    const ARENA: bool;
    /// Whether masks take a runtime number of units.
    const RUNTIME: bool;
    /// A placeholder value.
    const NONE: Self;
    /// Arena units per slot for masks of `u` units, by which a handle and the next
    /// free slot count: 1 where handles count slots.
    fn slot_units(u: usize) -> usize;
    /// What a split reads of its heavy side's mask: the mask, or where it starts.
    type Side: Copy;

    /// The carrier of every root of a group of `n` rows: all rows, or arena slot 0,
    /// which this writes.
    ///
    /// # Safety
    ///
    /// For handles, `arena` has room for one mask.
    unsafe fn root(arena: *mut M, n: usize, u: usize) -> Self;

    /// The split side of mask `field` of `masks`.
    ///
    /// # Safety
    ///
    /// `masks` holds more than `field` masks.
    unsafe fn side(masks: *const M, field: usize, u: usize) -> Self::Side;

    /// The heavy side of a split on varying predicate `pred` among the carried rows,
    /// its left rows by `scan`, or its right rows if `right`: in `scratch` for
    /// runtime-width masks.
    ///
    /// # Safety
    ///
    /// For handles, `self` is a written slot of `arena`; `scratch` has room for one
    /// mask, which the returned side reads.
    unsafe fn scan_side(
        self,
        arena: *const M,
        pred: usize,
        right: bool,
        scan: &mut impl Scan<M>,
        scratch: *mut M,
        u: usize,
    ) -> Self::Side;

    /// Split the rows at a split whose heavy side's rows are `side`: the heavy and
    /// the light side's carriers, and whether each side has rows. A handle keeps a slot
    /// for the continuing side and takes a new one for the light side only when that
    /// side has rows, or with `force` on every split; the shared root slot takes new
    /// ones on a real split.
    ///
    /// # Safety
    ///
    /// For handles, `self` is a written slot of `arena`, and `arena` has room for
    /// two slots past `*next`, which counts in [`Self::slot_units`].
    unsafe fn split(
        self,
        side: Self::Side,
        force: bool,
        arena: *mut M,
        next: &mut usize,
        u: usize,
    ) -> (Self, Self, bool, bool);

    /// Add `v` to the carried rows' runs in `sink`.
    ///
    /// # Safety
    ///
    /// For handles, `self` is a written slot of `arena`.
    unsafe fn add_runs<S: RunSink + ?Sized>(
        &self,
        arena: *const M,
        v: S::Value,
        sink: &mut S,
        u: usize,
    );
}

macro_rules! impl_by_value {
    ($($ty:ty),*) => {$(
        impl Carrier<$ty> for $ty {
            const ARENA: bool = false;
            const RUNTIME: bool = false;
            const NONE: Self = 0;
            type Side = $ty;

            #[inline(always)]
            fn slot_units(_: usize) -> usize {
                1
            }

            #[inline(always)]
            unsafe fn root(_: *mut $ty, n: usize, _: usize) -> Self {
                <$ty as Mask>::first(n)
            }

            #[inline(always)]
            unsafe fn side(masks: *const $ty, field: usize, _: usize) -> $ty {
                // SAFETY: the caller guarantees the mask exists.
                unsafe { *masks.add(field) }
            }

            #[inline(always)]
            unsafe fn scan_side(
                self,
                _: *const $ty,
                pred: usize,
                right: bool,
                scan: &mut impl Scan<$ty>,
                _: *mut $ty,
                _: usize,
            ) -> $ty {
                scan.left(pred, self).xor(<$ty as Mask>::ones_if(right))
            }

            #[inline(always)]
            unsafe fn split(
                self,
                side: $ty,
                _: bool,
                _: *mut $ty,
                _: &mut usize,
                _: usize,
            ) -> (Self, Self, bool, bool) {
                let heavy = self & side;
                let light = self.and_not(side);
                (heavy, light, heavy != 0, light != 0)
            }

            #[inline(always)]
            unsafe fn add_runs<S: RunSink + ?Sized>(
                &self,
                _: *const $ty,
                v: S::Value,
                sink: &mut S,
                _: usize,
            ) {
                Mask::add_runs(self, v, sink);
            }
        }
    )*};
}

impl_by_value!(u32, u64);

/// A slot of the group's mask arena of fixed-width masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Handle(u32);

macro_rules! impl_handle {
    ($($w:literal),*) => {$(
        impl<X: Isa> Carrier<Wide<$w, X>> for Handle {
            const ARENA: bool = true;
            const RUNTIME: bool = false;
            const NONE: Self = Self(0);
            type Side = Wide<$w, X>;

            #[inline(always)]
            fn slot_units(_: usize) -> usize {
                1
            }

            #[inline(always)]
            unsafe fn root(arena: *mut Wide<$w, X>, n: usize, _: usize) -> Self {
                // SAFETY: the caller guarantees slot 0.
                unsafe { *arena = Wide::<$w, X>::first(n) };
                Self(0)
            }

            #[inline(always)]
            unsafe fn side(masks: *const Wide<$w, X>, field: usize, _: usize) -> Wide<$w, X> {
                // SAFETY: the caller guarantees the mask exists.
                unsafe { *masks.add(field) }
            }

            #[inline(always)]
            unsafe fn scan_side(
                self,
                arena: *const Wide<$w, X>,
                pred: usize,
                right: bool,
                scan: &mut impl Scan<Wide<$w, X>>,
                _: *mut Wide<$w, X>,
                _: usize,
            ) -> Wide<$w, X> {
                // SAFETY: the caller guarantees self is a written slot.
                let rows = unsafe { *arena.add(self.0 as usize) };
                scan.left(pred, rows).xor(Wide::<$w, X>::ones_if(right))
            }

            #[inline(always)]
            unsafe fn split(
                self,
                side: Wide<$w, X>,
                force: bool,
                arena: *mut Wide<$w, X>,
                next: &mut usize,
                _: usize,
            ) -> (Self, Self, bool, bool) {
                // SAFETY: the caller guarantees self is written and two slots past
                // *next exist.
                unsafe {
                    let rows = *arena.add(self.0 as usize);
                    let heavy = rows.and(side);
                    let light = rows.and_not(side);
                    let (h, l) = (!heavy.is_zero(), !light.is_zero());
                    let split = (h & l) | force;
                    let shared = split & (self.0 == 0);
                    let keep = select_unpredictable(shared, *next as u32, self.0);
                    let fresh = *next + usize::from(shared);
                    *arena.add(keep as usize) = Wide::<$w, X>::select(h | split, heavy, light);
                    *arena.add(fresh) = light;
                    *next += usize::from(split) + usize::from(shared);
                    (
                        Self(keep),
                        Self(select_unpredictable(split, fresh as u32, keep)),
                        h,
                        l,
                    )
                }
            }

            #[inline(always)]
            unsafe fn add_runs<S: RunSink + ?Sized>(
                &self,
                arena: *const Wide<$w, X>,
                v: S::Value,
                sink: &mut S,
                _: usize,
            ) {
                // SAFETY: the caller guarantees self is a written slot.
                unsafe { (*arena.add(self.0 as usize)).add_runs(v, sink) };
            }
        }
    )*};
}

impl_handle!(2, 4, 8, 16);

/// A slot of the group's mask arena of runtime-width masks, as the unit where it
/// starts, so that neither reading a slot nor taking the next free one multiplies. A
/// slot is a header unit, which holds its mask's row span ([`Span`]) as two words,
/// then the mask's `u` units, of which only the span's are read or written: a split
/// reads and writes only its rows' units, and a leaf entry adds only its rows' words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Slot(u32);

impl Slot {
    /// The slot's mask.
    ///
    /// # Safety
    ///
    /// `arena` holds the slot.
    #[inline(always)]
    const unsafe fn at<X>(self, arena: *const Unit<X>) -> *const Unit<X> {
        // SAFETY: the caller guarantees the slot, whose mask follows its header.
        unsafe { arena.add(self.0 as usize + 1) }
    }

    /// The span of the slot's mask.
    ///
    /// # Safety
    ///
    /// `arena` holds the slot, written.
    #[inline(always)]
    const unsafe fn span<X>(self, arena: *const Unit<X>) -> Span {
        // SAFETY: the caller guarantees the slot's header.
        let [lo, hi] = unsafe { (*arena.add(self.0 as usize)).0 };
        Span {
            lo: lo as usize,
            hi: hi as usize,
        }
    }

    /// Writes span `s` into the header of the slot at unit `at`.
    ///
    /// # Safety
    ///
    /// `arena` holds the slot.
    #[inline(always)]
    const unsafe fn set_span<X>(arena: *mut Unit<X>, at: usize, s: Span) {
        // SAFETY: the caller guarantees the slot's header.
        unsafe { *arena.add(at) = Wide::new([s.lo as u64, s.hi as u64]) };
    }
}

impl<X: Isa> Carrier<Unit<X>> for Slot {
    const ARENA: bool = true;
    const RUNTIME: bool = true;
    const NONE: Self = Self(0);
    type Side = *const Unit<X>;

    #[inline(always)]
    fn slot_units(u: usize) -> usize {
        u + 1
    }

    #[inline(always)]
    unsafe fn root(arena: *mut Unit<X>, n: usize, u: usize) -> Self {
        // SAFETY: the caller guarantees slot 0, a header and u units.
        unsafe {
            Self::set_span(arena, 0, Span::first(n));
            rows::first(std::slice::from_raw_parts_mut(arena.add(1), u), n);
        }
        Self(0)
    }

    #[inline(always)]
    unsafe fn side(masks: *const Unit<X>, field: usize, u: usize) -> *const Unit<X> {
        // SAFETY: the caller guarantees the mask exists.
        unsafe { masks.add(field * u) }
    }

    /// Scans only the span's units, the ones a split reads.
    #[inline(always)]
    unsafe fn scan_side(
        self,
        arena: *const Unit<X>,
        pred: usize,
        right: bool,
        scan: &mut impl Scan<Unit<X>>,
        scratch: *mut Unit<X>,
        _: usize,
    ) -> *const Unit<X> {
        // SAFETY: the caller guarantees the slot, written.
        let units = unsafe { self.span(arena) }.units();
        // SAFETY: as above, and the scratch, which no slot overlaps, has room for a
        // mask's units, the span's among them.
        let (rows, left) = unsafe {
            (
                std::slice::from_raw_parts(self.at(arena).add(units.start), units.len()),
                std::slice::from_raw_parts_mut(scratch.add(units.start), units.len()),
            )
        };
        scan.left_rows(pred, rows, left, units.start);
        if right {
            for x in left {
                *x = x.xor(Unit::<X>::ones_if(true));
            }
        }
        scratch
    }

    /// Unlike [`Handle`]'s, which selects the continuing side's mask, the heavy side is
    /// written over the slot as it is read, and the light side to a new slot, which
    /// the split takes when the light side has rows; on the shared root slot, both
    /// go to new slots, which a real split takes. Both sides lie within the slot's
    /// span, so the split reads and writes only its units, and each side's header
    /// gets its own span.
    #[inline(always)]
    unsafe fn split(
        self,
        side: *const Unit<X>,
        force: bool,
        arena: *mut Unit<X>,
        next: &mut usize,
        u: usize,
    ) -> (Self, Self, bool, bool) {
        let stride = <Self as Carrier<Unit<X>>>::slot_units(u);
        let root = self.0 == 0;
        let keep = select_unpredictable(root, *next as u32, self.0);
        let fresh = *next as u32 + select_unpredictable(root, stride as u32, 0);
        // SAFETY: the caller guarantees self is written and two slots past *next
        // exist; keep is self or a new slot, and fresh a new slot past keep.
        let (heavy, light) = unsafe {
            let (k, f) = (Self(keep), Self(fresh));
            let spans = rows::split::<X>(
                self.at(arena),
                side,
                k.at(arena).cast_mut(),
                f.at(arena).cast_mut(),
                self.span(arena),
            );
            Self::set_span(arena, keep as usize, spans.0);
            Self::set_span(arena, fresh as usize, spans.1);
            spans
        };
        let (h, l) = (!heavy.is_empty(), !light.is_empty());
        let split = (h & l) | force;
        let unsplit_root = root && !split;
        let taken = select_unpredictable(root, 2 * usize::from(split), usize::from(l | force));
        *next += taken * stride;
        (
            Self(select_unpredictable(unsplit_root, 0, keep)),
            Self(select_unpredictable(unsplit_root, 0, fresh)),
            h,
            l,
        )
    }

    #[inline(always)]
    unsafe fn add_runs<S: RunSink + ?Sized>(
        &self,
        arena: *const Unit<X>,
        v: S::Value,
        sink: &mut S,
        _: usize,
    ) {
        // SAFETY: the caller guarantees self is a written slot, whose span's words
        // lie in its mask.
        let (w, lo) = unsafe {
            let s = self.span(arena);
            let words = self.at(arena).cast::<u64>().add(s.lo);
            (std::slice::from_raw_parts(words, s.hi - s.lo), s.lo)
        };
        rows::add_runs(w, lo, v, sink);
    }
}

/// What the counted resolve passes saw.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Splits {
    pub(super) splits: u64,
    pub(super) empty_sides: u64,
}

/// Where a resolve pass reads and writes.
pub(super) struct Resolve<M, C> {
    pub(super) stops: *const u32,
    pub(super) words: *const u64,
    pub(super) masks: *const M,
    pub(super) arena: *mut M,
    /// A mask's room for scanned sides, when precompute is off.
    pub(super) scratch: *mut M,
    /// Units of `M` per mask ([`Carrier`]).
    pub(super) units: usize,
    pub(super) next_slot: usize,
    pub(super) items: *const Item<C>,
    pub(super) out: *mut Item<C>,
    pub(super) leaves: *mut Item<C>,
    pub(super) n_leaves: usize,
    /// The positions of a pass's varying stops, for fixed-width handles.
    pub(super) listed: *mut u32,
}

/// Whether resolve lists fixed-width handles' varying stops with AVX-512's compress
/// (`list_avx512`) at level `X` rather than item by item: with AVX-512F, VL and DQ.
pub(super) const fn compress<X: Isa>() -> bool {
    cfg!(target_arch = "x86_64") && X::AVX512 && X::AVX512DQ
}

/// Entries past the last that listing fixed-width handles' varying stops may write, in
/// `leaves` and in `listed`: `list_avx512` stores a block of 8 whole, where the build
/// or a level it may dispatch to compresses; 0 for the loop item by item.
pub(super) const LISTED_SLACK: usize = if compress::<Native>() || DISPATCH {
    7
} else {
    0
};

/// Turn a pass's stops into the next pass's items, returned as a count, and leaf
/// entries.
///
/// # Safety
///
/// `pass.stops` and `pass.words` hold what [`walk`](super::walk::walk) wrote, with `VARYING`, for the
/// `n_items` items `pass.items`. `pass.masks` holds the all-rows mask, then every
/// predicate's left and right masks, `pass.out` has room for two items per item plus one and
/// `pass.leaves` for one entry per leaf plus one, `pass.scratch` for one mask, the
/// arena as [`Carrier::split`] requires, and for fixed-width handles `pass.leaves` has
/// [`LISTED_SLACK`] more entries and `pass.listed` room for `n_items` positions plus
/// [`LISTED_SLACK`].
#[inline(always)]
pub(super) unsafe fn resolve<
    X: Isa,
    M: Mask,
    C: Carrier<M>,
    const SCAN: bool,
    const EMPTY: bool,
    const STATS: bool,
>(
    pass: &mut Resolve<M, C>,
    n_items: usize,
    scan: &mut impl Scan<M>,
    seen: &mut Splits,
) -> usize {
    if C::ARENA && !C::RUNTIME {
        // SAFETY: the caller guarantees the bounds above.
        return unsafe { resolve_listed::<X, M, C, SCAN, EMPTY, STATS>(pass, n_items, scan, seen) };
    }
    let u = pass.units;
    // The split of an item, given its word 1, carry and kind, as (heavy, light,
    // has_heavy, has_light), counted with `STATS`. Leaves read entry 0, the all-rows
    // mask; a split on varying predicate i reads its heavy side's, 2i + 1 for the left
    // and 2i + 2 for the right. A leaf passes on no side. By value its split is a few
    // instructions and needs no branch; a slot's would read and write the arena (see
    // the module docs).
    macro_rules! split_of {
        ($w1:expr, $carry:expr, $is_leaf:expr) => {{
            let (w1, carry, is_leaf) = ($w1, $carry, $is_leaf);
            let is_varying = !is_leaf;
            let field = ((w1 >> MASK_SHIFT) & MASK_FIELD) as usize;
            let side = if SCAN && is_varying {
                let (pred, right) = ((field - 1) / 2, field.is_multiple_of(2));
                carry.scan_side(pass.arena, pred, right, scan, pass.scratch, u)
            } else {
                C::side(pass.masks, field, u)
            };
            let (heavy, light, has_heavy, has_light) = if C::RUNTIME && is_leaf {
                (carry, carry, false, false)
            } else {
                carry.split(side, EMPTY & is_varying, pass.arena, &mut pass.next_slot, u)
            };
            if STATS && is_varying {
                seen.splits += u64::from(has_heavy & has_light);
                if EMPTY {
                    seen.empty_sides += u64::from(!has_heavy) + u64::from(!has_light);
                }
            }
            (heavy, light, has_heavy, has_light)
        }};
    }
    if cfg!(target_arch = "x86_64") && !C::RUNTIME {
        // On x86-64, for carriers by value, cursors through the items, words, stops,
        // leaf entries and items written, rather than indices beside their bases:
        // three fewer live registers, which keeps the loop's bases out of stack
        // reloads (a runtime-width slot's frame grew instead).
        // SAFETY: the caller guarantees the bounds above; the cursors advance by one
        // item per item, and the outputs' by at most what the indices would.
        unsafe {
            let (mut w, mut it, mut st) = (pass.words, pass.items, pass.stops);
            let end = w.add(n_items);
            let (mut leaf_at, mut out) = (pass.leaves.add(pass.n_leaves), pass.out);
            while w < end {
                let (w1, carry, node) = (*w, (*it).carry, *st);
                (w, it, st) = (w.add(1), it.add(1), st.add(1));
                let child = w1 as u32;
                let is_leaf = w1 & FLAG_LEAF != 0;
                *leaf_at = Item { node: child, carry };
                leaf_at = leaf_at.add(usize::from(is_leaf));
                let is_varying = !is_leaf;
                let (heavy, light, has_heavy, has_light) = split_of!(w1, carry, is_leaf);
                let (pass_heavy, pass_light) = (has_heavy | EMPTY, has_light | EMPTY);
                *out = Item {
                    node: node + LINK_STRIDE,
                    carry: heavy,
                };
                out = out.add(usize::from(is_varying & pass_heavy));
                *out = Item {
                    node: child,
                    carry: light,
                };
                out = out.add(usize::from(is_varying & pass_light));
            }
            pass.n_leaves = leaf_at.offset_from_unsigned(pass.leaves);
            return out.offset_from_unsigned(pass.out);
        }
    }
    let mut n = 0usize;
    // SAFETY: the caller guarantees the bounds above.
    unsafe {
        for j in 0..n_items {
            let w1 = *pass.words.add(j);
            let carry = (*pass.items.add(j)).carry;
            let child = w1 as u32;
            let is_leaf = w1 & FLAG_LEAF != 0;
            *pass.leaves.add(pass.n_leaves) = Item { node: child, carry };
            pass.n_leaves += usize::from(is_leaf);
            let is_varying = !is_leaf;
            let node = *pass.stops.add(j);
            let (heavy, light, has_heavy, has_light) = split_of!(w1, carry, is_leaf);
            *pass.out.add(n) = Item {
                node: node + LINK_STRIDE,
                carry: heavy,
            };
            let (pass_heavy, pass_light) = (has_heavy | EMPTY, has_light | EMPTY);
            n += usize::from(is_varying & pass_heavy);
            *pass.out.add(n) = Item {
                node: child,
                carry: light,
            };
            n += usize::from(is_varying & pass_light);
        }
    }
    n
}

/// [`resolve`] for fixed-width handles: the leaf entries and the list of varying stops in
/// one loop, then the listed stops' splits in another, so that no leaf reads or
/// writes the arena and no branch depends on an item's kind. Items, leaf entries and
/// arena slots come out as one loop over the items would make them.
///
/// # Safety
///
/// As for [`resolve`].
#[inline(always)]
unsafe fn resolve_listed<
    X: Isa,
    M: Mask,
    C: Carrier<M>,
    const SCAN: bool,
    const EMPTY: bool,
    const STATS: bool,
>(
    pass: &mut Resolve<M, C>,
    n_items: usize,
    scan: &mut impl Scan<M>,
    seen: &mut Splits,
) -> usize {
    let u = pass.units;
    // SAFETY: the caller guarantees the bounds of [`resolve`], with the slack.
    let (n_leaves, listed) = unsafe {
        list::<X, C>(
            pass.words,
            pass.items,
            n_items,
            pass.leaves,
            pass.n_leaves,
            pass.listed,
        )
    };
    pass.n_leaves = n_leaves;
    // The split of listed position `$j`, binding its word 1, node, sides and whether
    // each side has rows.
    macro_rules! split {
        ($j:expr => $w1:ident, $node:ident, $heavy:ident, $light:ident, $has_heavy:ident, $has_light:ident) => {
            let j = $j;
            let $w1 = *pass.words.add(j);
            let carry = (*pass.items.add(j)).carry;
            let $node = *pass.stops.add(j);
            let field = (($w1 >> MASK_SHIFT) & MASK_FIELD) as usize;
            let side = if SCAN {
                let (pred, right) = ((field - 1) / 2, field.is_multiple_of(2));
                carry.scan_side(pass.arena, pred, right, scan, pass.scratch, u)
            } else {
                C::side(pass.masks, field, u)
            };
            let ($heavy, $light, $has_heavy, $has_light) =
                carry.split(side, EMPTY, pass.arena, &mut pass.next_slot, u);
            if STATS {
                seen.splits += u64::from($has_heavy & $has_light);
                if EMPTY {
                    seen.empty_sides += u64::from(!$has_heavy) + u64::from(!$has_light);
                }
            }
        };
    }
    if cfg!(target_arch = "x86_64") {
        // On x86-64, a cursor through the list and one through the items written,
        // rather than an index into each beside their bases: two fewer live registers,
        // which keeps the loop's state off the stack.
        // SAFETY: as above; every listed position is an item's, below `n_items`, and
        // the list holds `listed` of them.
        unsafe {
            let (mut at, end, mut out) =
                (pass.listed.cast_const(), pass.listed.add(listed), pass.out);
            while at.cast_mut() < end {
                split!(*at as usize => w1, node, heavy, light, has_heavy, has_light);
                at = at.add(1);
                *out = Item {
                    node: node + LINK_STRIDE,
                    carry: heavy,
                };
                out = out.add(usize::from(has_heavy | EMPTY));
                *out = Item {
                    node: w1 as u32,
                    carry: light,
                };
                out = out.add(usize::from(has_light | EMPTY));
            }
            return out.offset_from_unsigned(pass.out);
        }
    }
    let mut n = 0usize;
    // SAFETY: as above; every listed position is an item's, below `n_items`.
    unsafe {
        for i in 0..listed {
            split!(*pass.listed.add(i) as usize => w1, node, heavy, light, has_heavy, has_light);
            *pass.out.add(n) = Item {
                node: node + LINK_STRIDE,
                carry: heavy,
            };
            n += usize::from(has_heavy | EMPTY);
            *pass.out.add(n) = Item {
                node: w1 as u32,
                carry: light,
            };
            n += usize::from(has_light | EMPTY);
        }
    }
    n
}

/// Resolve's first loop for fixed-width handles: writes the leaf entries of
/// `items[..n_items]` from `n_leaves` on and the positions of their varying stops to
/// `listed`, and returns both counts. With AVX-512, items of 4-byte carriers go 8
/// at a time (`list_avx512`).
///
/// # Safety
///
/// `words` and `items` hold `n_items` entries; `leaves` has room for the leaf entries
/// past `n_leaves` plus 1 + [`LISTED_SLACK`], and `listed` for `n_items` +
/// [`LISTED_SLACK`] positions.
#[inline(always)]
unsafe fn list<X: Isa, C: Copy>(
    words: *const u64,
    items: *const Item<C>,
    n_items: usize,
    leaves: *mut Item<C>,
    n_leaves: usize,
    listed: *mut u32,
) -> (usize, usize) {
    #[cfg(target_arch = "x86_64")]
    if compress::<X>() && size_of::<C>() == 4 {
        // SAFETY: the caller guarantees the bounds, and the carrier's size; the level
        // has AVX-512F, VL and DQ.
        return unsafe { list_avx512(words, items, n_items, leaves, n_leaves, listed) };
    }
    // SAFETY: the caller guarantees the bounds.
    unsafe { list_scalar(words, items, n_items, leaves, n_leaves, listed) }
}

/// [`list`] item by item, without branches: every item writes a leaf entry and a
/// position, and the counts advance by its kind.
///
/// # Safety
///
/// As for [`list`], without the slack.
#[inline(always)]
unsafe fn list_scalar<C: Copy>(
    words: *const u64,
    items: *const Item<C>,
    n_items: usize,
    leaves: *mut Item<C>,
    mut n_leaves: usize,
    listed: *mut u32,
) -> (usize, usize) {
    let mut n_listed = 0usize;
    // SAFETY: the caller guarantees the bounds; `n_listed` stays at or below `j`, so
    // the list's writes stay below `n_items`.
    unsafe {
        for j in 0..n_items {
            let w1 = *words.add(j);
            let carry = (*items.add(j)).carry;
            let is_leaf = w1 & FLAG_LEAF != 0;
            *leaves.add(n_leaves) = Item {
                node: w1 as u32,
                carry,
            };
            n_leaves += usize::from(is_leaf);
            *listed.add(n_listed) = j as u32;
            n_listed += usize::from(!is_leaf);
        }
    }
    (n_leaves, n_listed)
}

/// [`list`] for 4-byte carriers, 8 items at a time: a mask of the leaves by
/// `vptestmq`, the entries (word 1's low half, the item's carry) by a bitwise select,
/// then the leaves' entries and the varying stops' positions compressed into
/// registers by `vpcompressq` and `vpcompressd`, stored whole, and the counts
/// advanced by the masks' populations. The register form and a plain store, because
/// the compress to memory is microcoded on Zen 4. A last block of fewer than 8 items
/// loads under a mask of them. Items and entries come out in the order of [`list_scalar`].
///
/// # Safety
///
/// As for [`list`], `C` is 4 bytes, so that an item is its node's 4 bytes then the
/// carry's, and the CPU has AVX-512F, VL and DQ.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq")]
#[inline]
unsafe fn list_avx512<C: Copy>(
    words: *const u64,
    items: *const Item<C>,
    n_items: usize,
    leaves: *mut Item<C>,
    mut n_leaves: usize,
    listed: *mut u32,
) -> (usize, usize) {
    use std::arch::x86_64::{
        __m256i, __m512i, __mmask8, _mm256_add_epi32, _mm256_maskz_compress_epi32,
        _mm256_set1_epi32, _mm256_setr_epi32, _mm256_storeu_si256, _mm512_maskz_compress_epi64,
        _mm512_maskz_loadu_epi64, _mm512_set1_epi64, _mm512_storeu_si512,
        _mm512_ternarylogic_epi64, _mm512_test_epi64_mask,
    };
    /// `!leaf & block` by one `kandnb` (AVX512DQ), a mask-register instruction on
    /// port 0. Written in Rust, LLVM folds the complement into a second test of the
    /// words (`vptestnmq`), a port-5 µop beside the test's and the compresses'.
    #[target_feature(enable = "avx512f,avx512dq")]
    #[inline]
    fn others(leaf: __mmask8, block: __mmask8) -> __mmask8 {
        let out: __mmask8;
        // SAFETY: the function enables AVX512DQ; the instruction reads and writes mask
        // registers only.
        unsafe {
            std::arch::asm!(
                "kandnb {o}, {l}, {b}",
                l = in(kreg) leaf,
                b = in(kreg) block,
                o = lateout(kreg) out,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        out
    }
    debug_assert_eq!(size_of::<Item<C>>(), 8);
    let mut n_listed = 0usize;
    // SAFETY: the caller guarantees the bounds, with the slack: a block's stores write
    // 8 entries from `n_leaves` and 8 positions from `n_listed`, which then advance by
    // at most 8, and loads read only the block's items below `n_items` (masked loads
    // do not touch the lanes they leave out).
    unsafe {
        let flag = _mm512_set1_epi64(FLAG_LEAF as i64);
        // Each item's low dword, from word 1, by a bitwise select on this rather than
        // a dword blend under a mask register, which LLVM rebuilt in every block.
        let low = _mm512_set1_epi64(0xFFFF_FFFF);
        let mut positions = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        // One block of 8 items from `j`, of which the `count` in `block` are loaded.
        let mut list_block = |j: usize, block: u8, count: usize| {
            let w = _mm512_maskz_loadu_epi64(block, words.add(j).cast());
            let it = _mm512_maskz_loadu_epi64(block, items.add(j).cast());
            let leaf = _mm512_test_epi64_mask(w, flag);
            let varying = others(leaf, block);
            // Odd dwords, the items' high halves, are the carries: `low ? w : it`.
            let entries: __m512i = _mm512_ternarylogic_epi64::<0xCA>(low, w, it);
            _mm512_storeu_si512(
                leaves.add(n_leaves).cast(),
                _mm512_maskz_compress_epi64(leaf, entries),
            );
            let at: __m256i = _mm256_maskz_compress_epi32(varying, positions);
            _mm256_storeu_si256(listed.add(n_listed).cast(), at);
            let leaf_count = leaf.count_ones() as usize;
            n_leaves += leaf_count;
            n_listed += count - leaf_count;
            positions = _mm256_add_epi32(positions, _mm256_set1_epi32(8));
        };
        let full = n_items - n_items % 8;
        let mut j = 0usize;
        while j < full {
            list_block(j, u8::MAX, 8);
            j += 8;
        }
        if full < n_items {
            let rest = n_items - full;
            list_block(full, (1u8 << rest) - 1, rest);
        }
    }
    (n_leaves, n_listed)
}

/// `list_avx512` against [`list_scalar`]. These run only where the build enables
/// AVX-512F and VL (`-C target-cpu=x86-64-v4`, or `native` on such a CPU): first run
/// on GCE, since neither Rosetta nor Miri executes AVX-512.
#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl"
))]
mod tests {
    use super::{FLAG_LEAF, Handle, Item, LISTED_SLACK, list_avx512, list_scalar};

    /// A guard value past the slack, which neither loop may write.
    const GUARD: u32 = 0xDEAD_BEEF;

    struct Lists {
        n_leaves: usize,
        n_listed: usize,
        leaves: Vec<Item<Handle>>,
        listed: Vec<u32>,
    }

    /// Both loops over the same input, each into buffers sized as [`super::list`]
    /// requires, with a guard past them.
    fn lists(words: &[u64], items: &[Item<Handle>], n_leaves: usize, avx512: bool) -> Lists {
        let n = words.len();
        let guard = Item {
            node: GUARD,
            carry: Handle(GUARD),
        };
        let mut leaves = vec![guard; n_leaves + n + 1 + LISTED_SLACK + 8];
        let mut listed = vec![GUARD; n + LISTED_SLACK + 8];
        let f = if avx512 {
            list_avx512::<Handle>
        } else {
            list_scalar::<Handle>
        };
        // SAFETY: the buffers are sized as `list` requires, and a handle is 4 bytes.
        let (end, n_listed) = unsafe {
            f(
                words.as_ptr(),
                items.as_ptr(),
                n,
                leaves.as_mut_ptr(),
                n_leaves,
                listed.as_mut_ptr(),
            )
        };
        let written = n_leaves + n + 1 + LISTED_SLACK;
        assert!(
            leaves[written..]
                .iter()
                .all(|e| e.node == GUARD && e.carry.0 == GUARD)
        );
        assert!(listed[n + LISTED_SLACK..].iter().all(|&p| p == GUARD));
        Lists {
            n_leaves: end,
            n_listed,
            leaves,
            listed,
        }
    }

    #[test]
    fn compressed_lists_match_the_loop_item_by_item() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for n in (0..=70).chain([127, 128, 129, 1000, 1023]) {
            for case in 0..8 {
                // Leaf shares from none to all; word 1's other bits random.
                let share = case % 5;
                let words: Vec<u64> = (0..n)
                    .map(|_| {
                        let w = next() & !FLAG_LEAF;
                        if next() % 4 < share { w | FLAG_LEAF } else { w }
                    })
                    .collect();
                let items: Vec<Item<Handle>> = (0..n)
                    .map(|_| {
                        let r = next();
                        Item {
                            node: r as u32,
                            carry: Handle((r >> 32) as u32),
                        }
                    })
                    .collect();
                let start = (next() % 9) as usize;
                let (a, b) = (
                    lists(&words, &items, start, false),
                    lists(&words, &items, start, true),
                );
                assert_eq!(
                    (a.n_leaves, a.n_listed),
                    (b.n_leaves, b.n_listed),
                    "{n} {case}"
                );
                for (x, y) in a.leaves[start..a.n_leaves]
                    .iter()
                    .zip(&b.leaves[start..b.n_leaves])
                {
                    assert_eq!((x.node, x.carry), (y.node, y.carry), "{n} {case}");
                }
                assert_eq!(a.listed[..a.n_listed], b.listed[..b.n_listed], "{n} {case}");
            }
        }
    }
}
