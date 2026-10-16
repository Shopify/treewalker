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

use super::mask::rows::{self, Span, Unit};
use super::mask::{Mask, Wide};
use super::scan::Scan;
use super::walk::Item;
use crate::exact::RunSink;
use crate::load::layout::{FLAG_LEAF, MASK_FIELD, MASK_SHIFT};

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
        impl Carrier<Wide<$w>> for Handle {
            const ARENA: bool = true;
            const RUNTIME: bool = false;
            const NONE: Self = Self(0);
            type Side = Wide<$w>;

            #[inline(always)]
            fn slot_units(_: usize) -> usize {
                1
            }

            #[inline(always)]
            unsafe fn root(arena: *mut Wide<$w>, n: usize, _: usize) -> Self {
                // SAFETY: the caller guarantees slot 0.
                unsafe { *arena = Wide::<$w>::first(n) };
                Self(0)
            }

            #[inline(always)]
            unsafe fn side(masks: *const Wide<$w>, field: usize, _: usize) -> Wide<$w> {
                // SAFETY: the caller guarantees the mask exists.
                unsafe { *masks.add(field) }
            }

            #[inline(always)]
            unsafe fn scan_side(
                self,
                arena: *const Wide<$w>,
                pred: usize,
                right: bool,
                scan: &mut impl Scan<Wide<$w>>,
                _: *mut Wide<$w>,
                _: usize,
            ) -> Wide<$w> {
                // SAFETY: the caller guarantees self is a written slot.
                let rows = unsafe { *arena.add(self.0 as usize) };
                scan.left(pred, rows).xor(Wide::<$w>::ones_if(right))
            }

            #[inline(always)]
            unsafe fn split(
                self,
                side: Wide<$w>,
                force: bool,
                arena: *mut Wide<$w>,
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
                    *arena.add(keep as usize) = Wide::<$w>::select(h | split, heavy, light);
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
                arena: *const Wide<$w>,
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
    const unsafe fn at(self, arena: *const Unit) -> *const Unit {
        // SAFETY: the caller guarantees the slot, whose mask follows its header.
        unsafe { arena.add(self.0 as usize + 1) }
    }

    /// The span of the slot's mask.
    ///
    /// # Safety
    ///
    /// `arena` holds the slot, written.
    #[inline(always)]
    const unsafe fn span(self, arena: *const Unit) -> Span {
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
    const unsafe fn set_span(arena: *mut Unit, at: usize, s: Span) {
        // SAFETY: the caller guarantees the slot's header.
        unsafe { *arena.add(at) = Wide([s.lo as u64, s.hi as u64]) };
    }
}

impl Carrier<Unit> for Slot {
    const ARENA: bool = true;
    const RUNTIME: bool = true;
    const NONE: Self = Self(0);
    type Side = *const Unit;

    #[inline(always)]
    fn slot_units(u: usize) -> usize {
        u + 1
    }

    #[inline(always)]
    unsafe fn root(arena: *mut Unit, n: usize, u: usize) -> Self {
        // SAFETY: the caller guarantees slot 0, a header and u units.
        unsafe {
            Self::set_span(arena, 0, Span::first(n));
            rows::first(std::slice::from_raw_parts_mut(arena.add(1), u), n);
        }
        Self(0)
    }

    #[inline(always)]
    unsafe fn side(masks: *const Unit, field: usize, u: usize) -> *const Unit {
        // SAFETY: the caller guarantees the mask exists.
        unsafe { masks.add(field * u) }
    }

    /// Scans only the span's units, the ones a split reads.
    #[inline(always)]
    unsafe fn scan_side(
        self,
        arena: *const Unit,
        pred: usize,
        right: bool,
        scan: &mut impl Scan<Unit>,
        scratch: *mut Unit,
        _: usize,
    ) -> *const Unit {
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
                *x = x.xor(Unit::ones_if(true));
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
        side: *const Unit,
        force: bool,
        arena: *mut Unit,
        next: &mut usize,
        u: usize,
    ) -> (Self, Self, bool, bool) {
        let stride = Self::slot_units(u);
        let root = self.0 == 0;
        let keep = select_unpredictable(root, *next as u32, self.0);
        let fresh = *next as u32 + select_unpredictable(root, stride as u32, 0);
        // SAFETY: the caller guarantees self is written and two slots past *next
        // exist; keep is self or a new slot, and fresh a new slot past keep.
        let (heavy, light) = unsafe {
            let (k, f) = (Self(keep), Self(fresh));
            let spans = rows::split(
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
        arena: *const Unit,
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

/// Turn a pass's stops into the next pass's items, returned as a count, and leaf
/// entries.
///
/// # Safety
///
/// `pass.stops` and `pass.words` hold what [`walk`](super::walk::walk) wrote, with `VARYING`, for the
/// `n_items` items `pass.items`. `pass.masks` holds the all-rows mask, then every
/// predicate's left and right masks, `pass.out` has room for two items per item plus one and
/// `pass.leaves` for one entry per leaf plus one, `pass.scratch` for one mask, the
/// arena as [`Carrier::split`] requires, and for fixed-width handles `pass.listed`
/// has room for `n_items` positions.
#[inline(always)]
pub(super) unsafe fn resolve<
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
        return unsafe { resolve_listed::<M, C, SCAN, EMPTY, STATS>(pass, n_items, scan, seen) };
    }
    let mut n = 0usize;
    let u = pass.units;
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
            // Leaves read entry 0, the all-rows mask; a split on varying predicate i
            // reads its heavy side's, 2i + 1 for the left and 2i + 2 for the right.
            let field = ((w1 >> MASK_SHIFT) & MASK_FIELD) as usize;
            let side = if SCAN && is_varying {
                let (pred, right) = ((field - 1) / 2, field.is_multiple_of(2));
                carry.scan_side(pass.arena, pred, right, scan, pass.scratch, u)
            } else {
                C::side(pass.masks, field, u)
            };
            // A leaf passes on no side. By value its split is a few instructions and
            // needs no branch; a slot's would read and write the arena (see the
            // module docs).
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
            *pass.out.add(n) = Item {
                node: node + 1,
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
    let mut listed = 0usize;
    // SAFETY: the caller guarantees the bounds of [`resolve`]; `listed` stays at or
    // below `j`, so the list's writes stay below `n_items`.
    unsafe {
        for j in 0..n_items {
            let w1 = *pass.words.add(j);
            let carry = (*pass.items.add(j)).carry;
            let is_leaf = w1 & FLAG_LEAF != 0;
            *pass.leaves.add(pass.n_leaves) = Item {
                node: w1 as u32,
                carry,
            };
            pass.n_leaves += usize::from(is_leaf);
            *pass.listed.add(listed) = j as u32;
            listed += usize::from(!is_leaf);
        }
    }
    let mut n = 0usize;
    // SAFETY: as above; every listed position is an item's, below `n_items`.
    unsafe {
        for i in 0..listed {
            let j = *pass.listed.add(i) as usize;
            let w1 = *pass.words.add(j);
            let carry = (*pass.items.add(j)).carry;
            let node = *pass.stops.add(j);
            let field = ((w1 >> MASK_SHIFT) & MASK_FIELD) as usize;
            let side = if SCAN {
                let (pred, right) = ((field - 1) / 2, field.is_multiple_of(2));
                carry.scan_side(pass.arena, pred, right, scan, pass.scratch, u)
            } else {
                C::side(pass.masks, field, u)
            };
            let (heavy, light, has_heavy, has_light) =
                carry.split(side, EMPTY, pass.arena, &mut pass.next_slot, u);
            if STATS {
                seen.splits += u64::from(has_heavy & has_light);
                if EMPTY {
                    seen.empty_sides += u64::from(!has_heavy) + u64::from(!has_light);
                }
            }
            *pass.out.add(n) = Item {
                node: node + 1,
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
