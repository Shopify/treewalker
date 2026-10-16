//! The walk pass: lanes that step constant splits until each item stops.
//!
//! A pass walks its items, each a node and the rows that reach it, with K lanes held
//! in registers. A lane steps constant splits without branches and stops at a leaf or
//! a varying split, and takes its next item; lane k takes items k, k + K, k + 2K and
//! so on. Every lane ends on the dummy node, which loops on itself, and the pass ends
//! when all of them do.
//!
//! Each item stops exactly once, so the walk keeps no event queue: every step writes
//! the lane's node and its word 1 at its item's position, and the last write is the
//! item's stop. The stores' address is the lane's cursor, which the lane holds anyway,
//! so a step pays two stores and no counter. The resolve pass ([`super::resolve`])
//! then reads the stops in item order.

use std::hint::select_unpredictable;

use crate::load::layout::{FLAG_LEAF, LayoutNode, SLOT_SHIFT, STOPS_VARYING};

pub(super) use crate::rules::{MAX_DISTANCE, MAX_LANES};

/// A lane's cursor, the position of its item: on x86-64 a `usize`, which addresses
/// the items, stops and words without extending it, where a `u32` cursor takes a
/// zero-extended copy for each address (x86-64 has no extending address mode), and
/// spills more at 8 lanes. Instructions per round of 8 lane steps, `usize` against
/// `u32` cursors, on emeraldrapids (x86-64-v4), rustc 1.97.1: the walk without varying
/// stops 180 against 190 at distance 0 (180, 190) and 202 against 219 at distance
/// 1 or more (213, 223); the walks with varying stops over 8-byte items 187 against
/// 196 (187, 194) and 216 against 232 (218, 230), over 16-byte items 200 against 211
/// (201, 211) and 236 against 237 (235, 238). Elsewhere a `u32`, which AArch64's
/// addressing extends for free (`uxtw`) and the M4's pass loop keeps narrow.
#[cfg(target_arch = "x86_64")]
type Cursor = usize;
#[cfg(not(target_arch = "x86_64"))]
type Cursor = u32;

/// The position `c` as an index.
#[inline(always)]
#[allow(clippy::unnecessary_cast, reason = "a usize on x86-64")]
const fn at(c: Cursor) -> usize {
    c as usize
}

/// Items past a pass's last that a walk of `lanes` lanes prefetching `distance`
/// items ahead may read: a lane's cursor stays below the item count plus the lanes,
/// and its prefetch reads `distance` lanes' worth of items further.
pub(super) const fn pad(lanes: usize, distance: usize) -> usize {
    lanes * (distance + 1)
}

/// The most padding any walk needs.
pub(super) const MAX_PAD: usize = pad(MAX_LANES, MAX_DISTANCE);

/// The lanes of a walk given `K`: the rule's for its stops when `K` is 0, else `K`.
pub(super) const fn lanes_of(k: usize, varying: bool) -> usize {
    if k == 0 {
        crate::rules::lanes(varying)
    } else {
        k
    }
}

/// A walk item: a node and the rows that reach it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct Item<C> {
    pub(super) node: u32,
    pub(super) carry: C,
}

/// What a walk reads of its items: their nodes. A pass without varying stops walks
/// the trees' roots in place, as bare nodes.
pub(super) trait Start: Copy {
    fn node(self) -> u32;
}

impl<C: Copy> Start for Item<C> {
    #[inline(always)]
    fn node(self) -> u32 {
        self.node
    }
}

impl Start for u32 {
    #[inline(always)]
    fn node(self) -> u32 {
        self
    }
}

/// What a pass reads of the model.
#[derive(Clone, Copy)]
pub(super) struct Pools {
    pub(super) nodes: *const LayoutNode,
    pub(super) bins: *const u32,
    pub(super) dummy: u32,
}

/// One lane's step: the next node, whether the lane stopped at `index`, and the
/// node's word 1. Every split is `bin <= k` on word 0's low 32 bits, so the slot
/// above them costs no masking.
///
/// # Safety
///
/// `index` is a node of the pool, and its slot a bin.
#[inline(always)]
unsafe fn step(p: &Pools, index: u32) -> (u32, u64) {
    // SAFETY: the caller guarantees index and its slot are in range.
    unsafe {
        let node = *p.nodes.add(index as usize);
        let x = *p.bins.add((node.w0 >> SLOT_SHIFT) as usize);
        let heavy = x <= node.w0 as u32;
        (
            select_unpredictable(heavy, index + 1, node.w1 as u32),
            node.w1,
        )
    }
}

/// Run `$step(&mut node, &mut cursor, prefetch)` for every lane, with each lane's node
/// and cursor in locals, round after round until every lane sits on `$dummy`, with
/// `prefetch` true in the first round of every `$r`.
macro_rules! lanes {
    ($k:expr, $r:expr, $dummy:expr, $first:expr, $step:expr) => {
        match $k {
            1 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0)),
            2 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0) (1, i1, c1)),
            3 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0) (1, i1, c1) (2, i2, c2)),
            4 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0) (1, i1, c1) (2, i2, c2) (3, i3, c3)),
            5 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0) (1, i1, c1) (2, i2, c2) (3, i3, c3) (4, i4, c4)),
            6 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0) (1, i1, c1) (2, i2, c2) (3, i3, c3) (4, i4, c4) (5, i5, c5)),
            7 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0) (1, i1, c1) (2, i2, c2) (3, i3, c3) (4, i4, c4) (5, i5, c5) (6, i6, c6)),
            8 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0) (1, i1, c1) (2, i2, c2) (3, i3, c3) (4, i4, c4) (5, i5, c5) (6, i6, c6) (7, i7, c7)),
            9 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0) (1, i1, c1) (2, i2, c2) (3, i3, c3) (4, i4, c4) (5, i5, c5) (6, i6, c6) (7, i7, c7) (8, i8, c8)),
            10 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0) (1, i1, c1) (2, i2, c2) (3, i3, c3) (4, i4, c4) (5, i5, c5) (6, i6, c6) (7, i7, c7) (8, i8, c8) (9, i9, c9)),
            11 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0) (1, i1, c1) (2, i2, c2) (3, i3, c3) (4, i4, c4) (5, i5, c5) (6, i6, c6) (7, i7, c7) (8, i8, c8) (9, i9, c9) (10, i10, c10)),
            12 => lanes!(@run $r, $dummy, $first, $step; (0, i0, c0) (1, i1, c1) (2, i2, c2) (3, i3, c3) (4, i4, c4) (5, i5, c5) (6, i6, c6) (7, i7, c7) (8, i8, c8) (9, i9, c9) (10, i10, c10) (11, i11, c11)),
            _ => unreachable!("1 to MAX_LANES lanes"),
        }
    };
    (@run $r:expr, $dummy:expr, $first:expr, $step:expr; $(($lane:literal, $i:ident, $c:ident))+) => {{
        $(
            let mut $i: u32 = $first($lane);
            let mut $c: Cursor = $lane;
        )+
        // The rounds of a rate are written out, each after the test that every
        // round has, so the prefetch's absence costs nothing in the others.
        'rounds: while !(true $(&& $i == $dummy)+) {
            $( $step(&mut $i, &mut $c, true); )+
            if $r >= 2 {
                if true $(&& $i == $dummy)+ {
                    break 'rounds;
                }
                $( $step(&mut $i, &mut $c, false); )+
            }
            if $r >= 4 {
                if true $(&& $i == $dummy)+ {
                    break 'rounds;
                }
                $( $step(&mut $i, &mut $c, false); )+
                if true $(&& $i == $dummy)+ {
                    break 'rounds;
                }
                $( $step(&mut $i, &mut $c, false); )+
            }
        }
    }};
}

/// One pass over the items `items[..n]`, which are followed by
/// `pad(lanes, D)` items on the dummy, with the lanes of [`lanes_of`] for `K` and
/// `VARYING`.
/// Writes each item's stop node to `stops` (with `VARYING` only) and the node's word
/// 1 to `words`, at the item's position. Items stop at leaves and, with `VARYING`, at
/// varying splits; without it, varying splits are stepped through their constant
/// encoding, as for a single row, whose bins hold every feature. When a lane takes
/// an item, it prefetches the root of the item `D` items later in its sequence; `D`
/// = 0 prefetches nothing.
///
/// A lane loads its next item once the step has moved its cursor, on the chain from
/// its node to its next one. Loading the item after its own at the start of each
/// step instead (an early load) takes it off the chain, but holds one more value per
/// lane through the step. At a lane count that reaches the issue bound
/// ([`crate::rules::lanes`]) the chain does not bind, and an early load is worth only
/// the difference in instructions. At x86-64's 8 lanes (instructions per round of 8
/// lane steps, emeraldrapids, rustc 1.97.1) it saved 3% on the walk without varying
/// stops at distance 0 and at prefetch rates 2 and 4 (180 → 174, 397 → 385) and lost
/// 2% at distance 1 or more, rate 1 (202 → 207; on x86-64-v4 it saved 7%, 213 → 198);
/// it cost 1% on the walks with varying stops over 8-byte items at distance 1 or more
/// (216 → 219), and 7–16% on those at distance 0 (187 → 217 over 8-byte items, 200 →
/// 232 over 16-byte items) and over 16-byte items at any distance (236 → 252).
/// On Emerald Rapids the walks with it ran in 0.985 and 0.983 of the time without it
/// (and `u32` cursors) at the two alignments, but 6 cells were slower at both, exactly
/// those that walk the instantiations it lengthens (the 64-row panels by 2.5–5.2%,
/// their instructions per group up 3.4–3.8% and their IPC unchanged), and the gains
/// came mostly from the `usize` cursors ([`Cursor`]). On the M4, whose 8 or 9 lanes
/// keep the walk bound by throughput as well, it cost 1 or 2 instructions per lane
/// step and spilled the varying walks' lanes.
///
/// # Safety
///
/// `items` holds `n + pad(lanes, D)` items on nodes of `p`, `stops` and `words` have
/// room for `n + lanes` entries, and `p` is valid as [`step`] requires.
#[inline(never)]
pub(super) unsafe fn walk<
    I: Start,
    const K: usize,
    const VARYING: bool,
    const D: usize,
    const R: usize,
    const STATS: bool,
>(
    p: &Pools,
    items: *const I,
    stops: *mut u32,
    words: *mut u64,
    steps: &mut Steps,
) {
    let k: usize = const {
        let k = lanes_of(K, VARYING);
        assert!(k >= 1 && k <= MAX_LANES && D <= MAX_DISTANCE);
        assert!(R == 1 || R == 2 || R == 4);
        k
    };
    // SAFETY: each item is visited once and stops once, and cursors advance by k per
    // stop and stay below n + k, so cursor + D * k is a padding item and every write
    // to stops and words is below n + k.
    unsafe {
        lanes!(
            k,
            R,
            p.dummy,
            |lane: u32| (*items.add(lane as usize)).node(),
            |i: &mut u32, c: &mut Cursor, ahead: bool| {
                if STATS {
                    steps.count(*i != p.dummy);
                }
                let (next, w1) = step(p, *i);
                let stops_at = if VARYING { STOPS_VARYING } else { FLAG_LEAF };
                let stop = w1 & stops_at != 0;
                // The lane's item stops here or later, so its last writes are its stop.
                // Resolve needs the node only to split at a varying stop.
                if VARYING {
                    *stops.add(at(*c)) = *i;
                }
                *words.add(at(*c)) = w1;
                let cursor = *c + k as Cursor * Cursor::from(stop);
                // The lane's next item, taken if it stops here.
                let item = (*items.add(at(cursor))).node();
                if D > 0 && ahead {
                    prefetch(
                        p.nodes
                            .add((*items.add(at(cursor) + D * k)).node() as usize)
                            .cast(),
                    );
                }
                *c = cursor;
                *i = select_unpredictable(stop, item, next);
            }
        );
    }
}

/// Lane steps of the counted walks: on nodes, and in all, dummies included.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Steps {
    pub(super) nodes: u64,
    pub(super) slots: u64,
}

impl Steps {
    #[inline(always)]
    fn count(&mut self, on_node: bool) {
        self.nodes += u64::from(on_node);
        self.slots += 1;
    }
}

/// Hint the cache to load the line at `p`; no effect on results.
#[inline(always)]
pub(super) fn prefetch(p: *const u8) {
    // Miri does not run inline assembly; there the prefetch is a no-op.
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    // SAFETY: a prefetch never faults and has no architectural effect.
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly));
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as above.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p.cast());
    }
    #[cfg(any(miri, not(any(target_arch = "aarch64", target_arch = "x86_64"))))]
    let _ = p;
}
