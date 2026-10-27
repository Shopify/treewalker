//! One chunk's walk and resolve passes ([`Pass`]), and how a group's passes run their
//! walks ([`Walks`]): the lanes and prefetch distance each walk is compiled for.

use super::counters::{Clock, Stats};
use super::isa::Isa;
use super::kernel::Options;
use super::mask::Mask;
use super::resolve::{Carrier, LISTED_SLACK, Resolve, Splits, compress};
use super::scan::Scan;
use super::scratch::Queues;
use super::walk::{self, Item, MAX_PAD, Pools, Steps, lanes_of, pad};
use crate::rules::{Distance, pass_distance};

/// How a group's passes run their walks: with the rule's lanes for each walk, or a
/// research ablation's, a prefetch distance from `rules::DISTANCES` and, for the walk
/// without varying stops, a rate from `rules::RATES`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Walks {
    pub(super) lanes: Option<usize>,
    pub(super) distance: usize,
    pub(super) rate: usize,
}

impl Walks {
    /// The walks of `options` with the rule's `distance` and `rate`. Forced lanes
    /// prefetch one item ahead every round, as before the rules, unless the distance
    /// or the rate is forced too.
    pub(super) fn of<const ABLATE: bool>(options: Options, rule: Distance) -> Self {
        if ABLATE {
            let forced = options.lanes.is_some();
            let rate = usize::from(rule.rate);
            Self {
                lanes: options.lanes,
                distance: options
                    .distance
                    .unwrap_or(if forced { 1 } else { rule.items }),
                rate: options.rate.unwrap_or(if forced { 1 } else { rate }),
            }
        } else {
            Self {
                lanes: None,
                distance: rule.items,
                rate: usize::from(rule.rate),
            }
        }
    }

    /// The walks of a pass of `n` items, with the distance of
    /// [`crate::rules::pass_distance`], and the padding such a walk reads past the
    /// items; `varying` and `bits` as for [`walk::walk`].
    pub(super) fn for_pass<const ABLATE: bool>(
        self,
        n: usize,
        varying: bool,
        bits: bool,
    ) -> (Self, usize) {
        let forced = if ABLATE { self.lanes } else { None };
        // The rule's lanes, evaluated at compile time. Called at run time, the const fn
        // is left to LLVM, which did not fold it at x86-64's dispatched base level:
        // there the rule's loop ran every pass, and `pass_distance` divided by a lane
        // count it could not see.
        let lanes = match (forced.unwrap_or(0), varying, bits) {
            (0, true, false) => const { lanes_of(0, true, false) },
            (0, true, true) => const { lanes_of(0, true, true) },
            (0, false, false) => const { lanes_of(0, false, false) },
            (0, false, true) => const { lanes_of(0, false, true) },
            (k, _, _) => k,
        };
        let wanted = match forced {
            Some(_) => self.distance.min(1),
            None => self.distance,
        };
        let distance = pass_distance(wanted, n, lanes);
        (
            Self {
                lanes: self.lanes,
                distance,
                rate: self.rate,
            },
            pad(lanes, distance),
        )
    }
}

/// Run `$call` with `K`, `D`, `R` and `B` as consts: the walk's lanes (0 for the
/// rule's), its prefetch distance, rounded up to a compiled one, its rate (1 without
/// prefetch), and whether it shifts for bit splits, `$bits`. Production, without
/// `$ablate`, never instantiates forced lanes.
macro_rules! walk_with {
    (@forced $lanes:ident, $d:ident, $($l:literal)+) => {
        match $lanes {
            $($l => if $d == 0 { go!($l, 0, 1) } else { go!($l, 1, 1) },)+
            _ => unreachable!("lanes are checked at creation"),
        }
    };
    (@rate $r:expr, $dd:literal) => {
        match $r {
            1 => go!(0, $dd, 1),
            2 => go!(0, $dd, 2),
            _ => go!(0, $dd, 4),
        }
    };
    ($ablate:expr, $walks:expr, $bits:expr, |$k:ident, $d:ident, $rr:ident, $b:ident| $call:expr) => {{
        macro_rules! go {
            ($kk:literal, $dd:literal, $r:literal) => {{
                const $k: usize = $kk;
                const $d: usize = $dd;
                const $rr: usize = $r;
                if $bits {
                    const $b: bool = true;
                    $call
                } else {
                    const $b: bool = false;
                    $call
                }
            }};
        }
        let w: Walks = $walks;
        match (if $ablate { w.lanes } else { None }, w.distance) {
            (None, 0) => go!(0, 0, 1),
            (None, 1) => walk_with!(@rate w.rate, 1),
            (None, 2) => walk_with!(@rate w.rate, 2),
            (None, 3..=4) => walk_with!(@rate w.rate, 4),
            (None, 5..=8) => walk_with!(@rate w.rate, 8),
            (None, _) => walk_with!(@rate w.rate, 16),
            // Forced lanes, research only: 1 to MAX_LANES, without prefetch or one
            // item ahead every round.
            (Some(lanes), d) => walk_with!(@forced lanes, d, 1 2 3 4 5 6 7 8 9 10 11 12),
        }
    }};
}

pub(super) use walk_with;

/// One chunk's walk and resolve passes.
pub(super) struct Pass<'a, M> {
    pub(super) p: &'a Pools,
    pub(super) roots: &'a [u32],
    /// The group's rows.
    pub(super) n: usize,
    /// Units of `M` per mask ([`Carrier`]).
    pub(super) units: usize,
    pub(super) masks: &'a [M],
    /// A mask's room for scanned sides.
    pub(super) scratch: *mut M,
    pub(super) stats: &'a mut Stats,
}

impl<M: Mask> Pass<'_, M> {
    /// [`Self::run`] at level `X` ([`Isa`]): where the build dispatches, out of line,
    /// with the body compiled inside the level's trampoline; elsewhere inlined.
    ///
    /// # Safety
    ///
    /// As for [`Self::run`].
    #[cfg_attr(not(dispatch), inline(always))]
    #[cfg_attr(dispatch, inline(never))]
    pub(super) unsafe fn run_at<
        X: Isa,
        C: Carrier<M>,
        const STATS: bool,
        const ABLATE: bool,
        const SCAN: bool,
        const EMPTY: bool,
    >(
        &mut self,
        walks: Walks,
        arena: &mut [M],
        queues: &mut Queues<C>,
        stops: &mut [u32],
        words: &mut [u64],
        scan: &mut impl Scan<M>,
    ) -> (usize, usize) {
        if X::NATIVE {
            // SAFETY: the caller's, which `run` shares.
            return unsafe {
                self.run::<X, C, STATS, ABLATE, SCAN, EMPTY>(
                    walks, arena, queues, stops, words, scan,
                )
            };
        }
        X::with7(
            #[inline(always)]
            |pass: &mut Self, walks, arena, queues, stops, words, scan| {
                // SAFETY: as above.
                unsafe {
                    pass.run::<X, C, STATS, ABLATE, SCAN, EMPTY>(
                        walks, arena, queues, stops, words, scan,
                    )
                }
            },
            self,
            walks,
            arena,
            queues,
            stops,
            words,
            scan,
        )
    }

    /// Every walk and resolve pass of the chunk; returns the number of leaf entries in
    /// `q.leaves` and how many of them, first, the first pass wrote. With `SCAN`,
    /// varying events scan their rows with `scan` instead of reading the precomputed
    /// masks; with `EMPTY`, they pass on sides without rows.
    ///
    /// # Safety
    ///
    /// The scratch holds the chunk as [`Model::masked_group`](crate::model::Model::masked_group) asserts, `masks` the all-rows
    /// mask and every predicate's left and right masks, `scratch` a mask, and `p` is
    /// valid for [`walk::walk`].
    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline(never))]
    pub(super) unsafe fn run<
        X: Isa,
        C: Carrier<M>,
        const STATS: bool,
        const ABLATE: bool,
        const SCAN: bool,
        const EMPTY: bool,
    >(
        &mut self,
        walks: Walks,
        arena: &mut [M],
        queues: &mut Queues<C>,
        stops: &mut [u32],
        words: &mut [u64],
        scan: &mut impl Scan<M>,
    ) -> (usize, usize) {
        let (p, u) = (self.p, self.units);
        // Listing fixed-width handles' varying stops may write LISTED_SLACK entries
        // past the last leaf entry and position: the scratch gives both that many more
        // than the chunk's leaves plus one, which the items' length bounds with the most
        // padding (`Model::masked_group` asserts it).
        if C::ARENA && !C::RUNTIME && compress::<X>() {
            let room = queues.items.len() - MAX_PAD + LISTED_SLACK;
            assert!(
                queues.leaves.len() >= room && queues.listed.len() >= room,
                "listing slack"
            );
        }
        // SAFETY: the caller sizes the arena for the chunk's slots, slot 0 included.
        let root = unsafe { C::root(arena.as_mut_ptr(), self.n, u) };
        for (item, &node) in queues.items.iter_mut().zip(self.roots) {
            // SAFETY: `item` is a live item.
            unsafe { put(item, node, root) };
        }
        let (mut items, mut next) = (queues.items.as_mut_ptr(), queues.next.as_mut_ptr());
        let (stops, words) = (stops.as_mut_ptr(), words.as_mut_ptr());
        let mut res = Resolve {
            stops: stops.cast_const(),
            words: words.cast_const(),
            masks: self.masks.as_ptr(),
            arena: arena.as_mut_ptr(),
            scratch: self.scratch,
            units: u,
            next_slot: C::slot_units(u),
            items,
            out: next,
            leaves: queues.leaves.as_mut_ptr(),
            n_leaves: 0,
            listed: queues.listed.as_mut_ptr(),
        };
        // Leaf entries of the first pass carry all rows.
        let mut first_pass = None;
        let mut n = self.roots.len();
        let dummy = Item {
            node: p.dummy,
            carry: C::NONE,
        };
        let (mut steps, mut seen) = (Steps::default(), Splits::default());
        let mut clock = Clock::<STATS>::start();
        while n > 0 {
            let (walks, pad) = walks.for_pass::<ABLATE>(n, true, p.bits);
            // SAFETY: the caller sizes both queues for a pass's items plus the most
            // padding, the stops likewise, and the arena for the chunk's splits.
            unsafe {
                for k in 0..pad {
                    put(items.add(n + k), dummy.node, dummy.carry);
                }
                // The walk with varying stops prefetches every round
                // (`rules::RATES`): a rate of 1 here leaves one walk per distance.
                let walks = Walks { rate: 1, ..walks };
                walk_with!(ABLATE, walks, p.bits, |K, D, R, B| walk::walk_at::<
                    X,
                    Item<C>,
                    K,
                    true,
                    D,
                    R,
                    STATS,
                    B,
                >(
                    p, items, stops, words, &mut steps
                ));
                clock = clock.lap(&mut self.stats.phases.walk);
                res.items = items;
                res.out = next;
                let leaves_before = res.n_leaves;
                let items_in_pass = n;
                n = super::resolve::resolve::<X, M, C, SCAN, EMPTY, STATS>(
                    &mut res,
                    items_in_pass,
                    scan,
                    &mut seen,
                );
                clock = clock.lap(&mut self.stats.phases.resolve);
                first_pass.get_or_insert(res.n_leaves);
                if STATS {
                    let c = &mut self.stats.counters;
                    let leaf_events = (res.n_leaves - leaves_before) as u64;
                    c.passes += 1;
                    c.leaf_events += leaf_events;
                    c.varying_events += items_in_pass as u64 - leaf_events;
                }
            }
            std::mem::swap(&mut items, &mut next);
        }
        if STATS {
            let c = &mut self.stats.counters;
            c.walk_steps += steps.nodes;
            c.lane_slots += steps.slots;
            c.splits += seen.splits;
            c.empty_sides += seen.empty_sides;
            if C::ARENA {
                c.arena_slots += (res.next_slot / C::slot_units(u) - 1) as u64;
            }
        }
        (res.n_leaves, first_pass.unwrap_or(0))
    }
}

/// `*to = Item { node, carry }`, on x86-64 as one 8-byte store for a 4-byte carrier:
/// LLVM otherwise writes the node and the (constant) carry with two 4-byte stores
/// there. AArch64 already pairs them (`stp`, or vector zips), which the 8-byte value
/// would undo.
///
/// # Safety
///
/// `to` is valid for writes, and a 4-byte `C` has no padding (every carrier).
#[inline(always)]
unsafe fn put<C: Copy>(to: *mut Item<C>, node: u32, carry: C) {
    if cfg!(target_arch = "x86_64") && size_of::<C>() == 4 {
        // SAFETY: the caller guarantees `to` and that `C` has no padding (the 4-byte
        // carriers are `u32` and its newtypes `Handle` and `Slot`), so `Item<C>`
        // (`repr(C)`) is the node's 4 bytes then the carry's.
        unsafe {
            let carry = u64::from(std::mem::transmute_copy::<C, u32>(&carry));
            let node = u64::from(node);
            // x86-64 is little-endian: the node is the low half.
            to.cast::<u64>().write_unaligned(node | carry << 32);
        }
    } else {
        // SAFETY: the caller guarantees `to`.
        unsafe { to.write(Item { node, carry }) };
    }
}
