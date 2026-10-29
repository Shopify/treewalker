//! The restart prefetch distance: how many items ahead a lane prefetches its next roots,
//! from the latency of the level that holds the nodes.

use std::sync::OnceLock;

use super::machine::{self, Caches, Latency};

/// Prefetch distances the walk is compiled for: none, then powers of two. A
/// distance between two is rounded up, as prefetching earlier only costs L1 room.
pub(crate) const DISTANCES: [usize; 6] = [0, 1, 2, 4, 8, 16];

/// The longest distance: with the most lanes of 8 that two-thirds of the targets
/// pick, 16 items ahead hold 8 × 16 lines of 64 bytes, a quarter of the smallest L1
/// assumed (32 KiB), so prefetched lines wait in L1 without evicting the walk's own.
pub(crate) const MAX_DISTANCE: usize = 16;

/// The distance a pass of `n` items walks with, for a latency-covering distance
/// `need` and `lanes` lanes: at most half of each lane's `n / lanes` items, rounded
/// down to a compiled distance.
///
/// A lane that prefetches `d` items ahead never prefetches its first `d` items, and
/// its last `d` prefetches fall past the pass, on the dummy. Over its `m` items, with
/// latency `L` and time per item `t`, it stalls about `d L` on its first items and
/// `(m − d) max(0, L − d t)` on the rest; while `d t < L` that sum falls as `d` grows
/// to `m / 2` and rises after, so the best distance is the smaller of `L / t`
/// (`need`) and `m / 2`. Passes after a group's first carry a few items each: a
/// distance fixed by latency alone would leave them with no useful prefetch (16 items
/// ahead cost the 1,000-tree panels 5–8% of their time against 1 or 4).
pub(crate) fn pass_distance(need: usize, n: usize, lanes: usize) -> usize {
    let most = (n / lanes / 2).min(need);
    DISTANCES
        .iter()
        .rev()
        .copied()
        .find(|&d| d <= most)
        .unwrap_or(0)
}

/// Prefetch rates the walk without varying stops is compiled for: a prefetch every 1,
/// 2 or 4 rounds.
///
/// The walk with varying stops prefetches every round, so the rule applies to the
/// walk without them only. Its items run to the next varying split or leaf and are
/// short: 37–80% of the panels' items take one step, so [`rate`] picked a rate above 1
/// for that walk on 1 of the 98 benchmark cells (at 1, 8 and 128 ns per level, the
/// M4 Pro's), for about 1%. Carrying a rate into the varying walk's pass
/// (`Pass::run`) cost more: on the M4 its frame grew by 16 bytes (the rate is a
/// second value live across the pass loop) and it dispatched on the rate every
/// pass, which slowed a 500-tree panel 0.9% and 3.5% at default and 64-byte
/// function alignment. On x86-64 that walk at rates 2 and 4 also reloads invariants
/// from the stack, which cancels the saved prefetches.
pub(crate) const RATES: [usize; 3] = [1, 2, 4];

/// The prefetch's share of a lane step's instructions in the walk without varying
/// stops, per target: its three instructions (the load of the item ahead, the node
/// address, the prefetch) over the instructions of a lane step at D > 0 and rate 1,
/// counting the round's exit test. Counted from the post-LTO assembly of
/// `examples/walk_asm.rs` (rustc 1.97.1), not tuned:
///
/// - AArch64 (apple-m4, 8 lanes): 3 of 16.
/// - x86-64 (8 lanes): 24 of the 219 instructions of a round of 8 lane steps on
///   emeraldrapids (223 on x86-64-v4; 3 of 23 at 2 lanes); the round at 8 lanes also
///   reloads and stores its spilled values.
#[cfg(target_arch = "x86_64")]
const PREFETCH_SHARE: f64 = 24.0 / 219.0;
#[cfg(not(target_arch = "x86_64"))]
const PREFETCH_SHARE: f64 = 3.0 / 16.0;

/// The rate of a walk without varying stops whose items take `steps` steps on
/// average, `short[j]` of them j + 1 steps, prefetching from a level `latency` L1
/// latencies away: one of [`RATES`].
///
/// A lane at rate r prefetches in one round of every r. An item that takes s steps
/// holds the lane's cursor for s consecutive rounds, and the lane prefetches for the
/// cursor only in those, so the item ahead gets its prefetch if s ≥ r, else with
/// probability s / r: a share `m(r) = Σ_{s<r} short[s−1] (1 − s/r)` of the items go
/// without, each stalling a lane for the latency less the L1's, `latency − 1` L1
/// latencies, which at the bound of two per step ([`distance`]) is `(latency − 1) / 2`
/// steps. Against that, a lane issues `steps (1 − 1/r)` fewer prefetches per item,
/// each `share` of a step. The rate is the one with the largest saving net of the
/// stalls; at a tie, the smaller. The bound on a step's time is low, so the stalls
/// are counted long, which errs toward prefetching every round.
pub(crate) fn rate(steps: f64, short: &[f64; super::steps::SHORT], latency: f64) -> usize {
    let share = PREFETCH_SHARE;
    let stall = (latency - 1.0).max(0.0) / 2.0;
    let net = |r: usize| {
        let r = r as f64;
        let missed: f64 = (1..=short.len())
            .map(|s| short[s - 1] * (1.0 - s as f64 / r).max(0.0))
            .sum();
        missed.mul_add(-stall, steps.max(1.0) * (1.0 - 1.0 / r) * share)
    };
    RATES
        .iter()
        .copied()
        .fold((1, 0.0), |(best, at), r| {
            let v = net(r);
            if v > at { (r, v) } else { (best, at) }
        })
        .0
}

/// A restart prefetch distance and what it came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Distance {
    /// Items ahead of a lane's current item whose root a lane prefetches; 0 for none.
    pub(crate) items: usize,
    /// Rounds per prefetch ([`rate`]), one of [`RATES`]. A byte, which shares the
    /// level's word, so that a `Distance` keeps its 24 bytes and the predictor's
    /// scratch, which holds two, its layout.
    pub(crate) rate: u8,
    /// The cache level that holds the nodes, or one past the last for memory.
    pub(crate) level: u8,
    /// That level's latency in L1 latencies.
    pub(crate) latency: f64,
}

/// The restart prefetch distance for a walk of `lanes` lanes whose items take
/// `steps` lane steps on average, over a node pool of `pool_bytes` whose nodes are
/// `nodes`, which the probe walks when they pass the last cache, keeping the result
/// in `cache` ([`machine::latency`]); and the walk's prefetch [`rate`], for a walk
/// without varying stops, given its items' lengths in `rated` (`short` of them
/// short), else 1.
///
/// When a lane takes item i, it prefetches the root of the item it will take
/// `items` items later. The lane walks those items first, `steps` steps each, and a
/// step is at least two dependent L1 loads (the node, then its slot's bin; the next
/// node's address depends on both), so the prefetch has at least
/// `items × steps × 2` L1 latencies to land. The distance is the fewest items for
/// which that covers the latency of the level holding the pool: the first cache
/// whose share per CPU holds it, else memory. A pool within L1 needs no prefetch.
/// Since the bound on a step's time is low (a step takes three to four times as
/// long when lanes contend for the core), the distance errs long, which only keeps
/// lines in L1 longer: the prefetches in flight at a time are `lanes × latency /
/// time per item` whatever the distance. Capped at [`MAX_DISTANCE`] and so that the
/// lanes' lines ahead take at most a quarter of L1.
///
/// At rate r a lane prefetches up to r − 1 rounds after it takes an item, so the
/// items ahead cover the latency plus those r − 1 steps.
pub(crate) fn distance(
    pool_bytes: usize,
    nodes: &[crate::load::layout::LayoutNode],
    cache: &OnceLock<Latency>,
    steps: f64,
    rated: Option<&[f64; super::steps::SHORT]>,
    lanes: usize,
    caches: &Caches,
) -> Distance {
    let level = caches
        .levels
        .iter()
        .find(|c| pool_bytes <= c.per_cpu())
        .map_or_else(
            || caches.levels.last().map_or(2, |c| c.level) + 1,
            |c| c.level,
        );
    if level == 1 {
        return Distance {
            items: 0,
            rate: 1,
            level,
            latency: 1.0,
        };
    }
    let latency = machine::latency(level, nodes, cache).l1s;
    let rate = rated.map_or(1, |short| rate(steps, short, latency));
    let per_item = 2.0 * steps.max(1.0);
    let need = (2.0f64.mul_add((rate - 1) as f64, latency) / per_item)
        .ceil()
        .max(1.0) as usize;
    let l1 = caches
        .level(1)
        .map_or(machine::ASSUMED[0].bytes, |c| c.bytes);
    let room = (l1 / 4 / (lanes * caches.line)).max(1);
    let cap = MAX_DISTANCE.min(1 << room.ilog2());
    Distance {
        items: need.next_power_of_two().min(cap),
        rate: rate as u8,
        level,
        latency,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances_round_up_and_cap() {
        let caches = Caches {
            line: 64,
            levels: machine::ASSUMED.to_vec(),
            source: machine::Source::Assumed,
        };
        let none = OnceLock::new();
        let items = Some(&[0.0; 3]);
        assert_eq!(
            distance(1 << 10, &[], &none, 5.0, items, 8, &caches).items,
            0
        );
        let d = distance(100 << 10, &[], &none, 5.0, items, 8, &caches);
        assert_eq!(d.level, 2);
        assert!(DISTANCES.contains(&d.items) && d.items >= 1);
        let far = distance(1 << 30, &[], &none, 1.0, items, 2, &caches);
        assert!(far.items <= MAX_DISTANCE && far.level == 3);
    }

    #[test]
    fn rates_trade_prefetches_for_coverage() {
        // Long items and no short ones: the fewest prefetches.
        assert_eq!(rate(7.0, &[0.0; 3], 8.0), 4);
        // Every item a single step: a rate above 1 loses every other prefetch.
        assert_eq!(rate(1.0, &[1.0, 0.0, 0.0], 8.0), 1);
        // Items of 3 steps: rate 2 covers them all, rate 4 misses a quarter.
        assert_eq!(rate(3.0, &[0.0, 0.0, 1.0], 8.0), 2);
        // Within L1 there is nothing to stall on.
        assert_eq!(rate(1.0, &[1.0, 0.0, 0.0], 1.0), 4);
    }
}
