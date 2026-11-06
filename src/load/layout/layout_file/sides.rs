//! Sparse side equivalence: index exceptions once, never rescan a feature for
//! every predicate. Long chains compare keyed hashes of their code sets instead of
//! exhausting the small-merge budget on repeated visits to one slot.

use super::{ChainPoint, ValidatedBins, check, invalid};
use crate::layout_file::validate::Work;
use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::io;

const fn truth(word: u32, point: ChainPoint) -> bool {
    super::super::takes_heavy(word.wrapping_shl(point.shift), point.bound) ^ point.invert
}

struct Index<'a> {
    defaults: &'a [u32],
    slots: HashMap<u32, usize>,
    entries: Vec<Vec<(u32, u32)>>,
}

impl ValidatedBins<'_> {
    pub(crate) fn check_sides(&self, pairs: &[[ChainPoint; 2]], work: &mut Work) -> io::Result<()> {
        if pairs.is_empty() {
            return Ok(());
        }
        let index = Index::new(self, pairs, work)?;
        let mut pending = Vec::new();
        for &[a, b] in pairs {
            // P comparisons: defaults cover code zero and all codes with no
            // exception in either slot, including unlisted input categories.
            work.spend(1)?;
            check(
                truth(index.defaults[a.slot as usize], a)
                    == truth(index.defaults[b.slot as usize], b),
                "categorical predicate sides",
            )?;
            let (left, right) = (index.entries(a), index.entries(b));
            if work.merge(left.len() + right.len())? {
                index.merge(a, b)?;
            } else {
                pending.push([a, b]);
            }
        }
        index.canonical(&pending, work)
    }
}

impl<'a> Index<'a> {
    fn new(
        bins: &ValidatedBins<'a>,
        pairs: &[[ChainPoint; 2]],
        work: &mut Work,
    ) -> io::Result<Self> {
        // Hash keys are seeded and unpredictable to the file, as in the source loader.
        let mut slots = HashMap::new();
        let mut features = HashMap::new();
        // Two endpoints per pair; only index the slots/features actually needed.
        work.spend(2 * pairs.len())?;
        for point in pairs.iter().flatten() {
            let next = slots.len();
            slots.entry(point.slot).or_insert(next);
            let owner = bins.owner(point.slot as usize)?;
            let c = owner
                .feature
                .categories
                .ok_or_else(|| invalid("categorical point owner"))?;
            features.insert(owner.feature.feature, c);
        }
        let mut entries = vec![Vec::new(); slots.len()];
        for c in features.values().copied() {
            let extent =
                bins.bins.offsets[c.offsets + c.codes.domain()] - bins.bins.offsets[c.offsets];
            // Disjoint per-feature tables: total D <= offsets.len(), E <= the
            // exception count. Each selected feature is scanned exactly once.
            work.spend(c.codes.domain() + extent as usize)?;
            for code in 0..c.codes.domain() {
                for (relative, word) in bins.bins.exceptions(c, code as u32) {
                    let slot = c.slot + relative;
                    if let Some(&at) = slots.get(&(slot as u32)) {
                        entries[at].push((code as u32, word));
                    }
                }
            }
        }
        Ok(Self {
            defaults: &bins.bins.defaults,
            slots,
            entries,
        })
    }

    fn entries(&self, point: ChainPoint) -> &[(u32, u32)] {
        &self.entries[self.slots[&point.slot]]
    }

    fn merge(&self, a: ChainPoint, b: ChainPoint) -> io::Result<()> {
        let (mut left, mut right) = (self.entries(a), self.entries(b));
        while !left.is_empty() || !right.is_empty() {
            let code = left
                .first()
                .map_or(u32::MAX, |e| e.0)
                .min(right.first().map_or(u32::MAX, |e| e.0));
            let word = |entries: &mut &[(u32, u32)], point: ChainPoint| {
                if entries.first().is_some_and(|e| e.0 == code) {
                    let value = entries[0].1;
                    *entries = &entries[1..];
                    value
                } else {
                    self.defaults[point.slot as usize]
                }
            };
            check(
                truth(word(&mut left, a), a) == truth(word(&mut right, b), b),
                "categorical predicate sides",
            )?;
        }
        Ok(())
    }

    /// A slot can hold a long chain, so even slot-only merges may cost P*E.
    /// Batch those queries by (slot, shift). Each field sweeps its exceptions
    /// once, toggling codes at the two endpoints where their truth differs from
    /// the default, and keyed set hashes compare those differences. The sides'
    /// agreement is semantic, single-row against grouped predictions, never an
    /// index the kernel trusts, so a collision (about 2^-128 a comparison) is no
    /// safety risk.
    fn canonical(&self, pairs: &[[ChainPoint; 2]], work: &mut Work) -> io::Result<()> {
        if pairs.is_empty() {
            return Ok(());
        }
        let mut queries: HashMap<(u32, u32), Vec<(u32, usize)>> = HashMap::new();
        for (i, pair) in pairs.iter().enumerate() {
            for (side, point) in pair.iter().enumerate() {
                queries
                    .entry((point.slot, point.shift))
                    .or_default()
                    .push((point.bound, 2 * i + side));
            }
        }
        // Keyed 128-bit set hashes: XOR of code keys is independent of toggle order.
        let state = RandomState::new();
        let mut roots = vec![0; 2 * pairs.len()];
        for ((slot, shift), mut queries) in queries {
            let entries = &self.entries[self.slots[&slot]];
            // At most 32 distinct shifts per slot: <=32E field visits and <=64E
            // toggle events, regardless of predicates or partner slots per field.
            work.spend(entries.len())?;
            let default = self.defaults[slot as usize].wrapping_shl(shift);
            let mut events = Vec::new();
            for &(code, word) in entries {
                let value = word.wrapping_shl(shift);
                if value != default {
                    events.extend([(value.min(default), code), (value.max(default), code)]);
                }
            }
            events.sort_unstable();
            queries.sort_unstable();
            let (mut next, mut root) = (0, 0);
            for (bound, at) in queries {
                while let Some(&(value, code)) = events.get(next) {
                    if !super::super::takes_heavy(value, bound) {
                        break;
                    }
                    work.spend(1)?;
                    root ^= u128::from(state.hash_one((code, 0u8))) << 64
                        | u128::from(state.hash_one((code, 1u8)));
                    next += 1;
                }
                work.spend(1)?; // at most two queries per predicate
                roots[at] = root;
            }
        }
        for pair in roots.chunks_exact(2) {
            check(pair[0] == pair[1], "categorical predicate sides")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_queries_match_two_pointer_merges() {
        let mut slots = HashMap::new();
        slots.insert(0, 0);
        slots.insert(1, 1);
        let index = Index {
            defaults: &[3, 5],
            slots,
            entries: vec![
                vec![(1, 1), (3, 9), (4, u32::MAX)],
                vec![(2, 7), (3, 0), (4, 1)],
            ],
        };
        for shift in 0..4 {
            for a in [0, 1, 3, 5, 8, u32::MAX] {
                for b in [0, 1, 3, 5, 8, u32::MAX] {
                    let left = ChainPoint {
                        slot: 0,
                        shift,
                        bound: a,
                        invert: false,
                    };
                    let right = ChainPoint {
                        slot: 1,
                        shift: 3 - shift,
                        bound: b,
                        invert: true,
                    };
                    if truth(index.defaults[0], left) != truth(index.defaults[1], right) {
                        continue;
                    }
                    let expected = index.merge(left, right).is_ok();
                    let actual = index
                        .canonical(&[[left, right]], &mut Work::new(10000))
                        .is_ok();
                    assert_eq!(actual, expected, "{left:?} {right:?}");
                }
            }
        }
    }

    #[test]
    fn batched_long_chain_queries_share_one_sweep_per_field() {
        let count = 1024u32;
        let mut slots = HashMap::new();
        slots.insert(0, 0);
        slots.insert(1, 1);
        let defaults = [count, 0];
        let index = Index {
            defaults: &defaults,
            slots,
            entries: vec![
                (0..count).map(|r| (r + 1, r)).collect(),
                (0..count).map(|r| (r + 1, count - r)).collect(),
            ],
        };
        let bias = crate::load::layout::BOUND_BIAS;
        let pairs: Vec<_> = (0..count)
            .map(|k| {
                [
                    ChainPoint {
                        slot: 0,
                        shift: 0,
                        bound: k + bias,
                        invert: false,
                    },
                    ChainPoint {
                        slot: 1,
                        shift: 0,
                        bound: count - k - 1 + bias,
                        invert: true,
                    },
                ]
            })
            .collect();
        index.canonical(&pairs, &mut Work::new(10000)).unwrap();
    }
}
