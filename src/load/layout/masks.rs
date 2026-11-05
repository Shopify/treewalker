//! What the precompute needs to build the varying predicates' masks, built at load.
//!
//! Per varying feature, the range of its numeric predicates and its chains that
//! categorical predicates test. A numeric predicate's mask comes from the feature's
//! rows sorted by value, merged with the predicates in threshold order. A categorical
//! predicate tests `rank <= k` on a field of one of the feature's slots ([`super`]),
//! so a chain is an ordinal feature too, with small integer values: its predicates'
//! masks come from the rows counted by rank rather than sorted ([`MaskChain`]).

use std::ops::Range;

use super::{BinMap, CategoryBins, takes_heavy};
use crate::LoadError;
use crate::model::FeatureRange;

/// A varying categorical predicate's test, its node's: heavy when its slot's word,
/// shifted left by `shift`, meets `bound`, its field's rank at most its k; and its
/// left rows are the heavy ones, or with `invert` the others. A word's fields have
/// distinct shifts, so the slot and the shift name the chain, and the bound orders
/// its predicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChainPoint {
    pub(crate) slot: u32,
    pub(crate) shift: u32,
    pub(crate) bound: u32,
    pub(crate) invert: bool,
}

impl ChainPoint {
    /// No test: a numeric predicate's entry.
    pub(crate) const NONE: Self = Self {
        slot: u32::MAX,
        shift: 0,
        bound: 0,
        invert: false,
    };
}

/// Whether a row whose slot holds `word` is heavy at a test of `shift` and `bound`.
const fn heavy(word: u32, shift: u32, bound: u32) -> bool {
    takes_heavy(word.wrapping_shl(shift), bound)
}

/// One varying feature's predicates.
#[derive(Clone, Debug)]
pub(crate) struct MaskFeature {
    pub(crate) feature: usize,
    /// Its numeric predicates' IDs, in increasing threshold order.
    pub(crate) num: Range<usize>,
    /// Its chains that categorical predicates test, in [`MaskPlan::chains`].
    pub(crate) chains: Range<usize>,
    /// Its codes' flip bounds, `domain + 1` entries from here in
    /// [`MaskPlan::flip_bounds`], if it has categorical predicates.
    pub(crate) flips: usize,
    /// Its categorical slots, if it has categorical predicates.
    pub(crate) categories: Option<CategoryBins>,
}

/// One chain's categorical predicates, as positions along the chain.
///
/// A row of rank r is a left row of the predicates at k ≥ r (inverted ones aside).
/// With the predicates in increasing k, each row is counted once, in the mask of the
/// first predicate at or past its rank, and a running union over the predicates in
/// order then turns those masks into each predicate's rows: one union per predicate,
/// rather than a sort of the rows or a test of every row per predicate. Most rows
/// take the default rank, no category's, so that first predicate's mask starts with
/// every row, `start`, and a row whose code has an exception in the chain moves from
/// it to its own: two bit flips per exception, nothing per row and chain. Which
/// masks a code's exceptions flip depends only on the code, so the plan lists them
/// per code ([`MaskPlan::flips`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MaskChain {
    /// Its first predicate at or past the default rank, `len` if none is.
    pub(crate) start: u32,
    /// Its `len` predicates in increasing k, from `points` in [`MaskPlan::points`].
    pub(crate) points: u32,
    pub(crate) len: u32,
}

/// Per varying feature, its predicates; per chain, its categorical predicates; per
/// code of a varying categorical feature, the masks its rows flip.
#[derive(Clone, Debug)]
pub(crate) struct MaskPlan {
    pub(crate) features: Vec<MaskFeature>,
    pub(crate) chains: Vec<MaskChain>,
    /// Each chain's predicates and whether each one's left rows are inverted.
    pub(crate) points: Vec<(u32, bool)>,
    /// Per varying categorical feature, per code, the predicates whose left masks a
    /// row of the code flips, from every row in its chains' `start` masks: code `e`'s
    /// at `flip_bounds[e]..flip_bounds[e + 1]` from its feature's `flips`. The
    /// precompute reads them once per row, so they take two loads and a flip per
    /// predicate, rather than a walk through the chains per exception. Each
    /// exception in a chain with predicates gives at most two, so they take at most
    /// 4 bytes per code and 8 per exception, under
    /// [`crate::LoadOptions::max_rank_table_bytes`] with the rank tables.
    pub(crate) flip_bounds: Vec<u32>,
    pub(crate) flips: Vec<u32>,
}

impl MaskPlan {
    /// The plan for the predicates indexed by `ranges` as the parser leaves them,
    /// with the categorical slots of `bins` and each categorical predicate's test in
    /// `points`. The flip lists and `bins`' rank tables take at most `max_bytes`.
    pub(crate) fn build(
        bins: &BinMap,
        ranges: &[FeatureRange],
        points: &[ChainPoint],
        max_bytes: u64,
    ) -> Result<Self, LoadError> {
        let mut plan = Self {
            features: Vec::with_capacity(ranges.len()),
            chains: Vec::new(),
            points: Vec::new(),
            flip_bounds: Vec::new(),
            flips: Vec::new(),
        };
        let index = |i: usize| {
            u32::try_from(i)
                .map_err(|_| LoadError::Limit("the categorical masks' plan exceeds u32".into()))
        };
        let rank_bytes = bins.rank_table_bytes() as u64;
        // (slot, shift, bound, predicate), in chain then k order; per chain of the
        // feature, its first test there, and per slot its chains.
        let mut tests: Vec<(u32, u32, u32, u32)> = Vec::new();
        let mut first_test: Vec<u32> = Vec::new();
        let mut of_slot: Vec<Range<u32>> = Vec::new();
        // The first predicate of a chain's tests at which a word is heavy, the first
        // at or past its rank.
        let first_heavy = |run: &[(u32, u32, u32, u32)], word: u32| {
            run.partition_point(|t| !heavy(word, t.1, t.2)) as u32
        };
        for range in ranges {
            let f = range.feature as usize;
            let num = range.num.start as usize..range.num.end as usize;
            let first = plan.chains.len();
            let flips = plan.flip_bounds.len();
            let categories = if range.cat.is_empty() {
                None
            } else {
                bins.varying_categories(f)
            };
            if let Some(c) = categories {
                tests.clear();
                tests.extend(range.cat.clone().map(|p| {
                    let t = points[p as usize];
                    debug_assert!(
                        t.slot as usize >= c.slot && (t.slot as usize) < c.slot + c.slots()
                    );
                    (t.slot - c.slot as u32, t.shift, t.bound, p)
                }));
                tests.sort_unstable();
                first_test.clear();
                of_slot.clear();
                of_slot.resize(c.slots(), 0..0);
                let mut from = 0;
                for run in tests.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)) {
                    let at = plan.points.len();
                    let local = index(plan.chains.len() - first)?;
                    first_test.push(index(from)?);
                    from += run.len();
                    plan.points
                        .extend(run.iter().map(|t| (t.3, points[t.3 as usize].invert)));
                    let slot = run[0].0 as usize;
                    if of_slot[slot].is_empty() {
                        of_slot[slot] = local..local;
                    }
                    of_slot[slot].end = local + 1;
                    plan.chains.push(MaskChain {
                        start: first_heavy(run, bins.defaults[c.slot + slot]),
                        points: index(at)?,
                        len: index(run.len())?,
                    });
                }
                // Per code, its exceptions' moves: from each chain's start to the
                // first predicate its word is heavy at, for the chains with
                // predicates in each of its words.
                let chains = &plan.chains[first..];
                plan.flip_bounds.push(index(plan.flips.len())?);
                for e in 0..c.codes.domain() {
                    for &[slot, word] in bins.exceptions(c, e as u32) {
                        for m in of_slot[slot as usize - c.slot].clone() {
                            let chain = chains[m as usize];
                            let from = first_test[m as usize] as usize;
                            let j = first_heavy(&tests[from..from + chain.len as usize], word);
                            if j != chain.start {
                                for j in [chain.start, j] {
                                    if j < chain.len {
                                        plan.flips.push(plan.points[(chain.points + j) as usize].0);
                                    }
                                }
                            }
                        }
                    }
                    plan.flip_bounds.push(index(plan.flips.len())?);
                    let bytes = size_of::<u32>() * (plan.flip_bounds.len() + plan.flips.len());
                    if rank_bytes + bytes as u64 > max_bytes {
                        return Err(LoadError::Limit(format!(
                            "the categorical rank tables and masks' flips exceed {max_bytes} \
                             bytes (LoadOptions::max_rank_table_bytes)"
                        )));
                    }
                }
            }
            plan.features.push(MaskFeature {
                feature: f,
                num,
                chains: first..plan.chains.len(),
                flips,
                categories,
            });
        }
        Ok(plan)
    }
}
