//! What the precompute needs to build the varying predicates' masks, built at load.
//!
//! Per varying feature, the range of its numeric predicates and its chains that
//! categorical predicates test. A numeric predicate's mask comes from the feature's
//! rows sorted by value, merged with the predicates in threshold order. A categorical
//! predicate tests `rank <= k` on one of the feature's chain slots ([`super`]), so a
//! chain is an ordinal feature too, with small integer values: its predicates' masks
//! come from the rows counted by rank rather than sorted ([`MaskChain`]).

use std::ops::Range;

use super::{BinMap, CategoryBins};
use crate::LoadError;
use crate::model::FeatureRange;

/// A varying categorical predicate's test: heavy when `rank[slot] <= k`, and its left
/// rows are the heavy ones, or with `invert` the others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChainPoint {
    pub(crate) slot: u32,
    pub(crate) k: u32,
    pub(crate) invert: bool,
}

impl ChainPoint {
    /// No test: a numeric predicate's entry.
    pub(crate) const NONE: Self = Self {
        slot: u32::MAX,
        k: 0,
        invert: false,
    };
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

/// A chain without varying predicates, while the plan is built.
const NO_CHAIN: u32 = u32::MAX;

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
        let mut tests: Vec<(u32, u32, u32, bool)> = Vec::new();
        // Per chain of the feature, its entry in `plan.chains` or NO_CHAIN, and its
        // first predicate at or past each rank `0..=k_max`, then `len`, from `from`
        // in `firsts`, with `past = k_max + 1`.
        let mut of_chain: Vec<u32> = Vec::new();
        let mut ranks: Vec<(usize, u32)> = Vec::new();
        let mut firsts: Vec<u32> = Vec::new();
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
                // (chain, k, predicate, invert), in chain then k order.
                tests.clear();
                tests.extend(range.cat.clone().map(|p| {
                    let t = points[p as usize];
                    debug_assert!(
                        t.slot as usize >= c.slot && (t.slot as usize) < c.slot + c.slots()
                    );
                    (t.slot - c.slot as u32, t.k, p, t.invert)
                }));
                tests.sort_unstable();
                of_chain.clear();
                of_chain.resize(c.slots(), NO_CHAIN);
                ranks.clear();
                firsts.clear();
                for run in tests.chunk_by(|a, b| a.0 == b.0) {
                    let at = plan.points.len();
                    plan.points.extend(run.iter().map(|t| (t.2, t.3)));
                    let from = firsts.len();
                    let k_max = run.last().expect("a run").1;
                    let mut j = 0;
                    for rank in 0..=k_max {
                        while run[j].1 < rank {
                            j += 1;
                        }
                        firsts.push(j as u32);
                    }
                    firsts.push(run.len() as u32);
                    let default = bins.defaults[c.slot + run[0].0 as usize].min(k_max + 1);
                    of_chain[run[0].0 as usize] = index(ranks.len())?;
                    ranks.push((from, k_max + 1));
                    plan.chains.push(MaskChain {
                        start: firsts[from + default as usize],
                        points: index(at)?,
                        len: index(run.len())?,
                    });
                }
                // Per code, its exceptions' moves: from each chain's start to the
                // first predicate at or past its rank there.
                let chains = &plan.chains[first..];
                plan.flip_bounds.push(index(plan.flips.len())?);
                for e in 0..c.codes.domain() {
                    for &[slot, rank] in bins.exceptions(c, e as u32) {
                        let m = of_chain[slot as usize - c.slot];
                        if m == NO_CHAIN {
                            continue;
                        }
                        let chain = chains[m as usize];
                        let (from, past) = ranks[m as usize];
                        let j = firsts[from + rank.min(past) as usize];
                        if j != chain.start {
                            for j in [chain.start, j] {
                                if j < chain.len {
                                    plan.flips.push(plan.points[(chain.points + j) as usize].0);
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
