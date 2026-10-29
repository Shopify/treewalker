//! The varying predicates' masks for one group.
//!
//! Per varying feature, the sweep sorts the group's non-missing values with their
//! rows and merges them with the feature's numeric predicates in threshold order,
//! adding rows to one running left mask, so each row is added once. Rows that already
//! arrive in order, as a monotonic feature's do, cost the sort one pass: it stops
//! after finding a single run. Categorical predicates test ranks on the feature's
//! chains, whose values are small integers: each row's code is computed once per
//! feature, every row starts in the mask of the first predicate at or past an
//! unlisted category's rank, each of its code's exceptions moves it to the first
//! predicate at or past its own, and a running union turns those masks into every
//! predicate's rows ([`MaskPlan`]'s chains).
//!
//! Each predicate's mask is written with its complement, its right rows, so that
//! resolve reads the heavy side's mask of a split directly ([`sides`]). Masks carried
//! by value are built in registers ([`Mask`]); runtime-width masks in rows of words
//! ([`Rows`]).

use super::WorkCounters;
use crate::load::layout::{BinMap, MaskChain, MaskFeature, MaskPlan};
use crate::model::{Model, VaryingPredicate, threshold_go_left};
use crate::predict::mask::Mask;

/// A predicate's masks of both sides: its left rows and their complement, which
/// holds its right rows and, past the group's rows, rows that no item carries.
#[inline(always)]
fn sides<M: Mask>(left: M) -> [M; 2] {
    [left, M::ones_if(true).and_not(left)]
}

/// A mask the precompute builds row by row: a [`Mask`] value, or a runtime-width
/// mask's words.
pub(crate) trait Bits {
    /// Clears every row.
    fn clear(&mut self);
    /// Sets row `r`, which the mask holds.
    fn set(&mut self, r: usize);
    /// Flips row `r`, which the mask holds.
    fn flip(&mut self, r: usize);
    /// Takes the rows of `other`.
    fn copy_from(&mut self, other: &Self);
}

impl<M: Mask> Bits for M {
    #[inline(always)]
    fn clear(&mut self) {
        *self = M::ZERO;
    }

    #[inline(always)]
    fn set(&mut self, r: usize) {
        Mask::set(self, r);
    }

    #[inline(always)]
    fn flip(&mut self, r: usize) {
        Mask::flip(self, r);
    }

    #[inline(always)]
    fn copy_from(&mut self, other: &Self) {
        *self = *other;
    }
}

impl Bits for [u64] {
    #[inline(always)]
    fn clear(&mut self) {
        self.fill(0);
    }

    #[inline(always)]
    fn set(&mut self, r: usize) {
        self[r / 64] |= 1 << (r % 64);
    }

    #[inline(always)]
    fn flip(&mut self, r: usize) {
        self[r / 64] ^= 1 << (r % 64);
    }

    #[inline(always)]
    fn copy_from(&mut self, other: &Self) {
        self.copy_from_slice(other);
    }
}

/// Copies of the sweep's running left mask that consecutive rows take turns in, for
/// masks held in memory. The sweep adds rows in value order, which for a column that
/// arrives sorted is row order, so consecutive rows mostly share a word: in one mask,
/// each set's read of the word waits for the previous set's store, 4 to 5 cycles of
/// store-to-load forwarding plus the load, while a set issues in 1 to 2 cycles, so four
/// turns keep the loads off the critical path (as [`crate::exact::WIDE_SUMS`] does for
/// the wide sums). A mask carried by value, 64 rows or fewer, is set in a register and
/// takes one turn.
pub(crate) const SWEEP_TURNS: usize = 4;

/// Whether a feature's sweep takes [`SWEEP_TURNS`] turns rather than one: when the
/// forwarding waits the turns save exceed the work they add.
///
/// Each of the sweep's sets whose row shares a word with the previous set's waits for
/// that store, about one L1 latency, and the turns take that wait off the chain:
/// `consecutive` such sets save as many L1 latencies. In exchange, each of the
/// feature's `predicates` writes reads the other `SWEEP_TURNS - 1` copies of each of
/// the mask's `words`, a load and an OR per word, of which the core issues about four
/// per L1 latency (four or more loads and ALU operations per cycle against a latency
/// of 4 to 5 cycles, on the M4 and current x86 cores). Turns pay when
/// `consecutive ≥ predicates × (SWEEP_TURNS - 1) × words / 4`, a ratio of the core's
/// latency to its width, which the probe does not measure.
///
/// The sweep counts its sets as consecutive when the sorted values' rows are in row
/// order, ascending or descending, as a column that arrives in order leaves them (all
/// sets but one per word, here rounded to all), and otherwise as none: an exact count
/// would cost a pass over the values, which measured more than it saved, while the
/// test of order stops at the first row out of it.
#[inline]
pub(crate) const fn sweep_turns(consecutive: usize, predicates: usize, words: usize) -> bool {
    4 * consecutive >= predicates * (SWEEP_TURNS - 1) * words
}

/// Whether the rows of `sorted` are in row order, ascending or descending.
#[inline]
fn in_row_order(sorted: &[(f64, u32)]) -> bool {
    sorted.windows(2).all(|w| w[0].1 < w[1].1) || sorted.windows(2).all(|w| w[0].1 > w[1].1)
}

/// The sweep's running left mask as up to [`Lefts::TURNS`] copies: the union of the
/// copies a sweep takes turns in is the mask.
pub(crate) trait Lefts<B: Bits + ?Sized> {
    /// Copies held.
    const TURNS: usize;
    /// True if it holds the copies of masks like `mask`, which [`Self::set`] relies on.
    fn holds(&self, mask: &B) -> bool;
    /// Clears the first `T` copies.
    fn clear<const T: usize>(&mut self);
    /// The copies' memory as a mask like `mask`, which the categorical predicates
    /// use as scratch once the numeric ones are written.
    fn scratch(&mut self, mask: &B) -> &mut B;
    /// Sets row `r` in copy `t % T`.
    ///
    /// # Safety
    ///
    /// [`Self::holds`] is true of a mask that holds row `r`, and `T <= TURNS`.
    unsafe fn set<const T: usize>(&mut self, t: usize, r: usize);
}

/// Masks carried in [`Mask`] values: the copies one after another.
impl<M: Mask> Lefts<M> for [M] {
    const TURNS: usize = if M::WIDTH > 64 { SWEEP_TURNS } else { 1 };

    #[inline(always)]
    fn holds(&self, _: &M) -> bool {
        self.len() == Self::TURNS
    }

    #[inline(always)]
    fn clear<const T: usize>(&mut self) {
        self[..T].fill(M::ZERO);
    }

    #[inline(always)]
    fn scratch(&mut self, _: &M) -> &mut M {
        &mut self[0]
    }

    #[inline(always)]
    unsafe fn set<const T: usize>(&mut self, t: usize, r: usize) {
        // SAFETY: the copies hold T masks, and r < M::WIDTH.
        Mask::set(unsafe { self.get_unchecked_mut(t % T) }, r);
    }
}

/// Runtime-width masks: `T` copies interleaved by word, word `i` of copy `t` at
/// `T * i + t`, so that a word's union reads adjacent words, and one copy is a mask.
impl Lefts<[u64]> for [u64] {
    const TURNS: usize = SWEEP_TURNS;

    #[inline(always)]
    fn holds(&self, mask: &[u64]) -> bool {
        self.len() == SWEEP_TURNS * mask.len()
    }

    #[inline(always)]
    fn clear<const T: usize>(&mut self) {
        let words = self.len() / SWEEP_TURNS;
        self[..T * words].fill(0);
    }

    #[inline(always)]
    fn scratch(&mut self, mask: &[u64]) -> &mut [u64] {
        &mut self[..mask.len()]
    }

    #[inline(always)]
    unsafe fn set<const T: usize>(&mut self, t: usize, r: usize) {
        let at = T * (r / 64) + t % T;
        // SAFETY: the mask holds row r, so r / 64 is below its words.
        unsafe { *self.get_unchecked_mut(at) |= 1 << (r % 64) };
    }
}

/// Where the precompute writes each predicate's left and right masks ([`sides`]).
pub(crate) trait Sides<B: Bits + ?Sized> {
    /// The sweep's running left mask.
    type Lefts: Lefts<B> + ?Sized;
    /// Number of predicates.
    fn len(&self) -> usize;
    /// Predicate `p`'s sides, from its left rows `left`.
    fn write(&mut self, p: usize, left: &B);
    /// Predicate `p`'s sides, from its left rows, the union of the first `T` copies in
    /// `lefts`, and, if `default_left`, the rows `missing`.
    fn write_or<const T: usize>(
        &mut self,
        p: usize,
        lefts: &Self::Lefts,
        missing: &B,
        default_left: bool,
    );
    /// Mask words, which a write reads per copy.
    fn words(&self) -> usize;
    /// Predicate `p`'s left mask, to build in place.
    fn left(&mut self, p: usize) -> &mut B;
    /// [`Self::left`] without a bounds check.
    ///
    /// # Safety
    ///
    /// `p` is below [`Self::len`].
    unsafe fn left_unchecked(&mut self, p: usize) -> &mut B;

    /// The sides of one chain's predicates `points`, in increasing k, from their left
    /// masks as the categorical flips leave them, each holding the rows counted first
    /// at it: each predicate's left rows are the union of the masks up to it, or with
    /// its `invert` that union's complement in `all`. One pass, the union carried:
    /// inverting a predicate's rows does not change it.
    fn chain(&mut self, points: &[(u32, bool)], all: &B);
}

impl<M: Mask> Sides<M> for [[M; 2]] {
    type Lefts = [M];

    #[inline(always)]
    fn len(&self) -> usize {
        <[_]>::len(self)
    }

    #[inline(always)]
    fn write(&mut self, p: usize, left: &M) {
        self[p] = sides(*left);
    }

    #[inline(always)]
    fn write_or<const T: usize>(&mut self, p: usize, lefts: &[M], missing: &M, default_left: bool) {
        let left = if T == 1 {
            lefts[0]
        } else {
            let [l0, l1, l2, l3] = lefts[..SWEEP_TURNS] else {
                unreachable!("the sweep's copies")
            };
            l0.or(l1).or(l2.or(l3))
        };
        self[p] = sides(left.or(M::ones_if(default_left).and(*missing)));
    }

    #[inline(always)]
    fn words(&self) -> usize {
        M::WIDTH.div_ceil(64)
    }

    #[inline(always)]
    fn left(&mut self, p: usize) -> &mut M {
        &mut self[p][0]
    }

    #[inline(always)]
    unsafe fn left_unchecked(&mut self, p: usize) -> &mut M {
        debug_assert!(p < <[_]>::len(self));
        // SAFETY: the caller keeps p below the predicates.
        unsafe { &mut self.get_unchecked_mut(p)[0] }
    }

    // Never inlined: once per chain, and inlined it would take registers and stack in
    // masked_group (the u64 instance's frame grew by 16 bytes) for code most groups
    // never run.
    #[inline(never)]
    fn chain(&mut self, points: &[(u32, bool)], all: &M) {
        let mut union = M::ZERO;
        for &(p, invert) in points {
            let p = p as usize;
            union = union.or(self[p][0]);
            self[p] = sides(if invert { all.and_not(union) } else { union });
        }
    }
}

/// Runtime-width masks of `words` words each: predicate `p`'s left mask, then its right
/// mask, at mask `2p` and `2p + 1` of `masks`.
pub(crate) struct Rows<'a> {
    pub(crate) masks: &'a mut [u64],
    pub(crate) words: usize,
}

impl Rows<'_> {
    /// The words of predicate `p`'s left and right masks.
    #[inline(always)]
    fn pair(&mut self, p: usize) -> (&mut [u64], &mut [u64]) {
        let w = self.words;
        self.masks[2 * p * w..(2 * p + 2) * w].split_at_mut(w)
    }
}

impl Sides<[u64]> for Rows<'_> {
    type Lefts = [u64];

    #[inline(always)]
    fn words(&self) -> usize {
        self.words
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.masks.len() / (2 * self.words)
    }

    #[inline(always)]
    fn write(&mut self, p: usize, left: &[u64]) {
        let (l, r) = self.pair(p);
        for ((l, r), &x) in l.iter_mut().zip(r).zip(left) {
            (*l, *r) = (x, !x);
        }
    }

    #[inline(always)]
    fn write_or<const T: usize>(
        &mut self,
        p: usize,
        lefts: &[u64],
        missing: &[u64],
        default_left: bool,
    ) {
        let default = u64::from(default_left).wrapping_neg();
        let (ls, rs) = self.pair(p);
        let copies = lefts.chunks_exact(T);
        for (((l, r), c), &m) in ls.iter_mut().zip(rs).zip(copies).zip(missing) {
            let x = c.iter().fold(default & m, |x, &c| x | c);
            (*l, *r) = (x, !x);
        }
    }

    #[inline(always)]
    fn left(&mut self, p: usize) -> &mut [u64] {
        self.pair(p).0
    }

    #[inline(always)]
    unsafe fn left_unchecked(&mut self, p: usize) -> &mut [u64] {
        let w = self.words;
        debug_assert!((2 * p + 1) * w <= self.masks.len());
        // SAFETY: the caller keeps p below the predicates, whose masks `masks` holds.
        unsafe { self.masks.get_unchecked_mut(2 * p * w..(2 * p + 1) * w) }
    }

    // Never inlined, as for value masks.
    #[inline(never)]
    fn chain(&mut self, points: &[(u32, bool)], all: &[u64]) {
        // Word by word, so that each word's union is carried in a register.
        let w = self.words;
        for (i, &a) in all.iter().enumerate() {
            let mut union = 0;
            for &(p, invert) in points {
                let at = 2 * p as usize * w + i;
                union |= self.masks[at];
                let left = if invert { a & !union } else { union };
                (self.masks[at], self.masks[at + w]) = (left, !left);
            }
        }
    }
}

impl Model {
    /// The left and right masks ([`sides`]) of every varying predicate for one
    /// group, by the sweep, into `out`. `order` and `codes` hold at least a value per
    /// row, `missing` holds every row, and `lefts` holds the running left mask's
    /// copies ([`Lefts`]). With `STATS`,
    /// the work is added to `counters`.
    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the group, its scratch and its masks"
    )]
    pub(crate) fn precompute_sweep<
        const F32: bool,
        B: Bits + ?Sized,
        O: Sides<B> + ?Sized,
        const STATS: bool,
    >(
        &self,
        rows: &[f64],
        order: &mut [(f64, u32)],
        codes: &mut [u32],
        out: &mut O,
        lefts: &mut O::Lefts,
        missing: &mut B,
        counters: &mut WorkCounters,
    ) {
        let plan = &self.masks;
        debug_assert_eq!(out.len(), self.varying_predicates.len());
        let nf = self.config.n_features();
        let n = rows.len() / nf;
        assert!(
            u32::try_from(n).is_ok() && lefts.holds(missing),
            "{n} rows exceed the sweep's row indices or scratch"
        );
        for feature in &plan.features {
            if !feature.num.is_empty() {
                let f = feature.feature;
                missing.clear();
                let mut len = 0;
                for (r, row) in rows.chunks_exact(nf).enumerate() {
                    let v = row[f];
                    if v.is_nan() {
                        missing.set(r);
                    } else {
                        order[len] = (v, r as u32);
                        len += 1;
                    }
                }
                let sorted = &mut order[..len];
                if STATS {
                    counters.precompute_row_evals += n as u64;
                    counters.precompute_mask_writes += feature.num.len() as u64;
                    let compares = &mut counters.precompute_sort_compares;
                    if F32 {
                        sorted.sort_unstable_by(|a, b| {
                            *compares += 1;
                            (a.0 as f32).total_cmp(&(b.0 as f32))
                        });
                    } else {
                        sorted.sort_unstable_by(|a, b| {
                            *compares += 1;
                            a.0.total_cmp(&b.0)
                        });
                    }
                } else if F32 {
                    sorted.sort_unstable_by(|a, b| (a.0 as f32).total_cmp(&(b.0 as f32)));
                } else {
                    sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                }
                let turns = <O::Lefts as Lefts<B>>::TURNS > 1
                    && sweep_turns(len, feature.num.len(), out.words())
                    && in_row_order(sorted);
                if turns {
                    self.merge::<F32, B, O, STATS, SWEEP_TURNS>(
                        feature, sorted, out, lefts, missing, counters,
                    );
                } else {
                    self.merge::<F32, B, O, STATS, 1>(
                        feature, sorted, out, lefts, missing, counters,
                    );
                }
            }
            chain_masks::<F32, B, O, STATS>(
                rows,
                nf,
                plan,
                &self.layout.bins,
                feature,
                &mut codes[..n],
                out,
                lefts.scratch(missing),
                counters,
            );
        }
    }

    /// The sweep's merge of `feature`'s `sorted` values with its numeric predicates,
    /// the running left mask in `T` copies of `lefts`.
    #[inline(always)]
    fn merge<
        const F32: bool,
        B: Bits + ?Sized,
        O: Sides<B> + ?Sized,
        const STATS: bool,
        const T: usize,
    >(
        &self,
        feature: &MaskFeature,
        sorted: &[(f64, u32)],
        out: &mut O,
        lefts: &mut O::Lefts,
        missing: &B,
        counters: &mut WorkCounters,
    ) {
        lefts.clear::<T>();
        let mut next = 0;
        for p in feature.num.clone() {
            let VaryingPredicate::Num {
                threshold,
                default_left,
                ..
            } = self.varying_predicates[p]
            else {
                unreachable!("numeric predicates come first")
            };
            while let Some(&(v, r)) = sorted.get(next) {
                if STATS {
                    counters.precompute_sweep_compares += 1;
                }
                if !threshold_go_left::<F32>(v, threshold.0) {
                    break;
                }
                // SAFETY: the sweep asserts that `lefts` holds copies of masks like
                // `missing`, which holds every row, and T <= TURNS.
                unsafe { lefts.set::<T>(next, r as usize) };
                next += 1;
            }
            out.write_or::<T>(p, lefts, missing, default_left);
        }
    }

    /// The left and right masks of every varying predicate for one group, by brute
    /// force: each predicate evaluated on every row's value, with `left` as the
    /// running mask. [`super::Options::sweep`] off selects it.
    pub(crate) fn precompute_brute_force<
        const F32: bool,
        B: Bits + ?Sized,
        O: Sides<B> + ?Sized,
        const STATS: bool,
    >(
        &self,
        rows: &[f64],
        out: &mut O,
        left: &mut B,
        counters: &mut WorkCounters,
    ) {
        debug_assert_eq!(out.len(), self.varying_predicates.len());
        let nf = self.config.n_features();
        if STATS {
            let n_preds = self.varying_predicates.len() as u64;
            counters.precompute_row_evals += n_preds * (rows.len() / nf) as u64;
            counters.precompute_mask_writes += n_preds;
        }
        for (p, pred) in self.varying_predicates.iter().enumerate() {
            let f = pred.feature() as usize;
            left.clear();
            for (r, row) in rows.chunks_exact(nf).enumerate() {
                if pred.goes_left::<F32>(row[f], &self.categories) {
                    left.set(r);
                }
            }
            out.write(p, left);
        }
    }
}

/// The masks of `feature`'s categorical predicates: each row's code into `codes`,
/// then per chain every row in the mask of the first predicate at or past the
/// default rank, then per row its code's flips, each exception moving the row to
/// the first predicate at or past its own rank, then per chain a running union over
/// its predicates ([`MaskPlan`]'s chains, [`Sides::chain`]). `all` takes all rows,
/// which inverted predicates take the complement in.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the group, the plan and the scratch"
)]
fn chain_masks<const F32: bool, B: Bits + ?Sized, O: Sides<B> + ?Sized, const STATS: bool>(
    rows: &[f64],
    nf: usize,
    plan: &MaskPlan,
    bins: &BinMap,
    feature: &MaskFeature,
    codes: &mut [u32],
    out: &mut O,
    all: &mut B,
    counters: &mut WorkCounters,
) {
    let Some(c) = feature.categories else {
        return;
    };
    let chains = &plan.chains[feature.chains.clone()];
    let points = |chain: &MaskChain| {
        &plan.points[chain.points as usize..(chain.points + chain.len) as usize]
    };
    all.clear();
    for (r, (code, row)) in codes.iter_mut().zip(rows.chunks_exact(nf)).enumerate() {
        *code = c.codes.code::<F32>(&bins.books, row[feature.feature]);
        all.set(r);
    }
    for chain in chains {
        for &(p, _) in points(chain) {
            out.left(p as usize).clear();
        }
        if chain.start < chain.len {
            let (p, _) = plan.points[(chain.points + chain.start) as usize];
            out.left(p as usize).copy_from(all);
        }
    }
    let bounds = &plan.flip_bounds[feature.flips..=feature.flips + c.codes.domain()];
    let mut seen = 0u64;
    // Unchecked: every index below is in range by the plan's construction at load
    // ([`MaskPlan::build`]), which the debug assertions restate. A code's flips lie
    // between its bounds, and name the feature's predicates, which `out` holds; and
    // the rows are the mask's.
    for (r, &code) in codes.iter().enumerate() {
        if STATS {
            seen += bins.exceptions(c, code).len() as u64;
        }
        let code = code as usize;
        debug_assert!(code + 1 < bounds.len());
        // SAFETY: as above; `code` is one of the feature's codes.
        let (from, to) = unsafe {
            (
                *bounds.get_unchecked(code) as usize,
                *bounds.get_unchecked(code + 1) as usize,
            )
        };
        debug_assert!(from <= to && to <= plan.flips.len());
        // SAFETY: as above.
        for &p in unsafe { plan.flips.get_unchecked(from..to) } {
            // SAFETY: as above.
            unsafe { out.left_unchecked(p as usize).flip(r) };
        }
    }
    if STATS {
        let n_points: usize = chains.iter().map(|chain| chain.len as usize).sum();
        counters.precompute_row_evals += codes.len() as u64 + seen;
        counters.precompute_mask_writes += n_points as u64;
    }
    for chain in chains {
        out.chain(points(chain), all);
    }
}

#[cfg(test)]
mod tests {
    //! Edge cases of the precompute on one varying feature, through models loaded
    //! from Treelite JSON so that the loader builds the predicates and their
    //! [`MaskPlan`]. Each case checks the sweep, the brute force and each
    //! predicate's float evaluation against hand-written masks.

    use crate::load::json;
    use crate::model::{Model, Threshold, ThresholdType, VaryingPredicate};
    use crate::predict::mask::{Mask, Wide};
    use crate::{Forest, LoadOptions, ModelFormat, WalkerConfig};

    /// A numeric split's JSON fields.
    fn numeric(op: &str, threshold: f64, default_left: bool) -> (String, bool) {
        (
            format!(
                r#""node_type": "numerical_test_node", "comparison_op": "{op}", "threshold": {threshold:?}"#
            ),
            default_left,
        )
    }

    /// A categorical split's JSON fields: `categories` go left.
    fn categorical(categories: &[u32], default_left: bool) -> (String, bool) {
        (
            format!(
                r#""node_type": "categorical_test_node", "category_list": {categories:?}, "category_list_right_child": false"#
            ),
            default_left,
        )
    }

    /// A model of one varying feature with one tree per split.
    fn forest(splits: &[(String, bool)], f32: bool) -> Forest {
        forest_with(splits, f32, 32)
    }

    /// [`forest`] for groups of up to `rows` rows.
    fn forest_with(splits: &[(String, bool)], f32: bool, rows: usize) -> Forest {
        let trees: Vec<String> = splits
            .iter()
            .map(|(test, default_left)| json::stump(test, *default_left, (1, 5)))
            .collect();
        let json = json::regressor(f32, 1, &trees);
        let config = WalkerConfig::builder(1)
            .max_group_width(rows)
            .varying([0])
            .build()
            .unwrap();
        Forest::from_bytes(
            json.as_bytes(),
            ModelFormat::TreeliteJson,
            config,
            &LoadOptions::default(),
        )
        .unwrap()
    }

    /// The left masks of `out`, checking that each right mask is its complement.
    fn lefts(out: &[[u32; 2]]) -> Vec<u32> {
        out.iter()
            .map(|&[left, right]| {
                assert_eq!(right, !left, "the right mask complements the left");
                left
            })
            .collect()
    }

    fn sweep<const F32: bool>(model: &Model, rows: &[f64]) -> Vec<u32> {
        let n = rows.len();
        let mut out = vec![[0; 2]; model.varying_predicates.len()];
        let (mut order, mut codes) = (vec![(0.0, 0); n], vec![0; n]);
        let (left, mut missing) = (0u32, 0u32);
        model.precompute_sweep::<F32, u32, _, false>(
            rows,
            &mut order,
            &mut codes,
            &mut out[..],
            &mut [left][..],
            &mut missing,
            &mut super::WorkCounters::default(),
        );
        lefts(&out)
    }

    fn brute<const F32: bool>(model: &Model, rows: &[f64]) -> Vec<u32> {
        let mut out = vec![[0; 2]; model.varying_predicates.len()];
        model.precompute_brute_force::<F32, u32, _, false>(
            rows,
            &mut out[..],
            &mut 0u32,
            &mut super::WorkCounters::default(),
        );
        lefts(&out)
    }

    /// The sweep's masks over `column`, checked against the brute force and each
    /// predicate's float evaluation of every row.
    fn masks(forest: &Forest, column: &[f64]) -> Vec<u32> {
        let model = &forest.model;
        let f32 = model.threshold_type == ThresholdType::F32;
        let (swept, brute) = if f32 {
            (sweep::<true>(model, column), brute::<true>(model, column))
        } else {
            (sweep::<false>(model, column), brute::<false>(model, column))
        };
        let reference: Vec<u32> = model
            .varying_predicates
            .iter()
            .map(|p| {
                column.iter().enumerate().fold(0, |m, (r, &x)| {
                    let left = if f32 {
                        p.goes_left::<true>(x, &model.categories)
                    } else {
                        p.goes_left::<false>(x, &model.categories)
                    };
                    m | u32::from(left) << r
                })
            })
            .collect();
        assert_eq!(swept, reference, "the sweep against the float predicates");
        assert_eq!(
            brute, reference,
            "the brute force against the float predicates"
        );
        swept
    }

    /// The mask of the one predicate that `is` picks.
    fn mask_of(forest: &Forest, masks: &[u32], is: impl Fn(&VaryingPredicate) -> bool) -> u32 {
        let found: Vec<usize> = (0..masks.len())
            .filter(|&p| is(&forest.model.varying_predicates[p]))
            .collect();
        assert_eq!(found.len(), 1, "one predicate");
        masks[found[0]]
    }

    fn num(threshold: f64, default_left: bool) -> impl Fn(&VaryingPredicate) -> bool {
        move |p| {
            *p == VaryingPredicate::Num {
                feature: 0,
                threshold: Threshold(threshold),
                default_left,
            }
        }
    }

    /// The Wide<16> sweep's masks of a 1,000-row column, each predicate's left rows.
    fn wide_masks(model: &Model, column: &[f64]) -> Vec<Wide<16>> {
        let n = column.len();
        let mut out = vec![[Wide::<16>::ZERO; 2]; model.varying_predicates.len()];
        let (mut order, mut codes) = (vec![(0.0, 0); n], vec![0; n]);
        let mut lefts = [Wide::<16>::ZERO; super::SWEEP_TURNS];
        let mut missing = Wide::<16>::ZERO;
        model.precompute_sweep::<false, Wide<16>, _, false>(
            column,
            &mut order,
            &mut codes,
            &mut out[..],
            &mut lefts[..],
            &mut missing,
            &mut super::WorkCounters::default(),
        );
        out.iter().map(|&[left, _]| left).collect()
    }

    #[test]
    fn sweeps_in_one_turn_and_in_four_build_the_same_masks() {
        // Few predicates on many rows in row order take four turns; the same rows
        // shuffled take one.
        let splits: Vec<_> = [100.5, 400.5, 999.5]
            .iter()
            .map(|&t| numeric("<=", t, t > 400.0))
            .collect();
        let column: Vec<f64> = (0..1000)
            .map(|r| if r % 97 == 5 { f64::NAN } else { f64::from(r) })
            .collect();
        let rows = column.iter().filter(|v| !v.is_nan()).count();
        assert!(super::sweep_turns(rows, splits.len(), 16));
        let model = &forest_with(&splits, false, 1024).model;
        let four = wide_masks(model, &column);
        // A fixed permutation of the rows: r -> 7r mod 1000 (7 is coprime to 1000).
        let shuffled: Vec<f64> = (0..1000).map(|r| column[7 * r % 1000]).collect();
        let one = wide_masks(model, &shuffled);
        for (p, (a, b)) in four.iter().zip(&one).enumerate() {
            for r in 0..1000 {
                let (x, y) = (7 * r % 1000, r);
                assert_eq!(
                    a.0[x / 64] >> (x % 64) & 1,
                    b.0[y / 64] >> (y % 64) & 1,
                    "predicate {p} row {x}"
                );
            }
            for (r, &v) in column.iter().enumerate() {
                let t = [100.5, 400.5, 999.5][p];
                let left = if v.is_nan() { t > 400.0 } else { v <= t };
                assert_eq!(
                    a.0[r / 64] >> (r % 64) & 1 == 1,
                    left,
                    "predicate {p} row {r}"
                );
            }
        }
    }

    #[test]
    fn an_all_missing_column_goes_the_default_ways() {
        let f = forest(
            &[numeric("<=", 5.0, true), numeric("<=", 10.0, false)],
            false,
        );
        let m = masks(&f, &[f64::NAN; 4]);
        assert_eq!(mask_of(&f, &m, num(5.0, true)), 0b1111);
        assert_eq!(mask_of(&f, &m, num(10.0, false)), 0);
    }

    #[test]
    fn a_single_row_goes_left_of_the_thresholds_above_it() {
        let f = forest(
            &[numeric("<=", 3.0, false), numeric("<=", 7.0, false)],
            false,
        );
        let m = masks(&f, &[5.0]);
        assert_eq!(mask_of(&f, &m, num(3.0, false)), 0);
        assert_eq!(mask_of(&f, &m, num(7.0, false)), 1);
    }

    #[test]
    fn identical_rows_go_together() {
        let f = forest(
            &[numeric("<=", 3.0, false), numeric("<=", 7.0, false)],
            false,
        );
        let m = masks(&f, &[5.0; 8]);
        assert_eq!(mask_of(&f, &m, num(3.0, false)), 0);
        assert_eq!(mask_of(&f, &m, num(7.0, false)), 0xFF);
    }

    #[test]
    fn the_widest_group_of_32_rows_splits_at_its_threshold() {
        let f = forest(&[numeric("<=", 16.0, false)], false);
        let column: Vec<f64> = (0..32).map(f64::from).collect();
        let m = masks(&f, &column);
        assert_eq!(mask_of(&f, &m, num(16.0, false)), (1 << 17) - 1);
        // Rows in reverse take the same masks, mirrored.
        let reversed: Vec<f64> = column.iter().rev().copied().collect();
        let m = masks(&f, &reversed);
        assert_eq!(mask_of(&f, &m, num(16.0, false)), !((1 << 15) - 1));
    }

    #[test]
    fn missing_rows_mix_with_values() {
        let f = forest(
            &[numeric("<=", 2.0, true), numeric("<=", 4.0, false)],
            false,
        );
        let m = masks(&f, &[f64::NAN, 1.0, f64::NAN, 5.0, 3.0]);
        assert_eq!(mask_of(&f, &m, num(2.0, true)), 0b00111);
        assert_eq!(mask_of(&f, &m, num(4.0, false)), 0b10010);
    }

    #[test]
    fn categorical_predicates_need_no_numeric_ones() {
        let f = forest(&[categorical(&[1, 3], false)], false);
        assert!(f.model.masks.features.iter().all(|m| m.num.is_empty()));
        let m = masks(&f, &[0.0, 1.0, 3.0]);
        assert_eq!(m, [0b110]);
    }

    #[test]
    fn float32_thresholds_and_values_round_before_comparing() {
        // Both thresholds round to 1.0f32, so the two splits differ only in their
        // default direction.
        let (t1, t2) = (1.0 + 2f64.powi(-24), 1.0 + 2f64.powi(-25));
        assert_eq!((t1 as f32).to_bits(), (t2 as f32).to_bits());
        let f = forest(&[numeric("<", t1, false), numeric("<", t2, true)], true);
        // 1 - 2^-26 is below 1.0 in float64 but rounds to 1.0f32, not below it; 1 -
        // 2^-23 stays below it.
        let column = [
            1.0,
            1.000_000_15,
            1.0 - 2f64.powi(-26),
            1.0 - 2f64.powi(-23),
            f64::NAN,
        ];
        let m = masks(&f, &column);
        assert_eq!(mask_of(&f, &m, num(1.0, false)), 0b01000);
        assert_eq!(mask_of(&f, &m, num(1.0, true)), 0b11000);
    }

    #[test]
    fn an_all_missing_categorical_column_goes_the_default_ways() {
        let f = forest(
            &[categorical(&[1, 3], true), categorical(&[0, 40], false)],
            false,
        );
        let mut m = masks(&f, &[f64::NAN; 5]);
        m.sort_unstable();
        assert_eq!(m, [0, 0b11111]);
    }

    #[test]
    fn out_of_range_values_are_in_no_category() {
        // Inline codes for categories below 32, pooled codes once a split names 40.
        for (categories, left_values) in [
            (vec![1, 3], vec![1.0, 1.5, 3.0, 3.99]),
            (vec![1, 40], vec![1.0, 1.5, 40.0, 40.7]),
        ] {
            let f = forest(&[categorical(&categories, false)], false);
            let mut column = vec![
                -1.0,
                -0.5,
                0.0,
                -0.0,
                2.0,
                31.0,
                32.0,
                41.0,
                1e10,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ];
            let others = column.len();
            column.extend(&left_values);
            let m = masks(&f, &column);
            let want = ((1u32 << left_values.len()) - 1) << others;
            assert_eq!(m, [want], "{categories:?}");
        }
    }
}
