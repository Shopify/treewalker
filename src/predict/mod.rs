//! Prediction: the [`Predictor`] and its call path down to the kernel.
//!
//! A call checks its input once, borrows the model once, and dispatches once on
//! (threshold type × mask width). The monomorphized path then runs every group of the
//! call in pieces of at most `M::WIDTH` rows; groups wider than [`MAX_PIECE_ROWS`]
//! run in pieces of that many rows, and exact sums make the split invisible.
//!
//! The kernel ([`kernel`]) is generic over `const F32: bool` (threshold comparison
//! type), `M: RowMask` (u16/u32/u64/`Bits<W>`, chosen from the maximum group width),
//! `const STATS: bool` (work counters) and `const ABLATE: bool` (runtime ablation
//! flags). It is compiled three ways: production `(false, false)`, with every
//! optimization on and nothing counted; and, with the `research` feature, research
//! timed `(false, true)` and research counted `(true, true)`. With `ABLATE = false`
//! the flags are the defaults at compile time.
//!
//! The helpers the research builds share with production are `#[inline(always)]`:
//! `run`, the input check, the group loop, the piece, the threshold sweep, the prefix
//! starts, the finalization and `Bits::set_bit`. With ordinary inlining, their extra
//! callers changed what LLVM inlined into the production path, so compiling or using
//! the research feature changed production code. `run_unchecked`, `sigmoid_inplace`
//! and smaller helpers keep ordinary inlining, and the language does not guarantee
//! the result: production code was verified instruction-identical with the feature
//! off, compiled in and in use with rustc 1.97.1 and fat LTO, by disassembly on
//! aarch64-apple-darwin with default and 64-byte function alignment. A compiler
//! upgrade should repeat that check.

#![expect(
    clippy::inline_always,
    reason = "keeps the production call path independent of the research builds"
)]

mod ablation;
mod counters;
mod kernel;

#[cfg(feature = "research")]
pub use counters::COUNTERS_VERSION;
pub use counters::WorkCounters;

use std::sync::Arc;

use crate::forest::{Forest, Model, ThresholdType};
use crate::mask::{Bits, RowMask};

/// Runtime ablation flags: each disables one optimization of the predict path.
///
/// Used by the research predictor to measure each optimization's contribution;
/// production prediction is compiled with every optimization on. Parse-time
/// ablations are load options instead, because they change the model's layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ablation {
    /// Fall back to per-row partition functions instead of precomputed varying masks.
    pub disable_varying_precompute: bool,
    /// Use brute-force O(P × n) precompute instead of the sorted-threshold sweep.
    /// Effective only with precompute on (the default).
    pub disable_predicate_sweep: bool,
    /// Always recurse into both children at varying splits.
    pub disable_unsplit: bool,
    /// Treat all monotonic varying features as non-monotonic: no early-break scans.
    /// Effective only with `disable_varying_precompute`, because the precompute
    /// evaluates every predicate the same way whatever its monotonicity.
    pub disable_monotonic: bool,
    /// Add leaf values per row in `f64`, in tree order, instead of summing them
    /// exactly. Effective only when the model supports exact sums.
    pub disable_exact_sums: bool,
}

/// Row masks for every varying predicate, in the width chosen at load.
enum Masks {
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    B2(Vec<Bits<2>>),
    B4(Vec<Bits<4>>),
    B8(Vec<Bits<8>>),
    B16(Vec<Bits<16>>),
}

/// Per-group scratch, sized for one piece of a group.
struct Buffers {
    /// One varying feature's values, by row.
    column: Vec<f64>,
    /// Sort buffer for the threshold sweep: (value, row).
    order: Vec<(f64, u16)>,
    prefix_starts: Vec<u16>,
    /// Exact-sum difference array: one entry per row plus one.
    diff: Vec<i128>,
}

struct Workspace {
    masks: Masks,
    buffers: Buffers,
    /// The variant the last call ran with, so tests can check what each entry point
    /// passes down.
    #[cfg(all(test, feature = "research"))]
    last_variant: Option<Ablation>,
}

#[cfg(all(test, feature = "research"))]
impl Predictor {
    pub(crate) const fn last_variant(&self) -> Option<Ablation> {
        self.workspace.last_variant
    }

    pub(crate) const fn clear_last_variant(&mut self) {
        self.workspace.last_variant = None;
    }
}

impl Workspace {
    fn new(model: &Model) -> Self {
        let np = model.varying_predicates.len();
        let rows = model.config.max_group_width().min(MAX_PIECE_ROWS);
        let masks = match rows {
            0..=16 => Masks::U16(vec![0; np]),
            17..=32 => Masks::U32(vec![0; np]),
            33..=64 => Masks::U64(vec![0; np]),
            65..=128 => Masks::B2(vec![Bits::ZERO; np]),
            129..=256 => Masks::B4(vec![Bits::ZERO; np]),
            257..=512 => Masks::B8(vec![Bits::ZERO; np]),
            _ => Masks::B16(vec![Bits::ZERO; np]),
        };
        Self {
            masks,
            buffers: Buffers {
                column: vec![0.0; rows],
                order: vec![(0.0, 0); rows],
                prefix_starts: vec![0; model.trees.len()],
                diff: vec![0; rows + 1],
            },
            #[cfg(all(test, feature = "research"))]
            last_variant: None,
        }
    }
}

/// Context for recursive `partial_eval` calls.
struct EvalCtx<'a, M: RowMask> {
    base: usize,
    const_features: &'a [f64],
    pred_left_masks: &'a [M],
    results: &'a mut [f64],
    start: usize,
    /// Exact sums: leaves add `value * 2^scale` over runs of rows into this
    /// difference array (one entry per row plus one).
    scale: Option<i32>,
    diff: &'a mut [i128],
    counters: &'a mut WorkCounters,
    ablation: Ablation,
    /// The group's rows, row-major, `n_features` values per row.
    rows: &'a [f64],
    n_features: usize,
}

/// Rows per piece of a group: the widest mask is `Bits<16>`. Masks are passed by
/// value down the recursion, so this also bounds its stack use in deep trees.
pub(crate) const MAX_PIECE_ROWS: usize = 1024;

/// How the rows of one call divide into groups.
#[derive(Clone, Copy)]
pub(crate) enum Groups<'a> {
    /// All rows form one group.
    One,
    /// Consecutive groups of this many rows; the last may be shorter.
    Fixed(usize),
    /// Group `g` is rows `offsets[g]..offsets[g + 1]`.
    Offsets(&'a [usize]),
}

impl Groups<'_> {
    const fn count(self, n_rows: usize) -> usize {
        match self {
            Self::One => (n_rows > 0) as usize,
            Self::Fixed(width) => n_rows.div_ceil(width),
            Self::Offsets(offsets) => offsets.len() - 1,
        }
    }

    /// Rows of group `g < count`. Never overflows: every bound is at most `n_rows`.
    #[inline]
    fn bounds(self, g: usize, n_rows: usize) -> (usize, usize) {
        match self {
            Self::One => (0, n_rows),
            Self::Fixed(width) => {
                let start = g * width;
                (start, start + width.min(n_rows - start))
            }
            Self::Offsets(offsets) => (offsets[g], offsets[g + 1]),
        }
    }
}

#[inline]
fn sigmoid_inplace(slice: &mut [f64]) {
    let (chunks, remainder) = slice.split_at_mut(slice.len() & !1);
    for pair in chunks.chunks_exact_mut(2) {
        let (a, b) = ((-pair[0]).exp(), (-pair[1]).exp());
        pair[0] = 1.0 / (1.0 + a);
        pair[1] = 1.0 / (1.0 + b);
    }
    for v in remainder {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Predicts groups of rows with a [`Forest`]'s model and its own scratch space.
///
/// Create one with [`Forest::predictor`] and reuse it, one per worker thread: creating
/// a predictor allocates the scratch space for the widest group, and calls never
/// allocate. A predictor is `Send`, so it can move to the thread that uses it, and it
/// keeps its model alive.
///
/// Every call takes row-major `f64` input, `n_features` values per row in the trained
/// column order, and writes one output per row. Within a group, constant features
/// must be equal in every row, and each monotonic feature must be monotonic in the
/// declared direction; these contracts are not checked. Empty input is a no-op.
///
/// # Panics
///
/// Each call panics, before writing any output, if the input length is not a multiple
/// of `n_features`, if `out` does not hold one value per row, or if the grouping is
/// invalid as documented on the call. Groups may have any width up to the configured
/// `max_group_width`.
pub struct Predictor {
    pub(crate) model: Arc<Model>,
    workspace: Workspace,
}

impl std::fmt::Debug for Predictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Predictor")
            .field("max_group_width", &self.model.config.max_group_width())
            .finish_non_exhaustive()
    }
}

impl Forest {
    /// Create a [`Predictor`] with scratch space for this model's widest group.
    ///
    /// Predictors are meant to be reused: create one per worker, not one per call.
    #[must_use]
    pub fn predictor(&self) -> Predictor {
        Predictor {
            workspace: Workspace::new(&self.model),
            model: Arc::clone(&self.model),
        }
    }
}

impl Predictor {
    /// Predict one group: every row of `rows` belongs to the same entity.
    ///
    /// # Panics
    ///
    /// If `rows.len()` is not a multiple of `n_features`, if `out.len()` is not the row
    /// count, or if the group has more than `max_group_width` rows.
    #[inline]
    pub fn predict_group(&mut self, rows: &[f64], out: &mut [f64]) {
        self.run::<false, false>(rows, out, Groups::One, Ablation::default(), None);
    }

    /// Predict consecutive groups of varying width: group `g` is rows
    /// `offsets[g]..offsets[g + 1]` of `data`.
    ///
    /// `offsets` starts at 0, ends at the row count and strictly increases, so it holds
    /// one more entry than there are groups: `[0]` for no rows.
    ///
    /// # Panics
    ///
    /// If `data.len()` is not a multiple of `n_features`, if `out.len()` is not the row
    /// count, if `offsets` is empty, does not start at 0, does not end at the row count
    /// or does not strictly increase, or if a group has more than `max_group_width`
    /// rows.
    #[inline]
    pub fn predict_groups(&mut self, data: &[f64], offsets: &[usize], out: &mut [f64]) {
        self.run::<false, false>(
            data,
            out,
            Groups::Offsets(offsets),
            Ablation::default(),
            None,
        );
    }

    /// Predict consecutive groups of `width` rows. The last group is shorter when the
    /// row count is not a multiple of `width`.
    ///
    /// # Panics
    ///
    /// If `data.len()` is not a multiple of `n_features`, if `out.len()` is not the row
    /// count, or if `width` is 0 or more than `max_group_width`.
    #[inline]
    pub fn predict_fixed(&mut self, data: &[f64], width: usize, out: &mut [f64]) {
        self.run::<false, false>(data, out, Groups::Fixed(width), Ablation::default(), None);
    }

    /// Check a call's input, then run its groups.
    #[inline(always)]
    pub(crate) fn run<const STATS: bool, const ABLATE: bool>(
        &mut self,
        data: &[f64],
        out: &mut [f64],
        groups: Groups<'_>,
        variant: Ablation,
        counters: Option<&mut WorkCounters>,
    ) {
        self.check(data, out.len(), groups);
        self.run_unchecked::<STATS, ABLATE>(data, out, groups, variant, counters, true);
    }

    /// Panic unless the call's input and output lengths match the model and the
    /// grouping is valid.
    #[inline(always)]
    pub(crate) fn check(&self, data: &[f64], out_len: usize, groups: Groups<'_>) {
        let config = &self.model.config;
        let (nf, max_width) = (config.n_features(), config.max_group_width());
        assert!(
            data.len().is_multiple_of(nf),
            "input length {} is not a multiple of n_features {nf}",
            data.len()
        );
        let n_rows = data.len() / nf;
        assert!(
            out_len == n_rows,
            "output length {out_len} differs from the row count {n_rows}"
        );
        match groups {
            Groups::One => assert!(
                n_rows <= max_width,
                "group of {n_rows} rows exceeds max_group_width {max_width}"
            ),
            Groups::Fixed(width) => {
                assert!(width > 0, "group width must be positive");
                assert!(
                    width <= max_width,
                    "group width {width} exceeds max_group_width {max_width}"
                );
            }
            Groups::Offsets(offsets) => {
                assert!(offsets.first() == Some(&0), "offsets must start at 0");
                assert!(
                    offsets.last() == Some(&n_rows),
                    "offsets must end at the row count {n_rows}"
                );
                for pair in offsets.windows(2) {
                    assert!(pair[0] < pair[1], "offsets must strictly increase");
                    assert!(
                        pair[1] - pair[0] <= max_width,
                        "group of {} rows exceeds max_group_width {max_width}",
                        pair[1] - pair[0]
                    );
                }
            }
        }
    }

    /// Dispatch once on (threshold type × mask width), then run every group. The input
    /// must have passed [`Self::check`]. Without `finalize`, `out` receives the tree
    /// sums, before averaging, the base score and the link function.
    pub(crate) fn run_unchecked<const STATS: bool, const ABLATE: bool>(
        &mut self,
        data: &[f64],
        out: &mut [f64],
        groups: Groups<'_>,
        variant: Ablation,
        counters: Option<&mut WorkCounters>,
        finalize: bool,
    ) {
        #[cfg(all(test, feature = "research"))]
        {
            self.workspace.last_variant = Some(variant);
        }
        let model: &Model = &self.model;
        let Workspace { masks, buffers, .. } = &mut self.workspace;
        let mut unused = WorkCounters::default();
        let counters = counters.unwrap_or(&mut unused);
        macro_rules! run {
            ($m:ty, $masks:expr) => {
                match model.threshold_type {
                    ThresholdType::F64 => model.predict_groups::<false, $m, STATS, ABLATE>(
                        data, out, groups, $masks, buffers, variant, counters, finalize,
                    ),
                    ThresholdType::F32 => model.predict_groups::<true, $m, STATS, ABLATE>(
                        data, out, groups, $masks, buffers, variant, counters, finalize,
                    ),
                }
            };
        }
        match masks {
            Masks::U16(m) => run!(u16, m),
            Masks::U32(m) => run!(u32, m),
            Masks::U64(m) => run!(u64, m),
            Masks::B2(m) => run!(Bits<2>, m),
            Masks::B4(m) => run!(Bits<4>, m),
            Masks::B8(m) => run!(Bits<8>, m),
            Masks::B16(m) => run!(Bits<16>, m),
        }
    }
}

impl Model {
    #[expect(clippy::float_cmp, reason = "exact fast-path identities")]
    #[inline(always)]
    pub(crate) fn apply_margin(&self, values: &mut [f64]) {
        let out = self.output;
        if out.divisor != 1.0 || out.base_score != 0.0 {
            for v in &mut *values {
                *v = *v / out.divisor + out.base_score;
            }
        }
    }

    #[expect(clippy::float_cmp, reason = "exact fast-path identity")]
    #[inline(always)]
    pub(crate) fn apply_link(&self, values: &mut [f64]) {
        match self.output.postprocessor {
            crate::parser::Postprocessor::Identity => {}
            crate::parser::Postprocessor::Sigmoid(alpha) => {
                if alpha != 1.0 {
                    for v in &mut *values {
                        *v *= alpha;
                    }
                }
                sigmoid_inplace(values);
            }
        }
    }

    /// Turn tree sums into outputs: the staged finalization, then the link function.
    #[inline]
    fn finalize(&self, values: &mut [f64]) {
        self.apply_margin(values);
        self.apply_link(values);
    }

    /// Run every group of a call, each in pieces of at most `M::WIDTH` rows. The
    /// pieces share the group's constant features; each row's prediction does not
    /// depend on the split.
    #[expect(
        clippy::too_many_arguments,
        reason = "the call, its workspace and its variant"
    )]
    #[inline(always)]
    fn predict_groups<const F32: bool, M: RowMask, const STATS: bool, const ABLATE: bool>(
        &self,
        data: &[f64],
        results: &mut [f64],
        groups: Groups<'_>,
        masks: &mut [M],
        buffers: &mut Buffers,
        variant: Ablation,
        counters: &mut WorkCounters,
        finalize: bool,
    ) {
        let n_rows = results.len();
        for g in 0..groups.count(n_rows) {
            let (start, end) = groups.bounds(g, n_rows);
            let mut s = start;
            while s < end {
                let e = end.min(s + M::WIDTH);
                self.predict_core::<F32, M, STATS, ABLATE>(
                    data, results, s, e, masks, buffers, variant, counters,
                );
                s = e;
            }
            if finalize {
                self.finalize(&mut results[start..end]);
            }
        }
    }

    /// Tree sums of one piece, rows `start..end` of `data`, into `results[start..end]`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the piece, its workspace and its variant"
    )]
    #[inline(always)]
    fn predict_core<const F32: bool, M: RowMask, const STATS: bool, const ABLATE: bool>(
        &self,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
        masks: &mut [M],
        buffers: &mut Buffers,
        variant: Ablation,
        counters: &mut WorkCounters,
    ) {
        let n = end - start;
        let Buffers {
            column,
            order,
            prefix_starts,
            diff,
        } = buffers;
        debug_assert!(n <= M::WIDTH && n < diff.len());
        let nf = self.config.n_features();
        // SAFETY: Predictor::check asserts that data holds n_features values per
        // output row, and groups give start < end <= results.len().
        let const_features = unsafe { data.get_unchecked(start * nf..(start + 1) * nf) };
        // SAFETY: as above.
        let rows = unsafe { data.get_unchecked(start * nf..end * nf) };
        // Production runs every optimization; the flags exist only with ABLATE.
        let ablation = if ABLATE { variant } else { Ablation::default() };

        // Precompute varying masks (unless ablation disables it).
        let use_precompute = !ablation.disable_varying_precompute;
        let pred_left_masks: &[M] = if use_precompute {
            if ablation.disable_predicate_sweep {
                self.precompute_bruteforce_generic::<F32, M, STATS>(rows, masks, counters);
            } else {
                self.precompute_varying_masks::<F32, M, STATS>(
                    rows, column, order, masks, counters,
                );
            }
            masks
        } else {
            &[]
        };

        let scale = if ablation.disable_exact_sums {
            None
        } else {
            self.fixed_scale
        };
        if scale.is_none() {
            results[start..end].fill(0.0);
        }
        let all_mask: M = M::from_width(n);

        let mut ctx = EvalCtx::<M> {
            base: 0,
            const_features,
            pred_left_masks,
            results,
            start,
            scale,
            diff,
            counters,
            ablation,
            rows,
            n_features: nf,
        };

        if !self.prefix_groups.is_empty() {
            let k = self.prefix_depth;
            prefix_starts.fill(0);
            for group in &self.prefix_groups {
                self.precompute_prefix_starts::<F32, STATS>(
                    const_features,
                    ctx.counters,
                    group,
                    k,
                    prefix_starts,
                );
            }
        }

        for (i, tree) in self.trees.iter().enumerate() {
            ctx.base = tree.node_start as usize;
            let start_idx = prefix_starts[i] as usize;
            self.partial_eval::<F32, M, STATS, ABLATE>(&mut ctx, start_idx, all_mask);
        }

        if let Some(e) = scale {
            let mut acc = 0i128;
            for (out, &d) in ctx.results[start..end].iter_mut().zip(&ctx.diff[..n]) {
                acc += d;
                *out = crate::exact::to_f64(acc, e);
            }
            ctx.diff[..=n].fill(0);
        }
    }
}
