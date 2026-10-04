//! Prediction: full tree walk and recursive partial evaluation.
//!
//! Three public methods on `Forest`:
//! - `predict` — fast production path (all optimizations, no stats)
//! - `predict_full` — baseline per-row walk (no partial evaluation)
//! - `predict_with_stats` — benchmark path (runtime ablation + stats counters)
//!
//! The hot path is generic over `const F32: bool` (threshold comparison type),
//! `M: RowMask` (u16/u32/u64/`Bits<W>`, chosen from the maximum group width) and
//! `const STATS: bool`. Stats counters and ablation flags exist only in the
//! `predict_with_stats` instantiation (`STATS = true`); `predict` compiles them out.
//! Groups wider than [`MAX_PIECE_ROWS`] run in pieces of that many rows; exact sums
//! make the split invisible.

mod ablation;
mod counters;
mod kernel;

pub use counters::PredictStats;

use crate::config::AblationMode;
use crate::forest::{Forest, ThresholdType};
use crate::mask::{Bits, RowMask};
// ---------------------------------------------------------------------------
// Workspace (private)
// ---------------------------------------------------------------------------

/// Row masks for every varying predicate, in the width chosen at load.
pub(crate) enum Masks {
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    B2(Vec<Bits<2>>),
    B4(Vec<Bits<4>>),
    B8(Vec<Bits<8>>),
    B16(Vec<Bits<16>>),
}

/// Per-group scratch, sized for one piece of a group.
pub(crate) struct Buffers {
    /// One varying feature's values, by row.
    column: Vec<f64>,
    /// Sort buffer for the threshold sweep: (value, row).
    order: Vec<(f64, u16)>,
    prefix_starts: Vec<u16>,
    /// Exact-sum difference array: one entry per row plus one.
    diff: Vec<i128>,
}

pub(crate) struct Workspace {
    masks: Masks,
    buffers: Buffers,
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
    stats: Option<PredictStats>,
    ablation: AblationMode,
    /// The group's rows, row-major, `n_features` values per row.
    rows: &'a [f64],
    n_features: usize,
}

/// Rows per piece of a group: the widest mask is `Bits<16>`. Masks are passed by
/// value down the recursion, so this also bounds its stack use in deep trees.
pub const MAX_PIECE_ROWS: usize = 1024;

/// Row-major rows from width-32 feature columns (test helpers).
#[cfg(any(test, feature = "test-helpers"))]
fn rows_from_columns(cols: &[[f64; 32]; 64], n_rows: usize, n_features: usize) -> Vec<f64> {
    (0..n_rows)
        .flat_map(|r| (0..n_features).map(move |f| cols[f][r]))
        .collect()
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

impl Forest {
    fn check_input(&self, data: &[f64], results: &[f64], start: usize, end: usize, grouped: bool) {
        assert!(
            self.config.structural_key() == self.compiled_config.structural_key(),
            "Forest.config structural fields changed after loading; reload the forest to change feature classification or maximum group width"
        );
        assert!(start <= end, "prediction start exceeds end");
        if grouped {
            assert!(start < end, "group must be nonempty");
            assert!(
                end - start <= self.config.max_group_width,
                "group exceeds configured max_group_width"
            );
        }
        let elements = end
            .checked_mul(self.config.n_features)
            .expect("prediction dimensions overflow");
        assert!(elements <= data.len(), "prediction data is too short");
        assert!(end <= results.len(), "prediction output is too short");
    }

    #[expect(clippy::float_cmp, reason = "exact fast-path identities")]
    fn finalize(&self, values: &mut [f64]) {
        let out = self.output;
        if out.divisor != 1.0 || out.base_score != 0.0 {
            for v in &mut *values {
                *v = *v / out.divisor + out.base_score;
            }
        }
        match out.postprocessor {
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

    // -----------------------------------------------------------------------
    // Workspace management (private)
    // -----------------------------------------------------------------------

    fn ensure_workspace(&mut self) -> &mut Workspace {
        if self.workspace.is_none() {
            let np = self.varying_predicates.len();
            let rows = self.config.max_group_width.min(MAX_PIECE_ROWS);
            let masks = match rows {
                0..=16 => Masks::U16(vec![0; np]),
                17..=32 => Masks::U32(vec![0; np]),
                33..=64 => Masks::U64(vec![0; np]),
                65..=128 => Masks::B2(vec![Bits::ZERO; np]),
                129..=256 => Masks::B4(vec![Bits::ZERO; np]),
                257..=512 => Masks::B8(vec![Bits::ZERO; np]),
                _ => Masks::B16(vec![Bits::ZERO; np]),
            };
            self.workspace = Some(Workspace {
                masks,
                buffers: Buffers {
                    column: vec![0.0; rows],
                    order: vec![(0.0, 0); rows],
                    prefix_starts: vec![0; self.trees.len()],
                    diff: vec![0; rows + 1],
                },
            });
        }
        self.workspace.as_mut().unwrap()
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Baseline: evaluate every tree for every row independently. No partial evaluation.
    pub fn predict_full(&self, data: &[f64], results: &mut [f64], start: usize, end: usize) {
        self.check_input(data, results, start, end, false);
        match self.threshold_type {
            ThresholdType::F64 => self.predict_full_inner::<false>(data, results, start, end),
            ThresholdType::F32 => self.predict_full_inner::<true>(data, results, start, end),
        }
    }

    /// Fast production path. All optimizations enabled, no stats.
    pub fn predict(&mut self, data: &[f64], results: &mut [f64], start: usize, end: usize) {
        self.check_input(data, results, start, end, true);
        self.ensure_workspace();
        self.dispatch::<false>(data, results, start, end, AblationMode::default(), None);
    }

    /// Benchmark path: runtime ablation flags + stats collection.
    pub fn predict_with_stats(
        &mut self,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
    ) -> PredictStats {
        self.check_input(data, results, start, end, true);
        self.ensure_workspace();
        let ablation = self.config.ablation;
        self.dispatch::<true>(
            data,
            results,
            start,
            end,
            ablation,
            Some(PredictStats::default()),
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // Dispatch: (threshold type × mask width), then pieces of the group
    // -----------------------------------------------------------------------

    fn dispatch<const STATS: bool>(
        &mut self,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
        ablation: AblationMode,
        stats: Option<PredictStats>,
    ) -> Option<PredictStats> {
        let mut ws = self.workspace.take().unwrap();
        let Workspace { masks, buffers } = &mut ws;
        macro_rules! run {
            ($m:ty, $masks:expr) => {
                match self.threshold_type {
                    ThresholdType::F64 => self.predict_pieces::<false, $m, STATS>(
                        data, results, start, end, $masks, buffers, ablation, stats,
                    ),
                    ThresholdType::F32 => self.predict_pieces::<true, $m, STATS>(
                        data, results, start, end, $masks, buffers, ablation, stats,
                    ),
                }
            };
        }
        let result = match masks {
            Masks::U16(m) => run!(u16, m),
            Masks::U32(m) => run!(u32, m),
            Masks::U64(m) => run!(u64, m),
            Masks::B2(m) => run!(Bits<2>, m),
            Masks::B4(m) => run!(Bits<4>, m),
            Masks::B8(m) => run!(Bits<8>, m),
            Masks::B16(m) => run!(Bits<16>, m),
        };
        self.workspace = Some(ws);
        result
    }

    /// Predict a group in pieces of at most `M::WIDTH` rows. The pieces share the
    /// group's constant features; each row's prediction does not depend on the split.
    #[expect(
        clippy::too_many_arguments,
        reason = "the group, its workspace and its variant"
    )]
    fn predict_pieces<const F32: bool, M: RowMask, const STATS: bool>(
        &self,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
        masks: &mut [M],
        buffers: &mut Buffers,
        ablation: AblationMode,
        stats: Option<PredictStats>,
    ) -> Option<PredictStats> {
        let mut total: Option<PredictStats> = None;
        let mut s = start;
        while s < end {
            let e = end.min(s + M::WIDTH);
            let piece = self.predict_core::<F32, M, STATS>(
                data, results, s, e, masks, buffers, ablation, stats,
            );
            total = match (total, piece) {
                (Some(mut t), Some(p)) => {
                    t += p;
                    Some(t)
                }
                (t, p) => t.or(p),
            };
            s = e;
        }
        total
    }

    // -----------------------------------------------------------------------
    // Core predict — one function, runtime ablation/stats
    // -----------------------------------------------------------------------

    #[expect(
        clippy::too_many_arguments,
        reason = "the group, its workspace and its variant"
    )]
    fn predict_core<const F32: bool, M: RowMask, const STATS: bool>(
        &self,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
        masks: &mut [M],
        buffers: &mut Buffers,
        ablation: AblationMode,
        stats: Option<PredictStats>,
    ) -> Option<PredictStats> {
        let n = end - start;
        let Buffers {
            column,
            order,
            prefix_starts,
            diff,
        } = buffers;
        debug_assert!(n <= M::WIDTH && n < diff.len());
        let nf = self.config.n_features;
        // SAFETY: check_input asserts that data holds end * nf values, and start < end.
        let const_features = unsafe { data.get_unchecked(start * nf..(start + 1) * nf) };
        // SAFETY: as above.
        let rows = unsafe { data.get_unchecked(start * nf..end * nf) };
        // `predict` runs every optimization; ablation exists only with STATS.
        let ablation = if STATS {
            ablation
        } else {
            AblationMode::default()
        };

        // Precompute varying masks (unless ablation disables it).
        let use_precompute = !ablation.disable_varying_precompute;
        let pred_left_masks: &[M] = if use_precompute {
            if ablation.disable_predicate_sweep {
                self.precompute_bruteforce_generic::<F32, M>(rows, masks);
            } else {
                self.precompute_varying_masks::<F32, M>(rows, column, order, masks);
            }
            masks
        } else {
            &[]
        };

        let scale = self.fixed_scale;
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
            stats: if STATS {
                let precompute_evals = if use_precompute {
                    let n_u64 = n as u64;
                    let num_advances: u64 = self
                        .feature_ranges
                        .iter()
                        .map(|r| if r.num_start < r.num_end { n_u64 } else { 0 })
                        .sum();
                    let cat_evals: u64 = self
                        .feature_ranges
                        .iter()
                        .map(|r| u64::from(r.cat_end - r.cat_start) * n_u64)
                        .sum();
                    num_advances + cat_evals
                } else {
                    0
                };
                stats.map(|mut s| {
                    s.precompute_row_evals = precompute_evals;
                    s
                })
            } else {
                None
            },
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
                    &mut ctx.stats,
                    group,
                    k,
                    prefix_starts,
                );
            }
        }

        for (i, tree) in self.trees.iter().enumerate() {
            ctx.base = tree.node_start as usize;
            let start_idx = prefix_starts[i] as usize;
            self.partial_eval::<F32, M, STATS>(&mut ctx, start_idx, all_mask);
        }

        if let Some(e) = scale {
            let mut acc = 0i128;
            for (out, &d) in ctx.results[start..end].iter_mut().zip(&ctx.diff[..n]) {
                acc += d;
                *out = crate::exact::to_f64(acc, e);
            }
            ctx.diff[..=n].fill(0);
        }
        self.finalize(&mut ctx.results[start..end]);
        ctx.stats
    }

    // -----------------------------------------------------------------------
    // Full walk baseline
    // -----------------------------------------------------------------------

    fn predict_full_inner<const F32: bool>(
        &self,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
    ) {
        let nf = self.config.n_features;
        let nodes = &self.nodes;
        results[start..end].fill(0.0);
        for tree in &self.trees {
            let base = tree.node_start as usize;
            for (j, res) in results[start..end].iter_mut().enumerate() {
                let features = &data[(start + j) * nf..(start + j + 1) * nf];
                let mut local = 0usize;
                while !nodes[base + local].is_leaf() {
                    local = self.step::<F32>(nodes, base + local, local, features);
                }
                *res += nodes[base + local].value;
            }
        }
        self.finalize(&mut results[start..end]);
    }

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    #[cfg(any(test, feature = "test-helpers"))]
    pub fn precompute_bruteforce(
        &self,
        varying_cols: &[[f64; 32]; 64],
        n_rows: usize,
        f32_mode: bool,
    ) -> Vec<u32> {
        assert!(n_rows <= 32, "test helpers take at most 32 rows");
        let mut out = vec![0u32; self.varying_predicates.len()];
        for (i, pred) in self.varying_predicates.iter().enumerate() {
            let f = pred.feature() as usize;
            let col = &varying_cols[f];
            let mut left_mask = 0u32;
            for r in 0..n_rows {
                // SAFETY: n_rows <= 32, the column length, as asserted above.
                let val = unsafe { *col.get_unchecked(r) };
                let goes_left = if f32_mode {
                    pred.goes_left::<true>(val, &self.bitsets)
                } else {
                    pred.goes_left::<false>(val, &self.bitsets)
                };
                if goes_left {
                    left_mask |= 1 << r;
                }
            }
            out[i] = left_mask;
        }
        out
    }

    #[cfg(any(test, feature = "test-helpers"))]
    pub fn precompute_sweep(
        &self,
        varying_cols: &[[f64; 32]; 64],
        n_rows: usize,
        f32_mode: bool,
    ) -> Vec<u32> {
        let rows = rows_from_columns(varying_cols, n_rows, self.config.n_features);
        let mut out = vec![0u32; self.varying_predicates.len()];
        let (mut column, mut order) = (vec![0.0; n_rows], vec![(0.0, 0); n_rows]);
        if f32_mode {
            self.precompute_varying_masks::<true, u32>(&rows, &mut column, &mut order, &mut out);
        } else {
            self.precompute_varying_masks::<false, u32>(&rows, &mut column, &mut order, &mut out);
        }
        out
    }
}
