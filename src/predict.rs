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

use crate::config::AblationMode;
use crate::forest::{
    Forest, Node, PrefixGroup, SPLIT_MONO_DEC, SPLIT_MONO_INC, ThresholdType, VaryingPredicate,
    threshold_go_left,
};
use crate::mask::{Bits, RowMask};

// ---------------------------------------------------------------------------
// PredictStats
// ---------------------------------------------------------------------------

/// Node visit counters for algorithmic analysis.
#[derive(Clone, Copy, Default)]
pub struct PredictStats {
    pub constant_steps: u64,
    pub varying_splits: u64,
    pub unsplit_skips: u64,
    pub recursive_calls: u64,
    pub leaf_hits: u64,
    pub partition_row_evals: u64,
    pub precompute_row_evals: u64,
}

impl std::fmt::Display for PredictStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "const={}, varying={}, unsplit={}, recurse={}, leaf={}, row_evals={}, precompute_evals={}",
            self.constant_steps,
            self.varying_splits,
            self.unsplit_skips,
            self.recursive_calls,
            self.leaf_hits,
            self.partition_row_evals,
            self.precompute_row_evals,
        )
    }
}

impl std::ops::AddAssign for PredictStats {
    fn add_assign(&mut self, rhs: Self) {
        self.constant_steps += rhs.constant_steps;
        self.varying_splits += rhs.varying_splits;
        self.unsplit_skips += rhs.unsplit_skips;
        self.recursive_calls += rhs.recursive_calls;
        self.leaf_hits += rhs.leaf_hits;
        self.partition_row_evals += rhs.partition_row_evals;
        self.precompute_row_evals += rhs.precompute_row_evals;
    }
}

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

    #[allow(clippy::float_cmp)] // Exact fast-path identities.
    pub(crate) fn finalize(&self, values: &mut [f64]) {
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
    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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
        let const_features = unsafe { data.get_unchecked(start * nf..(start + 1) * nf) };
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
    // Prefix group evaluation (mask-agnostic)
    // -----------------------------------------------------------------------

    pub(crate) fn precompute_prefix_starts<const F32: bool, const STATS: bool>(
        &self,
        const_features: &[f64],
        stats: &mut Option<PredictStats>,
        group: &PrefixGroup,
        k: usize,
        prefix_starts: &mut [u16],
    ) {
        let rep_base = group.node_base as usize;
        let nodes = &self.nodes;
        let mut bail_level = k;
        for j in 0..k {
            let node = unsafe { nodes.get_unchecked(rep_base + j) };
            if STATS && let Some(s) = stats {
                s.constant_steps += 1;
            }
            let go_left = self.eval_split::<F32>(node, const_features);
            if go_left != node.heavy_is_left() {
                bail_level = j;
                break;
            }
        }
        for &tree_idx in &group.trees {
            let start_idx = if bail_level == k {
                k as u16
            } else {
                let tree_base = self.trees[tree_idx as usize].node_start as usize;
                unsafe { nodes.get_unchecked(tree_base + bail_level) }.skip as u16
            };
            prefix_starts[tree_idx as usize] = start_idx;
        }
    }

    // -----------------------------------------------------------------------
    // Node step / eval
    // -----------------------------------------------------------------------

    #[inline]
    pub(crate) fn step<const F32: bool>(
        &self,
        nodes: &[Node],
        abs_idx: usize,
        local_idx: usize,
        features: &[f64],
    ) -> usize {
        let node = &nodes[abs_idx];
        if self.eval_split::<F32>(node, features) == node.heavy_is_left() {
            local_idx + 1
        } else {
            node.skip as usize
        }
    }

    #[inline]
    pub(crate) fn eval_split<const F32: bool>(&self, node: &Node, features: &[f64]) -> bool {
        let val = unsafe { *features.get_unchecked(node.feature as usize) };
        if val.is_nan() {
            node.default_left()
        } else if node.is_categorical() {
            let val = if F32 { f64::from(val as f32) } else { val };
            val >= 0.0 && self.cat_test(node, val as i32, node.default_left())
        } else {
            threshold_go_left::<F32>(val, node.value)
        }
    }

    // -----------------------------------------------------------------------
    // Partial eval — ablation and stats only when STATS
    // -----------------------------------------------------------------------

    fn partial_eval<const F32: bool, M: RowMask, const STATS: bool>(
        &self,
        ctx: &mut EvalCtx<M>,
        mut idx: usize,
        mut row_mask: M,
    ) {
        if row_mask.is_zero() {
            return;
        }
        let nodes = self.nodes.as_slice();
        let base = ctx.base;
        let ablation = if STATS {
            ctx.ablation
        } else {
            AblationMode::default()
        };
        let use_precompute = !ablation.disable_varying_precompute;
        let use_unsplit = !ablation.disable_unsplit;
        let use_mono = !ablation.disable_monotonic || use_precompute;

        loop {
            // Constant walk — single bit test per node.
            while unsafe { nodes.get_unchecked(base + idx) }.is_walkable() {
                if STATS && let Some(ref mut s) = ctx.stats {
                    s.constant_steps += 1;
                }
                idx = self.step::<F32>(nodes, base + idx, idx, ctx.const_features);
            }

            let node = unsafe { nodes.get_unchecked(base + idx) };

            // Leaf (walked past all constant nodes, could be leaf or varying).
            if node.is_leaf() {
                if STATS && let Some(ref mut s) = ctx.stats {
                    s.leaf_hits += 1;
                }
                if let Some(e) = ctx.scale {
                    // diff has n + 1 entries; runs lie within rows 0..n.
                    let x = crate::exact::to_fixed(node.value, e);
                    let diff = &mut *ctx.diff;
                    row_mask
                        .run_starts()
                        .for_each_bit(|a| unsafe { *diff.get_unchecked_mut(a) += x });
                    row_mask
                        .run_ends()
                        .for_each_bit(|b| unsafe { *diff.get_unchecked_mut(b + 1) -= x });
                } else {
                    let val = node.value;
                    let results = &mut ctx.results[ctx.start..];
                    row_mask.for_each_bit(|r| unsafe { *results.get_unchecked_mut(r) += val });
                }
                return;
            }

            // Varying split.
            if STATS && let Some(ref mut s) = ctx.stats {
                s.varying_splits += 1;
            }

            let (left_mask, right_mask) = if use_precompute {
                let pred_id = node.varying_pred_id as usize;
                let pred_left_mask = unsafe { *ctx.pred_left_masks.get_unchecked(pred_id) };
                (row_mask & pred_left_mask, row_mask & !pred_left_mask)
            } else {
                let (rows, nf, feat) = (ctx.rows, ctx.n_features, node.feature as usize);
                let col = |r: usize| rows[r * nf + feat];
                let result = if node.is_categorical() {
                    self.partition_per_row::<F32, M>(node, rows, nf, row_mask)
                } else if use_mono {
                    match node.varying_type() {
                        SPLIT_MONO_INC => Self::partition_mono_inc::<F32, M>(node, col, row_mask),
                        SPLIT_MONO_DEC => Self::partition_mono_dec::<F32, M>(node, col, row_mask),
                        _ => Self::partition_non_mono::<F32, M>(node, col, row_mask),
                    }
                } else {
                    Self::partition_non_mono::<F32, M>(node, col, row_mask)
                };
                if STATS && let Some(ref mut s) = ctx.stats {
                    s.partition_row_evals += u64::from(row_mask.count_ones());
                }
                result
            };

            let (heavy_idx, light_idx) = (idx + 1, node.skip as usize);
            let (heavy_mask, light_mask) = if node.heavy_is_left() {
                (left_mask, right_mask)
            } else {
                (right_mask, left_mask)
            };

            // Unsplit optimization.
            if use_unsplit {
                if light_mask.is_zero() {
                    if STATS && let Some(ref mut s) = ctx.stats {
                        s.unsplit_skips += 1;
                    }
                    idx = heavy_idx;
                    continue;
                }
                if heavy_mask.is_zero() {
                    if STATS && let Some(ref mut s) = ctx.stats {
                        s.unsplit_skips += 1;
                    }
                    idx = light_idx;
                    continue;
                }
            }

            if STATS && let Some(ref mut s) = ctx.stats {
                s.recursive_calls += 1;
            }
            self.partial_eval::<F32, M, STATS>(ctx, light_idx, light_mask);
            idx = heavy_idx;
            row_mask = heavy_mask;
        }
    }

    // -----------------------------------------------------------------------
    // Predicate precompute — sorted-threshold sweep
    // -----------------------------------------------------------------------

    /// Left masks of every varying predicate for one piece: per feature, sort the
    /// rows by value once and sweep the feature's sorted thresholds.
    #[inline]
    fn precompute_varying_masks<const F32: bool, M: RowMask>(
        &self,
        rows: &[f64],
        column: &mut [f64],
        order: &mut [(f64, u16)],
        out_left_masks: &mut [M],
    ) {
        debug_assert_eq!(out_left_masks.len(), self.varying_predicates.len());
        let nf = self.config.n_features;
        let column = &mut column[..rows.len() / nf];

        for range in &self.feature_ranges {
            let f = range.feature as usize;
            for (v, row) in column.iter_mut().zip(rows.chunks_exact(nf)) {
                *v = row[f];
            }

            if range.num_start < range.num_end {
                let mut nan_mask = M::ZERO;
                let mut n_non_nan = 0usize;
                for (r, &val) in column.iter().enumerate() {
                    if val.is_nan() {
                        nan_mask = nan_mask.set_bit(r);
                    } else {
                        order[n_non_nan] = (val, r as u16);
                        n_non_nan += 1;
                    }
                }

                let sorted = &mut order[..n_non_nan];
                if F32 {
                    #[allow(clippy::cast_possible_truncation)]
                    sorted.sort_unstable_by(|a, b| (a.0 as f32).total_cmp(&(b.0 as f32)));
                } else {
                    sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                }

                let mut row_ptr = 0usize;
                let mut non_nan_left = M::ZERO;
                let preds =
                    &self.varying_predicates[range.num_start as usize..range.num_end as usize];
                let masks = &mut out_left_masks[range.num_start as usize..range.num_end as usize];

                for (out, pred) in masks.iter_mut().zip(preds.iter()) {
                    let (threshold, default_left) = match pred {
                        VaryingPredicate::Num {
                            threshold,
                            default_left,
                            ..
                        } => (*threshold, *default_left),
                        _ => unreachable!(),
                    };
                    while row_ptr < n_non_nan {
                        let (val, r) = unsafe { *sorted.get_unchecked(row_ptr) };
                        if threshold_go_left::<F32>(val, threshold) {
                            non_nan_left = non_nan_left.set_bit(usize::from(r));
                            row_ptr += 1;
                        } else {
                            break;
                        }
                    }
                    *out = non_nan_left | if default_left { nan_mask } else { M::ZERO };
                }
            }

            let cat_preds =
                &self.varying_predicates[range.cat_start as usize..range.cat_end as usize];
            let cat_masks = &mut out_left_masks[range.cat_start as usize..range.cat_end as usize];
            for (out, pred) in cat_masks.iter_mut().zip(cat_preds.iter()) {
                let mut left_mask = M::ZERO;
                for (r, &val) in column.iter().enumerate() {
                    if pred.goes_left::<F32>(val, &self.bitsets) {
                        left_mask = left_mask.set_bit(r);
                    }
                }
                *out = left_mask;
            }
        }
    }

    /// Brute-force O(P × n) precompute — evaluates each predicate against every row.
    /// Used as the ablation baseline when `disable_predicate_sweep` is set.
    fn precompute_bruteforce_generic<const F32: bool, M: RowMask>(
        &self,
        rows: &[f64],
        out_left_masks: &mut [M],
    ) {
        debug_assert_eq!(out_left_masks.len(), self.varying_predicates.len());
        let nf = self.config.n_features;
        for (out, pred) in out_left_masks
            .iter_mut()
            .zip(self.varying_predicates.iter())
        {
            let f = pred.feature() as usize;
            let mut left_mask = M::ZERO;
            for (r, row) in rows.chunks_exact(nf).enumerate() {
                if pred.goes_left::<F32>(row[f], &self.bitsets) {
                    left_mask = left_mask.set_bit(r);
                }
            }
            *out = left_mask;
        }
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
        let mut out = vec![0u32; self.varying_predicates.len()];
        for (i, pred) in self.varying_predicates.iter().enumerate() {
            let f = pred.feature() as usize;
            let col = &varying_cols[f];
            let mut left_mask = 0u32;
            for r in 0..n_rows {
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

    // -----------------------------------------------------------------------
    // Partition functions — generic over M: RowMask
    // -----------------------------------------------------------------------

    #[inline]
    fn partition_mono_inc<const F32: bool, M: RowMask>(
        node: &Node,
        col: impl Fn(usize) -> f64,
        row_mask: M,
    ) -> (M, M) {
        let thresh = node.value;
        let default_left = node.default_left();
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            if col(r).is_nan() {
                if default_left {
                    left = left.set_bit(r);
                }
                m = m.clear_lowest();
            } else if threshold_go_left::<F32>(col(r), thresh) {
                left = left.set_bit(r);
                m = m.clear_lowest();
            } else {
                if default_left {
                    m = m.clear_lowest();
                    while !m.is_zero() {
                        let r2 = m.trailing_zeros() as usize;
                        if col(r2).is_nan() {
                            left = left.set_bit(r2);
                        }
                        m = m.clear_lowest();
                    }
                }
                break;
            }
        }
        (left, row_mask & !left)
    }

    #[inline]
    fn partition_mono_dec<const F32: bool, M: RowMask>(
        node: &Node,
        col: impl Fn(usize) -> f64,
        row_mask: M,
    ) -> (M, M) {
        let thresh = node.value;
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            if col(r).is_nan() {
                if node.default_left() {
                    left = left.set_bit(r);
                }
                m = m.clear_lowest();
            } else if threshold_go_left::<F32>(col(r), thresh) {
                left = left.set_bit(r);
                m = m.clear_lowest();
                while !m.is_zero() {
                    let r2 = m.trailing_zeros() as usize;
                    if col(r2).is_nan() {
                        if node.default_left() {
                            left = left.set_bit(r2);
                        }
                    } else {
                        left = left.set_bit(r2);
                    }
                    m = m.clear_lowest();
                }
                break;
            } else {
                m = m.clear_lowest();
            }
        }
        (left, row_mask & !left)
    }

    #[inline]
    fn partition_non_mono<const F32: bool, M: RowMask>(
        node: &Node,
        col: impl Fn(usize) -> f64,
        row_mask: M,
    ) -> (M, M) {
        let thresh = node.value;
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            m = m.clear_lowest();
            if col(r).is_nan() {
                if node.default_left() {
                    left = left.set_bit(r);
                }
            } else if threshold_go_left::<F32>(col(r), thresh) {
                left = left.set_bit(r);
            }
        }
        (left, row_mask & !left)
    }

    #[inline]
    fn partition_per_row<const F32: bool, M: RowMask>(
        &self,
        node: &Node,
        rows: &[f64],
        n_features: usize,
        row_mask: M,
    ) -> (M, M) {
        let mut left = M::ZERO;
        let mut m = row_mask;
        while !m.is_zero() {
            let r = m.trailing_zeros() as usize;
            m = m.clear_lowest();
            if self.eval_split::<F32>(node, &rows[r * n_features..(r + 1) * n_features]) {
                left = left.set_bit(r);
            }
        }
        (left, row_mask & !left)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AblationMode, WalkerConfig};
    use crate::forest::{FeatureRange, Node, VaryingPredicate};

    fn num_node(threshold: f64, default_left: bool) -> Node {
        Node {
            value: threshold,
            skip: -1,
            varying_pred_id: u16::MAX,
            feature: 0,
            flags: u8::from(default_left),
            cat_n_words: 0,
        }
    }

    #[test]
    fn test_partition_mono_inc_nan_after_threshold() {
        let node = num_node(3.0, true);
        let col = &[
            1.0,
            2.0,
            5.0,
            f64::NAN,
            8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ][..];
        let (left, right) = Forest::partition_mono_inc::<false, u32>(&node, |r| col[r], 0b11111u32);
        assert_eq!(left, 0b01011);
        assert_eq!(right, 0b10100);
    }

    #[test]
    fn test_partition_mono_inc_nan_no_default_left() {
        let node = num_node(3.0, false);
        let col = &[
            1.0,
            2.0,
            5.0,
            f64::NAN,
            8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ][..];
        let (left, right) = Forest::partition_mono_inc::<false, u32>(&node, |r| col[r], 0b11111u32);
        assert_eq!(left, 0b00011);
        assert_eq!(right, 0b11100);
    }

    #[test]
    fn test_partition_mono_inc_nan_before_threshold() {
        let node = num_node(3.0, true);
        let col = &[
            f64::NAN,
            1.0,
            2.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ][..];
        let (left, right) = Forest::partition_mono_inc::<false, u32>(&node, |r| col[r], 0b1111u32);
        assert_eq!(left, 0b0111);
        assert_eq!(right, 0b1000);
    }

    // --- Sweep tests ---

    fn sweep_forest(preds: Vec<VaryingPredicate>, ranges: Vec<FeatureRange>) -> Forest {
        Forest {
            output: crate::parser::Output {
                base_score: 0.0,
                divisor: 1.0,
                postprocessor: crate::parser::Postprocessor::Sigmoid(1.0),
            },
            compiled_config: WalkerConfig::try_new(64, 32, &(0..64).collect::<Vec<_>>(), &[], &[])
                .unwrap(),
            trees: Vec::new(),
            config: WalkerConfig {
                n_features: 64,
                max_group_width: 32,
                varying_mask: u128::from(u64::MAX),
                mono_inc_mask: 0,
                mono_dec_mask: 0,
                ablation: AblationMode::default(),
            },
            nodes: Vec::new(),
            bitsets: Vec::new(),
            varying_predicates: preds,
            feature_ranges: ranges,
            threshold_type: ThresholdType::F64,
            prefix_groups: Vec::new(),
            prefix_depth: 0,
            fixed_scale: None,
            workspace: None,
        }
    }

    fn num_pred(feature: u16, threshold: f64, default_left: bool) -> VaryingPredicate {
        VaryingPredicate::Num {
            feature,
            threshold,
            default_left,
        }
    }

    fn sweep<const F32: bool>(forest: &Forest, cols: &[[f64; 32]; 64], n_rows: usize) -> Vec<u32> {
        let rows = rows_from_columns(cols, n_rows, forest.config.n_features);
        let mut out = vec![0u32; forest.varying_predicates.len()];
        let (mut column, mut order) = (vec![0.0; n_rows], vec![(0.0, 0); n_rows]);
        forest.precompute_varying_masks::<F32, u32>(&rows, &mut column, &mut order, &mut out);
        out
    }

    fn run_sweep(forest: &Forest, cols: &[[f64; 32]; 64], n_rows: usize) -> Vec<u32> {
        sweep::<false>(forest, cols, n_rows)
    }

    #[test]
    fn test_sweep_all_nan_column() {
        let forest = sweep_forest(
            vec![num_pred(0, 5.0, true), num_pred(0, 10.0, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 2,
                cat_start: 2,
                cat_end: 2,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        for r in 0..4 {
            cols[0][r] = f64::NAN;
        }
        let masks = run_sweep(&forest, &cols, 4);
        assert_eq!(masks[0], 0b1111);
        assert_eq!(masks[1], 0b0000);
    }

    #[test]
    fn test_sweep_single_row() {
        let forest = sweep_forest(
            vec![num_pred(0, 3.0, false), num_pred(0, 7.0, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 2,
                cat_start: 2,
                cat_end: 2,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        cols[0][0] = 5.0;
        let masks = run_sweep(&forest, &cols, 1);
        assert_eq!(masks[0], 0b0);
        assert_eq!(masks[1], 0b1);
    }

    #[test]
    fn test_sweep_all_rows_identical() {
        let forest = sweep_forest(
            vec![num_pred(0, 3.0, false), num_pred(0, 7.0, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 2,
                cat_start: 2,
                cat_end: 2,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        for r in 0..8 {
            cols[0][r] = 5.0;
        }
        let masks = run_sweep(&forest, &cols, 8);
        assert_eq!(masks[0], 0);
        assert_eq!(masks[1], 0b1111_1111);
    }

    #[test]
    fn test_sweep_max_group_width() {
        let forest = sweep_forest(
            vec![num_pred(0, 16.0, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 1,
                cat_start: 1,
                cat_end: 1,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        for r in 0..32 {
            cols[0][r] = r as f64;
        }
        let masks = run_sweep(&forest, &cols, 32);
        let expected: u32 = (0..=16).fold(0u32, |m, r| m | (1 << r));
        assert_eq!(masks[0], expected);
    }

    #[test]
    fn test_sweep_mixed_nan_and_values() {
        let forest = sweep_forest(
            vec![num_pred(0, 2.0, true), num_pred(0, 4.0, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 2,
                cat_start: 2,
                cat_end: 2,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        cols[0][0] = f64::NAN;
        cols[0][1] = 1.0;
        cols[0][2] = f64::NAN;
        cols[0][3] = 5.0;
        cols[0][4] = 3.0;
        let masks = run_sweep(&forest, &cols, 5);
        assert_eq!(masks[0], 0b00111);
        assert_eq!(masks[1], 0b10010);
    }

    #[test]
    fn test_sweep_no_numerical_predicates() {
        let forest = sweep_forest(
            vec![VaryingPredicate::CatInline {
                feature: 0,
                default_left: false,
                word: 0b1010,
            }],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 0,
                cat_start: 0,
                cat_end: 1,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        cols[0][0] = 0.0;
        cols[0][1] = 1.0;
        cols[0][2] = 3.0;
        let masks = run_sweep(&forest, &cols, 3);
        assert_eq!(masks[0], 0b110);
    }

    #[test]
    fn test_sweep_f32_threshold_collisions() {
        let t1: f64 = 1.0 + f64::from(2.0f32.powi(-24));
        let t2: f64 = 1.0 + f64::from(2.0f32.powi(-25));
        assert_eq!(
            (t1 as f32).to_bits(),
            (t2 as f32).to_bits(),
            "precondition: same f32"
        );
        let forest = sweep_forest(
            vec![num_pred(0, t1, false), num_pred(0, t2, false)],
            vec![FeatureRange {
                feature: 0,
                num_start: 0,
                num_end: 2,
                cat_start: 2,
                cat_end: 2,
            }],
        );
        let mut cols = Box::new([[0.0f64; 32]; 64]);
        cols[0][0] = 1.0;
        cols[0][1] = 1.000_000_15;
        let out = sweep::<true>(&forest, &cols, 2);
        assert_eq!(out[0], out[1]);
    }
}
