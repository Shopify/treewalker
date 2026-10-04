//! Experimental kernels (opt-experiments branch).
//!
//! `Forest::predict` is untouched; everything here is a separate code path so the
//! baseline timing is unaffected. Variants:
//! - leaf accumulation: scalar (baseline), split u128 into two u64 loops,
//!   AVX-512 masked add (bit-exact), difference array over runs (not bit-exact)
//! - K-tree lockstep constant walk (bit-exact: leaves still applied in tree order)
//! - schedule-feature mask caching across groups of equal width
//! - mask width chosen per group / forced wide (dispatch experiment)
//! - profile-guided layout (profiling kernel + parser hook in `parser::layout`)
// Research code: correctness and suspicious lints stay on; style/pedantic lints are
// relaxed for this module only.
#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::style,
    clippy::complexity,
    clippy::perf
)]

use std::mem::MaybeUninit;

use crate::forest::{Forest, ThresholdType, VaryingPredicate, threshold_go_left};
use crate::mask::RowMask;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafMode {
    Scalar,
    Halves,
    Avx512,
    DiffArray,
    /// Fixed-point (i128) difference array: exact, order-independent sums, rounded once.
    Exact,
    /// Timing knock-out: leaves do no accumulation (results are wrong).
    Noop,
}

/// Per-group accumulation buffers for the difference-array leaf modes.
pub struct Acc {
    pub f: [f64; 130],
    pub x: [i128; 130],
    /// Fixed-point scale: leaf value v is accumulated as v * 2^e (an exact integer).
    pub e: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct OptFlags {
    pub leaf: LeafMode,
    /// 1 = no lockstep; 4 or 8 = walk that many trees' constant prefixes together.
    pub lockstep_k: usize,
    /// Bitmask of varying features whose values depend only on the row index.
    pub schedule_features: u128,
    /// Use the u128 / G=128 path regardless of group width.
    pub force_wide: bool,
    /// Choose mask width from the group's own row count instead of `max_group_width`.
    pub per_group_width: bool,
    /// Measurement only: populate columns and precompute masks, skip the trees.
    pub skip_trees: bool,
    /// Fixed-point scale for LeafMode::Exact (from `Forest::exact_scale`).
    pub exact_e: i32,
}

impl Default for OptFlags {
    fn default() -> Self {
        Self {
            leaf: LeafMode::Scalar,
            lockstep_k: 1,
            schedule_features: 0,
            force_wide: false,
            per_group_width: false,
            skip_trees: false,
            exact_e: 0,
        }
    }
}

pub struct OptWorkspace {
    m32: Vec<u32>,
    m64: Vec<u64>,
    m128: Vec<u128>,
    c32: Box<[[f64; 32]; 64]>,
    c64: Box<[[f64; 64]; 64]>,
    c128: Box<[[f64; 128]; 64]>,
    prefix_starts: Vec<u16>,
    diff: Box<Acc>,
    sched_n: [usize; 3],
}

impl OptWorkspace {
    pub fn new(forest: &Forest) -> Self {
        let np = forest.varying_predicates.len();
        Self {
            m32: vec![0; np],
            m64: vec![0; np],
            m128: vec![0; np],
            c32: Box::new([[0.0; 32]; 64]),
            c64: vec![[0.0; 64]; 64].into_boxed_slice().try_into().unwrap(),
            c128: vec![[0.0; 128]; 64].into_boxed_slice().try_into().unwrap(),
            prefix_starts: vec![0; forest.trees.len()],
            diff: Box::new(Acc {
                f: [0.0; 130],
                x: [0; 130],
                e: 0,
            }),
            sched_n: [0; 3],
        }
    }
}

struct Cx<'a, M: RowMask> {
    base: usize,
    const_features: &'a [f64],
    masks: &'a [M],
    results: &'a mut [f64],
    start: usize,
    diff: &'a mut Acc,
}

macro_rules! leaf_dispatch {
    ($self:ident, $f32:literal, $m:ty, $g:literal, $args:expr, $flags:expr) => {{
        let (data, results, start, end, masks, cols, ps, diff, sched_n) = $args;
        match ($flags.leaf, $flags.lockstep_k) {
            (LeafMode::Scalar, 2) => $self.core::<$f32, $m, $g, 0, 2>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Scalar, 3) => $self.core::<$f32, $m, $g, 0, 3>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Scalar, 5) => $self.core::<$f32, $m, $g, 0, 5>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Halves, 3) => $self.core::<$f32, $m, $g, 1, 3>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Halves, 5) => $self.core::<$f32, $m, $g, 1, 5>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Avx512, 3) => $self.core::<$f32, $m, $g, 2, 3>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Noop, 7) => $self.core::<$f32, $m, $g, 5, 7>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Exact, 1) => $self.core::<$f32, $m, $g, 4, 1>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Exact, 7) => $self.core::<$f32, $m, $g, 4, 7>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Exact, 9) => $self.core::<$f32, $m, $g, 4, 9>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Avx512, 9) => $self.core::<$f32, $m, $g, 2, 9>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Exact, 10) => $self.core::<$f32, $m, $g, 4, 10>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Avx512, 10) => $self.core::<$f32, $m, $g, 2, 10>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::DiffArray, 7) => $self.core::<$f32, $m, $g, 3, 7>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Avx512, 7) => $self.core::<$f32, $m, $g, 2, 7>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Halves, 7) => $self.core::<$f32, $m, $g, 1, 7>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Avx512, 6) => $self.core::<$f32, $m, $g, 2, 6>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Halves, 6) => $self.core::<$f32, $m, $g, 1, 6>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Avx512, 5) => $self.core::<$f32, $m, $g, 2, 5>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Avx512, 2) => $self.core::<$f32, $m, $g, 2, 2>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Scalar, 1) => $self.core::<$f32, $m, $g, 0, 1>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Scalar, 4) => $self.core::<$f32, $m, $g, 0, 4>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Scalar, 8) => $self.core::<$f32, $m, $g, 0, 8>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Halves, 1) => $self.core::<$f32, $m, $g, 1, 1>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Halves, 4) => $self.core::<$f32, $m, $g, 1, 4>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Halves, 8) => $self.core::<$f32, $m, $g, 1, 8>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Avx512, 1) => $self.core::<$f32, $m, $g, 2, 1>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Avx512, 4) => $self.core::<$f32, $m, $g, 2, 4>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::Avx512, 8) => $self.core::<$f32, $m, $g, 2, 8>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::DiffArray, 1) => $self.core::<$f32, $m, $g, 3, 1>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::DiffArray, 4) => $self.core::<$f32, $m, $g, 3, 4>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (LeafMode::DiffArray, 8) => $self.core::<$f32, $m, $g, 3, 8>(
                data, results, start, end, masks, cols, ps, diff, sched_n, $flags,
            ),
            (_, k) => panic!("unsupported lockstep_k {k}"),
        }
    }};
}

/// Rows a bitmask kernel of the given configured width can hold.
const fn mask_cap(width: usize) -> usize {
    if width <= 32 {
        32
    } else if width <= 64 {
        64
    } else {
        128
    }
}

impl Forest {
    /// Experimental prediction path. Same contract as `predict`.
    /// Input guard for the experimental entry points; mirrors `check_input` plus the
    /// kernel's row capacity.
    fn opt_guard(
        &self,
        data: &[f64],
        out_len: Option<usize>,
        start: usize,
        end: usize,
        cap: usize,
    ) {
        assert!(
            self.config.structural_key() == self.compiled_config.structural_key(),
            "Forest.config structural fields changed after loading; reload the forest"
        );
        assert!(start < end, "group must be nonempty");
        assert!(
            end - start <= cap,
            "group of {} rows exceeds kernel capacity {cap}",
            end - start
        );
        let elements = end
            .checked_mul(self.config.n_features)
            .expect("prediction dimensions overflow");
        assert!(elements <= data.len(), "prediction data is too short");
        if let Some(l) = out_len {
            assert!(end <= l, "prediction output is too short");
        }
    }

    pub fn predict_opt(
        &self,
        ws: &mut OptWorkspace,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
        flags: &OptFlags,
    ) {
        let n = end.saturating_sub(start);
        let width = if flags.force_wide {
            128
        } else if flags.per_group_width {
            n
        } else {
            self.config.max_group_width
        };
        self.opt_guard(data, Some(results.len()), start, end, mask_cap(width));
        let ps = &mut ws.prefix_starts;
        let diff = &mut *ws.diff;
        diff.e = flags.exact_e;
        match (self.threshold_type, width) {
            (ThresholdType::F64, 0..=32) => leaf_dispatch!(
                self,
                false,
                u32,
                32,
                (
                    data,
                    results,
                    start,
                    end,
                    &mut ws.m32[..],
                    &mut *ws.c32,
                    ps,
                    diff,
                    &mut ws.sched_n[0]
                ),
                flags
            ),
            (ThresholdType::F64, 33..=64) => leaf_dispatch!(
                self,
                false,
                u64,
                64,
                (
                    data,
                    results,
                    start,
                    end,
                    &mut ws.m64[..],
                    &mut *ws.c64,
                    ps,
                    diff,
                    &mut ws.sched_n[1]
                ),
                flags
            ),
            (ThresholdType::F64, _) => leaf_dispatch!(
                self,
                false,
                u128,
                128,
                (
                    data,
                    results,
                    start,
                    end,
                    &mut ws.m128[..],
                    &mut *ws.c128,
                    ps,
                    diff,
                    &mut ws.sched_n[2]
                ),
                flags
            ),
            (ThresholdType::F32, 0..=32) => leaf_dispatch!(
                self,
                true,
                u32,
                32,
                (
                    data,
                    results,
                    start,
                    end,
                    &mut ws.m32[..],
                    &mut *ws.c32,
                    ps,
                    diff,
                    &mut ws.sched_n[0]
                ),
                flags
            ),
            (ThresholdType::F32, 33..=64) => leaf_dispatch!(
                self,
                true,
                u64,
                64,
                (
                    data,
                    results,
                    start,
                    end,
                    &mut ws.m64[..],
                    &mut *ws.c64,
                    ps,
                    diff,
                    &mut ws.sched_n[1]
                ),
                flags
            ),
            (ThresholdType::F32, _) => leaf_dispatch!(
                self,
                true,
                u128,
                128,
                (
                    data,
                    results,
                    start,
                    end,
                    &mut ws.m128[..],
                    &mut *ws.c128,
                    ps,
                    diff,
                    &mut ws.sched_n[2]
                ),
                flags
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn core<const F32: bool, M: RowMask, const G: usize, const LEAF: u8, const K: usize>(
        &self,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
        masks: &mut [M],
        cols: &mut [[f64; G]; 64],
        prefix_starts: &mut [u16],
        diff: &mut Acc,
        sched_n: &mut usize,
        flags: &OptFlags,
    ) {
        let n = end - start;
        let nf = self.config.n_features;
        let const_features = unsafe { data.get_unchecked(start * nf..(start + 1) * nf) };

        // Schedule features: masks persist in the workspace between groups of equal width.
        let sched = flags.schedule_features & self.config.varying_mask;
        let cached = sched != 0 && *sched_n == n;
        let skip = if cached { sched } else { 0 };

        let mut vm = self.config.varying_mask & !skip;
        while vm != 0 {
            let f = vm.trailing_zeros() as usize;
            for r in 0..n {
                cols[f][r] = unsafe { *data.get_unchecked((start + r) * nf + f) };
            }
            vm &= vm - 1;
        }
        self.precompute_opt::<F32, M, G>(cols, n, masks, skip);
        if sched != 0 {
            *sched_n = n;
        }

        results[start..end].fill(0.0);
        if flags.skip_trees {
            return;
        }
        let all: M = M::from_width(n);

        if !self.prefix_groups.is_empty() && K != 7 && K != 9 {
            let k = self.prefix_depth;
            prefix_starts.fill(0);
            let mut none: Option<crate::predict::PredictStats> = None;
            for group in &self.prefix_groups {
                self.precompute_prefix_starts::<F32, false>(
                    const_features,
                    &mut none,
                    group,
                    k,
                    prefix_starts,
                );
            }
        }

        let mut cx = Cx::<M> {
            base: 0,
            const_features,
            masks,
            results,
            start,
            diff,
        };

        let trees = &self.trees;
        if K == 7 || K == 9 || K == 10 {
            // Unified fall-through compare (`flip_transform` + child hints): the step is
            // `cfx[feature'] <= t'`, branching directly on the compare flags.
            let nf = self.config.n_features;
            let mut cfx = [0.0f64; 256];
            for f in 0..nf {
                let x = const_features[f];
                if F32 {
                    let x32 = x as f32;
                    cfx[f] = f64::from(x32);
                    cfx[f + nf] = f64::from(-x32);
                } else {
                    cfx[f] = x;
                    cfx[f + nf] = -x;
                }
            }
            let nodes = self.nodes.as_slice();
            if !self.prefix_groups.is_empty() {
                let k = self.prefix_depth;
                prefix_starts.fill(0);
                for group in &self.prefix_groups {
                    let rep = group.node_base as usize;
                    let mut bail = k;
                    for j in 0..k {
                        if !self.flip_ft::<F32>(&nodes[rep + j], &cfx, const_features) {
                            bail = j;
                            break;
                        }
                    }
                    for &t in &group.trees {
                        prefix_starts[t as usize] = if bail == k {
                            k as u16
                        } else {
                            nodes[self.trees[t as usize].node_start as usize + bail].skip as u16
                        };
                    }
                }
            }
            for (i, tree) in trees.iter().enumerate() {
                cx.base = tree.node_start as usize;
                let s0 = prefix_starts[i] as usize;
                let kind = node_kind(unsafe { nodes.get_unchecked(cx.base + s0) });
                if K == 9 {
                    self.pe_flip_stack::<F32, M, G, LEAF, false>(&mut cx, &cfx, s0, all, kind);
                } else if K == 10 {
                    self.pe_flip_stack::<F32, M, G, LEAF, true>(&mut cx, &cfx, s0, all, kind);
                } else {
                    self.pe_flip::<F32, M, G, LEAF>(&mut cx, &cfx, s0, all, kind);
                }
            }
        } else if K == 3 || K == 5 || K == 6 {
            // K == 3: child-kind hints + branch-free inline categorical eval.
            // K == 5: branch-free inline categorical eval only (original loop structure).
            let nodes = self.nodes.as_slice();
            for (i, tree) in trees.iter().enumerate() {
                cx.base = tree.node_start as usize;
                let s0 = prefix_starts[i] as usize;
                if K == 3 || K == 6 {
                    let kind = node_kind(unsafe { nodes.get_unchecked(cx.base + s0) });
                    if K == 3 {
                        self.pe_hint::<F32, M, G, LEAF, true>(&mut cx, s0, all, kind);
                    } else {
                        self.pe_hint::<F32, M, G, LEAF, false>(&mut cx, s0, all, kind);
                    }
                } else {
                    self.pe_fastcat::<F32, M, G, LEAF>(&mut cx, s0, all);
                }
            }
        } else if K <= 2 {
            // K == 2 is the software-prefetch variant (lockstep is K >= 4).
            for (i, tree) in trees.iter().enumerate() {
                cx.base = tree.node_start as usize;
                if K == 2 {
                    self.pe_opt::<F32, M, G, LEAF, true>(&mut cx, prefix_starts[i] as usize, all);
                } else {
                    self.pe_opt::<F32, M, G, LEAF, false>(&mut cx, prefix_starts[i] as usize, all);
                }
            }
        } else {
            let nodes = self.nodes.as_slice();
            let full = trees.len() / K * K;
            let mut i = 0;
            while i < full {
                let mut base = [0usize; K];
                let mut idx = [0usize; K];
                for k in 0..K {
                    base[k] = trees[i + k].node_start as usize;
                    idx[k] = prefix_starts[i + k] as usize;
                }
                // Lockstep constant walk: K independent dependency chains in flight.
                loop {
                    let mut any = false;
                    for k in 0..K {
                        let node = unsafe { nodes.get_unchecked(base[k] + idx[k]) };
                        let w = node.is_walkable();
                        let gl = self.eval_split::<F32>(node, const_features);
                        let nxt = if gl == node.heavy_is_left() {
                            idx[k] + 1
                        } else {
                            node.skip as u16 as usize
                        };
                        idx[k] = if w { nxt } else { idx[k] };
                        any |= w;
                    }
                    if !any {
                        break;
                    }
                }
                // Finish each tree in order (leaf adds stay in tree order: bit-exact).
                for k in 0..K {
                    cx.base = base[k];
                    self.pe_opt::<F32, M, G, LEAF, false>(&mut cx, idx[k], all);
                }
                i += K;
            }
            for (j, tree) in trees.iter().enumerate().skip(full) {
                cx.base = tree.node_start as usize;
                self.pe_opt::<F32, M, G, LEAF, false>(&mut cx, prefix_starts[j] as usize, all);
            }
        }

        if LEAF == 4 {
            let e = cx.diff.e;
            let mut acc: i128 = 0;
            for r in 0..n {
                acc = acc.wrapping_add(cx.diff.x[r]);
                cx.results[start + r] = fixed_to_f64(acc, e);
            }
            cx.diff.x[..=n].fill(0);
        }
        if LEAF == 3 {
            let mut acc = 0.0;
            for r in 0..n {
                acc += cx.diff.f[r];
                cx.results[start + r] += acc;
            }
            cx.diff.f[..=n].fill(0.0);
        }

        // Same finalization as `predict` (divisor, base score, postprocessor).
        self.finalize(&mut cx.results[start..end]);
    }

    fn pe_opt<const F32: bool, M: RowMask, const G: usize, const LEAF: u8, const PF: bool>(
        &self,
        cx: &mut Cx<M>,
        mut idx: usize,
        mut row_mask: M,
    ) {
        if row_mask.is_zero() {
            return;
        }
        let nodes = self.nodes.as_slice();
        let base = cx.base;
        loop {
            while unsafe { nodes.get_unchecked(base + idx) }.is_walkable() {
                if PF {
                    // Software prefetch of the out-of-line children of the next two
                    // heavy-path nodes, so a taken jump finds its line already requested.
                    let last = nodes.len() - 1;
                    for d in 1..=2 {
                        let ahead = unsafe { nodes.get_unchecked((base + idx + d).min(last)) };
                        let tgt = nodes
                            .as_ptr()
                            .wrapping_add(base + (ahead.skip as u16 as usize));
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
                                tgt.cast(),
                            );
                        }
                    }
                }
                idx = self.step::<F32>(nodes, base + idx, idx, cx.const_features);
            }
            let node = unsafe { nodes.get_unchecked(base + idx) };
            if node.is_leaf() {
                leaf_add::<M, G, LEAF>(cx, node.value, row_mask);
                return;
            }
            let pm = unsafe { *cx.masks.get_unchecked(node.varying_pred_id as usize) };
            let (lm, rm) = (row_mask & pm, row_mask & !pm);
            let (heavy, light) = if node.heavy_is_left() {
                (lm, rm)
            } else {
                (rm, lm)
            };
            if light.is_zero() {
                idx += 1;
                continue;
            }
            if heavy.is_zero() {
                idx = node.skip as usize;
                continue;
            }
            self.pe_opt::<F32, M, G, LEAF, PF>(cx, node.skip as usize, light);
            idx += 1;
            row_mask = heavy;
        }
    }

    /// Copy of `precompute_varying_masks` that can skip features whose masks are cached.
    fn precompute_opt<const F32: bool, M: RowMask, const G: usize>(
        &self,
        cols: &[[f64; G]; 64],
        n_rows: usize,
        out: &mut [M],
        skip_features: u128,
    ) {
        for range in &self.feature_ranges {
            let f = range.feature as usize;
            if skip_features & (1u128 << f) != 0 {
                continue;
            }
            let col = &cols[f];
            if range.num_start < range.num_end {
                let mut nan_mask = M::ZERO;
                let mut nn = 0usize;
                let mut sorted: [(f64, M); G] = [(0.0, M::ZERO); G];
                for r in 0..n_rows {
                    let val = unsafe { *col.get_unchecked(r) };
                    if val.is_nan() {
                        nan_mask = nan_mask.set_bit(r);
                    } else {
                        sorted[nn] = (val, M::ZERO.set_bit(r));
                        nn += 1;
                    }
                }
                let sorted = &mut sorted[..nn];
                if F32 {
                    sorted.sort_unstable_by(|a, b| (a.0 as f32).total_cmp(&(b.0 as f32)));
                } else {
                    sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                }
                let mut ptr = 0usize;
                let mut left = M::ZERO;
                let preds =
                    &self.varying_predicates[range.num_start as usize..range.num_end as usize];
                let masks = &mut out[range.num_start as usize..range.num_end as usize];
                for (o, pred) in masks.iter_mut().zip(preds.iter()) {
                    let (thr, dl) = match pred {
                        VaryingPredicate::Num {
                            threshold,
                            default_left,
                            ..
                        } => (*threshold, *default_left),
                        _ => unreachable!(),
                    };
                    while ptr < nn {
                        let (val, bit) = unsafe { *sorted.get_unchecked(ptr) };
                        if threshold_go_left::<F32>(val, thr) {
                            left |= bit;
                            ptr += 1;
                        } else {
                            break;
                        }
                    }
                    *o = left | if dl { nan_mask } else { M::ZERO };
                }
            }
            let cat_preds =
                &self.varying_predicates[range.cat_start as usize..range.cat_end as usize];
            let cat_masks = &mut out[range.cat_start as usize..range.cat_end as usize];
            for (o, pred) in cat_masks.iter_mut().zip(cat_preds.iter()) {
                let mut m = M::ZERO;
                for r in 0..n_rows {
                    if pred.goes_left::<F32>(unsafe { *col.get_unchecked(r) }, &self.bitsets) {
                        m = m.set_bit(r);
                    }
                }
                *o = m;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Profiling kernel for profile-guided layout
    // -----------------------------------------------------------------------

    /// Run one group and accumulate per-node direction counts (absolute pool index):
    /// constant nodes count group-visits by direction; varying nodes count one-sided
    /// partitions (all rows left / all rows right). Splits that send rows both ways
    /// touch both children and are not counted.
    pub fn profile_group(
        &self,
        ws: &mut OptWorkspace,
        data: &[f64],
        start: usize,
        end: usize,
        counts: &mut [(u64, u64)],
    ) {
        assert!(
            self.prefix_groups.is_empty(),
            "profile with prefix grouping disabled"
        );
        self.opt_guard(
            data,
            None,
            start,
            end,
            mask_cap(self.config.max_group_width),
        );
        let n = end - start;
        match (self.threshold_type, self.config.max_group_width) {
            (ThresholdType::F64, 0..=32) => self.profile_core::<false, u32, 32>(
                data,
                start,
                n,
                &mut ws.m32,
                &mut ws.c32,
                counts,
            ),
            (ThresholdType::F64, 33..=64) => self.profile_core::<false, u64, 64>(
                data,
                start,
                n,
                &mut ws.m64,
                &mut ws.c64,
                counts,
            ),
            (ThresholdType::F64, _) => self.profile_core::<false, u128, 128>(
                data,
                start,
                n,
                &mut ws.m128,
                &mut ws.c128,
                counts,
            ),
            (ThresholdType::F32, 0..=32) => {
                self.profile_core::<true, u32, 32>(data, start, n, &mut ws.m32, &mut ws.c32, counts)
            }
            (ThresholdType::F32, 33..=64) => {
                self.profile_core::<true, u64, 64>(data, start, n, &mut ws.m64, &mut ws.c64, counts)
            }
            (ThresholdType::F32, _) => self.profile_core::<true, u128, 128>(
                data,
                start,
                n,
                &mut ws.m128,
                &mut ws.c128,
                counts,
            ),
        }
    }

    fn profile_core<const F32: bool, M: RowMask, const G: usize>(
        &self,
        data: &[f64],
        start: usize,
        n: usize,
        masks: &mut [M],
        cols: &mut [[f64; G]; 64],
        counts: &mut [(u64, u64)],
    ) {
        let nf = self.config.n_features;
        let cf = &data[start * nf..(start + 1) * nf];
        let mut vm = self.config.varying_mask;
        while vm != 0 {
            let f = vm.trailing_zeros() as usize;
            for r in 0..n {
                cols[f][r] = data[(start + r) * nf + f];
            }
            vm &= vm - 1;
        }
        self.precompute_opt::<F32, M, G>(cols, n, masks, 0);
        let all = M::from_width(n);
        for tree in &self.trees {
            self.profile_pe::<F32, M>(tree.node_start as usize, cf, masks, 0, all, counts);
        }
    }

    fn profile_pe<const F32: bool, M: RowMask>(
        &self,
        base: usize,
        cf: &[f64],
        masks: &[M],
        mut idx: usize,
        mut row_mask: M,
        counts: &mut [(u64, u64)],
    ) {
        let nodes = self.nodes.as_slice();
        loop {
            let node = &nodes[base + idx];
            if node.is_leaf() {
                return;
            }
            if node.is_constant() {
                let gl = self.eval_split::<F32>(node, cf);
                if gl {
                    counts[base + idx].0 += 1
                } else {
                    counts[base + idx].1 += 1
                }
                idx = if gl == node.heavy_is_left() {
                    idx + 1
                } else {
                    node.skip as usize
                };
                continue;
            }
            let pm = masks[node.varying_pred_id as usize];
            let (lm, rm) = (row_mask & pm, row_mask & !pm);
            if rm.is_zero() {
                counts[base + idx].0 += 1;
            } else if lm.is_zero() {
                counts[base + idx].1 += 1;
            }
            let (heavy, light) = if node.heavy_is_left() {
                (lm, rm)
            } else {
                (rm, lm)
            };
            if light.is_zero() {
                idx += 1;
                continue;
            }
            if heavy.is_zero() {
                idx = node.skip as usize;
                continue;
            }
            self.profile_pe::<F32, M>(base, cf, masks, node.skip as usize, light, counts);
            idx += 1;
            row_mask = heavy;
        }
    }

    /// Fraction of constant-walk steps that fall through to idx+1, and prefix-group
    /// bails, measured over the given groups with the production layout.
    pub fn fallthrough_rate(
        &self,
        ws: &mut OptWorkspace,
        data: &[f64],
        groups: &[(usize, usize)],
    ) -> (u64, u64) {
        let mut ft = 0u64;
        let mut total = 0u64;
        for &(s, e) in groups {
            self.opt_guard(data, None, s, e, mask_cap(self.config.max_group_width));
            let n = e - s;
            let nf = self.config.n_features;
            let cf = &data[s * nf..(s + 1) * nf];
            match (self.threshold_type, self.config.max_group_width) {
                (ThresholdType::F64, 0..=32) => self.ft_core::<false, u32, 32>(
                    data,
                    s,
                    n,
                    &mut ws.m32,
                    &mut ws.c32,
                    cf,
                    &mut ft,
                    &mut total,
                ),
                (ThresholdType::F64, 33..=64) => self.ft_core::<false, u64, 64>(
                    data,
                    s,
                    n,
                    &mut ws.m64,
                    &mut ws.c64,
                    cf,
                    &mut ft,
                    &mut total,
                ),
                (ThresholdType::F64, _) => self.ft_core::<false, u128, 128>(
                    data,
                    s,
                    n,
                    &mut ws.m128,
                    &mut ws.c128,
                    cf,
                    &mut ft,
                    &mut total,
                ),
                (ThresholdType::F32, 0..=32) => self.ft_core::<true, u32, 32>(
                    data,
                    s,
                    n,
                    &mut ws.m32,
                    &mut ws.c32,
                    cf,
                    &mut ft,
                    &mut total,
                ),
                (ThresholdType::F32, 33..=64) => self.ft_core::<true, u64, 64>(
                    data,
                    s,
                    n,
                    &mut ws.m64,
                    &mut ws.c64,
                    cf,
                    &mut ft,
                    &mut total,
                ),
                (ThresholdType::F32, _) => self.ft_core::<true, u128, 128>(
                    data,
                    s,
                    n,
                    &mut ws.m128,
                    &mut ws.c128,
                    cf,
                    &mut ft,
                    &mut total,
                ),
            }
        }
        (ft, total)
    }

    #[allow(clippy::too_many_arguments)]
    fn ft_core<const F32: bool, M: RowMask, const G: usize>(
        &self,
        data: &[f64],
        start: usize,
        n: usize,
        masks: &mut [M],
        cols: &mut [[f64; G]; 64],
        cf: &[f64],
        ft: &mut u64,
        total: &mut u64,
    ) {
        let nf = self.config.n_features;
        let mut vm = self.config.varying_mask;
        while vm != 0 {
            let f = vm.trailing_zeros() as usize;
            for r in 0..n {
                cols[f][r] = data[(start + r) * nf + f];
            }
            vm &= vm - 1;
        }
        self.precompute_opt::<F32, M, G>(cols, n, masks, 0);
        let all = M::from_width(n);
        for tree in &self.trees {
            self.ft_pe::<F32, M>(tree.node_start as usize, cf, masks, 0, all, ft, total);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ft_pe<const F32: bool, M: RowMask>(
        &self,
        base: usize,
        cf: &[f64],
        masks: &[M],
        mut idx: usize,
        mut row_mask: M,
        ft: &mut u64,
        total: &mut u64,
    ) {
        let nodes = self.nodes.as_slice();
        loop {
            let node = &nodes[base + idx];
            if node.is_leaf() {
                return;
            }
            if node.is_constant() {
                let gl = self.eval_split::<F32>(node, cf);
                *total += 1;
                if gl == node.heavy_is_left() {
                    *ft += 1;
                    idx += 1;
                } else {
                    idx = node.skip as usize;
                }
                continue;
            }
            let pm = masks[node.varying_pred_id as usize];
            let (lm, rm) = (row_mask & pm, row_mask & !pm);
            let (heavy, light) = if node.heavy_is_left() {
                (lm, rm)
            } else {
                (rm, lm)
            };
            if light.is_zero() {
                idx += 1;
                continue;
            }
            if heavy.is_zero() {
                idx = node.skip as usize;
                continue;
            }
            self.ft_pe::<F32, M>(base, cf, masks, node.skip as usize, light, ft, total);
            idx += 1;
            row_mask = heavy;
        }
    }

    /// Node visit counts (every node reached, leaves included) and cache-line footprint
    /// over the given groups: (mean distinct 64-byte lines per group, distinct lines over
    /// all groups). Walks from each tree's root (prefix grouping not applied).
    pub fn touch_profile(
        &self,
        ws: &mut OptWorkspace,
        data: &[f64],
        groups: &[(usize, usize)],
        visits: &mut [u64],
    ) -> (f64, usize) {
        let base_addr = self.nodes.as_ptr() as usize;
        let n_lines = (self.nodes.len() * 16 + base_addr % 64) / 64 + 2;
        let mut union = vec![false; n_lines];
        let mut lines: Vec<u32> = Vec::new();
        let mut sum = 0usize;
        for &(s, e) in groups {
            self.opt_guard(data, None, s, e, mask_cap(self.config.max_group_width));
            lines.clear();
            let n = e - s;
            match (self.threshold_type, self.config.max_group_width) {
                (ThresholdType::F64, 0..=32) => self.touch_core::<false, u32, 32>(
                    data,
                    s,
                    n,
                    &mut ws.m32,
                    &mut ws.c32,
                    visits,
                    &mut lines,
                ),
                (ThresholdType::F64, 33..=64) => self.touch_core::<false, u64, 64>(
                    data,
                    s,
                    n,
                    &mut ws.m64,
                    &mut ws.c64,
                    visits,
                    &mut lines,
                ),
                (ThresholdType::F64, _) => self.touch_core::<false, u128, 128>(
                    data,
                    s,
                    n,
                    &mut ws.m128,
                    &mut ws.c128,
                    visits,
                    &mut lines,
                ),
                (ThresholdType::F32, 0..=32) => self.touch_core::<true, u32, 32>(
                    data,
                    s,
                    n,
                    &mut ws.m32,
                    &mut ws.c32,
                    visits,
                    &mut lines,
                ),
                (ThresholdType::F32, 33..=64) => self.touch_core::<true, u64, 64>(
                    data,
                    s,
                    n,
                    &mut ws.m64,
                    &mut ws.c64,
                    visits,
                    &mut lines,
                ),
                (ThresholdType::F32, _) => self.touch_core::<true, u128, 128>(
                    data,
                    s,
                    n,
                    &mut ws.m128,
                    &mut ws.c128,
                    visits,
                    &mut lines,
                ),
            }
            lines.sort_unstable();
            lines.dedup();
            sum += lines.len();
            for &l in &lines {
                union[l as usize] = true;
            }
        }
        (
            sum as f64 / groups.len().max(1) as f64,
            union.iter().filter(|&&b| b).count(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn touch_core<const F32: bool, M: RowMask, const G: usize>(
        &self,
        data: &[f64],
        start: usize,
        n: usize,
        masks: &mut [M],
        cols: &mut [[f64; G]; 64],
        visits: &mut [u64],
        lines: &mut Vec<u32>,
    ) {
        let nf = self.config.n_features;
        let cf = &data[start * nf..(start + 1) * nf];
        let mut vm = self.config.varying_mask;
        while vm != 0 {
            let f = vm.trailing_zeros() as usize;
            for r in 0..n {
                cols[f][r] = data[(start + r) * nf + f];
            }
            vm &= vm - 1;
        }
        self.precompute_opt::<F32, M, G>(cols, n, masks, 0);
        let all = M::from_width(n);
        let base_addr = self.nodes.as_ptr() as usize;
        for tree in &self.trees {
            self.touch_pe::<F32, M>(
                tree.node_start as usize,
                cf,
                masks,
                0,
                all,
                visits,
                lines,
                base_addr,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn touch_pe<const F32: bool, M: RowMask>(
        &self,
        base: usize,
        cf: &[f64],
        masks: &[M],
        mut idx: usize,
        mut row_mask: M,
        visits: &mut [u64],
        lines: &mut Vec<u32>,
        base_addr: usize,
    ) {
        let nodes = self.nodes.as_slice();
        loop {
            let a = base + idx;
            visits[a] += 1;
            lines.push(((base_addr % 64 + a * 16) / 64) as u32);
            let node = &nodes[a];
            if node.is_leaf() {
                return;
            }
            if node.is_constant() {
                let gl = self.eval_split::<F32>(node, cf);
                idx = if gl == node.heavy_is_left() {
                    idx + 1
                } else {
                    node.skip as usize
                };
                continue;
            }
            let pm = masks[node.varying_pred_id as usize];
            let (lm, rm) = (row_mask & pm, row_mask & !pm);
            let (heavy, light) = if node.heavy_is_left() {
                (lm, rm)
            } else {
                (rm, lm)
            };
            if light.is_zero() {
                idx += 1;
                continue;
            }
            if heavy.is_zero() {
                idx = node.skip as usize;
                continue;
            }
            self.touch_pe::<F32, M>(
                base,
                cf,
                masks,
                node.skip as usize,
                light,
                visits,
                lines,
                base_addr,
            );
            idx += 1;
            row_mask = heavy;
        }
    }

    /// Move the node pool into memory advised for transparent huge pages (2 MiB).
    /// Returns the number of bytes inside 2 MiB-aligned advised ranges.
    pub fn hugepage_nodes(&mut self) -> usize {
        #[cfg(target_os = "linux")]
        {
            const HP: usize = 1 << 21;
            let n = self.nodes.len();
            let bytes = n * std::mem::size_of::<crate::forest::Node>();
            let layout = std::alloc::Layout::from_size_align(bytes.max(HP), HP).unwrap();
            unsafe {
                let p = std::alloc::alloc(layout);
                assert!(!p.is_null());
                let len = layout.size() & !(HP - 1);
                libc::madvise(p.cast(), len, libc::MADV_HUGEPAGE);
                std::ptr::copy_nonoverlapping(self.nodes.as_ptr().cast::<u8>(), p, bytes);
                let v = Vec::from_raw_parts(p.cast::<crate::forest::Node>(), n, layout.size() / 16);
                let old = std::mem::replace(&mut self.nodes, v);
                drop(old);
                return len.min(bytes);
            }
        }
        #[allow(unreachable_code)]
        0
    }

    /// Store each internal node's children kinds (walk / leaf / varying) in fields the
    /// kernels never read for that node type: varying_pred_id of constant nodes and
    /// cat_n_words of varying nodes. Lets the walk decide "leave the constant run?"
    /// from the parent it already holds instead of the child it has yet to load.
    pub fn add_child_hints(&mut self) {
        let ranges: Vec<(usize, usize)> = self
            .trees
            .iter()
            .map(|t| (t.node_start as usize, t.node_count as usize))
            .collect();
        for (base, cnt) in ranges {
            for i in 0..cnt {
                let a = base + i;
                let n = self.nodes[a];
                if n.is_leaf() {
                    continue;
                }
                let kh = node_kind(&self.nodes[a + 1]);
                let kl = node_kind(&self.nodes[base + n.skip as usize]);
                let code = kh | (kl << 2);
                if n.is_constant() {
                    self.nodes[a].varying_pred_id = u16::from(code);
                } else {
                    self.nodes[a].cat_n_words = code;
                }
            }
        }
    }

    /// Branch-free split evaluation for numeric and inline-categorical nodes: both
    /// outcomes are computed and selected, so the per-step "is it categorical?" branch
    /// disappears. Pool-bitset categoricals (more than 32 categories) take the slow path.
    #[inline(always)]
    fn eval_fast<const F32: bool>(&self, node: &crate::forest::Node, cf: &[f64]) -> bool {
        use crate::forest::{FLAG_CATEGORICAL, FLAG_INLINE_CAT};
        let fl = node.flags;
        if fl & (FLAG_CATEGORICAL | FLAG_INLINE_CAT) == FLAG_CATEGORICAL {
            return self.eval_split::<F32>(node, cf);
        }
        let x = unsafe { *cf.get_unchecked(node.feature as usize) };
        let dl = node.default_left();
        let num = threshold_go_left::<F32>(x, node.value);
        // Same rule as `eval_split`: f32 models round the category first; negative and
        // out-of-range categories take the non-membership branch (NaN handled below).
        let xc = if F32 { f64::from(x as f32) } else { x };
        let cat = xc as i32;
        let word = node.value as u32;
        let in_range = xc >= 0.0 && (0..32).contains(&cat);
        let bit = (word >> (cat as u32 & 31)) & 1 != 0;
        let catv = in_range & bit;
        let v = std::hint::select_unpredictable(fl & FLAG_CATEGORICAL != 0, catv, num);
        std::hint::select_unpredictable(x.is_nan(), dl, v)
    }

    fn pe_fastcat<const F32: bool, M: RowMask, const G: usize, const LEAF: u8>(
        &self,
        cx: &mut Cx<M>,
        mut idx: usize,
        mut row_mask: M,
    ) {
        if row_mask.is_zero() {
            return;
        }
        let nodes = self.nodes.as_slice();
        let base = cx.base;
        loop {
            loop {
                let node = unsafe { nodes.get_unchecked(base + idx) };
                if !node.is_walkable() {
                    break;
                }
                let gl = self.eval_fast::<F32>(node, cx.const_features);
                idx = if gl == node.heavy_is_left() {
                    idx + 1
                } else {
                    node.skip as usize
                };
            }
            let node = unsafe { nodes.get_unchecked(base + idx) };
            if node.is_leaf() {
                leaf_add::<M, G, LEAF>(cx, node.value, row_mask);
                return;
            }
            let pm = unsafe { *cx.masks.get_unchecked(node.varying_pred_id as usize) };
            let (lm, rm) = (row_mask & pm, row_mask & !pm);
            let (heavy, light) = if node.heavy_is_left() {
                (lm, rm)
            } else {
                (rm, lm)
            };
            if light.is_zero() {
                idx += 1;
                continue;
            }
            if heavy.is_zero() {
                idx = node.skip as usize;
                continue;
            }
            self.pe_fastcat::<F32, M, G, LEAF>(cx, node.skip as usize, light);
            idx += 1;
            row_mask = heavy;
        }
    }

    /// Walk using child-kind hints (requires `add_child_hints`): the loop-exit and
    /// leaf-vs-varying branches depend only on the parent node already in registers.
    fn pe_hint<const F32: bool, M: RowMask, const G: usize, const LEAF: u8, const FAST: bool>(
        &self,
        cx: &mut Cx<M>,
        mut idx: usize,
        mut row_mask: M,
        mut kind: u8,
    ) {
        if row_mask.is_zero() {
            return;
        }
        let nodes = self.nodes.as_slice();
        let base = cx.base;
        loop {
            if kind == K_WALK {
                loop {
                    let node = unsafe { nodes.get_unchecked(base + idx) };
                    let gl = if FAST {
                        self.eval_fast::<F32>(node, cx.const_features)
                    } else {
                        self.eval_split::<F32>(node, cx.const_features)
                    };
                    let h = node.varying_pred_id as u8;
                    if gl == node.heavy_is_left() {
                        idx += 1;
                        kind = h & 3;
                    } else {
                        idx = node.skip as usize;
                        kind = (h >> 2) & 3;
                    }
                    if kind != K_WALK {
                        break;
                    }
                }
            }
            let node = unsafe { nodes.get_unchecked(base + idx) };
            if kind == K_LEAF {
                leaf_add::<M, G, LEAF>(cx, node.value, row_mask);
                return;
            }
            let hv = node.cat_n_words;
            let pm = unsafe { *cx.masks.get_unchecked(node.varying_pred_id as usize) };
            let (lm, rm) = (row_mask & pm, row_mask & !pm);
            let (heavy, light) = if node.heavy_is_left() {
                (lm, rm)
            } else {
                (rm, lm)
            };
            if light.is_zero() {
                idx += 1;
                kind = hv & 3;
                continue;
            }
            if heavy.is_zero() {
                idx = node.skip as usize;
                kind = (hv >> 2) & 3;
                continue;
            }
            self.pe_hint::<F32, M, G, LEAF, FAST>(cx, node.skip as usize, light, (hv >> 2) & 3);
            idx += 1;
            row_mask = heavy;
            kind = hv & 3;
        }
    }

    /// Rewrite constant numerical splits so that "take the fall-through child" is always
    /// `v <= t'` with v drawn from a per-group table holding x and -x (or their f32
    /// images for XGBoost): heavy-left nodes keep (f, t) [F32: t' = prev(t)], heavy-right
    /// nodes use (f + nf, -next_up(t)) [F32: -t]. Exact for every non-NaN x; NaN keeps the
    /// default-direction rule. Nodes with infinite thresholds get flag 0x80 and keep the
    /// original evaluation. Only `pe_flip` may run on a transformed forest.
    pub fn flip_transform(&mut self) -> usize {
        let nf = self.config.n_features;
        assert!(2 * nf <= 256);
        let f32m = self.threshold_type == ThresholdType::F32;
        let mut slow = 0;
        for n in &mut self.nodes {
            if n.is_leaf() || !n.is_constant() || n.is_categorical() {
                continue;
            }
            let t = n.value;
            if !t.is_finite() {
                n.flags |= FLAG_SLOW;
                slow += 1;
                continue;
            }
            let hil = n.heavy_is_left();
            if f32m {
                let t32 = t as f32;
                if !t32.is_finite() {
                    n.flags |= FLAG_SLOW;
                    slow += 1;
                    continue;
                }
                if hil {
                    n.value = f64::from(t32.next_down());
                } else {
                    n.value = f64::from(-t32);
                    n.feature += nf as u16;
                }
            } else if hil {
                // x <= t: unchanged.
            } else {
                n.value = -t.next_up();
                n.feature += nf as u16;
            }
        }
        slow
    }

    /// Fall-through decision on a transformed node.
    #[inline(always)]
    fn flip_ft<const F32: bool>(
        &self,
        node: &crate::forest::Node,
        cfx: &[f64; 256],
        cf: &[f64],
    ) -> bool {
        use crate::forest::FLAG_CATEGORICAL;
        if node.flags & (FLAG_CATEGORICAL | FLAG_SLOW) != 0 {
            return self.eval_split::<F32>(node, cf) == node.heavy_is_left();
        }
        let v = unsafe { *cfx.get_unchecked(node.feature as usize) };
        if v <= node.value {
            true
        } else if v.is_nan() {
            node.default_left() == node.heavy_is_left()
        } else {
            false
        }
    }

    fn pe_flip<const F32: bool, M: RowMask, const G: usize, const LEAF: u8>(
        &self,
        cx: &mut Cx<M>,
        cfx: &[f64; 256],
        mut idx: usize,
        mut row_mask: M,
        mut kind: u8,
    ) {
        if row_mask.is_zero() {
            return;
        }
        let nodes = self.nodes.as_slice();
        let base = cx.base;
        loop {
            if kind == K_WALK {
                loop {
                    let node = unsafe { nodes.get_unchecked(base + idx) };
                    let h = node.varying_pred_id as u8;
                    if self.flip_ft::<F32>(node, cfx, cx.const_features) {
                        idx += 1;
                        kind = h & 3;
                    } else {
                        idx = node.skip as usize;
                        kind = (h >> 2) & 3;
                    }
                    if kind != K_WALK {
                        break;
                    }
                }
            }
            let node = unsafe { nodes.get_unchecked(base + idx) };
            if kind == K_LEAF {
                leaf_add::<M, G, LEAF>(cx, node.value, row_mask);
                return;
            }
            let hv = node.cat_n_words;
            let pm = unsafe { *cx.masks.get_unchecked(node.varying_pred_id as usize) };
            let (lm, rm) = (row_mask & pm, row_mask & !pm);
            let (heavy, light) = if node.heavy_is_left() {
                (lm, rm)
            } else {
                (rm, lm)
            };
            if light.is_zero() {
                idx += 1;
                kind = hv & 3;
                continue;
            }
            if heavy.is_zero() {
                idx = node.skip as usize;
                kind = (hv >> 2) & 3;
                continue;
            }
            self.pe_flip::<F32, M, G, LEAF>(cx, cfx, node.skip as usize, light, (hv >> 2) & 3);
            idx += 1;
            row_mask = heavy;
            kind = hv & 3;
        }
    }

    /// Smallest fixed-point scale e such that every leaf value is an exact integer
    /// multiple of 2^-e, provided the sum over all trees cannot overflow i128.
    /// Returns None when the model's leaf exponents span too wide a range.
    pub fn exact_scale(&self) -> Option<i32> {
        let mut min_ex = i32::MAX;
        let mut max_top = i32::MIN;
        for n in &self.nodes {
            if !n.is_leaf() || n.value == 0.0 {
                continue;
            }
            if !n.value.is_finite() {
                return None;
            }
            let bits = n.value.to_bits();
            let ef = ((bits >> 52) & 0x7ff) as i32;
            let frac = bits & ((1u64 << 52) - 1);
            let (m, ex) = if ef == 0 {
                (frac, -1074)
            } else {
                (frac | (1u64 << 52), ef - 1075)
            };
            min_ex = min_ex.min(ex + m.trailing_zeros() as i32);
            max_top = max_top.max(ex + 64 - m.leading_zeros() as i32);
        }
        if min_ex == i32::MAX {
            return Some(0);
        }
        let e = -min_ex;
        let tree_bits = (usize::BITS - self.trees.len().leading_zeros()) as i32;
        if max_top + e + tree_bits + 1 <= 126 {
            Some(e)
        } else {
            None
        }
    }

    /// `pe_flip` with the recursion replaced by an explicit stack of pending
    /// (heavy child, mask, kind) entries. Same visit order: light side first.
    fn pe_flip_stack<
        const F32: bool,
        M: RowMask,
        const G: usize,
        const LEAF: u8,
        const UNINIT: bool,
    >(
        &self,
        cx: &mut Cx<M>,
        cfx: &[f64; 256],
        idx0: usize,
        mask0: M,
        kind0: u8,
    ) {
        if mask0.is_zero() {
            return;
        }
        let nodes = self.nodes.as_slice();
        let base = cx.base;
        // UNINIT = false reproduces the original zero-initialised stack (~1.2 KB of
        // stores per tree); entries are always written before they are read.
        let mut st_idx: [MaybeUninit<u16>; 64] = [const { MaybeUninit::uninit() }; 64];
        let mut st_mask: [MaybeUninit<M>; 64] = [const { MaybeUninit::uninit() }; 64];
        let mut st_kind: [MaybeUninit<u8>; 64] = [const { MaybeUninit::uninit() }; 64];
        if !UNINIT {
            for i in 0..64 {
                st_idx[i].write(0);
                st_mask[i].write(M::ZERO);
                st_kind[i].write(0);
            }
        }
        let mut sp = 0usize;
        let (mut idx, mut row_mask, mut kind) = (idx0, mask0, kind0);
        loop {
            loop {
                if kind == K_WALK {
                    loop {
                        let node = unsafe { nodes.get_unchecked(base + idx) };
                        let h = node.varying_pred_id as u8;
                        if self.flip_ft::<F32>(node, cfx, cx.const_features) {
                            idx += 1;
                            kind = h & 3;
                        } else {
                            idx = node.skip as usize;
                            kind = (h >> 2) & 3;
                        }
                        if kind != K_WALK {
                            break;
                        }
                    }
                }
                let node = unsafe { nodes.get_unchecked(base + idx) };
                if kind == K_LEAF {
                    leaf_add::<M, G, LEAF>(cx, node.value, row_mask);
                    break;
                }
                let hv = node.cat_n_words;
                let pm = unsafe { *cx.masks.get_unchecked(node.varying_pred_id as usize) };
                let (lm, rm) = (row_mask & pm, row_mask & !pm);
                let (heavy, light) = if node.heavy_is_left() {
                    (lm, rm)
                } else {
                    (rm, lm)
                };
                if light.is_zero() {
                    idx += 1;
                    kind = hv & 3;
                    continue;
                }
                if heavy.is_zero() {
                    idx = node.skip as usize;
                    kind = (hv >> 2) & 3;
                    continue;
                }
                if sp == 64 {
                    // Trees may be up to 256 edges deep: past the explicit stack, finish the
                    // light side recursively (same visit order) and continue with the heavy side.
                    self.pe_flip::<F32, M, G, LEAF>(
                        cx,
                        cfx,
                        node.skip as usize,
                        light,
                        (hv >> 2) & 3,
                    );
                    idx += 1;
                    row_mask = heavy;
                    kind = hv & 3;
                    continue;
                }
                unsafe {
                    st_idx.get_unchecked_mut(sp).write((idx + 1) as u16);
                    st_mask.get_unchecked_mut(sp).write(heavy);
                    st_kind.get_unchecked_mut(sp).write(hv & 3);
                }
                sp += 1;
                idx = node.skip as usize;
                row_mask = light;
                kind = (hv >> 2) & 3;
            }
            if sp == 0 {
                return;
            }
            sp -= 1;
            unsafe {
                idx = st_idx.get_unchecked(sp).assume_init_read() as usize;
                row_mask = st_mask.get_unchecked(sp).assume_init_read();
                kind = st_kind.get_unchecked(sp).assume_init_read();
            }
        }
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn tree_ranges(&self) -> Vec<(usize, usize)> {
        self.trees
            .iter()
            .map(|t| (t.node_start as usize, t.node_count as usize))
            .collect()
    }
}

#[inline(always)]
fn leaf_add<M: RowMask, const G: usize, const LEAF: u8>(cx: &mut Cx<M>, v: f64, m: M) {
    let start = cx.start;
    match LEAF {
        1 => {
            let mm = m.to_u128();
            let (mut lo, mut hi) = (mm as u64, (mm >> 64) as u64);
            while lo != 0 {
                let r = lo.trailing_zeros() as usize;
                unsafe {
                    *cx.results.get_unchecked_mut(start + r) += v;
                }
                lo &= lo - 1;
            }
            while hi != 0 {
                let r = hi.trailing_zeros() as usize;
                unsafe {
                    *cx.results.get_unchecked_mut(start + 64 + r) += v;
                }
                hi &= hi - 1;
            }
        }
        2 => {
            #[cfg(target_feature = "avx512f")]
            unsafe {
                avx512_masked_add::<G>(cx.results.as_mut_ptr().add(start), v, m.to_u128());
            }
            #[cfg(not(target_feature = "avx512f"))]
            {
                let mut mm = m;
                while !mm.is_zero() {
                    let r = mm.trailing_zeros() as usize;
                    unsafe {
                        *cx.results.get_unchecked_mut(start + r) += v;
                    }
                    mm = mm.clear_lowest();
                }
            }
        }
        3 => {
            let mm = m.to_u128();
            let pc = mm.count_ones();
            let mut s = mm & !(mm << 1);
            if 2 * s.count_ones() <= pc {
                let mut e = mm & !(mm >> 1);
                while s != 0 {
                    let a = s.trailing_zeros() as usize;
                    let b = e.trailing_zeros() as usize + 1;
                    unsafe {
                        *cx.diff.f.get_unchecked_mut(a) += v;
                        *cx.diff.f.get_unchecked_mut(b) -= v;
                    }
                    s &= s.wrapping_sub(1);
                    e &= e.wrapping_sub(1);
                }
            } else {
                let mut mm = m;
                while !mm.is_zero() {
                    let r = mm.trailing_zeros() as usize;
                    unsafe {
                        *cx.results.get_unchecked_mut(start + r) += v;
                    }
                    mm = mm.clear_lowest();
                }
            }
        }
        5 => {
            cx.diff.f[0] += v;
        }
        4 => {
            let x = to_fixed(v, cx.diff.e);
            let mm = m.to_u128();
            let mut s = mm & !(mm << 1);
            let mut e = mm & !(mm >> 1);
            while s != 0 {
                let a = s.trailing_zeros() as usize;
                let b = e.trailing_zeros() as usize + 1;
                unsafe {
                    let pa = cx.diff.x.get_unchecked_mut(a);
                    *pa = pa.wrapping_add(x);
                    let pb = cx.diff.x.get_unchecked_mut(b);
                    *pb = pb.wrapping_sub(x);
                }
                s &= s.wrapping_sub(1);
                e &= e.wrapping_sub(1);
            }
        }
        _ => {
            let mut mm = m;
            while !mm.is_zero() {
                let r = mm.trailing_zeros() as usize;
                unsafe {
                    *cx.results.get_unchecked_mut(start + r) += v;
                }
                mm = mm.clear_lowest();
            }
        }
    }
}

/// v * 2^e as an exact integer (caller guarantees v * 2^e is integral and fits; see
/// `Forest::exact_scale`). Pure bit manipulation: no float rounding, no libcall.
#[inline(always)]
pub(crate) fn to_fixed(v: f64, e: i32) -> i128 {
    let bits = v.to_bits();
    let ef = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (m, ex) = if ef == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), ef - 1075)
    };
    let sh = ex + e;
    let mag = if sh >= 0 {
        (m as i128) << sh
    } else {
        m.checked_shr((-sh) as u32).unwrap_or(0) as i128
    };
    if bits >> 63 != 0 { -mag } else { mag }
}

/// Correctly rounded f64 of acc * 2^-e (the int-to-float cast rounds to nearest even;
/// scaling by a power of two is exact for normal results).
#[inline(always)]
pub(crate) fn fixed_to_f64(acc: i128, e: i32) -> f64 {
    let mut r = acc as f64;
    let mut k = e;
    while k > 0 {
        let step = k.min(1000);
        r *= f64::from_bits(((1023 - step) as u64) << 52);
        k -= step;
    }
    r
}

/// Bit-exact masked accumulation: each selected row gets exactly one `+= v`, as in the
/// scalar loop. Masked loads/stores touch only selected lanes, so rows past the group
/// end are never read or written.
#[cfg(target_feature = "avx512f")]
#[inline(always)]
unsafe fn avx512_masked_add<const G: usize>(p: *mut f64, v: f64, m: u128) {
    use std::arch::x86_64::{
        _mm512_add_pd, _mm512_mask_storeu_pd, _mm512_maskz_loadu_pd, _mm512_set1_pd,
    };
    unsafe {
        let vb = _mm512_set1_pd(v);
        let chunks = G.div_ceil(8);
        for c in 0..chunks {
            let k = (m >> (8 * c)) as u8;
            if k != 0 {
                let q = p.add(8 * c);
                let a = _mm512_maskz_loadu_pd(k, q);
                _mm512_mask_storeu_pd(q, k, _mm512_add_pd(a, vb));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compact 8-byte nodes with per-group threshold ranks
// ---------------------------------------------------------------------------
//
// Numerical constant splits store the rank of their threshold among the
// feature's sorted unique thresholds. Per group, each constant feature value is
// converted once to r = #{t : t < x} (F64, `x <= t`) or #{t : t <= x} (F32,
// `x < t`), and the split is `rank >= r`: exactly the original comparison, so
// results stay bit-identical. Leaves store an index into a dense leaf-value
// array; varying splits store their predicate id; categorical splits fall back
// to the original 16-byte node.

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CNode {
    pub payload: u32,
    pub skip: i16,
    pub feature: u8,
    pub flags: u8,
}
const _: () = assert!(std::mem::size_of::<CNode>() == 8);
const CF_LEAF: u8 = 0x80;
const CF_RANK_NAN: u32 = u32::MAX;

pub struct CompactForest {
    pub nodes: Vec<CNode>,
    leaf_vals: Vec<f64>,
    thr64: Vec<Vec<f64>>,
    thr32: Vec<Vec<f32>>,
    ranks: Vec<u32>,
}

impl CompactForest {
    pub fn bytes(&self) -> usize {
        self.nodes.len() * 8 + self.leaf_vals.len() * 8
    }

    /// Move the compact pool into memory advised for 2 MiB transparent huge pages.
    pub fn hugepage_nodes(&mut self) -> usize {
        #[cfg(target_os = "linux")]
        {
            const HP: usize = 1 << 21;
            let n = self.nodes.len();
            let bytes = n * 8;
            let layout = std::alloc::Layout::from_size_align(bytes.max(HP), HP).unwrap();
            unsafe {
                let p = std::alloc::alloc(layout);
                assert!(!p.is_null());
                let len = layout.size() & !(HP - 1);
                libc::madvise(p.cast(), len, libc::MADV_HUGEPAGE);
                std::ptr::copy_nonoverlapping(self.nodes.as_ptr().cast::<u8>(), p, bytes);
                let v = Vec::from_raw_parts(p.cast::<CNode>(), n, layout.size() / 8);
                drop(std::mem::replace(&mut self.nodes, v));
                return len.min(bytes);
            }
        }
        #[allow(unreachable_code)]
        0
    }
}

impl Forest {
    pub fn compact(&self) -> CompactForest {
        let nf = self.config.n_features;
        assert!(nf <= 256);
        let f32m = self.threshold_type == ThresholdType::F32;
        let mut thr64: Vec<Vec<f64>> = vec![Vec::new(); nf];
        let mut thr32: Vec<Vec<f32>> = vec![Vec::new(); nf];
        for n in &self.nodes {
            if !n.is_leaf() && n.is_constant() && !n.is_categorical() {
                assert!(!n.value.is_nan(), "NaN threshold");
                let f = n.feature as usize;
                if f32m {
                    thr32[f].push(n.value as f32)
                } else {
                    thr64[f].push(n.value)
                }
            }
        }
        for v in &mut thr64 {
            v.sort_by(f64::total_cmp);
            v.dedup_by(|a, b| a.to_bits() == b.to_bits());
        }
        for v in &mut thr32 {
            v.sort_by(f32::total_cmp);
            v.dedup_by(|a, b| a.to_bits() == b.to_bits());
        }
        let mut nodes = Vec::with_capacity(self.nodes.len());
        let mut leaf_vals = Vec::new();
        for (abs, n) in self.nodes.iter().enumerate() {
            if n.is_leaf() {
                nodes.push(CNode {
                    payload: leaf_vals.len() as u32,
                    skip: -1,
                    feature: 0,
                    flags: CF_LEAF,
                });
                leaf_vals.push(n.value);
                continue;
            }
            let payload = if !n.is_constant() {
                u32::from(n.varying_pred_id)
            } else if n.is_categorical() {
                abs as u32
            } else {
                let f = n.feature as usize;
                if f32m {
                    let t = n.value as f32;
                    thr32[f].binary_search_by(|x| x.total_cmp(&t)).unwrap() as u32
                } else {
                    thr64[f]
                        .binary_search_by(|x| x.total_cmp(&n.value))
                        .unwrap() as u32
                }
            };
            nodes.push(CNode {
                payload,
                skip: n.skip,
                feature: n.feature as u8,
                flags: n.flags,
            });
        }
        CompactForest {
            nodes,
            leaf_vals,
            thr64,
            thr32,
            ranks: vec![0; nf],
        }
    }

    pub fn predict_compact(
        &self,
        cf: &mut CompactForest,
        ws: &mut OptWorkspace,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
        flags: &OptFlags,
    ) {
        self.opt_guard(
            data,
            Some(results.len()),
            start,
            end,
            mask_cap(self.config.max_group_width),
        );
        let ps = &mut ws.prefix_starts;
        let diff = &mut *ws.diff;
        let avx = flags.leaf == LeafMode::Avx512;
        macro_rules! go {
            ($f32:literal, $m:ty, $g:literal, $masks:expr, $cols:expr, $sn:expr) => {
                if avx {
                    self.compact_core::<$f32, $m, $g, 2>(
                        cf, data, results, start, end, $masks, $cols, ps, diff, $sn, flags,
                    )
                } else {
                    self.compact_core::<$f32, $m, $g, 0>(
                        cf, data, results, start, end, $masks, $cols, ps, diff, $sn, flags,
                    )
                }
            };
        }
        match (self.threshold_type, self.config.max_group_width) {
            (ThresholdType::F64, 0..=32) => go!(
                false,
                u32,
                32,
                &mut ws.m32[..],
                &mut *ws.c32,
                &mut ws.sched_n[0]
            ),
            (ThresholdType::F64, 33..=64) => go!(
                false,
                u64,
                64,
                &mut ws.m64[..],
                &mut *ws.c64,
                &mut ws.sched_n[1]
            ),
            (ThresholdType::F64, _) => go!(
                false,
                u128,
                128,
                &mut ws.m128[..],
                &mut *ws.c128,
                &mut ws.sched_n[2]
            ),
            (ThresholdType::F32, 0..=32) => go!(
                true,
                u32,
                32,
                &mut ws.m32[..],
                &mut *ws.c32,
                &mut ws.sched_n[0]
            ),
            (ThresholdType::F32, 33..=64) => go!(
                true,
                u64,
                64,
                &mut ws.m64[..],
                &mut *ws.c64,
                &mut ws.sched_n[1]
            ),
            (ThresholdType::F32, _) => go!(
                true,
                u128,
                128,
                &mut ws.m128[..],
                &mut *ws.c128,
                &mut ws.sched_n[2]
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compact_core<const F32: bool, M: RowMask, const G: usize, const LEAF: u8>(
        &self,
        cfo: &mut CompactForest,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
        masks: &mut [M],
        cols: &mut [[f64; G]; 64],
        prefix_starts: &mut [u16],
        diff: &mut Acc,
        sched_n: &mut usize,
        flags: &OptFlags,
    ) {
        let n = end - start;
        let nf = self.config.n_features;
        let const_features = unsafe { data.get_unchecked(start * nf..(start + 1) * nf) };
        let sched = flags.schedule_features & self.config.varying_mask;
        let cached = sched != 0 && *sched_n == n;
        let skip = if cached { sched } else { 0 };
        let mut vm = self.config.varying_mask & !skip;
        while vm != 0 {
            let f = vm.trailing_zeros() as usize;
            for r in 0..n {
                cols[f][r] = unsafe { *data.get_unchecked((start + r) * nf + f) };
            }
            vm &= vm - 1;
        }
        self.precompute_opt::<F32, M, G>(cols, n, masks, skip);
        if sched != 0 {
            *sched_n = n;
        }
        // Per-group threshold ranks for every feature (constant splits only use them).
        for f in 0..nf {
            let x = const_features[f];
            cfo.ranks[f] = if x.is_nan() {
                CF_RANK_NAN
            } else if F32 {
                let x32 = x as f32;
                cfo.thr32[f].partition_point(|t| *t <= x32) as u32
            } else {
                cfo.thr64[f].partition_point(|t| *t < x) as u32
            };
        }
        results[start..end].fill(0.0);
        let all: M = M::from_width(n);
        let cn = cfo.nodes.as_slice();
        let ranks = cfo.ranks.as_slice();
        if !self.prefix_groups.is_empty() {
            let k = self.prefix_depth;
            prefix_starts.fill(0);
            for group in &self.prefix_groups {
                let rep = group.node_base as usize;
                let mut bail = k;
                for j in 0..k {
                    let c = &cn[rep + j];
                    let gl = self.c_go_left::<F32>(c, ranks, const_features);
                    if gl != (c.flags & crate::forest::FLAG_HEAVY_IS_LEFT != 0) {
                        bail = j;
                        break;
                    }
                }
                for &t in &group.trees {
                    prefix_starts[t as usize] = if bail == k {
                        k as u16
                    } else {
                        cn[self.trees[t as usize].node_start as usize + bail].skip as u16
                    };
                }
            }
        }
        let leaf_vals = cfo.leaf_vals.as_slice();
        let mut cx = Cx::<M> {
            base: 0,
            const_features,
            masks,
            results,
            start,
            diff,
        };
        for (i, tree) in self.trees.iter().enumerate() {
            cx.base = tree.node_start as usize;
            self.c_pe::<F32, M, G, LEAF>(
                cn,
                leaf_vals,
                ranks,
                &mut cx,
                prefix_starts[i] as usize,
                all,
            );
        }
        self.finalize(&mut cx.results[start..end]);
    }

    #[inline(always)]
    fn c_go_left<const F32: bool>(&self, c: &CNode, ranks: &[u32], cf: &[f64]) -> bool {
        if c.flags & crate::forest::FLAG_CATEGORICAL != 0 {
            let n = unsafe { self.nodes.get_unchecked(c.payload as usize) };
            self.eval_split::<F32>(n, cf)
        } else {
            let r = unsafe { *ranks.get_unchecked(c.feature as usize) };
            if r == CF_RANK_NAN {
                c.flags & crate::forest::FLAG_DEFAULT_LEFT != 0
            } else {
                c.payload >= r
            }
        }
    }

    fn c_pe<const F32: bool, M: RowMask, const G: usize, const LEAF: u8>(
        &self,
        cn: &[CNode],
        leaf_vals: &[f64],
        ranks: &[u32],
        cx: &mut Cx<M>,
        mut idx: usize,
        mut row_mask: M,
    ) {
        if row_mask.is_zero() {
            return;
        }
        let base = cx.base;
        loop {
            loop {
                let c = unsafe { cn.get_unchecked(base + idx) };
                if c.flags & crate::forest::FLAG_WALKABLE == 0 {
                    break;
                }
                let gl = self.c_go_left::<F32>(c, ranks, cx.const_features);
                idx = if gl == (c.flags & crate::forest::FLAG_HEAVY_IS_LEFT != 0) {
                    idx + 1
                } else {
                    c.skip as usize
                };
            }
            let c = unsafe { cn.get_unchecked(base + idx) };
            if c.flags & CF_LEAF != 0 {
                let v = unsafe { *leaf_vals.get_unchecked(c.payload as usize) };
                leaf_add::<M, G, LEAF>(cx, v, row_mask);
                return;
            }
            let pm = unsafe { *cx.masks.get_unchecked(c.payload as usize) };
            let (lm, rm) = (row_mask & pm, row_mask & !pm);
            let (heavy, light) = if c.flags & crate::forest::FLAG_HEAVY_IS_LEFT != 0 {
                (lm, rm)
            } else {
                (rm, lm)
            };
            if light.is_zero() {
                idx += 1;
                continue;
            }
            if heavy.is_zero() {
                idx = c.skip as usize;
                continue;
            }
            self.c_pe::<F32, M, G, LEAF>(cn, leaf_vals, ranks, cx, c.skip as usize, light);
            idx += 1;
            row_mask = heavy;
        }
    }
}

const K_WALK: u8 = 0;
const K_LEAF: u8 = 1;
const K_VARY: u8 = 2;

#[inline(always)]
fn node_kind(n: &crate::forest::Node) -> u8 {
    if n.is_leaf() {
        K_LEAF
    } else if n.is_constant() {
        K_WALK
    } else {
        K_VARY
    }
}

/// Experimental flag (bit 7 of Node.flags): constant numerical split excluded from the
/// flip transform (infinite threshold); evaluated with the original rule.
const FLAG_SLOW: u8 = 0x80;

// ---------------------------------------------------------------------------
// Run-list masks: row sets as sorted disjoint intervals (panels of any width)
// ---------------------------------------------------------------------------
//
// For survival panels every varying feature is, within a group, monotone, unimodal
// or valley-shaped in the row index (time step, its square, remaining steps, x0*t,
// sin(pi*t/h)). The left set of any numerical threshold on such a feature is then at
// most two runs of rows, computed for all thresholds of a feature by a pointer sweep.
// Row masks become short run lists, partitions are run-list intersections, and leaf
// accumulation writes two fixed-point entries per run into a difference array. Cost
// per tree no longer depends on the number of rows; only the per-group sweep and the
// final prefix sum do. Groups whose varying features are not of these shapes are
// rejected (the caller falls back to the bitmask kernels).

pub const RUNS: usize = 16;

#[derive(Clone, Copy)]
pub struct RunMask {
    n: u8,
    s: [u16; RUNS],
    e: [u16; RUNS],
}

#[derive(Clone, Copy, Default)]
pub struct PredRuns {
    nl: u8,
    l: [(u16, u16); 2],
    nr: u8,
    r: [(u16, u16); 3],
}

impl RunMask {
    /// Branch-free intersection: every (row run, predicate run) pair is written and the
    /// output cursor advances only when the overlap is non-empty.
    #[inline(always)]
    fn and_bf(&self, b: &[(u16, u16)], overflow: &mut bool) -> Self {
        let mut out = Self {
            n: 0,
            s: [0; RUNS],
            e: [0; RUNS],
        };
        let mut k = 0usize;
        for i in 0..self.n as usize {
            let (s0, e0) = (self.s[i], self.e[i]);
            for &(bs, be) in b {
                if k == RUNS {
                    *overflow = true;
                    return out;
                }
                let lo = s0.max(bs);
                let hi = e0.min(be);
                out.s[k] = lo;
                out.e[k] = hi;
                k += usize::from(lo < hi);
            }
        }
        out.n = k as u8;
        out
    }

    #[inline(always)]
    fn all(n: usize) -> Self {
        let mut m = Self {
            n: 1,
            s: [0; RUNS],
            e: [0; RUNS],
        };
        m.e[0] = n as u16;
        m
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.n == 0
    }
    /// Intersection with a short sorted run list; sets *overflow when RUNS is exceeded.
    #[inline(always)]
    fn and(&self, b: &[(u16, u16)], overflow: &mut bool) -> Self {
        let mut out = Self {
            n: 0,
            s: [0; RUNS],
            e: [0; RUNS],
        };
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.n as usize && j < b.len() {
            let lo = self.s[i].max(b[j].0);
            let hi = self.e[i].min(b[j].1);
            if lo < hi {
                if (out.n as usize) == RUNS {
                    *overflow = true;
                    return out;
                }
                out.s[out.n as usize] = lo;
                out.e[out.n as usize] = hi;
                out.n += 1;
            }
            if self.e[i] < b[j].1 { i += 1 } else { j += 1 }
        }
        out
    }
}

pub struct RunWorkspace {
    preds: Vec<PredRuns>,
    xdiff: Vec<i128>,
    col: Vec<f64>,
    /// Row-major gather of all varying columns: cols[k * max_rows + r] for the k-th range.
    cols: Vec<f64>,
    prefix_starts: Vec<u16>,
    pub overflows: u64,
    pub rejected: u64,
    /// Measurement knock-outs: 0 full, 1 precompute only, 2 no leaf updates, 3 no epilogue.
    pub knock: u8,
    /// Optimised path: cached schedule-feature runs, single-pass gather, branch-free AND.
    pub fast: bool,
    /// Uninitialised explicit stack in `pe_runs` (no per-tree zeroing).
    pub uninit: bool,
    /// Varying features whose values depend only on the row index (cached per width).
    pub schedule: u128,
    sched_n: usize,
    max_rows: usize,
}

impl RunWorkspace {
    pub fn new(forest: &Forest, max_rows: usize) -> Self {
        assert!(
            max_rows < usize::from(u16::MAX),
            "run lists use u16 row positions"
        );
        Self {
            preds: vec![PredRuns::default(); forest.varying_predicates.len()],
            xdiff: vec![0; max_rows + 2],
            col: vec![0.0; max_rows],
            cols: vec![0.0; max_rows * forest.feature_ranges.len().max(1)],
            prefix_starts: vec![0; forest.trees.len()],
            overflows: 0,
            rejected: 0,
            knock: 0,
            fast: false,
            uninit: false,
            schedule: 0,
            sched_n: 0,
            max_rows,
        }
    }
}

/// Fast correctly rounded i128 -> f64 (64-bit window with a sticky bit), times 2^-e.
#[inline(always)]
pub(crate) fn fixed_to_f64_fast(acc: i128, e: i32) -> f64 {
    if acc == 0 {
        return 0.0;
    }
    let neg = acc < 0;
    let mag = acc.unsigned_abs();
    let lz = mag.leading_zeros() as i32;
    let width = 128 - lz;
    let (top, shift) = if width > 64 {
        let sh = (width - 64) as u32;
        let t = (mag >> sh) as u64;
        let sticky = u64::from(mag & ((1u128 << sh) - 1) != 0);
        (t | sticky, sh as i32)
    } else {
        (mag as u64, 0)
    };
    // u64 -> f64 rounds to nearest even; the sticky bit sits below the rounding position.
    let mut r = top as f64;
    let mut k = shift - e;
    while k != 0 {
        let step = k.clamp(-1000, 1000);
        r *= f64::from_bits(((1023 + step) as u64) << 52);
        k -= step;
    }
    if neg { -r } else { r }
}

impl Forest {
    /// Left-set run lists of every varying predicate for one group. Returns false when
    /// some varying feature is NaN, categorical, or not monotone/unimodal/valley-shaped.
    fn precompute_runs<const F32: bool>(
        &self,
        data: &[f64],
        start: usize,
        n: usize,
        ws: &mut RunWorkspace,
    ) -> bool {
        let nf = self.config.n_features;
        let cached = ws.fast && ws.schedule != 0 && ws.sched_n == n;
        if ws.fast {
            // One pass over the group's rows gathers every varying column that is needed.
            let nr = self.feature_ranges.len();
            let mr = ws.max_rows;
            for r in 0..n {
                let row = &data[(start + r) * nf..(start + r + 1) * nf];
                for (k, range) in self.feature_ranges.iter().enumerate() {
                    let f = range.feature as usize;
                    if cached && ws.schedule & (1u128 << f) != 0 {
                        continue;
                    }
                    unsafe {
                        *ws.cols.get_unchecked_mut(k * mr + r) = *row.get_unchecked(f);
                    }
                }
            }
            let _ = nr;
        }
        for (k, range) in self.feature_ranges.iter().enumerate() {
            if range.cat_start < range.cat_end {
                return false;
            }
            let f = range.feature as usize;
            if cached && ws.schedule & (1u128 << f) != 0 {
                continue;
            }
            let col = &mut ws.col[..n];
            if ws.fast {
                col.copy_from_slice(&ws.cols[k * ws.max_rows..k * ws.max_rows + n]);
                if col.iter().any(|v| v.is_nan()) {
                    return false;
                }
            } else {
                for r in 0..n {
                    let v = data[(start + r) * nf + f];
                    if v.is_nan() {
                        return false;
                    }
                    col[r] = v;
                }
            }
            // Shape: split point p such that [0, p] is one monotone part and (p, n) the other.
            let mut dir0 = 0i8;
            let mut turn: Option<usize> = None;
            for r in 1..n {
                let d = if col[r] > col[r - 1] {
                    1
                } else if col[r] < col[r - 1] {
                    -1
                } else {
                    0
                };
                if d == 0 {
                    continue;
                }
                if dir0 == 0 {
                    dir0 = d;
                } else if d != dir0 {
                    if turn.is_some() {
                        return false;
                    }
                    turn = Some(r - 1);
                    dir0 = d;
                }
            }
            // Parts: A = [0, pa), B = [pa, n). Each part is monotone; record its direction.
            let pa = turn.map_or(n, |p| p + 1);
            let dir_of = |lo: usize, hi: usize| -> i8 {
                for r in lo + 1..hi {
                    if col[r] > col[r - 1] {
                        return 1;
                    }
                    if col[r] < col[r - 1] {
                        return -1;
                    }
                }
                0
            };
            let (da, db) = (dir_of(0, pa), dir_of(pa, n));
            let go = |v: f64, t: f64| threshold_go_left::<F32>(v, t);
            // In a non-decreasing part, the rows going left form a prefix; in a
            // non-increasing part, a suffix. Counts grow with the threshold, so one
            // pointer per part sweeps the sorted thresholds.
            let preds = &self.varying_predicates[range.num_start as usize..range.num_end as usize];
            let (mut ca, mut cb) = (0usize, 0usize);
            for (k, pred) in preds.iter().enumerate() {
                let t = match pred {
                    VaryingPredicate::Num { threshold, .. } => *threshold,
                    _ => return false,
                };
                let la = pa;
                let lb = n - pa;
                // part A
                if da >= 0 {
                    while ca < la && go(col[ca], t) {
                        ca += 1;
                    }
                } else {
                    while ca < la && go(col[pa - 1 - ca], t) {
                        ca += 1;
                    }
                }
                if da == 0 && ca > 0 && ca < la {
                    ca = la;
                }
                // part B
                if db >= 0 {
                    while cb < lb && go(col[pa + cb], t) {
                        cb += 1;
                    }
                } else {
                    while cb < lb && go(col[n - 1 - cb], t) {
                        cb += 1;
                    }
                }
                if db == 0 && cb > 0 && cb < lb {
                    cb = lb;
                }
                // Left runs of each part.
                let ra = if ca == 0 {
                    None
                } else if da >= 0 {
                    Some((0, ca))
                } else {
                    Some((pa - ca, pa))
                };
                let rb = if cb == 0 {
                    None
                } else if db >= 0 {
                    Some((pa, pa + cb))
                } else {
                    Some((n - cb, n))
                };
                let mut pr = PredRuns::default();
                match (ra, rb) {
                    (None, None) => {}
                    (Some(a), None) | (None, Some(a)) => {
                        pr.l[0] = (a.0 as u16, a.1 as u16);
                        pr.nl = 1;
                    }
                    (Some(a), Some(b)) => {
                        if a.1 == b.0 {
                            pr.l[0] = (a.0 as u16, b.1 as u16);
                            pr.nl = 1;
                        } else {
                            pr.l[0] = (a.0 as u16, a.1 as u16);
                            pr.l[1] = (b.0 as u16, b.1 as u16);
                            pr.nl = 2;
                        }
                    }
                }
                // Complement within [0, n).
                let mut c = 0u16;
                for q in 0..pr.nl as usize {
                    let (s0, e0) = pr.l[q];
                    if s0 > c {
                        pr.r[pr.nr as usize] = (c, s0);
                        pr.nr += 1;
                    }
                    c = e0;
                }
                if (c as usize) < n {
                    pr.r[pr.nr as usize] = (c, n as u16);
                    pr.nr += 1;
                }
                ws.preds[range.num_start as usize + k] = pr;
            }
        }
        if ws.fast && ws.schedule != 0 {
            ws.sched_n = n;
        }
        true
    }

    /// Run-list kernel on a forest prepared with `add_child_hints` + `flip_transform`.
    /// Exact fixed-point accumulation (scale `e`). Returns false if the group was rejected
    /// (unsupported feature shape or run overflow); results are then left untouched.
    pub fn predict_runs(
        &self,
        ws: &mut RunWorkspace,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
        e: i32,
    ) -> bool {
        self.opt_guard(
            data,
            Some(results.len()),
            start,
            end,
            ws.max_rows.min(usize::from(u16::MAX) - 1),
        );
        match self.threshold_type {
            ThresholdType::F64 => self.runs_core::<false>(ws, data, results, start, end, e),
            ThresholdType::F32 => self.runs_core::<true>(ws, data, results, start, end, e),
        }
    }

    fn runs_core<const F32: bool>(
        &self,
        ws: &mut RunWorkspace,
        data: &[f64],
        results: &mut [f64],
        start: usize,
        end: usize,
        e: i32,
    ) -> bool {
        let n = end - start;
        let nf = self.config.n_features;
        if !self.precompute_runs::<F32>(data, start, n, ws) {
            ws.rejected += 1;
            return false;
        }
        let cf = &data[start * nf..(start + 1) * nf];
        let mut cfx = [0.0f64; 256];
        for f in 0..nf {
            let x = cf[f];
            if F32 {
                let x32 = x as f32;
                cfx[f] = f64::from(x32);
                cfx[f + nf] = f64::from(-x32);
            } else {
                cfx[f] = x;
                cfx[f + nf] = -x;
            }
        }
        let nodes = self.nodes.as_slice();
        if !self.prefix_groups.is_empty() {
            let k = self.prefix_depth;
            ws.prefix_starts.fill(0);
            for group in &self.prefix_groups {
                let rep = group.node_base as usize;
                let mut bail = k;
                for j in 0..k {
                    if !self.flip_ft::<F32>(&nodes[rep + j], &cfx, cf) {
                        bail = j;
                        break;
                    }
                }
                for &t in &group.trees {
                    ws.prefix_starts[t as usize] = if bail == k {
                        k as u16
                    } else {
                        nodes[self.trees[t as usize].node_start as usize + bail].skip as u16
                    };
                }
            }
        }
        if ws.knock == 1 {
            return true;
        }
        let all = RunMask::all(n);
        let mut overflow = false;
        let (noleaf, fast, un) = (ws.knock == 2, ws.fast, ws.uninit);
        for (i, tree) in self.trees.iter().enumerate() {
            let base = tree.node_start as usize;
            let s0 = ws.prefix_starts[i] as usize;
            let kind = node_kind(&nodes[base + s0]);
            match (fast, noleaf) {
                _ if un && fast && !noleaf => self.pe_runs::<F32, true, false, true>(
                    base,
                    &cfx,
                    cf,
                    &ws.preds,
                    &mut ws.xdiff,
                    e,
                    s0,
                    all,
                    kind,
                    &mut overflow,
                ),
                _ if un && fast && noleaf => self.pe_runs::<F32, true, true, true>(
                    base,
                    &cfx,
                    cf,
                    &ws.preds,
                    &mut ws.xdiff,
                    e,
                    s0,
                    all,
                    kind,
                    &mut overflow,
                ),
                (false, false) => self.pe_runs::<F32, false, false, false>(
                    base,
                    &cfx,
                    cf,
                    &ws.preds,
                    &mut ws.xdiff,
                    e,
                    s0,
                    all,
                    kind,
                    &mut overflow,
                ),
                (false, true) => self.pe_runs::<F32, false, true, false>(
                    base,
                    &cfx,
                    cf,
                    &ws.preds,
                    &mut ws.xdiff,
                    e,
                    s0,
                    all,
                    kind,
                    &mut overflow,
                ),
                (true, false) => self.pe_runs::<F32, true, false, false>(
                    base,
                    &cfx,
                    cf,
                    &ws.preds,
                    &mut ws.xdiff,
                    e,
                    s0,
                    all,
                    kind,
                    &mut overflow,
                ),
                (true, true) => self.pe_runs::<F32, true, true, false>(
                    base,
                    &cfx,
                    cf,
                    &ws.preds,
                    &mut ws.xdiff,
                    e,
                    s0,
                    all,
                    kind,
                    &mut overflow,
                ),
            }
        }
        if ws.knock == 3 {
            ws.xdiff[..=n].fill(0);
            return !overflow;
        }
        if overflow {
            ws.overflows += 1;
            ws.xdiff[..=n].fill(0);
            return false;
        }
        let mut acc: i128 = 0;
        let mut last_acc: i128 = 0;
        let mut last = 0.0f64;
        for r in 0..n {
            acc = acc.wrapping_add(ws.xdiff[r]);
            if r == 0 || acc != last_acc {
                last = fixed_to_f64_fast(acc, e);
                last_acc = acc;
            }
            results[start + r] = last;
        }
        self.finalize(&mut results[start..start + n]);
        ws.xdiff[..=n].fill(0);
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn pe_runs<const F32: bool, const BF: bool, const NOLEAF: bool, const UNINIT: bool>(
        &self,
        base: usize,
        cfx: &[f64; 256],
        cf: &[f64],
        preds: &[PredRuns],
        xdiff: &mut [i128],
        e: i32,
        idx0: usize,
        mask0: RunMask,
        kind0: u8,
        overflow: &mut bool,
    ) {
        let nodes = self.nodes.as_slice();
        let mut st: [MaybeUninit<(u16, RunMask, u8)>; 32] = [const { MaybeUninit::uninit() }; 32];
        if !UNINIT {
            for x in st.iter_mut() {
                x.write((
                    0,
                    RunMask {
                        n: 0,
                        s: [0; RUNS],
                        e: [0; RUNS],
                    },
                    0,
                ));
            }
        }
        let mut sp = 0usize;
        let (mut idx, mut row_mask, mut kind) = (idx0, mask0, kind0);
        loop {
            loop {
                if kind == K_WALK {
                    loop {
                        let node = unsafe { nodes.get_unchecked(base + idx) };
                        let h = node.varying_pred_id as u8;
                        if self.flip_ft::<F32>(node, cfx, cf) {
                            idx += 1;
                            kind = h & 3;
                        } else {
                            idx = node.skip as usize;
                            kind = (h >> 2) & 3;
                        }
                        if kind != K_WALK {
                            break;
                        }
                    }
                }
                let node = unsafe { nodes.get_unchecked(base + idx) };
                if kind == K_LEAF {
                    if NOLEAF {
                        unsafe {
                            let p = xdiff.get_unchecked_mut(0);
                            *p = p.wrapping_add(i128::from(row_mask.n));
                        }
                        break;
                    }
                    let x = to_fixed(node.value, e);
                    for q in 0..row_mask.n as usize {
                        let (a, b) = (row_mask.s[q] as usize, row_mask.e[q] as usize);
                        unsafe {
                            let pa = xdiff.get_unchecked_mut(a);
                            *pa = pa.wrapping_add(x);
                            let pb = xdiff.get_unchecked_mut(b);
                            *pb = pb.wrapping_sub(x);
                        }
                    }
                    break;
                }
                let hv = node.cat_n_words;
                let pr = unsafe { preds.get_unchecked(node.varying_pred_id as usize) };
                let (lm, rm) = if BF {
                    (
                        row_mask.and_bf(&pr.l[..pr.nl as usize], overflow),
                        row_mask.and_bf(&pr.r[..pr.nr as usize], overflow),
                    )
                } else {
                    (
                        row_mask.and(&pr.l[..pr.nl as usize], overflow),
                        row_mask.and(&pr.r[..pr.nr as usize], overflow),
                    )
                };
                let (heavy, light) = if node.heavy_is_left() {
                    (lm, rm)
                } else {
                    (rm, lm)
                };
                if light.is_zero() {
                    idx += 1;
                    kind = hv & 3;
                    continue;
                }
                if heavy.is_zero() {
                    idx = node.skip as usize;
                    kind = (hv >> 2) & 3;
                    continue;
                }
                if sp == st.len() {
                    *overflow = true;
                    return;
                }
                unsafe {
                    st.get_unchecked_mut(sp)
                        .write(((idx + 1) as u16, heavy, hv & 3))
                };
                sp += 1;
                idx = node.skip as usize;
                row_mask = light;
                kind = (hv >> 2) & 3;
            }
            if sp == 0 {
                return;
            }
            sp -= 1;
            let (i2, m2, k2) = unsafe { st.get_unchecked(sp).assume_init_read() };
            idx = i2 as usize;
            row_mask = m2;
            kind = k2;
        }
    }
}
