//! A call's groups through the kernel: per group, the bins of its first row and
//! prefix starts, the varying masks, the walk and resolve passes over chunks of
//! trees, the leaf pass and the finish, at the narrowest mask width that holds the group:
//! `u32` or `u64` by value, `Wide` masks up to [`WIDE_ROWS`] rows and runtime-width
//! masks of [`rows::units`] 128-bit units above, both in an arena. A single
//! row has bins for every feature, so its walk steps through varying splits as
//! constants: it needs no masks and stops only at leaves, which all carry its row.
//! Its one pass per chunk walks the roots in place and sums the leaf values from the
//! walk's words, without items, leaf entries or difference arrays.
//!
//! Scratch lives in [`Scratch`], allocated once per predictor. Trees are processed in
//! chunks whose working set across passes (walk queues, stops, words, leaf entries
//! and mask arena, and the predicates' masks) fits the chunk budget of
//! [`crate::rules::chunk_budget`] at the group's mask width, so scratch scales with
//! a chunk rather than the model, and wide masks get chunks of fewer trees.

use super::counters::{Clock, Stats};
use super::isa::Isa;
use super::leaves::leaf_pass_at;
use super::mask::rows::{self, Unit};
use super::mask::{Mask, Wide};
use super::passes::{Pass, Walks, walk_with};
use super::precompute::{Lefts, Rows, SWEEP_TURNS};
use super::resolve::{Carrier, Handle, Slot};
use super::scan::{NoScan, Scanner, Scans};
use super::scratch::{Common, Queues, Scratch, Sums, WIDE_ROWS, Width, Words, view};
use super::walk::{self, MAX_PAD, Pools, Steps};
use super::{Groups, WorkCounters};
use crate::exact::{self, Plan, WIDE_SUMS, WideRuns};
use crate::load::layout::LeafValues;
use crate::model::Model;

/// The kernel's runtime choices; production uses [`Options::default`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Options {
    /// Walk single rows' varying splits as constants, without masks.
    pub(crate) g1_constant: bool,
    /// Run each group at the narrowest mask width that holds it, rather than the
    /// widest the configuration allows.
    pub(crate) width_dispatch: bool,
    /// Precompute the varying predicates' masks; without it, each varying event scans
    /// its rows.
    pub(crate) precompute: bool,
    /// Precompute with the sorted-threshold sweep rather than by brute force.
    pub(crate) sweep: bool,
    /// Let event scans stop comparing early on monotonic features.
    pub(crate) monotonic: bool,
    /// Drop sides without rows at varying events.
    pub(crate) unsplit: bool,
    /// Walk every walk with this many lanes, rather than the rule's for each, and
    /// prefetch one item ahead unless `distance` says otherwise.
    pub(crate) lanes: Option<usize>,
    /// Prefetch this many items ahead, rather than the rule's distance.
    pub(crate) distance: Option<usize>,
    /// Prefetch once every this many rounds in the walk without varying stops,
    /// rather than at the rule's rate.
    pub(crate) rate: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            g1_constant: true,
            width_dispatch: true,
            precompute: true,
            sweep: true,
            monotonic: true,
            unsplit: true,
            lanes: None,
            distance: None,
            rate: None,
        }
    }
}

impl Model {
    /// [`Self::kernel_groups`] at level `X` ([`Isa`]), inside the level's trampoline, so
    /// that the code it inlines compiles with the level's features.
    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the call, its scratch and its variant"
    )]
    pub(super) fn kernel_groups_at<
        X: Isa,
        const F32: bool,
        const STATS: bool,
        const ABLATE: bool,
    >(
        &self,
        data: &[f64],
        out: &mut [f64],
        groups: Groups<'_>,
        s: &mut Scratch,
        options: Options,
        stats: &mut Stats,
        finalize: bool,
    ) {
        if X::NATIVE {
            return self.kernel_groups::<X, F32, STATS, ABLATE>(
                data, out, groups, s, options, stats, finalize,
            );
        }
        X::with8(
            #[inline(always)]
            |m: &Self, data, out, groups, s, options, stats, finalize| {
                m.kernel_groups::<X, F32, STATS, ABLATE>(
                    data, out, groups, s, options, stats, finalize,
                );
            },
            self,
            data,
            out,
            groups,
            s,
            options,
            stats,
            finalize,
        );
    }

    /// Tree sums, or with `finalize` outputs, of every group of a call, through the
    /// kernel. With `STATS`, the work and phase times are added to `stats`; with
    /// `ABLATE`, `options` apply, otherwise the defaults do. The input must have
    /// passed [`super::Predictor::check`].
    #[expect(
        clippy::too_many_arguments,
        reason = "the call, its scratch and its variant"
    )]
    fn kernel_groups<X: Isa, const F32: bool, const STATS: bool, const ABLATE: bool>(
        &self,
        data: &[f64],
        out: &mut [f64],
        groups: Groups<'_>,
        s: &mut Scratch,
        options: Options,
        stats: &mut Stats,
        finalize: bool,
    ) {
        let options = if ABLATE { options } else { Options::default() };
        let nf = self.config.n_features();
        let n_rows = out.len();
        for g in 0..groups.count(n_rows) {
            let (start, end) = groups.bounds(g, n_rows);
            let rows = &data[start * nf..end * nf];
            let results = &mut out[start..end];
            let constant = results.len() == 1 && options.g1_constant;
            let c = &mut s.common;
            let t = Clock::<STATS>::start();
            self.layout
                .bins
                .fill_with::<F32>(&rows[..nf], &mut c.bins, constant);
            let t = t.lap(&mut stats.phases.bins);
            self.prefix_starts_at::<X, STATS, ABLATE>(c, &mut stats.counters);
            t.lap(&mut stats.phases.prefix);
            if STATS {
                stats.counters.groups += 1;
                stats.counters.bins_computed += 1;
            }
            // The rows the group's masks hold: its own, or the widest group's.
            let held = if options.width_dispatch {
                results.len()
            } else {
                s.common.max_rows
            };
            let c = &mut s.common;
            macro_rules! group {
                ($m:ty, $c:ty, $q:expr, $u:expr) => {
                    self.masked_group_at::<X, F32, $m, $c, STATS, ABLATE>(
                        rows, results, c, $q, $u, options, stats,
                    )
                };
            }
            match Width::of(held) {
                _ if held > WIDE_ROWS => group!(Unit<X>, Slot, &mut s.slots, rows::units(held)),
                Width::W32 if constant => {
                    self.constant_group_at::<X, STATS, ABLATE>(results, c, options, stats);
                }
                Width::W32 => group!(u32, u32, &mut s.q32, 1),
                Width::W64 => group!(u64, u64, &mut s.q64, 1),
                Width::W128 => group!(Wide<2, X>, Handle, &mut s.handles, 1),
                Width::W256 => group!(Wide<4, X>, Handle, &mut s.handles, 1),
                Width::W512 => group!(Wide<8, X>, Handle, &mut s.handles, 1),
                Width::W1024 => group!(Wide<16, X>, Handle, &mut s.handles, 1),
                _ => unreachable!("wider groups take runtime-width masks"),
            }
            if finalize {
                let t = Clock::<STATS>::start();
                self.finalize_at::<X>(&mut out[start..end]);
                t.lap(&mut stats.phases.finish);
            }
        }
    }

    /// [`Self::prefix_starts`] at level `X` ([`Isa`]): where the build dispatches, out of line,
    /// with the body compiled inside the level's trampoline; elsewhere inlined.
    #[cfg_attr(not(dispatch), inline(always))]
    fn prefix_starts_at<X: Isa, const STATS: bool, const ABLATE: bool>(
        &self,
        c: &mut Common,
        counters: &mut WorkCounters,
    ) {
        if X::NATIVE {
            return self.prefix_starts::<STATS, ABLATE>(c, counters);
        }
        X::with3(
            #[inline(always)]
            |m: &Self, c, counters| m.prefix_starts::<STATS, ABLATE>(c, counters),
            self,
            c,
            counters,
        );
    }

    /// Each tree's start: its root, or after its prefix group's shared constant splits
    /// on the bins, the first node the group does not share.
    ///
    /// `ABLATE` only keeps the research builds' instantiation apart from
    /// production's, so that compiling `research` in cannot change production code.
    #[cfg_attr(dispatch, inline(always))]
    fn prefix_starts<const STATS: bool, const ABLATE: bool>(
        &self,
        c: &mut Common,
        counters: &mut WorkCounters,
    ) {
        let layout = &self.layout;
        for group in &self.prefix_groups {
            let (base, k) = (group.node_base as usize, group.depth as usize);
            let bail = (0..k)
                .find(|&j| !layout.nodes[base + j].goes_heavy(&c.bins))
                .unwrap_or(k);
            if STATS {
                counters.prefix_steps += (bail + 1).min(k) as u64;
            }
            let trees = group.first as usize..group.end as usize;
            let (roots, starts) = (&mut c.roots[trees.clone()], &self.tree_starts[trees]);
            if bail == k {
                // Every tree's rows follow the whole prefix: a vector add.
                for (root, &start) in roots.iter_mut().zip(starts) {
                    *root = start + k as u32;
                }
            } else {
                for (root, &start) in roots.iter_mut().zip(starts) {
                    *root = layout.nodes[start as usize + bail].child();
                }
            }
        }
    }

    fn pools(&self, bins: &[u32]) -> Pools {
        assert!(bins.len() >= self.layout.bins.n_slots());
        Pools {
            nodes: self.layout.nodes.as_ptr(),
            bins: bins.as_ptr(),
            dummy: self.layout.dummy(),
        }
    }

    /// [`Self::masked_group`] at level `X` ([`Isa`]): where the build dispatches, out of line,
    /// with the body compiled inside the level's trampoline; elsewhere inlined.
    #[cfg_attr(not(dispatch), inline(always))]
    #[cfg_attr(dispatch, inline(never))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the group, its scratch and its variant"
    )]
    fn masked_group_at<X: Isa, const F32: bool, M, C, const STATS: bool, const ABLATE: bool>(
        &self,
        rows: &[f64],
        results: &mut [f64],
        c: &mut Common,
        queues: &mut Queues<C>,
        runtime_units: usize,
        options: Options,
        stats: &mut Stats,
    ) where
        M: Mask + Words,
        C: Carrier<M>,
    {
        if X::NATIVE {
            return self.masked_group::<X, F32, M, C, STATS, ABLATE>(
                rows,
                results,
                c,
                queues,
                runtime_units,
                options,
                stats,
            );
        }
        X::with8(
            #[inline(always)]
            |m: &Self, rows, results, c, queues, u, options, stats| {
                m.masked_group::<X, F32, M, C, STATS, ABLATE>(
                    rows, results, c, queues, u, options, stats,
                );
            },
            self,
            rows,
            results,
            c,
            queues,
            runtime_units,
            options,
            stats,
        );
    }

    /// Tree sums of one group at mask type `M` carried by `C`, its masks of
    /// `runtime_units` units of `M` for runtime-width carriers and one otherwise. `ABLATE` keeps research's
    /// instantiation apart, as in [`Self::prefix_starts`].
    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline(never))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the group, its scratch and its variant"
    )]
    fn masked_group<X: Isa, const F32: bool, M, C, const STATS: bool, const ABLATE: bool>(
        &self,
        rows: &[f64],
        results: &mut [f64],
        c: &mut Common,
        queues: &mut Queues<C>,
        runtime_units: usize,
        options: Options,
        stats: &mut Stats,
    ) where
        M: Mask + Words,
        C: Carrier<M>,
    {
        let n = results.len();
        let np = self.varying_predicates.len();
        let (units, width) = if C::RUNTIME {
            (runtime_units, Width::of(128 * runtime_units))
        } else {
            (1, Width::of(M::WIDTH))
        };
        let t = Clock::<STATS>::start();
        // The all-rows mask, which leaves read, then each predicate's left and right
        // masks, which the splits with that heavy side read.
        let counters = &mut stats.counters;
        if C::RUNTIME {
            let words = 2 * units;
            let masks = rows::words_mut(&mut c.masks[..(2 * np + 1) * units]);
            let (all_rows, sides) = masks.split_at_mut(words);
            // The sweep's copies of its running left mask, then the left and missing
            // masks.
            let bits = rows::words_mut(&mut c.bits);
            let (lefts, bits) = bits.split_at_mut(SWEEP_TURNS * words);
            let (left, missing) = bits[..2 * words].split_at_mut(words);
            let sides = &mut Rows {
                masks: sides,
                words,
            };
            // Production (not ABLATE) always precomputes by the sweep: the defaults.
            if ABLATE && !options.precompute {
            } else if !ABLATE || options.sweep {
                self.precompute_sweep::<F32, _, _, STATS>(
                    rows,
                    &mut c.order,
                    &mut c.codes,
                    sides,
                    lefts,
                    missing,
                    counters,
                );
            } else {
                self.precompute_brute_force::<F32, _, _, STATS>(rows, sides, left, counters);
            }
            all_rows.fill(u64::MAX);
        } else {
            let masks = &mut view::<M>(&mut c.masks)[..=2 * np];
            let (all_rows, sides) = masks.split_first_mut().expect("the all-rows mask");
            let (sides, _) = sides.as_chunks_mut::<2>();
            let (mut left, mut missing) = (M::ZERO, M::ZERO);
            // Production (not ABLATE) always precomputes by the sweep: the defaults.
            if ABLATE && !options.precompute {
            } else if !ABLATE || options.sweep {
                // The sweep's copies of its running left mask: `left` itself for masks
                // carried by value, else in the scratch.
                let lefts = if <[M] as Lefts<M>>::TURNS == 1 {
                    core::slice::from_mut(&mut left)
                } else {
                    &mut view::<M>(&mut c.bits)[..<[M] as Lefts<M>>::TURNS]
                };
                self.precompute_sweep::<F32, M, _, STATS>(
                    rows,
                    &mut c.order,
                    &mut c.codes,
                    sides,
                    lefts,
                    &mut missing,
                    counters,
                );
            } else {
                self.precompute_brute_force::<F32, M, _, STATS>(rows, sides, &mut left, counters);
            }
            *all_rows = M::ones_if(true);
        }
        let masks = &view::<M>(&mut c.masks)[..(2 * np + 1) * units];
        let mut clock = t.lap(&mut stats.phases.precompute);
        let arena = view::<M>(&mut c.arena);
        let scratch = view::<M>(&mut c.scan).as_mut_ptr();
        let p = self.pools(&c.bins);
        let chunks = &c.chunks[width.index()];
        let mut scanner = Scanner {
            model: self,
            rows,
            monotonic: options.monotonic,
            counters: WorkCounters::default(),
        };
        // Production passes a scan without data, so its passes cannot be merged with
        // research's, whose scan holds the scanner.
        let mut scan = Scans::<F32, STATS>(&mut scanner);
        let walks = Walks::of::<ABLATE>(options, c.tuning.varying);
        for chunk in chunks {
            assert!(
                queues.items.len() > chunk.leaves + MAX_PAD
                    && queues.next.len() > chunk.leaves
                    && queues.leaves.len() > chunk.leaves
                    && c.stops.len() > chunk.leaves + MAX_PAD
                    && c.words.len() > chunk.leaves + MAX_PAD
                    && (!C::ARENA || arena.len() >= (chunk.slots + 3) * C::slot_units(units))
                    && (!C::ARENA || C::RUNTIME || queues.listed.len() > chunk.leaves)
                    && (!C::RUNTIME || c.scan.len() >= units),
                "chunk scratch"
            );
            let roots = &c.roots[chunk.trees.0..chunk.trees.1];
            let (stops, words) = (&mut c.stops[..], &mut c.words[..]);
            let mut pass = Pass {
                p: &p,
                roots,
                n,
                units,
                masks,
                scratch,
                stats: &mut *stats,
            };
            macro_rules! passes {
                ($scan:expr) => {
                    match (options.precompute, options.unsplit) {
                        (true, true) => pass.run_at::<X, C, STATS, ABLATE, false, false>(
                            walks, arena, queues, stops, words, $scan,
                        ),
                        (true, false) => pass.run_at::<X, C, STATS, ABLATE, false, true>(
                            walks, arena, queues, stops, words, $scan,
                        ),
                        (false, true) => pass.run_at::<X, C, STATS, ABLATE, true, false>(
                            walks, arena, queues, stops, words, $scan,
                        ),
                        (false, false) => pass.run_at::<X, C, STATS, ABLATE, true, true>(
                            walks, arena, queues, stops, words, $scan,
                        ),
                    }
                };
            }
            // SAFETY: the assert above sizes the scratch for the chunk, and the
            // layout's children and slots are in range by construction.
            let (n_leaves, n_full) = unsafe {
                // Production (not ABLATE) runs the default options only, precompute
                // and unsplit, so it instantiates only their pass.
                if ABLATE {
                    passes!(&mut scan)
                } else {
                    pass.run_at::<X, C, STATS, ABLATE, false, false>(
                        walks,
                        arena,
                        queues,
                        stops,
                        words,
                        &mut NoScan,
                    )
                }
            };
            clock = Clock::<STATS>::start();
            let (full, entries) = queues.leaves[..n_leaves].split_at(n_full);
            let runs = match (&self.layout.leaves, &mut c.sums) {
                (LeafValues::Fixed(values), Sums::Fixed(diff)) => {
                    let half = diff.len() / 2;
                    let (even, odd) = diff.split_at_mut(half);
                    leaf_pass_at::<X, M, C, _, STATS>(
                        entries,
                        full,
                        arena,
                        units,
                        values,
                        &mut even[..=n],
                        &mut odd[..=n],
                    )
                }
                (
                    LeafValues::Wide(values),
                    Sums::Wide {
                        wide, diff, sums, ..
                    },
                ) => {
                    let limbs = (n + 1) * wide.limbs();
                    let half = diff.len() / 2;
                    let (even, odd) = diff.split_at_mut(half);
                    leaf_pass_at::<X, M, C, _, STATS>(
                        entries,
                        full,
                        arena,
                        units,
                        values,
                        &mut WideRuns::new(*wide, &mut even[..limbs], sums),
                        &mut WideRuns::new(*wide, &mut odd[..limbs], &mut []),
                    )
                }
                _ => unreachable!("leaf values follow the plan"),
            };
            clock = clock.lap(&mut stats.phases.leaf_pass);
            if STATS {
                stats.counters.runs_written += runs;
            }
        }
        if STATS {
            stats.counters.scan_rows += scanner.counters.scan_rows;
            stats.counters.scan_evals += scanner.counters.scan_evals;
        }
        self.finish_at::<X>(results, &mut c.sums);
        clock.lap(&mut stats.phases.finish);
    }

    /// [`Self::constant_group`] at level `X` ([`Isa`]): where the build dispatches, out of line,
    /// with the body compiled inside the level's trampoline; elsewhere inlined.
    #[cfg_attr(not(dispatch), inline(always))]
    #[cfg_attr(dispatch, inline(never))]
    fn constant_group_at<X: Isa, const STATS: bool, const ABLATE: bool>(
        &self,
        results: &mut [f64],
        c: &mut Common,
        options: Options,
        stats: &mut Stats,
    ) {
        if X::NATIVE {
            return self.constant_group::<X, STATS, ABLATE>(results, c, options, stats);
        }
        X::with5(
            #[inline(always)]
            |m: &Self, results, c, options, stats| {
                m.constant_group::<X, STATS, ABLATE>(results, c, options, stats);
            },
            self,
            results,
            c,
            options,
            stats,
        );
    }

    /// Tree sums of a group whose walks stop only at leaves: rows whose bins hold
    /// every feature, as a single row's do, so that varying splits step through their
    /// constant encoding. Every leaf a walk reaches carries all the rows: each chunk's
    /// one pass walks the roots in place, the leaf values are summed in a register from
    /// the walk's words, which hold each stop's leaf-value index, and the sum is every
    /// row's. No mask, item, leaf entry or difference array is touched. `ABLATE` keeps
    /// research's instantiation apart, as in [`Self::prefix_starts`].
    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline(never))]
    fn constant_group<X: Isa, const STATS: bool, const ABLATE: bool>(
        &self,
        results: &mut [f64],
        c: &mut Common,
        options: Options,
        stats: &mut Stats,
    ) {
        let mut clock = Clock::<STATS>::start();
        if STATS {
            stats.counters.single_rows += 1;
        }
        if let Sums::Wide { acc, .. } = &mut c.sums {
            acc.fill(0);
        }
        let fixed = self.constant_chunks::<X, STATS, ABLATE>(c, options, stats, &mut clock);
        let sum = match (&mut c.sums, self.plan) {
            (Sums::Fixed(_), Plan::Fixed(e)) => exact::to_f64(fixed, e),
            (Sums::Wide { wide, acc, .. }, _) => {
                let (first, others) = acc.split_at_mut(wide.limbs());
                for other in others.chunks_exact(wide.limbs()) {
                    wide.add_acc(first, other);
                }
                wide.to_f64(first)
            }
            _ => unreachable!("sums follow the plan"),
        };
        results.fill(sum);
        clock.lap(&mut stats.phases.finish);
    }

    /// The chunks of [`Self::constant_group`]: returns the `i128` sum, or adds a wide
    /// model's to its accumulators.
    #[inline(always)]
    fn constant_chunks<X: Isa, const STATS: bool, const ABLATE: bool>(
        &self,
        c: &mut Common,
        options: Options,
        stats: &mut Stats,
        clock: &mut Clock<STATS>,
    ) -> i128 {
        let p = self.pools(&c.bins);
        let n_trees = c.roots.len() - MAX_PAD;
        let walks = Walks::of::<ABLATE>(options, c.tuning.constant);
        let mut fixed = 0i128;
        for chunk in &c.constant {
            let n = chunk.trees.1 - chunk.trees.0;
            let (walks, pad) = walks.for_pass::<ABLATE>(n, false);
            // The last chunk's roots are followed by the dummies; others are copied.
            let roots: &[u32] = if chunk.trees.1 == n_trees {
                &c.roots[chunk.trees.0..]
            } else {
                let roots = &mut c.constant_roots[..n + pad];
                roots[..n].copy_from_slice(&c.roots[chunk.trees.0..chunk.trees.1]);
                roots[n..].fill(p.dummy);
                roots
            };
            // The roots' dummies are written when the scratch is made, and the prefix
            // starts write trees only.
            assert!(
                roots.len() >= n + pad && c.words.len() > n + pad,
                "chunk scratch"
            );
            debug_assert!(roots[n..n + pad].iter().all(|&r| r == p.dummy));
            let mut steps = Steps::default();
            let (stops, words) = (c.stops.as_mut_ptr(), c.words.as_mut_ptr());
            // SAFETY: the assert above sizes the roots, with their dummies, and the
            // words for the chunk; the walk writes no stops without varying ones; and
            // the layout's children and slots are in range by construction.
            unsafe {
                let roots = roots.as_ptr();
                walk_with!(ABLATE, walks, |K, D, R| walk::walk_at::<
                    X,
                    u32,
                    K,
                    false,
                    D,
                    R,
                    STATS,
                >(
                    &p, roots, stops, words, &mut steps
                ));
            }
            *clock = clock.lap(&mut stats.phases.walk);
            // Every stop is a leaf, whose word 1 starts with its leaf-value index.
            let leaves = c.words[..n].iter().map(|&w1| w1 as u32 as usize);
            match (&self.layout.leaves, &mut c.sums) {
                (LeafValues::Fixed(values), _) => {
                    for leaf in leaves {
                        fixed += values[leaf];
                    }
                }
                (LeafValues::Wide(values), Sums::Wide { wide, acc, .. }) => {
                    // Leaf i adds to sum i % WIDE_SUMS, as in WideRuns: consecutive
                    // leaves mostly share their slots.
                    let l = wide.limbs();
                    assert!(acc.len() >= WIDE_SUMS * l, "the sums of a single row");
                    let leaves = leaves.map(|leaf| values[leaf]);
                    // SAFETY: the assert above holds the WIDE_SUMS accumulators, and the
                    // model's leaf values are leaves of its layout.
                    unsafe { exact::add_in_turns(acc.as_mut_ptr(), l, 0, leaves) };
                }
                _ => unreachable!("leaf values follow the plan"),
            }
            *clock = clock.lap(&mut stats.phases.leaf_pass);
            if STATS {
                let counters = &mut stats.counters;
                counters.passes += 1;
                counters.leaf_events += n as u64;
                counters.runs_written += n as u64;
                counters.walk_steps += steps.nodes;
                counters.lane_slots += steps.slots;
            }
        }
        fixed
    }
}
