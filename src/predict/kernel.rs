//! A call's groups through the kernel: per group, the bins of its first row and
//! prefix starts, the walk over chunks of trees and the finish. A single row has bins
//! for every feature, so its walk steps through varying splits as constants: it needs
//! no masks and stops only at leaves, which all carry its row. Its one pass per chunk
//! walks the roots in place and sums the leaf values from the walk's words, without
//! items, leaf entries or difference arrays.
//!
//! Scratch lives in [`Scratch`], allocated once per predictor. Trees are processed in
//! chunks whose working set fits the chunk budget of [`crate::rules::chunk_budget`],
//! so scratch scales with a chunk rather than the model.

use super::counters::{Clock, Stats};
use super::passes::{Walks, walk_with};
use super::scratch::{Common, Scratch, Sums};
use super::walk::{self, MAX_PAD, Pools, Steps};
use super::{Groups, WorkCounters};
use crate::exact::{self, Plan, WIDE_SUMS};
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
    /// Tree sums, or with `finalize` outputs, of every group of a call, through the
    /// kernel. With `STATS`, the work and phase times are added to `stats`; with
    /// `ABLATE`, `options` apply, otherwise the defaults do. The input must have
    /// passed [`super::Predictor::check`].
    #[expect(
        clippy::too_many_arguments,
        reason = "the call, its scratch and its variant"
    )]
    pub(super) fn kernel_groups<const F32: bool, const STATS: bool, const ABLATE: bool>(
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
            self.prefix_starts::<STATS, ABLATE>(c, &mut stats.counters);
            t.lap(&mut stats.phases.prefix);
            if STATS {
                stats.counters.groups += 1;
                stats.counters.bins_computed += 1;
            }
            // Groups of more than one row take the masked walk, which lands next.
            assert!(constant, "a single row's constant walk");
            self.constant_group::<STATS, ABLATE>(results, &mut s.common, options, stats);
            if finalize {
                let t = Clock::<STATS>::start();
                self.finalize(&mut out[start..end]);
                t.lap(&mut stats.phases.finish);
            }
        }
    }

    /// Each tree's start: its root, or after its prefix group's shared constant splits
    /// on the bins, the first node the group does not share.
    ///
    /// `ABLATE` only keeps the research builds' instantiation apart from
    /// production's, so that compiling `research` in cannot change production code.
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

    /// Tree sums of a group whose walks stop only at leaves: rows whose bins hold
    /// every feature, as a single row's do, so that varying splits step through their
    /// constant encoding. Every leaf a walk reaches carries all the rows: each chunk's
    /// one pass walks the roots in place, the leaf values are summed in a register from
    /// the walk's words, which hold each stop's leaf-value index, and the sum is every
    /// row's. No mask, item, leaf entry or difference array is touched. `ABLATE` keeps
    /// research's instantiation apart, as in [`Self::prefix_starts`].
    #[inline(never)]
    fn constant_group<const STATS: bool, const ABLATE: bool>(
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
        let fixed = self.constant_chunks::<STATS, ABLATE>(c, options, stats, &mut clock);
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
    fn constant_chunks<const STATS: bool, const ABLATE: bool>(
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
                walk_with!(ABLATE, walks, |K, D, R| walk::walk::<
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
