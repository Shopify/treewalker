//! Work counters and phase times of the counted predict path.

use std::time::{Duration, Instant};

/// Version of the [`WorkCounters`] definitions. It changes whenever a counter's
/// meaning does, so recorded counters can say which definitions they follow.
///
/// Version 1 is treewalker-gbdt 1.x's `PredictStats` and version 2 the 2.x counters of
/// the recursive kernel. Version 3 counts the 3.x kernel's work: only the four
/// `precompute_*` counters keep their version 2 meaning. Version 4 walks single rows
/// through the same passes as groups, so `pieces`, `passes`, `leaf_events` and
/// `runs_written` count them too. Version 5 runs every group whole, at a runtime
/// mask width past 1,024 rows, so `pieces` counts groups. Version 6 drops `pieces`,
/// which equals `groups`; the other counters keep their version 5 meaning.
#[cfg(feature = "research")]
pub const COUNTERS_VERSION: u32 = 6;

/// Work done by one or more counted predictions, summed over their groups.
///
/// The definitions are those of [`COUNTERS_VERSION`]. Counters of a disabled path
/// stay 0: with precompute on, the `scan_*` counters; with it off, the `precompute_*`
/// counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct WorkCounters {
    /// Groups predicted, single rows included.
    pub groups: u64,
    /// Single-row groups whose walks stepped through varying splits as constants.
    pub single_rows: u64,
    /// Bin fills: one per group, from its first row.
    pub bins_computed: u64,
    /// Row values the precompute evaluates: per numerical feature, one missing-value
    /// check per row in the threshold sweep; one evaluation per row for each
    /// categorical predicate, and for every predicate in the brute force.
    pub precompute_row_evals: u64,
    /// Comparisons made sorting each numerical feature's rows for the sweep.
    pub precompute_sort_compares: u64,
    /// Threshold comparisons in the sweep's merge of sorted rows and thresholds.
    pub precompute_sweep_compares: u64,
    /// Predicate masks computed by the precompute: one per predicate per group,
    /// each written with its complement.
    pub precompute_mask_writes: u64,
    /// Rows the event scans visit, with precompute off: a varying event scans its
    /// rows for the predicate's mask. A monotonic scan stops comparing at the first
    /// row past the threshold.
    pub scan_rows: u64,
    /// Predicate evaluations in the event scans; below `scan_rows` only when a
    /// monotonic scan stops early.
    pub scan_evals: u64,
    /// Shared prefix splits evaluated: per group and prefix group, up to the first
    /// split that leaves the shared path.
    pub prefix_steps: u64,
    /// Nodes the walk lanes step through, dummies excluded.
    pub walk_steps: u64,
    /// Lane steps the walk loops run, dummies included: lanes times loop iterations.
    /// `walk_steps / lane_slots` is the lane utilization.
    pub lane_slots: u64,
    /// Walk passes, per chunk of trees.
    pub passes: u64,
    /// Walk stops at varying splits.
    pub varying_events: u64,
    /// Varying events whose rows go both ways.
    pub splits: u64,
    /// Sides with no rows that varying events pass on as items, which only
    /// `disable_unsplit` does.
    pub empty_sides: u64,
    /// Walk stops at leaves.
    pub leaf_events: u64,
    /// Runs of rows the leaf pass adds a leaf value to, two difference-array writes
    /// each.
    pub runs_written: u64,
    /// Mask-arena slots taken by splits of masks wider than 64 bits.
    pub arena_slots: u64,
}

macro_rules! counters {
    ($($field:ident),*) => {
        impl std::fmt::Display for WorkCounters {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut first = true;
                $(
                    if !first {
                        f.write_str(", ")?;
                    }
                    first = false;
                    write!(f, "{}={}", stringify!($field), self.$field)?;
                )*
                let _ = first;
                Ok(())
            }
        }

        impl std::ops::AddAssign for WorkCounters {
            fn add_assign(&mut self, rhs: Self) {
                $( self.$field += rhs.$field; )*
            }
        }

        impl WorkCounters {
            /// Every counter's name and value, in declaration order.
            #[cfg(feature = "research")]
            #[must_use]
            pub const fn fields(&self) -> [(&'static str, u64); count!($($field)*)] {
                [$( (stringify!($field), self.$field) ),*]
            }
        }
    };
}

#[cfg(feature = "research")]
macro_rules! count {
    () => { 0 };
    ($head:ident $($tail:ident)*) => { 1 + count!($($tail)*) };
}

counters!(
    groups,
    single_rows,
    bins_computed,
    precompute_row_evals,
    precompute_sort_compares,
    precompute_sweep_compares,
    precompute_mask_writes,
    scan_rows,
    scan_evals,
    prefix_steps,
    walk_steps,
    lane_slots,
    passes,
    varying_events,
    splits,
    empty_sides,
    leaf_events,
    runs_written,
    arena_slots
);

/// Time a counted prediction spends in each phase, summed over its groups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Phases {
    /// Filling the bins from each group's first row.
    pub bins: Duration,
    /// The varying predicates' masks.
    pub precompute: Duration,
    /// The prefix groups' shared splits.
    pub prefix: Duration,
    /// Walk passes.
    pub walk: Duration,
    /// Resolve passes.
    pub resolve: Duration,
    /// Leaf passes.
    pub leaf_pass: Duration,
    /// Prefix sums into tree sums, and finalization.
    pub finish: Duration,
}

impl std::ops::AddAssign for Phases {
    fn add_assign(&mut self, rhs: Self) {
        self.bins += rhs.bins;
        self.precompute += rhs.precompute;
        self.prefix += rhs.prefix;
        self.walk += rhs.walk;
        self.resolve += rhs.resolve;
        self.leaf_pass += rhs.leaf_pass;
        self.finish += rhs.finish;
    }
}

/// What the counted path records.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Stats {
    pub(crate) counters: WorkCounters,
    pub(crate) phases: Phases,
}

/// A phase timer that exists only in the counted build.
#[derive(Clone, Copy)]
pub(super) struct Clock<const STATS: bool>(Option<Instant>);

impl<const STATS: bool> Clock<STATS> {
    #[inline(always)]
    pub(super) fn start() -> Self {
        Self(if STATS { Some(Instant::now()) } else { None })
    }

    /// Add the time since the start to `phase`, and restart.
    #[inline(always)]
    pub(super) fn lap(self, phase: &mut Duration) -> Self {
        if let (true, Some(start)) = (STATS, self.0) {
            let now = Instant::now();
            *phase += now - start;
            Self(Some(now))
        } else {
            self
        }
    }
}
