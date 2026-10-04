//! Work counters of the counted predict path.

/// Version of the [`WorkCounters`] definitions. It changes whenever a counter's
/// meaning does, so recorded counters can say which definitions they follow.
///
/// Version 1 is treewalker-gbdt 1.x's `PredictStats`. Version 2 adds `leaf_adds`, the
/// `scan_*` counters and the `precompute_*` counters other than
/// `precompute_row_evals`, and makes `precompute_row_evals` count the brute force's
/// predicate × row evaluations, which version 1 reported as the sweep's. Every other
/// counter keeps its version 1 meaning.
#[cfg(feature = "research")]
pub const COUNTERS_VERSION: u32 = 2;

/// Work done by one or more counted predictions, summed over their pieces.
///
/// The definitions are those of [`COUNTERS_VERSION`]. Counters of a disabled path
/// stay 0: with precompute on, the `scan_*` counters and `partition_row_evals`; with
/// it off, the `precompute_*` counters.
///
/// The walk counters (`constant_steps`, `varying_splits`, `leaf_hits`) count the nodes
/// the walk physically visits. With the unsplit shortcut on, every visit has at least
/// one active row. With `Ablation::disable_unsplit`, a
/// split recurses into its light child, which returns at once when no row goes there,
/// and the walk then continues into the heavy child even when no row goes there
/// either: those empty heavy-side visits are counted too. `leaf_adds` counts
/// accumulator writes, so an empty visit adds none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct WorkCounters {
    /// Constant splits the walk visits, including the shared prefix splits.
    pub constant_steps: u64,
    /// Varying splits the walk visits.
    pub varying_splits: u64,
    /// Varying splits where every active row went one way, so the walk continued
    /// without recursion.
    pub unsplit_skips: u64,
    /// Recursions into a split's light child, including those with no active row.
    pub recursive_calls: u64,
    /// Leaves the walk visits.
    pub leaf_hits: u64,
    /// Accumulator writes at leaves: two per run of consecutive rows with exact sums,
    /// one per row without.
    pub leaf_adds: u64,
    /// Active rows at each per-row partition (precompute off), whether or not the
    /// partition loop visits them.
    pub partition_row_evals: u64,
    /// Rows the per-row partition loops visit. A monotonic scan stops early unless
    /// it must still route missing values.
    pub scan_row_evals: u64,
    /// Threshold comparisons and category tests in the per-row partitions.
    pub scan_compares: u64,
    /// Missing-value checks in the per-row partitions. After a monotonic scan stops
    /// comparing, it can still check the remaining rows for missing values.
    pub scan_missing_checks: u64,
    /// Row values the precompute evaluates: per numerical feature, one missing-value
    /// check per row in the threshold sweep; one evaluation per row for each
    /// categorical predicate, and for every predicate in the brute force.
    pub precompute_row_evals: u64,
    /// Comparisons made sorting each numerical feature's rows for the sweep.
    pub precompute_sort_compares: u64,
    /// Threshold comparisons in the sweep's merge of sorted rows and thresholds.
    pub precompute_sweep_compares: u64,
    /// Predicate masks written by the precompute: one per predicate per piece.
    pub precompute_mask_writes: u64,
}

impl std::fmt::Display for WorkCounters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "const={}, varying={}, unsplit={}, recurse={}, leaf={}, leaf_adds={}, \
             row_evals={}, scan_rows={}, scan_compares={}, scan_missing={}, \
             precompute_evals={}, sort_compares={}, sweep_compares={}, mask_writes={}",
            self.constant_steps,
            self.varying_splits,
            self.unsplit_skips,
            self.recursive_calls,
            self.leaf_hits,
            self.leaf_adds,
            self.partition_row_evals,
            self.scan_row_evals,
            self.scan_compares,
            self.scan_missing_checks,
            self.precompute_row_evals,
            self.precompute_sort_compares,
            self.precompute_sweep_compares,
            self.precompute_mask_writes,
        )
    }
}

impl std::ops::AddAssign for WorkCounters {
    fn add_assign(&mut self, rhs: Self) {
        self.constant_steps += rhs.constant_steps;
        self.varying_splits += rhs.varying_splits;
        self.unsplit_skips += rhs.unsplit_skips;
        self.recursive_calls += rhs.recursive_calls;
        self.leaf_hits += rhs.leaf_hits;
        self.leaf_adds += rhs.leaf_adds;
        self.partition_row_evals += rhs.partition_row_evals;
        self.scan_row_evals += rhs.scan_row_evals;
        self.scan_compares += rhs.scan_compares;
        self.scan_missing_checks += rhs.scan_missing_checks;
        self.precompute_row_evals += rhs.precompute_row_evals;
        self.precompute_sort_compares += rhs.precompute_sort_compares;
        self.precompute_sweep_compares += rhs.precompute_sweep_compares;
        self.precompute_mask_writes += rhs.precompute_mask_writes;
    }
}
