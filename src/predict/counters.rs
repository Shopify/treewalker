//! Work counters of the counted predict path.

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
