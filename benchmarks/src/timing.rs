//! Blocked paired multi-method timing harness with per-group measurement.
//!
//! All methods run on the same batch of groups in each block, with cyclic
//! rotation of method order. Each group is timed individually with
//! `Instant::now()`, producing a distribution of per-group latencies.

use std::time::{Duration, Instant};

/// Result of a per-group benchmark run.
#[derive(Debug, Clone)]
pub struct TimingResult {
    /// Median per-group latency in microseconds.
    pub median_us: f64,
    /// 5th percentile per-group latency in microseconds.
    pub p5_us: f64,
    /// 95th percentile per-group latency in microseconds.
    pub p95_us: f64,
    /// Number of groups (observations) measured.
    pub n_obs: usize,
    /// Actual number of blocks completed.
    pub actual_blocks: usize,
}

/// Compute a percentile from a sorted slice.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// A method to be benchmarked.
pub struct BenchMethod<'a> {
    pub name: String,
    pub predict_group: Box<dyn FnMut(usize, usize) + 'a>,
}

/// Configuration for the blocked protocol.
pub struct BlockConfig {
    /// Number of batches to cycle through.
    pub n_batches: usize,
    /// Minimum blocks before adaptive stopping is allowed.
    pub min_blocks: usize,
    /// Maximum blocks.
    pub max_blocks: usize,
    /// Number of warmup passes (each pass = all methods on first batch).
    pub warmup: usize,
    /// Overall wall-clock budget.
    pub max_time: Option<Duration>,
    /// Stop early when all methods have CV below this percentage.
    pub precision_target_pct: f64,
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self {
            n_batches: 12,
            min_blocks: 11,
            max_blocks: 21,
            warmup: 3,
            max_time: None,
            precision_target_pct: 3.0,
        }
    }
}

/// Check if all methods have coefficient of variation below `target_pct`.
fn all_methods_stable(per_method_medians: &[Vec<f64>], target_pct: f64) -> bool {
    per_method_medians.iter().all(|medians| {
        if medians.len() < 3 {
            return false;
        }
        let n = medians.len() as f64;
        let mean = medians.iter().sum::<f64>() / n;
        if mean <= 0.0 {
            return true;
        }
        let var = medians.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let cv = var.sqrt() / mean * 100.0;
        cv < target_pct
    })
}

/// Split a pool of group boundaries into approximately equal batches.
fn split_into_batches(pool: &[(usize, usize)], n_batches: usize) -> Vec<Vec<(usize, usize)>> {
    let batch_size = pool.len().div_ceil(n_batches);
    pool.chunks(batch_size)
        .map(<[(usize, usize)]>::to_vec)
        .collect()
}

/// Run the blocked paired multi-method benchmark with per-group timing.
///
/// For each block:
///   1. Pick a batch of groups (round-robin).
///   2. For each method (cyclic rotation):
///      Time each group individually with `Instant::now()`.
///      Record the median per-group time for this block.
///
/// Returns one `TimingResult` per method. The median/p5/p95 are over
/// per-block median values, capturing both group variance and run variance.
pub fn bench_blocked(
    group_boundaries: &[(usize, usize)],
    methods: &mut [BenchMethod<'_>],
    cfg: &BlockConfig,
) -> Vec<TimingResult> {
    let n_methods = methods.len();
    let n_obs = group_boundaries.len();

    if n_methods == 0 || n_obs == 0 {
        return methods.iter().map(|_| TimingResult {
            median_us: 0.0, p5_us: 0.0, p95_us: 0.0, n_obs, actual_blocks: 0,
        }).collect();
    }

    let batches = split_into_batches(group_boundaries, cfg.n_batches);

    // Warmup: run all methods on the first batch (no timing).
    for _ in 0..cfg.warmup {
        for m in methods.iter_mut() {
            for &(s, e) in &batches[0] {
                (m.predict_group)(s, e);
            }
        }
    }

    eprintln!(
        "  Block protocol: {} methods, {} batches, {} pool groups, {}-{} blocks",
        n_methods, batches.len(), n_obs, cfg.min_blocks, cfg.max_blocks,
    );

    // Per-method, per-block median: one median per block.
    let mut per_method_block_medians: Vec<Vec<f64>> =
        vec![Vec::with_capacity(cfg.max_blocks); n_methods];
    // Scratch buffer for per-group times within one block.
    let mut group_times: Vec<f64> = Vec::new();

    let wall = Instant::now();

    for block_id in 0..cfg.max_blocks {
        let batch = &batches[block_id % batches.len()];

        // Cyclic rotation of method order to cancel positional bias.
        let method_order: Vec<usize> = (0..n_methods)
            .map(|i| (i + block_id) % n_methods)
            .collect();

        for &mi in &method_order {
            let m = &mut methods[mi];

            // Time each group individually.
            group_times.clear();
            for &(s, e) in batch {
                let t0 = Instant::now();
                (m.predict_group)(s, e);
                group_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
            }

            // Block-level summary: median of per-group times.
            group_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let block_median = percentile(&group_times, 50.0);
            per_method_block_medians[mi].push(block_median);
        }

        // Adaptive stopping after min_blocks.
        if block_id + 1 >= cfg.min_blocks
            && let Some(budget) = cfg.max_time
            && wall.elapsed() >= budget
        {
            eprintln!(
                "  Block {}/{}: time budget reached ({:.1}s)",
                block_id + 1, cfg.max_blocks, wall.elapsed().as_secs_f64(),
            );
            break;
        }
        if block_id + 1 >= cfg.min_blocks
            && all_methods_stable(&per_method_block_medians, cfg.precision_target_pct)
        {
            eprintln!(
                "  Block {}/{}: all methods stable (CV < {:.0}%)",
                block_id + 1, cfg.max_blocks, cfg.precision_target_pct,
            );
            break;
        }
    }

    // Summarize: for each method, compute median/p5/p95 over block medians.
    methods.iter().enumerate().map(|(i, _)| {
        let medians = &mut per_method_block_medians[i];
        let actual_blocks = medians.len();
        medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
        TimingResult {
            median_us: percentile(medians, 50.0),
            p5_us: percentile(medians, 5.0),
            p95_us: percentile(medians, 95.0),
            n_obs,
            actual_blocks,
        }
    }).collect()
}
