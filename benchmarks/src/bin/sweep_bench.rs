//! Sweep benchmark binary: load a model + raw test data, run all ablation modes,
//! output JSON with timing and stats.
//!
//! Usage:
//!     sweep_bench <model_dir> [--data-dir DIR] [--warmup N] [--iters N] [--mode NAME]
//!                 [--min-iters N] [--max-time-secs S]
//!                 [--group-offsets FILE] [--skip-full] [--skip-stats]
//!                 [--disable-precompute] [--disable-unsplit] [--disable-monotonic]
//!                 [--disable-tree-ordering] [--disable-prefix-grouping] [--disable-bitset-intern]
//!                 [--emit-extended]
//!
//! Expects:
//!     <model_dir>/model_treelite.json    — treelite JSON model
//!     <data_dir>/walker_config.json      — feature classification
//!     <data_dir>/test_data.bin           — raw f64 LE: u64 n_rows, u64 n_cols, then n_rows*n_cols f64s
//!
//! If --data-dir is not set, data files are loaded from <model_dir>.
//! If --mode is set, only run that single ablation mode (for perf stat isolation).
//! If --group-offsets is set, iterate by variable-length groups instead of fixed
//! group_width stride. Format: u64 n_groups, then (n_groups + 1) u64 LE offsets.
//!
//! If any --disable-* flag is given (without --mode), constructs a single ablation
//! configuration from the flags. Otherwise runs all 9 legacy modes.
//!
//! If --emit-extended is set, includes rss_kb, parse_time_us, model_bytes, ns_per_row
//! in the JSON output.
//!
//! Outputs JSON to stdout with timing (median, p5, p95 µs) and PredictStats
//! for each (ablation_mode, group_width) combination.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use treewalker_gbdt::ParseConfig;
use treewalker_gbdt::config::AblationMode;
use treewalker_gbdt::forest::Forest;
use treewalker_gbdt::predict::PredictStats;
use treewalker_bench::get_rss_kb;

/// Load variable-length group offsets from a binary file.
///
/// Format: `u64 n_groups`, then `(n_groups + 1)` u64 LE values representing
/// cumulative row offsets: `[0, end_of_group_0, end_of_group_1, ...]`.
fn load_group_offsets(path: &std::path::Path) -> Vec<usize> {
    let bytes = std::fs::read(path).expect("failed to read group_offsets.bin");
    assert!(bytes.len() >= 8, "group_offsets.bin too short for header");

    let n_groups = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let expected = 8 + (n_groups + 1) * 8;
    assert_eq!(
        bytes.len(),
        expected,
        "group_offsets.bin size mismatch: got {}, expected {} ({n_groups} groups)",
        bytes.len(),
        expected,
    );

    let offsets: Vec<usize> = bytes[8..]
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) as usize)
        .collect();

    assert_eq!(offsets[0], 0, "group_offsets first offset must be 0");
    for (i, pair) in offsets.windows(2).enumerate() {
        let width = pair[1] - pair[0];
        assert!(
            pair[0] <= pair[1],
            "group_offsets must be monotonically increasing (group {i}: {} > {})",
            pair[0], pair[1],
        );
        assert!(
            (1..=128).contains(&width),
            "group {i} has {width} rows (offsets {}..{}), but predict supports 1..=128 rows per entity",
            pair[0], pair[1],
        );
    }

    offsets
}

/// Group boundaries: either fixed-stride or variable from offsets file.
enum Groups {
    /// Fixed group width: groups are [0..w, w..2*w, ...].
    Fixed { n_obs: usize, group_width: usize },
    /// Variable-width: cumulative offsets [0, end0, end1, ...].
    Variable { offsets: Vec<usize> },
}

impl Groups {
    const fn n_obs(&self) -> usize {
        match self {
            Self::Fixed { n_obs, .. } => *n_obs,
            Self::Variable { offsets } => offsets.len() - 1,
        }
    }

    fn start_end(&self, idx: usize) -> (usize, usize) {
        match self {
            Self::Fixed { group_width, .. } => {
                let s = idx * group_width;
                (s, s + group_width)
            }
            Self::Variable { offsets } => (offsets[idx], offsets[idx + 1]),
        }
    }

    /// Total rows across all groups.
    fn total_rows(&self) -> usize {
        match self {
            Self::Fixed { n_obs, group_width } => n_obs * group_width,
            Self::Variable { offsets } => *offsets.last().unwrap_or(&0),
        }
    }

    /// Group width for JSON output. Fixed returns the stride; variable returns None.
    const fn group_width(&self) -> Option<usize> {
        match self {
            Self::Fixed { group_width, .. } => Some(*group_width),
            Self::Variable { .. } => None,
        }
    }
}

/// Timing summary: median, p5, p95 in microseconds.
struct TimingSummary {
    median: f64,
    p5: f64,
    p95: f64,
    actual_iters: usize,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Generic timing harness: run `predict_fn` up to `max_iters` times, stopping early
/// if `max_time` budget is exceeded (after at least `min_iters` samples).
fn bench<F: FnMut()>(
    mut predict_fn: F,
    n_obs: usize,
    max_iters: usize,
    min_iters: usize,
    max_time: Option<Duration>,
) -> TimingSummary {
    let mut timings = Vec::with_capacity(max_iters);
    let wall = Instant::now();
    for i in 0..max_iters {
        let start = Instant::now();
        predict_fn();
        timings.push(start.elapsed().as_nanos() as f64 / 1000.0 / n_obs as f64);
        if let Some(budget) = max_time && i + 1 >= min_iters && wall.elapsed() >= budget {
            break;
        }
    }
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = timings.len();
    TimingSummary {
        median: timings[n / 2],
        p5: percentile(&timings, 5.0),
        p95: percentile(&timings, 95.0),
        actual_iters: n,
    }
}

/// Collect stats across all observations.
fn collect_stats(forest: &mut Forest, data: &[f64], groups: &Groups) -> PredictStats {
    let n_obs = groups.n_obs();
    let mut results = vec![0.0f64; data.len() / forest.config.n_features];
    let mut total = PredictStats::default();

    for s in 0..n_obs {
        let (row_start, row_end) = groups.start_end(s);
        let stats = forest.predict_with_stats(data, &mut results, row_start, row_end);
        total += stats;
    }

    total
}

/// Compute model memory footprint: nodes*16 + bitsets + trees*12.
fn model_bytes(forest: &Forest) -> usize {
    forest.nodes().len() * 16 + forest.bitset_bytes() + forest.trees().len() * 12
}

/// JSON result object for one ablation mode.
#[derive(serde::Serialize)]
struct BenchResult {
    mode: String,
    group_width: Option<usize>,
    latency_partial_us: f64,
    latency_partial_p5_us: f64,
    latency_partial_p95_us: f64,
    latency_full_us: f64,
    latency_full_p5_us: f64,
    latency_full_p95_us: f64,
    actual_iters_partial: usize,
    actual_iters_full: usize,
    speedup_vs_full: f64,
    constant_steps: u64,
    varying_splits: u64,
    unsplit_skips: u64,
    recursive_calls: u64,
    leaf_hits: u64,
    partition_row_evals: u64,
    precompute_row_evals: u64,
    n_obs: usize,
    n_trees: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    ns_per_row: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rss_kb: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parse_time_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_bytes: Option<usize>,
}

/// Shared timing/control parameters for benchmark runs.
struct BenchControl {
    warmup: usize,
    iters: usize,
    min_iters: usize,
    max_time: Option<Duration>,
    emit_extended: bool,
    skip_full: bool,
    skip_stats: bool,
    parse_time_us: f64,
}

/// Run a full-walk (no partial evaluation) baseline benchmark.
fn bench_full_baseline(
    forest: &Forest,
    data: &[f64],
    groups: &Groups,
    ctl: &BenchControl,
) -> TimingSummary {
    let n_obs = groups.n_obs();
    let total_rows = groups.total_rows();
    let mut results = vec![0.0f64; total_rows];
    let warmup_wall = Instant::now();
    for w in 0..ctl.warmup {
        for s in 0..n_obs {
            let (start, end) = groups.start_end(s);
            forest.predict_full(data, &mut results, start, end);
        }
        // Respect max_time budget during warmup.
        if let Some(budget) = ctl.max_time && warmup_wall.elapsed() >= budget {
            eprintln!("  (full warmup cut short after {}/{} passes)", w + 1, ctl.warmup);
            break;
        }
    }
    bench(|| {
        for s in 0..n_obs {
            let (start, end) = groups.start_end(s);
            forest.predict_full(data, &mut results, start, end);
        }
    }, n_obs, ctl.iters, ctl.min_iters, ctl.max_time)
}

/// Run one ablation mode: warmup, partial bench, stats, and produce a `BenchResult`.
fn run_one_mode(
    mode_name: &str,
    forest: &mut Forest,
    full: &TimingSummary,
    data: &[f64],
    groups: &Groups,
    ctl: &BenchControl,
) -> BenchResult {
    let n_obs = groups.n_obs();
    let total_rows = groups.total_rows();
    let group_width = groups.group_width();

    // Warmup (time-bounded)
    let mut results = vec![0.0f64; total_rows];
    let warmup_wall = Instant::now();
    for w in 0..ctl.warmup {
        for s in 0..n_obs {
            let (start, end) = groups.start_end(s);
            forest.predict(data, &mut results, start, end);
        }
        if let Some(budget) = ctl.max_time && warmup_wall.elapsed() >= budget {
            eprintln!("  (partial warmup cut short after {}/{} passes)", w + 1, ctl.warmup);
            break;
        }
    }

    // Partial bench
    let partial = bench(|| {
        for s in 0..n_obs {
            let (start, end) = groups.start_end(s);
            forest.predict(data, &mut results, start, end);
        }
    }, n_obs, ctl.iters, ctl.min_iters, ctl.max_time);

    let rss_kb = get_rss_kb();
    let stats = if ctl.skip_stats {
        PredictStats::default()
    } else {
        collect_stats(forest, data, groups)
    };

    eprintln!(
        "  {mode_name:<25} partial={:.1}\u{00b5}s  full={:.1}\u{00b5}s  speedup={:.1}x  iters={}",
        partial.median, full.median, full.median / partial.median, partial.actual_iters,
    );

    let (ns_per_row, ext_rss, ext_parse, ext_model) = if ctl.emit_extended {
        (
            Some(partial.median * 1000.0 / total_rows as f64 * n_obs as f64),
            Some(rss_kb),
            Some(ctl.parse_time_us),
            Some(model_bytes(forest)),
        )
    } else {
        (None, None, None, None)
    };

    BenchResult {
        mode: mode_name.to_string(),
        group_width,
        latency_partial_us: partial.median,
        latency_partial_p5_us: partial.p5,
        latency_partial_p95_us: partial.p95,
        latency_full_us: full.median,
        latency_full_p5_us: full.p5,
        latency_full_p95_us: full.p95,
        actual_iters_partial: partial.actual_iters,
        actual_iters_full: full.actual_iters,
        speedup_vs_full: full.median / partial.median,
        constant_steps: stats.constant_steps,
        varying_splits: stats.varying_splits,
        unsplit_skips: stats.unsplit_skips,
        recursive_calls: stats.recursive_calls,
        leaf_hits: stats.leaf_hits,
        partition_row_evals: stats.partition_row_evals,
        precompute_row_evals: stats.precompute_row_evals,
        n_obs,
        n_trees: forest.trees().len(),
        ns_per_row,
        rss_kb: ext_rss,
        parse_time_us: ext_parse,
        model_bytes: ext_model,
    }
}

/// All known ablation mode names.
const ALL_MODES: &[&str] = &[
    "baseline", "no_monotonic", "no_unsplit",
    "no_varying_precompute", "all_disabled",
    "no_tree_ordering", "no_bitset_intern", "no_prefix_grouping",
];

/// Build a mode name from sorted active disable flags.
fn compose_mode_name(flags: &TuningFlags) -> String {
    let d = TuningFlags::default();
    let mut parts = Vec::new();
    if flags.precompute { parts.push("no_precompute".to_string()); }
    if flags.unsplit { parts.push("no_unsplit".to_string()); }
    if flags.monotonic { parts.push("no_monotonic".to_string()); }
    if flags.predicate_sweep { parts.push("no_sweep".to_string()); }
    if flags.tree_ordering { parts.push("no_tree_ordering".to_string()); }
    if flags.prefix_depth != d.prefix_depth {
        parts.push(format!("prefix_depth_{}", flags.prefix_depth));
    }
    if flags.bitset_intern { parts.push("no_bitset_intern".to_string()); }
    if flags.predicate_dedup { parts.push("no_dedup".to_string()); }
    if parts.is_empty() {
        "baseline".into()
    } else {
        parts.join("+")
    }
}

/// Parsed tuning flags from CLI.
struct TuningFlags {
    precompute: bool,
    unsplit: bool,
    monotonic: bool,
    predicate_sweep: bool,
    tree_ordering: bool,
    prefix_depth: usize,
    bitset_intern: bool,
    predicate_dedup: bool,
}

impl Default for TuningFlags {
    fn default() -> Self {
        Self {
            precompute: false,
            unsplit: false,
            monotonic: false,
            predicate_sweep: false,
            tree_ordering: false,
            prefix_depth: 2,
            bitset_intern: false,
            predicate_dedup: false,
        }
    }
}

impl TuningFlags {
    fn any_set(&self) -> bool {
        let d = Self::default();
        self.precompute || self.unsplit || self.monotonic || self.predicate_sweep
            || self.tree_ordering || self.bitset_intern || self.predicate_dedup
            || self.prefix_depth != d.prefix_depth
    }

    const fn to_ablation_mode(&self) -> AblationMode {
        AblationMode {
            disable_varying_precompute: self.precompute,
            disable_unsplit: self.unsplit,
            disable_monotonic: self.monotonic,
            disable_predicate_sweep: self.predicate_sweep,
        }
    }

    const fn to_parse_config(&self) -> ParseConfig {
        ParseConfig {
            disable_tree_ordering: self.tree_ordering,
            disable_bitset_intern: self.bitset_intern,
            prefix_depth: self.prefix_depth,
            disable_predicate_dedup: self.predicate_dedup,
            hoist_constants: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Block-mode benchmark protocol
// ---------------------------------------------------------------------------

/// Pool manifest: binary file with u64 count, then count × u64 group indices.
fn load_pool_manifest(path: &std::path::Path) -> Vec<usize> {
    let bytes = std::fs::read(path).expect("failed to read pool manifest");
    assert!(bytes.len() >= 8, "pool manifest too short");
    let count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    assert_eq!(bytes.len(), 8 + count * 8, "pool manifest size mismatch");
    bytes[8..]
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect()
}

/// Split pool indices into n_batches deterministic mini-batches.
fn split_into_batches(pool: &[usize], n_batches: usize) -> Vec<Vec<usize>> {
    let mut batches: Vec<Vec<usize>> = (0..n_batches).map(|_| Vec::new()).collect();
    for (i, &idx) in pool.iter().enumerate() {
        batches[i % n_batches].push(idx);
    }
    batches
}

/// A named mode with its forest ready to predict.
struct BlockMode {
    name: String,
    forest: Forest,
}

/// Per-block per-mode timing record, emitted as one JSONL line.
#[derive(serde::Serialize)]
struct BlockRecord {
    block_id: usize,
    batch_id: usize,
    mode: String,
    order_idx: usize,
    inner_repeats: usize,
    us_per_obs: f64,
    n_groups: usize,
    total_rows: usize,
}

/// Calibrate inner_repeats so one timed sample ≈ target_ms.
fn calibrate_repeats(
    forest: &mut Forest,
    data: &[f64],
    results: &mut [f64],
    batch_indices: &[usize],
    groups: &Groups,
    target_ms: f64,
) -> usize {
    // Pilot: time one pass over the batch.
    let start = Instant::now();
    for &g in batch_indices {
        let (s, e) = groups.start_end(g);
        forest.predict(data, results, s, e);
    }
    let pilot_ms = start.elapsed().as_secs_f64() * 1000.0;
    if pilot_ms >= target_ms {
        return 1;
    }
    // Round up to hit target.
    let reps = (target_ms / pilot_ms).ceil() as usize;
    reps.clamp(1, 10_000) // cap at 10k to prevent runaway
}

/// Block-mode configuration.
struct BlockConfig {
    n_batches: usize,
    target_ms: f64,
    min_blocks: usize,
    max_blocks: usize,
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self {
            n_batches: 12,
            target_ms: 75.0,
            min_blocks: 11,
            max_blocks: 21,
        }
    }
}

/// Run the blocked paired measurement protocol.
///
/// For each block:
///   1. Pick a batch (round-robin).
///   2. For each mode (cyclic rotation by block_id):
///      a. On first block, calibrate inner_repeats.
///      b. Time inner_repeats passes over the batch.
///      c. Emit one JSONL line.
fn run_block_mode(
    modes: &mut [BlockMode],
    data: &[f64],
    groups: &Groups,
    pool: &[usize],
    cfg: &BlockConfig,
) {
    let batches = split_into_batches(pool, cfg.n_batches);
    let n_modes = modes.len();
    let total_rows = groups.total_rows();
    let mut results = vec![0.0f64; total_rows];

    // Warmup: one pass per mode on the first batch.
    for m in modes.iter_mut() {
        for &g in &batches[0] {
            let (s, e) = groups.start_end(g);
            m.forest.predict(data, &mut results, s, e);
        }
    }

    // Calibrate inner_repeats per mode using the first batch.
    let mut repeats_per_mode: Vec<usize> = Vec::with_capacity(n_modes);
    for m in modes.iter_mut() {
        let r = calibrate_repeats(
            &mut m.forest, data, &mut results, &batches[0], groups, cfg.target_ms,
        );
        repeats_per_mode.push(r);
    }

    eprintln!(
        "Block mode: {} modes, {} batches, {} pool groups, {}-{} blocks",
        n_modes, cfg.n_batches, pool.len(), cfg.min_blocks, cfg.max_blocks,
    );
    for (i, m) in modes.iter().enumerate() {
        eprintln!("  mode[{i}] = {} (inner_repeats={})", m.name, repeats_per_mode[i]);
    }

    for block_id in 0..cfg.max_blocks {
        let batch_id = block_id % batches.len();
        let batch = &batches[batch_id];
        let batch_rows: usize = batch.iter().map(|&g| {
            let (s, e) = groups.start_end(g);
            e - s
        }).sum();
        let n_groups = batch.len();

        // Cyclic rotation of mode order.
        let mode_order: Vec<usize> = (0..n_modes)
            .map(|i| (i + block_id) % n_modes)
            .collect();

        for (order_idx, &mi) in mode_order.iter().enumerate() {
            let m = &mut modes[mi];
            let inner = repeats_per_mode[mi];

            let is_full = m.name == "full";
            let start = Instant::now();
            for _ in 0..inner {
                for &g in batch {
                    let (s, e) = groups.start_end(g);
                    if is_full {
                        m.forest.predict_full(data, &mut results, s, e);
                    } else {
                        m.forest.predict(data, &mut results, s, e);
                    }
                }
            }
            let elapsed_us = start.elapsed().as_nanos() as f64 / 1000.0;
            let us_per_obs = elapsed_us / (n_groups as f64 * inner as f64);

            let record = BlockRecord {
                block_id,
                batch_id,
                mode: m.name.clone(),
                order_idx,
                inner_repeats: inner,
                us_per_obs,
                n_groups,
                total_rows: batch_rows,
            };
            // JSONL: one line per measurement.
            println!("{}", simd_json::serde::to_string(&record).unwrap());
        }

        if block_id + 1 >= cfg.min_blocks {
            // Adaptive stopping: Python checks the JSONL stream.
            // For now, emit a progress line to stderr.
            eprintln!(
                "  block {}/{}: {} measurements",
                block_id + 1, cfg.max_blocks, (block_id + 1) * n_modes,
            );
        }
    }
}

/// Parse a comma-separated mode spec into (name, AblationMode, ParseConfig) triples.
///
/// Modes prefixed with "p:" use non-default parse configs; others share the
/// default forest.  Recognized names:
///   baseline, no_unsplit, no_precompute, no_monotonic, no_sweep, all_disabled,
///   full (full-walk baseline),
///   p:no_tree_ordering, p:no_bitset_intern, p:no_prefix_grouping
fn parse_mode_specs(spec: &str) -> Vec<(String, AblationMode, ParseConfig)> {
    let mut modes = Vec::new();
    for name in spec.split(',') {
        let name = name.trim();
        if name.is_empty() { continue; }
        let (ablation, parse) = match name {
            "baseline" | "full" => (AblationMode::default(), ParseConfig::default()),
            "no_unsplit" => (
                AblationMode { disable_unsplit: true, ..Default::default() },
                ParseConfig::default(),
            ),
            "no_precompute" => (
                AblationMode { disable_varying_precompute: true, ..Default::default() },
                ParseConfig::default(),
            ),
            "no_monotonic" => (
                AblationMode {
                    disable_monotonic: true,
                    disable_varying_precompute: true,
                    ..Default::default()
                },
                ParseConfig::default(),
            ),
            "no_sweep" => (
                AblationMode { disable_predicate_sweep: true, ..Default::default() },
                ParseConfig::default(),
            ),
            "all_disabled" => (
                AblationMode {
                    disable_monotonic: true,
                    disable_unsplit: true,
                    disable_varying_precompute: true,
                    disable_predicate_sweep: true,
                },
                ParseConfig::default(),
            ),
            "p:no_tree_ordering" => (
                AblationMode::default(),
                ParseConfig { disable_tree_ordering: true, ..Default::default() },
            ),
            "p:no_bitset_intern" => (
                AblationMode::default(),
                ParseConfig { disable_bitset_intern: true, ..Default::default() },
            ),
            "p:no_prefix_grouping" => (
                AblationMode::default(),
                ParseConfig { prefix_depth: 0, ..Default::default() },
            ),
            _ => {
                eprintln!("Unknown block-mode mode: {name}");
                std::process::exit(1);
            }
        };
        modes.push((name.to_string(), ablation, parse));
    }
    modes
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: sweep_bench <model_dir> [--data-dir DIR] [--warmup N] [--iters N] [--mode NAME]\n\
             \x20      [--min-iters N] [--max-time-secs S] [--group-offsets FILE]\n\
             \x20      [--skip-full] [--skip-stats]\n\
             \x20      [--disable-precompute] [--disable-unsplit] [--disable-monotonic]\n\
             \x20      [--disable-tree-ordering] [--disable-prefix-grouping] [--disable-bitset-intern]\n\
             \x20      [--emit-extended]\n\
             \n\
             Block mode (new protocol):\n\
             \x20      [--block-mode] [--pool FILE] [--modes MODE1,MODE2,...]\n\
             \x20      [--n-batches N] [--target-ms MS] [--min-blocks N] [--max-blocks N]"
        );
        eprintln!("Modes: {}", ALL_MODES.join(", "));
        eprintln!("Block modes: baseline,full,no_unsplit,no_precompute,no_monotonic,no_sweep,all_disabled,p:no_tree_ordering,p:no_bitset_intern,p:no_prefix_grouping");
        eprintln!("Tuning: --disable-{{precompute,unsplit,monotonic,predicate-sweep,tree-ordering,bitset-intern,predicate-dedup}} --prefix-depth N");
        eprintln!("\n--min-iters N   Minimum iterations before max-time can exit (default: 11)");
        eprintln!("--skip-full     Skip full-walk baseline (report speedup as NaN)");
        eprintln!("--skip-stats    Skip stats collection pass (zero out stats fields)");
        std::process::exit(1);
    }

    // -----------------------------------------------------------------------
    // Grid mode: scan artifacts dir, run all methods, write CSV directly.
    // Triggered by --grid. Remaining args are grid-specific.
    // -----------------------------------------------------------------------
    if args.iter().any(|a| a == "--grid") {
        run_grid_mode(&args);
        return;
    }

    // -----------------------------------------------------------------------
    // Validate mode: run TreeWalker on a cell and compare to an f64 reference.
    // Triggered by --validate FILE. Used by prepare_scenario.py as the
    // per-cell correctness gate (TOL_F64 = 1e-14 for LightGBM).
    // -----------------------------------------------------------------------
    if args.iter().any(|a| a == "--validate") {
        run_validate_mode(&args);
        return;
    }

    // -----------------------------------------------------------------------
    // Single-cell mode (legacy): benchmark one model_dir.
    // -----------------------------------------------------------------------
    let model_dir = PathBuf::from(&args[1]);
    let mut data_dir: Option<PathBuf> = None;
    let mut warmup = 3;
    let mut iters = 11;
    let mut min_iters = 11;
    let mut mode_filter: Option<String> = None;
    let mut group_offsets_path: Option<PathBuf> = None;
    let mut max_time: Option<Duration> = None;
    let mut tuning = TuningFlags::default();
    let mut emit_extended = false;
    let mut skip_full = false;
    let mut skip_stats = false;
    // Block-mode args
    let mut block_mode = false;
    let mut pool_path: Option<PathBuf> = None;
    let mut block_modes_spec: Option<String> = None;
    let mut block_cfg = BlockConfig::default();
    #[cfg(feature = "quickscorer-bench")]
    let mut quickscorer_model: Option<PathBuf> = None;

    let mut i = 2;
    while i < args.len() {
        let flag = args[i].as_str();
        // Helper for flags that require a value.
        let next = |i: usize, flag: &str| -> &String {
            args.get(i + 1).unwrap_or_else(|| {
                eprintln!("Missing value for {flag}");
                std::process::exit(1);
            })
        };
        match flag {
            "--data-dir" => {
                data_dir = Some(PathBuf::from(next(i, flag)));
                i += 2;
            }
            "--warmup" => {
                let val = next(i, flag);
                warmup = val.parse().unwrap_or_else(|e| {
                    eprintln!("{flag}: invalid value '{val}': {e}");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--iters" => {
                let val = next(i, flag);
                iters = val.parse().unwrap_or_else(|e| {
                    eprintln!("{flag}: invalid value '{val}': {e}");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--min-iters" => {
                let val = next(i, flag);
                min_iters = val.parse().unwrap_or_else(|e| {
                    eprintln!("{flag}: invalid value '{val}': {e}");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--mode" => {
                let m = next(i, flag).clone();
                if !ALL_MODES.contains(&m.as_str()) {
                    eprintln!("Unknown mode: {m}");
                    eprintln!("Valid modes: {}", ALL_MODES.join(", "));
                    std::process::exit(1);
                }
                mode_filter = Some(m);
                i += 2;
            }
            "--group-offsets" => {
                group_offsets_path = Some(PathBuf::from(next(i, flag)));
                i += 2;
            }
            "--max-time-secs" => {
                let val = next(i, flag);
                let secs: f64 = val.parse().unwrap_or_else(|e| {
                    eprintln!("{flag}: invalid value '{val}': {e}");
                    std::process::exit(1);
                });
                max_time = Some(Duration::from_secs_f64(secs));
                i += 2;
            }
            // Block-mode flags.
            "--block-mode" => { block_mode = true; i += 1; }
            "--pool" => {
                pool_path = Some(PathBuf::from(next(i, flag)));
                i += 2;
            }
            "--modes" => {
                block_modes_spec = Some(next(i, flag).clone());
                i += 2;
            }
            "--n-batches" => {
                let val = next(i, flag);
                block_cfg.n_batches = val.parse().unwrap_or_else(|e| {
                    eprintln!("{flag}: invalid value '{val}': {e}");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--target-ms" => {
                let val = next(i, flag);
                block_cfg.target_ms = val.parse().unwrap_or_else(|e| {
                    eprintln!("{flag}: invalid value '{val}': {e}");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--min-blocks" => {
                let val = next(i, flag);
                block_cfg.min_blocks = val.parse().unwrap_or_else(|e| {
                    eprintln!("{flag}: invalid value '{val}': {e}");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--max-blocks" => {
                let val = next(i, flag);
                block_cfg.max_blocks = val.parse().unwrap_or_else(|e| {
                    eprintln!("{flag}: invalid value '{val}': {e}");
                    std::process::exit(1);
                });
                i += 2;
            }
            // Boolean disable flags — no value needed.
            "--disable-precompute" => { tuning.precompute = true; i += 1; }
            "--disable-unsplit" => { tuning.unsplit = true; i += 1; }
            "--disable-monotonic" => { tuning.monotonic = true; i += 1; }
            "--disable-predicate-sweep" => { tuning.predicate_sweep = true; i += 1; }
            "--disable-tree-ordering" => { tuning.tree_ordering = true; i += 1; }
            "--prefix-depth" => {
                let val = next(i, flag);
                tuning.prefix_depth = val.parse().unwrap_or_else(|e| {
                    eprintln!("{flag}: invalid value '{val}': {e}");
                    std::process::exit(1);
                });
                i += 2;
            }
            "--disable-bitset-intern" => { tuning.bitset_intern = true; i += 1; }
            "--disable-predicate-dedup" => { tuning.predicate_dedup = true; i += 1; }
            "--emit-extended" => { emit_extended = true; i += 1; }
            "--skip-full" => { skip_full = true; i += 1; }
            "--skip-stats" => { skip_stats = true; i += 1; }
            #[cfg(feature = "quickscorer-bench")]
            "--quickscorer" => {
                quickscorer_model = Some(PathBuf::from(next(i, flag)));
                i += 2;
            }
            _ => {
                eprintln!("Unknown argument: {flag}");
                std::process::exit(1);
            }
        }
    }

    let data_dir = data_dir.as_ref().unwrap_or(&model_dir);
    let bin_path = model_dir.join("model_treelite.bin");
    let json_path = model_dir.join("model_treelite.json");
    let model_path = if bin_path.exists() { bin_path } else { json_path };
    let config_path = data_dir.join("walker_config.json");
    let data_path = data_dir.join("test_data.bin");

    // -----------------------------------------------------------------------
    // Block mode: new paired measurement protocol.
    // -----------------------------------------------------------------------
    if block_mode {
        let modes_spec = block_modes_spec.unwrap_or_else(|| "baseline,full".to_string());
        let mode_specs = parse_mode_specs(&modes_spec);

        let (data, n_rows, _n_cols) = treewalker_bench::load_raw_f64(&data_path);

        // Build groups (needed for index → row mapping).
        let temp_forest = Forest::load(&model_path, &config_path);
        let groups = build_groups(&temp_forest, n_rows, group_offsets_path.as_ref());
        let n_total_groups = groups.n_obs();
        drop(temp_forest);

        // Build pool: from manifest or all groups.
        let pool: Vec<usize> = pool_path.as_ref().map_or_else(
            || (0..n_total_groups).collect(),
            |pp| {
                let p = load_pool_manifest(pp);
                for &idx in &p {
                    assert!(
                        idx < n_total_groups,
                        "pool index {idx} >= n_total_groups {n_total_groups}",
                    );
                }
                p
            },
        );

        // Build forests: group by ParseConfig to avoid redundant parses.
        let mut modes: Vec<BlockMode> = Vec::with_capacity(mode_specs.len());
        for (name, ablation, parse_config) in &mode_specs {
            if name == "full" {
                // Full-walk mode uses predict_full; handled specially.
                // We still need a forest for it.
                let f = Forest::load_with_config(&model_path, &config_path, parse_config);
                modes.push(BlockMode { name: name.clone(), forest: f });
            } else {
                let mut f = Forest::load_with_config(&model_path, &config_path, parse_config);
                f.config.ablation = *ablation;
                modes.push(BlockMode { name: name.clone(), forest: f });
            }
        }

        eprintln!(
            "Block mode: {} groups in pool (of {} total), {} modes",
            pool.len(), n_total_groups, modes.len(),
        );

        run_block_mode(&mut modes, &data, &groups, &pool, &block_cfg);
        return;
    }

    // -----------------------------------------------------------------------
    // Legacy dispatch: --mode, --disable-* flags, or all modes.
    // -----------------------------------------------------------------------
    if mode_filter.is_some() && tuning.any_set() {
        eprintln!("Cannot combine --mode with --disable-* flags");
        std::process::exit(1);
    }

    let ctl = BenchControl { warmup, iters, min_iters, max_time, emit_extended, skip_full, skip_stats, parse_time_us: 0.0 };

    if tuning.any_set() {
        // Composable single-config mode from --disable-* flags.
        let mode_name = compose_mode_name(&tuning);
        let ablation = tuning.to_ablation_mode();
        let parse_config = tuning.to_parse_config();

        let parse_start = Instant::now();
        let mut forest = Forest::load_with_config(&model_path, &config_path, &parse_config);
        let parse_time_us = parse_start.elapsed().as_nanos() as f64 / 1000.0;
        forest.config.ablation = ablation;

        let (data, n_rows, _n_cols) = treewalker_bench::load_raw_f64(&data_path);
        let groups = build_groups(&forest, n_rows, group_offsets_path.as_ref());

        eprintln!(
            "Loaded: {} trees, {} features, group_width={}, {} observations{}",
            forest.trees().len(), forest.config.n_features, forest.config.max_group_width,
            groups.n_obs(), if group_offsets_path.is_some() { " (variable groups)" } else { "" },
        );
        eprintln!("  mode: {mode_name} (composed from --disable-* flags)");

        let ctl = BenchControl { parse_time_us, ..ctl };
        let full = if ctl.skip_full {
            TimingSummary { median: f64::NAN, p5: f64::NAN, p95: f64::NAN, actual_iters: 0 }
        } else {
            bench_full_baseline(&forest, &data, &groups, &ctl)
        };

        let result = run_one_mode(&mode_name, &mut forest, &full, &data, &groups, &ctl);
        let output = simd_json::serde::to_string_pretty(&[result]).unwrap();
        println!("{output}");
    } else {
        // Legacy mode: --mode filter or all 9 modes.
        let parse_start = Instant::now();
        let forest = Forest::load(&model_path, &config_path);
        let default_parse_time_us = parse_start.elapsed().as_nanos() as f64 / 1000.0;

        let (data, n_rows, _n_cols) = treewalker_bench::load_raw_f64(&data_path);
        let groups = build_groups(&forest, n_rows, group_offsets_path.as_ref());

        eprintln!(
            "Loaded: {} trees, {} features, group_width={}, {} observations{}",
            forest.trees().len(), forest.config.n_features, forest.config.max_group_width,
            groups.n_obs(), if group_offsets_path.is_some() { " (variable groups)" } else { "" },
        );
        drop(forest);

        let should_run = |name: &str| -> bool {
            mode_filter.as_ref().is_none_or(|f| f == name)
        };

        let runtime_modes: &[(&str, AblationMode)] = &[
            ("baseline", AblationMode::default()),
            ("no_unsplit", AblationMode { disable_unsplit: true, ..Default::default() }),
            ("no_varying_precompute", AblationMode { disable_varying_precompute: true, ..Default::default() }),
            // Monotonic only has effect when precompute is also disabled (precompute
            // evaluates all predicates identically regardless of monotonicity).
            ("no_monotonic", AblationMode { disable_monotonic: true, disable_varying_precompute: true, ..Default::default() }),
            ("all_disabled", AblationMode { disable_monotonic: true, disable_unsplit: true, disable_varying_precompute: true, disable_predicate_sweep: true }),
        ];

        let parse_modes: &[(&str, ParseConfig)] = &[
            ("no_tree_ordering", ParseConfig { disable_tree_ordering: true, ..Default::default() }),
            ("no_bitset_intern", ParseConfig { disable_bitset_intern: true, ..Default::default() }),
            ("no_prefix_grouping", ParseConfig { prefix_depth: 0, ..Default::default() }),
        ];

        // Full baseline
        let ctl = BenchControl { parse_time_us: default_parse_time_us, ..ctl };
        let full = if ctl.skip_full {
            TimingSummary { median: f64::NAN, p5: f64::NAN, p95: f64::NAN, actual_iters: 0 }
        } else {
            let full_forest = Forest::load(&model_path, &config_path);
            bench_full_baseline(&full_forest, &data, &groups, &ctl)
        };

        let mut json_results = Vec::new();

        for &(name, ablation) in runtime_modes {
            if !should_run(name) { continue; }
            let mut f = Forest::load(&model_path, &config_path);
            f.config.ablation = ablation;
            json_results.push(run_one_mode(name, &mut f, &full, &data, &groups, &ctl));
        }

        for &(name, ref pc) in parse_modes {
            if !should_run(name) { continue; }
            let pt_start = Instant::now();
            let mut f = Forest::load_with_config(&model_path, &config_path, pc);
            let pt_us = pt_start.elapsed().as_nanos() as f64 / 1000.0;
            let mode_ctl = BenchControl { parse_time_us: pt_us, ..ctl };
            let mode_full = if ctl.skip_full {
                TimingSummary { median: f64::NAN, p5: f64::NAN, p95: f64::NAN, actual_iters: 0 }
            } else {
                bench_full_baseline(&f, &data, &groups, &mode_ctl)
            };
            json_results.push(run_one_mode(name, &mut f, &mode_full, &data, &groups, &mode_ctl));
        }

        let output = simd_json::serde::to_string_pretty(&json_results).unwrap();
        println!("{output}");
    }

    // -----------------------------------------------------------------------
    #[cfg(feature = "quickscorer-bench")]
    if let Some(ref qs_model_path) = quickscorer_model {
        run_quickscorer_bench(
            qs_model_path, &model_path, &config_path, data_dir,
            group_offsets_path.as_ref(), warmup, iters, min_iters, max_time,
        );
    }
}

/// Run the QuickScorer baseline benchmark.
///
/// Iterates over groups (observations), not flat rows.  For each group the
/// inner loop calls `score_fast` once per row, so QuickScorer pays the real
/// per-group cost including loop and function-call overhead.
#[cfg(feature = "quickscorer-bench")]
fn run_quickscorer_bench(
    qs_model_path: &std::path::Path,
    model_path: &std::path::Path,
    config_path: &std::path::Path,
    data_dir: &std::path::Path,
    group_offsets_path: Option<&PathBuf>,
    warmup: usize,
    iters: usize,
    min_iters: usize,
    max_time: Option<Duration>,
) {
    let data_path = data_dir.join("test_data.bin");
    let (data, n_rows, n_cols) = treewalker_bench::load_raw_f64(&data_path);

    // Build groups so we iterate per-observation, not per-row.
    let forest = Forest::load(model_path, config_path);
    let groups = build_groups(&forest, n_rows, group_offsets_path);
    let n_obs = groups.n_obs();
    drop(forest);

    eprintln!("\n--- QuickScorer baseline (per-group, {n_obs} obs) ---");
    let qs_start = Instant::now();
    let qs_result = std::panic::catch_unwind(|| {
        quickscorer::QuickScorer::from_model_file(qs_model_path)
    });
    let mut qs = match qs_result {
        Ok(Ok(qs)) => qs,
        Ok(Err(e)) => {
            eprintln!("  QuickScorer load error: {e}");
            eprintln!("  (skipping QuickScorer benchmark)");
            return;
        }
        Err(_) => {
            eprintln!("  QuickScorer panicked during load (likely >128 leaves per tree)");
            eprintln!("  (skipping QuickScorer benchmark)");
            return;
        }
    };
    let qs_parse_us = qs_start.elapsed().as_nanos() as f64 / 1000.0;
    eprintln!("  QuickScorer loaded in {qs_parse_us:.0}µs");

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

    // Warmup (time-bounded): iterate over groups, score each row in the group.
    let warmup_wall = Instant::now();
    for w in 0..warmup {
        for g in 0..n_obs {
            let (start, end) = groups.start_end(g);
            for row_idx in start..end {
                let row = &data_f32[row_idx * n_cols..(row_idx + 1) * n_cols];
                let _ = qs.score_fast(row);
            }
        }
        if let Some(budget) = max_time && warmup_wall.elapsed() >= budget {
            eprintln!("  (qs warmup cut short after {}/{warmup} passes)", w + 1);
            break;
        }
    }

    // Bench: iterate over groups per iteration, report µs per observation.
    let qs_timing = {
        let mut timings = Vec::with_capacity(iters);
        let wall = Instant::now();
        for i in 0..iters {
            let iter_start = Instant::now();
            for g in 0..n_obs {
                let (start, end) = groups.start_end(g);
                for row_idx in start..end {
                    let row = &data_f32[row_idx * n_cols..(row_idx + 1) * n_cols];
                    let _ = qs.score_fast(row);
                }
            }
            timings.push(iter_start.elapsed().as_nanos() as f64 / 1000.0 / n_obs as f64);
            if let Some(budget) = max_time && i + 1 >= min_iters && wall.elapsed() >= budget {
                break;
            }
        }
        timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = timings.len();
        TimingSummary {
            median: timings[n / 2],
            p5: percentile(&timings, 5.0),
            p95: percentile(&timings, 95.0),
            actual_iters: n,
        }
    };

    eprintln!(
        "  quickscorer              {:.1}µs/obs  iters={}",
        qs_timing.median, qs_timing.actual_iters,
    );

    let qs_json = format!(
        "{{\"mode\":\"quickscorer\",\"latency_us_per_obs\":{:.6},\"p5_us\":{:.6},\"p95_us\":{:.6},\"iters\":{},\"n_obs\":{},\"parse_time_us\":{:.0}}}",
        qs_timing.median, qs_timing.p5, qs_timing.p95, qs_timing.actual_iters, n_obs, qs_parse_us,
    );
    println!("{qs_json}");
}

/// Build group iteration from forest config and optional offsets file.
// Allow: the assert_eq! inside the Some branch makes map_or_else less readable.
#[allow(clippy::option_if_let_else)]
fn build_groups(forest: &Forest, n_rows: usize, group_offsets_path: Option<&PathBuf>) -> Groups {
    if let Some(gpath) = group_offsets_path {
        let offsets = load_group_offsets(gpath);
        assert_eq!(
            *offsets.last().unwrap(), n_rows,
            "group_offsets last offset ({}) != n_rows ({n_rows})",
            offsets.last().unwrap(),
        );
        Groups::Variable { offsets }
    } else {
        let pl = forest.config.max_group_width;
        assert_eq!(
            n_rows % pl, 0,
            "n_rows ({n_rows}) is not divisible by group_width ({pl}); \
             tail rows would be silently dropped. Use --group-offsets for variable groups.",
        );
        Groups::Fixed { n_obs: n_rows / pl, group_width: pl }
    }
}

// ---------------------------------------------------------------------------
// Validate mode: TreeWalker output vs an f64 reference file.
// ---------------------------------------------------------------------------

/// Run TreeWalker (default partial-evaluation path) over every group in a cell
/// and compare per-row sigmoid probabilities to a precomputed reference.
///
/// The reference file uses the same layout as `write_raw_f64`:
/// `u64 n_rows`, `u64 n_cols`, then `n_rows * n_cols` LE f64 values. For the
/// scenario gate `n_cols = 1`. Prints `VALIDATE PASS max_delta=...` on success
/// or `VALIDATE FAIL ...` on failure and exits non-zero.
fn run_validate_mode(args: &[String]) {
    let model_dir = PathBuf::from(&args[1]);
    let mut data_dir: Option<PathBuf> = None;
    let mut group_offsets_path: Option<PathBuf> = None;
    let mut ref_path: Option<PathBuf> = None;
    let mut tol: f64 = 1e-14;

    let mut i = 2;
    while i < args.len() {
        let flag = args[i].as_str();
        let next = |i: usize, flag: &str| -> &String {
            args.get(i + 1).unwrap_or_else(|| {
                eprintln!("Missing value for {flag}");
                std::process::exit(1);
            })
        };
        match flag {
            "--data-dir" => { data_dir = Some(PathBuf::from(next(i, flag))); i += 2; }
            "--group-offsets" => { group_offsets_path = Some(PathBuf::from(next(i, flag))); i += 2; }
            "--validate" => { ref_path = Some(PathBuf::from(next(i, flag))); i += 2; }
            "--tol" => {
                tol = next(i, flag).parse().unwrap_or_else(|e| {
                    eprintln!("--tol: invalid value: {e}");
                    std::process::exit(1);
                });
                i += 2;
            }
            _ => { i += 1; }
        }
    }

    let data_dir = data_dir.as_ref().unwrap_or(&model_dir);
    let ref_path = ref_path.expect("--validate requires a reference file path");

    let bin_path = model_dir.join("model_treelite.bin");
    let json_path = model_dir.join("model_treelite.json");
    let model_path = if bin_path.exists() { bin_path } else { json_path };
    let config_path = data_dir.join("walker_config.json");
    let data_path = data_dir.join("test_data.bin");

    let mut forest = Forest::load(&model_path, &config_path);
    let (data, n_rows, _n_cols) = treewalker_bench::load_raw_f64(&data_path);
    let groups = build_groups(&forest, n_rows, group_offsets_path.as_ref());
    let n_obs = groups.n_obs();

    // Run TreeWalker over every group.
    let mut results = vec![0.0f64; n_rows];
    for g in 0..n_obs {
        let (s, e) = groups.start_end(g);
        forest.predict(&data, &mut results, s, e);
    }

    // Load reference (write_raw_f64 layout: u64 n_rows, u64 n_cols, f64s).
    let (ref_data, ref_n, ref_cols) = treewalker_bench::load_raw_f64(&ref_path);
    if ref_cols != 1 {
        eprintln!("VALIDATE FAIL: reference n_cols={ref_cols} != 1");
        std::process::exit(1);
    }
    if ref_n != n_rows {
        eprintln!("VALIDATE FAIL: reference has {ref_n} rows, data has {n_rows}");
        std::process::exit(1);
    }

    let mut max_delta = 0.0f64;
    let mut worst = 0usize;
    for r in 0..n_rows {
        let d = (results[r] - ref_data[r]).abs();
        if d > max_delta {
            max_delta = d;
            worst = r;
        }
    }

    if max_delta <= tol {
        println!("VALIDATE PASS max_delta={max_delta:.6e} tol={tol:.0e} n_rows={n_rows} n_obs={n_obs}");
        eprintln!("  VALIDATE PASS max_delta={max_delta:.6e} (worst row {worst}) tol={tol:.0e}");
    } else {
        eprintln!("VALIDATE FAIL max_delta={max_delta:.6e} > tol={tol:.0e} (worst row {worst})");
        eprintln!("  tw={:.17e} ref={:.17e}", results[worst], ref_data[worst]);
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Grid mode: all methods, all cells, CSV output
// ---------------------------------------------------------------------------

/// Parse --grid mode arguments and dispatch to bench::run_grid().
fn run_grid_mode(args: &[String]) {
    use treewalker_bench as bench;

    let artifacts_dir = PathBuf::from(&args[1]);

    let mut grid = bench::grid::Grid::All;
    let mut output_dir = PathBuf::from(".");
    let mut datasets_filter: Option<Vec<String>> = None;
    let mut warmup: usize = 3;
    let mut max_iters: usize = 21;
    let mut min_iters: usize = 11;
    let mut max_time_secs: Option<f64> = None;
    let mut seed: u64 = 42;
    #[cfg(feature = "external-bench")]
    let mut lgb_lib: Option<PathBuf> = None;
    #[cfg(feature = "external-bench")]
    let mut xgb_lib: Option<PathBuf> = None;

    let mut i = 2;
    while i < args.len() {
        let flag = args[i].as_str();
        let next_val = |idx: usize, f: &str| -> &String {
            args.get(idx + 1).unwrap_or_else(|| {
                eprintln!("Missing value for {f}");
                std::process::exit(1);
            })
        };
        match flag {
            "--grid" => {
                let val = next_val(i, flag);
                grid = match val.as_str() {
                    "1" => bench::grid::Grid::G1,
                    "3" => bench::grid::Grid::G3,
                    "4" => bench::grid::Grid::G4,
                    "scen" => bench::grid::Grid::Scen,
                    "all" => bench::grid::Grid::All,
                    other => {
                        eprintln!("Unknown grid: {other}. Use 1, 3, 4, scen, or all.");
                        std::process::exit(1);
                    }
                };
                i += 2;
            }
            "--output-dir" => {
                output_dir = PathBuf::from(next_val(i, flag));
                i += 2;
            }
            "--datasets" => {
                let val = next_val(i, flag);
                datasets_filter = Some(val.split(',').map(String::from).collect());
                i += 2;
            }
            "--warmup" => {
                warmup = next_val(i, flag).parse().unwrap_or_else(|e| {
                    eprintln!("--warmup: {e}"); std::process::exit(1);
                });
                i += 2;
            }
            "--iters" => {
                max_iters = next_val(i, flag).parse().unwrap_or_else(|e| {
                    eprintln!("--iters: {e}"); std::process::exit(1);
                });
                i += 2;
            }
            "--min-iters" => {
                min_iters = next_val(i, flag).parse().unwrap_or_else(|e| {
                    eprintln!("--min-iters: {e}"); std::process::exit(1);
                });
                i += 2;
            }
            "--max-time-secs" => {
                max_time_secs = Some(next_val(i, flag).parse().unwrap_or_else(|e| {
                    eprintln!("--max-time-secs: {e}"); std::process::exit(1);
                }));
                i += 2;
            }
            "--seed" => {
                seed = next_val(i, flag).parse().unwrap_or_else(|e| {
                    eprintln!("--seed: {e}"); std::process::exit(1);
                });
                i += 2;
            }
            #[cfg(feature = "external-bench")]
            "--lgb-lib" => {
                lgb_lib = Some(PathBuf::from(next_val(i, flag)));
                i += 2;
            }
            #[cfg(feature = "external-bench")]
            "--xgb-lib" => {
                xgb_lib = Some(PathBuf::from(next_val(i, flag)));
                i += 2;
            }
            _ => {
                // Skip unknown flags silently (they might be for single-cell mode).
                i += 1;
            }
        }
    }

    let config = bench::RunConfig {
        artifacts_dir,
        output_dir,
        grid,
        datasets_filter,
        warmup,
        max_iters,
        min_iters,
        max_time_secs,
        seed,
        collect_stats: true,
        #[cfg(feature = "external-bench")]
        lgb_lib,
        #[cfg(feature = "external-bench")]
        xgb_lib,
    };

    bench::run_grid(&config);
}
