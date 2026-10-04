//! TreeWalker benchmark orchestration: run all methods on all grid cells, write CSV.
//!
//! This module replaces `sweep.py` — the Rust binary handles grid enumeration,
//! method loading, per-group timing, and CSV output directly.
//!
//! # Architecture
//!
//! `CellData` loads test data, model, and group boundaries for one cell.
//! `collect_methods` builds `BenchMethod` closures for all available prediction
//! methods. Grid runners iterate cells, call these helpers, run [`bench_blocked`],
//! and write CSV rows.

pub mod csv;
mod data;
#[cfg(feature = "external-bench")]
pub mod external;
pub mod grid;
mod system;
pub mod timing;

pub use data::{load_group_offsets, load_raw_f64, try_load_raw_f64};
pub use system::get_rss_kb;

use std::path::{Path, PathBuf};

use treewalker_gbdt::research::{Ablation, WorkCounters};
use treewalker_gbdt::{Forest, LoadOptions};

use self::csv::CsvWriter;
use self::grid::GridCell;
use self::timing::{BenchMethod, BlockConfig, TimingResult, bench_blocked};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of benchmarking one method on one cell.
#[derive(Debug)]
pub struct MethodResult {
    pub method: String,
    pub timing: TimingResult,
}

/// Configuration for the grid orchestrator.
pub struct RunConfig {
    pub artifacts_dir: PathBuf,
    pub output_dir: PathBuf,
    pub grid: grid::Grid,
    pub datasets_filter: Option<Vec<String>>,
    pub warmup: usize,
    pub max_iters: usize,
    pub min_iters: usize,
    pub max_time_secs: Option<f64>,
    pub seed: u64,
    /// Collect work counters after Grid 3 ablation (separate CSV, not in timing loop).
    pub collect_stats: bool,
    #[cfg(feature = "external-bench")]
    pub lgb_lib: Option<PathBuf>,
    #[cfg(feature = "external-bench")]
    pub xgb_lib: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Cell data loading (shared by all grid runners)
// ---------------------------------------------------------------------------

/// Loaded data for one benchmark cell: test array, forest, group boundaries.
struct CellData {
    data: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
    forest: Forest,
    bounds: Vec<(usize, usize)>,
    /// Time to parse the forest model (microseconds).
    parse_time_us: f64,
    /// Model file size in bytes.
    model_bytes: u64,
}

impl CellData {
    /// Load test data, model, and group boundaries. Returns `None` if files are missing.
    fn load(data_dir: &Path, fw_dir: &Path, options: &LoadOptions) -> Option<Self> {
        let data_path = data_dir.join("test_data.bin");
        if !data_path.exists() {
            return None;
        }
        let (data, n_rows, n_cols) = crate::load_raw_f64(&data_path);

        let config_path = if data_dir.join("walker_config.json").exists() {
            data_dir.join("walker_config.json")
        } else {
            data_dir.parent()?.join("walker_config.json")
        };
        if !config_path.exists() {
            return None;
        }

        let bin_path = fw_dir.join("model_treelite.bin");
        let json_path = fw_dir.join("model_treelite.json");
        let model_path = if bin_path.exists() {
            bin_path
        } else {
            json_path
        };
        if !model_path.exists() {
            return None;
        }

        let model_bytes = std::fs::metadata(&model_path).map_or(0, |m| m.len());
        let t0 = std::time::Instant::now();
        let forest =
            Forest::load_with(&model_path, &config_path, options).unwrap_or_else(|e| panic!("{e}"));
        let parse_time_us = t0.elapsed().as_nanos() as f64 / 1000.0;

        let bounds = build_group_boundaries(&forest, n_rows, data_dir);
        Some(Self {
            data,
            n_rows,
            n_cols,
            forest,
            bounds,
            parse_time_us,
            model_bytes,
        })
    }

    fn data_ref(&self) -> &[f64] {
        &self.data
    }
}

/// Build group boundaries from forest config and optional offsets file.
fn build_group_boundaries(forest: &Forest, n_rows: usize, data_dir: &Path) -> Vec<(usize, usize)> {
    let go_path = data_dir.join("group_offsets.bin");
    if go_path.exists() {
        let offsets = crate::load_group_offsets(&go_path);
        (0..offsets.len() - 1)
            .map(|i| (offsets[i], offsets[i + 1]))
            .collect()
    } else {
        let gw = forest.config().max_group_width();
        assert_eq!(
            n_rows % gw,
            0,
            "n_rows ({n_rows}) not divisible by group_width ({gw})"
        );
        let n_obs = n_rows / gw;
        (0..n_obs).map(|g| (g * gw, (g + 1) * gw)).collect()
    }
}

// ---------------------------------------------------------------------------
// Method collection (shared by all grid runners)
// ---------------------------------------------------------------------------

/// Build `BenchMethod` closures for all available prediction methods.
///
/// Always includes TreeWalker baseline and full. External methods (lleaves,
/// tl2cgen, LightGBM, XGBoost, QuickScorer) added when available.
fn collect_methods<'a>(
    cell_data: &'a CellData,
    fw_dir: &Path,
    framework: &str,
    config: &RunConfig,
) -> Vec<BenchMethod<'a>> {
    let data_ref = cell_data.data_ref();
    let n_rows = cell_data.n_rows;
    let n_cols = cell_data.n_cols;
    let forest = &cell_data.forest;

    // These inputs are only needed by the optional external baselines.
    #[cfg(not(feature = "external-bench"))]
    let _ = (fw_dir, framework, config, data_ref, n_rows, n_cols, forest);

    let methods = treewalker_methods(cell_data);

    #[cfg(feature = "external-bench")]
    let methods = {
        use self::external::ExternalMethod;

        let mut methods = methods;

        if framework == "lightgbm"
            && let Some(mut m) = external::LleavesBench::load(&fw_dir.join("lleaves.so"), n_rows)
        {
            methods.push(BenchMethod {
                name: m.name().to_string(),
                predict_group: Box::new(move |s, e| {
                    m.predict_group(data_ref, n_cols, s, e);
                }),
            });
        }

        if let Some(mut m) = external::Tl2cgenBench::load(&fw_dir.join("tl2cgen.so")) {
            methods.push(BenchMethod {
                name: m.name().to_string(),
                predict_group: Box::new(move |s, e| {
                    m.predict_group(data_ref, n_cols, s, e);
                }),
            });
        }

        if framework == "lightgbm"
            && let Some(ref lgb_lib) = config.lgb_lib
        {
            let max_gw = forest.config().max_group_width();
            if let Some(mut m) = external::LightGBMBench::load(
                lgb_lib,
                &fw_dir.join("model_native.txt"),
                n_cols,
                max_gw,
            ) {
                methods.push(BenchMethod {
                    name: m.name().to_string(),
                    predict_group: Box::new(move |s, e| {
                        m.predict_group(data_ref, n_cols, s, e);
                    }),
                });
            }
        }

        if framework == "xgboost"
            && let Some(ref xgb_lib) = config.xgb_lib
            && let Some(mut m) =
                external::XGBoostBench::load(xgb_lib, &fw_dir.join("model_native.json"), n_cols)
        {
            methods.push(BenchMethod {
                name: m.name().to_string(),
                predict_group: Box::new(move |s, e| {
                    m.predict_group(data_ref, n_cols, s, e);
                }),
            });
        }

        // QuickScorer: only works with LightGBM TXT format.
        if framework == "lightgbm"
            && let Some(mut m) =
                external::QuickScorerBench::load(&fw_dir.join("model_native.txt"), data_ref, n_cols)
        {
            methods.push(BenchMethod {
                name: m.name().to_string(),
                predict_group: Box::new(move |s, e| {
                    m.predict_group(data_ref, n_cols, s, e);
                }),
            });
        }
        methods
    };

    methods
}

/// Build a `BlockConfig` from `RunConfig`.
fn block_config(config: &RunConfig) -> BlockConfig {
    BlockConfig {
        n_batches: 12,
        min_blocks: config.min_iters,
        max_blocks: config.max_iters,
        warmup: config.warmup,
        max_time: config.max_time_secs.map(std::time::Duration::from_secs_f64),
        precision_target_pct: 3.0,
    }
}

/// Architecture label for output files.
const fn detect_architecture() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "arm"
    } else {
        "intel"
    }
}

// ---------------------------------------------------------------------------
// Grid 1: Full factorial comparison (all methods)
// ---------------------------------------------------------------------------

/// Sentinel drift detection period (re-time the first cell every N cells).
const SENTINEL_PERIOD: usize = 25;

fn run_grid1(config: &RunConfig) {
    let arch = detect_architecture();
    let cells = grid::discover_cells(
        &config.artifacts_dir,
        grid::Grid::G1,
        config.datasets_filter.as_deref(),
    );
    if cells.is_empty() {
        eprintln!("Grid 1: no cells found, skipping");
        return;
    }

    let csv_path = config.output_dir.join(format!("grid1_results_{arch}.csv"));
    let columns = [
        "dataset",
        "framework",
        "arch",
        "n_trees",
        "max_depth",
        "horizon",
        "method",
        "n_obs",
        "iters",
        "median_us",
        "p5_us",
        "p95_us",
    ];
    let key_cols = [
        "dataset",
        "framework",
        "n_trees",
        "max_depth",
        "horizon",
        "method",
    ];
    let mut csv = CsvWriter::new(&csv_path, &columns, &key_cols);
    let bcfg = block_config(config);

    eprintln!("\n{}", "=".repeat(60));
    eprintln!(
        "Grid 1: {} cells, output: {}",
        cells.len(),
        csv_path.display()
    );
    eprintln!("{}", "=".repeat(60));

    let mut sentinel_baseline: Option<f64> = None;

    for (i, cell) in cells.iter().enumerate() {
        eprintln!("\n[{}/{}] {}", i + 1, cells.len(), cell.label());

        let Some(cd) = CellData::load(&cell.param_dir, &cell.fw_dir, &LoadOptions::default())
        else {
            eprintln!("  skipping (missing files)");
            continue;
        };
        let mut methods = collect_methods(&cd, &cell.fw_dir, &cell.framework, config);
        let timings = bench_blocked(&cd.bounds, &mut methods, &bcfg);

        let h_str = if cell.is_ctr {
            String::new()
        } else {
            cell.horizon.to_string()
        };
        for (m, t) in methods.iter().zip(&timings) {
            eprintln!(
                "    {}: {:.1}µs/obs (blocks={})",
                m.name, t.median_us, t.actual_blocks
            );
            csv.write_row(&[
                cell.dataset.as_str(),
                cell.framework.as_str(),
                arch,
                &cell.nt.to_string(),
                &cell.md.to_string(),
                &h_str,
                &m.name,
                &t.n_obs.to_string(),
                &t.actual_blocks.to_string(),
                &format!("{:.6}", t.median_us),
                &format!("{:.6}", t.p5_us),
                &format!("{:.6}", t.p95_us),
            ]);
        }

        // Record sentinel from first cell's baseline.
        if i == 0
            && let Some(t) = timings.first()
        {
            sentinel_baseline = Some(t.median_us);
        }
        // Check sentinel drift every SENTINEL_PERIOD cells.
        if i > 0
            && i % SENTINEL_PERIOD == 0
            && let Some(baseline_us) = sentinel_baseline
        {
            // Re-time the first cell.
            let sentinel_cell = &cells[0];
            if let Some(scd) = CellData::load(
                &sentinel_cell.param_dir,
                &sentinel_cell.fw_dir,
                &LoadOptions::default(),
            ) && let Some(st) = {
                let mut smethods = collect_methods(
                    &scd,
                    &sentinel_cell.fw_dir,
                    &sentinel_cell.framework,
                    config,
                );
                let stimings = bench_blocked(&scd.bounds, &mut smethods, &bcfg);
                stimings.into_iter().next()
            } {
                let drift_pct = ((st.median_us - baseline_us) / baseline_us * 100.0).abs();
                if drift_pct > 3.0 {
                    eprintln!(
                        "  \u{26a0} SENTINEL DRIFT: {drift_pct:.1}% (baseline={baseline_us:.1}, now={:.1})",
                        st.median_us
                    );
                } else {
                    eprintln!("  \u{2713} sentinel: {drift_pct:.1}% drift");
                }
            }
        }
    }
    eprintln!(
        "\nGrid 1 done: {} rows \u{2192} {}",
        csv.n_rows(),
        csv.path().display()
    );
}

// ---------------------------------------------------------------------------
// Grid 3: Ablation interaction study
// ---------------------------------------------------------------------------

fn run_grid3_ablation(config: &RunConfig) {
    let arch = detect_architecture();
    let cells = grid::discover_cells(
        &config.artifacts_dir,
        grid::Grid::G3,
        config.datasets_filter.as_deref(),
    );
    if cells.is_empty() {
        eprintln!("Grid 3: no cells found, skipping");
        return;
    }

    let csv_path = config.output_dir.join(format!("grid3_results_{arch}.csv"));
    let columns = [
        "dataset",
        "framework",
        "arch",
        "n_trees",
        "max_depth",
        "horizon",
        "disable_precompute",
        "disable_unsplit",
        "disable_monotonic",
        "disable_tree_ordering",
        "disable_prefix_grouping",
        "disable_bitset_intern",
        "n_obs",
        "iters",
        "median_us",
        "p5_us",
        "p95_us",
    ];
    let key_cols = [
        "dataset",
        "framework",
        "n_trees",
        "max_depth",
        "horizon",
        "disable_precompute",
        "disable_unsplit",
        "disable_monotonic",
        "disable_tree_ordering",
        "disable_prefix_grouping",
        "disable_bitset_intern",
    ];
    let mut csv = CsvWriter::new(&csv_path, &columns, &key_cols);
    let bcfg = block_config(config);

    let b_prime: std::collections::HashSet<(usize, usize, usize)> =
        grid::ABLATION_ANCHORS_B_PRIME.iter().copied().collect();

    eprintln!("\n{}", "=".repeat(60));
    eprintln!("Grid 3 (ablation): {} anchor cells", cells.len());
    eprintln!("{}", "=".repeat(60));

    for (i, cell) in cells.iter().enumerate() {
        let is_prime = b_prime.contains(&(cell.nt, cell.md, cell.horizon));
        let combos = ablation_combos(is_prime);
        eprintln!(
            "\n[{}/{}] {} ({} combos, {})",
            i + 1,
            cells.len(),
            cell.label(),
            combos.len(),
            if is_prime {
                "full-cross"
            } else {
                "within-group"
            },
        );

        // Group combos by LoadOptions to minimize forest reloads.
        let mut by_parse: std::collections::BTreeMap<(bool, bool, bool), Vec<(Ablation, String)>> =
            std::collections::BTreeMap::new();
        for (am, pc, label) in &combos {
            let key = (
                pc.disable_tree_ordering,
                pc.prefix_depth == 0,
                pc.disable_bitset_intern,
            );
            by_parse.entry(key).or_default().push((*am, label.clone()));
        }

        for (pkey, runtime_combos) in &by_parse {
            let pc = LoadOptions {
                disable_tree_ordering: pkey.0,
                prefix_depth: if pkey.1 { 0 } else { 2 },
                disable_bitset_intern: pkey.2,
                disable_predicate_dedup: false,
            };
            let Some(cd) = CellData::load(&cell.param_dir, &cell.fw_dir, &pc) else {
                continue;
            };
            let data_ref = cd.data_ref();
            let (n_rows, nf) = (cd.n_rows, cd.n_cols);

            // One BenchMethod per runtime combo, each a research predictor with its
            // flags, the unablated reference included, so ratios compare one build.
            let mut methods: Vec<BenchMethod<'_>> = Vec::new();
            let mut labels: Vec<&str> = Vec::new();
            for (ablation, label) in runtime_combos {
                let mut results_buf = vec![0.0f64; n_rows];
                let mut predictor = cd.forest.research_predictor(*ablation);
                methods.push(BenchMethod {
                    name: label.clone(),
                    predict_group: Box::new(move |s, e| {
                        predictor.predict_group(&data_ref[s * nf..e * nf], &mut results_buf[s..e]);
                    }),
                });
                labels.push(label);
            }

            let timings = bench_blocked(&cd.bounds, &mut methods, &bcfg);

            let h_str = if cell.is_ctr {
                String::new()
            } else {
                cell.horizon.to_string()
            };
            for (label, t) in labels.iter().zip(&timings) {
                eprintln!(
                    "    {label}: {:.1}µs/obs (blocks={})",
                    t.median_us, t.actual_blocks
                );
                // Parse the label back into flag values.
                let flags: Vec<&str> = label
                    .split(',')
                    .map(|f| f.split('=').nth(1).unwrap_or("0"))
                    .collect();
                let nt_s = cell.nt.to_string();
                let md_s = cell.md.to_string();
                let n_obs_s = t.n_obs.to_string();
                let blocks_s = t.actual_blocks.to_string();
                let med_s = format!("{:.6}", t.median_us);
                let p5_s = format!("{:.6}", t.p5_us);
                let p95_s = format!("{:.6}", t.p95_us);
                csv.write_row(&[
                    cell.dataset.as_str(),
                    cell.framework.as_str(),
                    arch,
                    &nt_s,
                    &md_s,
                    &h_str,
                    flags.first().unwrap_or(&"0"),
                    flags.get(1).unwrap_or(&"0"),
                    flags.get(2).unwrap_or(&"0"),
                    flags.get(3).unwrap_or(&"0"),
                    flags.get(4).unwrap_or(&"0"),
                    flags.get(5).unwrap_or(&"0"),
                    &n_obs_s,
                    &blocks_s,
                    &med_s,
                    &p5_s,
                    &p95_s,
                ]);
            }
        }
    }

    eprintln!(
        "\nGrid 3 done: {} rows → {}",
        csv.n_rows(),
        csv.path().display()
    );

    // Stats collection pass (separate CSV).
    if config.collect_stats {
        collect_grid3_stats(config, &cells, arch);
    }
}

/// Collect work counters for Grid 3 (separate from timing).
fn collect_grid3_stats(config: &RunConfig, cells: &[GridCell], arch: &str) {
    let stats_path = config.output_dir.join(format!("grid3_stats_{arch}.csv"));
    let columns = [
        "dataset",
        "framework",
        "arch",
        "n_trees",
        "max_depth",
        "horizon",
        "disable_precompute",
        "disable_unsplit",
        "disable_monotonic",
        "disable_tree_ordering",
        "disable_prefix_grouping",
        "disable_bitset_intern",
        "n_obs",
        "constant_steps",
        "varying_splits",
        "unsplit_skips",
        "recursive_calls",
        "leaf_hits",
        "partition_row_evals",
        "precompute_row_evals",
        "parse_time_us",
        "rss_kb",
        "model_bytes",
    ];
    let key_cols = [
        "dataset",
        "framework",
        "n_trees",
        "max_depth",
        "horizon",
        "disable_precompute",
        "disable_unsplit",
        "disable_monotonic",
        "disable_tree_ordering",
        "disable_prefix_grouping",
        "disable_bitset_intern",
    ];
    let mut csv = CsvWriter::new(&stats_path, &columns, &key_cols);
    let b_prime: std::collections::HashSet<(usize, usize, usize)> =
        grid::ABLATION_ANCHORS_B_PRIME.iter().copied().collect();

    eprintln!("\nCollecting work counters for Grid 3...");

    for cell in cells {
        let is_prime = b_prime.contains(&(cell.nt, cell.md, cell.horizon));
        let combos = ablation_combos(is_prime);

        // Group by LoadOptions.
        let mut by_parse: std::collections::BTreeMap<(bool, bool, bool), Vec<(Ablation, String)>> =
            std::collections::BTreeMap::new();
        for (am, pc, label) in &combos {
            let key = (
                pc.disable_tree_ordering,
                pc.prefix_depth == 0,
                pc.disable_bitset_intern,
            );
            by_parse.entry(key).or_default().push((*am, label.clone()));
        }

        for (pkey, runtime_combos) in &by_parse {
            let pc = LoadOptions {
                disable_tree_ordering: pkey.0,
                prefix_depth: if pkey.1 { 0 } else { 2 },
                disable_bitset_intern: pkey.2,
                disable_predicate_dedup: false,
            };
            let Some(cd) = CellData::load(&cell.param_dir, &cell.fw_dir, &pc) else {
                continue;
            };
            let mut results = vec![0.0f64; cd.n_rows];

            for (ablation, label) in runtime_combos {
                let total = count_work(&cd, *ablation, &mut results);
                let h_str = if cell.is_ctr {
                    String::new()
                } else {
                    cell.horizon.to_string()
                };
                let flags: Vec<&str> = label
                    .split(',')
                    .map(|f| f.split('=').nth(1).unwrap_or("0"))
                    .collect();
                let nt_s = cell.nt.to_string();
                let md_s = cell.md.to_string();
                let n_obs_s = cd.bounds.len().to_string();
                csv.write_row(&[
                    cell.dataset.as_str(),
                    cell.framework.as_str(),
                    arch,
                    &nt_s,
                    &md_s,
                    &h_str,
                    flags.first().unwrap_or(&"0"),
                    flags.get(1).unwrap_or(&"0"),
                    flags.get(2).unwrap_or(&"0"),
                    flags.get(3).unwrap_or(&"0"),
                    flags.get(4).unwrap_or(&"0"),
                    flags.get(5).unwrap_or(&"0"),
                    &n_obs_s,
                    &total.constant_steps.to_string(),
                    &total.varying_splits.to_string(),
                    &total.unsplit_skips.to_string(),
                    &total.recursive_calls.to_string(),
                    &total.leaf_hits.to_string(),
                    &total.partition_row_evals.to_string(),
                    &total.precompute_row_evals.to_string(),
                    &format!("{:.0}", cd.parse_time_us),
                    &crate::get_rss_kb().to_string(),
                    &cd.model_bytes.to_string(),
                ]);
            }
        }
        eprintln!("  {} stats collected", cell.label());
    }
    eprintln!(
        "Grid 3 stats done: {} rows → {}",
        csv.n_rows(),
        stats_path.display()
    );
}

// ---------------------------------------------------------------------------
// Grid 4: Group distribution experiments (Expedia only)
// ---------------------------------------------------------------------------

/// Known distribution subdirectory names (from prepare.py --prepare-groups).
const GROUP_DISTRIBUTIONS: &[&str] = &["empirical", "fixed8", "fixed16", "fixed32", "geom8"];

fn run_grid4_distributions(config: &RunConfig) {
    let arch = detect_architecture();
    let expedia_dir = config.artifacts_dir.join("expedia");
    if !expedia_dir.exists() {
        eprintln!("Grid 4: no expedia directory, skipping");
        return;
    }

    let csv_path = config.output_dir.join(format!("grid4_results_{arch}.csv"));
    let columns = [
        "dataset",
        "framework",
        "arch",
        "n_trees",
        "max_depth",
        "group_dist",
        "mean_group_size",
        "n_groups",
        "method",
        "n_obs",
        "iters",
        "median_us",
        "p5_us",
        "p95_us",
    ];
    let key_cols = [
        "dataset",
        "framework",
        "n_trees",
        "max_depth",
        "group_dist",
        "method",
    ];
    let mut csv = CsvWriter::new(&csv_path, &columns, &key_cols);
    let bcfg = block_config(config);

    // Discover Expedia param dirs.
    let Ok(entries) = std::fs::read_dir(&expedia_dir) else {
        return;
    };
    let mut param_dirs: Vec<_> = entries
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|ft| ft.is_dir()))
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with("nt") && !name.contains('h')
        })
        .collect();
    param_dirs.sort_by_key(std::fs::DirEntry::file_name);

    eprintln!("\n{}", "=".repeat(60));
    eprintln!(
        "Grid 4 (distributions): {} param dirs × {} dists × 2 fw",
        param_dirs.len(),
        GROUP_DISTRIBUTIONS.len()
    );
    eprintln!("{}", "=".repeat(60));

    let mut cell_idx = 0;
    for pd_entry in &param_dirs {
        let param_dir = pd_entry.path();
        let param_name = pd_entry.file_name().to_string_lossy().to_string();
        let Some((nt, md, _)) = grid::parse_param_dir(&param_name) else {
            continue;
        };

        for &dist_name in GROUP_DISTRIBUTIONS {
            // Empirical reuses the parent directory's data.
            let dist_data_dir = if dist_name == "empirical" {
                param_dir.clone()
            } else {
                param_dir.join(dist_name)
            };
            let go_path = dist_data_dir.join("group_offsets.bin");
            if !go_path.exists() {
                continue;
            }

            // Compute mean group size from offsets.
            let offsets = crate::load_group_offsets(&go_path);
            let n_groups = offsets.len() - 1;
            if n_groups == 0 {
                continue;
            }
            let mean_gs: f64 = offsets
                .windows(2)
                .map(|w| (w[1] - w[0]) as f64)
                .sum::<f64>()
                / n_groups as f64;

            for framework in &["lightgbm", "xgboost"] {
                let fw_dir = param_dir.join(framework);
                let Some(cd) = CellData::load(&dist_data_dir, &fw_dir, &LoadOptions::default())
                else {
                    continue;
                };
                cell_idx += 1;
                eprintln!("\n[{cell_idx}] expedia/{param_name}/{framework}/{dist_name}");

                let mut methods = collect_methods(&cd, &fw_dir, framework, config);
                let timings = bench_blocked(&cd.bounds, &mut methods, &bcfg);

                let nt_s = nt.to_string();
                let md_s = md.to_string();
                let gs_s = format!("{mean_gs:.1}");
                let ng_s = n_groups.to_string();
                for (m, t) in methods.iter().zip(&timings) {
                    eprintln!(
                        "    {}: {:.1}µs/obs (blocks={})",
                        m.name, t.median_us, t.actual_blocks
                    );
                    csv.write_row(&[
                        "expedia",
                        framework,
                        arch,
                        &nt_s,
                        &md_s,
                        dist_name,
                        &gs_s,
                        &ng_s,
                        &m.name,
                        &t.n_obs.to_string(),
                        &t.actual_blocks.to_string(),
                        &format!("{:.6}", t.median_us),
                        &format!("{:.6}", t.p5_us),
                        &format!("{:.6}", t.p95_us),
                    ]);
                }
            }
        }
    }
    eprintln!(
        "\nGrid 4 done: {} rows → {}",
        csv.n_rows(),
        csv.path().display()
    );
}

// ---------------------------------------------------------------------------
// E1: Scenario-analysis benchmark (UCI Default of Credit)
// ---------------------------------------------------------------------------

/// Scenario cell on disk: parsed `(k, G)` plus its directory path.
struct ScenCell {
    k: usize,
    g: usize,
    dir: PathBuf,
}

/// Core 3x3 `{1,4,8} x {4,16,128}` first, then the remaining 7 cells.
fn scen_cell_order(cells: &[ScenCell]) -> Vec<&ScenCell> {
    let core: std::collections::HashSet<(usize, usize)> = [1, 4, 8]
        .iter()
        .flat_map(|&k| [4, 16, 128].iter().map(move |&g| (k, g)))
        .collect();
    let mut indexed: Vec<(bool, &ScenCell)> = cells
        .iter()
        .map(|c| (core.contains(&(c.k, c.g)), c))
        .collect();
    // Core cells {k=1,4,8}x{G=4,16,128} first in fixed (k, G) ascending order,
    // then the remaining cells in fixed (k, G) ascending order. Deterministic
    // run-to-run and independent of filesystem read_dir order, so CSV row order
    // is reproducible across machines.
    indexed.sort_by_key(|(core_a, a)| (!*core_a, a.k, a.g));
    indexed.into_iter().map(|(_, c)| c).collect()
}

/// Build the two `BenchMethod` closures the E1 scenario grid times: TreeWalker
/// (partial evaluation) and the same-layout row-independent full walk.
///
/// The binding spec for `--grid scen` is EXACTLY these two methods. External
/// baselines (lleaves, tl2cgen, native LightGBM/XGBoost, QuickScorer) are NEVER
/// timed here, regardless of `external-bench` flags or artifacts on disk.
/// `collect_methods` is intentionally not used so flags/artifacts cannot leak
/// other methods into the scenario timing CSV.
fn scen_collect_methods(cell_data: &CellData) -> Vec<BenchMethod<'_>> {
    treewalker_methods(cell_data)
}

/// TreeWalker's full walk and production partial evaluation, timed one group at a time.
fn treewalker_methods(cell_data: &CellData) -> Vec<BenchMethod<'_>> {
    let data_ref = cell_data.data_ref();
    let (n_rows, nf) = (cell_data.n_rows, cell_data.n_cols);
    let forest = &cell_data.forest;

    let mut tw_base_results = vec![0.0f64; n_rows];
    let mut tw_full_results = vec![0.0f64; n_rows];
    let mut predictor = forest.predictor();

    vec![
        // Row-independent baseline: walks each row fully through every tree,
        // no partial evaluation. Paper label: "TreeWalker (full walk)".
        BenchMethod {
            name: "treewalker_fullwalk".to_string(),
            predict_group: Box::new(move |s, e| {
                forest.predict_full_walk(&data_ref[s * nf..e * nf], &mut tw_base_results[s..e]);
            }),
        },
        // Optimized TreeWalker: partial evaluation exploiting constant features.
        // Paper label: "TreeWalker".
        BenchMethod {
            name: "treewalker".to_string(),
            predict_group: Box::new(move |s, e| {
                predictor.predict_group(&data_ref[s * nf..e * nf], &mut tw_full_results[s..e]);
            }),
        },
    ]
}

/// Work counters of one runtime variant, summed over every group of the cell.
fn count_work(cd: &CellData, ablation: Ablation, results: &mut [f64]) -> WorkCounters {
    let nf = cd.n_cols;
    let mut predictor = cd.forest.research_predictor(ablation);
    let mut total = WorkCounters::default();
    for &(s, e) in &cd.bounds {
        total += predictor.predict_group_counted(&cd.data[s * nf..e * nf], &mut results[s..e]);
    }
    total
}

/// Run the E1 scenario-analysis benchmark: TreeWalker + same-layout full walk
/// over all `(k, G)` cells, with work counters and an inline
/// correctness check against the GTIL f64 reference.
///
/// Layout (written by `prepare_scenario.py`):
///   `<artifacts>/scenario_credit/lightgbm/`  — shared trained model
///   `<artifacts>/scenario_credit/cells/k{K}_G{G}/` — per-cell data + config
///        + `reference.bin` (GTIL f64 reference for the correctness gate)
fn run_grid_scen(config: &RunConfig) {
    let arch = detect_architecture();
    let scen_root = config.artifacts_dir.join("scenario_credit");
    let model_dir = scen_root.join("lightgbm");
    let cells_root = scen_root.join("cells");
    if !model_dir.exists() || !cells_root.exists() {
        eprintln!(
            "Scen: no scenario_credit/ under {}, skipping",
            config.artifacts_dir.display()
        );
        return;
    }

    // Discover cell directories.
    let mut cells: Vec<ScenCell> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&cells_root) {
        for e in entries.flatten() {
            if !e.file_type().is_ok_and(|ft| ft.is_dir()) {
                continue;
            }
            let name = e.file_name().to_string_lossy().to_string();
            if let Some((k, g)) = grid::parse_scen_cell(&name) {
                cells.push(ScenCell {
                    k,
                    g,
                    dir: e.path(),
                });
            }
        }
    }
    if cells.is_empty() {
        eprintln!(
            "Scen: no k{{K}}_G{{G}} cells under {}, skipping",
            cells_root.display()
        );
        return;
    }
    let ordered = scen_cell_order(&cells);

    let results_path = config
        .output_dir
        .join(format!("scenario_credit_results_{arch}.csv"));
    let stats_path = config
        .output_dir
        .join(format!("scenario_credit_stats_{arch}.csv"));

    let res_cols = [
        "dataset",
        "framework",
        "arch",
        "k",
        "G",
        "n_trees",
        "max_depth",
        "n_groups",
        "method",
        "n_obs",
        "iters",
        "median_us",
        "p5_us",
        "p95_us",
    ];
    let res_keys = ["k", "G", "method"];
    let mut res_csv = CsvWriter::new(&results_path, &res_cols, &res_keys);

    let stat_cols = [
        "dataset",
        "framework",
        "arch",
        "k",
        "G",
        "n_trees",
        "max_depth",
        "evaluator",
        "n_obs",
        "constant_steps",
        "varying_splits",
        "unsplit_skips",
        "recursive_calls",
        "leaf_hits",
        "partition_row_evals",
        "precompute_row_evals",
    ];
    let stat_keys = ["k", "G", "evaluator"];
    let mut stat_csv = CsvWriter::new(&stats_path, &stat_cols, &stat_keys);

    let bcfg = block_config(config);

    eprintln!("\n{}", "=".repeat(60));
    eprintln!("Scen (scenario-analysis): {} cells", ordered.len());
    eprintln!("  results -> {}", results_path.display());
    eprintln!("  stats   -> {}", stats_path.display());
    eprintln!("{}", "=".repeat(60));

    for (i, cell) in ordered.iter().enumerate() {
        let label = format!("scenario_credit/k{}_G{}", cell.k, cell.g);
        eprintln!("\n[{}/{}] {}", i + 1, ordered.len(), label);

        let Some(cd) = CellData::load(&cell.dir, &model_dir, &LoadOptions::default()) else {
            eprintln!("  skipping (missing files)");
            continue;
        };
        let n_trees = cd.forest.trees().len();
        let n_groups = cd.bounds.len();

        // Inline correctness: TreeWalker vs GTIL f64 reference (TOL_F64).
        if !validate_scen_cell(&cd, &cell.dir) {
            eprintln!("  CORRECTNESS FAIL: aborting scen grid");
            std::process::exit(1);
        }

        // Timing: TreeWalker + same-layout full walk ONLY (scen_collect_methods;
        // external baselines are never timed under --grid scen regardless of flags).
        let mut methods = scen_collect_methods(&cd);
        let timings = bench_blocked(&cd.bounds, &mut methods, &bcfg);

        let k_s = cell.k.to_string();
        let g_s = cell.g.to_string();
        let nt_s = n_trees.to_string();
        // max_depth is fixed (8) but read from the model for provenance.
        let md_s = "8".to_string();
        let ng_s = n_groups.to_string();
        for (m, t) in methods.iter().zip(&timings) {
            eprintln!(
                "    {}: {:.2}\u{00b5}s/obs (blocks={})",
                m.name, t.median_us, t.actual_blocks
            );
            res_csv.write_row(&[
                "scenario_credit",
                "lightgbm",
                arch,
                &k_s,
                &g_s,
                &nt_s,
                &md_s,
                &ng_s,
                &m.name,
                &t.n_obs.to_string(),
                &t.actual_blocks.to_string(),
                &format!("{:.6}", t.median_us),
                &format!("{:.6}", t.p5_us),
                &format!("{:.6}", t.p95_us),
            ]);
        }

        // Work counters: default (precompute) + trace (disable_varying_precompute),
        // summed over all groups — mirrors grid3_stats.
        for (evaluator, ablation) in [
            ("precompute", Ablation::default()),
            (
                "trace",
                Ablation {
                    disable_varying_precompute: true,
                    ..Default::default()
                },
            ),
        ] {
            let mut results = vec![0.0f64; cd.n_rows];
            let total = count_work(&cd, ablation, &mut results);
            stat_csv.write_row(&[
                "scenario_credit",
                "lightgbm",
                arch,
                &k_s,
                &g_s,
                &nt_s,
                &md_s,
                evaluator,
                &ng_s,
                &total.constant_steps.to_string(),
                &total.varying_splits.to_string(),
                &total.unsplit_skips.to_string(),
                &total.recursive_calls.to_string(),
                &total.leaf_hits.to_string(),
                &total.partition_row_evals.to_string(),
                &total.precompute_row_evals.to_string(),
            ]);
            eprintln!(
                "    stats/{evaluator}: C={} V={} leaf={} recurse={} part={} precomp={}",
                total.constant_steps,
                total.varying_splits,
                total.leaf_hits,
                total.recursive_calls,
                total.partition_row_evals,
                total.precompute_row_evals
            );
        }
    }

    eprintln!(
        "\nScen done: {} result rows, {} stat rows",
        res_csv.n_rows(),
        stat_csv.n_rows()
    );
}

/// Inline correctness check for one scenario cell.
///
/// Loads `cell_dir/reference.bin` (write_raw_f64 layout, n_cols=1) and compares
/// TreeWalker's per-row sigmoid output at `TOL_F64 = 1e-14`. Returns false (and
/// logs) on mismatch or missing reference.
fn validate_scen_cell(cd: &CellData, cell_dir: &Path) -> bool {
    const TOL: f64 = 1e-14;
    let ref_path = cell_dir.join("reference.bin");
    if !ref_path.exists() {
        // FAIL-CLOSED gate: a missing GTIL reference is a correctness hazard,
        // not a skip. Aborts the scen grid (the caller exits non-zero) so no
        // timing rows are written without a verified reference.
        eprintln!("  no reference.bin; FAIL-CLOSED gate: missing reference aborts the scen grid");
        return false;
    }
    let (ref_data, ref_n, ref_cols) = match crate::try_load_raw_f64(&ref_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  reference load error: {e}");
            return false;
        }
    };
    if ref_cols != 1 || ref_n != cd.n_rows {
        eprintln!(
            "  reference shape mismatch: cols={ref_cols} n={ref_n} vs n_rows={}",
            cd.n_rows
        );
        return false;
    }
    let mut results = vec![0.0f64; cd.n_rows];
    let mut predictor = cd.forest.predictor();
    let nf = cd.n_cols;
    for &(s, e) in &cd.bounds {
        predictor.predict_group(&cd.data[s * nf..e * nf], &mut results[s..e]);
    }
    let mut max_delta = 0.0f64;
    for r in 0..cd.n_rows {
        let d = (results[r] - ref_data[r]).abs();
        if d > max_delta {
            max_delta = d;
        }
    }
    if max_delta <= TOL {
        eprintln!("  CORRECTNESS PASS max_delta={max_delta:.2e} (tol={TOL:.0e})");
        true
    } else {
        eprintln!("  CORRECTNESS FAIL max_delta={max_delta:.2e} > tol={TOL:.0e}");
        false
    }
}

// ---------------------------------------------------------------------------
// Ablation combo generation
// ---------------------------------------------------------------------------

/// Generate ablation combos as `(Ablation, LoadOptions, label)` tuples.
///
/// `is_full_cross=true`: 2^6 = 64 combos (all runtime × all parse).
/// `is_full_cross=false`: within-group powerset (8 runtime + 7 parse = 15).
fn ablation_combos(is_full_cross: bool) -> Vec<(Ablation, LoadOptions, String)> {
    let mut combos = Vec::new();
    let bits_range = if is_full_cross { 64 } else { 8 };

    // Runtime combos: bits 0-2 = precompute, unsplit, monotonic
    for rt_bits in 0..bits_range {
        let (dp, du, dm, dt, dpg, db) = if is_full_cross {
            (
                rt_bits & 1 != 0,
                rt_bits & 2 != 0,
                rt_bits & 4 != 0,
                rt_bits & 8 != 0,
                rt_bits & 16 != 0,
                rt_bits & 32 != 0,
            )
        } else {
            (
                rt_bits & 1 != 0,
                rt_bits & 2 != 0,
                rt_bits & 4 != 0,
                false,
                false,
                false,
            )
        };
        let am = Ablation {
            disable_varying_precompute: dp,
            disable_unsplit: du,
            disable_monotonic: dm,
            ..Default::default()
        };
        let pc = LoadOptions {
            disable_tree_ordering: dt,
            prefix_depth: if dpg { 0 } else { 2 },
            disable_bitset_intern: db,
            disable_predicate_dedup: false,
        };
        let label = format!(
            "dp={},du={},dm={},dt={},dpg={},db={}",
            u8::from(dp),
            u8::from(du),
            u8::from(dm),
            u8::from(dt),
            u8::from(dpg),
            u8::from(db)
        );
        combos.push((am, pc, label));
    }

    // Within-group: add parse-only combos (runtime flags off).
    if !is_full_cross {
        for pt_bits in 1..8u8 {
            let (dt, dpg, db) = (pt_bits & 1 != 0, pt_bits & 2 != 0, pt_bits & 4 != 0);
            let am = Ablation::default();
            let pc = LoadOptions {
                disable_tree_ordering: dt,
                prefix_depth: if dpg { 0 } else { 2 },
                disable_bitset_intern: db,
                disable_predicate_dedup: false,
            };
            let label = format!(
                "dp=0,du=0,dm=0,dt={},dpg={},db={}",
                u8::from(dt),
                u8::from(dpg),
                u8::from(db)
            );
            combos.push((am, pc, label));
        }
    }
    combos
}

// ---------------------------------------------------------------------------
// Top-level dispatcher
// ---------------------------------------------------------------------------

/// Run benchmark grids and write results to CSV.
pub fn run_grid(config: &RunConfig) {
    match config.grid {
        grid::Grid::All => {
            run_grid1(config);
            run_grid3_ablation(config);
            run_grid4_distributions(config);
        }
        grid::Grid::G1 => run_grid1(config),
        grid::Grid::G3 => run_grid3_ablation(config),
        grid::Grid::G4 => run_grid4_distributions(config),
        grid::Grid::Scen => run_grid_scen(config),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_cells() {
        let artifacts = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../artifacts");
        if !artifacts.exists() {
            return;
        }
        let cells =
            grid::discover_cells(&artifacts, grid::Grid::All, Some(&["expedia".to_string()]));
        if cells.is_empty() {
            eprintln!("No cells found (walker_config.json may be absent locally)");
            return;
        }
        for cell in &cells {
            assert_eq!(cell.dataset, "expedia");
            assert!(cell.is_ctr);
        }
        eprintln!("Discovered {} cells", cells.len());
    }

    #[test]
    fn test_run_single_cell() {
        let artifacts = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../artifacts");
        if !artifacts.exists() {
            return;
        }
        let cells = grid::discover_cells(&artifacts, grid::Grid::All, None);
        let cell = cells
            .iter()
            .find(|c| c.framework == "lightgbm" && c.param_dir.join("test_data.bin").exists());
        let Some(cell) = cell else {
            eprintln!("No testable cell found locally (need test_data.bin)");
            return;
        };
        eprintln!("Testing cell: {}", cell.label());

        let cd = CellData::load(&cell.param_dir, &cell.fw_dir, &LoadOptions::default()).unwrap();
        let config = RunConfig {
            artifacts_dir: artifacts,
            output_dir: PathBuf::from("/tmp"),
            grid: grid::Grid::All,
            datasets_filter: None,
            warmup: 1,
            max_iters: 3,
            min_iters: 3,
            max_time_secs: Some(5.0),
            seed: 42,
            collect_stats: false,
            #[cfg(feature = "external-bench")]
            lgb_lib: None,
            #[cfg(feature = "external-bench")]
            xgb_lib: None,
        };
        let mut methods = collect_methods(&cd, &cell.fw_dir, &cell.framework, &config);
        let bcfg = block_config(&config);
        let timings = bench_blocked(&cd.bounds, &mut methods, &bcfg);

        assert!(
            !timings.is_empty(),
            "should have at least TreeWalker results"
        );
        for (m, t) in methods.iter().zip(&timings) {
            eprintln!("  {}: {:.1}µs/obs", m.name, t.median_us);
            assert!(t.median_us > 0.0);
        }
    }
}
