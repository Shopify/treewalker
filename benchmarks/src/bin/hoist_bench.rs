//! Paired baseline/hoisted measurement on an existing benchmark cell.
//! Both forests are checked against the original full walk before timing.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use treewalker_bench::timing::{BenchMethod, BlockConfig, bench_blocked};
use treewalker_gbdt::forest::Forest;
use treewalker_gbdt::{HoistStats, ParseConfig, PredictStats};

#[derive(Serialize)]
#[serde(remote = "HoistStats")]
struct HoistReport {
    trees_examined: usize,
    trees_changed: usize,
    swaps: usize,
    collapsed_splits: usize,
    nodes_before: usize,
    nodes_after: usize,
    varying_roots_before: usize,
    varying_roots_after: usize,
    pass_limit_hits: usize,
}

#[derive(Serialize)]
#[serde(remote = "PredictStats")]
struct WorkReport {
    constant_steps: u64,
    varying_splits: u64,
    unsplit_skips: u64,
    recursive_calls: u64,
    leaf_hits: u64,
    partition_row_evals: u64,
    precompute_row_evals: u64,
}

#[derive(Serialize)]
struct ModeReport {
    mode: &'static str,
    load_us: f64,
    nodes: usize,
    /// Compact node, bitset and tree pools; excludes workspace/metadata.
    model_pool_bytes: usize,
    #[serde(with = "WorkReport")]
    work: PredictStats,
    median_us: f64,
    p5_us: f64,
    p95_us: f64,
    blocks: usize,
}

#[derive(Serialize)]
struct Report {
    model: PathBuf,
    config: PathBuf,
    data: PathBuf,
    groups: usize,
    rows: usize,
    trees: usize,
    prefix_depth: usize,
    tree_ordering: bool,
    max_prediction_delta: f64,
    #[serde(with = "HoistReport")]
    hoisting: HoistStats,
    speedup: f64,
    modes: Vec<ModeReport>,
}

fn mode_report(mode: &'static str, forest: &Forest, load_us: f64) -> ModeReport {
    ModeReport {
        mode,
        load_us,
        nodes: forest.nodes().len(),
        model_pool_bytes: std::mem::size_of_val(forest.nodes())
            + forest.bitset_bytes()
            + std::mem::size_of_val(forest.trees()),
        work: PredictStats::default(),
        median_us: 0.0,
        p5_us: 0.0,
        p95_us: 0.0,
        blocks: 0,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || args.iter().any(|a| a == "--help") {
        eprintln!(
            "Usage: hoist_bench MODEL CONFIG DATA [--group-offsets FILE] \
            [--min-blocks N] [--max-blocks N] [--warmup N] \
            [--no-tree-ordering] [--prefix-depth N]"
        );
        if args.iter().any(|a| a == "--help") {
            return;
        }
        std::process::exit(2);
    }
    let model = PathBuf::from(&args[1]);
    let config = PathBuf::from(&args[2]);
    let data_path = PathBuf::from(&args[3]);
    let mut offsets = None;
    let mut timing = BlockConfig::default();
    let mut parse = ParseConfig::default();
    let mut i = 4;
    while i < args.len() {
        if args[i] == "--no-tree-ordering" {
            parse.disable_tree_ordering = true;
            i += 1;
            continue;
        }
        let value = args.get(i + 1).expect("missing option value");
        match args[i].as_str() {
            "--group-offsets" => {
                offsets = Some(treewalker_bench::load_group_offsets(Path::new(value)));
            }
            "--min-blocks" => timing.min_blocks = value.parse().expect("invalid min-blocks"),
            "--max-blocks" => timing.max_blocks = value.parse().expect("invalid max-blocks"),
            "--warmup" => timing.warmup = value.parse().expect("invalid warmup"),
            "--prefix-depth" => parse.prefix_depth = value.parse().expect("invalid prefix-depth"),
            flag => panic!("unknown option: {flag}"),
        }
        i += 2;
    }
    assert!(timing.min_blocks > 0 && timing.max_blocks >= timing.min_blocks);
    let start = Instant::now();
    let mut baseline = Forest::load_with_config(&model, &config, &parse);
    let baseline_us = start.elapsed().as_secs_f64() * 1e6;
    let start = Instant::now();
    let mut hoisted = Forest::load_with_config(
        &model,
        &config,
        &ParseConfig {
            hoist_constants: true,
            ..parse
        },
    );
    let hoisted_us = start.elapsed().as_secs_f64() * 1e6;
    let (data, rows, cols) = treewalker_bench::load_raw_f64(&data_path);
    assert_eq!(cols, baseline.config.n_features, "feature count mismatch");
    // The engine currently has 64 feature columns, regardless of mask width.
    assert!(cols <= 64);
    let width = baseline.config.max_group_width;
    assert!((1..=128).contains(&width));
    let offsets = offsets.unwrap_or_else(|| {
        assert_eq!(rows % width, 0, "use --group-offsets for variable groups");
        (0..=rows).step_by(width).collect()
    });
    assert_eq!(offsets.first(), Some(&0));
    assert_eq!(offsets.last(), Some(&rows));
    let groups: Vec<_> = offsets
        .windows(2)
        .map(|p| {
            assert!(p[0] < p[1] && p[1] - p[0] <= width);
            (p[0], p[1])
        })
        .collect();
    assert!(!groups.is_empty(), "empty dataset");

    let mut reports = vec![
        mode_report("baseline", &baseline, baseline_us),
        mode_report("hoisted", &hoisted, hoisted_us),
    ];
    let mut reference = vec![0.0; rows];
    let mut original = vec![0.0; rows];
    let mut transformed = vec![0.0; rows];
    baseline.predict_full(&data, &mut reference, 0, rows);
    let mut max_delta: f64 = 0.0;
    for &(s, e) in &groups {
        reports[0].work += baseline.predict_with_stats(&data, &mut original, s, e);
        reports[1].work += hoisted.predict_with_stats(&data, &mut transformed, s, e);
    }
    for output in [&original, &transformed] {
        for (row, (&actual, &expected)) in output.iter().zip(&reference).enumerate() {
            let delta = (actual - expected).abs();
            assert!(
                actual.is_finite() && expected.is_finite() && delta <= 1e-14,
                "prediction mismatch at row {row}: {actual} vs {expected}"
            );
            max_delta = max_delta.max(delta);
        }
    }
    let hoisting = *hoisted.hoist_stats();
    eprintln!("Correctness passed: max_delta={max_delta:.3e}; {hoisting:?}");
    let trees = baseline.trees().len();
    let mut methods = [
        BenchMethod {
            name: "baseline".into(),
            predict_group: Box::new(|s, e| {
                baseline.predict(&data, &mut original, s, e);
                black_box(&original[s..e]);
            }),
        },
        BenchMethod {
            name: "hoisted".into(),
            predict_group: Box::new(|s, e| {
                hoisted.predict(&data, &mut transformed, s, e);
                black_box(&transformed[s..e]);
            }),
        },
    ];
    let timings = bench_blocked(&groups, &mut methods, &timing);
    for (report, result) in reports.iter_mut().zip(&timings) {
        report.median_us = result.median_us;
        report.p5_us = result.p5_us;
        report.p95_us = result.p95_us;
        report.blocks = result.actual_blocks;
    }
    let report = Report {
        model,
        config,
        data: data_path,
        groups: groups.len(),
        rows,
        trees,
        prefix_depth: parse.prefix_depth,
        tree_ordering: !parse.disable_tree_ordering,
        max_prediction_delta: max_delta,
        hoisting,
        speedup: reports[0].median_us / reports[1].median_us,
        modes: reports,
    };
    println!("{}", simd_json::serde::to_string_pretty(&report).unwrap());
}
