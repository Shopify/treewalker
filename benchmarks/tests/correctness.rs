use std::path::{Path, PathBuf};

use treewalker_gbdt::AblationMode;
use treewalker_gbdt::forest::{Forest, ThresholdType};
use treewalker_gbdt::config::WalkerConfig;

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// F64 models (LightGBM): GTIL reference is exact. Only FP accumulation noise.
const TOL_F64: f64 = 1e-14;

/// F32 models (XGBoost): f32 leaf accumulation noise between TW (f64 sum) and
/// native XGBoost (f32 sum). Split decisions are identical (f32 comparison).
const TOL_F32: f64 = 1e-5;

/// Partial eval vs full walk: must be bitwise identical (same code path, same precision).
const TOL_EXACT: f64 = 1e-15;

/// Current importer bounds JSON memory; larger models use streaming binary.
const SIMD_JSON_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// Treelite model for a config: the binary export when present (the benchmark
/// harness prefers it), else JSON. The largest grid model's JSON export
/// (Expedia, T=2000, L=16, LightGBM) exceeds the simd-json size limit.
fn model_file(param_dir: &Path, framework: &str) -> PathBuf {
    let bin = param_dir.join(format!("{framework}/model_treelite.bin"));
    if bin.exists() { bin } else { param_dir.join(format!("{framework}/model_treelite.json")) }
}

// ---------------------------------------------------------------------------
// Config discovery — test every artifact
// ---------------------------------------------------------------------------

const ARTIFACTS_BASE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../paper/experiments/artifacts");
const FRAMEWORKS: &[&str] = &["lightgbm", "xgboost"];

struct TestConfig {
    label: String,
    param_dir: PathBuf,
    framework: &'static str,
    tol: f64,
    /// Group offsets for variable-length entities. None = fixed panel_length stride.
    group_offsets: Option<Vec<usize>>,
}

/// Discover all (dataset, params, framework) combos that have artifacts.
fn all_configs() -> Vec<TestConfig> {
    let base = std::env::var_os("TEST_ARTIFACTS_BASE")
        .map_or_else(|| PathBuf::from(ARTIFACTS_BASE), PathBuf::from);
    let mut configs = Vec::new();

    let mut datasets: Vec<_> = std::fs::read_dir(&base)
        .expect("artifacts dir missing — run prepare.py")
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|ft| ft.is_dir()))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    datasets.sort();

    for dataset in &datasets {
        let dataset_dir = base.join(dataset);
        let mut params: Vec<_> = std::fs::read_dir(&dataset_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_ok_and(|ft| ft.is_dir()))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        params.sort();

        for param in &params {
            let param_dir = dataset_dir.join(param);
            for &fw in FRAMEWORKS {
                let model = model_file(&param_dir, fw);
                let preds = param_dir.join(format!("{fw}/predictions.npy"));
                if !model.exists() || !preds.exists() {
                    continue;
                }
                let forest = Forest::load(&model, param_dir.join("walker_config.json"));
                let tol = match forest.threshold_type() {
                    ThresholdType::F64 => TOL_F64,
                    ThresholdType::F32 => TOL_F32,
                };
                let group_offsets = load_group_offsets_if_present(&param_dir);
                configs.push(TestConfig {
                    label: format!("{dataset}/{param}/{fw}"),
                    param_dir: param_dir.clone(),
                    framework: fw,
                    tol,
                    group_offsets,
                });
            }
        }
    }

    assert!(!configs.is_empty(), "No artifact configs found. Run prepare.py first.");
    configs
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_forest(param_dir: &Path, framework: &str) -> Forest {
    Forest::load(model_file(param_dir, framework), param_dir.join("walker_config.json"))
}

fn load_forest_at_width(param_dir: &Path, framework: &str, width: usize) -> Forest {
    let mut config = WalkerConfig::from_file(param_dir.join("walker_config.json"));
    config.max_group_width = width;
    let path = model_file(param_dir, framework);
    let format = if path.extension().is_some_and(|e| e == "bin") {
        treewalker_gbdt::ModelFormat::TreeliteBinaryV4
    } else {
        treewalker_gbdt::ModelFormat::TreeliteJson
    };
    Forest::from_reader(
        std::fs::File::open(path).unwrap(), format, config,
        &treewalker_gbdt::ParseConfig::default(),
    ).unwrap()
}

fn load_test_data(param_dir: &Path) -> (Vec<f64>, usize) {
    let (data, n_rows, _) = treewalker_bench::load_raw_f64(param_dir.join("test_data.bin"));
    (data, n_rows)
}

/// Load group_offsets.bin if it exists. Format: u64 n_groups, then (n_groups+1) u64 LE offsets.
fn load_group_offsets_if_present(param_dir: &Path) -> Option<Vec<usize>> {
    let path = param_dir.join("group_offsets.bin");
    if !path.exists() {
        return None;
    }
    let bytes = std::fs::read(&path).expect("failed to read group_offsets.bin");
    let n_groups = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let offsets: Vec<usize> = bytes[8..]
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    assert_eq!(offsets.len(), n_groups + 1);
    Some(offsets)
}

fn load_reference(param_dir: &Path, framework: &str) -> Vec<f64> {
    let arr: ndarray::Array1<f64> =
        ndarray_npy::read_npy(param_dir.join(format!("{framework}/predictions.npy"))).unwrap();
    arr.to_vec()
}

/// Iterate groups: fixed panel_length stride or variable offsets.
fn for_each_group(
    group_width: usize, n_rows: usize, offsets: Option<&[usize]>,
    mut f: impl FnMut(usize, usize),
) {
    if let Some(offs) = offsets {
        for pair in offs.windows(2) {
            f(pair[0], pair[1]);
        }
    } else {
        for s in 0..(n_rows / group_width) {
            f(s * group_width, (s + 1) * group_width);
        }
    }
}

fn predict_all(forest: &mut Forest, data: &[f64], n_rows: usize, offsets: Option<&[usize]>) -> Vec<f64> {
    let mut results = vec![0.0f64; n_rows];
    let h = forest.config.max_group_width;
    if let Some(offs) = offsets {
        for pair in offs.windows(2) {
            forest.predict(data, &mut results, pair[0], pair[1]);
        }
    } else {
        for s in 0..(n_rows / h) {
            forest.predict(data, &mut results, s * h, (s + 1) * h);
        }
    }
    results
}

fn predict_full_all(forest: &Forest, data: &[f64], n_rows: usize, offsets: Option<&[usize]>) -> Vec<f64> {
    let mut results = vec![0.0f64; n_rows];
    let h = forest.config.max_group_width;
    for_each_group(h, n_rows, offsets, |start, end| {
        forest.predict_full(data, &mut results, start, end);
    });
    results
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

// ---------------------------------------------------------------------------
// Core correctness — runs on EVERY artifact
// ---------------------------------------------------------------------------

#[test]
fn test_reference_match() {
    for cfg in &all_configs() {
        let mut forest = load_forest(&cfg.param_dir, cfg.framework);
        let (data, n_rows) = load_test_data(&cfg.param_dir);
        let reference = load_reference(&cfg.param_dir, cfg.framework);

        let partial = predict_all(&mut forest, &data, n_rows, cfg.group_offsets.as_deref());
        let full = predict_full_all(&forest, &data, n_rows, cfg.group_offsets.as_deref());

        let d_partial = max_diff(&partial, &reference);
        let d_full = max_diff(&full, &reference);
        eprintln!("{}: partial={d_partial:.2e} full={d_full:.2e}", cfg.label);

        assert!(d_partial < cfg.tol,
            "{} partial vs ref: {d_partial:.2e} exceeds {:.0e}", cfg.label, cfg.tol);
        assert!(d_full < cfg.tol,
            "{} full vs ref: {d_full:.2e} exceeds {:.0e}", cfg.label, cfg.tol);
    }
}

#[test]
fn test_partial_matches_full() {
    for cfg in &all_configs() {
        let mut forest = load_forest(&cfg.param_dir, cfg.framework);
        let (data, n_rows) = load_test_data(&cfg.param_dir);

        let partial = predict_all(&mut forest, &data, n_rows, cfg.group_offsets.as_deref());
        let full = predict_full_all(&forest, &data, n_rows, cfg.group_offsets.as_deref());

        let d = max_diff(&partial, &full);
        eprintln!("{}: partial_vs_full={d:.2e}", cfg.label);
        assert!(d < TOL_EXACT, "{} partial vs full: {d:.2e}", cfg.label);
    }
}

#[test]
fn test_single_row() {
    for cfg in &all_configs() {
        let forest = load_forest(&cfg.param_dir, cfg.framework);
        let (data, _) = load_test_data(&cfg.param_dir);
        let reference = load_reference(&cfg.param_dir, cfg.framework);
        let nf = forest.config.n_features;

        let mut single = vec![0.0f64; 1];
        forest.predict_full(&data[0..nf], &mut single, 0, 1);

        let d = (single[0] - reference[0]).abs();
        assert!(d < cfg.tol, "{} single row diff: {d:.2e}", cfg.label);
    }
}

// --- Edge case tests (first config only for speed) ---

fn first_config() -> TestConfig {
    // Edge-case tests use fixed panel_length stride, so skip variable-group
    // configs, and need more than one row per group (G=1 has no varying splits).
    all_configs()
        .into_iter()
        .find(|c| {
            c.group_offsets.is_none()
                && WalkerConfig::from_file(c.param_dir.join("walker_config.json")).max_group_width > 1
        })
        .unwrap()
}

fn assert_partial_matches_full_on_obs(forest: &mut Forest, data: &[f64], obs: usize) {
    let h = forest.config.max_group_width;
    let start = obs * h;
    let end = start + h;

    let mut full = vec![0.0f64; end];
    let mut partial = vec![0.0f64; end];
    forest.predict_full(data, &mut full, start, end);
    forest.predict(data, &mut partial, start, end);

    let d = max_diff(&full[start..end], &partial[start..end]);
    assert!(d < TOL_EXACT, "Partial vs full max_diff={d:.2e} on observation {obs}");
}

#[test]
fn test_all_nan_tv_features() {
    let cfg = first_config();
    let mut forest = load_forest(&cfg.param_dir, cfg.framework);
    let (mut data, _) = load_test_data(&cfg.param_dir);
    let nf = forest.config.n_features;
    let h = forest.config.max_group_width;

    for r in 0..h {
        for f in 0..nf {
            if forest.config.varying_mask & (1 << f) != 0 {
                data[r * nf + f] = f64::NAN;
            }
        }
    }
    assert_partial_matches_full_on_obs(&mut forest, &data, 0);
}

#[test]
fn test_monotonic_columns_with_ties() {
    let cfg = first_config();
    let mut forest = load_forest(&cfg.param_dir, cfg.framework);
    let (mut data, _) = load_test_data(&cfg.param_dir);
    let nf = forest.config.n_features;
    let h = forest.config.max_group_width;

    for r in 0..h {
        for f in 0..nf {
            if forest.config.is_mono_inc(f) || forest.config.is_mono_dec(f) {
                data[r * nf + f] = 5.0;
            }
        }
    }
    assert_partial_matches_full_on_obs(&mut forest, &data, 0);
}

#[test]
fn test_all_nan_monotonic_features() {
    let cfg = first_config();
    let mut forest = load_forest(&cfg.param_dir, cfg.framework);
    let (mut data, _) = load_test_data(&cfg.param_dir);
    let nf = forest.config.n_features;
    let h = forest.config.max_group_width;

    for r in 0..h {
        for f in 0..nf {
            if forest.config.is_mono_inc(f) || forest.config.is_mono_dec(f) {
                data[r * nf + f] = f64::NAN;
            }
        }
    }
    assert_partial_matches_full_on_obs(&mut forest, &data, 0);
}

#[test]
fn test_extreme_feature_values() {
    let cfg = first_config();
    let mut forest = load_forest(&cfg.param_dir, cfg.framework);
    let (mut data, _) = load_test_data(&cfg.param_dir);
    let nf = forest.config.n_features;
    let h = forest.config.max_group_width;
    let tv_mask = forest.config.varying_mask;

    for &val in &[f64::MIN, f64::MAX, 0.0, -0.0, 1e300, -1e300] {
        for r in 0..h {
            for f in 0..nf {
                if tv_mask & (1 << f) == 0 {
                    data[r * nf + f] = val;
                }
            }
        }
        assert_partial_matches_full_on_obs(&mut forest, &data, 0);
    }
}

// --- Ablation correctness (all configs × all modes) ---

#[test]
fn test_ablation_all_configs() {
    let ablations: &[(&str, AblationMode)] = &[
        ("no_unsplit", AblationMode { disable_unsplit: true, ..Default::default() }),
        ("no_varying_precompute", AblationMode { disable_varying_precompute: true, ..Default::default() }),
        // Monotonic only has effect when precompute is also disabled.
        ("no_monotonic", AblationMode { disable_monotonic: true, disable_varying_precompute: true, ..Default::default() }),
        ("no_mono_no_unsplit", AblationMode { disable_monotonic: true, disable_unsplit: true, disable_varying_precompute: true, ..Default::default() }),
        ("all_disabled", AblationMode { disable_monotonic: true, disable_unsplit: true, disable_varying_precompute: true, disable_predicate_sweep: true }),
    ];

    for cfg in &all_configs() {
        for &(mode, ablation) in ablations {
            let mut forest = load_forest(&cfg.param_dir, cfg.framework);
            forest.config.ablation = ablation;
            let (data, n_rows) = load_test_data(&cfg.param_dir);

            let partial = predict_all(&mut forest, &data, n_rows, cfg.group_offsets.as_deref());
            let full = predict_full_all(&forest, &data, n_rows, cfg.group_offsets.as_deref());

            let d = max_diff(&partial, &full);
            assert!(d < TOL_EXACT, "{}/{mode} max_diff={d:.2e}", cfg.label);
        }
        eprintln!("{}: all ablations ok", cfg.label);
    }
}

// --- Parse config correctness ---

#[test]
fn test_parse_configs() {
    use treewalker_gbdt::ParseConfig;

    let parse_configs: &[(&str, ParseConfig)] = &[
        ("no_tree_ordering", ParseConfig { disable_tree_ordering: true, ..Default::default() }),
        ("no_bitset_intern", ParseConfig { disable_bitset_intern: true, ..Default::default() }),
        ("no_prefix_grouping", ParseConfig { prefix_depth: 0, ..Default::default() }),
        ("all_parse_disabled", ParseConfig {
            disable_tree_ordering: true,
            disable_bitset_intern: true,
            prefix_depth: 0,
            ..Default::default()
        }),
        // disable_predicate_dedup tested separately — large models exceed u16 limit.
    ];

    for cfg in &all_configs() {
        for &(mode, ref pc) in parse_configs {
            let mut forest = Forest::load_with_config(
                model_file(&cfg.param_dir, cfg.framework),
                cfg.param_dir.join("walker_config.json"),
                pc,
            );
            let (data, n_rows) = load_test_data(&cfg.param_dir);
            let partial = predict_all(&mut forest, &data, n_rows, cfg.group_offsets.as_deref());
            let full = predict_full_all(&forest, &data, n_rows, cfg.group_offsets.as_deref());
            let d = max_diff(&partial, &full);
            assert!(d < TOL_EXACT, "{}/{mode} max_diff={d:.2e}", cfg.label);
        }
        eprintln!("{}: all parse configs ok", cfg.label);
    }
}

// --- Binary format correctness ---

#[test]
fn test_binary_matches_json() {
    let mut tested = 0;
    for cfg in &all_configs() {
        let bin_path = cfg.param_dir.join(format!("{}/model_treelite.bin", cfg.framework));
        let json_path = cfg.param_dir.join(format!("{}/model_treelite.json", cfg.framework));
        if !bin_path.exists() || !json_path.exists() {
            continue;
        }
        if std::fs::metadata(&json_path).unwrap().len() > SIMD_JSON_MAX_BYTES {
            eprintln!("{}: JSON export exceeds the simd-json size limit; skipped", cfg.label);
            continue;
        }

        let mut forest_json = Forest::load(&json_path, cfg.param_dir.join("walker_config.json"));
        let mut forest_bin = Forest::load(
            &bin_path,
            cfg.param_dir.join("walker_config.json"),
        );

        assert_eq!(
            forest_json.trees().len(),
            forest_bin.trees().len(),
            "{}: tree count mismatch",
            cfg.label,
        );

        let (data, n_rows) = load_test_data(&cfg.param_dir);
        let pred_json = predict_all(&mut forest_json, &data, n_rows, cfg.group_offsets.as_deref());
        let pred_bin = predict_all(&mut forest_bin, &data, n_rows, cfg.group_offsets.as_deref());

        let d = max_diff(&pred_json, &pred_bin);
        assert!(
            d < TOL_F64,
            "{}: binary vs json max_diff={d:.2e}",
            cfg.label,
        );
        tested += 1;
    }
    eprintln!("test_binary_matches_json: {tested} configs tested");
    assert!(tested > 0, "No .bin files found — run prepare.py or generate test binaries");
}

// --- Out-of-range categorical values ---

/// Verify that out-of-range category values (beyond bitset width) produce
/// identical results for partial eval and full walk. This exercises the fix
/// for inverted bitsets (XGBoost `category_list_right_child=true`): both
/// `cat_test` and `VaryingPredicate::goes_left` must return `default_left`
/// for out-of-range categories, not unconditional `false`.
#[test]
fn test_cat_out_of_range_partial_matches_full() {
    // Find an XGBoost config with categorical splits.
    let cfg = all_configs()
        .into_iter()
        .find(|c| {
            c.framework == "xgboost" && c.group_offsets.is_none() && {
                let forest = load_forest(&c.param_dir, c.framework);
                forest.nodes().iter().any(|n| n.is_categorical() && !n.is_leaf())
            }
        });
    let Some(cfg) = cfg else {
        eprintln!("No XGBoost config with categorical splits found, skipping");
        return;
    };

    let mut forest = load_forest(&cfg.param_dir, cfg.framework);
    let (mut data, _n_rows) = load_test_data(&cfg.param_dir);
    let nf = forest.config.n_features;
    let h = forest.config.max_group_width;

    // Find a categorical feature by scanning nodes.
    let cat_feature = forest.nodes().iter()
        .find(|n| n.is_categorical() && !n.is_leaf())
        .map(|n| n.feature as usize)
        .unwrap();

    // Set the categorical feature to an out-of-range value (50) for all rows
    // in the first observation. This value exceeds the inline bitset range (0..31)
    // and would trigger the bug if out-of-range returned false instead of default_left.
    for r in 0..h {
        data[r * nf + cat_feature] = 50.0;
    }

    let start = 0;
    let end = h;
    let mut full_results = vec![0.0f64; end];
    let mut partial_results = vec![0.0f64; end];
    forest.predict_full(&data, &mut full_results, start, end);
    forest.predict(&data, &mut partial_results, start, end);

    let d = max_diff(&full_results[start..end], &partial_results[start..end]);
    eprintln!("{}: cat_out_of_range partial_vs_full={d:.2e}", cfg.label);
    assert!(d < TOL_EXACT,
        "{}: cat out-of-range partial vs full max_diff={d:.2e}", cfg.label);
}

// --- Width-32 boundary test ---

/// Verify that partial eval works correctly with 32-row groups (the u32 mask boundary).
/// This exercises `u32::MAX >> (32 - 32) == u32::MAX` — the exact analog of the u16
/// overflow bug at width=16 (where `(1u16 << 16) - 1` wrapped to 0).
#[test]
fn test_width_32_boundary() {
    // Load a real model with max_group_width set to the u32 boundary.
    let base_cfg = first_config();
    let mut forest = load_forest_at_width(&base_cfg.param_dir, base_cfg.framework, 32);
    let nf = forest.config.n_features;

    // Build synthetic test data: 32 rows × n_features.
    // Use the first observation's constant features, vary the TV features linearly.
    let (real_data, _) = load_test_data(&base_cfg.param_dir);
    let mut data = vec![0.0f64; 32 * nf];
    for row in 0..32 {
        for f in 0..nf {
            if forest.config.is_varying(f) {
                // Linearly spaced values for varying features
                if forest.config.is_mono_inc(f) {
                    data[row * nf + f] = row as f64;
                } else if forest.config.is_mono_dec(f) {
                    data[row * nf + f] = 31.0 - row as f64;
                } else {
                    data[row * nf + f] = (row as f64) * 0.1;
                }
            } else {
                // Copy constant features from the real data
                data[row * nf + f] = real_data[f];
            }
        }
    }

    // Predict: full walk vs partial eval on the 32-row group.
    let mut full_results = vec![0.0f64; 32];
    let mut partial_results = vec![0.0f64; 32];
    forest.predict_full(&data, &mut full_results, 0, 32);
    forest.predict(&data, &mut partial_results, 0, 32);

    let d = max_diff(&full_results, &partial_results);
    eprintln!(
        "width_32_boundary: partial_vs_full max_diff={d:.2e} ({} trees, {} features)",
        forest.trees().len(), nf,
    );
    assert!(d < TOL_EXACT,
        "Width-32 boundary: partial vs full max_diff={d:.2e} — u32 mask arithmetic may be wrong");

    // Also verify that predict (with internal workspace) matches.
    let mut ws_results = vec![0.0f64; 32];
    forest.predict(&data, &mut ws_results, 0, 32);
    let d2 = max_diff(&full_results, &ws_results);
    assert!(d2 < TOL_EXACT,
        "Width-32 boundary (workspace): max_diff={d2:.2e}");
}

// --- Stats (first config) ---

#[test]
fn test_node_visit_stats() {
    use treewalker_gbdt::PredictStats;

    let cfg = first_config();
    let mut forest = load_forest(&cfg.param_dir, cfg.framework);
    let (data, n_rows) = load_test_data(&cfg.param_dir);
    let h = forest.config.max_group_width;
    let n_obs = n_rows / h;

    let mut results = vec![0.0f64; n_rows];
    let mut total = PredictStats::default();
    for s in 0..n_obs {
        total += forest.predict_with_stats(&data, &mut results, s * h, (s + 1) * h);
    }

    assert!(total.constant_steps > 0);
    assert!(total.varying_splits > 0);
    assert!(total.leaf_hits > 0);
    assert!(total.precompute_row_evals > 0);
}

// ---------------------------------------------------------------------------
// Differential test: sweep precompute vs brute-force
// ---------------------------------------------------------------------------

/// Verify that the sorted-threshold sweep produces bit-exact masks compared to
/// the brute-force O(P × n) precompute for every observation in every config.
#[test]
fn test_sweep_matches_bruteforce() {
    for cfg in &all_configs() {
        let forest = load_forest(&cfg.param_dir, cfg.framework);
        let (data, n_rows) = load_test_data(&cfg.param_dir);
        let nf = forest.config.n_features;
        let f32_mode = forest.threshold_type() == ThresholdType::F32;

        let mut n_obs_tested = 0usize;
        let mut n_obs_skipped_wide = 0usize;
        let gw = forest.config.max_group_width;
        // precompute_sweep/precompute_bruteforce are width-32 test helpers: they
        // build u32 row masks and store columns in [[f64; 32]; 64]. Groups wider
        // than 32 rows (e.g. E2's 128-row chunks) use the u64/u128 mask path,
        // which these helpers do not exercise. Skip them here; the wider mask
        // paths are validated by test_reference_match / test_partial_matches_full
        // / test_wide_group_{48,64,96,128}. Validating only n <= 32 avoids an
        // out-of-bounds index into the width-32 column buffer.
        const SWEEP_HELPER_WIDTH: usize = 32;
        for_each_group(gw, n_rows, cfg.group_offsets.as_deref(), |start, end| {
            let n = end - start;
            if n > SWEEP_HELPER_WIDTH {
                n_obs_skipped_wide += 1;
                return;
            }

            // Build varying_cols the same way predict_typed_ablation does.
            let mut varying_cols = Box::new([[0.0f64; 32]; 64]);
            let mut m = forest.config.varying_mask;
            while m != 0 {
                let f = m.trailing_zeros() as usize;
                for r in 0..n {
                    varying_cols[f][r] = data[(start + r) * nf + f];
                }
                m &= m - 1;
            }

            let sweep = forest.precompute_sweep(&varying_cols, n, f32_mode);
            let brute = forest.precompute_bruteforce(&varying_cols, n, f32_mode);

            assert_eq!(
                sweep.len(), brute.len(),
                "{}: mask vec lengths differ", cfg.label
            );
            for (pred_id, (s, b)) in sweep.iter().zip(brute.iter()).enumerate() {
                assert_eq!(
                    s, b,
                    "{}: obs {start}..{end}, pred_id {pred_id}: \
                     sweep=0b{s:032b} brute=0b{b:032b}",
                    cfg.label
                );
            }
            n_obs_tested += 1;
        });
        eprintln!(
            "{}: sweep matches bruteforce on {n_obs_tested} observations{}",
            cfg.label,
            if n_obs_skipped_wide > 0 {
                format!(" ({n_obs_skipped_wide} skipped: groups > {SWEEP_HELPER_WIDTH} rows use u64/u128 masks, not the width-32 helpers)")
            } else {
                String::new()
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Synthetic wide-group tests — u64 and u128 mask paths
// ---------------------------------------------------------------------------

/// Build a synthetic wide group by taking an existing model's test data and
/// constructing groups wider than 32 rows. Each synthetic group copies the
/// constant features from the first row and fills varying features with
/// values sampled from the original data.
///
/// The test verifies that `predict` (partial eval, using u64/u128 masks)
/// produces the same predictions as `predict_full` (full walk, no masks).
fn test_wide_group_partial_matches_full(target_width: usize) {
    // Use first available config.
    let first_cfg = all_configs().into_iter().next()
        .expect("No configs found — run prepare.py");

    let mut forest = load_forest_at_width(&first_cfg.param_dir, first_cfg.framework, target_width);
    let (orig_data, orig_n_rows) = load_test_data(&first_cfg.param_dir);
    let nf = forest.config.n_features;
    let orig_gw = WalkerConfig::from_file(first_cfg.param_dir.join("walker_config.json"))
        .max_group_width;

    // Build synthetic groups: take the first 10 original groups' constant
    // features, expand each to `target_width` rows by cycling varying features
    // from the original data.
    let n_synthetic_groups = 10;
    let mut synth_data = Vec::new();

    for g in 0..n_synthetic_groups {
        // Source group's first row (for constant features).
        let src_start = if let Some(offs) = &first_cfg.group_offsets {
            if g >= offs.len() - 1 { break; }
            offs[g]
        } else {
            g * orig_gw
        };
        if (src_start + 1) * nf > orig_data.len() { break; }
        let const_row = &orig_data[src_start * nf..(src_start + 1) * nf];

        // Build `target_width` rows. Row 0 gets the original constant features.
        // All rows share constant features; varying features are cycled from
        // original data rows.
        for r in 0..target_width {
            let mut row = const_row.to_vec();
            // Fill varying features from a cycling source row.
            let src_row_idx = (src_start + r) % orig_n_rows;
            let src_row = &orig_data[src_row_idx * nf..(src_row_idx + 1) * nf];
            let mut vm = forest.config.varying_mask;
            while vm != 0 {
                let f = vm.trailing_zeros() as usize;
                row[f] = src_row[f];
                vm &= vm - 1;
            }
            synth_data.extend_from_slice(&row);
        }
    }

    let total_rows = n_synthetic_groups * target_width;
    assert_eq!(synth_data.len(), total_rows * nf);

    // Predict with partial eval (exercises u64/u128 mask path).
    let mut results_partial = vec![0.0f64; total_rows];
    for g in 0..n_synthetic_groups {
        let start = g * target_width;
        let end = start + target_width;
        forest.predict(&synth_data, &mut results_partial, start, end);
    }

    // Predict with full walk (no masks, always correct).
    let mut results_full = vec![0.0f64; total_rows];
    for g in 0..n_synthetic_groups {
        let start = g * target_width;
        let end = start + target_width;
        forest.predict_full(&synth_data, &mut results_full, start, end);
    }

    let diff = max_diff(&results_partial, &results_full);
    eprintln!(
        "wide_group(width={target_width}): partial vs full max_diff={diff:.2e} ({n_synthetic_groups} groups, {} framework)",
        first_cfg.framework,
    );
    assert!(
        diff < TOL_EXACT,
        "wide_group(width={target_width}): partial eval diverges from full walk: max_diff={diff:.2e}"
    );
}

#[test]
fn test_wide_group_48_u64() {
    test_wide_group_partial_matches_full(48);
}

#[test]
fn test_wide_group_64_u64() {
    test_wide_group_partial_matches_full(64);
}

#[test]
fn test_wide_group_96_u128() {
    test_wide_group_partial_matches_full(96);
}

#[test]
fn test_wide_group_128_u128() {
    test_wide_group_partial_matches_full(128);
}
