use std::path::{Path, PathBuf};

use treewalker_gbdt::research::{Ablation, ThresholdType};
use treewalker_gbdt::{Forest, LoadError, WalkerConfig};

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// F64 models (LightGBM). GTIL adds leaves in f64 in tree order, so its rounding
/// grows with the partial sums: on the 1,008-step FLCHAIN panel, where margins reach
/// ±500, it is up to 1.1e-14 from the correctly rounded sum that `predict` returns.
const TOL_F64: f64 = 1e-13;

/// F32 models (XGBoost): f32 leaf accumulation noise between TW (f64 sum) and
/// native XGBoost (f32 sum). Split decisions are identical (f32 comparison).
const TOL_F32: f64 = 1e-5;

/// Partial eval vs full walk. `predict` returns the correctly rounded sum of the leaf
/// values; the full walk adds them in f64 in tree order, so the two differ by the full
/// walk's rounding: the same accumulation noise as the GTIL comparison.
const TOL_FULL_WALK: f64 = TOL_F64;

/// Current importer bounds JSON memory; larger models use streaming binary.
const SIMD_JSON_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// Treelite model for a config: the binary export when present (the benchmark
/// harness prefers it), else JSON. The largest grid model's JSON export
/// (Expedia, T=2000, L=16, LightGBM) exceeds the simd-json size limit.
fn model_file(param_dir: &Path, framework: &str) -> PathBuf {
    let bin = param_dir.join(format!("{framework}/model_treelite.bin"));
    if bin.exists() {
        bin
    } else {
        param_dir.join(format!("{framework}/model_treelite.json"))
    }
}

/// Number of `"threshold": ,` entries in a Treelite JSON export, found without the
/// parser under test. Treelite writes an empty value for nonfinite thresholds, and
/// the loader rejects such dumps rather than repairing them (docs/treelite-loading.md).
fn empty_thresholds(path: &Path) -> usize {
    const KEY: &[u8] = b"\"threshold\":";
    let bytes = std::fs::read(path).unwrap();
    (0..bytes.len().saturating_sub(KEY.len()))
        .filter(|&i| bytes[i..].starts_with(KEY))
        .filter(|&i| {
            bytes[i + KEY.len()..]
                .iter()
                .find(|b| !b.is_ascii_whitespace())
                .is_some_and(|&b| b == b',' || b == b'}')
        })
        .count()
}

// ---------------------------------------------------------------------------
// Config discovery — test every artifact
// ---------------------------------------------------------------------------

const ARTIFACTS_BASE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../artifacts");
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
        .expect("artifacts dir missing — run treewalker-exp prepare")
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
                    eprintln!("{dataset}/{param}/{fw}: no model or predictions; skipped");
                    continue;
                }
                let forest = Forest::load(&model, param_dir.join("walker_config.json")).unwrap();
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

    assert!(
        !configs.is_empty(),
        "No artifact configs found. Run treewalker-exp prepare first."
    );
    configs
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_forest(param_dir: &Path, framework: &str) -> Forest {
    Forest::load(
        model_file(param_dir, framework),
        param_dir.join("walker_config.json"),
    )
    .unwrap()
}

fn load_forest_at_width(param_dir: &Path, framework: &str, width: usize) -> Forest {
    let file = WalkerConfig::from_file(param_dir.join("walker_config.json")).unwrap();
    let nf = file.n_features();
    let features = |role: fn(&WalkerConfig, usize) -> bool| {
        (0..nf).filter(|&f| role(&file, f)).collect::<Vec<_>>()
    };
    let config = WalkerConfig::builder(nf)
        .max_group_width(width)
        .varying(features(WalkerConfig::is_varying))
        .increasing(features(WalkerConfig::is_increasing))
        .decreasing(features(WalkerConfig::is_decreasing))
        .build()
        .unwrap();
    let path = model_file(param_dir, framework);
    let format = if path.extension().is_some_and(|e| e == "bin") {
        treewalker_gbdt::ModelFormat::TreeliteBinaryV4
    } else {
        treewalker_gbdt::ModelFormat::TreeliteJson
    };
    Forest::from_reader(
        std::fs::File::open(path).unwrap(),
        format,
        config,
        &treewalker_gbdt::LoadOptions::default(),
    )
    .unwrap()
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
    group_width: usize,
    n_rows: usize,
    offsets: Option<&[usize]>,
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

fn predict_all(
    forest: &Forest,
    data: &[f64],
    n_rows: usize,
    offsets: Option<&[usize]>,
) -> Vec<f64> {
    let mut results = vec![0.0f64; n_rows];
    let mut predictor = forest.predictor();
    if let Some(offs) = offsets {
        predictor.predict_groups(data, offs, &mut results);
    } else {
        predictor.predict_fixed(data, forest.config().max_group_width(), &mut results);
    }
    results
}

fn predict_full_all(forest: &Forest, data: &[f64], n_rows: usize) -> Vec<f64> {
    let mut results = vec![0.0f64; n_rows];
    forest.predict_full_walk(data, &mut results);
    results
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Whether every group's declared monotonic features are monotonic (ignoring NaN).
/// The tiled chunked-G cells repeat a 16-step panel inside each group, so they break
/// this caller contract on purpose.
fn honors_monotonic_contract(
    config: &WalkerConfig,
    data: &[f64],
    n_rows: usize,
    offsets: Option<&[usize]>,
) -> bool {
    let nf = config.n_features();
    let mut ok = true;
    for_each_group(config.max_group_width(), n_rows, offsets, |s, e| {
        for f in (0..nf).filter(|&f| config.is_increasing(f) || config.is_decreasing(f)) {
            let values: Vec<f64> = (s..e)
                .map(|r| data[r * nf + f])
                .filter(|v| !v.is_nan())
                .collect();
            ok &= values.windows(2).all(|w| {
                if config.is_increasing(f) {
                    w[0] <= w[1]
                } else {
                    w[0] >= w[1]
                }
            });
        }
    });
    ok
}

fn assert_same_bits(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    if let Some(r) = (0..actual.len()).find(|&r| actual[r].to_bits() != expected[r].to_bits()) {
        panic!("{label}: row {r}: {} != {}", actual[r], expected[r]);
    }
}

// ---------------------------------------------------------------------------
// Core correctness — runs on EVERY artifact
// ---------------------------------------------------------------------------

#[test]
fn test_reference_match() {
    for cfg in &all_configs() {
        let forest = load_forest(&cfg.param_dir, cfg.framework);
        let (data, n_rows) = load_test_data(&cfg.param_dir);
        let reference = load_reference(&cfg.param_dir, cfg.framework);

        let partial = predict_all(&forest, &data, n_rows, cfg.group_offsets.as_deref());
        let full = predict_full_all(&forest, &data, n_rows);

        let d_partial = max_diff(&partial, &reference);
        let d_full = max_diff(&full, &reference);
        eprintln!("{}: partial={d_partial:.2e} full={d_full:.2e}", cfg.label);

        assert!(
            d_partial < cfg.tol,
            "{} partial vs ref: {d_partial:.2e} exceeds {:.0e}",
            cfg.label,
            cfg.tol
        );
        assert!(
            d_full < cfg.tol,
            "{} full vs ref: {d_full:.2e} exceeds {:.0e}",
            cfg.label,
            cfg.tol
        );
    }
}

#[test]
fn test_partial_matches_full() {
    for cfg in &all_configs() {
        let forest = load_forest(&cfg.param_dir, cfg.framework);
        let (data, n_rows) = load_test_data(&cfg.param_dir);

        let partial = predict_all(&forest, &data, n_rows, cfg.group_offsets.as_deref());
        let full = predict_full_all(&forest, &data, n_rows);

        let d = max_diff(&partial, &full);
        eprintln!("{}: partial_vs_full={d:.2e}", cfg.label);
        assert!(d < TOL_FULL_WALK, "{} partial vs full: {d:.2e}", cfg.label);
    }
}

#[test]
fn test_single_row() {
    for cfg in &all_configs() {
        let forest = load_forest(&cfg.param_dir, cfg.framework);
        let (data, _) = load_test_data(&cfg.param_dir);
        let reference = load_reference(&cfg.param_dir, cfg.framework);
        let nf = forest.config().n_features();

        let mut single = vec![0.0f64; 1];
        forest.predict_full_walk(&data[0..nf], &mut single);

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
                && WalkerConfig::from_file(c.param_dir.join("walker_config.json"))
                    .unwrap()
                    .max_group_width()
                    > 1
        })
        .unwrap()
}

fn assert_partial_matches_full_on_obs(forest: &Forest, data: &[f64], obs: usize) {
    let (nf, h) = (
        forest.config().n_features(),
        forest.config().max_group_width(),
    );
    let rows = &data[obs * h * nf..(obs + 1) * h * nf];

    let mut full = vec![0.0f64; h];
    let mut partial = vec![0.0f64; h];
    forest.predict_full_walk(rows, &mut full);
    forest.predictor().predict_group(rows, &mut partial);

    let d = max_diff(&full, &partial);
    assert!(
        d < TOL_FULL_WALK,
        "Partial vs full max_diff={d:.2e} on observation {obs}"
    );
}

#[test]
fn test_all_nan_tv_features() {
    let cfg = first_config();
    let forest = load_forest(&cfg.param_dir, cfg.framework);
    let (mut data, _) = load_test_data(&cfg.param_dir);
    let nf = forest.config().n_features();
    let h = forest.config().max_group_width();

    for r in 0..h {
        for f in 0..nf {
            if forest.config().is_varying(f) {
                data[r * nf + f] = f64::NAN;
            }
        }
    }
    assert_partial_matches_full_on_obs(&forest, &data, 0);
}

#[test]
fn test_monotonic_columns_with_ties() {
    let cfg = first_config();
    let forest = load_forest(&cfg.param_dir, cfg.framework);
    let (mut data, _) = load_test_data(&cfg.param_dir);
    let nf = forest.config().n_features();
    let h = forest.config().max_group_width();

    for r in 0..h {
        for f in 0..nf {
            if forest.config().is_increasing(f) || forest.config().is_decreasing(f) {
                data[r * nf + f] = 5.0;
            }
        }
    }
    assert_partial_matches_full_on_obs(&forest, &data, 0);
}

#[test]
fn test_all_nan_monotonic_features() {
    let cfg = first_config();
    let forest = load_forest(&cfg.param_dir, cfg.framework);
    let (mut data, _) = load_test_data(&cfg.param_dir);
    let nf = forest.config().n_features();
    let h = forest.config().max_group_width();

    for r in 0..h {
        for f in 0..nf {
            if forest.config().is_increasing(f) || forest.config().is_decreasing(f) {
                data[r * nf + f] = f64::NAN;
            }
        }
    }
    assert_partial_matches_full_on_obs(&forest, &data, 0);
}

#[test]
fn test_extreme_feature_values() {
    let cfg = first_config();
    let forest = load_forest(&cfg.param_dir, cfg.framework);
    let (mut data, _) = load_test_data(&cfg.param_dir);
    let nf = forest.config().n_features();
    let h = forest.config().max_group_width();

    for &val in &[f64::MIN, f64::MAX, 0.0, -0.0, 1e300, -1e300] {
        for r in 0..h {
            for f in 0..nf {
                if !forest.config().is_varying(f) {
                    data[r * nf + f] = val;
                }
            }
        }
        assert_partial_matches_full_on_obs(&forest, &data, 0);
    }
}

// --- Ablation correctness (all configs × all modes) ---

#[test]
fn test_ablation_all_configs() {
    let ablations: &[(&str, Ablation)] = &[
        (
            "no_unsplit",
            Ablation {
                disable_unsplit: true,
                ..Default::default()
            },
        ),
        (
            "no_varying_precompute",
            Ablation {
                disable_varying_precompute: true,
                ..Default::default()
            },
        ),
        // Monotonic only has effect when precompute is also disabled.
        (
            "no_monotonic",
            Ablation {
                disable_monotonic: true,
                disable_varying_precompute: true,
                ..Default::default()
            },
        ),
        (
            "no_mono_no_unsplit",
            Ablation {
                disable_monotonic: true,
                disable_unsplit: true,
                disable_varying_precompute: true,
                ..Default::default()
            },
        ),
        (
            "all_disabled",
            Ablation {
                disable_monotonic: true,
                disable_unsplit: true,
                disable_varying_precompute: true,
                disable_predicate_sweep: true,
                disable_exact_sums: false,
            },
        ),
    ];

    // Every mode reaches the same leaves with the same rows, so predictions are
    // bitwise identical to the default.
    for cfg in &all_configs() {
        let (data, n_rows) = load_test_data(&cfg.param_dir);
        let offsets = cfg.group_offsets.as_deref();
        let forest = load_forest(&cfg.param_dir, cfg.framework);
        let reference = predict_all(&forest, &data, n_rows, offsets);
        let monotonic = honors_monotonic_contract(forest.config(), &data, n_rows, offsets);
        let (nf, width) = (
            forest.config().n_features(),
            forest.config().max_group_width(),
        );
        for &(mode, ablation) in ablations {
            // Without precompute, monotonic features are partitioned by prefix/suffix
            // scans that are only correct when the data honor the declared contract.
            let scans = ablation.disable_varying_precompute && !ablation.disable_monotonic;
            if scans && !monotonic {
                eprintln!(
                    "{}/{mode}: skipped, data break the monotonic contract",
                    cfg.label
                );
                continue;
            }
            let mut predictor = forest.research_predictor(ablation);
            let mut ablated = vec![0.0f64; n_rows];
            for_each_group(width, n_rows, offsets, |s, e| {
                predictor.predict_group(&data[s * nf..e * nf], &mut ablated[s..e]);
            });
            assert_same_bits(&ablated, &reference, &format!("{}/{mode}", cfg.label));
        }
        eprintln!("{}: all ablations ok", cfg.label);
    }
}

// --- Parse config correctness ---

#[test]
fn test_optionss() {
    use treewalker_gbdt::LoadOptions;

    let optionss: &[(&str, LoadOptions)] = &[
        (
            "no_tree_ordering",
            LoadOptions {
                disable_tree_ordering: true,
                ..Default::default()
            },
        ),
        (
            "no_bitset_intern",
            LoadOptions {
                disable_bitset_intern: true,
                ..Default::default()
            },
        ),
        (
            "no_prefix_grouping",
            LoadOptions {
                prefix_depth: 0,
                ..Default::default()
            },
        ),
        (
            "all_parse_disabled",
            LoadOptions {
                disable_tree_ordering: true,
                disable_bitset_intern: true,
                prefix_depth: 0,
                ..Default::default()
            },
        ),
        // disable_predicate_dedup tested separately — large models exceed u16 limit.
    ];

    // With exact sums, tree order and node layout cannot change a prediction.
    for cfg in &all_configs() {
        let (data, n_rows) = load_test_data(&cfg.param_dir);
        let offsets = cfg.group_offsets.as_deref();
        let reference_forest = load_forest(&cfg.param_dir, cfg.framework);
        let reference = predict_all(&reference_forest, &data, n_rows, offsets);
        for &(mode, ref pc) in optionss {
            let forest = Forest::load_with(
                model_file(&cfg.param_dir, cfg.framework),
                cfg.param_dir.join("walker_config.json"),
                pc,
            )
            .unwrap();
            let partial = predict_all(&forest, &data, n_rows, offsets);
            let label = format!("{}/{mode}", cfg.label);
            if reference_forest.exact_sums() {
                assert_same_bits(&partial, &reference, &label);
            } else {
                let full = predict_full_all(&forest, &data, n_rows);
                let d = max_diff(&partial, &full);
                assert!(d < TOL_FULL_WALK, "{label} max_diff={d:.2e}");
            }
        }
        eprintln!("{}: all parse configs ok", cfg.label);
    }
}

// --- Binary format correctness ---

#[test]
fn test_binary_matches_json() {
    let mut tested = 0;
    for cfg in &all_configs() {
        let bin_path = cfg
            .param_dir
            .join(format!("{}/model_treelite.bin", cfg.framework));
        let json_path = cfg
            .param_dir
            .join(format!("{}/model_treelite.json", cfg.framework));
        if !bin_path.exists() || !json_path.exists() {
            eprintln!("{}: no binary or JSON export; skipped", cfg.label);
            continue;
        }
        if std::fs::metadata(&json_path).unwrap().len() > SIMD_JSON_MAX_BYTES {
            eprintln!(
                "{}: JSON export exceeds the simd-json size limit; skipped",
                cfg.label
            );
            continue;
        }

        let forest_json = match Forest::load(&json_path, cfg.param_dir.join("walker_config.json")) {
            Ok(forest) => forest,
            Err(e) => {
                let empty = empty_thresholds(&json_path);
                assert!(
                    empty > 0
                        && matches!(&e, LoadError::MalformedModel(m) if m.starts_with("JSON: ")),
                    "{}: {e}",
                    json_path.display()
                );
                eprintln!(
                    "{}: {empty} empty thresholds ({e}); skipped",
                    json_path.display()
                );
                continue;
            }
        };
        let forest_bin = Forest::load(&bin_path, cfg.param_dir.join("walker_config.json")).unwrap();

        assert_eq!(
            forest_json.trees().len(),
            forest_bin.trees().len(),
            "{}: tree count mismatch",
            cfg.label,
        );

        let (data, n_rows) = load_test_data(&cfg.param_dir);
        let pred_json = predict_all(&forest_json, &data, n_rows, cfg.group_offsets.as_deref());
        let pred_bin = predict_all(&forest_bin, &data, n_rows, cfg.group_offsets.as_deref());

        let d = max_diff(&pred_json, &pred_bin);
        assert!(
            d < TOL_F64,
            "{}: binary vs json max_diff={d:.2e}",
            cfg.label,
        );
        tested += 1;
    }
    eprintln!("test_binary_matches_json: {tested} configs tested");
    assert!(
        tested > 0,
        "No .bin files found — run treewalker-exp prepare or generate test binaries"
    );
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
    let cfg = all_configs().into_iter().find(|c| {
        c.framework == "xgboost" && c.group_offsets.is_none() && {
            let forest = load_forest(&c.param_dir, c.framework);
            forest
                .nodes()
                .iter()
                .any(|n| n.is_categorical() && !n.is_leaf())
        }
    });
    let Some(cfg) = cfg else {
        eprintln!("No XGBoost config with categorical splits found, skipping");
        return;
    };

    let forest = load_forest(&cfg.param_dir, cfg.framework);
    let (mut data, _n_rows) = load_test_data(&cfg.param_dir);
    let nf = forest.config().n_features();
    let h = forest.config().max_group_width();

    // Find a categorical feature by scanning nodes.
    let cat_feature = forest
        .nodes()
        .iter()
        .find(|n| n.is_categorical() && !n.is_leaf())
        .map(|n| n.feature as usize)
        .unwrap();

    // Set the categorical feature to an out-of-range value (50) for all rows
    // in the first observation. This value exceeds the inline bitset range (0..31)
    // and would trigger the bug if out-of-range returned false instead of default_left.
    for r in 0..h {
        data[r * nf + cat_feature] = 50.0;
    }

    let rows = &data[..h * nf];
    let mut full_results = vec![0.0f64; h];
    let mut partial_results = vec![0.0f64; h];
    forest.predict_full_walk(rows, &mut full_results);
    forest.predictor().predict_group(rows, &mut partial_results);

    let d = max_diff(&full_results, &partial_results);
    eprintln!("{}: cat_out_of_range partial_vs_full={d:.2e}", cfg.label);
    assert!(
        d < TOL_FULL_WALK,
        "{}: cat out-of-range partial vs full max_diff={d:.2e}",
        cfg.label
    );
}

// --- Width-32 boundary test ---

/// Verify that partial eval works correctly with 32-row groups (the u32 mask boundary).
/// This exercises `u32::MAX >> (32 - 32) == u32::MAX` — the exact analog of the u16
/// overflow bug at width=16 (where `(1u16 << 16) - 1` wrapped to 0).
#[test]
fn test_width_32_boundary() {
    // Load a real model with max_group_width set to the u32 boundary.
    let base_cfg = first_config();
    let forest = load_forest_at_width(&base_cfg.param_dir, base_cfg.framework, 32);
    let nf = forest.config().n_features();

    // Build synthetic test data: 32 rows × n_features.
    // Use the first observation's constant features, vary the TV features linearly.
    let (real_data, _) = load_test_data(&base_cfg.param_dir);
    let mut data = vec![0.0f64; 32 * nf];
    for row in 0..32 {
        for f in 0..nf {
            if forest.config().is_varying(f) {
                // Linearly spaced values for varying features
                if forest.config().is_increasing(f) {
                    data[row * nf + f] = row as f64;
                } else if forest.config().is_decreasing(f) {
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
    let mut predictor = forest.predictor();
    forest.predict_full_walk(&data, &mut full_results);
    predictor.predict_group(&data, &mut partial_results);

    let d = max_diff(&full_results, &partial_results);
    eprintln!(
        "width_32_boundary: partial_vs_full max_diff={d:.2e} ({} trees, {} features)",
        forest.trees().len(),
        nf,
    );
    assert!(
        d < TOL_FULL_WALK,
        "Width-32 boundary: partial vs full max_diff={d:.2e} — u32 mask arithmetic may be wrong"
    );

    // A second call reuses the predictor's workspace.
    let mut ws_results = vec![0.0f64; 32];
    predictor.predict_group(&data, &mut ws_results);
    let d2 = max_diff(&full_results, &ws_results);
    assert!(
        d2 < TOL_FULL_WALK,
        "Width-32 boundary (workspace): max_diff={d2:.2e}"
    );
}

// --- Stats (first config) ---

#[test]
fn test_node_visit_stats() {
    use treewalker_gbdt::research::WorkCounters;

    let cfg = first_config();
    let forest = load_forest(&cfg.param_dir, cfg.framework);
    let (data, n_rows) = load_test_data(&cfg.param_dir);
    let (nf, h) = (
        forest.config().n_features(),
        forest.config().max_group_width(),
    );

    let mut results = vec![0.0f64; n_rows];
    let mut total = WorkCounters::default();
    let mut predictor = forest.research_predictor(Ablation::default());
    for (rows, out) in data.chunks(h * nf).zip(results.chunks_mut(h)) {
        total += predictor.predict_group_counted(rows, out);
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
    // predicate_masks builds u32 row masks, so it takes groups of at most 32 rows.
    // Wider groups use the u64 and Bits<W> mask paths, which test_reference_match,
    // test_partial_matches_full and test_wide_group_* validate.
    const SWEEP_HELPER_WIDTH: usize = 32;
    for cfg in &all_configs() {
        let forest = load_forest(&cfg.param_dir, cfg.framework);
        let (data, n_rows) = load_test_data(&cfg.param_dir);
        let nf = forest.config().n_features();

        let mut n_obs_tested = 0usize;
        let mut n_obs_skipped_wide = 0usize;
        let gw = forest.config().max_group_width();
        for_each_group(gw, n_rows, cfg.group_offsets.as_deref(), |start, end| {
            let n = end - start;
            if n > SWEEP_HELPER_WIDTH {
                n_obs_skipped_wide += 1;
                return;
            }

            let rows = &data[start * nf..end * nf];
            let sweep = forest.predicate_masks(rows, true);
            let brute = forest.predicate_masks(rows, false);

            assert_eq!(
                sweep.len(),
                brute.len(),
                "{}: mask vec lengths differ",
                cfg.label
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
                format!(
                    " ({n_obs_skipped_wide} skipped: groups > {SWEEP_HELPER_WIDTH} rows use wider masks)"
                )
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
    let first_cfg = all_configs()
        .into_iter()
        .next()
        .expect("No configs found — run treewalker-exp prepare");

    let forest = load_forest_at_width(&first_cfg.param_dir, first_cfg.framework, target_width);
    let (orig_data, orig_n_rows) = load_test_data(&first_cfg.param_dir);
    let nf = forest.config().n_features();
    let orig_gw = WalkerConfig::from_file(first_cfg.param_dir.join("walker_config.json"))
        .unwrap()
        .max_group_width();

    // Build synthetic groups: take the first 10 original groups' constant
    // features, expand each to `target_width` rows by cycling varying features
    // from the original data.
    let n_synthetic_groups = 10;
    let mut synth_data = Vec::new();

    for g in 0..n_synthetic_groups {
        // Source group's first row (for constant features).
        let src_start = if let Some(offs) = &first_cfg.group_offsets {
            if g >= offs.len() - 1 {
                break;
            }
            offs[g]
        } else {
            g * orig_gw
        };
        if (src_start + 1) * nf > orig_data.len() {
            break;
        }
        let const_row = &orig_data[src_start * nf..(src_start + 1) * nf];

        // Build `target_width` rows. Row 0 gets the original constant features.
        // All rows share constant features; varying features are cycled from
        // original data rows.
        for r in 0..target_width {
            let mut row = const_row.to_vec();
            // Fill varying features from a cycling source row.
            let src_row_idx = (src_start + r) % orig_n_rows;
            let src_row = &orig_data[src_row_idx * nf..(src_row_idx + 1) * nf];
            for f in (0..nf).filter(|&f| forest.config().is_varying(f)) {
                row[f] = src_row[f];
            }
            synth_data.extend_from_slice(&row);
        }
    }

    let total_rows = n_synthetic_groups * target_width;
    assert_eq!(synth_data.len(), total_rows * nf);

    // Predict with partial eval (exercises the mask width chosen for target_width).
    let mut results_partial = vec![0.0f64; total_rows];
    forest
        .predictor()
        .predict_fixed(&synth_data, target_width, &mut results_partial);

    // Predict with full walk (no masks, always correct).
    let mut results_full = vec![0.0f64; total_rows];
    forest.predict_full_walk(&synth_data, &mut results_full);

    let diff = max_diff(&results_partial, &results_full);
    eprintln!(
        "wide_group(width={target_width}): partial vs full max_diff={diff:.2e} ({n_synthetic_groups} groups, {} framework)",
        first_cfg.framework,
    );
    assert!(
        diff < TOL_FULL_WALK,
        "wide_group(width={target_width}): partial eval diverges from full walk: max_diff={diff:.2e}"
    );
}

#[test]
fn test_wide_group_48() {
    test_wide_group_partial_matches_full(48);
}

#[test]
fn test_wide_group_64() {
    test_wide_group_partial_matches_full(64);
}

#[test]
fn test_wide_group_96() {
    test_wide_group_partial_matches_full(96);
}

#[test]
fn test_wide_group_128() {
    test_wide_group_partial_matches_full(128);
}

#[test]
fn test_wide_group_200() {
    test_wide_group_partial_matches_full(200);
}

#[test]
fn test_wide_group_1000() {
    test_wide_group_partial_matches_full(1000);
}

/// Wider than one 1,024-row piece.
#[test]
fn test_wide_group_2500() {
    test_wide_group_partial_matches_full(2500);
}
