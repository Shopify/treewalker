use std::path::{Path, PathBuf};

use treewalker_gbdt::{Forest, LoadError, LoadOptions, WalkerConfig};

fn test_dir() -> PathBuf {
    PathBuf::from(std::env::var("TEST_ARTIFACTS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../artifacts/flchain/nt500_md8_h16"
        )
        .into()
    }))
}

/// Guard: skip tests that require artifacts not present on this machine.
fn require_artifacts(dir: &Path) -> bool {
    let needed = dir.join("walker_config.json");
    if !needed.exists() {
        eprintln!("Skipping: {} not found (run prepare.py)", needed.display());
        return false;
    }
    true
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

/// Load the cell's LightGBM JSON export, or `None` if the export is missing or has
/// empty thresholds. Every other load error fails the test.
fn load_json(dir: &Path, parse: &LoadOptions) -> Option<Forest> {
    let model = dir.join("lightgbm/model_treelite.json");
    if !model.exists() {
        eprintln!("Skipping: {} not found (run prepare.py)", model.display());
        return None;
    }
    match Forest::load_with(&model, dir.join("walker_config.json"), parse) {
        Ok(forest) => Some(forest),
        Err(e) => {
            let empty = empty_thresholds(&model);
            assert!(
                empty > 0 && matches!(&e, LoadError::MalformedModel(m) if m.starts_with("JSON: ")),
                "{}: {e}",
                model.display()
            );
            eprintln!(
                "Skipping: {} has {empty} empty thresholds ({e})",
                model.display()
            );
            None
        }
    }
}

/// A five-field grouping configuration under the test's own temporary directory.
fn temp_config(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(test);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("walker_config.json");
    std::fs::write(
        &path,
        r#"{"n_features": 1, "max_group_width": 1, "varying_features": [], "mono_inc_features": [], "mono_dec_features": []}"#,
    )
    .unwrap();
    path
}

#[test]
fn test_parse_model_basic() {
    let dir = test_dir();
    if !require_artifacts(&dir) {
        return;
    }

    let Some(forest) = load_json(&dir, &LoadOptions::default()) else {
        return;
    };
    assert!(!forest.trees().is_empty(), "should parse at least one tree");

    let t0 = &forest.trees()[0];
    let t0_nodes = forest.tree_nodes(t0);
    assert!(t0_nodes.len() >= 3, "tree 0 should have at least 3 nodes");
    assert!(!t0_nodes[0].is_leaf());

    let n_leaves = t0_nodes.iter().filter(|n| n.is_leaf()).count();
    let n_internal = t0_nodes.iter().filter(|n| !n.is_leaf()).count();
    assert_eq!(
        n_leaves,
        n_internal + 1,
        "binary tree: leaves = internal + 1"
    );

    for node in t0_nodes.iter().filter(|n| !n.is_leaf()) {
        assert!((node.skip as usize) < t0_nodes.len(), "skip out of range");
    }
}

#[test]
fn test_tree_order_is_deterministic() {
    let dir = test_dir();
    if !require_artifacts(&dir) {
        return;
    }

    let Some(forest_a) = load_json(&dir, &LoadOptions::default()) else {
        return;
    };
    let forest_b = load_json(&dir, &LoadOptions::default()).unwrap();

    assert_eq!(forest_a.trees().len(), forest_b.trees().len());
    for (a, b) in forest_a.trees().iter().zip(forest_b.trees()) {
        assert_eq!(a.node_start, b.node_start);
        assert_eq!(a.node_count, b.node_count);
        assert_eq!(a.bitset_start, b.bitset_start);
    }
}

#[test]
fn test_tree_ordering_changes_order_but_not_output() {
    let dir = test_dir();
    if !require_artifacts(&dir) {
        return;
    }

    let config = WalkerConfig::from_file(dir.join("walker_config.json")).unwrap();
    let Some(forest_ordered) = load_json(&dir, &LoadOptions::default()) else {
        return;
    };
    let forest_original = load_json(
        &dir,
        &LoadOptions {
            disable_tree_ordering: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(forest_ordered.trees().len(), forest_original.trees().len());

    let pl = config.max_group_width();
    let n_features = config.n_features();
    let n_obs = 2;
    let n_rows = n_obs * pl;
    let mut data = vec![0.5f64; n_rows * n_features];
    for r in 0..n_rows {
        data[r * n_features] = r as f64 * 0.1;
        if n_features > 1 {
            data[r * n_features + 1] = (r % 3) as f64;
        }
    }

    let mut results_ordered = vec![0.0f64; n_rows];
    let mut results_original = vec![0.0f64; n_rows];

    forest_ordered
        .predictor()
        .predict_fixed(&data, pl, &mut results_ordered);
    forest_original
        .predictor()
        .predict_fixed(&data, pl, &mut results_original);

    for (i, (a, b)) in results_ordered
        .iter()
        .zip(results_original.iter())
        .enumerate()
    {
        assert!((a - b).abs() < 1e-10, "row {i}: ordered={a}, original={b}");
    }
}

#[test]
fn test_lightgbm_json_matches_native() {
    let dir = test_dir();
    if !require_artifacts(&dir) {
        return;
    }

    let Some(forest) = load_json(&dir, &LoadOptions::default()) else {
        return;
    };
    let preds_path = dir.join("lightgbm/predictions.npy");
    if !preds_path.exists() {
        eprintln!("Skipping: {} not found", preds_path.display());
        return;
    }
    let reference: Vec<f64> = {
        let arr: ndarray::Array1<f64> = ndarray_npy::read_npy(&preds_path).unwrap();
        arr.to_vec()
    };

    let (data, n_rows) = load_test_bin(&dir.join("test_data.bin"));
    let mut results = vec![0.0f64; n_rows];
    let offsets = load_group_offsets_if_present(&dir);

    let mut predictor = forest.predictor();
    if let Some(offs) = &offsets {
        predictor.predict_groups(&data, offs, &mut results);
    } else {
        predictor.predict_fixed(&data, forest.config().max_group_width(), &mut results);
    }

    let max_diff = results
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    eprintln!("LightGBM JSON vs native max_diff: {max_diff:.2e}");
    assert!(max_diff < 1e-10, "JSON vs native max_diff={max_diff:.2e}");
}

#[test]
fn test_reject_non_json() {
    let err = Forest::load("model.csv", temp_config("reject_non_json")).unwrap_err();
    assert!(matches!(err, LoadError::Unsupported(_)), "{err}");
    assert!(
        err.to_string().contains("must end in .bin or .json"),
        "{err}"
    );
}

#[test]
fn test_reject_unknown_json_schema() {
    let config = temp_config("reject_unknown_json_schema");
    let model = config.with_file_name("bad_model.json");
    std::fs::write(&model, r#"{"foo": "bar"}"#).unwrap();
    let err = Forest::load(&model, &config).unwrap_err();
    assert!(err.to_string().contains("unknown JSON field"), "{err}");
}

fn load_test_bin(path: &Path) -> (Vec<f64>, usize) {
    let (data, n_rows, _) = treewalker_bench::load_raw_f64(path);
    (data, n_rows)
}

fn load_group_offsets_if_present(dir: &Path) -> Option<Vec<usize>> {
    let path = dir.join("group_offsets.bin");
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
