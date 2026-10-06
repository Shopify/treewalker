use treewalker_gbdt::ParseConfig;
use treewalker_gbdt::config::WalkerConfig;
use treewalker_gbdt::forest::Forest;

fn test_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("TEST_ARTIFACTS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../paper/experiments/artifacts/expedia/nt50_md8"
        )
        .into()
    }))
}

/// Guard: skip tests that require artifacts not present on this machine.
fn require_artifacts(dir: &std::path::Path) -> bool {
    let needed = dir.join("walker_config.json");
    if !needed.exists() {
        eprintln!("Skipping: {} not found (run prepare.py)", needed.display());
        return false;
    }
    true
}

#[test]
fn test_parse_model_basic() {
    let dir = test_dir();
    if !require_artifacts(&dir) {
        return;
    }

    let forest = Forest::load(
        dir.join("lightgbm/model_treelite.json"),
        dir.join("walker_config.json"),
    );
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

    let forest_a = Forest::load(
        dir.join("lightgbm/model_treelite.json"),
        dir.join("walker_config.json"),
    );
    let forest_b = Forest::load(
        dir.join("lightgbm/model_treelite.json"),
        dir.join("walker_config.json"),
    );

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

    let config = WalkerConfig::from_file(dir.join("walker_config.json"));
    let mut forest_ordered = Forest::load(
        dir.join("lightgbm/model_treelite.json"),
        dir.join("walker_config.json"),
    );
    let mut forest_original = Forest::load_with_config(
        dir.join("lightgbm/model_treelite.json"),
        dir.join("walker_config.json"),
        &ParseConfig {
            disable_tree_ordering: true,
            ..Default::default()
        },
    );

    assert_eq!(forest_ordered.trees().len(), forest_original.trees().len());

    let pl = config.max_group_width;
    let n_features = config.n_features;
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

    for s in 0..n_obs {
        let start = s * pl;
        let end = start + pl;
        forest_ordered.predict(&data, &mut results_ordered, start, end);
        forest_original.predict(&data, &mut results_original, start, end);
    }

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

    let mut forest = Forest::load(
        dir.join("lightgbm/model_treelite.json"),
        dir.join("walker_config.json"),
    );
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

    if let Some(offs) = &offsets {
        for pair in offs.windows(2) {
            forest.predict(&data, &mut results, pair[0], pair[1]);
        }
    } else {
        let pl = forest.config.max_group_width;
        for s in 0..(n_rows / pl) {
            forest.predict(&data, &mut results, s * pl, (s + 1) * pl);
        }
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
#[should_panic(expected = "must end in .bin or .json")]
fn test_reject_non_json() {
    let dir = test_dir();
    Forest::load("model.csv", dir.join("walker_config.json"));
}

#[test]
#[should_panic(expected = "unknown JSON field")]
fn test_reject_unknown_json_schema() {
    let dir = test_dir();
    if !require_artifacts(&dir) {
        return;
    }
    std::fs::write("/tmp/bad_model.json", r#"{"foo": "bar"}"#).unwrap();
    Forest::load("/tmp/bad_model.json", dir.join("walker_config.json"));
}

fn load_test_bin(path: &std::path::Path) -> (Vec<f64>, usize) {
    let (data, n_rows, _) = treewalker_bench::load_raw_f64(path);
    (data, n_rows)
}

fn load_group_offsets_if_present(dir: &std::path::Path) -> Option<Vec<usize>> {
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
