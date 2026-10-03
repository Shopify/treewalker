use std::path::PathBuf;

use treewalker_gbdt::forest::Forest;
use treewalker_gbdt::{AblationMode, ParseConfig};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hoisting")
        .join(name)
}

fn read_array(name: &str) -> Vec<f64> {
    let bytes = std::fs::read(fixture(name)).unwrap();
    bytes[16..]
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

fn load(name: &str, extension: &str, hoist: bool, prefix_depth: usize, ordering: bool) -> Forest {
    Forest::load_with_config(
        fixture(&format!("{name}.{extension}")),
        fixture("walker_config.json"),
        &ParseConfig {
            hoist_constants: hoist,
            prefix_depth,
            disable_tree_ordering: !ordering,
            // Exercise categorical matching without shared bitset offsets.
            disable_bitset_intern: true,
            ..ParseConfig::default()
        },
    )
}

#[test]
fn both_readers_and_all_prediction_paths_preserve_reference_outputs() {
    let data = read_array("data.bin");
    let rows = data.len() / 3;
    for name in ["numeric_f64", "numeric_f32", "categorical_f64"] {
        let gtil = read_array(&format!("{name}_reference.bin"));
        for extension in ["json", "bin"] {
            for (prefix, ordering) in [(0, false), (2, true)] {
                let baseline = load(name, extension, false, prefix, ordering);
                let mut transformed = load(name, extension, true, prefix, ordering);
                let mut expected = vec![0.0; rows];
                let mut actual = vec![0.0; rows];
                baseline.predict_full(&data, &mut expected, 0, rows);
                transformed.predict_full(&data, &mut actual, 0, rows);
                assert!(transformed.hoist_stats().swaps > 0);
                assert_eq!(transformed.hoist_stats().varying_roots_after, 0);
                assert!(transformed.nodes().len() < baseline.nodes().len());
                for (i, ((&x, &y), &reference)) in
                    actual.iter().zip(&expected).zip(&gtil).enumerate()
                {
                    assert!((x - y).abs() < 1e-15, "{name}.{extension}, row {i}");
                    let tolerance = if name == "numeric_f32" { 1e-7 } else { 1e-14 };
                    assert!(
                        (x - reference).abs() < tolerance,
                        "GTIL {name}.{extension}, row {i}: {x} vs {reference}"
                    );
                }
                for width in [1, 32, 33, 64, 65, 128] {
                    for ablation in [
                        AblationMode::default(),
                        AblationMode {
                            disable_varying_precompute: true,
                            ..AblationMode::default()
                        },
                        AblationMode {
                            disable_unsplit: true,
                            ..AblationMode::default()
                        },
                        AblationMode {
                            disable_predicate_sweep: true,
                            ..AblationMode::default()
                        },
                    ] {
                        transformed.config.ablation = ablation;
                        // Chunk within each true group, never across constant-feature changes.
                        for group in (0..rows).step_by(128) {
                            for start in (group..group + 128).step_by(width) {
                                let end = (start + width).min(group + 128);
                                transformed.predict_with_stats(&data, &mut actual, start, end);
                            }
                        }
                        for (i, (&x, &y)) in actual.iter().zip(&expected).enumerate() {
                            assert!(
                                (x - y).abs() < 1e-15,
                                "{name}.{extension}, width {width}, row {i}"
                            );
                        }
                    }
                }
                for start in (0..rows).step_by(128) {
                    transformed.predict(&data, &mut actual, start, start + 128);
                }
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(x, y)| (x - y).abs() < 1e-15)
                );
            }
        }
    }
}

#[test]
fn hoisting_reduces_traversal_work_but_preserves_global_precompute_cost() {
    let data = read_array("data.bin");
    let mut baseline = load("numeric_f64", "bin", false, 0, false);
    let mut hoisted = load("numeric_f64", "bin", true, 0, false);
    let mut results = vec![0.0; data.len() / 3];
    let before = baseline.predict_with_stats(&data, &mut results, 0, 128);
    let after = hoisted.predict_with_stats(&data, &mut results, 0, 128);
    assert!(after.constant_steps < before.constant_steps);
    assert!(after.varying_splits < before.varying_splits);
    assert!(after.recursive_calls < before.recursive_calls);
    assert_eq!(after.precompute_row_evals, before.precompute_row_evals);
    assert_eq!(baseline.hoist_stats().trees_examined, 0);
}
