//! The research API: ablated builds use their flags, counted and timed builds agree,
//! and the stages compose to production's output.
#![expect(clippy::float_cmp, reason = "exact hand-computable expectations")]
use simd_json::{OwnedValue as Value, prelude::*};
use treewalker_gbdt::research::{Ablation, Stages};
use treewalker_gbdt::{Forest, LoadOptions, ModelFormat, WalkerConfig};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/import/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}
fn raw(name: &str) -> Vec<f64> {
    fixture(name)[16..]
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn config() -> WalkerConfig {
    WalkerConfig::builder(2)
        .max_group_width(128)
        .varying([0])
        .build()
        .unwrap()
}
fn load(name: &str, parse: &LoadOptions) -> Forest {
    Forest::from_bytes(
        &fixture(name),
        ModelFormat::TreeliteBinaryV4,
        config(),
        parse,
    )
    .unwrap()
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
/// Every combination of the five runtime flags.
fn variants() -> impl Iterator<Item = Ablation> {
    (0..32u8).map(|b| Ablation {
        disable_varying_precompute: b & 1 != 0,
        disable_predicate_sweep: b & 2 != 0,
        disable_unsplit: b & 4 != 0,
        disable_monotonic: b & 8 != 0,
        disable_exact_sums: b & 16 != 0,
    })
}
const MODELS: [&str; 6] = [
    "sigmoid_f64.bin",
    "sigmoid_f32.bin",
    "identity.bin",
    "average_base.bin",
    "categories_left.bin",
    "random_forest.bin",
];

/// Three trees that send `x0 <= 1` to leaves 1e16, 1 and -1e16, in that order. The
/// exact sum is 1; adding in f64 in tree order loses the 1 and gives 0.
fn cancellation() -> Forest {
    let mut model = simd_json::to_owned_value(&mut fixture("identity.json")).unwrap();
    let tree = model["trees"][0].clone();
    let trees: Vec<Value> = [1e16, 1.0, -1e16]
        .into_iter()
        .map(|left| {
            let mut t = tree.clone();
            t["nodes"][1]["leaf_value"] = left.into();
            t["nodes"][2]["leaf_value"] = 0.0.into();
            t
        })
        .collect();
    model["trees"] = Value::Array(Box::new(trees));
    model["target_id"] = Value::Array(Box::new(vec![0.into(); 3]));
    model["class_id"] = Value::Array(Box::new(vec![0.into(); 3]));
    model["base_scores"] = Value::Array(Box::new(vec![0.0.into()]));
    // Keep the tree order, which decides the f64 sum.
    let parse = LoadOptions {
        disable_tree_ordering: true,
        ..Default::default()
    };
    Forest::from_bytes(
        &simd_json::to_vec(&model).unwrap(),
        ModelFormat::TreeliteJson,
        config(),
        &parse,
    )
    .unwrap()
}

#[test]
fn disabling_exact_sums_changes_the_timed_and_counted_outputs() {
    let forest = cancellation();
    assert!(forest.exact_sums());
    let rows = [0.5, 0.0, 0.5, 0.0, 2.0, 0.0];
    let mut production = [f64::NAN; 3];
    forest.predictor().predict_group(&rows, &mut production);
    assert_eq!(production, [1.0, 1.0, 0.0]);
    let mut full = [f64::NAN; 3];
    forest.predict_full_walk(&rows, &mut full);
    assert_eq!(full, [0.0, 0.0, 0.0]);
    for variant in variants() {
        let expected = if variant.disable_exact_sums {
            full
        } else {
            production
        };
        let mut r = forest.research_predictor(variant);
        let (mut timed, mut counted) = ([f64::NAN; 3], [f64::NAN; 3]);
        r.predict_group(&rows, &mut timed);
        r.predict_group_counted(&rows, &mut counted);
        assert_eq!(bits(&timed), bits(&expected), "timed {variant:?}");
        assert_eq!(bits(&counted), bits(&expected), "counted {variant:?}");
        let mut stages = Stages::default();
        r.predict_group_stages(&rows, &mut stages);
        assert_eq!(
            bits(&stages.tree_sum),
            bits(&expected),
            "stages {variant:?}"
        );
    }
}

#[test]
fn every_variant_matches_production_or_the_full_walk() {
    // Variants that keep exact sums are bit-identical to production. Variants that add
    // in f64 add in tree order, as the full walk does, so they match it bit for bit.
    let data = raw("data.bin");
    let n = data.len() / 2;
    for name in MODELS {
        let forest = load(name, &LoadOptions::default());
        let mut production = vec![0.0; n];
        forest
            .predictor()
            .predict_fixed(&data, 128, &mut production);
        let mut full = vec![0.0; n];
        forest.predict_full_walk(&data, &mut full);
        for variant in variants() {
            let expected = if variant.disable_exact_sums && forest.exact_sums() {
                &full
            } else {
                &production
            };
            let mut r = forest.research_predictor(variant);
            let (mut timed, mut counted) = (vec![f64::NAN; n], vec![f64::NAN; n]);
            for ((rows, t), c) in data
                .chunks(256)
                .zip(timed.chunks_mut(128))
                .zip(counted.chunks_mut(128))
            {
                r.predict_group(rows, t);
                r.predict_group_counted(rows, c);
            }
            assert_eq!(bits(&timed), bits(expected), "{name} timed {variant:?}");
            assert_eq!(bits(&counted), bits(&timed), "{name} counted {variant:?}");
        }
    }
}

#[test]
fn counters_show_each_flag() {
    let data = raw("data.bin");
    let forest = load("sigmoid_f64.bin", &LoadOptions::default());
    // One row: every varying split leaves one side empty.
    let count = |variant: Ablation| {
        let mut r = forest.research_predictor(variant);
        r.predict_group_counted(&data[..2], &mut [0.0])
    };
    let all_on = count(Ablation::default());
    assert!(all_on.precompute_row_evals > 0);
    assert_eq!(all_on.partition_row_evals, 0);
    assert!(all_on.unsplit_skips > 0);
    let no_precompute = count(Ablation {
        disable_varying_precompute: true,
        ..Default::default()
    });
    assert_eq!(no_precompute.precompute_row_evals, 0);
    assert!(no_precompute.partition_row_evals > 0);
    let no_unsplit = count(Ablation {
        disable_unsplit: true,
        ..Default::default()
    });
    assert_eq!(no_unsplit.unsplit_skips, 0);
    assert!(no_unsplit.recursive_calls > all_on.recursive_calls);
}

#[test]
fn stages_compose_to_the_output() {
    let data = raw("data.bin");
    for name in MODELS {
        let forest = load(name, &LoadOptions::default());
        let mut production = vec![0.0; 128];
        forest
            .predictor()
            .predict_group(&data[..256], &mut production);
        let mut stages = Stages::default();
        let mut r = forest.research_predictor(Ablation::default());
        r.predict_group_stages(&data[..256], &mut stages);
        assert_eq!(bits(&stages.output), bits(&production), "{name}");
        assert_eq!(stages.tree_sum.len(), 128);
        assert_eq!(stages.raw_margin.len(), 128);
    }
    // average_base averages two trees and adds a base score of 3: tree sums are
    // halved, then shifted, each step rounded on its own.
    let forest = load("average_base.bin", &LoadOptions::default());
    let mut stages = Stages::default();
    let mut r = forest.research_predictor(Ablation::default());
    r.predict_group_stages(&[0.0, 0.0], &mut stages);
    assert_eq!(stages.raw_margin, [stages.tree_sum[0] / 2.0 + 3.0]);
    assert_eq!(stages.output, stages.raw_margin);
    assert_eq!(stages.output, [3.375]);
    // A sigmoid applies the link to the raw margin.
    let forest = load("sigmoid_f64.bin", &LoadOptions::default());
    let mut r = forest.research_predictor(Ablation::default());
    r.predict_group_stages(&data[..256], &mut stages);
    for (&m, &o) in stages.raw_margin.iter().zip(&stages.output) {
        assert_eq!(o, 1.0 / (1.0 + (-m).exp()));
    }
}

#[test]
fn predicate_masks_from_the_sweep_match_the_brute_force() {
    let data = raw("data.bin");
    for name in MODELS {
        let forest = load(name, &LoadOptions::default());
        for rows in [1, 7, 32] {
            let group = &data[..rows * 2];
            assert_eq!(
                forest.predicate_masks(group, true),
                forest.predicate_masks(group, false),
                "{name} {rows}"
            );
        }
    }
}
