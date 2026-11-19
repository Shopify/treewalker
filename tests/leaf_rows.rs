//! Leaf rows of random models: the rows reach every leaf that sampled inputs reach,
//! as the generator's own evaluator walks them, independently of the library.
#![expect(clippy::float_cmp, reason = "the evaluator's exact category tests")]

mod common;

use std::collections::HashSet;

use common::model::{self, Model, Settings};
use common::rng::{Rng, mix};
use treewalker_gbdt::{LoadError, LoadOptions, ModelFormat, WalkerConfig, leaf_rows_from_bytes};

const SEED: u64 = 0x1EAF_2026_1118;

/// The leaves, as (tree, node), that `rows` reach.
fn reached(model: &Model, rows: &[f64]) -> HashSet<(usize, usize)> {
    rows.chunks_exact(model.n_features)
        .flat_map(|row| {
            model
                .trees
                .iter()
                .enumerate()
                .map(move |(t, tree)| (t, model.leaf_index(tree, row)))
        })
        .collect()
}

/// Check `cases` random models drawn with `settings`; the share of them whose leaf
/// rows are unsupported, as features split both ways are.
fn run(name: &str, cases: u64, settings: &Settings) -> f64 {
    let (mut unsupported, mut f32, mut categorical, mut leaves, mut n_rows) = (0u64, 0, 0, 0, 0);
    for case in 0..cases {
        let mut rng = Rng::new(mix(SEED ^ mix(case)));
        let (model, features) = model::generate(&mut rng, settings);
        let (bytes, format) = if model.has_infinite_threshold() || rng.chance(0.5) {
            (model.to_binary(&mut rng), ModelFormat::TreeliteBinaryV4)
        } else {
            (model.to_json(&mut rng), ModelFormat::TreeliteJson)
        };
        let config = WalkerConfig::builder(model.n_features)
            .max_group_width(1)
            .varying([])
            .build()
            .expect("a configuration");
        let rows = match leaf_rows_from_bytes(&bytes, format, &config, &LoadOptions::default()) {
            Err(LoadError::Unsupported(m)) if m.starts_with("leaf rows") => {
                unsupported += 1;
                continue;
            }
            rows => rows.unwrap_or_else(|e| panic!("case {case}: {e}")),
        };
        assert_eq!(rows.len() % model.n_features, 0);
        let by_rows = reached(&model, &rows);
        let sampled: Vec<f64> = (0..2000 * model.n_features)
            .map(|i| {
                let f = i % model.n_features;
                model::input(&mut rng, &features[f], model.precision, true)
            })
            .collect();
        let missed: Vec<_> = reached(&model, &sampled)
            .difference(&by_rows)
            .copied()
            .collect();
        assert!(
            missed.is_empty(),
            "case {case}: sampled inputs reach leaves {missed:?} that the {} leaf rows miss",
            rows.len() / model.n_features
        );
        // Each row reaches a leaf that no earlier row reaches.
        let rows = rows.len() / model.n_features;
        assert!(
            rows <= by_rows.len(),
            "case {case}: {rows} rows for {} leaves",
            by_rows.len()
        );
        f32 += usize::from(model.precision == model::Precision::F32);
        categorical += usize::from(model.max_category().is_some());
        leaves += by_rows.len();
        n_rows += rows;
    }
    eprintln!(
        "{name}: {cases} cases, {unsupported} unsupported, {f32} float32, {categorical} categorical; \
         {n_rows} rows reach {leaves} leaves"
    );
    unsupported as f64 / cases as f64
}

#[test]
fn ordinary_models() {
    let unsupported = run(
        "ordinary_models",
        300,
        &Settings {
            max_trees: 30,
            max_nodes: 1500,
            wide_leaves: 0.0,
            categorical: 0.2,
            wide: false,
            max_depth: 12,
            nested: 0.0,
        },
    );
    assert!(unsupported < 0.5, "{unsupported}");
}

/// Categories anywhere in `u32`, nested category sets and deep trees. Few features
/// are categorical: with hundreds of features, one of them split both ways would
/// leave few models supported.
#[test]
fn categorical_models_past_the_old_limits() {
    let unsupported = run(
        "categorical_models_past_the_old_limits",
        300,
        &Settings {
            max_trees: 20,
            max_nodes: 3000,
            wide_leaves: 0.0,
            categorical: 0.03,
            wide: true,
            max_depth: 40,
            nested: 0.5,
        },
    );
    assert!(unsupported < 0.5, "{unsupported}");
}
