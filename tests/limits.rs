//! Models past the representation limits of earlier versions, which now have none
//! but the input caps of `LoadOptions`, 32-bit node indices and memory: more than
//! 65,535 varying predicates, more than 1,000,000 trees, and malformed category
//! segments. Trees of more than 32,767 nodes, deeper than 256, with more than 64
//! features and with any `u32` category run in the property tests. Each model is
//! checked against the oracle and the exact full walk.
//!
//! Not run at full scale: 32,000,000 nodes and more (a 512 MB layout), 2^29 varying
//! predicates and 2^30 bin slots, the layout's field widths; unit tests check those
//! bounds' logic on their counts instead.
#![expect(clippy::float_cmp, reason = "the generator's exact category test")]

mod common;

use common::bits::same;
use common::model::{Model, Node, Op, Precision, Task, Tree};
use common::rng::Rng;
use treewalker_gbdt::research::LoadAblation;
use treewalker_gbdt::{Forest, LoadError, LoadOptions, ModelFormat, WalkerConfig};

const fn leaf(value: f64) -> Node {
    Node::Leaf {
        value,
        count: Some(1),
    }
}

const fn model(trees: Vec<Tree>, n_features: usize) -> Model {
    Model {
        precision: Precision::F64,
        n_features,
        trees,
        average: false,
        base_score: 0.0,
        sigmoid: None,
        task: Task::Regressor,
    }
}

/// A complete binary tree in breadth-first order, `internal` splits on feature 0 at
/// distinct thresholds, and leaves valued by position.
fn complete(internal: usize) -> Tree {
    let total = 2 * internal + 1;
    let nodes = (0..total)
        .map(|i| {
            if i < internal {
                Node::Numeric {
                    feature: 0,
                    op: Op::Le,
                    threshold: i as f64 + 0.5,
                    default_left: i % 3 == 0,
                    left: 2 * i + 1,
                    right: 2 * i + 2,
                    count: Some((total - i) as u64),
                }
            } else {
                leaf((i % 97) as f64 / 8.0)
            }
        })
        .collect();
    Tree { nodes }
}

/// Load `m` from binary with `config` and `options`, and check groups of `rows`
/// against the oracle and the full walk.
fn check(m: &Model, config: WalkerConfig, ablation: &LoadAblation, rows: &[f64]) -> Forest {
    let mut rng = Rng::new(1);
    let bytes = m.to_binary(&mut rng);
    let forest = Forest::from_bytes_ablated(
        &bytes,
        ModelFormat::TreeliteBinaryV4,
        config,
        &LoadOptions::default(),
        ablation,
    )
    .unwrap_or_else(|e| panic!("load: {e}"));
    let nf = m.n_features;
    let n = rows.len() / nf;
    let expected: Vec<f64> = rows.chunks_exact(nf).map(|r| m.predict(r)).collect();
    let mut full = vec![f64::NAN; n];
    forest.predict_full_walk(rows, &mut full);
    let mut group = vec![f64::NAN; n];
    forest.predictor().predict_group(rows, &mut group);
    let mut single = vec![f64::NAN; n];
    forest.predictor().predict_fixed(rows, 1, &mut single);
    same(&full, &expected, "full walk");
    same(&group, &expected, "group");
    same(&single, &expected, "single rows");
    forest
}

/// The number of varying predicates: one left mask each, for a row of the two
/// features.
fn predicates(forest: &Forest) -> usize {
    forest.predicate_masks(&[0.0, 0.0], true).len()
}

#[test]
fn more_than_65535_varying_predicates() {
    // One tree of 140,001 nodes, every split on varying feature 0 at its own
    // threshold: 70,000 predicates, past the old u16 IDs.
    let m = model(vec![complete(70_000)], 2);
    let mut rng = Rng::new(7);
    let rows: Vec<f64> = (0..300)
        .flat_map(|r| {
            let x = match r % 5 {
                0 => f64::NAN,
                1 => rng.range(0, 70_000) as f64,
                _ => rng.range(0, 70_000) as f64 + 0.5,
            };
            [x, 1.0]
        })
        .collect();
    for dedup in [true, false] {
        let config = WalkerConfig::builder(2)
            .max_group_width(300)
            .varying([0])
            .build()
            .unwrap();
        let ablation = LoadAblation {
            disable_predicate_dedup: !dedup,
            ..LoadAblation::default()
        };
        let forest = check(&m, config, &ablation, &rows);
        assert_eq!(predicates(&forest), 70_000);
    }
}

#[test]
fn more_than_a_million_trees() {
    // 1,000,001 trees, past the old cap: single leaves and, every 1,000th, a split
    // on the constant or the varying feature.
    let n = 1_000_001;
    let trees: Vec<Tree> = (0..n)
        .map(|t| {
            if t % 1000 == 0 {
                Tree {
                    nodes: vec![
                        Node::Numeric {
                            feature: t / 1000 % 2,
                            op: Op::Lt,
                            threshold: (t % 7) as f64,
                            default_left: t % 3 == 0,
                            left: 1,
                            right: 2,
                            count: Some(3),
                        },
                        leaf(1.0),
                        leaf(-0.25),
                    ],
                }
            } else {
                Tree {
                    nodes: vec![leaf(if t % 2 == 0 { 0.5 } else { -0.125 })],
                }
            }
        })
        .collect();
    let m = model(trees, 2);
    let rows: Vec<f64> = (0..40)
        .flat_map(|r| [f64::from(r % 9) - 1.0, 3.0])
        .collect();
    let config = WalkerConfig::builder(2)
        .max_group_width(40)
        .varying([0])
        .build()
        .unwrap();
    check(&m, config, &LoadAblation::default(), &rows);
}

#[test]
fn overlapping_category_segments_are_rejected() {
    let split = |categories: Vec<u32>, left, right| Node::Categorical {
        feature: 0,
        categories,
        right_child: false,
        default_left: false,
        left,
        right,
        count: Some(2),
    };
    let tree = Tree {
        nodes: vec![
            split(vec![1, 2, 40], 1, 2),
            split(vec![3], 3, 4),
            leaf(1.0),
            leaf(2.0),
            leaf(3.0),
        ],
    };
    let m = model(vec![tree], 1);
    let config = || {
        WalkerConfig::builder(1)
            .max_group_width(4)
            .varying([])
            .build()
            .unwrap()
    };
    let mut rng = Rng::new(3);
    for overlap in [false, true] {
        let bytes = m.to_binary_with(&mut rng, overlap);
        let result = Forest::from_bytes(
            &bytes,
            ModelFormat::TreeliteBinaryV4,
            config(),
            &LoadOptions::default(),
        );
        match (overlap, result) {
            (false, Ok(_)) => {}
            (true, Err(LoadError::MalformedModel(msg))) => {
                assert!(msg.contains("overlap"), "{msg}");
            }
            (overlap, other) => panic!("overlap {overlap}: {other:?}"),
        }
    }
}
