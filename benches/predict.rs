//! Developer benchmarks: `cargo bench`.
//!
//! Production prediction on the committed import fixtures, over synthetic groups
//! from 16 to 2,500 rows: one varying feature cycling through the fixtures'
//! boundary values, one constant feature. A quick check of the library alone; the
//! paper's measurements come from the benchmark runner (`experiments/`).

use divan::{Bencher, black_box};
use treewalker_gbdt::{Forest, LoadOptions, ModelFormat, WalkerConfig};

fn main() {
    divan::main();
}

const MODELS: &[&str] = &[
    "sigmoid_f64",
    "sigmoid_f32",
    "categories_right",
    "random_forest",
];
const WIDTHS: &[usize] = &[16, 17, 64, 128, 129, 1024, 1025, 2500];

fn forest(name: &str, width: usize) -> Forest {
    let path = format!(
        "{}/tests/fixtures/import/{name}.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let config = WalkerConfig::builder(2)
        .max_group_width(width)
        .varying([0])
        .build()
        .unwrap();
    Forest::from_bytes(
        &std::fs::read(path).unwrap(),
        ModelFormat::TreeliteBinaryV4,
        config,
        &LoadOptions::default(),
    )
    .unwrap()
}

/// One group of `width` rows: feature 0 varies, feature 1 is constant.
fn group(width: usize) -> Vec<f64> {
    const VALUES: [f64; 12] = [
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        31.0,
        32.0,
        40.0,
        64.0,
        f64::NAN,
    ];
    (0..width)
        .flat_map(|r| [VALUES[r % VALUES.len()], 1.0])
        .collect()
}

#[divan::bench(args = WIDTHS, consts = [0, 1, 2, 3])]
fn predict_group<const M: usize>(bencher: Bencher, width: usize) {
    let f = forest(MODELS[M], width);
    let mut predictor = f.predictor();
    let rows = group(width);
    let mut out = vec![0.0; width];
    bencher
        .counter(divan::counter::ItemsCount::new(width))
        .bench_local(|| predictor.predict_group(black_box(&rows), black_box(&mut out)));
}
