//! Developer benchmarks: `cargo bench`.
//!
//! Production prediction on the committed import fixtures: synthetic groups from
//! 16 to 2,500 rows with one varying feature cycling through the fixtures'
//! boundary values and one constant feature; single rows (G = 1) in a fixed-seed
//! shuffle, one call per row with no varying feature; and one call over groups of
//! mixed sizes across the mask-width boundaries. A quick check of the library
//! alone; the paper's measurements come from the benchmark runner (`experiments/`).

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
/// The mixed case's group sizes, each once, in this order.
const MIXED: &[usize] = &[
    1, 33, 2, 129, 16, 1025, 17, 65, 257, 32, 513, 64, 3, 128, 1024,
];
/// Single rows per iteration.
const ROWS: usize = 1024;

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
/// The constant feature's values, as in the fixtures' data.
const CONSTANTS: [f64; 7] = [-1.0, -0.5, 0.0, 1.0, 40.0, 64.0, f64::NAN];

fn forest(name: &str, width: usize, varying: &[usize]) -> Forest {
    let path = format!(
        "{}/tests/fixtures/import/{name}.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let config = WalkerConfig::builder(2)
        .max_group_width(width)
        .varying(varying.iter().copied())
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
    (0..width)
        .flat_map(|r| [VALUES[r % VALUES.len()], 1.0])
        .collect()
}

#[divan::bench(args = WIDTHS, consts = [0, 1, 2, 3])]
fn predict_group<const M: usize>(bencher: Bencher, width: usize) {
    let f = forest(MODELS[M], width, &[0]);
    let mut predictor = f.predictor();
    let rows = group(width);
    let mut out = vec![0.0; width];
    bencher
        .counter(divan::counter::ItemsCount::new(width))
        .bench_local(|| predictor.predict_group(black_box(&rows), black_box(&mut out)));
}

/// `ROWS` rows covering every pair of values, in a fixed-seed shuffle, so
/// consecutive rows are independent.
fn independent_rows() -> Vec<f64> {
    let mut order: Vec<usize> = (0..ROWS).collect();
    let mut x = 0x9E37_79B9_7F4A_7C15_u64;
    for i in (1..ROWS).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        order.swap(i, (x % (i as u64 + 1)) as usize);
    }
    order
        .iter()
        .flat_map(|&r| {
            [
                VALUES[r % VALUES.len()],
                CONSTANTS[r / VALUES.len() % CONSTANTS.len()],
            ]
        })
        .collect()
}

/// G = 1: one call per row, no feature varying.
#[divan::bench(consts = [0, 1, 2, 3])]
fn predict_rows<const M: usize>(bencher: Bencher) {
    let f = forest(MODELS[M], 1, &[]);
    let mut predictor = f.predictor();
    let rows = independent_rows();
    let mut out = vec![0.0; ROWS];
    bencher
        .counter(divan::counter::ItemsCount::new(ROWS))
        .bench_local(|| {
            for (row, y) in black_box(&rows).chunks_exact(2).zip(out.iter_mut()) {
                predictor.predict_group(row, std::slice::from_mut(y));
            }
            black_box(&mut out);
        });
}

/// Groups of `MIXED` sizes in one `predict_groups` call, configured for the widest.
#[divan::bench(consts = [0, 1, 2, 3])]
fn predict_mixed<const M: usize>(bencher: Bencher) {
    let width = *MIXED.iter().max().unwrap();
    let f = forest(MODELS[M], width, &[0]);
    let mut predictor = f.predictor();
    let mut offsets = vec![0];
    let mut rows = Vec::new();
    for (g, &n) in MIXED.iter().enumerate() {
        for r in 0..n {
            rows.extend([VALUES[r % VALUES.len()], CONSTANTS[g % CONSTANTS.len()]]);
        }
        offsets.push(offsets.last().unwrap() + n);
    }
    let n_rows = *offsets.last().unwrap();
    let mut out = vec![0.0; n_rows];
    bencher
        .counter(divan::counter::ItemsCount::new(n_rows))
        .bench_local(|| {
            predictor.predict_groups(black_box(&rows), black_box(&offsets), black_box(&mut out));
        });
}
