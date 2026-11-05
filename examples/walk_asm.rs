//! Production prediction on an opaque model, so that every walk the production kernel
//! can run is compiled in: the binary that `experiments/scripts/walk_spills.py` emits
//! as post-LTO assembly to check that the walk loops spill exactly where the lanes
//! model says.
//! It runs only when given a model and its configuration.

use std::hint::black_box;
use treewalker_gbdt::Forest;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, model, config] = args.as_slice() else {
        eprintln!("usage: walk_asm MODEL CONFIG");
        return;
    };
    let forest = Forest::load(model, config).expect("a model");
    let nf = forest.config().n_features();
    let mut p = forest.predictor();
    let n = black_box(64usize);
    let data = vec![0.5; n * nf];
    let data: &[f64] = black_box(&data);
    let mut out = vec![0.0; n];
    let out: &mut [f64] = black_box(&mut out);
    p.predict_group(data, out);
    p.predict_groups(data, black_box(&[0, 1, 64]), out);
    p.predict_fixed(data, black_box(3), out);
    println!("{}", out[0]);
}
