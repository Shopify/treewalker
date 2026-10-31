//! Small production-shaped cases for Miri, which checks the unsafe code: each runs every
//! kind of unsafe operation of its path once. The README's development checks list the
//! commands: x86-64 with SSE2 and with AVX2, and RISC-V for the plain-word fallback; the
//! lib's `predict::tests::prefetching_walks` adds the walks that prefetch, and its
//! `production_scratch_holds_a_leaf_per_row` fills chunks in buffers sized as production
//! builds size them (these cases, built with `research`, size them for its plans too).
//! On AArch64, Miri lacks a NEON intrinsic the masks use. Whether the kernel is right on every path is the property tests'
//! question (`tests/property.rs`); natively these cases run once, as under Miri.
//!
//! Each case is a list of group sizes on one generated model:
//!
//! - `single_rows`: the constant walk, and wide leaves summed in turns;
//! - `masks_by_value`: masks of 32 and 64 bits in registers, and `i128` leaf runs;
//! - `masks_in_the_arena`: fixed-width masks of 128 and 256 bits in the arena, their
//!   SIMD loads and stores, resolve's lists, and wide leaf runs;
//! - `widest_masks`: masks of 512 and 1,024 bits;
//! - `runtime_width_*`: a group past 1,024 rows on runtime-width masks, with most
//!   features constant and with every feature varying;
//! - `a_leaf_per_row`: trees of more leaves than a group's class has rows, which bound
//!   a tree's items in a chunk, with every row at its own leaf;
//! - `category_bits`: the walks that shift, with every constant categorical feature in
//!   bits, on a single row and on groups.
#![expect(clippy::float_cmp, reason = "the generator's exact category test")]

mod common;

use common::bits::same;
use common::groups::{Role, config, group, offsets};
use common::model::{self, Model, Node, Op, Precision, Settings, Task, Tree};
use common::rng::{Rng, mix};
use treewalker_gbdt::research::LoadAblation;
use treewalker_gbdt::{Forest, LoadOptions, ModelFormat, WalkerConfig};

/// Generate the case `name` for `sizes`, with ordinary leaves or leaves past the
/// 126-bit rule (`wide`), and most features constant or all varying; check every row
/// against the exact full walk, bit for bit. The seed comes from `name`.
fn check(name: &str, sizes: &[usize], wide: bool, all_varying: bool) {
    check_with(name, sizes, wide, all_varying, false);
}

/// [`check`], with every constant categorical feature in bits when `bits`.
fn check_with(name: &str, sizes: &[usize], wide: bool, all_varying: bool, bits: bool) {
    let base = name.bytes().fold(0u64, |h, b| mix(h ^ u64::from(b)));
    let seed = mix(base ^ (u64::from(wide) << 1 | u64::from(all_varying)));
    let mut rng = Rng::new(seed);
    let settings = Settings {
        max_trees: 6,
        max_nodes: 90,
        wide_leaves: if wide { 1.0 } else { 0.0 },
        categorical: 0.4,
        wide: false,
        max_depth: 12,
        nested: 0.0,
    };
    // Few features keep the masks' threshold sweep small.
    let (m, features) = loop {
        let (m, features) = model::generate(&mut rng, &settings);
        if m.n_features <= 8 {
            break (m, features);
        }
    };
    // Binary, which needs no JSON parser beyond the empty attributes.
    let bytes = m.to_binary(&mut rng);
    let nf = m.n_features;
    let roles: Vec<Role> = (0..nf)
        .map(|_| {
            if all_varying {
                *rng.pick(&[Role::Varying, Role::Increasing, Role::Decreasing])
            } else {
                *rng.pick(&[
                    Role::Constant,
                    Role::Constant,
                    Role::Constant,
                    Role::Varying,
                ])
            }
        })
        .collect();
    let widest = *sizes.iter().max().unwrap();
    let config = config(nf, widest + rng.below(2) * 100, &roles);
    // No prefix groups, or the rule's.
    let ablation = LoadAblation {
        prefix_depth: *rng.pick(&[Some(0), None]),
        pack_above: bits.then_some(0),
        ..Default::default()
    };
    let forest = Forest::from_bytes_ablated(
        &bytes,
        ModelFormat::TreeliteBinaryV4,
        config,
        &LoadOptions::default(),
        &ablation,
    )
    .unwrap_or_else(|e| panic!("{name}: load: {e}"));
    assert_eq!(forest.layout_stats().packed, bits, "{name}");
    let data: Vec<f64> = sizes
        .iter()
        .flat_map(|&n| group(&mut rng, &features, &roles, m.precision, n, f64::total_cmp))
        .collect();
    let n = data.len() / nf;
    let mut full = vec![f64::NAN; n];
    forest.predict_full_walk(&data, &mut full);
    let mut out = vec![f64::NAN; n];
    forest
        .predictor()
        .predict_groups(&data, &offsets(sizes), &mut out);
    same(&out, &full, &format!("{name} sizes {sizes:?}"));
}

#[test]
fn single_rows() {
    check("single_rows", &[1, 1, 1], true, false);
}

#[test]
fn masks_by_value() {
    check("masks_by_value", &[2, 31, 32, 33, 64], false, false);
}

#[test]
fn masks_in_the_arena() {
    check("masks_in_the_arena", &[65, 128, 129], true, false);
}

#[test]
fn widest_masks() {
    check("widest_masks", &[257, 513], false, true);
}

// Seeded by the shape's former name, "pieces".
#[test]
fn runtime_width_mostly_constant() {
    check("pieces", &[1025], false, false);
}

#[test]
fn runtime_width_all_varying() {
    check("pieces", &[1025], false, true);
}

#[test]
fn category_bits() {
    check_with("category_bits_walks", &[1, 33], false, false, true);
}

/// Eight complete trees of 64 leaves on one varying feature, and two groups of 32
/// rows whose values 0 to 63 each reach their own leaf: a tree holds 32 items, the
/// most its chunk counts at the narrowest class.
#[test]
fn a_leaf_per_row() {
    // Leaves `lo..hi` in preorder, leaf `k` taking the rows of value `k`.
    fn grow(nodes: &mut Vec<Node>, lo: usize, hi: usize, t: usize) {
        if hi - lo == 1 {
            let value = ((lo * 7 + t * 3) % 17) as f64 / 8.0;
            nodes.push(Node::Leaf {
                value,
                count: Some(1),
            });
            return;
        }
        let (at, mid) = (nodes.len(), lo.midpoint(hi));
        nodes.push(Node::Numeric {
            feature: 0,
            op: Op::Le,
            threshold: mid as f64 - 0.5,
            default_left: false,
            left: at + 1,
            right: 0,
            count: Some((hi - lo) as u64),
        });
        grow(nodes, lo, mid, t);
        let next = nodes.len();
        if let Node::Numeric { right, .. } = &mut nodes[at] {
            *right = next;
        }
        grow(nodes, mid, hi, t);
    }
    let trees = (0..8)
        .map(|t| {
            let mut nodes = Vec::new();
            grow(&mut nodes, 0, 64, t);
            Tree { nodes }
        })
        .collect();
    let m = Model {
        precision: Precision::F64,
        n_features: 1,
        trees,
        average: false,
        base_score: 0.0,
        sigmoid: None,
        task: Task::Regressor,
    };
    let bytes = m.to_binary(&mut Rng::new(11));
    let config = WalkerConfig::builder(1)
        .max_group_width(32)
        .varying([0])
        .build()
        .expect("a valid configuration");
    // No prefix groups: every tree's root is an item of the first pass.
    let ablation = LoadAblation {
        prefix_depth: Some(0),
        ..Default::default()
    };
    let forest = Forest::from_bytes_ablated(
        &bytes,
        ModelFormat::TreeliteBinaryV4,
        config,
        &LoadOptions::default(),
        &ablation,
    )
    .expect("load");
    let data: Vec<f64> = (0..64).map(f64::from).collect();
    let mut full = vec![f64::NAN; 64];
    forest.predict_full_walk(&data, &mut full);
    let mut out = vec![f64::NAN; 64];
    forest
        .predictor()
        .predict_groups(&data, &offsets(&[32, 32]), &mut out);
    same(&out, &full, "a leaf per row");
}
