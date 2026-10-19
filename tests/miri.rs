//! Small production-shaped cases for Miri, which checks the unsafe code: each runs every
//! kind of unsafe operation of its path once. The README's development checks list the
//! commands: x86-64 with SSE2 and with AVX2, and RISC-V for the plain-word fallback; the
//! lib's `predict::tests::prefetching_walks` adds the walks that prefetch. Miri perturbs
//! `exp` unless floats are deterministic (`-Zmiri-deterministic-floats`), and the
//! sigmoid would then differ between two calls; on AArch64 it lacks a NEON intrinsic
//! the masks use. Whether the kernel is right on every path is the property tests'
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
//!   features constant and with every feature varying.
#![expect(clippy::float_cmp, reason = "the generator's exact category test")]

mod common;

use common::bits::same;
use common::groups::{Role, config, group, offsets};
use common::model::{self, Settings};
use common::rng::{Rng, mix};
use treewalker_gbdt::research::LoadAblation;
use treewalker_gbdt::{Forest, LoadOptions, ModelFormat};

/// Generate the case `name` for `sizes`, with ordinary leaves or leaves past the
/// 126-bit rule (`wide`), and most features constant or all varying; check every row
/// against the exact full walk, bit for bit. The seed comes from `name`.
fn check(name: &str, sizes: &[usize], wide: bool, all_varying: bool) {
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
