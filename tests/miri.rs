//! Small cases that run every path of the kernel, sized for Miri, which checks the
//! unsafe code in the walk, the resolve passes, the leaf pass and the masks. The
//! README's development checks list the commands: x86-64 with and without AVX2, and
//! RISC-V for the plain-word fallback. Miri perturbs `exp` unless floats are
//! deterministic (`-Zmiri-deterministic-floats`), and the sigmoid would then differ
//! between two calls; on AArch64 it lacks a NEON intrinsic the masks use.
//!
//! Each shape is a list of group sizes: single rows (the constant path), masks of 32
//! and 64 bits carried by value, wider masks in the arena, and a group past 1,024 rows
//! at a runtime mask width. Each runs on models with ordinary leaves and with leaves past the
//! 126-bit rule, with every feature varying and with most features constant. Outside
//! Miri the same shapes run on more seeds.
#![expect(clippy::float_cmp, reason = "the generator's exact category test")]

mod common;

use common::model::{self, Feature, Precision, Settings};
use common::rng::{Rng, mix};
use treewalker_gbdt::research::LoadAblation;
use treewalker_gbdt::{Forest, LoadOptions, ModelFormat, WalkerConfig};

const MIRI: bool = cfg!(miri);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Constant,
    Varying,
    Increasing,
    Decreasing,
}

/// Generate a case for `sizes` and check every call against the exact full walk.
/// With `chained`, categorical features take nested sets, and a model without a
/// chain of two or more sets is skipped; returns whether the case ran.
fn check(seed: u64, sizes: &[usize], wide: bool, all_varying: bool, chained: bool) -> bool {
    let mut rng = Rng::new(seed);
    let settings = Settings {
        max_trees: 6,
        max_nodes: 90,
        wide_leaves: if wide { 1.0 } else { 0.0 },
        categorical: 0.4,
        wide: false,
        max_depth: 12,
        nested: if chained { 1.0 } else { 0.0 },
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
    let roles_ref = &roles;
    let with = |role| (0..nf).filter(move |&f| roles_ref[f] == role);
    let widest = *sizes.iter().max().unwrap();
    let config = WalkerConfig::builder(nf)
        .max_group_width(widest + rng.below(2) * 100)
        .varying(with(Role::Varying))
        .increasing(with(Role::Increasing))
        .decreasing(with(Role::Decreasing))
        .build()
        .unwrap();
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
    .unwrap_or_else(|e| panic!("seed {seed:#x}: load: {e}"));
    let stats = forest.layout_stats();
    if chained && stats.chain_sets <= stats.chain_slots {
        return false;
    }
    let data: Vec<f64> = sizes
        .iter()
        .flat_map(|&n| group(&mut rng, &features, &roles, m.precision, n))
        .collect();
    let n = data.len() / nf;
    let mut full = vec![f64::NAN; n];
    forest.predict_full_walk(&data, &mut full);
    let offsets: Vec<usize> = std::iter::once(0)
        .chain(sizes.iter().scan(0, |end, &n| {
            *end += n;
            Some(*end)
        }))
        .collect();
    let mut out = vec![f64::NAN; n];
    let mut predictor = forest.predictor();
    predictor.predict_groups(&data, &offsets, &mut out);
    for (r, (a, e)) in out.iter().zip(&full).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "seed {seed:#x} sizes {sizes:?}: row {r}: {a:e} != {e:e}"
        );
    }
    // The first group again on its own, through the same scratch.
    let mut first = vec![f64::NAN; sizes[0]];
    predictor.predict_group(&data[..sizes[0] * nf], &mut first);
    for (r, (a, e)) in first.iter().zip(&full).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "seed {seed:#x}: alone, row {r}");
    }
    true
}

/// One group's rows: constant features equal in every row, monotonic ones ordered.
fn group(
    rng: &mut Rng,
    features: &[Feature],
    roles: &[Role],
    precision: Precision,
    n: usize,
) -> Vec<f64> {
    let columns: Vec<Vec<f64>> = features
        .iter()
        .zip(roles)
        .map(|(f, &role)| match role {
            Role::Constant => vec![model::input(rng, f, precision, true); n],
            Role::Varying => (0..n)
                .map(|_| model::input(rng, f, precision, true))
                .collect(),
            Role::Increasing | Role::Decreasing => {
                let mut column: Vec<f64> = (0..n)
                    .map(|_| model::input(rng, f, precision, false))
                    .collect();
                column.sort_by(f64::total_cmp);
                if role == Role::Decreasing {
                    column.reverse();
                }
                column
            }
        })
        .collect();
    (0..n)
        .flat_map(|r| columns.iter().map(move |c| c[r]))
        .collect()
}

/// Run `sizes` on every combination of leaves and roles, `seeds` times each.
fn shape(name: &str, sizes: &[usize]) {
    let seeds = if MIRI { 1 } else { 40 };
    let base = name.bytes().fold(0u64, |h, b| mix(h ^ u64::from(b)));
    for s in 0..seeds {
        for (k, (wide, all_varying)) in [(false, false), (false, true), (true, false), (true, true)]
            .into_iter()
            .enumerate()
        {
            check(
                mix(base ^ (s << 2 | k as u64)),
                sizes,
                wide,
                all_varying,
                false,
            );
        }
    }
}

#[test]
fn chained_categories() {
    // Features whose splits' sets nest take chains of several sets: rank tables in
    // the bins and rank counts in the precompute, for a single row, masks by value
    // and in the arena.
    let base = mix(0x9001);
    let want = if MIRI { 2 } else { 20 };
    let mut ran = 0;
    for attempt in 0..2000 {
        let all_varying = attempt % 2 == 1;
        ran += usize::from(check(
            mix(base ^ attempt),
            &[1, 33, 65],
            false,
            all_varying,
            true,
        ));
        if ran == want {
            return;
        }
    }
    panic!("{ran} models with chained categories");
}

#[test]
fn single_rows() {
    shape("single_rows", &[1, 1, 1]);
}

#[test]
fn masks_by_value() {
    shape("masks_by_value", &[2, 31, 32, 33, 64]);
}

#[test]
fn masks_in_the_arena() {
    shape("masks_in_the_arena", &[65, 128, 129]);
}

#[test]
fn widest_masks() {
    shape("widest_masks", &[257, 513]);
}

#[test]
fn runtime_width() {
    // Seeded by the test's former name, "pieces", so that it keeps its cases.
    shape("pieces", &[1025]);
}
