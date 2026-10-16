//! The kernel's outputs across lane counts, width dispatch, single rows' constant walk,
//! prefetch distances and rates, the research ablations and chunk sizes, and the
//! precompute's masks against the float predicates. The kernel against the exact oracle
//! is the property tests' (`tests/property.rs`).

// The property tests' generator, with the exact module's copy of their oracle.
#[expect(clippy::float_cmp, reason = "exact expectations")]
#[allow(dead_code, reason = "these tests use a part")]
#[path = "../../tests/common/model.rs"]
mod model;
#[allow(dead_code, reason = "these tests use a part")]
#[path = "../../tests/common/rng.rs"]
mod rng;

use crate::exact::oracle;
use model::{Feature, Precision, Settings};
use rng::Rng;

use super::{Groups, Options, Predictor, Workspace};
use crate::config::Choices;
use crate::rules::{Prefix, Tuning};
use crate::{Forest, LoadOptions, ModelFormat, WalkerConfig};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Constant,
    Varying,
    Increasing,
    Decreasing,
}

struct Case {
    forest: Forest,
    data: Vec<f64>,
    offsets: Vec<usize>,
}

/// Under Miri, which checks the unsafe code, a few small models and groups.
const MIRI: bool = cfg!(miri);

const SIZES: [usize; 14] = [1, 1, 2, 17, 32, 33, 64, 65, 129, 257, 513, 1024, 1025, 1100];

fn case(seed: u64) -> Case {
    let mut rng = Rng::new(seed);
    let settings = Settings {
        max_trees: if MIRI { 3 } else { 20 },
        max_nodes: if MIRI { 60 } else { 1200 },
        wide_leaves: 0.25,
        categorical: 0.3,
        wide: false,
        max_depth: 12,
        nested: 0.3,
    };
    let (m, features) = model::generate(&mut rng, &settings);
    let (bytes, format) = if m.has_infinite_threshold() || rng.chance(0.3) {
        (m.to_binary(&mut rng), ModelFormat::TreeliteBinaryV4)
    } else {
        (m.to_json(&mut rng), ModelFormat::TreeliteJson)
    };
    let roles: Vec<Role> = (0..m.n_features)
        .map(|_| {
            *rng.pick(&[
                Role::Constant,
                Role::Constant,
                Role::Varying,
                Role::Increasing,
                Role::Decreasing,
            ])
        })
        .collect();
    let mut sizes = Vec::new();
    while sizes.iter().sum::<usize>() < if MIRI { 150 } else { 1500 } {
        sizes.push(*rng.pick(if MIRI { &SIZES[..9] } else { &SIZES }));
    }
    let widest = *sizes.iter().max().expect("a group");
    let roles_ref = &roles;
    let with = |role| (0..m.n_features).filter(move |&f| roles_ref[f] == role);
    let config = WalkerConfig::builder(m.n_features)
        .max_group_width(widest.max(*rng.pick(&[1, 64, 2048])))
        .varying(with(Role::Varying))
        .increasing(with(Role::Increasing))
        .decreasing(with(Role::Decreasing))
        .build()
        .expect("a valid configuration");
    let choices = Choices {
        tree_ordering: !rng.chance(0.3),
        bitset_intern: !rng.chance(0.3),
        prefix: *rng.pick(&[
            Prefix::Depth(0),
            Prefix::Depth(1),
            Prefix::Rule,
            Prefix::Depth(3),
        ]),
        predicate_dedup: !rng.chance(0.1),
    };
    let forest = Forest::from_reader_with(
        std::io::Cursor::new(&bytes),
        format,
        config,
        &LoadOptions::default(),
        &choices,
    )
    .expect("load");
    let data: Vec<f64> = sizes
        .iter()
        .flat_map(|&n| group(&mut rng, &features, &roles, m.precision, n))
        .collect();
    let offsets = std::iter::once(0)
        .chain(sizes.iter().scan(0, |end, &n| {
            *end += n;
            Some(*end)
        }))
        .collect();
    Case {
        forest,
        data,
        offsets,
    }
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

/// A predictor whose chunks take `chunk_bytes`, or the rule's budget.
fn predictor(forest: &Forest, chunk_bytes: Option<usize>) -> Predictor {
    let rule = Tuning::new(&forest.model);
    let tuning = Tuning {
        chunk_bytes: chunk_bytes.unwrap_or(rule.chunk_bytes),
        ..rule
    };
    Predictor {
        workspace: Workspace::with_tuning(&forest.model, tuning),
        model: std::sync::Arc::clone(&forest.model),
    }
}

/// The kernel's outputs with `options`, in chunks of `chunk_bytes` or the
/// rule's.
fn kernel(c: &Case, chunk_bytes: Option<usize>, options: Options) -> Vec<f64> {
    let mut p = predictor(&c.forest, chunk_bytes);
    let mut out = vec![f64::NAN; c.data.len() / c.forest.config().n_features()];
    let groups = Groups::Offsets(&c.offsets);
    p.check(&c.data, out.len(), groups);
    p.kernel::<false, true>(&c.data, &mut out, groups, options, None, true);
    out
}

fn same(actual: &[f64], expected: &[f64], what: &str) {
    for (r, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "{what}: row {r}: {a:e} != {e:e}");
    }
}

#[test]
fn outputs_do_not_depend_on_how_the_kernel_runs() {
    let default = Options::default();
    for seed in 0..if MIRI { 1 } else { 25 } {
        let c = case(0xC0FF_EE00 + seed);
        let reference = kernel(&c, None, default);
        let check = |out: Vec<f64>, what: &str| same(&out, &reference, &format!("{seed} {what}"));
        let lanes = |k, distance| Options {
            lanes: Some(k),
            distance,
            ..default
        };
        let k = |n: usize| if MIRI { [1, 3, 12][n % 3] } else { n };
        for n in 1..=if MIRI { 3 } else { 12 } {
            let distance = [None, Some(0)][n % 2];
            check(kernel(&c, Some(12_000), lanes(k(n), distance)), "lanes");
        }
        for d in [0, 1, 2, 4, 8, 16] {
            let o = Options {
                distance: Some(d),
                ..default
            };
            check(kernel(&c, Some(12_000), o), "distance");
        }
        for r in [1, 2, 4] {
            let o = Options {
                rate: Some(r),
                ..default
            };
            check(kernel(&c, Some(12_000), o), "rate");
        }
        let narrowest = Options {
            width_dispatch: false,
            ..default
        };
        check(kernel(&c, Some(1), narrowest), "widest masks");
        let no_single = Options {
            g1_constant: false,
            ..default
        };
        check(kernel(&c, Some(1), no_single), "single rows masked");
        let both_off = Options {
            g1_constant: false,
            width_dispatch: false,
            ..default
        };
        check(kernel(&c, Some(2_000), both_off), "every path off");
        // The research ablations: the brute-force precompute, event scans with and
        // without the monotonic shortcut, and empty sides passed on.
        for (precompute, sweep, monotonic, unsplit) in [
            (true, false, true, true),
            (false, true, true, true),
            (false, true, false, true),
            (true, true, true, false),
            (false, false, false, false),
        ] {
            let o = Options {
                precompute,
                sweep,
                monotonic,
                unsplit,
                ..default
            };
            check(kernel(&c, Some(12_000), o), "ablations");
            let o = Options {
                width_dispatch: false,
                g1_constant: false,
                ..o
            };
            check(kernel(&c, Some(1_500), o), "ablations, widest masks");
        }
        for chunk in [1, 1_500, 40_000] {
            check(kernel(&c, Some(chunk), default), "chunks");
        }
    }
}

#[test]
fn chunks_follow_the_byte_budget_at_each_width() {
    use super::scratch::{Width, chunk_trees};
    use crate::load::layout::Need;
    // 100 trees of 50 leaves and 20 arena slots each.
    let need = Need {
        leaves: 50,
        slots: 20,
    };
    let needs = vec![need; 100];
    let trees = |w, budget| {
        let chunks = chunk_trees(&needs, w, budget);
        assert_eq!(chunks.first().map(|c| c.trees.0), Some(0));
        assert_eq!(chunks.last().map(|c| c.trees.1), Some(100));
        assert!(chunks.windows(2).all(|p| p[0].trees.1 == p[1].trees.0));
        chunks.iter().map(|c| c.trees.1 - c.trees.0).max().unwrap()
    };
    // 36 bytes per leaf for u32 masks, 60 for u64 masks and 40 for fixed-width
    // handles, which list varying stops; 128-byte slots for 1,024-bit masks.
    assert_eq!(trees(Width::W32, 20_000), 11);
    assert_eq!(trees(Width::W64, 20_000), 6);
    assert_eq!(trees(Width::W1024, 20_000), 4);
    assert_eq!(trees(Width::W32, 1), 1);
    assert_eq!(trees(Width::W1024, usize::MAX), 100);
}

/// Each varying predicate's left mask over `rows`, as runtime-width words, from its
/// float comparison.
fn reference_masks(c: &Case, rows: &[f64]) -> Vec<Vec<u64>> {
    let model = &c.forest.model;
    let nf = model.config.n_features();
    let words = 2 * crate::predict::mask::rows::units(rows.len() / nf);
    let f32 = model.threshold_type == crate::model::ThresholdType::F32;
    model
        .varying_predicates
        .iter()
        .map(|p| {
            let mut left = vec![0u64; words];
            for (r, row) in rows.chunks_exact(nf).enumerate() {
                let x = row[p.feature() as usize];
                let goes = if f32 {
                    p.goes_left::<true>(x, &model.categories)
                } else {
                    p.goes_left::<false>(x, &model.categories)
                };
                left[r / 64] |= u64::from(goes) << (r % 64);
            }
            left
        })
        .collect()
}

/// The precompute's left masks as runtime-width words, by the sweep or the brute
/// force, checking that each right mask is its complement; for groups of up to 1,024
/// rows, the 1,024-bit masks carried by value must agree.
fn precomputed_masks(c: &Case, rows: &[f64], sweep: bool) -> Vec<Vec<u64>> {
    use crate::predict::mask::{Mask, Wide};
    let model = &c.forest.model;
    let n = rows.len() / model.config.n_features();
    let np = model.varying_predicates.len();
    let words = 2 * crate::predict::mask::rows::units(n);
    let (mut order, mut codes) = (vec![(0.0, 0); n], vec![0; n]);
    let counters = &mut super::WorkCounters::default();
    let f32 = model.threshold_type == crate::model::ThresholdType::F32;
    let mut masks = vec![0u64; 2 * np * words];
    let (mut left, mut missing) = (vec![0u64; words], vec![0u64; words]);
    let out = &mut super::precompute::Rows {
        masks: &mut masks,
        words,
    };
    let (left, missing) = (&mut left[..], &mut missing[..]);
    let mut copies = vec![0u64; super::precompute::SWEEP_TURNS * words];
    let lefts = &mut copies[..];
    match (f32, sweep) {
        (false, true) => model.precompute_sweep::<false, _, _, false>(
            rows, &mut order, &mut codes, out, lefts, missing, counters,
        ),
        (true, true) => model.precompute_sweep::<true, _, _, false>(
            rows, &mut order, &mut codes, out, lefts, missing, counters,
        ),
        (false, false) => {
            model.precompute_brute_force::<false, _, _, false>(rows, out, left, counters);
        }
        (true, false) => {
            model.precompute_brute_force::<true, _, _, false>(rows, out, left, counters);
        }
    }
    let lefts: Vec<Vec<u64>> = masks
        .chunks_exact(2 * words)
        .map(|pair| {
            let (left, right) = pair.split_at(words);
            assert!(
                left.iter().zip(right).all(|(l, r)| *r == !l),
                "complementary sides"
            );
            left.to_vec()
        })
        .collect();
    if n <= 1024 {
        let mut out = vec![[Wide::<16>::ZERO; 2]; np];
        let (mut left, mut missing) = (Wide::<16>::ZERO, Wide::<16>::ZERO);
        let mut copies = [Wide::<16>::ZERO; super::precompute::SWEEP_TURNS];
        let out_ = &mut out[..];
        match (f32, sweep) {
            (false, true) => model.precompute_sweep::<false, Wide<16>, _, false>(
                rows,
                &mut order,
                &mut codes,
                out_,
                &mut copies[..],
                &mut missing,
                counters,
            ),
            (true, true) => model.precompute_sweep::<true, Wide<16>, _, false>(
                rows,
                &mut order,
                &mut codes,
                out_,
                &mut copies[..],
                &mut missing,
                counters,
            ),
            (false, false) => model.precompute_brute_force::<false, Wide<16>, _, false>(
                rows, out_, &mut left, counters,
            ),
            (true, false) => model.precompute_brute_force::<true, Wide<16>, _, false>(
                rows, out_, &mut left, counters,
            ),
        }
        for (&[left, right], words) in out.iter().zip(&lefts) {
            assert!(
                right == left.xor(Mask::ones_if(true)),
                "complementary sides"
            );
            assert!(
                left.0[..words.len()] == words[..],
                "fixed and runtime widths"
            );
        }
    }
    lefts
}

#[test]
fn precomputed_masks_match_the_float_predicates() {
    let (mut checked, mut categorical, mut monotonic, mut f32) = (0, 0, 0, 0);
    for seed in 0..if MIRI { 2 } else { 40 } {
        let c = case(0xB175_0000 + seed);
        let nf = c.forest.config().n_features();
        let plan = &c.forest.model.masks;
        if plan.features.is_empty() {
            continue;
        }
        categorical += usize::from(plan.features.iter().any(|f| !f.chains.is_empty()));
        let config = c.forest.config();
        monotonic += usize::from(plan.features.iter().any(|f| {
            !f.num.is_empty()
                && (config.is_increasing(f.feature) || config.is_decreasing(f.feature))
        }));
        f32 += usize::from(c.forest.model.threshold_type == crate::model::ThresholdType::F32);
        let mut rng = Rng::new(seed);
        for g in c.offsets.windows(2) {
            let group = &c.data[g[0] * nf..g[1] * nf];
            // As generated, monotonic features are ordered; reversed or shuffled, they
            // break the promise, which the masks must not depend on.
            let mut rows: Vec<&[f64]> = group.chunks_exact(nf).collect();
            for order in 0..3 {
                match order {
                    1 => rows.reverse(),
                    2 => {
                        for i in (1..rows.len()).rev() {
                            rows.swap(i, rng.range(0, i));
                        }
                    }
                    _ => {}
                }
                let group: Vec<f64> = rows.concat();
                let expected = reference_masks(&c, &group);
                assert!(
                    precomputed_masks(&c, &group, true) == expected,
                    "case {seed}: the sweep, order {order}"
                );
                assert!(
                    precomputed_masks(&c, &group, false) == expected,
                    "case {seed}: the brute force, order {order}"
                );
                checked += 1;
            }
        }
    }
    assert!(
        MIRI || (checked > 0 && categorical > 5 && monotonic > 5 && f32 > 5),
        "{checked} groups; models with categorical predicates {categorical}, monotonic \
         numeric ones {monotonic}, float32 thresholds {f32}"
    );
}
