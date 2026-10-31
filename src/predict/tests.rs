//! The kernel's outputs across lane counts, width dispatch, single rows' constant walk,
//! prefetch distances and rates, the research ablations and chunk sizes, and the
//! precompute's masks against the float predicates. The kernel against the exact oracle
//! is the property tests' (`tests/property.rs`).

// The property tests' generator, groups and comparisons, with the exact module's copy
// of their oracle and the crate's own sigmoid for their evaluator.
#[allow(dead_code, reason = "these tests use a part")]
#[path = "../../tests/common/bits.rs"]
mod bits;
#[allow(dead_code, reason = "these tests use a part")]
#[path = "../../tests/common/groups.rs"]
mod groups;
#[expect(clippy::float_cmp, reason = "exact expectations")]
#[allow(dead_code, reason = "these tests use a part")]
#[path = "../../tests/common/model.rs"]
mod model;

use crate::exact::oracle;
use crate::rng;
use bits::same;
use groups::{Role, config, group, offsets};
use model::{Precision, Settings};
use rng::Rng;

use super::sigmoid::sigmoid;
use super::{Groups, Level, Options, Predictor, Stats, Workspace};
use crate::config::Choices;
use crate::load::layout::PACK_ABOVE;
use crate::rules::{Prefix, Tuning};
use crate::{Forest, LoadOptions, ModelFormat, WalkerConfig};

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
    let config = config(m.n_features, widest.max(*rng.pick(&[1, 64, 2048])), &roles);
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
        pack_above: *rng.pick(&[0, PACK_ABOVE]),
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
        .flat_map(|&n| group(&mut rng, &features, &roles, m.precision, n, f64::total_cmp))
        .collect();
    Case {
        forest,
        data,
        offsets: offsets(&sizes),
    }
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
        level: super::level(),
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

/// The production kernel's outputs at `level`, and its counted ones (which also run the
/// ablation's instantiations) with `options`.
fn at_level(c: &Case, level: Level, options: Options) -> [Vec<f64>; 2] {
    let mut p = predictor(&c.forest, None);
    p.level = level;
    let n = c.data.len() / c.forest.config().n_features();
    let groups = Groups::Offsets(&c.offsets);
    p.check(&c.data, n, groups);
    let (mut out, mut counted) = (vec![f64::NAN; n], vec![f64::NAN; n]);
    p.kernel::<false, false>(&c.data, &mut out, groups, Options::default(), None, true);
    let mut stats = Stats::default();
    p.kernel::<true, true>(
        &c.data,
        &mut counted,
        groups,
        options,
        Some(&mut stats),
        true,
    );
    [out, counted]
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

/// Every level this CPU has gives the same bits as the build's own: the levels change
/// instructions, not operations (software `fma` rounds as FMA does). Off x86-64, and on
/// x86-64 builds that enable AVX2 and FMA, the only level is the build's.
#[test]
fn outputs_do_not_depend_on_the_level() {
    let levels = super::isa::levels();
    for seed in 0..if MIRI { 1 } else { 25 } {
        let c = case(0xC0FF_EE00 + seed);
        let scan = Options {
            precompute: false,
            ..Options::default()
        };
        let options = [Options::default(), scan][seed as usize % 2];
        let [reference, counted] = at_level(&c, Level::Native, options);
        same(&counted, &reference, &format!("{seed} counted"));
        for &level in &levels[1..] {
            let [out, counted] = at_level(&c, level, options);
            same(&out, &reference, &format!("{seed} {}", level.name()));
            same(
                &counted,
                &reference,
                &format!("{seed} {} counted", level.name()),
            );
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
        let chunks = chunk_trees(&needs, w, budget, usize::MAX);
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

#[test]
fn chunks_count_at_most_the_class_rows_of_a_trees_leaves() {
    use super::scratch::{Width, chunk_trees};
    use crate::load::layout::Need;
    // 100 trees of 4,096 leaves: a tree's leaves alone pass a 100,000-byte budget at
    // 36 bytes each, the 32 rows of the narrowest class take 1,152 bytes.
    let need = Need {
        leaves: 4096,
        slots: 0,
    };
    let needs = vec![need; 100];
    let every_leaf = chunk_trees(&needs, Width::W32, 100_000, usize::MAX);
    assert_eq!(every_leaf.len(), 100);
    assert!(every_leaf.iter().all(|c| c.leaves == 4096));
    let rows = chunk_trees(&needs, Width::W32, 100_000, Width::W32.rows());
    assert_eq!(
        rows.iter().map(|c| c.trees).collect::<Vec<_>>(),
        [(0, 86), (86, 100)]
    );
    assert_eq!(rows[0].leaves, 86 * 32);
    // Trees with fewer leaves than the class's rows keep their count.
    let small = vec![Need { leaves: 20, ..need }; 100];
    let capped = chunk_trees(&small, Width::W32, 100_000, Width::W32.rows());
    assert_eq!(capped[0].leaves, 20 * capped[0].trees.1);
    for (leaves, counted) in [(31, 31), (32, 32), (33, 32)] {
        let one = chunk_trees(&[Need { leaves, ..need }], Width::W32, 100_000, 32);
        assert_eq!(one[0].leaves, counted, "{leaves}");
    }
    // Arena slots keep their count at fixed and runtime widths: a runtime slot can be
    // taken at every light-only continuation, more than the rows.
    for w in [Width::W128, Width::of(1025)] {
        let deep = vec![
            Need {
                leaves: 4096,
                slots: 3000
            };
            4
        ];
        let chunks = chunk_trees(&deep, w, usize::MAX, w.rows());
        assert_eq!(
            (chunks[0].leaves, chunks[0].slots),
            (4 * w.rows(), 12_000),
            "{w:?}"
        );
    }
}

/// A tree of a leaf per value `0..values` of varying feature 0, each range split
/// unevenly (a third or two thirds of it left, by `t`), so that leaves end at
/// different depths.
fn leaf_per_value(values: usize, t: usize) -> model::Tree {
    use model::{Node, Op, Tree};
    fn grow(nodes: &mut Vec<Node>, lo: usize, hi: usize, t: usize) {
        if hi - lo == 1 {
            nodes.push(Node::Leaf {
                value: ((lo * 7 + t * 3) % 17) as f64 / 8.0,
                count: Some(1),
            });
            return;
        }
        let mid = lo + ((hi - lo) * (1 + t % 2) / 3).max(1);
        let at = nodes.len();
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
    let mut nodes = Vec::new();
    grow(&mut nodes, 0, values, t);
    Tree { nodes }
}

/// Groups whose every row reaches its own leaf, in trees of more leaves than any
/// group's class has rows, so that a tree's items and its leaves already reached
/// fill its chunk's capped count, with production's kernel in buffers sized as
/// production sizes them: masks by value (32 and 64 rows), fixed-width handles (128)
/// and runtime slots (2,048). The research walk that keeps sides without rows checks
/// the chunks sized for every leaf on the same groups.
#[test]
fn production_scratch_holds_a_leaf_per_row() {
    let sizes: &[usize] = if MIRI {
        &[32, 128, 2048]
    } else {
        &[32, 64, 128, 2048]
    };
    let (c, mut p) = leaf_per_row(sizes, if MIRI { 2 } else { 6 });
    let n = c.data.len();
    let mut full = vec![f64::NAN; n];
    c.forest.predict_full_walk(&c.data, &mut full);
    let groups = Groups::Offsets(&c.offsets);
    let mut out = vec![f64::NAN; n];
    p.check(&c.data, n, groups);
    p.kernel::<false, false>(&c.data, &mut out, groups, Options::default(), None, true);
    same(&out, &full, "production's kernel and scratch");
    let kept = Options {
        unsplit: false,
        ..Options::default()
    };
    same(&kernel(&c, None, kept), &full, "sides without rows kept");
}

/// Debug builds check `resolve`'s contract after each pass: with each chunk's count
/// one short of the leaf entries its group reaches, the kernel fails in the pass that
/// passes it, with items still to walk, rather than after the last.
#[test]
#[cfg(debug_assertions)]
fn a_pass_past_its_chunks_count_fails() {
    let (c, mut p) = leaf_per_row(&[32], 2);
    for chunk in p.workspace.kernel.common.chunks.iter_mut().flatten() {
        chunk.leaves = chunk.leaves.saturating_sub(1);
    }
    let groups = Groups::Offsets(&c.offsets);
    let mut out = vec![f64::NAN; c.data.len()];
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        p.kernel::<false, false>(&c.data, &mut out, groups, Options::default(), None, true);
    }))
    .expect_err("the check fails");
    let message = err.downcast_ref::<String>().expect("a formatted message");
    let items: usize = message
        .split_once("pass the chunk's bound: ")
        .and_then(|(_, rest)| rest.split_once(" items"))
        .and_then(|(k, _)| k.parse().ok())
        .expect("the message counts the items");
    assert!(items > 0, "{message}");
}

/// Groups of `sizes` rows whose every row reaches its own leaf, in `trees` trees of
/// more leaves than any group's class has rows, and a predictor with the scratch
/// production builds.
fn leaf_per_row(sizes: &[usize], trees: usize) -> (Case, Predictor) {
    let trees = (0..trees).map(|t| leaf_per_value(2100, t)).collect();
    let m = model::Model {
        precision: Precision::F64,
        n_features: 1,
        trees,
        average: false,
        base_score: 0.0,
        sigmoid: None,
        task: model::Task::Regressor,
    };
    let bytes = m.to_binary(&mut Rng::new(13));
    let config = WalkerConfig::builder(1)
        .max_group_width(2048)
        .varying([0])
        .build()
        .expect("a valid configuration");
    // No prefix groups: every tree's root is an item of the first pass.
    let choices = Choices {
        prefix: Prefix::Depth(0),
        ..Choices::default()
    };
    let forest = Forest::from_reader_with(
        std::io::Cursor::new(&bytes),
        ModelFormat::TreeliteBinaryV4,
        config,
        &LoadOptions::default(),
        &choices,
    )
    .expect("load");
    // Each group's values 0..n, a leaf each, in an order of its own (7 is prime to
    // every n).
    let data: Vec<f64> = sizes
        .iter()
        .enumerate()
        .flat_map(|(g, &n)| (0..n).map(move |r| ((r * 7 + g * 13) % n) as f64))
        .collect();
    let c = Case {
        forest,
        data,
        offsets: offsets(sizes),
    };
    let model = &c.forest.model;
    let p = Predictor {
        workspace: Workspace::production(model, Tuning::new(model)),
        model: std::sync::Arc::clone(model),
        level: super::level(),
    };
    (c, p)
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
    let (mut order, mut codes) = (vec![[0; 2]; 2 * n], vec![0; n]);
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

/// Stumps enough that a pass over one item per tree prefetches at the longest distance:
/// `rules::pass_distance` gives a pass of `n` items at most `n / lanes / 2`.
const STUMPS: usize = 2 * super::walk::MAX_DISTANCE * super::walk::MAX_LANES;

/// `STUMPS` stumps, every 7th on the varying feature 0 and the others on feature 1,
/// which is constant in each group, so that a group's first pass and a single row's
/// walk hold one item per tree; groups of 1, 2, 33, 65 and 1 rows, whose walks take
/// `u32` items, items carrying masks by value and items carrying arena handles. With
/// `bits`, feature 1's stumps are categorical and take bits, so the walks shift.
fn stumps(bits: bool) -> (Case, model::Model) {
    use model::{Node, Op, Task, Tree};
    let leaf = |value| Node::Leaf {
        value,
        count: Some(1),
    };
    let trees = (0..STUMPS)
        .map(|t| {
            let feature = usize::from(t % 7 != 0);
            let split = if bits && feature == 1 {
                let mut categories = vec![(t % 4) as u32, (t % 3) as u32 + 1];
                categories.sort_unstable();
                categories.dedup();
                Node::Categorical {
                    feature,
                    categories,
                    right_child: t % 2 == 0,
                    default_left: t % 3 == 0,
                    left: 1,
                    right: 2,
                    count: Some(3),
                }
            } else {
                Node::Numeric {
                    feature,
                    op: Op::Le,
                    threshold: (t % 5) as f64 - 1.5,
                    default_left: t % 3 == 0,
                    left: 1,
                    right: 2,
                    count: Some(3),
                }
            };
            Tree {
                nodes: vec![
                    split,
                    leaf((t % 11) as f64 / 8.0),
                    leaf(-((t % 13) as f64) / 16.0),
                ],
            }
        })
        .collect();
    let m = model::Model {
        precision: Precision::F64,
        n_features: 2,
        trees,
        average: false,
        base_score: 0.0,
        sigmoid: None,
        task: Task::Regressor,
    };
    let bytes = m.to_binary(&mut Rng::new(7));
    let sizes = [1, 2, 33, 65, 1];
    let config = WalkerConfig::builder(2)
        .max_group_width(65)
        .varying([0])
        .build()
        .expect("a valid configuration");
    // No prefix groups: every tree's root is an item of the first pass.
    let choices = Choices {
        tree_ordering: true,
        bitset_intern: true,
        prefix: Prefix::Depth(0),
        pack_above: if bits { 0 } else { usize::MAX },
        ..Choices::default()
    };
    let forest = Forest::from_reader_with(
        std::io::Cursor::new(&bytes),
        ModelFormat::TreeliteBinaryV4,
        config,
        &LoadOptions::default(),
        &choices,
    )
    .expect("load");
    assert_eq!(forest.model.layout.bins.has_shifts(), bits);
    let data: Vec<f64> = sizes
        .iter()
        .enumerate()
        .flat_map(|(g, &n)| {
            let constant = (g % 4) as f64 - 1.0;
            (0..n).flat_map(move |r| [(r % 6) as f64 - 2.0, constant])
        })
        .collect();
    let case = Case {
        forest,
        data,
        offsets: offsets(&sizes),
    };
    (case, m)
}

#[test]
fn prefetching_walks() {
    // Under Miri, the walks' prefetch addresses and their padding: the rule's choices,
    // then the shortest and the longest distance and, for the walk without varying
    // stops, a prefetch every 4 rounds; on numeric stumps and on stumps whose walks
    // shift, which take their own lanes.
    for bits in [false, true] {
        let (c, m) = stumps(bits);
        let reference = kernel(&c, None, Options::default());
        let expected: Vec<f64> = c.data.chunks_exact(2).map(|row| m.predict(row)).collect();
        same(&reference, &expected, &format!("the oracle, bits {bits}"));
        let longest = super::walk::MAX_DISTANCE;
        for (distance, rate) in [(1, 1), (longest, 1), (longest, 4)] {
            let o = Options {
                distance: Some(distance),
                rate: Some(rate),
                ..Options::default()
            };
            let what = format!("distance {distance}, rate {rate}, bits {bits}");
            same(&kernel(&c, Some(1 << 20), o), &reference, &what);
        }
    }
}
