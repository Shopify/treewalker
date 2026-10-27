//! Property tests: random Treelite models, configurations and groups, checked
//! against the exact full walk and an independent oracle.
//!
//! Each case derives its own seed from the base seed, the test and the case index,
//! so a failure replays alone. Environment variables:
//!
//! - `TW_PROPTEST_CASES`: cases per test, instead of each test's default.
//! - `TW_PROPTEST_FULL`: each test's full-run cases, the full checks' count, set by the
//!   test's cost (`TW_PROPTEST_CASES` overrides it).
//! - `TW_PROPTEST_SEED`: the base seed (decimal or 0x-prefixed hex).
//! - `TW_PROPTEST_CASE`: run only this case index.
//! - `TW_PROPTEST_DUMP`: a directory to write each case's model, configuration and
//!   rows to.
#![expect(clippy::float_cmp, reason = "exact expectations")]

mod common;

use common::groups::{Role, config, offsets};
use common::model::{self, Feature, Model, Precision, Settings};
use common::rng::{Rng, mix};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use treewalker_gbdt::research::{Ablation, LoadAblation};
use treewalker_gbdt::{Forest, LoadOptions, ModelFormat};

const DEFAULT_SEED: u64 = 0x7EE5_A1C3_2026_1008;

/// Group sizes at the mask-width boundaries, runtime widths past 1,024 rows included.
const BOUNDARIES: [usize; 18] = [
    1, 2, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513, 1024, 1025, 2048, 2049,
];

struct Profile {
    name: &'static str,
    cases: usize,
    /// Cases with `TW_PROPTEST_FULL`: about 40 s at most in release on an M4 Pro.
    full_cases: usize,
    settings: Settings,
    /// Rows per case, at most (a single group may reach 2,049).
    max_rows: usize,
    /// Share of groups whose size is a boundary.
    boundaries: f64,
    /// Share of groups whose varying and monotonic columns repeat a few values,
    /// a quarter of them one value in every row.
    repeats: f64,
}

/// What a run covered, printed with `--nocapture` and checked on default runs.
#[derive(Debug, Default)]
struct Coverage {
    cases: usize,
    /// Models summed in `i128`, and in the wide accumulator.
    i128_sums: usize,
    wide_sums: usize,
    f32: usize,
    binary: usize,
    json: usize,
    pooled: usize,
    inline: usize,
    sigmoid: usize,
    average: usize,
    depth12: usize,
    fixed: usize,
    widest_group: usize,
    rows: usize,
    nodes: usize,
    trees: usize,
    /// Cases past the old representation limits.
    over_64_features: usize,
    categories_over_8159: usize,
    trees_over_32767_nodes: usize,
    depth_over_256: usize,
    /// Models whose layout has rank tables, and those with a chain of two or more
    /// sets, and of five or more on average.
    rank_tables: usize,
    chained: usize,
    long_chains: usize,
    /// Models loaded again with bits for some constant categorical feature.
    bits: usize,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn parse(name: &str, value: &str) -> u64 {
    let parsed = value
        .strip_prefix("0x")
        .map_or_else(|| value.parse(), |hex| u64::from_str_radix(hex, 16));
    parsed.unwrap_or_else(|e| panic!("{name}={value}: {e}"))
}

/// Run a profile's cases; on failure, name the seed and case that replay it.
fn run(profile: &Profile) -> Coverage {
    let seed = env("TW_PROPTEST_SEED").map_or(DEFAULT_SEED, |v| parse("TW_PROPTEST_SEED", &v));
    let default = if env("TW_PROPTEST_FULL").is_some() {
        profile.full_cases
    } else {
        profile.cases
    };
    let cases =
        env("TW_PROPTEST_CASES").map_or(default, |v| parse("TW_PROPTEST_CASES", &v) as usize);
    let only = env("TW_PROPTEST_CASE").map(|v| parse("TW_PROPTEST_CASE", &v) as usize);
    let name_hash = profile
        .name
        .bytes()
        .fold(0u64, |h, b| mix(h ^ u64::from(b)));
    let mut coverage = Coverage::default();
    for case in only.map_or(0..cases, |c| c..c + 1) {
        let case_seed = mix(seed ^ name_hash ^ mix(case as u64));
        let result = catch_unwind(AssertUnwindSafe(|| {
            check(profile, case, case_seed, &mut coverage);
        }));
        if let Err(e) = result {
            eprintln!(
                "{}: case {case} failed (case seed {case_seed:#x}); replay with \
                 TW_PROPTEST_SEED={seed:#x} TW_PROPTEST_CASE={case} \
                 cargo test --test property {}",
                profile.name, profile.name
            );
            resume_unwind(e);
        }
    }
    eprintln!("{}: {coverage:?}", profile.name);
    coverage
}

/// Generate a case and check it.
fn check(profile: &Profile, case: usize, seed: u64, coverage: &mut Coverage) {
    let mut rng = Rng::new(seed);
    let (model, features) = model::generate(&mut rng, &profile.settings);
    let binary = model.has_infinite_threshold() || rng.chance(0.25);
    let (bytes, format) = if binary {
        (model.to_binary(&mut rng), ModelFormat::TreeliteBinaryV4)
    } else {
        (model.to_json(&mut rng), ModelFormat::TreeliteJson)
    };
    let roles = roles(&mut rng, model.n_features);
    let (sizes, fixed) = sizes(&mut rng, profile);
    let widest = *sizes.iter().max().expect("a group");
    let max_width = max_group_width(&mut rng, fixed.unwrap_or(widest));
    let config = config(model.n_features, max_width, &roles);
    // Prefix depth None is the rule's groups.
    let ablation = LoadAblation {
        disable_tree_ordering: rng.chance(0.3),
        disable_bitset_intern: rng.chance(0.3),
        prefix_depth: *rng.pick(&[Some(0), Some(1), None, Some(2), Some(3), Some(5)]),
        disable_predicate_dedup: rng.chance(0.1),
        category_bits_above: None,
    };
    let groups: Vec<Vec<f64>> = sizes
        .iter()
        .map(|&n| {
            group(
                &mut rng,
                &features,
                &roles,
                model.precision,
                n,
                profile.repeats,
            )
        })
        .collect();
    let data: Vec<f64> = groups.concat();
    let n_rows = data.len() / model.n_features;
    if let Some(dir) = env("TW_PROPTEST_DUMP") {
        dump(
            &dir,
            profile.name,
            case,
            binary,
            &bytes,
            max_width,
            &roles,
            &data,
        );
    }

    let forest = Forest::from_bytes_ablated(
        &bytes,
        format,
        config.clone(),
        &LoadOptions::default(),
        &ablation,
    )
    .unwrap_or_else(|e| panic!("load: {e}; {}", describe(&model)));

    // The exact full walk equals the oracle on every model.
    let expected: Vec<f64> = data
        .chunks_exact(model.n_features)
        .map(|row| model.predict(row))
        .collect();
    let mut full = vec![f64::NAN; n_rows];
    forest.predict_full_walk(&data, &mut full);
    same(
        &full,
        &expected,
        "full walk against the oracle",
        &data,
        &model,
    );

    // Production equals the exact full walk on every model.
    let i128_sums = forest.layout_stats().wide_limbs == 0;
    let mut predictor = forest.predictor();
    let mut out = vec![f64::NAN; n_rows];
    let mut start = 0;
    for g in &groups {
        let rows = g.len() / model.n_features;
        predictor.predict_group(g, &mut out[start..start + rows]);
        start += rows;
    }
    let offsets = offsets(&sizes);
    let mut batch = vec![f64::NAN; n_rows];
    forest
        .predictor()
        .predict_groups(&data, &offsets, &mut batch);
    same(
        &out,
        &full,
        "predict_group against the full walk",
        &data,
        &model,
    );
    same(
        &batch,
        &full,
        "predict_groups against the full walk",
        &data,
        &model,
    );
    if let Some(width) = fixed {
        let mut fixed_out = vec![f64::NAN; n_rows];
        predictor.predict_fixed(&data, width, &mut fixed_out);
        same(
            &fixed_out,
            &full,
            "predict_fixed against the full walk",
            &data,
            &model,
        );
    }
    // The research predictor's timed build, separate code from production's.
    let mut research = vec![f64::NAN; n_rows];
    forest
        .research_predictor(Ablation::default())
        .predict_groups(&data, &offsets, &mut research);
    same(
        &research,
        &full,
        "the research predictor against the full walk",
        &data,
        &model,
    );
    // The brute-force precompute, on the same bins as the sweep.
    forest
        .research_predictor(Ablation::default().with("disable_predicate_sweep").unwrap())
        .predict_groups(&data, &offsets, &mut research);
    same(
        &research,
        &full,
        "the brute-force precompute against the full walk",
        &data,
        &model,
    );
    // Every constant categorical feature in bits rather than chains: the walks that
    // shift, on a group at a time and on the batch.
    let bits = LoadAblation {
        category_bits_above: Some(0),
        ..ablation
    };
    let forest_bits =
        Forest::from_bytes_ablated(&bytes, format, config, &LoadOptions::default(), &bits)
            .unwrap_or_else(|e| panic!("load with bits: {e}; {}", describe(&model)));
    let has_bits = forest_bits.layout_stats().bit_slots > 0;
    if has_bits {
        let mut predictor = forest_bits.predictor();
        let mut start = 0;
        for g in &groups {
            let rows = g.len() / model.n_features;
            predictor.predict_group(g, &mut out[start..start + rows]);
            start += rows;
        }
        same(
            &out,
            &full,
            "predict_group with bits against the full walk",
            &data,
            &model,
        );
        predictor.predict_groups(&data, &offsets, &mut batch);
        same(
            &batch,
            &full,
            "predict_groups with bits against the full walk",
            &data,
            &model,
        );
    }

    coverage.cases += 1;
    coverage.i128_sums += usize::from(i128_sums);
    coverage.wide_sums += usize::from(!i128_sums);
    coverage.f32 += usize::from(model.precision == Precision::F32);
    coverage.binary += usize::from(binary);
    coverage.json += usize::from(!binary);
    coverage.pooled += usize::from(model.has_pooled_categories());
    coverage.inline += usize::from(model.has_inline_categories());
    coverage.sigmoid += usize::from(model.sigmoid.is_some());
    coverage.average += usize::from(model.average);
    coverage.depth12 += usize::from(model.depth() == 12);
    coverage.fixed += usize::from(fixed.is_some());
    coverage.widest_group = coverage.widest_group.max(widest);
    coverage.rows += n_rows;
    coverage.nodes += model.node_count();
    coverage.trees += model.trees.len();
    coverage.over_64_features += usize::from(model.n_features > 64);
    coverage.categories_over_8159 += usize::from(model.max_category().is_some_and(|c| c > 8159));
    coverage.trees_over_32767_nodes +=
        usize::from(model.trees.iter().any(|t| t.nodes.len() > 32_767));
    coverage.depth_over_256 += usize::from(model.depth() > 256);
    let stats = forest.layout_stats();
    coverage.rank_tables += usize::from(stats.rank_table_bytes > 0);
    coverage.chained += usize::from(stats.chain_sets > stats.chain_slots);
    coverage.long_chains += usize::from(stats.chain_sets >= 5 * stats.chain_slots.max(1));
    coverage.bits += usize::from(has_bits);
}

/// Panic at the first row whose bits differ.
fn same(actual: &[f64], expected: &[f64], what: &str, data: &[f64], model: &Model) {
    let nf = model.n_features;
    for (r, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.to_bits() == e.to_bits(),
            "{what}: row {r}: {a:e} ({:#018x}) != {e:e} ({:#018x}); oracle tree sum {:e}; \
             row {:?}; {}",
            a.to_bits(),
            e.to_bits(),
            model.tree_sum(&data[r * nf..(r + 1) * nf]),
            &data[r * nf..(r + 1) * nf],
            describe(model),
        );
    }
}

fn describe(model: &Model) -> String {
    format!(
        "{:?} model, {} trees, {} nodes, depth {}, {} features",
        model.precision,
        model.trees.len(),
        model.node_count(),
        model.depth(),
        model.n_features
    )
}

/// Feature roles: all constant, all varying, or a mix.
fn roles(rng: &mut Rng, n_features: usize) -> Vec<Role> {
    let weights: [f64; 4] = match rng.below(4) {
        0 => [1.0, 0.0, 0.0, 0.0],
        1 => [0.0, 1.0, 0.0, 0.0],
        2 => [0.0, 0.4, 0.3, 0.3],
        _ => [0.5, 0.2, 0.15, 0.15],
    };
    (0..n_features)
        .map(|_| {
            let mut u = rng.unit();
            for (role, w) in [
                Role::Constant,
                Role::Varying,
                Role::Increasing,
                Role::Decreasing,
            ]
            .into_iter()
            .zip(weights)
            {
                if u < w {
                    return role;
                }
                u -= w;
            }
            Role::Constant
        })
        .collect()
}

fn group_size(rng: &mut Rng, profile: &Profile) -> usize {
    if rng.chance(profile.boundaries) {
        *rng.pick(&BOUNDARIES)
    } else {
        rng.range(1, 2500)
    }
}

/// Group sizes within the profile's row budget and, for a call to
/// `predict_fixed`, the width: groups of that width and a shorter last group.
fn sizes(rng: &mut Rng, profile: &Profile) -> (Vec<usize>, Option<usize>) {
    let budget = profile.max_rows;
    if rng.chance(0.3) {
        let width = group_size(rng, profile);
        let full = rng.range(0, budget / width);
        let mut sizes = vec![width; full];
        if full == 0 || rng.chance(0.7) {
            sizes.push(rng.range(1, width));
        }
        return (sizes, Some(width));
    }
    let mut sizes = vec![group_size(rng, profile)];
    loop {
        let n = group_size(rng, profile);
        if sizes.iter().sum::<usize>() + n > budget {
            break;
        }
        sizes.push(n);
    }
    (sizes, None)
}

/// The configured maximum: the widest group, or more.
fn max_group_width(rng: &mut Rng, widest: usize) -> usize {
    match rng.below(6) {
        0..=2 => widest,
        3 => widest + rng.range(1, widest),
        4 => BOUNDARIES
            .iter()
            .copied()
            .find(|&b| b > widest)
            .unwrap_or(widest + 1),
        _ => *rng.pick(&[2500, 5000, 100_000]).max(&widest),
    }
}

/// One group's rows, row-major: each constant feature equal in every row, each
/// monotonic feature ordered in its direction and free of NaN. With probability
/// `repeats`, varying and monotonic columns draw from a few values, sometimes one,
/// and varying ones are sometimes sorted.
fn group(
    rng: &mut Rng,
    features: &[Feature],
    roles: &[Role],
    precision: Precision,
    n: usize,
    repeats: f64,
) -> Vec<f64> {
    let few = (repeats > 0.0 && rng.chance(repeats))
        .then(|| if rng.chance(0.25) { 1 } else { rng.range(2, 4) });
    let columns: Vec<Vec<f64>> = features
        .iter()
        .zip(roles)
        .map(|(f, &role)| {
            let nan = role == Role::Varying;
            let mut column: Vec<f64> = match few {
                Some(k) => {
                    let pool: Vec<f64> = (0..k)
                        .map(|_| model::input(rng, f, precision, nan))
                        .collect();
                    (0..n).map(|_| *rng.pick(&pool)).collect()
                }
                None if role == Role::Constant => vec![model::input(rng, f, precision, true); n],
                None => (0..n)
                    .map(|_| model::input(rng, f, precision, nan))
                    .collect(),
            };
            match role {
                Role::Constant => vec![column[0]; n],
                Role::Varying => {
                    if few.is_some() && rng.chance(0.3) {
                        column.sort_by(f64::total_cmp);
                    }
                    column
                }
                Role::Increasing | Role::Decreasing => {
                    column.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
                    if role == Role::Decreasing {
                        column.reverse();
                    }
                    column
                }
            }
        })
        .collect();
    (0..n)
        .flat_map(|r| columns.iter().map(move |c| c[r]))
        .collect()
}

#[expect(clippy::too_many_arguments, reason = "everything that replays a case")]
fn dump(
    dir: &str,
    name: &str,
    case: usize,
    binary: bool,
    model: &[u8],
    max_width: usize,
    roles: &[Role],
    data: &[f64],
) {
    let base = std::path::Path::new(dir).join(format!("{name}-{case}"));
    let extension = if binary { "bin" } else { "json" };
    std::fs::write(base.with_extension(extension), model).unwrap();
    let list = |role| {
        let v: Vec<String> = (0..roles.len())
            .filter(|&f| roles[f] == role)
            .map(|f| f.to_string())
            .collect();
        v.join(",")
    };
    let config = format!(
        r#"{{"n_features":{},"max_group_width":{max_width},"varying_features":[{}],"mono_inc_features":[{}],"mono_dec_features":[{}]}}"#,
        roles.len(),
        list(Role::Varying),
        list(Role::Increasing),
        list(Role::Decreasing)
    );
    std::fs::write(base.with_extension("config.json"), config).unwrap();
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(base.with_extension("rows.f64"), bytes).unwrap();
}

/// Whether the run covers the profile's default or full cases, so coverage can be checked.
fn full_run() -> bool {
    env("TW_PROPTEST_CASE").is_none() && env("TW_PROPTEST_CASES").is_none()
}

#[test]
fn ordinary_models() {
    let c = run(&Profile {
        name: "ordinary_models",
        cases: 400,
        full_cases: 5_000,
        settings: Settings {
            max_trees: 30,
            max_nodes: 1500,
            wide_leaves: 0.0,
            categorical: 0.2,
            wide: false,
            max_depth: 12,
            nested: 0.0,
        },
        max_rows: 1500,
        boundaries: 0.7,
        repeats: 0.0,
    });
    if full_run() {
        assert!(
            c.i128_sums == c.cases,
            "ordinary leaves fit the 126-bit rule"
        );
        assert!(c.f32 > 0 && c.binary > 0 && c.json > 0 && c.fixed > 0);
        assert!(c.sigmoid > 0 && c.average > 0 && c.pooled > 0 && c.inline > 0);
        assert!(c.depth12 > 0);
    }
}

#[test]
fn leaves_past_the_126_bit_rule() {
    let c = run(&Profile {
        name: "leaves_past_the_126_bit_rule",
        cases: 300,
        full_cases: 5_000,
        settings: Settings {
            max_trees: 30,
            max_nodes: 1500,
            wide_leaves: 1.0,
            categorical: 0.2,
            wide: false,
            max_depth: 12,
            nested: 0.0,
        },
        max_rows: 1500,
        boundaries: 0.7,
        repeats: 0.0,
    });
    if full_run() {
        assert!(c.wide_sums > c.cases / 2, "{} of {}", c.wide_sums, c.cases);
        assert!(c.i128_sums > 0 && c.f32 > 0);
    }
}

#[test]
fn categorical_models() {
    let c = run(&Profile {
        name: "categorical_models",
        cases: 300,
        full_cases: 5_000,
        settings: Settings {
            max_trees: 20,
            max_nodes: 1500,
            wide_leaves: 0.1,
            categorical: 0.9,
            wide: false,
            max_depth: 12,
            nested: 0.0,
        },
        max_rows: 1500,
        boundaries: 0.7,
        repeats: 0.0,
    });
    if full_run() {
        assert!(c.pooled > c.cases / 2 && c.inline > c.cases / 2);
    }
}

#[test]
fn repeated_varying_values() {
    // Masks from bins: ties between rows, varying columns already sorted, groups
    // whose varying features hold one value, missing and categorical varying
    // values, and float32 models.
    let c = run(&Profile {
        name: "repeated_varying_values",
        cases: 300,
        full_cases: 5_000,
        settings: Settings {
            max_trees: 20,
            max_nodes: 1500,
            wide_leaves: 0.1,
            categorical: 0.4,
            wide: false,
            max_depth: 12,
            nested: 0.0,
        },
        max_rows: 2500,
        boundaries: 0.5,
        repeats: 0.8,
    });
    if full_run() {
        assert!(c.f32 > 0 && c.pooled > 0 && c.inline > 0);
    }
}

#[test]
fn deep_trees_and_wide_groups() {
    let c = run(&Profile {
        name: "deep_trees_and_wide_groups",
        cases: 80,
        full_cases: 5_000,
        settings: Settings {
            max_trees: 6,
            max_nodes: 9000,
            wide_leaves: 0.2,
            categorical: 0.2,
            wide: false,
            max_depth: 12,
            nested: 0.0,
        },
        max_rows: 5000,
        boundaries: 0.9,
        repeats: 0.0,
    });
    if full_run() {
        assert!(
            c.depth12 > 0 && c.widest_group >= 2049,
            "{}",
            c.widest_group
        );
    }
}

#[test]
fn past_the_old_feature_and_category_limits() {
    // 65 to 300 features, and category IDs anywhere in u32, most features with
    // more than the 30 categories an inline split holds.
    let c = run(&Profile {
        name: "past_the_old_feature_and_category_limits",
        cases: 150,
        full_cases: 5_000,
        settings: Settings {
            max_trees: 20,
            max_nodes: 1500,
            wide_leaves: 0.1,
            categorical: 0.5,
            wide: true,
            max_depth: 12,
            nested: 0.0,
        },
        max_rows: 1200,
        boundaries: 0.6,
        repeats: 0.3,
    });
    if full_run() {
        assert!(c.over_64_features == c.cases && c.categories_over_8159 > c.cases / 2);
        assert!(c.rank_tables > c.cases / 2, "{c:?}");
        assert!(c.f32 > 0 && c.binary > 0 && c.json > 0);
    }
}

#[test]
fn nested_categorical_sets() {
    // Categorical features whose splits take runs of a shuffled order from one of
    // its first entries: sets that nest into long chains and overlap across them,
    // both heavy sides, so that a feature needs several chain slots.
    let c = run(&Profile {
        name: "nested_categorical_sets",
        cases: 200,
        full_cases: 5_000,
        settings: Settings {
            max_trees: 30,
            max_nodes: 2000,
            wide_leaves: 0.1,
            categorical: 0.7,
            wide: false,
            max_depth: 12,
            nested: 0.9,
        },
        max_rows: 1200,
        boundaries: 0.7,
        repeats: 0.2,
    });
    if full_run() {
        assert!(
            c.chained > c.cases / 2 && c.long_chains > c.cases / 10 && c.bits > c.cases / 4,
            "{c:?}"
        );
    }
}

#[test]
fn past_the_old_tree_size_and_depth_limits() {
    // Trees of more than 32,767 nodes and deeper than 256.
    let c = run(&Profile {
        name: "past_the_old_tree_size_and_depth_limits",
        cases: 12,
        full_cases: 500,
        settings: Settings {
            max_trees: 2,
            max_nodes: 90_000,
            wide_leaves: 0.1,
            categorical: 0.3,
            wide: false,
            max_depth: 600,
            nested: 0.0,
        },
        max_rows: 600,
        boundaries: 0.5,
        repeats: 0.2,
    });
    if full_run() {
        assert!(
            c.trees_over_32767_nodes > 0 && c.depth_over_256 > 0,
            "{c:?}"
        );
    }
}
