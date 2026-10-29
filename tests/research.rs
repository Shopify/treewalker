//! The research API: every ablation computes production's outputs, counted and timed
//! builds agree, the counters count the work each ablation changes, and the stages
//! compose to production's output.
#![expect(clippy::float_cmp, reason = "exact hand-computable expectations")]

mod common;

use common::bits::{bits, same};
use common::fixtures::{bytes, values};
use simd_json::{OwnedValue as Value, prelude::*};
use treewalker_gbdt::research::{
    Ablation, COUNTERS_VERSION, LoadAblation, Phases, Stages, WorkCounters,
};
use treewalker_gbdt::{Forest, LoadOptions, ModelFormat, WalkerConfig};

fn config() -> WalkerConfig {
    WalkerConfig::builder(2)
        .max_group_width(128)
        .varying([0])
        .build()
        .unwrap()
}
fn load(name: &str, parse: &LoadOptions) -> Forest {
    Forest::from_bytes(&bytes(name), ModelFormat::TreeliteBinaryV4, config(), parse).unwrap()
}
/// Every combination of the seven runtime flags, each with a lane count drawn from
/// its index.
fn variants() -> impl Iterator<Item = Ablation> {
    (0..128u32).map(|b| Ablation {
        disable_predicate_sweep: b & 1 != 0,
        disable_varying_precompute: b & 2 != 0,
        disable_monotonic: b & 4 != 0,
        disable_unsplit: b & 8 != 0,
        disable_g1_constant: b & 16 != 0,
        disable_width_dispatch: b & 32 != 0,
        disable_restart_prefetch: b & 64 != 0,
        lanes: [None, Some(1), Some(2), Some(3), Some(4), Some(8), Some(12)][(b % 7) as usize],
    })
}
const MODELS: [&str; 6] = [
    "sigmoid_f64.bin",
    "sigmoid_f32.bin",
    "identity.bin",
    "average_base.bin",
    "categories_left.bin",
    "random_forest.bin",
];

/// Three trees that send `x0 <= 1` to leaves `big`, 1 and `-big`, in that order. The
/// exact sum is 1; adding in f64 in tree order loses the 1 and gives 0.
fn cancellation(big: f64) -> Forest {
    let mut model = simd_json::to_owned_value(&mut bytes("identity.json")).unwrap();
    let tree = model["trees"][0].clone();
    let trees: Vec<Value> = [big, 1.0, -big]
        .into_iter()
        .map(|left| {
            let mut t = tree.clone();
            t["nodes"][1]["leaf_value"] = left.into();
            t["nodes"][2]["leaf_value"] = 0.0.into();
            t
        })
        .collect();
    model["trees"] = Value::Array(Box::new(trees));
    model["target_id"] = Value::Array(Box::new(vec![0.into(); 3]));
    model["class_id"] = Value::Array(Box::new(vec![0.into(); 3]));
    model["base_scores"] = Value::Array(Box::new(vec![0.0.into()]));
    // Keep the tree order, which decides the f64 sum.
    let ablation = LoadAblation {
        disable_tree_ordering: true,
        ..Default::default()
    };
    Forest::from_bytes_ablated(
        &simd_json::to_vec(&model).unwrap(),
        ModelFormat::TreeliteJson,
        config(),
        &LoadOptions::default(),
        &ablation,
    )
    .unwrap()
}

#[test]
fn every_variant_matches_production_and_the_full_walk() {
    let data = values("data.bin");
    let n = data.len() / 2;
    // The rows come in blocks of 128 that share their constant feature. Groups of 1
    // to 128 rows within the blocks give single rows and every width up to 128 bits.
    let splits: [&[usize]; 3] = [&[1, 2, 17, 33, 64, 11], &[65, 63], &[128]];
    let mut offsets = vec![0];
    for b in 0..n / 128 {
        for &size in splits[b % 3] {
            offsets.push(offsets.last().unwrap() + size);
        }
    }
    for name in MODELS {
        let forest = load(name, &LoadOptions::default());
        let mut production = vec![0.0; n];
        forest
            .predictor()
            .predict_groups(&data, &offsets, &mut production);
        let mut full = vec![0.0; n];
        forest.predict_full_walk(&data, &mut full);
        same(&full, &production, &format!("{name} full walk"));
        for variant in variants() {
            let mut r = forest.research_predictor(variant);
            let mut batch = vec![f64::NAN; n];
            r.predict_groups(&data, &offsets, &mut batch);
            same(&batch, &production, &format!("{name} batch {variant:?}"));
            let (mut timed, mut counted) = (vec![f64::NAN; n], vec![f64::NAN; n]);
            for w in offsets.windows(2) {
                let (a, b) = (w[0], w[1]);
                r.predict_group(&data[2 * a..2 * b], &mut timed[a..b]);
                r.predict_group_counted(&data[2 * a..2 * b], &mut counted[a..b]);
            }
            same(&timed, &production, &format!("{name} timed {variant:?}"));
            same(
                &counted,
                &production,
                &format!("{name} counted {variant:?}"),
            );
            let mut stages = Stages::default();
            r.predict_group_stages(&data[..256], &mut stages);
            same(
                &stages.output,
                &production[..128],
                &format!("{name} {variant:?}"),
            );
        }
    }
}

#[test]
fn ablation_ids_round_trip() {
    for variant in variants() {
        assert_eq!(
            Ablation::from_id(&variant.id()),
            Ok(variant),
            "{}",
            variant.id()
        );
    }
    assert_eq!(Ablation::default().id(), "all-on");
    let a = Ablation::default()
        .with("disable_unsplit")
        .and_then(|a| a.with("lanes=2"))
        .unwrap();
    assert_eq!(a.id(), "disable_unsplit+lanes=2");
    assert_eq!(a.tokens(), ["disable_unsplit", "lanes=2"]);
    for bad in [
        "disable_exact_sums",
        "lanes=13",
        "lanes=0",
        "lanes=x",
        "distance=4",
        "rate=2",
        "layout=2e",
        "",
    ] {
        assert!(Ablation::from_id(bad).is_err(), "{bad}");
    }
}

#[test]
#[should_panic(expected = "lanes must be 1 to 12")]
fn research_predictors_take_1_to_12_lanes() {
    let forest = load("identity.bin", &LoadOptions::default());
    let _ = forest.research_predictor(Ablation {
        lanes: Some(13),
        ..Default::default()
    });
}

fn counted(forest: &Forest, variant: Ablation, rows: &[f64]) -> WorkCounters {
    let mut out = vec![0.0; rows.len() / 2];
    forest
        .research_predictor(variant)
        .predict_group_counted(rows, &mut out)
}

#[test]
fn counters_count_the_work_of_a_group() {
    assert_eq!(COUNTERS_VERSION, 6);
    let data = values("data.bin");
    let rows = &data[..64];
    let forest = load("sigmoid_f64.bin", &LoadOptions::default());
    let c = counted(&forest, Ablation::default(), rows);
    assert_eq!((c.groups, c.bins_computed, c.single_rows), (1, 1, 0));
    // The sweep reads each row once for the one varying feature.
    assert_eq!(c.precompute_row_evals, 32);
    assert!(c.precompute_sort_compares >= 31 && c.precompute_sweep_compares > 0);
    assert_eq!(
        c.precompute_mask_writes as usize,
        forest_predicates(&forest)
    );
    assert_eq!((c.scan_rows, c.scan_evals), (0, 0));
    // Every tree reaches at least one leaf, and each split adds one item.
    let trees = forest.trees().len() as u64;
    assert!(c.passes >= 1 && c.walk_steps > 0 && c.lane_slots >= c.walk_steps);
    assert!(c.splits <= c.varying_events);
    assert_eq!(c.leaf_events, trees + c.splits);
    assert!(c.runs_written >= c.leaf_events);
    assert_eq!((c.empty_sides, c.arena_slots), (0, 0));
    // Two groups count twice the work of one.
    let mut r = forest.research_predictor(Ablation::default());
    let (mut a, mut b) = (vec![0.0; 32], vec![0.0; 32]);
    let mut twice = r.predict_group_counted(rows, &mut a);
    twice += r.predict_group_counted(rows, &mut b);
    assert_eq!(twice.groups, 2);
    assert_eq!(twice.walk_steps, 2 * c.walk_steps);
    let phases = r.take_phases();
    assert!(phases.walk + phases.resolve + phases.leaf_pass > std::time::Duration::ZERO);
    assert_eq!(r.take_phases(), Phases::default());
    assert!(
        c.to_string()
            .starts_with("groups=1, single_rows=0, bins_computed=1")
    );
    assert_eq!(c.fields().len(), 19);
}

fn forest_predicates(forest: &Forest) -> usize {
    let masks = forest.predicate_masks(&[0.0, 0.0], true);
    masks.len()
}

#[test]
fn counters_show_each_ablation() {
    let data = values("data.bin");
    let rows = &data[..64];
    let forest = load("sigmoid_f64.bin", &LoadOptions::default());
    let base = counted(&forest, Ablation::default(), rows);
    let with = |token: &str| counted(&forest, Ablation::default().with(token).unwrap(), rows);
    // The brute force evaluates every predicate on every row, without sorting.
    let brute = with("disable_predicate_sweep");
    assert_eq!(brute.precompute_mask_writes, base.precompute_mask_writes);
    assert_eq!(
        brute.precompute_row_evals,
        32 * brute.precompute_mask_writes
    );
    assert_eq!(
        (
            brute.precompute_sort_compares,
            brute.precompute_sweep_compares
        ),
        (0, 0)
    );
    // Without precompute, varying events scan their rows.
    let scans = with("disable_varying_precompute");
    assert_eq!(scans.precompute_row_evals + scans.precompute_mask_writes, 0);
    assert!(scans.scan_rows > 0);
    assert_eq!(scans.scan_evals, scans.scan_rows, "no monotonic feature");
    assert_eq!(scans.walk_steps, base.walk_steps);
    // Without the unsplit shortcut, empty sides become items the walk steps through.
    // One row stopping at varying splits leaves one side of every varying event
    // empty.
    let one_row = Ablation::from_id("disable_g1_constant").unwrap();
    let single = counted(&forest, one_row, &data[..2]);
    let empty = counted(
        &forest,
        one_row.with("disable_unsplit").unwrap(),
        &data[..2],
    );
    assert!(single.varying_events > 0 && single.splits == 0);
    assert_eq!(empty.empty_sides, single.varying_events);
    assert!(empty.walk_steps > single.walk_steps);
    assert_eq!(empty.runs_written, single.runs_written);
    // Lanes change how the walk runs, not what it visits.
    for token in ["lanes=1", "lanes=2", "lanes=4", "lanes=8"] {
        let c = with(token);
        assert_eq!(c.walk_steps, base.walk_steps, "{token}");
        assert_eq!(c.leaf_events, base.leaf_events, "{token}");
        assert_eq!(c.varying_events, base.varying_events, "{token}");
        if let Some(n) = token.strip_prefix("lanes=") {
            assert_eq!(c.lane_slots % n.parse::<u64>().unwrap(), 0, "{token}");
        }
    }
    assert_eq!(with("disable_restart_prefetch"), base);
    // A 32-row group fits 32-bit masks; at the configured 128 rows it takes an arena.
    let widest = with("disable_width_dispatch");
    assert!(widest.arena_slots > 0 && base.arena_slots == 0);
    assert_eq!(widest.walk_steps, base.walk_steps);
    // A single row steps through varying splits as constants, without masks: one
    // pass reaches a leaf of every tree.
    let trees = forest.trees().len() as u64;
    let one = counted(&forest, Ablation::default(), &data[..2]);
    assert_eq!((one.single_rows, one.groups, one.passes), (1, 1, 1));
    assert_eq!((one.precompute_mask_writes, one.varying_events), (0, 0));
    assert_eq!((one.leaf_events, one.runs_written), (trees, trees));
    let one_masked = counted(
        &forest,
        Ablation::default().with("disable_g1_constant").unwrap(),
        &data[..2],
    );
    assert_eq!((one_masked.single_rows, one_masked.groups), (0, 1));
    assert!(one_masked.varying_events > 0 && one_masked.precompute_mask_writes > 0);
    assert_eq!(one_masked.leaf_events, trees);
    assert_eq!(one_masked.walk_steps, one.walk_steps);
}

#[test]
fn monotonic_scans_stop_comparing_early() {
    // Feature 0 declared increasing, rows sorted on it: a scan compares until the
    // first row past the threshold, unless disable_monotonic.
    let config = WalkerConfig::builder(2)
        .max_group_width(128)
        .varying([])
        .increasing([0])
        .build()
        .unwrap();
    let forest = Forest::from_bytes(
        &bytes("sigmoid_f64.bin"),
        ModelFormat::TreeliteBinaryV4,
        config,
        &LoadOptions::default(),
    )
    .unwrap();
    let mut rows: Vec<[f64; 2]> = values("data.bin")
        .chunks_exact(2)
        .take(64)
        .map(|r| [r[0], 0.25])
        .collect();
    rows.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let rows: Vec<f64> = rows.concat();
    let scan = counted(
        &forest,
        Ablation::default()
            .with("disable_varying_precompute")
            .unwrap(),
        &rows,
    );
    let full = counted(
        &forest,
        Ablation::from_id("disable_monotonic+disable_varying_precompute").unwrap(),
        &rows,
    );
    assert_eq!(scan.scan_rows, full.scan_rows);
    assert!(scan.scan_evals < full.scan_evals);
    assert_eq!(full.scan_evals, full.scan_rows);
    let mut a = vec![0.0; 64];
    let mut b = vec![0.0; 64];
    forest.predictor().predict_group(&rows, &mut a);
    forest
        .research_predictor(
            Ablation::default()
                .with("disable_varying_precompute")
                .unwrap(),
        )
        .predict_group(&rows, &mut b);
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn monotonic_promises_do_not_change_outputs() {
    // Feature 0 declared increasing or decreasing: rows in the promised order or out
    // of it predict as without the promise, in production and every precompute.
    let load_as = |increasing: bool, decreasing: bool| {
        let builder = WalkerConfig::builder(2).max_group_width(128).varying([0]);
        let builder = if increasing {
            builder.increasing([0])
        } else if decreasing {
            builder.decreasing([0])
        } else {
            builder
        };
        Forest::from_bytes(
            &bytes("sigmoid_f64.bin"),
            ModelFormat::TreeliteBinaryV4,
            builder.build().unwrap(),
            &LoadOptions::default(),
        )
        .unwrap()
    };
    let mut rows: Vec<[f64; 2]> = values("data.bin")
        .chunks_exact(2)
        .take(64)
        .map(|r| [r[0], 0.25])
        .collect();
    rows.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let ascending: Vec<f64> = rows.concat();
    rows.reverse();
    let descending: Vec<f64> = rows.concat();
    let outputs = |forest: &Forest, variant: &str, rows: &[f64]| {
        let mut out = vec![0.0; 64];
        forest
            .research_predictor(Ablation::from_id(variant).unwrap())
            .predict_group(rows, &mut out);
        bits(&out)
    };
    let unordered = load_as(false, false);
    for forest in [load_as(true, false), load_as(false, true)] {
        for rows in [&ascending, &descending] {
            for variant in ["all-on", "disable_predicate_sweep"] {
                assert_eq!(
                    outputs(&forest, variant, rows),
                    outputs(&unordered, variant, rows),
                    "{variant}"
                );
            }
        }
    }
}

#[test]
fn layout_stats_describe_the_kernel_choices() {
    let forest = load("sigmoid_f64.bin", &LoadOptions::default());
    let s = forest.layout_stats();
    assert!((0.0..=1.0).contains(&s.varying_share));
    assert!((1..=12).contains(&s.lanes) && (1..=12).contains(&s.lanes_constant));
    assert!(s.registers >= 14 && s.l1_bytes > 0 && s.l2_share_bytes > 0);
    assert!(s.line_bytes.is_power_of_two() && s.chunk_bytes == s.l2_share_bytes);
    assert!(s.steps_leaf >= s.steps_stop && s.steps_stop >= 1.0);
    // A tiny model's nodes fit L1: no prefetch.
    assert_eq!((s.pool_level, s.distance, s.distance_constant), (1, 0, 0));
    assert!(s.prefix_trees >= 2 * s.prefix_groups && s.prefix_saved_steps >= s.prefix_cost_steps);
    assert_eq!(s.pool_bytes, 16 * (forest.nodes().len() + 1));
    let leaves = forest.nodes().iter().filter(|n| n.is_leaf()).count();
    assert_eq!(s.leaf_bytes, 16 * leaves);
    assert_eq!((s.rank_table_bytes, s.chain_slots, s.chain_sets), (0, 0, 0));
    assert_eq!(s.flip_table_bytes, 0);
    // Two categorical features, each with one split: a chain of one set apiece,
    // each with a default rank and 5 exception bounds (for 2 categories, no
    // category and missing, and one past), and an exception per listed code: 1
    // and 40 with missing sent left, then 1 and 2.
    let categories = load("categories_left.bin", &LoadOptions::default());
    let s = categories.layout_stats();
    assert_eq!(
        (s.rank_table_bytes, s.chain_slots, s.chain_sets),
        (2 * (4 + 4 * 5) + 8 * (3 + 2), 2, 2)
    );
}

#[test]
fn the_full_walk_sums_exactly_past_the_126_bit_rule() {
    // Leaves 2^200 and 1 leave no 126-bit fixed point. Production and the full walk
    // sum in the wide accumulator, so neither loses the 1.
    let forest = cancellation(2f64.powi(200));
    // 201 bits of span: seven 29-bit digits in the wide accumulator.
    assert_eq!(forest.layout_stats().wide_limbs, 7);
    let rows = [0.5, 0.0, 2.0, 0.0];
    let mut production = [f64::NAN; 2];
    forest.predictor().predict_group(&rows, &mut production);
    assert_eq!(production, [1.0, 0.0]);
    let mut full = [f64::NAN; 2];
    forest.predict_full_walk(&rows, &mut full);
    assert_eq!(full, [1.0, 0.0]);
}

#[test]
fn stages_compose_to_the_output() {
    let data = values("data.bin");
    for name in MODELS {
        let forest = load(name, &LoadOptions::default());
        let mut production = vec![0.0; 128];
        forest
            .predictor()
            .predict_group(&data[..256], &mut production);
        let mut stages = Stages::default();
        let mut r = forest.research_predictor(Ablation::default());
        r.predict_group_stages(&data[..256], &mut stages);
        assert_eq!(bits(&stages.output), bits(&production), "{name}");
        assert_eq!(stages.tree_sum.len(), 128);
        assert_eq!(stages.raw_margin.len(), 128);
    }
    // average_base averages two trees and adds a base score of 3: tree sums are
    // halved, then shifted, each step rounded on its own.
    let forest = load("average_base.bin", &LoadOptions::default());
    let mut stages = Stages::default();
    let mut r = forest.research_predictor(Ablation::default());
    r.predict_group_stages(&[0.0, 0.0], &mut stages);
    assert_eq!(stages.raw_margin, [stages.tree_sum[0] / 2.0 + 3.0]);
    assert_eq!(stages.output, stages.raw_margin);
    assert_eq!(stages.output, [3.375]);
    // A sigmoid applies the link to the raw margin.
    let forest = load("sigmoid_f64.bin", &LoadOptions::default());
    let mut r = forest.research_predictor(Ablation::default());
    r.predict_group_stages(&data[..256], &mut stages);
    for (&m, &o) in stages.raw_margin.iter().zip(&stages.output) {
        assert_eq!(o, 1.0 / (1.0 + (-m).exp()));
    }
}

#[test]
fn predicate_masks_from_the_sweep_match_the_brute_force() {
    let data = values("data.bin");
    for name in MODELS {
        let forest = load(name, &LoadOptions::default());
        for rows in [1, 7, 32] {
            let group = &data[..rows * 2];
            assert_eq!(
                forest.predicate_masks(group, true),
                forest.predicate_masks(group, false),
                "{name} {rows}"
            );
        }
    }
}

#[test]
fn staged_calls_check_their_input_before_resizing() {
    let forest = load("sigmoid_f64.bin", &LoadOptions::default());
    let mut r = forest.research_predictor(Ablation::default());
    let mut stages = Stages {
        tree_sum: vec![f64::NAN; 4],
        ..Default::default()
    };
    let rows = vec![0.0; 129 * 2];
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        r.predict_group_stages(&rows, &mut stages);
    }));
    assert!(err.is_err(), "129 rows exceed max_group_width 128");
    assert_eq!(stages.tree_sum.len(), 4);
    assert!(stages.tree_sum.iter().all(|v| v.is_nan()));
}
