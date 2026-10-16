use treewalker_gbdt::Forest;
use treewalker_gbdt::research::WorkCounters;

use super::counters::*;
use super::validate::*;
use crate::artifacts::VariantSpec;
use crate::external::Precision;
use crate::external::{TL2CGEN_F64_MISSING_ALIAS, tl2cgen_f64_missing_alias};

#[test]
fn tl2cgen_alias_is_found_from_the_input_alone() {
    // Low 32 bits 0xffffffff: tl2cgen's `missing != -1` test reads it as missing.
    assert_eq!(54.952_331_542_968_74_f64.to_bits() as u32, u32::MAX);
    assert!(tl2cgen_f64_missing_alias(54.952_331_542_968_74));
    assert!(!tl2cgen_f64_missing_alias(74.934_997_558_593_75));
    assert!(!tl2cgen_f64_missing_alias(f64::NAN));
}

#[test]
fn affected_rows_are_attributed_and_other_mismatches_still_fail() {
    let rows = [[1.0, 54.952_331_542_968_74], [1.0, 2.0], [3.0, 4.0]];
    let mask: Vec<bool> = rows
        .iter()
        .map(|r| r.iter().any(|&v| tl2cgen_f64_missing_alias(v)))
        .collect();
    assert_eq!(mask, [true, false, false]);
    let want = [0.5, 0.25, 0.75];
    let attributed = Some((TL2CGEN_F64_MISSING_ALIAS, mask.as_slice()));
    // Only the affected row differs: the comparison passes and records it.
    let (max_abs, bad, defect) =
        compare_attributed(&[0.7, 0.25, 0.75], &want, false, 1e-13, attributed);
    let defect = defect.unwrap();
    assert_eq!((max_abs, bad), (0.0, 0));
    assert_eq!((defect.affected_rows, defect.affected_differ), (1, 1));
    assert!((defect.affected_max_abs - 0.2).abs() < 1e-12);
    // An unaffected row that differs still fails.
    let (_, bad, _) = compare_attributed(&[0.5, 0.3, 0.75], &want, false, 1e-13, attributed);
    assert_eq!(bad, 1);
    // Without a defect every row counts.
    let (_, bad, defect) = compare_attributed(&[0.7, 0.25, 0.75], &want, false, 1e-13, None);
    assert_eq!((bad, defect.is_none()), (1, true));
}

#[test]
fn f64_sum_tolerance_scales_with_the_trees_and_leaves() {
    let f = fixture_forest(&[0]);
    let leaves: f64 = f
        .trees()
        .iter()
        .map(|t| {
            f.tree_nodes(t)
                .iter()
                .filter(|n| n.is_leaf())
                .map(|n| n.value.abs())
                .fold(0.0, f64::max)
        })
        .sum();
    let want = ((f.trees().len() as f64 + 1.0) * f64::EPSILON * (leaves + 2.0))
        .max(Precision::F64.tolerance());
    assert_eq!(f64_sum_tolerance(&f, -2.0).to_bits(), want.to_bits());
    // A small model keeps the fixed floor; a huge base score raises it.
    assert!(f64_sum_tolerance(&f, 0.0) >= Precision::F64.tolerance());
    assert!(f64_sum_tolerance(&f, 1e6) > 1e-11);
}

pub(super) fn fixture_forest(varying: &[usize]) -> Forest {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/import/identity.bin"
    );
    let config = treewalker_gbdt::WalkerConfig::builder(2)
        .max_group_width(128)
        .varying(varying.iter().copied())
        .build()
        .unwrap();
    Forest::from_bytes(
        &std::fs::read(path).unwrap(),
        treewalker_gbdt::ModelFormat::TreeliteBinaryV4,
        config,
        &treewalker_gbdt::LoadOptions::default(),
    )
    .unwrap()
}

fn facts(f: &Forest, rows: &[f64]) -> Facts {
    let n = rows.len() / 2;
    let stats = f.layout_stats();
    Facts {
        monotonic: false,
        varying_predicates: !f.predicate_masks(rows, true).is_empty(),
        lanes: stats.lanes,
        lanes_constant: stats.lanes_constant,
        distance: stats.distance,
        distance_constant: stats.distance_constant,
        rate_constant: stats.rate_constant,
        single_rows: n == 1,
        mask_widths: if n > 1 {
            vec![mask_width(n)]
        } else {
            Vec::new()
        },
        config_width: mask_width(f.config().max_group_width()),
    }
}

fn counted(f: &Forest, rows: &[f64], flags: &[&str]) -> WorkCounters {
    let spec = VariantSpec {
        runtime: flags.iter().map(ToString::to_string).collect(),
        load: Vec::new(),
    };
    let mut out = vec![0.0; rows.len() / 2];
    f.research_predictor(spec.ablation().unwrap())
        .predict_group_counted(rows, &mut out)
}

const PATH_FLAGS: [&str; 5] = [
    "disable_varying_precompute",
    "disable_unsplit",
    "disable_predicate_sweep",
    "disable_g1_constant",
    "lanes=1",
];

#[test]
fn single_row_groups_show_every_path() {
    let f = fixture_forest(&[0]);
    let rows = [0.5, 1.0];
    let facts = facts(&f, &rows);
    assert!(facts.varying_predicates);
    for flag in PATH_FLAGS {
        let spec = VariantSpec {
            runtime: vec![flag.into()],
            load: Vec::new(),
        };
        assert_eq!(no_op(flag, &spec, &facts), None, "{flag}");
        assert!(
            signature(flag, &counted(&f, &rows, &[flag]), &facts),
            "{flag}"
        );
    }
}

#[test]
fn two_identical_rows_show_every_path() {
    let f = fixture_forest(&[0]);
    let rows = [0.5, 1.0, 0.5, 1.0];
    let facts = facts(&f, &rows);
    for flag in PATH_FLAGS {
        assert!(
            signature(flag, &counted(&f, &rows, &[flag]), &facts),
            "{flag}"
        );
    }
    // One row stopping at varying splits leaves a side of every varying event
    // empty.
    let empty = counted(&f, &rows[..2], &["disable_g1_constant", "disable_unsplit"]);
    assert!(empty.empty_sides > 0);
}

#[test]
fn no_varying_predicates_make_the_path_flags_no_ops() {
    // The fixture splits both features; with neither varying there are no
    // varying predicates, as in a horizon-1 survival model.
    let f = fixture_forest(&[]);
    let rows = [0.5, 1.0];
    let facts = facts(&f, &rows);
    assert!(!facts.varying_predicates);
    for flag in [
        "disable_varying_precompute",
        "disable_unsplit",
        "disable_predicate_sweep",
    ] {
        let spec = VariantSpec {
            runtime: vec![flag.into()],
            load: Vec::new(),
        };
        assert!(no_op(flag, &spec, &facts).is_some(), "{flag}");
    }
}

#[test]
fn the_kernel_flags_are_no_ops_where_the_policy_already_chose_them() {
    let f = fixture_forest(&[0]);
    let spec = |flags: &[&str]| VariantSpec {
        runtime: flags.iter().map(ToString::to_string).collect(),
        load: Vec::new(),
    };
    let why = |facts: &Facts, flags: &[&str]| no_op(flags[0], &spec(flags), facts);
    assert_eq!(
        (mask_width(1), mask_width(32), mask_width(33)),
        (32, 32, 64)
    );
    assert_eq!(
        (
            mask_width(65),
            mask_width(1024),
            mask_width(1025),
            mask_width(2049)
        ),
        (128, 1024, 1152, 2176)
    );
    let policy = f.layout_stats();
    let lanes = format!("lanes={}", policy.lanes);
    // A single row: no masks unless its path is off.
    let one = facts(&f, &[0.5, 1.0]);
    // The fixture's nodes fit L1, so the rules prefetch nothing, and forced lanes
    // prefetch one item ahead: no lane count is a no-op.
    assert_eq!((one.distance, one.distance_constant), (0, 0));
    assert!(why(&one, &[&lanes]).is_none() && why(&one, &["lanes=1"]).is_none());
    let prefetching = Facts {
        distance: 1,
        distance_constant: 1,
        lanes_constant: one.lanes,
        ..one.clone()
    };
    assert!(why(&prefetching, &[&lanes]).is_some());
    // Forced lanes prefetch every round, so not at a rule's rate of 2.
    let every_other = Facts {
        rate_constant: 2,
        ..prefetching
    };
    assert!(why(&every_other, &[&lanes]).is_none());
    assert!(why(&one, &["disable_width_dispatch"]).is_some());
    assert!(why(&one, &["disable_g1_constant"]).is_none());
    // Off, it takes 32-bit masks, and the configured 128 rows take 128 bits.
    assert!(why(&one, &["disable_width_dispatch", "disable_g1_constant"]).is_none());
    // Two rows of a 128-row config: dispatch matters.
    let two = facts(&f, &[0.5, 1.0, 0.5, 1.0]);
    assert!(why(&two, &["disable_g1_constant"]).is_some());
    assert!(why(&two, &["disable_width_dispatch"]).is_none());
    assert!(why(&two, &["disable_restart_prefetch"]).is_none());
}

#[test]
fn matches_are_established_positively() {
    assert!(matches(1.0, 1.0 + 1e-14, false, 1e-13));
    assert!(!matches(f64::NAN, 1.0, false, 1e-13));
    assert!(!matches(1.0, f64::NAN, false, 1e-13));
    assert!(!matches(f64::INFINITY, 1e308, false, f64::INFINITY));
    assert!(matches(f64::INFINITY, f64::INFINITY, false, 1e-13));
    assert!(!matches(f64::INFINITY, f64::NEG_INFINITY, false, 1e-13));
    assert!(!matches(
        f64::NAN,
        f64::from_bits(f64::NAN.to_bits() ^ 1),
        true,
        0.0
    ));
    assert!(matches(f64::NAN, f64::NAN, true, 0.0));
    let (max_abs, bad) = compare(
        &[1.0, f64::NAN, 2.0],
        &[1.0, 1.0, f64::INFINITY],
        false,
        1e-13,
    );
    assert_eq!(bad, 2);
    assert!(max_abs.is_infinite());
}
