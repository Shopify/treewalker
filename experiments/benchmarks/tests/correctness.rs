//! The artifact suite: every prepared cell under `experiments/artifacts` (or
//! `TEST_ARTIFACTS_BASE`), loaded through the runner's artifacts module.
//!
//! The checks are a table run on every cell; each failure names its cell and
//! check, and the suite reports them all before failing. Edge cases that need a
//! particular cell (a fixed-width panel, categorical splits) take the first cell
//! that qualifies.

use std::path::{Path, PathBuf};

use treewalker_bench::artifacts::{self, Cell};
use treewalker_gbdt::research::{Ablation, Stages, ThresholdType, WorkCounters};
use treewalker_gbdt::{Forest, LoadError, LoadOptions, WalkerConfig};

/// Outputs of `f64` additions in tree order (the full walk, GTIL) against
/// production's correctly rounded sums.
const TOL_F64: f64 = 1e-13;
/// Native XGBoost adds its `f32` leaves in `f32`.
const TOL_F32: f64 = 1e-5;
/// The JSON loader's size limit (docs/treelite-loading.md).
const JSON_MAX_BYTES: u64 = 64 * 1024 * 1024;

fn base() -> PathBuf {
    std::env::var_os("TEST_ARTIFACTS_BASE").map_or_else(
        || PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../artifacts")),
        PathBuf::from,
    )
}

/// Every ready cell, loaded once per test.
fn cells() -> Vec<Cell> {
    let dirs = artifacts::discover(&base());
    assert!(
        !dirs.is_empty(),
        "no prepared cells under {}; run treewalker-exp prepare",
        base().display()
    );
    let mut out = Vec::new();
    for d in dirs {
        match Cell::load(&d) {
            Ok(c) if c.status == "ready" => out.push(c),
            Ok(c) => eprintln!("{}: status {}; skipped", c.id, c.status),
            Err(e) => panic!("{}: {e:#}", d.display()),
        }
    }
    out
}

fn forest(c: &Cell) -> Forest {
    c.forest(&LoadOptions::default()).unwrap()
}

fn predict(f: &Forest, c: &Cell) -> Vec<f64> {
    let mut out = vec![0.0; c.n_rows];
    f.predictor().predict_groups(&c.data, &c.offsets, &mut out);
    out
}

fn full_walk(f: &Forest, c: &Cell) -> Vec<f64> {
    let mut out = vec![0.0; c.n_rows];
    f.predict_full_walk(&c.data, &mut out);
    out
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn same_bits(a: &[f64], b: &[f64]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("{} values against {}", a.len(), b.len()));
    }
    (0..a.len())
        .find(|&r| a[r].to_bits() != b[r].to_bits())
        .map_or(Ok(()), |r| Err(format!("row {r}: {} != {}", a[r], b[r])))
}

/// Whether every group's declared monotonic features are monotonic (NaN aside).
fn honors_monotonic(c: &Cell, config: &WalkerConfig) -> bool {
    let nf = c.n_features;
    (0..c.n_groups()).all(|g| {
        let rows = c.group(g);
        (0..nf)
            .filter(|&f| config.is_increasing(f) || config.is_decreasing(f))
            .all(|f| {
                let v: Vec<f64> = rows
                    .chunks_exact(nf)
                    .map(|r| r[f])
                    .filter(|x| !x.is_nan())
                    .collect();
                v.windows(2).all(|w| {
                    if config.is_increasing(f) {
                        w[0] <= w[1]
                    } else {
                        w[0] >= w[1]
                    }
                })
            })
    })
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

type CheckFn = fn(&Cell, &Forest) -> Result<(), String>;

/// The stage oracle: `math.fsum` of GTIL's per-tree outputs and the staged
/// finalization, bit for bit when the model supports exact sums.
fn oracle(c: &Cell, f: &Forest) -> Result<(), String> {
    let Some(o) = &c.oracle else {
        return Err("no oracle; prepare writes one (PREP_POLICY 3)".into());
    };
    let mut r = f.research_predictor(Ablation::default());
    let mut stages = Stages::default();
    let mut want = o.rows.iter();
    for &g in &o.groups {
        r.predict_group_stages(c.group(g), &mut stages);
        for (i, (ts, rm)) in stages.tree_sum.iter().zip(&stages.raw_margin).enumerate() {
            let &(row, o_sum, o_margin) = want.next().ok_or("oracle has fewer rows")?;
            let ok = if f.exact_sums() {
                ts.to_bits() == o_sum.to_bits() && rm.to_bits() == o_margin.to_bits()
            } else {
                (ts - o_sum).abs() <= TOL_F64 * o_sum.abs().max(1.0)
            };
            if !ok {
                return Err(format!(
                    "row {row}: tree_sum {ts} vs {o_sum}, raw_margin {rm} vs {o_margin}"
                ));
            }
            // The link, as production applies it, to the margin.
            if o.link(*rm).to_bits() != stages.output[i].to_bits() {
                return Err(format!(
                    "row {row}: output {} vs link {}",
                    stages.output[i],
                    o.link(*rm)
                ));
            }
        }
    }
    Ok(())
}

/// Native XGBoost's reference within its f32 accumulation; LightGBM's GTIL
/// reference adds in tree order, so the oracle checks it instead.
fn native_reference(c: &Cell, f: &Forest) -> Result<(), String> {
    if f.threshold_type() != ThresholdType::F32 || c.framework != "xgboost" {
        return Ok(());
    }
    let Some(reference) = &c.reference else {
        return Ok(());
    };
    let d = max_diff(&predict(f, c), reference);
    (d < TOL_F32)
        .then_some(())
        .ok_or_else(|| format!("{d:.2e} from native XGBoost"))
}

fn partial_matches_full(c: &Cell, f: &Forest) -> Result<(), String> {
    let d = max_diff(&predict(f, c), &full_walk(f, c));
    (d < TOL_F64)
        .then_some(())
        .ok_or_else(|| format!("{d:.2e} from the full walk"))
}

fn single_row(c: &Cell, f: &Forest) -> Result<(), String> {
    let nf = c.n_features;
    let (mut one, mut walk) = ([0.0], [0.0]);
    f.predictor().predict_group(&c.data[..nf], &mut one);
    f.predict_full_walk(&c.data[..nf], &mut walk);
    let d = (one[0] - walk[0]).abs();
    (d < TOL_F64)
        .then_some(())
        .ok_or_else(|| format!("{d:.2e}"))
}

/// Every runtime flag but `disable_exact_sums` reaches the same leaves with the
/// same rows, so outputs are bit-identical to production.
fn runtime_ablations(c: &Cell, f: &Forest) -> Result<(), String> {
    let reference = predict(f, c);
    let monotonic = honors_monotonic(c, f.config());
    for bits in 0..16u8 {
        let a = Ablation {
            disable_varying_precompute: bits & 1 != 0,
            disable_predicate_sweep: bits & 2 != 0,
            disable_unsplit: bits & 4 != 0,
            disable_monotonic: bits & 8 != 0,
            disable_exact_sums: false,
        };
        // Monotonic scans are correct only when the data honor the contract.
        if a.disable_varying_precompute && !a.disable_monotonic && !monotonic {
            continue;
        }
        let mut out = vec![0.0; c.n_rows];
        f.research_predictor(a)
            .predict_groups(&c.data, &c.offsets, &mut out);
        same_bits(&out, &reference).map_err(|e| format!("{a:?}: {e}"))?;
    }
    // Without exact sums, leaves add in f64 in tree order, as the full walk does.
    let mut out = vec![0.0; c.n_rows];
    f.research_predictor(Ablation {
        disable_exact_sums: true,
        ..Default::default()
    })
    .predict_groups(&c.data, &c.offsets, &mut out);
    same_bits(&out, &full_walk(f, c)).map_err(|e| format!("disable_exact_sums: {e}"))
}

/// With exact sums, tree order and node layout cannot change a prediction.
fn load_options(c: &Cell, f: &Forest) -> Result<(), String> {
    let reference = predict(f, c);
    for o in [
        LoadOptions {
            disable_tree_ordering: true,
            ..Default::default()
        },
        LoadOptions {
            disable_bitset_intern: true,
            ..Default::default()
        },
        LoadOptions {
            prefix_depth: 0,
            ..Default::default()
        },
        LoadOptions {
            disable_tree_ordering: true,
            disable_bitset_intern: true,
            prefix_depth: 0,
            ..Default::default()
        },
    ] {
        let g = c.forest(&o).map_err(|e| format!("{o:?}: {e}"))?;
        let out = predict(&g, c);
        if f.exact_sums() {
            same_bits(&out, &reference).map_err(|e| format!("{o:?}: {e}"))?;
        } else if max_diff(&out, &full_walk(&g, c)) >= TOL_F64 {
            return Err(format!("{o:?}: differs from its full walk"));
        }
    }
    Ok(())
}

/// Number of `"threshold": ,` entries, found without the parser under test.
fn empty_thresholds(path: &Path) -> usize {
    const KEY: &[u8] = b"\"threshold\":";
    let bytes = std::fs::read(path).unwrap();
    (0..bytes.len().saturating_sub(KEY.len()))
        .filter(|&i| bytes[i..].starts_with(KEY))
        .filter(|&i| {
            bytes[i + KEY.len()..]
                .iter()
                .find(|b| !b.is_ascii_whitespace())
                .is_some_and(|&b| b == b',' || b == b'}')
        })
        .count()
}

/// The JSON dump predicts bit for bit as the binary export. Treelite writes empty
/// thresholds for nonfinite values; such dumps are rejected, not repaired, and
/// skipped with a message (docs/treelite-loading.md).
fn json_matches_binary(c: &Cell, f: &Forest) -> Result<(), String> {
    let Some(json) = c.model_json() else {
        return Ok(());
    };
    if std::fs::metadata(&json).map_or(0, |m| m.len()) > JSON_MAX_BYTES {
        return Ok(());
    }
    let g = match Forest::load(&json, &c.walker_config) {
        Ok(g) => g,
        Err(e) => {
            let empty = empty_thresholds(&json);
            if empty > 0 && matches!(&e, LoadError::MalformedModel(m) if m.starts_with("JSON: ")) {
                eprintln!(
                    "{}: {empty} empty thresholds ({e}); skipped",
                    json.display()
                );
                return Ok(());
            }
            return Err(format!("{}: {e}", json.display()));
        }
    };
    if g.trees().len() != f.trees().len() {
        return Err("tree counts differ".into());
    }
    same_bits(&predict(&g, c), &predict(f, c))
}

/// The sorted-threshold sweep gives the brute force's masks for groups of up to
/// 32 rows (the helper's u32 masks).
fn sweep_matches_bruteforce(c: &Cell, f: &Forest) -> Result<(), String> {
    for g in (0..c.n_groups()).filter(|&g| c.offsets[g + 1] - c.offsets[g] <= 32) {
        let rows = c.group(g);
        if f.predicate_masks(rows, true) != f.predicate_masks(rows, false) {
            return Err(format!("group {g}"));
        }
    }
    Ok(())
}

const TABLE: &[(&str, CheckFn)] = &[
    ("oracle", oracle),
    ("native_reference", native_reference),
    ("partial_matches_full", partial_matches_full),
    ("single_row", single_row),
    ("runtime_ablations", runtime_ablations),
    ("load_options", load_options),
    ("json_matches_binary", json_matches_binary),
    ("sweep_matches_bruteforce", sweep_matches_bruteforce),
];

#[test]
fn every_check_on_every_cell() {
    let mut failures = Vec::new();
    let cells = cells();
    for c in &cells {
        let f = forest(c);
        for (name, check) in TABLE {
            if let Err(e) = check(c, &f) {
                failures.push(format!("{} / {name}: {e}", c.id));
            }
        }
        eprintln!("{}: {} checks", c.id, TABLE.len());
    }
    assert!(
        failures.is_empty(),
        "{} of {} cells failed:\n{}",
        failures.len(),
        cells.len(),
        failures.join("\n")
    );
}

/// The released credit scenario model (`artifacts/scenario_credit/lightgbm`) is
/// one model shared by sixteen cells outside the standard layout, so its JSON dump
/// gets its own comparison against the binary export and the GTIL reference.
#[test]
fn scenario_credit_json_matches_binary() {
    let dirs: Vec<PathBuf> = artifacts::discover(&base().join("scenario_credit"));
    if dirs.is_empty() {
        eprintln!("no scenario_credit cells; skipped (prepare --suite scenario-v1)");
        return;
    }
    for d in dirs {
        let c = Cell::load(&d).unwrap();
        let json = c.model_json().expect("scenario_credit has a JSON dump");
        let from_json = Forest::load(&json, &c.walker_config).unwrap();
        same_bits(&predict(&from_json, &c), &predict(&forest(&c), &c)).unwrap();
        let reference = treewalker_bench::load_raw_f64(d.join("reference.bin")).0;
        let diff = max_diff(&predict(&from_json, &c), &reference);
        assert!(diff < TOL_F64, "{}: {diff:.2e} from GTIL", c.id);
    }
}

// ---------------------------------------------------------------------------
// Edge cases on one qualifying cell
// ---------------------------------------------------------------------------

/// A fixed-width panel with more than one row per group.
fn panel() -> Option<Cell> {
    cells().into_iter().find(|c| {
        c.doc["generator"] == "panel-v2"
            && c.n_groups() > 1
            && c.offsets[1] > 1
            && c.offsets.windows(2).all(|w| w[1] - w[0] == c.offsets[1])
    })
}

fn partial_matches_full_on_first_group(f: &Forest, rows: &[f64]) {
    let n = rows.len() / f.config().n_features();
    let (mut full, mut partial) = (vec![0.0; n], vec![0.0; n]);
    f.predict_full_walk(rows, &mut full);
    f.predictor().predict_group(rows, &mut partial);
    let d = max_diff(&full, &partial);
    assert!(d < TOL_F64, "partial vs full {d:.2e}");
}

/// Rewrite the first group's features matching `pick` to `value`, then compare.
fn edited_first_group(pick: fn(&WalkerConfig, usize) -> bool, value: f64) {
    let Some(c) = panel() else {
        eprintln!("no panel cell; skipped");
        return;
    };
    let f = forest(&c);
    let nf = c.n_features;
    let mut rows = c.group(0).to_vec();
    for r in rows.chunks_exact_mut(nf) {
        for (i, v) in r.iter_mut().enumerate() {
            if pick(f.config(), i) {
                *v = value;
            }
        }
    }
    partial_matches_full_on_first_group(&f, &rows);
}

#[test]
fn all_nan_varying_features() {
    edited_first_group(WalkerConfig::is_varying, f64::NAN);
}

#[test]
fn monotonic_columns_with_ties() {
    edited_first_group(|c, f| c.is_increasing(f) || c.is_decreasing(f), 5.0);
}

#[test]
fn all_nan_monotonic_features() {
    edited_first_group(|c, f| c.is_increasing(f) || c.is_decreasing(f), f64::NAN);
}

#[test]
fn extreme_constant_features() {
    for v in [f64::MIN, f64::MAX, 0.0, -0.0, 1e300, -1e300] {
        let Some(c) = panel() else {
            return;
        };
        let f = forest(&c);
        let nf = c.n_features;
        let mut rows = c.group(0).to_vec();
        for r in rows.chunks_exact_mut(nf) {
            for (i, x) in r.iter_mut().enumerate() {
                if !f.config().is_varying(i) {
                    *x = v;
                }
            }
        }
        partial_matches_full_on_first_group(&f, &rows);
    }
}

/// Out-of-range categories follow the default direction in both paths.
#[test]
fn categories_out_of_range() {
    let cell = cells().into_iter().find(|c| {
        c.framework == "xgboost"
            && forest(c)
                .nodes()
                .iter()
                .any(|n| n.is_categorical() && !n.is_leaf())
    });
    let Some(c) = cell else {
        eprintln!("no XGBoost cell with categorical splits; skipped");
        return;
    };
    let f = forest(&c);
    let feature = f
        .nodes()
        .iter()
        .find(|n| n.is_categorical() && !n.is_leaf())
        .map(|n| n.feature as usize)
        .unwrap();
    let mut rows = c.group(0).to_vec();
    for r in rows.chunks_exact_mut(c.n_features) {
        r[feature] = 50.0;
    }
    partial_matches_full_on_first_group(&f, &rows);
}

/// Synthetic groups of `width` rows: the first source group's constant features,
/// varying features cycled from the data. Exercises the mask width chosen for
/// `width`, and the 1,024-row pieces above it.
fn wide_groups(width: usize) {
    let Some(c) = panel() else {
        eprintln!("no panel cell; skipped");
        return;
    };
    let f = c.forest_at_width(width).unwrap();
    let nf = c.n_features;
    let mut data = Vec::new();
    for g in 0..c.n_groups().min(10) {
        let constant = &c.group(g)[..nf];
        for r in 0..width {
            let src = (c.offsets[g] + r) % c.n_rows;
            let src = &c.data[src * nf..(src + 1) * nf];
            data.extend((0..nf).map(|i| {
                if f.config().is_varying(i) {
                    src[i]
                } else {
                    constant[i]
                }
            }));
        }
    }
    let n = data.len() / nf;
    let (mut partial, mut full) = (vec![0.0; n], vec![0.0; n]);
    f.predictor().predict_fixed(&data, width, &mut partial);
    f.predict_full_walk(&data, &mut full);
    let d = max_diff(&partial, &full);
    assert!(d < TOL_F64, "width {width}: {d:.2e}");
}

#[test]
fn wide_groups_at_every_mask_and_piece_transition() {
    for width in [
        16, 17, 32, 33, 48, 64, 65, 96, 128, 129, 200, 1000, 1024, 1025, 2500,
    ] {
        wide_groups(width);
    }
}

#[test]
fn counters_record_work() {
    let Some(c) = panel() else {
        return;
    };
    let f = forest(&c);
    let mut r = f.research_predictor(Ablation::default());
    let mut total = WorkCounters::default();
    let mut out = vec![0.0; c.max_group_rows()];
    for g in 0..c.n_groups() {
        let rows = c.group(g);
        total += r.predict_group_counted(rows, &mut out[..rows.len() / c.n_features]);
    }
    assert!(total.constant_steps > 0 && total.leaf_hits > 0 && total.leaf_adds > 0);
}
