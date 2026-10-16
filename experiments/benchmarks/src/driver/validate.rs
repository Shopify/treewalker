//! Validation before timing: every method's outputs against the references and the
//! stage oracle.

use anyhow::{Context as _, Result};
use serde_json::{Value, json};
use treewalker_gbdt::Forest;
use treewalker_gbdt::research::{Ablation, Stages};

use crate::artifacts::Cell;
use crate::external::Precision;
use crate::methods::TimedMethod;
use crate::output::{Table, V};

use super::Excluded;
use super::load::{Gathered, Loaded, calls_for};
use super::timing::Mode;

/// The result of comparing one method's outputs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Check {
    pub method: String,
    pub variant: String,
    /// The multi-row interface checked; one-row serving calls use the single-row one.
    pub interface: String,
    pub against: String,
    pub mode: String,
    pub bitwise: bool,
    pub tolerance: f64,
    pub max_abs: f64,
    pub mismatches: usize,
    pub rows: usize,
    pub passed: bool,
    /// A known library defect the comparison attributes, when there is one: the
    /// unaffected rows are checked as usual, the affected ones recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defect: Option<Defect>,
    /// Why the call failed, when it did.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Rows a known library defect affects, and how they compare.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Defect {
    pub name: String,
    /// Rows the defect can affect, found from their inputs alone.
    pub affected_rows: usize,
    /// Affected rows whose outputs differ beyond the tolerance.
    pub affected_differ: usize,
    pub affected_max_abs: f64,
}

/// Compare outputs, attributing rows that `affected` marks to a known defect: the
/// other rows must pass as usual. Returns `(max_abs, mismatches, defect)`.
pub fn compare_attributed(
    got: &[f64],
    want: &[f64],
    bitwise: bool,
    tol: f64,
    affected: Option<(&str, &[bool])>,
) -> (f64, usize, Option<Defect>) {
    let Some((name, mask)) = affected else {
        let (max_abs, bad) = compare(got, want, bitwise, tol);
        return (max_abs, bad, None);
    };
    let pick = |hit: bool| -> (Vec<f64>, Vec<f64>) {
        (0..got.len().min(want.len()))
            .filter(|&r| mask[r] == hit)
            .map(|r| (got[r], want[r]))
            .unzip()
    };
    let (g, w) = pick(false);
    let (max_abs, bad) = compare(&g, &w, bitwise, tol);
    let (g, w) = pick(true);
    let (affected_max_abs, affected_differ) = compare(&g, &w, bitwise, tol);
    let defect = Defect {
        name: name.to_string(),
        affected_rows: g.len(),
        affected_differ,
        affected_max_abs,
    };
    (max_abs, bad + got.len().abs_diff(want.len()), Some(defect))
}

/// Whether `got` matches `want`: bit for bit, or within `tol`.
///
/// A match must be established positively: a NaN difference is a mismatch, and a nonfinite value
/// on either side matches only an identical bit pattern.
pub fn matches(got: f64, want: f64, bitwise: bool, tol: f64) -> bool {
    if bitwise || !got.is_finite() || !want.is_finite() {
        return got.to_bits() == want.to_bits();
    }
    (got - want).abs() <= tol
}

pub(super) fn compare(a: &[f64], b: &[f64], bitwise: bool, tol: f64) -> (f64, usize) {
    let mut max_abs = 0.0f64;
    let mut bad = 0;
    for (&x, &y) in a.iter().zip(b) {
        let ok = matches(x, y, bitwise, tol);
        let d = if x.to_bits() == y.to_bits() {
            0.0
        } else {
            (x - y).abs()
        };
        // A NaN difference counts as the largest.
        if d.is_nan() || d > max_abs {
            max_abs = if d.is_nan() { f64::INFINITY } else { d };
        }
        bad += usize::from(!ok);
    }
    (max_abs, bad + a.len().abs_diff(b.len()))
}

/// Outputs of one method over the validation groups, through its calls in `mode`:
/// one call per group, or the batch calls.
pub(super) fn outputs(
    method: &mut TimedMethod,
    groups: &Gathered,
    nf: usize,
    budget: usize,
    mode: Mode,
) -> Result<Vec<f64>, String> {
    let total = *groups.offsets.last().unwrap_or(&0);
    let mut out = vec![0.0; total];
    match mode {
        Mode::Serving => {
            for i in 0..groups.offsets.len() - 1 {
                let (start, end) = (groups.offsets[i], groups.offsets[i + 1]);
                method.serve(groups.group(i, nf), end - start)?;
                out[start..end].copy_from_slice(&method.output(end - start));
            }
        }
        Mode::Batch => {
            for call in calls_for(groups, nf, budget) {
                let rows = *call.offsets.last().unwrap_or(&0);
                method.batch(&groups.rows[call.rows.clone()], &call.offsets)?;
                let start = groups.offsets[call.first];
                out[start..start + rows].copy_from_slice(&method.output(rows));
            }
        }
    }
    Ok(out)
}

/// Check the stage oracle: `tree_sum` and `raw_margin` must equal the oracle's bit
/// for bit, and the stages' output must equal production `predict`'s. Each oracle
/// row's stages go to `stages_t`, so the analysis can compare them with the oracle
/// and with the baselines.
pub(super) fn check_oracle(
    cell: &Cell,
    forest: &Forest,
    suite: &str,
    stages_t: &mut Table,
) -> Result<Value> {
    let Some(o) = &cell.oracle else {
        return Ok(json!({"status": "no oracle"}));
    };
    let mut r = forest.research_predictor(Ablation::default());
    let mut p = forest.predictor();
    let mut stages = Stages::default();
    let (mut sum_bad, mut margin_bad, mut out_bad, mut rows) = (0usize, 0usize, 0usize, 0usize);
    let (mut sum_max, mut margin_max) = (0.0f64, 0.0f64);
    let mut want = o.rows.iter();
    for &g in &o.groups {
        anyhow::ensure!(g < cell.n_groups(), "oracle group {g} out of range");
        let group = cell.group(g);
        r.predict_group_stages(group, &mut stages);
        let mut out = vec![0.0; stages.output.len()];
        p.predict_group(group, &mut out);
        for (i, (&ts, &rm)) in stages.tree_sum.iter().zip(&stages.raw_margin).enumerate() {
            let (row, o_sum, o_margin) = *want.next().context("oracle has fewer rows")?;
            anyhow::ensure!(row == cell.offsets[g] + i, "oracle row {row} out of order");
            stages_t.push(&[
                V::S(suite),
                V::S(&cell.id),
                V::U32(g as u32),
                V::U64(row as u64),
                V::F64(ts),
                V::F64(rm),
                V::F64(stages.output[i]),
            ]);
            let (ds, dm) = ((ts - o_sum).abs(), (rm - o_margin).abs());
            // NaN differences count as infinite.
            sum_max = if ds.is_nan() {
                f64::INFINITY
            } else {
                sum_max.max(ds)
            };
            margin_max = if dm.is_nan() {
                f64::INFINITY
            } else {
                margin_max.max(dm)
            };
            // Every model sums exactly since 3.0: both stages are bitwise.
            sum_bad += usize::from(ts.to_bits() != o_sum.to_bits());
            margin_bad += usize::from(rm.to_bits() != o_margin.to_bits());
            out_bad += usize::from(out[i].to_bits() != stages.output[i].to_bits());
            rows += 1;
        }
    }
    let passed = sum_bad == 0 && margin_bad == 0 && out_bad == 0 && want.next().is_none();
    Ok(json!({
        "status": if passed { "pass" } else { "fail" },
        "rows": rows,
        "groups": o.groups.len(),
        "tree_sum": {"mismatches": sum_bad, "max_abs": sum_max, "bitwise": true},
        "raw_margin": {"mismatches": margin_bad, "max_abs": margin_max, "bitwise": true},
        "output_vs_predict": {"mismatches": out_bad},
    }))
}

/// The tolerance for an output summed in `f64` in some other order than
/// production's exact sums: the rounding bound of summing the trees, n_trees x
/// epsilon x the largest possible sum of absolute leaf values (each tree's largest
/// leaf, and the base score), and never below [`Precision::F64`]'s 1e-13. The link's
/// slope is at most 1 (sigmoid's is 1/4), so the bound holds after it too. A wrong
/// branch or tree moves an output by a leaf value, many orders larger.
pub(super) fn f64_sum_tolerance(forest: &Forest, base_score: f64) -> f64 {
    let trees = forest.trees();
    let leaf_sum: f64 = trees
        .iter()
        .map(|t| {
            forest
                .tree_nodes(t)
                .iter()
                .filter(|n| n.is_leaf())
                .map(|n| n.value.abs())
                .fold(0.0, f64::max)
        })
        .sum::<f64>()
        + base_score.abs();
    let bound = (trees.len() as f64 + 1.0) * f64::EPSILON * leaf_sum;
    bound.max(Precision::F64.tolerance())
}

/// Validate every method on the validation groups, through its calls in each timed
/// mode. Returns the checks; methods that fail are removed from `l.methods` and
/// recorded in `l.excluded`.
pub(super) fn validate(
    cell: &Cell,
    l: &mut Loaded,
    g: &Gathered,
    budget: usize,
    modes: &[Mode],
) -> (Vec<Check>, Vec<f64>) {
    let nf = cell.n_features;
    let n = *g.offsets.last().unwrap_or(&0);
    let base = &l.forests[""];
    let f64_tol = f64_sum_tolerance(base, cell.oracle.as_ref().map_or(0.0, |o| o.base_score));
    let mut prod = vec![0.0; n];
    base.predictor()
        .predict_groups(&g.rows, &g.offsets, &mut prod);
    let mut checks = Vec::new();
    let mut failed = vec![false; l.methods.len()];
    for (i, m) in l.methods.iter_mut().enumerate() {
        let mut targets: Vec<(String, Vec<f64>, bool, f64)> = Vec::new();
        if m.is_treewalker() {
            // Every model sums exactly, so tree order, layout, the full walk and every
            // variant give production's output bit for bit.
            targets.push(("predict".into(), prod.clone(), true, 0.0));
        } else {
            let tol = match m.precision() {
                Precision::F64 => f64_tol,
                p @ Precision::F32 => p.tolerance(),
            };
            targets.push(("predict".into(), prod.clone(), false, tol));
        }
        let defect = m.known_defect().map(|(name, affected)| {
            let mask: Vec<bool> = g.rows.chunks_exact(nf).map(affected).collect();
            (name, mask)
        });
        // Every candidate interface the probe may choose is validated; one that
        // fails is never timed, and a method with none left is excluded.
        let candidates = m.candidates(false);
        let mut passing = Vec::new();
        let mut call_error = None;
        for (ci, &iface) in candidates.iter().enumerate() {
            m.select(ci);
            let got: Result<Vec<(Mode, Vec<f64>)>, String> = modes
                .iter()
                .map(|&mode| outputs(m, g, nf, budget, mode).map(|o| (mode, o)))
                .collect();
            let got = match got {
                Ok(v) => v,
                Err(e) => {
                    // Kept as a failed check, even when another candidate passes.
                    checks.push(Check {
                        method: m.method.clone(),
                        variant: m.variant.clone(),
                        interface: iface.into(),
                        against: "call".into(),
                        mode: "-".into(),
                        bitwise: false,
                        tolerance: 0.0,
                        max_abs: f64::INFINITY,
                        mismatches: n,
                        rows: n,
                        passed: false,
                        defect: None,
                        error: Some(e.clone()),
                    });
                    call_error = Some(if candidates.len() > 1 {
                        format!("call failed: {iface}: {e}")
                    } else {
                        format!("call failed: {e}")
                    });
                    continue;
                }
            };
            let mut ok = true;
            for (against, want, bitwise, tol) in &targets {
                for (mode, got) in &got {
                    let attributed = defect.as_ref().map(|(n, mask)| (*n, mask.as_slice()));
                    let (max_abs, mismatches, defect) =
                        compare_attributed(got, want, *bitwise, *tol, attributed);
                    let passed = mismatches == 0;
                    ok &= passed;
                    checks.push(Check {
                        method: m.method.clone(),
                        variant: m.variant.clone(),
                        interface: iface.into(),
                        against: against.clone(),
                        mode: mode.name().into(),
                        bitwise: *bitwise,
                        tolerance: *tol,
                        max_abs,
                        mismatches,
                        rows: n,
                        passed,
                        defect,
                        error: None,
                    });
                }
            }
            if ok {
                passing.push(ci);
            }
        }
        if let Some(&first) = passing.first() {
            m.select(first);
        } else {
            failed[i] = true;
            l.excluded.push(Excluded {
                method: m.method.clone(),
                variant: m.variant.clone(),
                reason: call_error.unwrap_or_else(|| {
                    "outputs differ from production predict beyond tolerance".into()
                }),
            });
        }
        m.allowed = passing;
    }
    drop_failed(l, &failed);
    (checks, prod)
}

pub(super) fn drop_failed(l: &mut Loaded, failed: &[bool]) {
    let mut i = 0;
    l.methods.retain(|_| {
        i += 1;
        !failed[i - 1]
    });
}
