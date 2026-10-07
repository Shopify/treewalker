//! The driver: one cell from loading to its output directory.
//!
//! For each cell it loads the methods the suite asks for, validates every method and
//! variant before timing, checks each ablation's counter signature, records the work
//! counters per group, then times the methods in serving mode (one call per group)
//! and batch mode (calls of up to the batch row budget), one method at a time, reading
//! the hardware counters around each timed pass over a batch (a block).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result};
use serde_json::{Value, json};
use treewalker_gbdt::research::{Ablation, COUNTERS_VERSION, Stages, WorkCounters};
use treewalker_gbdt::{Forest, LoadOptions};

use crate::artifacts::{Cell, ManifestCell, Natives, RunConfig, VariantSpec};
use crate::external::Precision;
use crate::methods::{CHUNK_ROWS, Engine, TimedMethod};
use crate::output::{self, Table, V};
use crate::pmu::Pmu;
use crate::schedule::{Schedule, batch_calls};
use crate::suites::MethodSet;

/// Everything a cell needs from the run.
pub struct Context {
    pub suite: String,
    pub config: RunConfig,
    pub natives: Natives,
    /// `libtl2cgen`, for tl2cgen's multi-row calls, with its SHA-256.
    pub tl2cgen_runtime: Option<(PathBuf, String)>,
    /// Only these methods, when set (`cell --methods`).
    pub only: Option<Vec<String>>,
    /// Replace the cell's tl2cgen library (the `parallel_comp` pilot).
    pub tl2cgen_override: Option<PathBuf>,
    pub pmu: Option<Pmu>,
    /// Why there are no hardware counters, when there are none.
    pub pmu_status: String,
    /// Fail a cell whose counter group was multiplexed or not read.
    pub require_pmu: bool,
    /// 0 in the run's process; k in XGBoost's extra process k.
    pub process: u32,
    /// Cells whose timing began in this process: the sentinel's cadence counts
    /// them, never cells that failed before timing or were reused.
    pub timing_began: usize,
    /// Interfaces to time instead of probing, by method and mode: the sentinel
    /// keeps its first run's choices, so drift never mixes in an interface change.
    pub interface_override: BTreeMap<String, BTreeMap<String, String>>,
}

/// A method or variant that is not timed, and why.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Excluded {
    pub method: String,
    pub variant: String,
    pub reason: String,
}

/// The rows of a set of groups, gathered contiguously.
struct Gathered {
    rows: Vec<f64>,
    /// Local offsets, from 0.
    offsets: Vec<usize>,
}

impl Gathered {
    fn new(cell: &Cell, groups: &[usize]) -> Self {
        let nf = cell.n_features;
        let mut rows = Vec::new();
        let mut offsets = vec![0];
        for &g in groups {
            rows.extend_from_slice(cell.group(g));
            offsets.push(rows.len() / nf);
        }
        Self { rows, offsets }
    }

    fn group(&self, i: usize, nf: usize) -> &[f64] {
        &self.rows[self.offsets[i] * nf..self.offsets[i + 1] * nf]
    }

    fn sizes(&self) -> Vec<usize> {
        self.offsets.windows(2).map(|w| w[1] - w[0]).collect()
    }
}

/// One batch-mode call: positions `first..first + count` of the block.
struct Call {
    first: usize,
    count: usize,
    rows: std::ops::Range<usize>,
    offsets: Vec<usize>,
}

fn calls_for(g: &Gathered, nf: usize, budget: usize) -> Vec<Call> {
    batch_calls(&g.sizes(), budget)
        .into_iter()
        .map(|(first, count)| {
            let base = g.offsets[first];
            Call {
                first,
                count,
                rows: base * nf..g.offsets[first + count] * nf,
                offsets: g.offsets[first..=first + count]
                    .iter()
                    .map(|o| o - base)
                    .collect(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Loading methods
// ---------------------------------------------------------------------------

struct Loaded {
    methods: Vec<TimedMethod>,
    excluded: Vec<Excluded>,
    /// Forests by load options, for counters and full walks.
    forests: BTreeMap<String, Forest>,
    load_seconds: BTreeMap<String, f64>,
}

fn load_key(spec: &VariantSpec) -> String {
    let mut l = spec.load.clone();
    l.sort();
    l.join("+")
}

fn wanted(ctx: &Context, method: &str) -> bool {
    ctx.only
        .as_ref()
        .is_none_or(|only| only.iter().any(|m| m == method))
}

fn load_methods(cell: &Cell, mc: &ManifestCell, ctx: &Context, max_rows: usize) -> Result<Loaded> {
    let set = MethodSet::parse(&mc.methods)?;
    let mut l = Loaded {
        methods: Vec::new(),
        excluded: Vec::new(),
        forests: BTreeMap::new(),
        load_seconds: BTreeMap::new(),
    };
    let t0 = Instant::now();
    let base = cell.forest(&LoadOptions::default())?;
    l.load_seconds
        .insert(String::new(), t0.elapsed().as_secs_f64());
    l.forests.insert(String::new(), base.clone());

    if wanted(ctx, "treewalker") {
        l.methods.push(TimedMethod::new(
            "treewalker",
            "all-on",
            Engine::Production(base.predictor()),
            max_rows,
        ));
    }
    if set.full_walk() && wanted(ctx, "treewalker_fullwalk") {
        l.methods.push(TimedMethod::new(
            "treewalker_fullwalk",
            "-",
            Engine::FullWalk(base),
            max_rows,
        ));
    }
    if set.full_walk() && cell.max_group_rows() > CHUNK_ROWS && wanted(ctx, "treewalker_chunked128")
    {
        match cell.forest_at_width(CHUNK_ROWS) {
            Ok(f) => l.methods.push(TimedMethod::new(
                "treewalker_chunked128",
                "-",
                Engine::Chunked(f.predictor()),
                max_rows,
            )),
            Err(e) => l.excluded.push(Excluded {
                method: "treewalker_chunked128".into(),
                variant: "-".into(),
                reason: format!("load: {e}"),
            }),
        }
    }
    for spec in &mc.variants {
        if !wanted(ctx, "treewalker_research") {
            break;
        }
        let id = spec.id();
        let key = load_key(spec);
        let forest = if let Some(f) = l.forests.get(&key) {
            f.clone()
        } else {
            let t0 = Instant::now();
            match spec.load_options().and_then(|o| cell.forest(&o)) {
                Ok(f) => {
                    l.load_seconds
                        .insert(key.clone(), t0.elapsed().as_secs_f64());
                    l.forests.insert(key, f.clone());
                    f
                }
                Err(e) => {
                    l.excluded.push(Excluded {
                        method: "treewalker_research".into(),
                        variant: id,
                        reason: format!("load: {e:#}"),
                    });
                    continue;
                }
            }
        };
        let ablation = spec.ablation()?;
        let mut m = TimedMethod::new(
            "treewalker_research",
            &id,
            Engine::Research(forest.research_predictor(ablation)),
            max_rows,
        );
        m.spec = Some(spec.clone());
        l.methods.push(m);
    }
    if set.baselines() {
        load_baselines(cell, ctx, max_rows, &mut l);
    }
    Ok(l)
}

#[cfg(not(feature = "quickscorer-bench"))]
fn load_baselines(cell: &Cell, ctx: &Context, _max_rows: usize, l: &mut Loaded) {
    let _ = (cell, ctx);
    l.excluded.push(Excluded {
        method: "baselines".into(),
        variant: "-".into(),
        reason: "sweep_bench was built without external-bench".into(),
    });
}

#[cfg(feature = "quickscorer-bench")]
fn load_baselines(cell: &Cell, ctx: &Context, max_rows: usize, l: &mut Loaded) {
    use crate::external::{Link, QuickScorerBench};

    let mut add = |name: &str, r: Result<Box<dyn crate::external::ExternalMethod>, String>| {
        if !wanted(ctx, name) {
            return;
        }
        match r {
            Ok(m) => l
                .methods
                .push(TimedMethod::new(name, "-", Engine::External(m), max_rows)),
            Err(reason) => l.excluded.push(Excluded {
                method: name.into(),
                variant: "-".into(),
                reason,
            }),
        }
    };
    #[cfg(feature = "external-bench")]
    {
        use crate::external::{LightGbmBench, LleavesBench, Tl2cgenBench, XgBoostBench};
        let nf = cell.n_features;
        // A compiled library is loaded only when it still has its recorded hash.
        let checked = |b: &crate::artifacts::BaselineLib| -> Result<std::path::PathBuf, String> {
            let got = crate::data::sha256_file(&b.path).map_err(|e| format!("{e:#}"))?;
            if got == b.sha256 {
                Ok(b.path.clone())
            } else {
                Err(format!(
                    "{} has sha256 {got}, cell.json records {}; rerun compile-baselines",
                    b.path.display(),
                    b.sha256
                ))
            }
        };
        let tl2cgen = match (&ctx.tl2cgen_override, &cell.tl2cgen) {
            (Some(p), _) => Some(Ok(p.clone())),
            (None, Some(b)) => Some(checked(b)),
            (None, None) => None,
        };
        add(
            "tl2cgen",
            match (tl2cgen, &ctx.tl2cgen_runtime) {
                (Some(Err(e)), _) => Err(e),
                (Some(Ok(lib)), Some((rt, want))) => {
                    // The runtime must still have the hash the run recorded.
                    match crate::data::sha256_file(rt) {
                        Ok(got) if got == *want => {
                            Tl2cgenBench::load(&lib, rt, nf, max_rows).map(|m| Box::new(m) as _)
                        }
                        Ok(got) => Err(format!(
                            "{} has sha256 {got}, the run recorded {want}",
                            rt.display()
                        )),
                        Err(e) => Err(format!("{e:#}")),
                    }
                }
                (None, _) => Err("not compiled; run compile-baselines".into()),
                (_, None) => Err("no libtl2cgen runtime (uv sync --group baselines)".into()),
            },
        );
        if cell.framework == "lightgbm" {
            add(
                "lleaves",
                cell.lleaves.as_ref().map_or_else(
                    || Err("not compiled; run compile-baselines".into()),
                    |lib| {
                        let path = checked(lib)?;
                        LleavesBench::load(&path, nf, max_rows).map(|m| Box::new(m) as _)
                    },
                ),
            );
            add(
                "lightgbm_native",
                match (&ctx.natives.lightgbm, &cell.model_native) {
                    (Some(lib), Some(model)) => LightGbmBench::load(&lib.path, model, nf, max_rows)
                        .map(|m| Box::new(m) as _),
                    (None, _) => Err("no native LightGBM library; run build-native".into()),
                    (_, None) => Err("no LightGBM text model".into()),
                },
            );
        }
        if cell.framework == "xgboost" {
            add(
                "xgboost_native",
                match (&ctx.natives.xgboost, &cell.model_native) {
                    (Some(lib), Some(model)) => {
                        XgBoostBench::load(&lib.path, model, nf, max_rows).map(|m| Box::new(m) as _)
                    }
                    (None, _) => Err("no native XGBoost library; run build-native".into()),
                    (_, None) => Err("no XGBoost JSON model".into()),
                },
            );
        }
    }
    let link = cell.oracle.as_ref().map(|o| Link {
        divisor: o.divisor,
        base_score: o.base_score,
        sigmoid_alpha: (o.postprocessor == "sigmoid").then_some(o.sigmoid_alpha),
    });
    // The XML must still have its recorded hash; LightGBM's text model is a cell
    // file, already checked.
    let quickscorer = match (&cell.quickscorer, cell.quickscorer_sha256.as_deref()) {
        (Ok(p), Some(want)) => match crate::data::sha256_file(p) {
            Ok(got) if got == want => Ok(p.clone()),
            Ok(got) => Err(format!(
                "{} has sha256 {got}, cell.json records {want}",
                p.display()
            )),
            Err(e) => Err(format!("{e:#}")),
        },
        (other, _) => other.clone(),
    };
    add(
        "quickscorer",
        match (&quickscorer, link) {
            (Ok(model), Some(link)) => {
                QuickScorerBench::load(model, link, cell.n_features, max_rows)
                    .map(|m| Box::new(m) as _)
            }
            (Err(reason), _) => Err(reason.clone()),
            (_, None) => Err("no oracle header with the model's link".into()),
        },
    );
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

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

fn compare(a: &[f64], b: &[f64], bitwise: bool, tol: f64) -> (f64, usize) {
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
fn outputs(
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
/// for bit when the model supports exact sums, and the stages' output must equal
/// production `predict`'s.
fn check_oracle(cell: &Cell, forest: &Forest) -> Result<Value> {
    let Some(o) = &cell.oracle else {
        return Ok(json!({"status": "no oracle"}));
    };
    let exact = forest.exact_sums();
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
            if exact {
                sum_bad += usize::from(ts.to_bits() != o_sum.to_bits());
                margin_bad += usize::from(rm.to_bits() != o_margin.to_bits());
            } else {
                sum_bad += usize::from(!matches(ts, o_sum, false, 1e-12 * o_sum.abs().max(1.0)));
                margin_bad += usize::from(!matches(
                    rm,
                    o_margin,
                    false,
                    1e-12 * o_margin.abs().max(1.0),
                ));
            }
            out_bad += usize::from(out[i].to_bits() != stages.output[i].to_bits());
            rows += 1;
        }
    }
    let passed = sum_bad == 0 && margin_bad == 0 && out_bad == 0 && want.next().is_none();
    Ok(json!({
        "status": if passed { "pass" } else { "fail" },
        "exact_sums": exact,
        "rows": rows,
        "groups": o.groups.len(),
        "tree_sum": {"mismatches": sum_bad, "max_abs": sum_max, "bitwise": exact},
        "raw_margin": {"mismatches": margin_bad, "max_abs": margin_max, "bitwise": exact},
        "output_vs_predict": {"mismatches": out_bad},
    }))
}

/// The tolerance for an output summed in `f64` in some other order than
/// production's exact sums: the rounding bound of summing the trees, n_trees x
/// epsilon x the largest possible sum of absolute leaf values (each tree's largest
/// leaf, and the base score), and never below [`Precision::F64`]'s 1e-13. The link's
/// slope is at most 1 (sigmoid's is 1/4), so the bound holds after it too. A wrong
/// branch or tree moves an output by a leaf value, many orders larger.
fn f64_sum_tolerance(forest: &Forest, base_score: f64) -> f64 {
    let trees = forest.trees();
    let leaf_sum: f64 = trees
        .iter()
        .map(|t| {
            forest
                .tree_nodes(t)
                .iter()
                .filter(|n| n.skip == -1)
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
fn validate(
    cell: &Cell,
    l: &mut Loaded,
    g: &Gathered,
    budget: usize,
    modes: &[Mode],
) -> (Vec<Check>, Vec<f64>) {
    let nf = cell.n_features;
    let n = *g.offsets.last().unwrap_or(&0);
    let base = &l.forests[""];
    let exact = base.exact_sums();
    let f64_tol = f64_sum_tolerance(base, cell.oracle.as_ref().map_or(0.0, |o| o.base_score));
    let mut prod = vec![0.0; n];
    base.predictor()
        .predict_groups(&g.rows, &g.offsets, &mut prod);
    // Full walks per load key: the no-exact-sums variants must match them bit for bit.
    let mut walks: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut checks = Vec::new();
    let mut failed = vec![false; l.methods.len()];
    for (i, m) in l.methods.iter_mut().enumerate() {
        let spec = m.spec.clone();
        let spec = spec.as_ref();
        let adds_in_f64 = matches!(m.engine, Engine::FullWalk(_))
            || spec.is_some_and(|s| s.runtime.iter().any(|f| f == "disable_exact_sums"))
            || !exact;
        let mut targets: Vec<(String, Vec<f64>, bool, f64)> = Vec::new();
        if m.is_treewalker() {
            if adds_in_f64 {
                let key = spec.map(load_key).unwrap_or_default();
                let walk = walks.entry(key.clone()).or_insert_with(|| {
                    let mut out = vec![0.0; n];
                    l.forests[&key].predict_full_walk(&g.rows, &mut out);
                    out
                });
                targets.push((
                    format!("full walk {key}").trim().into(),
                    walk.clone(),
                    true,
                    0.0,
                ));
                targets.push(("predict".into(), prod.clone(), false, f64_tol));
            } else {
                targets.push(("predict".into(), prod.clone(), true, 0.0));
            }
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

fn drop_failed(l: &mut Loaded, failed: &[bool]) {
    let mut i = 0;
    l.methods.retain(|_| {
        i += 1;
        !failed[i - 1]
    });
}

// ---------------------------------------------------------------------------
// Work counters and ablation signatures
// ---------------------------------------------------------------------------

fn count(
    forest: &Forest,
    ablation: Ablation,
    cell: &Cell,
    per_group: bool,
) -> (WorkCounters, Vec<WorkCounters>) {
    let mut r = forest.research_predictor(ablation);
    let mut out = vec![0.0; cell.max_group_rows()];
    let mut total = WorkCounters::default();
    let mut groups = Vec::new();
    for g in 0..cell.n_groups() {
        let rows = cell.group(g);
        let c = r.predict_group_counted(rows, &mut out[..rows.len() / cell.n_features]);
        total += c;
        if per_group {
            groups.push(c);
        }
    }
    (total, groups)
}

/// What a cell's no-op rules and signatures depend on.
#[derive(Debug, Clone, Copy)]
pub struct Facts {
    /// The cell declares monotonic features.
    pub monotonic: bool,
    /// The model supports exact sums.
    pub exact: bool,
    /// The model has varying predicates: splits on a varying feature. A horizon-1
    /// survival model has none: its time features are constant in training.
    pub varying_predicates: bool,
    /// Rows times trees: the leaf adds of per-row accumulation.
    pub row_leaves: u64,
}

impl Facts {
    pub fn of(forest: &Forest, cell: &Cell) -> Self {
        let config = forest.config();
        let rows = cell.group(0);
        let first = &rows[..rows.len().min(32 * cell.n_features)];
        Self {
            monotonic: (0..config.n_features())
                .any(|f| config.is_increasing(f) || config.is_decreasing(f)),
            exact: forest.exact_sums(),
            varying_predicates: !forest.predicate_masks(first, true).is_empty(),
            row_leaves: (cell.n_rows * forest.trees().len()) as u64,
        }
    }
}

/// Whether `flag` is a no-op for this variant and cell, and why.
pub fn no_op(flag: &str, spec: &VariantSpec, facts: &Facts) -> Option<&'static str> {
    let has = |f: &str| spec.runtime.iter().any(|x| x == f);
    match flag {
        "disable_monotonic" if !has("disable_varying_precompute") => {
            Some("precompute is on, and it ignores monotonicity")
        }
        "disable_monotonic" if !facts.monotonic => Some("the cell has no monotonic features"),
        "disable_predicate_sweep" if has("disable_varying_precompute") => Some("precompute is off"),
        "disable_exact_sums" if !facts.exact => Some("the model has no exact sums; it adds in f64"),
        "disable_unsplit"
        | "disable_varying_precompute"
        | "disable_predicate_sweep"
        | "disable_monotonic"
            if !facts.varying_predicates =>
        {
            Some("the model has no varying predicates")
        }
        _ => None,
    }
}

/// Whether `v`, the variant's counters, show that `flag`'s path ran.
///
/// Judged by the path's own counters, never by totals differing from all-on: with
/// two identical rows, exact and per-row accumulation both make 4 leaf adds.
pub fn signature(flag: &str, v: &WorkCounters, facts: &Facts) -> bool {
    match flag {
        // The per-row partitions ran, and the precompute did not.
        "disable_varying_precompute" => {
            v.precompute_row_evals == 0
                && v.precompute_mask_writes == 0
                && (v.partition_row_evals > 0 || v.varying_splits == 0)
        }
        // No split was skipped as unsplit.
        "disable_unsplit" => v.unsplit_skips == 0,
        // The brute force wrote the masks, with no sort or sweep.
        "disable_predicate_sweep" => {
            v.precompute_sweep_compares == 0
                && v.precompute_sort_compares == 0
                && v.precompute_mask_writes > 0
        }
        // Every partition scanned all its active rows: no monotonic early stop.
        "disable_monotonic" => v.scan_row_evals == v.partition_row_evals,
        // One add per row per tree: per-row accumulation.
        "disable_exact_sums" => v.leaf_adds == facts.row_leaves,
        _ => false,
    }
}

fn push_counters(
    t: &mut Table,
    suite: &str,
    cell: &str,
    variant: &str,
    g: usize,
    c: &WorkCounters,
) {
    t.push(&[
        V::S(suite),
        V::S(cell),
        V::S(variant),
        V::U32(g as u32),
        V::U32(COUNTERS_VERSION),
        V::U64(c.constant_steps),
        V::U64(c.varying_splits),
        V::U64(c.unsplit_skips),
        V::U64(c.recursive_calls),
        V::U64(c.leaf_hits),
        V::U64(c.leaf_adds),
        V::U64(c.partition_row_evals),
        V::U64(c.scan_row_evals),
        V::U64(c.scan_compares),
        V::U64(c.scan_missing_checks),
        V::U64(c.precompute_row_evals),
        V::U64(c.precompute_sort_compares),
        V::U64(c.precompute_sweep_compares),
        V::U64(c.precompute_mask_writes),
    ]);
}

/// Record per-group counters for every research variant (and all-on), and check
/// each ablation's signature. Variants whose counters do not show an active flag
/// are removed from timing.
fn counters(
    ctx: &Context,
    cell: &Cell,
    l: &mut Loaded,
    table: &mut Table,
) -> Result<BTreeMap<String, Value>> {
    let facts = Facts::of(&l.forests[""], cell);
    let mut totals: BTreeMap<(String, Vec<String>), WorkCounters> = BTreeMap::new();
    let mut total_of =
        |forests: &BTreeMap<String, Forest>, key: &str, rt: &[String], per_group: bool| {
            let mut rt = rt.to_vec();
            rt.sort();
            let spec = VariantSpec {
                runtime: rt.clone(),
                load: Vec::new(),
            };
            if !per_group && let Some(t) = totals.get(&(key.to_string(), rt.clone())) {
                return Ok::<_, anyhow::Error>((*t, Vec::new()));
            }
            let (t, g) = count(&forests[key], spec.ablation()?, cell, per_group);
            totals.insert((key.to_string(), rt), t);
            Ok((t, g))
        };
    let (_, all_on) = total_of(&l.forests, "", &[], true)?;
    for (g, c) in all_on.iter().enumerate() {
        push_counters(table, &ctx.suite, &cell.id, "all-on", g, c);
    }
    let mut report = BTreeMap::new();
    let mut failed = vec![false; l.methods.len()];
    let variants: Vec<(usize, VariantSpec)> = (l.methods.iter().enumerate())
        .filter_map(|(i, m)| Some((i, m.spec.clone()?)))
        .collect();
    for (i, spec) in variants {
        let id = spec.id();
        let key = load_key(&spec);
        let (v, groups) = if id == "all-on" {
            (
                all_on.iter().fold(WorkCounters::default(), |mut a, c| {
                    a += *c;
                    a
                }),
                Vec::new(),
            )
        } else {
            total_of(&l.forests, &key, &spec.runtime, true)?
        };
        for (g, c) in groups.iter().enumerate() {
            push_counters(table, &ctx.suite, &cell.id, &id, g, c);
        }
        let mut flags = BTreeMap::new();
        for flag in &spec.runtime {
            let status = match no_op(flag, &spec, &facts) {
                Some(why) => format!("no-op: {why}"),
                None if signature(flag, &v, &facts) => "active".to_string(),
                None => {
                    failed[i] = true;
                    "failed: the counters do not show this flag's path".to_string()
                }
            };
            flags.insert(flag.clone(), Value::String(status));
        }
        if failed[i] {
            l.excluded.push(Excluded {
                method: "treewalker_research".into(),
                variant: id.clone(),
                reason: "ablation self-check failed".into(),
            });
        }
        report.insert(
            id,
            json!({"flags": flags, "totals": format!("{v}"), "load": spec.load}),
        );
    }
    drop_failed(l, &failed);
    Ok(report)
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// The mode of a timing phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Serving,
    Batch,
}

impl Mode {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Serving => "serving",
            Self::Batch => "batch",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "serving" => Ok(Self::Serving),
            "batch" => Ok(Self::Batch),
            other => anyhow::bail!("unknown mode {other}; expected serving or batch"),
        }
    }

    /// A suite's modes, in order, each once; at least one.
    pub fn parse_list(names: &[String]) -> Result<Vec<Self>> {
        let mut out = Vec::new();
        for n in names {
            let m = Self::parse(n)?;
            if !out.contains(&m) {
                out.push(m);
            }
        }
        anyhow::ensure!(!out.is_empty(), "no timing modes");
        Ok(out)
    }
}

/// Why a mode stopped, and how much it timed.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Stopped {
    pub reason: &'static str,
    pub rounds: usize,
    pub blocks: usize,
    pub blocks_per_round: usize,
    pub seconds: f64,
    /// Relative standard error of each method's per-round mean ticks per row, in
    /// percent, at the stop.
    pub rse_pct: BTreeMap<String, f64>,
    /// Each method's timed ticks and rows over every round, by `method/variant`.
    pub totals: BTreeMap<String, [u64; 2]>,
    /// Methods that failed during timing; their samples up to the failure are kept.
    pub failures: Vec<Excluded>,
    /// Blocks whose counter group was multiplexed or could not be read.
    pub incomplete_reads: usize,
    /// Rounds per order cycle: forward, then reversed. A mode can stop after any
    /// round from `min_rounds` on, so an odd round count leaves the positions
    /// balanced only up to the last round.
    pub cycle: usize,
    /// Each method's interface probe, where it had a choice.
    pub probes: Vec<Probe>,
    /// The input-conversion passes of the methods that convert outside their
    /// library (QuickScorer's f64-to-f32 copy), one per round.
    pub conversion: Vec<Conversion>,
}

/// One input-conversion pass over the batch the method timed last.
///
/// Right after its timed passes: the calls' conversions alone, back to back in one interval
/// (amortized: one timer read pair for the batch), against a headline-style pass
/// over the same calls. Neither enters the samples.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Conversion {
    pub method: String,
    pub round: usize,
    pub batch: usize,
    pub calls: usize,
    pub rows: usize,
    pub ticks: u64,
    pub conversion_ticks: u64,
}

/// The conversion pass of a method over one batch: calls, rows, the ticks of a
/// headline-style pass, and the ticks of the same calls' conversions alone, all in
/// one interval, so the timer's overhead is one read pair over the batch. `None`
/// when the method has no conversion of its own, or a call failed.
fn conversion_pass(
    m: &mut TimedMethod,
    g: &Gathered,
    calls: Option<&[Call]>,
    nf: usize,
) -> Option<(usize, usize, u64, u64)> {
    let list: Vec<(&[f64], usize, bool)> = calls.map_or_else(
        || {
            (0..g.offsets.len() - 1)
                .map(|i| (g.group(i, nf), g.offsets[i + 1] - g.offsets[i], false))
                .collect()
        },
        |calls| {
            let call = |c: &Call| {
                (
                    &g.rows[c.rows.clone()],
                    *c.offsets.last().unwrap_or(&0),
                    true,
                )
            };
            calls.iter().map(call).collect()
        },
    );
    let (first, n, multi) = *list.first()?;
    if !m.convert(first, n, multi) {
        return None;
    }
    let t0 = crate::timer::start();
    let mut ok = true;
    for &(rows, n, multi) in &list {
        ok &= m.convert(rows, n, multi);
    }
    let conversion = crate::timer::stop().wrapping_sub(t0);
    let mut ticks = 0u64;
    run_pass(m, g, calls, nf, |_, _, _, t| ticks += t).ok()?;
    let rows = list.iter().map(|l| l.1).sum();
    ok.then_some((list.len(), rows, ticks, conversion))
}

fn rse_pct(per_round: &[f64]) -> f64 {
    let n = per_round.len() as f64;
    if per_round.len() < 2 {
        return f64::INFINITY;
    }
    let mean = per_round.iter().sum::<f64>() / n;
    let var = per_round.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (var / n).sqrt() / mean * 100.0
}

/// One mode's timing over the given groups: every round gives each method its own
/// phase, a warm-up on the batch it times last and then a timed pass over every
/// batch, in the schedule's method order.
#[expect(clippy::too_many_lines, reason = "the rounds and phases of one mode")]
fn time_mode(
    ctx: &mut Context,
    cell: &Cell,
    groups: &[usize],
    methods: &mut [TimedMethod],
    mode: Mode,
    samples: &mut Table,
    hw: &mut Table,
) -> Stopped {
    let cfg = ctx.config.clone();
    let nf = cell.n_features;
    // Seeded per cell, so the method order's position advantage differs by cell.
    let seed = crate::schedule::cell_seed(cfg.seed, &cell.id);
    let schedule = Schedule::new(groups.len(), methods.len(), cfg.target_batches, seed);
    let bpr = schedule.blocks_per_round();
    let batch_groups =
        |b: usize| -> Vec<usize> { schedule.batches[b].iter().map(|&i| groups[i]).collect() };
    let gather = |b: usize| {
        let g = Gathered::new(cell, &batch_groups(b));
        let calls = (mode == Mode::Batch).then(|| calls_for(&g, nf, cfg.batch_rows));
        (g, calls)
    };
    let mut alive = vec![true; methods.len()];
    let mut failures = Vec::new();
    let mut per_round: Vec<Vec<f64>> = vec![Vec::new(); methods.len()];
    let started = Instant::now();
    let mut sample_id = samples.len() as u64;
    let (mut rounds, mut blocks, mut reason) = (0, 0, "block_cap");
    let cycle = schedule.cycle();
    // The minimum binds first: neither the precision target nor the budget stops a
    // mode before it.
    let min_rounds = cfg.min_rounds.max(1);
    let max_rounds = cfg.max_rounds.max(min_rounds);
    let mut incomplete_reads = 0;
    let mut probes = Vec::new();
    let mut conversion = Vec::new();
    let mut totals = vec![[0u64; 2]; methods.len()];
    // Serving calls of one row always use the single-row interface, so only cells
    // with wider groups, and batch mode, have a choice to probe.
    let multi_row = mode == Mode::Batch
        || groups
            .iter()
            .any(|&g| cell.offsets[g + 1] - cell.offsets[g] > 1);
    for round in 0..max_rounds {
        let mut round_ticks = vec![0u64; methods.len()];
        let mut round_rows = vec![0u64; methods.len()];
        for (pos, mi) in schedule.method_order(round).into_iter().enumerate() {
            if !alive[mi] {
                continue;
            }
            let m = &mut methods[mi];
            let (method, variant) = (m.method.clone(), m.variant.clone());
            let (wg, wcalls) = gather(schedule.warm_batch());
            // The mode's first phase of each method chooses its interface: every
            // candidate runs on the warm-up batch, and the faster is timed in every
            // round.
            if round == 0 {
                let names = m.candidates(mode == Mode::Batch);
                let fixed = ctx
                    .interface_override
                    .get(&m.method)
                    .and_then(|by_mode| by_mode.get(mode.name()))
                    .and_then(|want| names.iter().position(|n| n == want))
                    .filter(|i| m.allowed.contains(i));
                let choice = if let Some(i) = fixed {
                    Ok(i)
                } else if multi_row && m.allowed.len() > 1 {
                    probe(m, &wg, wcalls.as_deref(), nf, schedule.warm_batch()).map(|p| {
                        let i = p.index;
                        probes.push(p);
                        i
                    })
                } else {
                    m.allowed
                        .first()
                        .copied()
                        .ok_or_else(|| "no interface".into())
                };
                match choice {
                    Ok(i) => {
                        m.select(i);
                        // All-1-row serving cells only ever use the single-row call.
                        let iface = if multi_row {
                            m.interface(2, mode == Mode::Batch)
                        } else {
                            m.interface(1, false)
                        };
                        m.chosen.insert(mode.name().into(), iface);
                    }
                    Err(e) => {
                        alive[mi] = false;
                        failures.push(Excluded {
                            method,
                            variant,
                            reason: format!("probe: {e}"),
                        });
                        continue;
                    }
                }
            }
            let (iface_one, iface_multi) =
                (m.interface(1, false), m.interface(2, mode == Mode::Batch));
            // Warm up, untimed, on the batch this phase times last.
            let warm = run_pass(m, &wg, wcalls.as_deref(), nf, |_, _, _, _| {});
            drop(wg);
            if let Err(e) = warm {
                alive[mi] = false;
                failures.push(Excluded {
                    method,
                    variant,
                    reason: e,
                });
                continue;
            }
            for b in 0..bpr {
                let block = round * bpr + b;
                let batch = batch_groups(b);
                let (g, calls) = gather(b);
                let marked = ctx.pmu.as_mut().map(Pmu::mark);
                let result = run_pass(m, &g, calls.as_deref(), nf, |first, count, rows, ticks| {
                    let iface = if mode == Mode::Serving && rows == 1 {
                        iface_one
                    } else {
                        iface_multi
                    };
                    samples.push(&[
                        V::U64(sample_id),
                        V::S(&ctx.suite),
                        V::S(&cell.id),
                        V::S(&variant),
                        V::S(&method),
                        V::S(mode.name()),
                        V::S(iface),
                        V::U32(block as u32),
                        V::U32(b as u32),
                        V::U32(pos as u32),
                        V::U32(round as u32),
                        V::U32(ctx.process),
                        V::U32(batch[first] as u32),
                        V::U32(count as u32),
                        V::U32(rows as u32),
                        V::U64(ticks),
                    ]);
                    sample_id += 1;
                    round_ticks[mi] += ticks;
                    totals[mi][0] += ticks;
                    round_rows[mi] += rows as u64;
                    totals[mi][1] += rows as u64;
                });
                let reading = match (marked, ctx.pmu.as_mut()) {
                    (Some(Ok(())), Some(p)) => Some(p.since_mark()),
                    (Some(Err(e)), _) => Some(Err(e)),
                    _ => None,
                };
                if matches!(&reading, Some(Err(_)))
                    || matches!(&reading, Some(Ok(r)) if !r.complete())
                {
                    incomplete_reads += 1;
                }
                push_hw(hw, ctx, &cell.id, &variant, &method, mode, block, reading);
                if let Err(e) = result {
                    alive[mi] = false;
                    failures.push(Excluded {
                        method: method.clone(),
                        variant: variant.clone(),
                        reason: e,
                    });
                    break;
                }
            }
            // QuickScorer's f32 copy, timed on its own over the batch just timed
            // last, outside the samples.
            if alive[mi] {
                let wb = schedule.warm_batch();
                let (g, calls) = gather(wb);
                if let Some((n_calls, rows, ticks, conversion_ticks)) =
                    conversion_pass(m, &g, calls.as_deref(), nf)
                {
                    conversion.push(Conversion {
                        method: method.clone(),
                        round,
                        batch: wb,
                        calls: n_calls,
                        rows,
                        ticks,
                        conversion_ticks,
                    });
                }
            }
        }
        blocks += bpr;
        rounds = round + 1;
        for mi in 0..methods.len() {
            if alive[mi] && round_rows[mi] > 0 {
                per_round[mi].push(round_ticks[mi] as f64 / round_rows[mi] as f64);
            }
        }
        if rounds < min_rounds {
            continue;
        }
        // The relative standard error needs two rounds.
        let precise = (0..methods.len())
            .filter(|&mi| alive[mi])
            .all(|mi| rse_pct(&per_round[mi]) < cfg.precision_pct);
        if precise {
            reason = "precision";
            break;
        }
        if started.elapsed().as_secs_f64() >= cfg.mode_budget_secs && rounds < max_rounds {
            reason = "time_budget";
            break;
        }
    }
    Stopped {
        reason,
        rounds,
        blocks,
        blocks_per_round: bpr,
        seconds: started.elapsed().as_secs_f64(),
        rse_pct: methods
            .iter()
            .enumerate()
            .map(|(mi, m)| {
                (
                    format!("{}/{}", m.method, m.variant),
                    rse_pct(&per_round[mi]),
                )
            })
            .collect(),
        totals: methods
            .iter()
            .zip(&totals)
            .map(|(m, t)| (format!("{}/{}", m.method, m.variant), *t))
            .collect(),
        failures,
        incomplete_reads,
        cycle,
        probes,
        conversion,
    }
}

/// The interface probe of one method on one batch.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Probe {
    pub method: String,
    pub variant: String,
    /// The warm-up batch, by index.
    pub batch: usize,
    pub rows: usize,
    pub candidates: Vec<ProbeCandidate>,
    /// How much slower the runner-up was than the chosen, in percent: below the
    /// probe's noise (about 1-2%) the choice is a coin toss between near-equals.
    pub margin_pct: Option<f64>,
    /// The candidate timed in the mode.
    pub chosen: String,
    #[serde(skip)]
    pub index: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProbeCandidate {
    pub interface: String,
    /// Ticks of its two timed passes; `None` when a call failed.
    pub ticks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Probe a method's validated candidates on one batch, untimed for the mode: each
/// runs one warm pass, then timed passes in order and in reverse order (A B B A).
/// The candidate with the fewest ticks is chosen. An error when every one fails.
fn probe(
    m: &mut TimedMethod,
    g: &Gathered,
    calls: Option<&[Call]>,
    nf: usize,
    batch: usize,
) -> Result<Probe, String> {
    let names = m.candidates(calls.is_some());
    let allowed = m.allowed.clone();
    let mut ticks: Vec<Option<u64>> = vec![Some(0); allowed.len()];
    let mut errors: Vec<Option<String>> = vec![None; allowed.len()];
    let mut pass = |m: &mut TimedMethod, k: usize, timed: bool, ticks: &mut [Option<u64>]| {
        if ticks[k].is_none() {
            return;
        }
        m.select(allowed[k]);
        let mut sum = 0u64;
        match run_pass(m, g, calls, nf, |_, _, _, t| sum += t) {
            Ok(()) if timed => ticks[k] = ticks[k].map(|t| t + sum),
            Ok(()) => {}
            Err(e) => {
                ticks[k] = None;
                errors[k] = Some(e);
            }
        }
    };
    for k in 0..allowed.len() {
        pass(m, k, false, &mut ticks);
    }
    for k in (0..allowed.len()).chain((0..allowed.len()).rev()) {
        pass(m, k, true, &mut ticks);
    }
    let best = (0..allowed.len())
        .filter_map(|k| ticks[k].map(|t| (t, k)))
        .min()
        .map(|(_, k)| k);
    let candidates = (0..allowed.len())
        .map(|k| ProbeCandidate {
            interface: names[allowed[k]].into(),
            ticks: ticks[k],
            error: errors[k].clone(),
        })
        .collect();
    let Some(best) = best else {
        return Err(format!("every candidate failed: {errors:?}"));
    };
    Ok(Probe {
        method: m.method.clone(),
        variant: m.variant.clone(),
        batch,
        rows: *g.offsets.last().unwrap_or(&0),
        margin_pct: (0..allowed.len())
            .filter(|&k| k != best)
            .filter_map(|k| ticks[k])
            .min()
            .zip(ticks[best])
            .map(|(second, first)| (second as f64 / first.max(1) as f64 - 1.0) * 100.0),
        candidates,
        chosen: names[allowed[best]].into(),
        index: allowed[best],
    })
}

/// One pass of a method over a block: one call per group, or the batch calls.
/// `record(first, count, rows, ticks)` receives each sample.
fn run_pass(
    m: &mut TimedMethod,
    g: &Gathered,
    calls: Option<&[Call]>,
    nf: usize,
    mut record: impl FnMut(usize, usize, usize, u64),
) -> Result<(), String> {
    match calls {
        None => {
            for i in 0..g.offsets.len() - 1 {
                let rows = g.offsets[i + 1] - g.offsets[i];
                let ticks = m.serve(g.group(i, nf), rows)?;
                record(i, 1, rows, ticks);
            }
        }
        Some(calls) => {
            for c in calls {
                let ticks = m.batch(&g.rows[c.rows.clone()], &c.offsets)?;
                record(c.first, c.count, *c.offsets.last().unwrap_or(&0), ticks);
            }
        }
    }
    Ok(())
}

#[expect(clippy::too_many_arguments, reason = "one row of the hw table")]
fn push_hw(
    hw: &mut Table,
    ctx: &Context,
    cell: &str,
    variant: &str,
    method: &str,
    mode: Mode,
    block: usize,
    reading: Option<Result<crate::pmu::Reading, String>>,
) {
    let mut row = vec![
        V::S(&ctx.suite),
        V::S(cell),
        V::S(variant),
        V::S(method),
        V::S(mode.name()),
        V::U32(block as u32),
        V::U32(ctx.process),
    ];
    let status = match reading {
        Some(Ok(r)) => {
            row.extend(r.values.iter().map(|v| V::Opt(*v)));
            row.extend([
                V::Opt(Some(r.time_enabled)),
                V::Opt(Some(r.time_running)),
                V::Opt(Some(r.rusage_nvcsw)),
                V::Opt(Some(r.rusage_nivcsw)),
            ]);
            if r.complete() { "ok" } else { "multiplexed" }.to_string()
        }
        Some(Err(e)) => {
            row.extend([V::Opt(None); 12]);
            format!("read failed: {e}")
        }
        None => {
            row.extend([V::Opt(None); 12]);
            format!("unsupported: {}", ctx.pmu_status)
        }
    };
    row.push(V::S(&status));
    hw.push(&row);
}

// ---------------------------------------------------------------------------
// One cell
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// XGBoost's extra processes
// ---------------------------------------------------------------------------

/// What every extra process times like the main one: the groups, modes and
/// output size, and the main process's ticks and rows per mode and
/// `method/variant`.
struct Shared<'a> {
    cell_dir: &'a Path,
    /// The cell and its production forest, for the output check.
    cell: &'a Cell,
    forest: &'a Forest,
    groups: &'a [usize],
    modes: &'a [Mode],
    max_rows: usize,
    totals: &'a BTreeMap<String, BTreeMap<String, [u64; 2]>>,
}

/// What a child process times and checks: [`extra_processes`] writes it,
/// [`xgboost_process`] reads it.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ChildRequest {
    cell_dir: PathBuf,
    suite: String,
    config: RunConfig,
    xgboost: crate::artifacts::NativeLib,
    process: u32,
    groups: Vec<usize>,
    modes: Vec<String>,
    /// The parent's interface per mode, by index into the candidates.
    interfaces: BTreeMap<String, usize>,
    max_rows: usize,
    require_pmu: bool,
    /// The output check: production predict's outputs on these groups.
    check_groups: Vec<usize>,
    expected: Vec<f64>,
}

const XGBOOST: &str = "xgboost_native/-";
/// Groups each extra process checks its outputs on.
const CHECK_GROUPS: usize = 32;

/// One process's measurement of XGBoost in one mode.
fn process_summary(process: u32, mode: &str, ticks_rows: Option<[u64; 2]>, status: &str) -> Value {
    let [ticks, rows] = ticks_rows.unwrap_or_default();
    json!({
        "process": process,
        "mode": mode,
        "ticks": ticks,
        "rows": rows,
        "ticks_per_row": (rows > 0).then(|| ticks as f64 / rows as f64),
        "status": status,
    })
}

/// Time XGBoost alone in `xgboost_processes` extra processes, one round each, with
/// the main process's groups, schedule and interfaces. Their samples and counter
/// blocks join the cell's tables as process k. Returns the per-process record;
/// a process that fails is recorded, never fatal.
fn extra_processes(
    ctx: &Context,
    methods: &mut [TimedMethod],
    shared: &Shared<'_>,
    samples: &mut Table,
    hw: &mut Table,
) -> Value {
    let cfg = &ctx.config;
    let Some(m) = methods.iter_mut().find(|m| m.method == "xgboost_native") else {
        return Value::Null;
    };
    let arch = std::env::consts::ARCH;
    let skip = |why: String| json!({"count": 0, "arch": arch, "skipped": why});
    if cfg.xgboost_processes == 0 {
        return skip("xgboost_processes is 0".into());
    }
    if !cfg.xgboost_process_arches.iter().any(|a| a == arch) {
        return skip(format!("{arch} is not in xgboost_process_arches"));
    }
    let Some(lib) = &ctx.natives.xgboost else {
        return skip("no native XGBoost library".into());
    };
    let mut processes: Vec<Value> = shared
        .modes
        .iter()
        .map(|mode| {
            let mode = mode.name();
            process_summary(
                0,
                mode,
                shared
                    .totals
                    .get(mode)
                    .and_then(|t| t.get(XGBOOST))
                    .copied(),
                "ok",
            )
        })
        .collect();
    let names = m.candidates(false);
    let interfaces: BTreeMap<String, usize> = m
        .chosen
        .iter()
        .filter_map(|(mode, iface)| Some((mode.clone(), names.iter().position(|n| n == iface)?)))
        .collect();
    // The output check: production predict on a few groups, which each child's
    // XGBoost must match within its validation tolerance.
    let check: Vec<usize> = shared.groups.iter().copied().take(CHECK_GROUPS).collect();
    m.select(interfaces.get("serving").copied().unwrap_or(0));
    // Production predict's outputs, which validation held XGBoost to.
    let g = Gathered::new(shared.cell, &check);
    let mut expected = vec![0.0; *g.offsets.last().unwrap_or(&0)];
    shared
        .forest
        .predictor()
        .predict_groups(&g.rows, &g.offsets, &mut expected);
    let exe = std::env::current_exe();
    for k in 1..=cfg.xgboost_processes as u32 {
        let request = ChildRequest {
            cell_dir: shared.cell_dir.to_path_buf(),
            suite: ctx.suite.clone(),
            config: cfg.clone(),
            xgboost: lib.clone(),
            process: k,
            groups: shared.groups.to_vec(),
            modes: shared.modes.iter().map(|m| m.name().to_string()).collect(),
            interfaces: interfaces.clone(),
            max_rows: shared.max_rows,
            // Where the parent reads counters, a child must too: XGBoost's modes
            // are told apart by its instructions.
            require_pmu: ctx.require_pmu || ctx.pmu.is_some(),
            check_groups: check.clone(),
            expected: expected.clone(),
        };
        let result = (|| -> Result<Value> {
            let dir = std::env::temp_dir()
                .join(format!("sweep_bench-xgboost-{}-{k}", std::process::id()));
            std::fs::create_dir_all(&dir)?;
            let (req, out) = (dir.join("request.json"), dir.join("result.json"));
            std::fs::write(&req, serde_json::to_string(&request)?)?;
            let exe = exe.as_ref().map_err(|e| anyhow::anyhow!("{e}"))?;
            let status = std::process::Command::new(exe)
                .arg("xgboost-process")
                .arg(&req)
                .arg(&out)
                .status()?;
            let doc = std::fs::read_to_string(&out);
            std::fs::remove_dir_all(&dir)?;
            anyhow::ensure!(status.success(), "the process exited with {status}");
            Ok(serde_json::from_str(&doc?)?)
        })();
        let doc = match result {
            Ok(d) => d,
            Err(e) => {
                for mode in shared.modes {
                    let status = format!("failed: {e:#}");
                    processes.push(process_summary(k, mode.name(), None, &status));
                }
                continue;
            }
        };
        // A child is valid only when it timed to completion, read every counter
        // block in full where counters are on, and reproduced the parent's outputs.
        let counters = cfg.hardware_counters && ctx.pmu.is_some();
        let invalid = child_invalid(&doc, shared.modes, counters);
        if !invalid.is_empty() {
            let status = format!("invalid: {}", invalid.join("; "));
            for mode in shared.modes {
                let stop = &doc["stops"][mode.name()];
                let ticks_rows = serde_json::from_value(stop["totals"][XGBOOST].clone()).ok();
                processes.push(process_summary(k, mode.name(), ticks_rows, &status));
            }
            continue;
        }
        // Sample IDs continue the cell's.
        let imported = (|| -> Result<()> {
            for row in doc["samples"].as_array().into_iter().flatten() {
                let mut row = row.as_array().cloned().unwrap_or_default();
                if let Some(id) = row.first_mut() {
                    *id = json!(samples.len());
                }
                samples.push_json(&row)?;
            }
            for row in doc["hw"].as_array().into_iter().flatten() {
                hw.push_json(row.as_array().map_or(&[][..], Vec::as_slice))?;
            }
            Ok(())
        })();
        for mode in shared.modes {
            let stop = &doc["stops"][mode.name()];
            let ticks_rows = serde_json::from_value(stop["totals"][XGBOOST].clone()).ok();
            let status = match &imported {
                Err(e) => format!("failed: {e:#}"),
                Ok(()) => "ok".into(),
            };
            processes.push(process_summary(k, mode.name(), ticks_rows, &status));
        }
    }
    json!({"count": cfg.xgboost_processes, "arch": arch, "processes": processes})
}

/// Why an extra process's result cannot stand, if it cannot: a failed output
/// check, a failure during timing, or, where the parent reads counters, a child
/// without them or with any block multiplexed or unread.
fn child_invalid(doc: &Value, modes: &[Mode], counters: bool) -> Vec<String> {
    let mut invalid = Vec::new();
    if doc["check"]["passed"] != true {
        invalid.push(format!(
            "outputs differ from production predict's: {}",
            doc["check"]
        ));
    }
    for mode in modes {
        let stop = &doc["stops"][mode.name()];
        if let Some(f) = stop["failures"].as_array().filter(|f| !f.is_empty()) {
            invalid.push(format!("{} failed: {f:?}", mode.name()));
        }
        let incomplete = stop["incomplete_reads"].as_u64().unwrap_or(0);
        if counters && incomplete > 0 {
            invalid.push(format!(
                "{}: {incomplete} counter blocks multiplexed or unread",
                mode.name()
            ));
        }
    }
    if counters && doc["pmu_status"] != "ok" {
        invalid.push(format!("no hardware counters: {}", doc["pmu_status"]));
    }
    invalid
}

/// The child side of [`extra_processes`]: XGBoost alone, one round per mode, with
/// the parent's groups and interfaces, its rows returned as JSON.
#[cfg(feature = "external-bench")]
pub fn xgboost_process(request: &Value) -> Result<Value> {
    use crate::external::XgBoostBench;
    let ChildRequest {
        cell_dir,
        suite,
        mut config,
        xgboost: lib,
        process,
        groups,
        modes,
        interfaces,
        max_rows,
        require_pmu,
        check_groups,
        expected,
    } = serde::Deserialize::deserialize(request).context("the child's request")?;
    anyhow::ensure!(!check_groups.is_empty(), "the request has no check groups");
    config.min_rounds = 1;
    config.max_rounds = 1;
    let cell = Cell::load(&cell_dir)?;
    let model = cell
        .model_native
        .as_ref()
        .context("the cell has no XGBoost model")?;
    let bench = XgBoostBench::load(&lib.path, model, cell.n_features, max_rows)
        .map_err(anyhow::Error::msg)?;
    let mut methods = vec![TimedMethod::new(
        "xgboost_native",
        "-",
        Engine::External(Box::new(bench)),
        max_rows,
    )];
    let (pmu, pmu_status) = if config.hardware_counters {
        match Pmu::open() {
            Ok(p) => (Some(p), "ok".to_string()),
            Err(e) if require_pmu => anyhow::bail!("hardware counters: {e}"),
            Err(e) => (None, e),
        }
    } else {
        (None, "disabled in the run config".to_string())
    };
    let mut ctx = Context {
        suite,
        config,
        natives: Natives::default(),
        tl2cgen_runtime: None,
        only: None,
        tl2cgen_override: None,
        pmu,
        pmu_status,
        require_pmu,
        process,
        timing_began: 0,
        interface_override: BTreeMap::new(),
    };
    let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
    let mut stops = serde_json::Map::new();
    // Production predict's outputs, through the serving interface.
    methods[0].select(interfaces.get("serving").copied().unwrap_or(0));
    let g = Gathered::new(&cell, &check_groups);
    let check = match outputs(
        &mut methods[0],
        &g,
        cell.n_features,
        ctx.config.batch_rows,
        Mode::Serving,
    ) {
        Ok(got) if got.len() == expected.len() => {
            let (max_abs, bad) = compare(&got, &expected, false, Precision::F32.tolerance());
            json!({"passed": bad == 0, "mismatches": bad, "max_abs": max_abs, "rows": got.len()})
        }
        Ok(got) => json!({"passed": false, "error":
            format!("{} outputs, {} expected", got.len(), expected.len())}),
        Err(e) => json!({"passed": false, "error": e}),
    };
    for name in &modes {
        let mode = Mode::parse(name)?;
        // The parent's interface, so the processes differ only in themselves.
        let chosen = interfaces.get(name).copied().unwrap_or(0);
        methods[0].allowed = vec![chosen];
        methods[0].select(chosen);
        // One untimed round first: the parent's methods ran a whole cell's
        // validation and rounds before their timed passes, a fresh child none.
        warm_round(&ctx, &cell, &groups, &mut methods[0], mode);
        let s = time_mode(
            &mut ctx,
            &cell,
            &groups,
            &mut methods,
            mode,
            &mut samples,
            &mut hw,
        );
        stops.insert(name.clone(), serde_json::to_value(&s)?);
    }
    Ok(json!({
        "samples": samples.rows_json(),
        "hw": hw.rows_json(),
        "stops": stops,
        "check": check,
        "pmu_status": ctx.pmu_status,
    }))
}

/// One untimed pass of a method over every batch of the mode's schedule, in the
/// timed rounds' batch order (the schedule is seeded per cell).
#[cfg_attr(not(feature = "external-bench"), allow(dead_code))]
fn warm_round(ctx: &Context, cell: &Cell, groups: &[usize], m: &mut TimedMethod, mode: Mode) {
    let cfg = &ctx.config;
    let seed = crate::schedule::cell_seed(cfg.seed, &cell.id);
    let schedule = Schedule::new(groups.len(), 1, cfg.target_batches, seed);
    for batch in &schedule.batches {
        let ids: Vec<usize> = batch.iter().map(|&i| groups[i]).collect();
        let g = Gathered::new(cell, &ids);
        let calls = (mode == Mode::Batch).then(|| calls_for(&g, cell.n_features, cfg.batch_rows));
        let _ = run_pass(m, &g, calls.as_deref(), cell.n_features, |_, _, _, _| {});
    }
}

#[cfg(not(feature = "external-bench"))]
pub fn xgboost_process(_: &Value) -> Result<Value> {
    anyhow::bail!("sweep_bench was built without external-bench")
}

/// What a finished cell reports to the run.
#[derive(Debug, Clone)]
pub struct CellOutcome {
    pub id: String,
    pub dir: PathBuf,
    pub timed_methods: usize,
    pub excluded: usize,
    pub seconds: f64,
    /// Each timed method's ticks and rows over every round, by mode and then
    /// `method/variant`.
    pub totals: BTreeMap<String, BTreeMap<String, [u64; 2]>>,
    /// Each timed method's multi-row interface per mode.
    pub interfaces: BTreeMap<String, BTreeMap<String, String>>,
}

/// A cell that may not be published as a completed measurement. Its diagnostics
/// go to `<run_dir>/failed/<cell>.json`, never into `cells/`.
fn refuse(run_dir: &Path, cell_id: &str, reason: &str, diagnostics: &Value) -> anyhow::Error {
    let name = Cell::slug(cell_id).unwrap_or_else(|_| "unsafe-cell-id".into());
    let dir = run_dir.join("failed");
    let written = std::fs::create_dir_all(&dir).and_then(|()| {
        let doc = json!({"id": cell_id, "reason": reason, "diagnostics": diagnostics});
        std::fs::write(dir.join(format!("{name}.json")), doc.to_string())
    });
    match written {
        Ok(()) => anyhow::anyhow!("{cell_id}: {reason} (diagnostics in failed/{name}.json)"),
        Err(e) => anyhow::anyhow!("{cell_id}: {reason} (diagnostics not written: {e})"),
    }
}

/// Validate, count and time one cell; write its directory under `run_dir`.
///
/// With `timing`, the cell is published only when it is ready and matches its
/// execution entry, the stage oracle passes on a nonempty sample, every required
/// method validates and times to completion, and, when hardware counters are
/// required, every block's group was read in full. Otherwise its diagnostics go
/// to `failed/` and the call returns an error.
#[expect(
    clippy::too_many_lines,
    reason = "one cell from loading to its directory"
)]
pub fn run_cell(
    ctx: &mut Context,
    mc: &ManifestCell,
    cell_dir: &Path,
    run_dir: &Path,
    resume_key: &str,
    timing: bool,
) -> Result<CellOutcome> {
    let started = Instant::now();
    let cell = Cell::load(cell_dir).with_context(|| format!("loading {}", cell_dir.display()))?;
    cell.verify_files()?;
    if timing && let Err(e) = crate::run::check_entry(mc, &cell.doc) {
        return Err(refuse(run_dir, &mc.id, &format!("{e:#}"), &Value::Null));
    }
    let cfg = ctx.config.clone();
    let max_rows = cfg.batch_rows.max(cell.max_group_rows());
    let mut loaded = load_methods(&cell, mc, ctx, max_rows)?;

    // The groups every method times: all of them, or above the row cap a seeded
    // sample stratified by group size.
    let sizes: Vec<usize> = cell.offsets.windows(2).map(|w| w[1] - w[0]).collect();
    let sample = crate::schedule::sample_groups(
        &sizes,
        cfg.max_rows_per_round,
        cfg.min_groups_per_round,
        // Its own stream: the schedule's order uses cell_seed(seed, id).
        crate::schedule::cell_seed(cfg.seed, &format!("{}#sample", cell.id)),
    );
    let timed_groups: Vec<usize> = sample
        .as_ref()
        .map_or_else(|| (0..cell.n_groups()).collect(), |s| s.groups.clone());

    // Validation on exactly the groups the timing covers, gathered as it gathers
    // them: every timed input is checked, and no more.
    let order = &timed_groups;
    let vgroups = Gathered::new(&cell, order);
    let oracle = check_oracle(&cell, &loaded.forests[""])?;
    let oracle_ok = oracle["status"] == "pass" && oracle["rows"].as_u64().unwrap_or(0) > 0;
    if timing && !oracle_ok {
        return Err(refuse(
            run_dir,
            &cell.id,
            "the stage oracle did not pass",
            &oracle,
        ));
    }
    let modes = Mode::parse_list(&mc.modes)?;
    let (checks, prod) = validate(&cell, &mut loaded, &vgroups, cfg.batch_rows, &modes);
    let reference = cell.reference.as_ref().map(|r| {
        let want: Vec<f64> = order
            .iter()
            .flat_map(|&g| r[cell.offsets[g]..cell.offsets[g + 1]].iter().copied())
            .collect();
        let (max_abs, _) = compare(&prod, &want, false, 0.0);
        json!({"max_abs_vs_predict": max_abs, "rows": want.len(),
               "note": "prepare's reference: GTIL for LightGBM, native XGBoost; informational"})
    });

    let mut counters_t = output::counters_table();
    let ablations = counters(ctx, &cell, &mut loaded, &mut counters_t)?;
    let required: Vec<&str> = MethodSet::parse(&mc.methods)?
        .required(&cell.framework)
        .iter()
        .copied()
        .filter(|m| wanted(ctx, m))
        .collect();
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|r| !loaded.methods.iter().any(|m| m.method == *r))
        .collect();
    if timing && !missing.is_empty() {
        let diag = json!({"checks": checks, "excluded": loaded.excluded, "oracle": oracle});
        let reason = format!("required methods failed to load or validate: {missing:?}");
        return Err(refuse(run_dir, &cell.id, &reason, &diag));
    }

    if timing && timed_groups.is_empty() {
        return Err(refuse(run_dir, &cell.id, "no groups to time", &Value::Null));
    }
    let mut is_timed = vec![false; cell.n_groups()];
    for &g in &timed_groups {
        is_timed[g] = true;
    }
    let mut groups_t = output::groups_table();
    for (g, &timed) in is_timed.iter().enumerate() {
        groups_t.push(&[
            V::S(&ctx.suite),
            V::S(&cell.id),
            V::U32(g as u32),
            V::I64(cell.entities[g]),
            V::U64(cell.offsets[g] as u64),
            V::U32((cell.offsets[g + 1] - cell.offsets[g]) as u32),
            V::U32(u32::from(timed)),
        ]);
    }

    let mut samples = output::samples_table();
    let mut hw = output::hw_table();
    let mut stops = serde_json::Map::new();
    let mut failed_timing: Vec<Excluded> = Vec::new();
    let mut totals: BTreeMap<String, BTreeMap<String, [u64; 2]>> = BTreeMap::new();
    let mut incomplete_reads = 0usize;
    if timing && !loaded.methods.is_empty() {
        ctx.timing_began += 1;
        for &mode in &modes {
            let s = time_mode(
                ctx,
                &cell,
                &timed_groups,
                &mut loaded.methods,
                mode,
                &mut samples,
                &mut hw,
            );
            eprintln!(
                "    {}: {} rounds of {} blocks, stopped by {} ({:.1}s)",
                mode.name(),
                s.rounds,
                s.blocks_per_round,
                s.reason,
                s.seconds
            );
            failed_timing.extend(s.failures.iter().cloned());
            incomplete_reads += s.incomplete_reads;
            stops.insert(mode.name().into(), serde_json::to_value(&s)?);
            totals.insert(mode.name().to_string(), s.totals);
        }
    }
    if ctx.require_pmu && incomplete_reads > 0 {
        let reason =
            format!("{incomplete_reads} blocks' hardware counters were multiplexed or not read");
        return Err(refuse(run_dir, &cell.id, &reason, &Value::Object(stops)));
    }
    // A method that failed during timing loses its whole measurement, both modes.
    if !failed_timing.is_empty() {
        let drop: Vec<(String, String)> = failed_timing
            .iter()
            .map(|e| (e.method.clone(), e.variant.clone()))
            .collect();
        samples.retain_pairs("method", "variant", &drop);
        hw.retain_pairs("method", "variant", &drop);
        loaded.methods.retain(|m| {
            !drop
                .iter()
                .any(|(mm, vv)| *mm == m.method && *vv == m.variant)
        });
        for e in &failed_timing {
            loaded.excluded.push(Excluded {
                reason: format!("failed during timing: {}", e.reason),
                ..e.clone()
            });
        }
        if let Some(r) = required.iter().find(|r| drop.iter().any(|(m, _)| m == *r)) {
            let reason = format!("required method {r} failed during timing");
            return Err(refuse(
                run_dir,
                &cell.id,
                &reason,
                &json!({"failures": failed_timing}),
            ));
        }
    }

    // XGBoost's extra processes, tagged in the samples and counters.
    let xgboost_processes = if timing {
        let shared = Shared {
            cell_dir,
            cell: &cell,
            forest: &loaded.forests[""],
            groups: &timed_groups,
            modes: &modes,
            max_rows,
            totals: &totals,
        };
        extra_processes(ctx, &mut loaded.methods, &shared, &mut samples, &mut hw)
    } else {
        Value::Null
    };
    let out = output::CellDir::create(run_dir, &Cell::slug(&cell.id)?, &cell.id)?;
    samples.write(&out.path("samples.parquet"))?;
    groups_t.write(&out.path("groups.parquet"))?;
    counters_t.write(&out.path("counters.parquet"))?;
    hw.write(&out.path("hw.parquet"))?;

    let timed: Vec<Value> = loaded
        .methods
        .iter()
        .map(|m| {
            json!({"method": m.method, "variant": m.variant,
                   "interface_one_row": m.interface(1, false),
                   "candidates": m.candidates(false),
                   "validated": m.allowed.iter().map(|&i| m.candidates(false)[i]).collect::<Vec<_>>(),
                   "interfaces": m.chosen,
                   "precision": m.precision(),
                   "tolerance": m.precision().tolerance(),
                   "settings": m.settings()})
        })
        .collect();
    let forest = &loaded.forests[""];
    let manifest = json!({
        "resume_key": resume_key,
        "suite": ctx.suite,
        "id": cell.id,
        "cell": cell.doc,
        "model_bytes": std::fs::metadata(&cell.model_treelite).map_or(0, |m| m.len()),
        "model": {
            "trees": forest.trees().len(),
            "nodes": forest.nodes().len(),
            "bitset_bytes": forest.bitset_bytes(),
            "threshold_type": format!("{:?}", forest.threshold_type()),
            "exact_sums": forest.exact_sums(),
        },
        "load_seconds": loaded.load_seconds,
        "groups": cell.n_groups(),
        "timed_groups": timed_groups.len(),
        "sample": sample,
        "rows": cell.n_rows,
        "validation": {
            "groups": order.len(),
            "oracle": oracle,
            "checks": checks,
            "reference": reference,
        },
        "ablations": ablations,
        "required_methods": required,
        "methods": timed,
        "excluded": loaded.excluded,
        "stop": stops,
        "xgboost_processes": xgboost_processes,
        "run": cfg,
        "seconds": started.elapsed().as_secs_f64(),
    });
    output::write_json(&out.path("manifest.json"), &manifest)?;
    let dir = out.finish()?;
    Ok(CellOutcome {
        id: cell.id,
        dir,
        timed_methods: loaded.methods.len(),
        excluded: loaded.excluded.len(),
        seconds: started.elapsed().as_secs_f64(),
        interfaces: loaded
            .methods
            .iter()
            .map(|m| {
                let by_mode = m.chosen.iter().map(|(k, v)| (k.clone(), (*v).to_string()));
                (m.method.clone(), by_mode.collect())
            })
            .collect(),
        // Only the methods whose measurement stands.
        totals: totals
            .into_iter()
            .map(|(mode, t)| {
                let kept = t
                    .into_iter()
                    .filter(|(k, _)| {
                        loaded
                            .methods
                            .iter()
                            .any(|m| *k == format!("{}/{}", m.method, m.variant))
                    })
                    .collect();
                (mode, kept)
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
                    .filter(|n| n.skip == -1)
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
            &LoadOptions::default(),
        )
        .unwrap()
    }

    fn facts(f: &Forest, rows: &[f64]) -> Facts {
        Facts {
            monotonic: false,
            exact: f.exact_sums(),
            varying_predicates: !f.predicate_masks(rows, true).is_empty(),
            row_leaves: (rows.len() / 2 * f.trees().len()) as u64,
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

    const PATH_FLAGS: [&str; 4] = [
        "disable_varying_precompute",
        "disable_unsplit",
        "disable_predicate_sweep",
        "disable_exact_sums",
    ];

    #[test]
    fn single_row_groups_show_every_path() {
        let f = fixture_forest(&[0]);
        let rows = [0.5, 1.0];
        let facts = facts(&f, &rows);
        assert!(facts.varying_predicates && facts.exact);
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
    fn two_identical_rows_keep_the_exact_sums_signature() {
        let f = fixture_forest(&[0]);
        let rows = [0.5, 1.0, 0.5, 1.0];
        let facts = facts(&f, &rows);
        let (exact, per_row) = (
            counted(&f, &rows, &[]),
            counted(&f, &rows, &["disable_exact_sums"]),
        );
        // Equal totals: two adds per run of two rows, one per row per tree.
        assert_eq!((exact.leaf_adds, per_row.leaf_adds), (4, 4));
        assert!(signature("disable_exact_sums", &per_row, &facts));
        for flag in PATH_FLAGS {
            assert!(
                signature(flag, &counted(&f, &rows, &[flag]), &facts),
                "{flag}"
            );
        }
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
        let spec = VariantSpec {
            runtime: vec!["disable_exact_sums".into()],
            load: Vec::new(),
        };
        assert_eq!(no_op("disable_exact_sums", &spec, &facts), None);
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
}

#[cfg(test)]
mod timing_tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::external::ExternalMethod;

    /// Every call a fake method receives: its name, the first row's group and the
    /// row count.
    type Log = Rc<RefCell<Vec<(String, usize, usize)>>>;

    struct Fake {
        name: String,
        log: Log,
        per_row: u64,
        /// Candidate interfaces and each one's ticks per row, when there is a choice.
        candidates: Vec<(&'static str, u64)>,
        selected: usize,
        /// Report a conversion of one tick per row.
        split: bool,
    }

    impl ExternalMethod for Fake {
        fn interface(&self, n: usize, multi: bool) -> &'static str {
            match self.candidates.get(self.selected) {
                Some(&(name, _)) if n > 1 || multi => name,
                _ => "fake",
            }
        }
        fn candidates(&self) -> Vec<&'static str> {
            if self.candidates.is_empty() {
                vec!["fake"]
            } else {
                self.candidates.iter().map(|c| c.0).collect()
            }
        }
        fn select(&mut self, i: usize) {
            self.selected = i;
        }
        fn precision(&self) -> Precision {
            Precision::F64
        }
        fn predict_timed(&mut self, rows: &[f64], n: usize, _: bool) -> Result<u64, String> {
            let name = format!("{}{}", self.name, self.selected);
            self.log.borrow_mut().push((name, rows[0] as usize, n));
            let per_row = self
                .candidates
                .get(self.selected)
                .map_or(self.per_row, |c| c.1);
            Ok(per_row * n as u64)
        }
        fn output(&self, n: usize) -> Vec<f64> {
            vec![0.0; n]
        }
        fn convert_only(&mut self, _: &[f64], _: usize, _: bool) -> bool {
            self.split
        }
    }

    /// A cell of `sizes.len()` groups whose rows hold their group index.
    pub(super) fn cell(sizes: &[usize]) -> Cell {
        let nf = 2;
        let mut offsets = vec![0];
        let mut data = Vec::new();
        for (g, &s) in sizes.iter().enumerate() {
            data.extend(std::iter::repeat_n(g as f64, s * nf));
            offsets.push(offsets.last().unwrap() + s);
        }
        Cell {
            id: "test/cell".into(),
            dir: PathBuf::new(),
            doc: Value::Null,
            framework: "test".into(),
            status: "ready".into(),
            n_rows: *offsets.last().unwrap(),
            n_features: nf,
            entities: (0..sizes.len() as i64).collect(),
            offsets,
            walker_config: PathBuf::new(),
            model_treelite: PathBuf::new(),
            model_native: None,
            reference: None,
            oracle: None,
            tl2cgen: None,
            lleaves: None,
            quickscorer: Err(String::new()),
            quickscorer_sha256: None,
            data,
        }
    }

    pub(super) fn context(config: RunConfig) -> Context {
        Context {
            suite: "test".into(),
            config,
            natives: Natives::default(),
            tl2cgen_runtime: None,
            only: None,
            tl2cgen_override: None,
            pmu: None,
            pmu_status: "test".into(),
            require_pmu: false,
            process: 0,
            timing_began: 0,
            interface_override: BTreeMap::new(),
        }
    }

    fn fake(name: &str, log: &Log, per_row: u64) -> TimedMethod {
        choosing(name, log, per_row, vec![])
    }

    fn choosing(
        name: &str,
        log: &Log,
        per_row: u64,
        candidates: Vec<(&'static str, u64)>,
    ) -> TimedMethod {
        let m = Fake {
            name: name.into(),
            log: log.clone(),
            per_row,
            candidates,
            selected: 0,
            split: name == "q",
        };
        TimedMethod::new(name, "-", Engine::External(Box::new(m)), 64)
    }

    #[test]
    fn extra_xgboost_processes_run_only_where_configured() {
        let log = Log::default();
        let modes = [Mode::Serving];
        let (cell, forest) = (cell(&[1]), super::tests::fixture_forest(&[]));
        let totals = BTreeMap::new();
        let shared = Shared {
            cell_dir: Path::new("."),
            cell: &cell,
            forest: &forest,
            groups: &[0],
            modes: &modes,
            max_rows: 1,
            totals: &totals,
        };
        let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
        let mut run = |config: RunConfig, name: &str| {
            let mut methods = [fake(name, &log, 1)];
            extra_processes(
                &context(config),
                &mut methods,
                &shared,
                &mut samples,
                &mut hw,
            )
        };
        // Not an XGBoost cell: nothing to record.
        assert_eq!(run(RunConfig::default(), "lightgbm_native"), Value::Null);
        let elsewhere = RunConfig {
            xgboost_process_arches: vec!["no-such-arch".into()],
            ..RunConfig::default()
        };
        let r = run(elsewhere, "xgboost_native");
        assert_eq!(r["count"], 0);
        assert!(
            r["skipped"]
                .as_str()
                .unwrap()
                .contains("not in xgboost_process_arches")
        );
        let none = RunConfig {
            xgboost_processes: 0,
            ..RunConfig::default()
        };
        assert_eq!(
            run(none, "xgboost_native")["skipped"],
            "xgboost_processes is 0"
        );
        let here = RunConfig {
            xgboost_process_arches: vec![std::env::consts::ARCH.into()],
            ..RunConfig::default()
        };
        assert_eq!(
            run(here, "xgboost_native")["skipped"],
            "no native XGBoost library"
        );
        let s = process_summary(2, "serving", Some([300, 3]), "ok");
        assert_eq!(s["ticks_per_row"].as_f64(), Some(100.0));
        assert_eq!(
            process_summary(2, "serving", None, "failed")["ticks_per_row"],
            Value::Null
        );
    }

    #[test]
    fn the_childs_warm_up_round_follows_the_timed_batches() {
        let cell = cell(&[1; 30]);
        let config = RunConfig {
            target_batches: 4,
            ..RunConfig::default()
        };
        let ctx = context(config.clone());
        let log = Log::default();
        let mut m = fake("w", &log, 1);
        let groups: Vec<usize> = (0..30).rev().collect();
        warm_round(&ctx, &cell, &groups, &mut m, Mode::Serving);
        let seed = crate::schedule::cell_seed(config.seed, &cell.id);
        let want: Vec<usize> = Schedule::new(30, 1, 4, seed)
            .batches
            .iter()
            .flatten()
            .map(|&i| groups[i])
            .collect();
        let got: Vec<usize> = log.borrow().iter().map(|c| c.1).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn a_child_with_unread_counters_or_other_outputs_is_invalid() {
        let modes = [Mode::Serving];
        let ok = json!({"check": {"passed": true}, "pmu_status": "ok",
                        "stops": {"serving": {"failures": [], "incomplete_reads": 0}}});
        assert!(child_invalid(&ok, &modes, true).is_empty());
        let mut unread = ok.clone();
        unread["stops"]["serving"]["incomplete_reads"] = json!(2);
        assert_eq!(child_invalid(&unread, &modes, true).len(), 1);
        // Without counters in the parent, unread blocks do not count.
        assert!(child_invalid(&unread, &modes, false).is_empty());
        let mut no_pmu = ok.clone();
        no_pmu["pmu_status"] = json!("perf_event_open: EACCES");
        assert!(child_invalid(&no_pmu, &modes, true)[0].contains("no hardware counters"));
        let mut differ = ok;
        differ["check"] = json!({"passed": false, "mismatches": 3});
        assert!(child_invalid(&differ, &modes, false)[0].contains("outputs differ"));
    }

    #[test]
    fn the_conversion_pass_follows_the_timed_passes_outside_the_samples() {
        let cell = cell(&[2; 6]);
        let config = RunConfig {
            target_batches: 3,
            min_rounds: 2,
            max_rounds: 2,
            hardware_counters: false,
            ..RunConfig::default()
        };
        let log = Log::default();
        let mut methods = vec![fake("q", &log, 10), fake("b", &log, 20)];
        let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
        let s = time_mode(
            &mut context(config),
            &cell,
            &(0..6).collect::<Vec<_>>(),
            &mut methods,
            Mode::Serving,
            &mut samples,
            &mut hw,
        );
        // One pass per round, of the method with a conversion, on the last batch:
        // 2 groups of 2 rows at 10 ticks a row, and the conversions in one interval.
        assert_eq!(s.conversion.len(), 2);
        let c = &s.conversion[0];
        assert_eq!(
            (c.method.as_str(), c.batch, c.calls, c.rows),
            ("q", 2, 2, 4)
        );
        assert_eq!(c.ticks, 40);
        assert_eq!(samples.len(), 2 * 2 * 6);
    }

    #[test]
    fn all_one_row_serving_cells_record_the_single_row_interface() {
        let cell = cell(&[1; 6]);
        let config = RunConfig {
            min_rounds: 1,
            max_rounds: 1,
            hardware_counters: false,
            ..RunConfig::default()
        };
        let log = Log::default();
        let mut methods = vec![choosing("a", &log, 0, vec![("mat", 30), ("loop", 10)])];
        let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
        let s = time_mode(
            &mut context(config),
            &cell,
            &(0..6).collect::<Vec<_>>(),
            &mut methods,
            Mode::Serving,
            &mut samples,
            &mut hw,
        );
        assert!(s.probes.is_empty());
        assert_eq!(methods[0].chosen["serving"], "fake");
    }

    #[test]
    fn the_probe_times_the_faster_candidate_and_records_it() {
        let cell = cell(&[1, 3, 3, 3, 3, 3]);
        let config = RunConfig {
            target_batches: 2,
            min_rounds: 2,
            max_rounds: 2,
            hardware_counters: false,
            ..RunConfig::default()
        };
        let log = Log::default();
        let mut methods = vec![
            choosing("a", &log, 0, vec![("mat", 30), ("loop", 10)]),
            fake("b", &log, 20),
        ];
        let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
        let s = time_mode(
            &mut context(config),
            &cell,
            &(0..6).collect::<Vec<_>>(),
            &mut methods,
            Mode::Serving,
            &mut samples,
            &mut hw,
        );
        // One probe, of the method with a choice, on its warm-up batch.
        assert_eq!(s.probes.len(), 1);
        let p = &s.probes[0];
        assert_eq!(
            (p.method.as_str(), p.chosen.as_str(), p.batch),
            ("a", "loop", 1)
        );
        let ticks: Vec<Option<u64>> = p.candidates.iter().map(|c| c.ticks).collect();
        // The runner-up, the multi-row call at 30 ticks a row, is 200% slower.
        assert!((p.margin_pct.unwrap() - 200.0).abs() < 1e-9);
        // Two timed passes each over the batch's rows: 30 and 10 ticks a row.
        assert_eq!(ticks, [Some(60 * p.rows as u64), Some(20 * p.rows as u64)]);
        assert_eq!(methods[0].chosen["serving"], "loop");
        assert_eq!(methods[1].chosen["serving"], "fake");
        // The probe ran each candidate three times on the warm-up batch: one warm
        // pass and two timed ones; every later call of "a" used the loop.
        let calls = log.borrow();
        let a: Vec<&String> = calls
            .iter()
            .filter(|c| c.0.starts_with('a'))
            .map(|c| &c.0)
            .collect();
        let probe_calls = 3
            * 2
            * Schedule::new(6, 2, 2, crate::schedule::cell_seed(42, "test/cell")).batches[1].len();
        assert!(a[..probe_calls].contains(&&"a0".to_string()));
        assert!(a[probe_calls..].iter().all(|n| *n == "a1"));
    }

    #[test]
    fn a_dropped_method_takes_its_variant_spec_with_it() {
        let log = Log::default();
        let mut l = Loaded {
            methods: Vec::new(),
            excluded: Vec::new(),
            forests: BTreeMap::new(),
            load_seconds: BTreeMap::new(),
        };
        for name in ["a", "b", "c"] {
            let mut m = fake(name, &log, 1);
            m.spec = Some(VariantSpec {
                runtime: vec![name.into()],
                load: Vec::new(),
            });
            l.methods.push(m);
        }
        l.methods.push(fake("baseline", &log, 1));
        drop_failed(&mut l, &[false, true, false, false]);
        let left: Vec<(&str, Option<&str>)> = l
            .methods
            .iter()
            .map(|m| {
                let spec = m.spec.as_ref().map(|s| s.runtime[0].as_str());
                (m.method.as_str(), spec)
            })
            .collect();
        assert_eq!(
            left,
            [("a", Some("a")), ("c", Some("c")), ("baseline", None)]
        );
    }

    #[test]
    fn a_cell_that_fails_before_timing_does_not_count_as_timed() {
        let mut ctx = context(RunConfig::default());
        let mc = ManifestCell {
            id: "x/y".into(),
            dir: PathBuf::from("no/such/cell"),
            status: "ready".into(),
            key: "k".into(),
            methods: "factorial".into(),
            variants: Vec::new(),
            modes: vec!["serving".into()],
        };
        let tmp = std::env::temp_dir().join(format!("tw-notimed-{}", std::process::id()));
        assert!(run_cell(&mut ctx, &mc, Path::new("/no/such/cell"), &tmp, "", true).is_err());
        assert_eq!(ctx.timing_began, 0);
    }

    #[test]
    fn the_minimum_rounds_bind_before_the_budget() {
        let cell = cell(&[1; 6]);
        for (precision_pct, reason) in [(0.0, "time_budget"), (1e9, "precision")] {
            let config = RunConfig {
                min_rounds: 3,
                max_rounds: 10,
                mode_budget_secs: 0.0,
                precision_pct,
                hardware_counters: false,
                ..RunConfig::default()
            };
            let log = Log::default();
            let mut methods = vec![fake("a", &log, 10), fake("b", &log, 20)];
            let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
            let s = time_mode(
                &mut context(config),
                &cell,
                &(0..6).collect::<Vec<_>>(),
                &mut methods,
                Mode::Serving,
                &mut samples,
                &mut hw,
            );
            // A spent budget and a met target both wait for the third round, an
            // odd one: stopping is not held to the A B B A cycle.
            assert_eq!((s.rounds, s.reason), (3, reason));
        }
    }

    #[test]
    fn every_method_times_the_same_group_sample() {
        let sizes: Vec<usize> = (0..40).map(|g| 1 + g % 3).collect();
        let cell = cell(&sizes);
        let sample = crate::schedule::sample_groups(&sizes, 20, 0, 5).unwrap();
        let config = RunConfig {
            min_rounds: 2,
            max_rounds: 2,
            hardware_counters: false,
            ..RunConfig::default()
        };
        let log = Log::default();
        let mut methods = vec![fake("a", &log, 10), fake("b", &log, 20)];
        let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
        time_mode(
            &mut context(config),
            &cell,
            &sample.groups,
            &mut methods,
            Mode::Serving,
            &mut samples,
            &mut hw,
        );
        for name in ["a", "b"] {
            let mut seen: Vec<usize> = log
                .borrow()
                .iter()
                .filter(|(m, _, _)| *m == format!("{name}0"))
                .map(|&(_, g, _)| g)
                .collect();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen, sample.groups, "{name}");
        }
    }

    #[test]
    fn each_method_times_every_batch_in_its_own_phase() {
        let cell = cell(&[2; 10]);
        let config = RunConfig {
            target_batches: 3,
            min_rounds: 4,
            max_rounds: 4,
            precision_pct: 0.0,
            hardware_counters: false,
            ..RunConfig::default()
        };
        let mut ctx = context(config.clone());
        let log = Log::default();
        let mut methods = vec![fake("a", &log, 10), fake("b", &log, 20)];
        let (mut samples, mut hw) = (output::samples_table(), output::hw_table());
        let groups: Vec<usize> = (0..10).collect();
        let s = time_mode(
            &mut ctx,
            &cell,
            &groups,
            &mut methods,
            Mode::Serving,
            &mut samples,
            &mut hw,
        );
        assert_eq!((s.rounds, s.blocks_per_round), (4, 3));
        let schedule = Schedule::new(
            10,
            2,
            3,
            crate::schedule::cell_seed(config.seed, "test/cell"),
        );
        let batch = |b: usize| schedule.batches[b].clone();
        // Each phase: the warm-up on the last batch, then batches 0, 1, 2.
        let mut want = Vec::new();
        for round in 0..4 {
            for mi in schedule.method_order(round) {
                let name = ["a", "b"][mi];
                let warm = batch(schedule.warm_batch());
                for b in std::iter::once(warm).chain((0..3).map(batch)) {
                    want.extend(b.iter().map(|&g| (format!("{name}0"), g, 2)));
                }
            }
        }
        assert_eq!(*log.borrow(), want);
        // A B B A across rounds, and every method times every group every round.
        let order: Vec<Vec<usize>> = (0..4).map(|r| schedule.method_order(r)).collect();
        assert_eq!(order[0], order[1].iter().rev().copied().collect::<Vec<_>>());
        assert_eq!(order[0], order[2]);
        assert_eq!(samples.len(), 4 * 2 * 10);
    }
}
