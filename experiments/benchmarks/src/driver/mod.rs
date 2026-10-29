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
use treewalker_gbdt::research::Ablation;

use crate::artifacts::{Cell, ManifestCell, Natives, RunConfig};
use crate::output::{self, V};
use crate::pmu::Pmu;
use crate::suites::MethodSet;

use counters::{counters, layout_stats};
use load::{Gathered, load_methods, wanted};
pub use processes::xgboost_process;
use processes::{Shared, extra_processes};
use timing::{Mode, time_mode};
use validate::{check_oracle, compare, validate};

mod counters;
mod load;
mod processes;
mod timing;
mod validate;

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
    /// This invocation's index in `run.json`'s `invocations`, which the runner
    /// appends at its end; a failure in `failed/` records it.
    pub invocation: usize,
}

/// A method or variant that is not timed, and why.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Excluded {
    pub method: String,
    pub variant: String,
    pub reason: String,
}

/// What a finished cell reports to the run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
/// go to `<run_dir>/failed/<cell>.json`, never into `cells/`, with the invocation
/// it failed in, so a later attempt that finished can be told from an earlier one.
fn refuse(
    run_dir: &Path,
    invocation: usize,
    cell_id: &str,
    reason: &str,
    diagnostics: &Value,
) -> anyhow::Error {
    let name = Cell::slug(cell_id).unwrap_or_else(|_| "unsafe-cell-id".into());
    let dir = run_dir.join("failed");
    let written = std::fs::create_dir_all(&dir).and_then(|()| {
        let doc = json!({"id": cell_id, "invocation": invocation, "reason": reason,
                         "diagnostics": diagnostics});
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
        return Err(refuse(
            run_dir,
            ctx.invocation,
            &mc.id,
            &format!("{e:#}"),
            &Value::Null,
        ));
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
    let mut stages_t = output::stages_table();
    let oracle = check_oracle(&cell, &loaded.forests[""], &ctx.suite, &mut stages_t)?;
    let oracle_ok = oracle["status"] == "pass" && oracle["rows"].as_u64().unwrap_or(0) > 0;
    if timing && !oracle_ok {
        return Err(refuse(
            run_dir,
            ctx.invocation,
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
        return Err(refuse(run_dir, ctx.invocation, &cell.id, &reason, &diag));
    }

    if timing && timed_groups.is_empty() {
        return Err(refuse(
            run_dir,
            ctx.invocation,
            &cell.id,
            "no groups to time",
            &Value::Null,
        ));
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
            V::B(timed),
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
        return Err(refuse(
            run_dir,
            ctx.invocation,
            &cell.id,
            &reason,
            &Value::Object(stops),
        ));
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
                ctx.invocation,
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
    stages_t.write(&out.path("stages.parquet"))?;

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
            "layout": layout_stats(forest),
        },
        // The instruction-set level the cell's predictors ran at: each is chosen when a
        // predictor is made, from the CPU and TREEWALKER_ISA, which a run does not change.
        "isa": forest.research_predictor(Ablation::default()).isa(),
        "load_seconds": loaded.load_seconds,
        "memory": {"source": crate::memory::SOURCE, "methods": loaded.memory},
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
mod tests;
#[cfg(test)]
mod timing_tests;
