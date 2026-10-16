//! XGBoost's extra processes: the parent's side and the child's.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::{Value, json};
use treewalker_gbdt::Forest;

use crate::artifacts::{Cell, RunConfig};
use crate::methods::TimedMethod;
use crate::output::Table;

use super::Context;
use super::load::Gathered;
use super::timing::Mode;

/// What every extra process times like the main one: the groups, modes and
/// output size, and the main process's ticks and rows per mode and
/// `method/variant`.
pub(super) struct Shared<'a> {
    pub(super) cell_dir: &'a Path,
    /// The cell and its production forest, for the output check.
    pub(super) cell: &'a Cell,
    pub(super) forest: &'a Forest,
    pub(super) groups: &'a [usize],
    pub(super) modes: &'a [Mode],
    pub(super) max_rows: usize,
    pub(super) totals: &'a BTreeMap<String, BTreeMap<String, [u64; 2]>>,
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
pub(super) fn process_summary(
    process: u32,
    mode: &str,
    ticks_rows: Option<[u64; 2]>,
    status: &str,
) -> Value {
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
pub(super) fn extra_processes(
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
pub(super) fn child_invalid(doc: &Value, modes: &[Mode], counters: bool) -> Vec<String> {
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
    use anyhow::Context as _;

    use super::timing::{time_mode, warm_round};
    use super::validate::{compare, outputs};
    use crate::artifacts::Natives;
    use crate::external::{Precision, XgBoostBench};
    use crate::methods::Engine;
    use crate::output;
    use crate::pmu::Pmu;
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
        invocation: 0,
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

#[cfg(not(feature = "external-bench"))]
pub fn xgboost_process(_: &Value) -> Result<Value> {
    anyhow::bail!("sweep_bench was built without external-bench")
}
