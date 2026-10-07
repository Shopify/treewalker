//! Runs: `run.json`, resume keys, preflight, and the loop over a manifest's cells.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use serde_json::{Value, json};

use crate::artifacts::{Cell, Manifest, ManifestCell, Natives, RunConfig};
use crate::driver::{self, Context};
use crate::output::{self, Kind, Table, V};
use crate::pmu::{self, Pmu};
use crate::suites::MethodSet;
use crate::{data, system, timer};

/// Version of the run directory layout and its tables.
///
/// 2: samples and hw gain `process`, groups gains `timed`, and the run gains
/// `sentinel.parquet` and `sentinel/`; `rowloop.parquet` is gone.
/// 3: the manifest's `sample` records `expected_rows`, `overshoot` and an
/// `exception` only where the minimum binds; its seed is its own stream.
/// 4: `run.json`'s sentinel records hold raw ticks and rows (`totals`), the
/// analysis derives drift and the first-load effect, and `sentinel.parquet` is
/// gone.
/// 5: the manifest's `stop` loses `instructions` and each conversion pass its
/// `share`, and `xgboost_processes` its instruction totals (the `hw` table
/// holds every block's counters).
pub const OUTPUT_SCHEMA: u32 = 5;

/// Options shared by `run`, `cell` and `preflight`.
#[derive(Debug, Clone, Default)]
pub struct Options {
    pub lgb_lib: Option<PathBuf>,
    pub xgb_lib: Option<PathBuf>,
    pub tl2cgen_runtime: Option<PathBuf>,
    pub tl2cgen_override: Option<PathBuf>,
    pub only: Option<Vec<String>>,
    /// Fail when hardware counters cannot be read.
    pub require_pmu: bool,
    /// A JSON file of host facts the runner cannot see (image, turbo mode, tools).
    pub system_info: Option<PathBuf>,
}

fn native(path: &Path, version: &str) -> Result<crate::artifacts::NativeLib> {
    Ok(crate::artifacts::NativeLib {
        path: path.to_path_buf(),
        sha256: data::sha256_file(path)?,
        version: version.into(),
        commit: String::new(),
        build: Value::Null,
    })
}

/// The run's context: natives from the manifest unless overridden, and the PMU.
pub fn context(
    suite: &str,
    config: RunConfig,
    natives: &Natives,
    tl2cgen_runtime: Option<(PathBuf, String)>,
    opts: &Options,
) -> Result<Context> {
    // libtl2cgen: the command line's, hashed now, or the manifest's, which must
    // still have its recorded hash.
    let tl2cgen_runtime = match (&opts.tl2cgen_runtime, tl2cgen_runtime) {
        (Some(p), _) => Some((p.clone(), data::sha256_file(p)?)),
        (None, Some((p, want))) => {
            let got = data::sha256_file(&p)?;
            if got != want {
                bail!(
                    "{} has sha256 {got}, the manifest records {want}; rerun treewalker-exp manifest",
                    p.display()
                );
            }
            Some((p, got))
        }
        (None, None) => None,
    };
    let mut natives = natives.clone();
    if let Some(p) = &opts.lgb_lib {
        natives.lightgbm = Some(native(p, "command line")?);
    }
    if let Some(p) = &opts.xgb_lib {
        natives.xgboost = Some(native(p, "command line")?);
    }
    for lib in [&natives.lightgbm, &natives.xgboost].into_iter().flatten() {
        let got = data::sha256_file(&lib.path)?;
        if got != lib.sha256 {
            bail!(
                "{} has sha256 {got}, the manifest records {}; rerun build-native and manifest",
                lib.path.display(),
                lib.sha256
            );
        }
    }
    if opts.require_pmu {
        if !config.hardware_counters {
            bail!("--require-pmu with hardware counters disabled in the run config");
        }
        // The whole group must be scheduled, never multiplexed.
        pmu::check().map_err(|e| anyhow::anyhow!("hardware counters: {e}"))?;
    }
    let (pmu, pmu_status) = if config.hardware_counters {
        match Pmu::open() {
            Ok(p) => (Some(p), "ok".to_string()),
            Err(e) if opts.require_pmu => bail!("hardware counters: {e}"),
            Err(e) => (None, e),
        }
    } else {
        (None, "disabled in the run config".to_string())
    };
    Ok(Context {
        suite: suite.to_string(),
        config,
        natives,
        tl2cgen_runtime,
        only: opts.only.clone(),
        tl2cgen_override: opts.tl2cgen_override.clone(),
        pmu,
        pmu_status,
        require_pmu: opts.require_pmu,
        process: 0,
        timing_began: 0,
        interface_override: std::collections::BTreeMap::new(),
    })
}

/// SHA-256 of this executable.
pub fn binary_sha256() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| data::sha256_file(&p).ok())
        .unwrap_or_default()
}

/// The resolved source commit: an `export-subst` file in archives, else git.
pub fn source_commit(repo_hint: &Path) -> Value {
    for dir in repo_hint.ancestors() {
        let file = dir.join("experiments/SOURCE_COMMIT");
        if let Ok(s) = std::fs::read_to_string(&file) {
            let s = s.trim();
            if !s.starts_with("$Format") && !s.is_empty() {
                return json!({"commit": s, "from": "experiments/SOURCE_COMMIT (git archive)"});
            }
            let git = |args: &[&str]| {
                std::process::Command::new("git")
                    .args(args)
                    .current_dir(dir)
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            };
            return json!({
                "commit": git(&["rev-parse", "HEAD"]),
                "dirty": git(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty()),
                "from": "git",
            });
        }
    }
    Value::Null
}

/// The measurement domain of a run.
///
/// Everything that decides what a tick means and how it was produced. A run directory holds one domain; resuming under another is
/// refused, so retained samples never acquire another calibration or host.
pub fn domain(header: &Value) -> Value {
    let cal = &header["timer"]["calibration"];
    let hz = cal["hz"].as_f64().unwrap_or(0.0);
    // A measured frequency varies in its last digits from run to run.
    let hz = if cal["source"] == "measured" {
        let scale = 10f64.powi(6 - hz.abs().log10().ceil() as i32);
        (hz * scale).round() / scale
    } else {
        hz
    };
    let host = &header["host"];
    json!({
        "output_schema": header["output_schema"],
        "source": header["source"],
        "binary_sha256": header["binary_sha256"],
        "host": {
            "os": host["os"], "arch": host["arch"], "cpu": host["cpu"],
            "logical_cpus": host["logical_cpus"], "pinned_cpus": host["pinned_cpus"],
            "governor": host["governor"], "smt_active": host["smt_active"],
            "kernel": host["kernel"], "rustc": host["rustc"], "rustflags": host["rustflags"],
            "features": host["features"],
        },
        "system_info": header["system_info"],
        "timer": {"counter": cal["counter"], "source": cal["source"], "hz": hz},
        "events": header["events"],
        "hardware_counters": header["hardware_counters"],
        "natives": header["natives"],
        "tl2cgen_runtime": header["tl2cgen_runtime"],
        "run": header["run"],
        "suite": header["suite"],
    })
}

/// The SHA-256 of each compiled baseline file the cell names, as found on disk.
fn baseline_files(cell_dir: &Path, cell_doc: &Value) -> Value {
    let Some(model) = cell_doc["files"]["model_treelite_bin"]["path"].as_str() else {
        return Value::Null;
    };
    let fw_dir = cell_dir
        .join(model)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let mut out = serde_json::Map::new();
    if let Some(b) = cell_doc["baselines"].as_object() {
        for (name, rec) in b {
            if let Some(lib) = rec["library"].as_str() {
                let path = fw_dir.join(lib);
                out.insert(name.clone(), json!(data::sha256_file(&path).ok()));
            }
        }
    }
    Value::Object(out)
}

/// A manifest entry and its `cell.json` must agree exactly: ready, the same ID and
/// the same key.
pub fn check_entry(mc: &ManifestCell, cell_doc: &Value) -> Result<()> {
    let (id, key, status) = (
        cell_doc["id"].as_str().unwrap_or_default(),
        cell_doc["key"].as_str().unwrap_or_default(),
        cell_doc["status"].as_str().unwrap_or_default(),
    );
    anyhow::ensure!(
        status == "ready",
        "{}: cell.json status is {status:?}",
        mc.id
    );
    anyhow::ensure!(id == mc.id, "{}: cell.json has ID {id:?}", mc.id);
    anyhow::ensure!(
        !key.is_empty() && key == mc.key,
        "{}: cell.json has key {key:?}, the manifest {:?}",
        mc.id,
        mc.key
    );
    Ok(())
}

/// The resume key: a finished cell is reused only when this matches. It covers
/// the run's whole measurement domain and everything the cell's measurement used.
pub fn resume_key(
    run_header: &Value,
    mc: &ManifestCell,
    cell_dir: &Path,
    cell_doc: &Value,
    ctx: &Context,
) -> String {
    let doc = json!({
        "domain": domain(run_header),
        "entry": {"id": mc.id, "key": mc.key, "status": mc.status},
        "cell": {"id": cell_doc["id"], "status": cell_doc["status"], "key": cell_doc["key"],
                 "files": cell_doc["files"],
                 "baselines": cell_doc["baselines"], "oracle": cell_doc["oracle"],
                 "baseline_files": baseline_files(cell_dir, cell_doc)},
        "methods": mc.methods,
        "variants": mc.variants,
        "modes": mc.modes,
        "only": ctx.only,
        "tl2cgen_override": ctx.tl2cgen_override.as_ref().and_then(|p| data::sha256_file(p).ok()),
    });
    data::sha256_bytes(doc.to_string().as_bytes())
}

/// Start or resume the run in `run_dir`. A new run writes `run.json` and
/// `timer.parquet`; a resumed run keeps both and must match their domain.
pub fn open_run(run_dir: &Path, header: Value, overhead: &[u64]) -> Result<Value> {
    std::fs::create_dir_all(run_dir.join("cells"))?;
    let path = run_dir.join("run.json");
    if path.exists() {
        let old: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        let (a, b) = (domain(&old), domain(&header));
        if a != b {
            let differ: Vec<&String> = a
                .as_object()
                .into_iter()
                .flatten()
                .filter(|(k, v)| b.get(k.as_str()) != Some(*v))
                .map(|(k, _)| k)
                .collect();
            bail!(
                "{} holds a run measured under another configuration ({differ:?} differ); \
                 use a new run ID",
                run_dir.display()
            );
        }
        return Ok(old);
    }
    write_overhead(run_dir, overhead)?;
    output::write_json(&path, &header)?;
    Ok(header)
}

/// Calibrate the timer and describe the host: `run.json`'s header.
pub fn header(
    ctx: &Context,
    manifest: Option<&Manifest>,
    opts: &Options,
    repo: &Path,
) -> Result<(Value, Vec<u64>)> {
    let (cal, overhead) = timer::calibrate(200, 100_000);
    let amortized = timer::amortized_check(20_000, cal.overhead.p50);
    let events = match pmu::events_for_host() {
        Ok((table, defs)) => json!({"table": table, "events": defs}),
        Err(e) => json!({"table": null, "reason": e}),
    };
    let system_info = match &opts.system_info {
        Some(p) => serde_json::from_str(
            &std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?,
        )?,
        None => Value::Null,
    };
    Ok((
        json!({
            "output_schema": OUTPUT_SCHEMA,
            "suite": ctx.suite,
            "manifest": manifest.map(|m| json!({
                "suite": m.suite, "grids_sha256": m.grids_sha256, "source": m.source,
                "artifacts_dir": m.artifacts_dir, "cells": m.cells.len(),
            })),
            "source": source_commit(repo),
            "binary_sha256": binary_sha256(),
            "host": system::host(),
            "system_info": system_info,
            "timer": {
                "calibration": cal,
                // Per-call samples against one long interval over the same calls:
                // what timing each call on its own adds, as measured.
                "amortized_check": amortized,
            },
            "events": events,
            "hardware_counters": ctx.pmu_status,
            "natives": ctx.natives,
            "tl2cgen_runtime": ctx.tl2cgen_runtime.as_ref().map(|(p, sha)| json!({"path": p, "sha256": sha})),
            "run": ctx.config,
            "started": unix_time(),
        }),
        overhead,
    ))
}

fn unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn write_overhead(run_dir: &Path, samples: &[u64]) -> Result<()> {
    let mut t = Table::new(&[("ticks", Kind::U64)]);
    for &s in samples {
        t.push(&[V::U64(s)]);
    }
    t.write(&run_dir.join("timer.parquet"))
}

/// Run every ready cell of a manifest into `run_dir`, resuming finished cells.
///
/// Returns false when any selected cell failed or could not be timed, so the
/// caller exits nonzero and an unattended run never reports success.
pub fn run(
    manifest_path: &Path,
    run_dir: &Path,
    artifacts: Option<&Path>,
    cells: Option<&str>,
    opts: &Options,
) -> Result<bool> {
    let manifest = Manifest::load(manifest_path)?;
    let artifacts = artifacts.map_or_else(|| manifest.artifacts_dir.clone(), Path::to_path_buf);
    let mut ctx = context(
        &manifest.suite,
        manifest.run.clone(),
        &manifest.native,
        manifest
            .tl2cgen_runtime
            .as_ref()
            .map(|r| (r.path.clone(), r.sha256.clone())),
        opts,
    )?;
    let (header, overhead) = header(&ctx, Some(&manifest), opts, manifest_path)?;
    let started = header["started"].clone();
    let mut header = open_run(run_dir, header, &overhead)?;
    // Every cell must have its own output directory.
    let mut names = std::collections::BTreeMap::new();
    for c in &manifest.cells {
        let name = Cell::slug(&c.id)?;
        if let Some(first) = names.insert(name.clone(), c.id.clone()) {
            bail!(
                "cells {first:?} and {:?} share the output name {name}",
                c.id
            );
        }
    }
    // The sentinel: at the start of every invocation, then every
    // sentinel_every measured cells.
    let mut sentinel = crate::sentinel::Sentinel::new(
        manifest.sentinel.as_ref(),
        &artifacts,
        ctx.config.sentinel_every,
    );
    sentinel.start(&mut ctx, run_dir, &mut header)?;
    let selected: Vec<&ManifestCell> = manifest
        .cells
        .iter()
        .filter(|c| cells.is_none_or(|pat| glob(pat, &c.id)))
        .collect();
    let (mut done, mut skipped, mut failed) = (Vec::new(), Vec::new(), Vec::new());
    for (i, mc) in selected.iter().enumerate() {
        eprintln!("[{}/{}] {}", i + 1, selected.len(), mc.id);
        if mc.status != "ready" {
            // Unsupported cells (import limits) are recorded; any other status is a
            // preparation failure.
            let entry = json!({"id": mc.id, "reason": format!("status {}", mc.status)});
            eprintln!("    not timed: status {}", mc.status);
            if mc.status == "unsupported" {
                skipped.push(entry);
            } else {
                failed.push(entry);
            }
            continue;
        }
        let dir = artifacts.join(&mc.dir);
        let began = ctx.timing_began;
        let outcome = (|| -> Result<driver::CellOutcome> {
            let slug = Cell::slug(&mc.id)?;
            let cell_doc: Value = serde_json::from_str(
                &std::fs::read_to_string(dir.join("cell.json"))
                    .with_context(|| format!("reading {}/cell.json", dir.display()))?,
            )?;
            // Readiness and identity before any reuse: a finished result is only
            // ever the result of this exact, ready cell.
            check_entry(mc, &cell_doc)?;
            let key = resume_key(&header, mc, &dir, &cell_doc, &ctx);
            if output::finished_key(run_dir, &slug).as_deref() == Some(&key) {
                eprintln!("    finished in an earlier invocation; reused");
                return Ok(driver::CellOutcome {
                    id: mc.id.clone(),
                    dir: run_dir.join("cells").join(&slug),
                    timed_methods: 0,
                    excluded: 0,
                    seconds: 0.0,
                    totals: std::collections::BTreeMap::new(),
                    interfaces: std::collections::BTreeMap::new(),
                });
            }
            driver::run_cell(&mut ctx, mc, &dir, run_dir, &key, true)
        })();
        let measured = ctx.timing_began > began;
        match outcome {
            Ok(o) => {
                if o.seconds > 0.0 {
                    eprintln!(
                        "    {} methods timed, {} excluded, {:.1}s",
                        o.timed_methods, o.excluded, o.seconds
                    );
                }
                done.push(json!({"id": o.id, "seconds": o.seconds}));
            }
            Err(e) => {
                eprintln!("    FAILED: {e:#}");
                failed.push(json!({"id": mc.id, "reason": format!("{e:#}")}));
            }
        }
        if measured && sentinel.after_timed_cell() {
            sentinel.run(
                &mut ctx,
                run_dir,
                &mut header,
                done.len() + failed.len(),
                false,
            )?;
        }
    }
    let invocation = json!({
        "started": started,
        "finished": unix_time(),
        "cells_done": done,
        "cells_skipped": skipped,
        "cells_failed": failed,
        "peak_rss_bytes": system::peak_rss_bytes(),
    });
    match header.get_mut("invocations").and_then(Value::as_array_mut) {
        Some(list) => list.push(invocation),
        None => header["invocations"] = json!([invocation]),
    }
    output::write_json(&run_dir.join("run.json"), &header)?;
    if !failed.is_empty() {
        eprintln!("{} cells failed; see run.json and failed/", failed.len());
    }
    Ok(failed.is_empty())
}

/// Shell-style `*` matching, enough for cell IDs.
pub fn glob(pattern: &str, s: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == s;
    }
    let mut rest = s;
    for (i, p) in parts.iter().enumerate() {
        if i == 0 {
            let Some(r) = rest.strip_prefix(p) else {
                return false;
            };
            rest = r;
        } else if i == parts.len() - 1 {
            return rest.ends_with(p);
        } else if let Some(at) = rest.find(p) {
            rest = &rest[at + p.len()..];
        } else {
            return false;
        }
    }
    true
}

/// Check that a required native library exists, has its recorded hash, loads and
/// exports the calls the adapter makes.
fn probe_native(lib: Option<&crate::artifacts::NativeLib>, kind: &str) -> Result<(), String> {
    let lib = lib.ok_or("no library; run build-native")?;
    let got = data::sha256_file(&lib.path).map_err(|e| format!("{e:#}"))?;
    if got != lib.sha256 {
        return Err(format!(
            "{} has sha256 {got}, recorded {}",
            lib.path.display(),
            lib.sha256
        ));
    }
    #[cfg(feature = "external-bench")]
    {
        crate::external::probe(&lib.path, kind)
    }
    #[cfg(not(feature = "external-bench"))]
    {
        let _ = kind;
        Err("sweep_bench was built without external-bench".into())
    }
}

/// Report required and optional methods and unsupported variants; fail when a
/// required native baseline or a valid PMU configuration is missing.
pub fn preflight(manifest_path: &Path, artifacts: Option<&Path>, opts: &Options) -> Result<bool> {
    let manifest = Manifest::load(manifest_path)?;
    let artifacts = artifacts.map_or_else(|| manifest.artifacts_dir.clone(), Path::to_path_buf);
    let mut ok = true;
    let pmu = if manifest.run.hardware_counters {
        match pmu::check() {
            Ok((table, r)) => {
                println!(
                    "hardware counters: {table}, the whole group scheduled (time_running {} = time_enabled)",
                    r.time_running
                );
                Some(true)
            }
            Err(e) => {
                println!("hardware counters: {e}");
                Some(false)
            }
        }
    } else {
        None
    };
    if pmu == Some(false) && (opts.require_pmu || cfg!(target_os = "linux")) {
        ok = false;
    }
    let mut natives = manifest.native.clone();
    if let Some(p) = &opts.lgb_lib {
        natives.lightgbm = Some(native(p, "command line")?);
    }
    if let Some(p) = &opts.xgb_lib {
        natives.xgboost = Some(native(p, "command line")?);
    }
    // libtl2cgen must exist with its recorded hash.
    let tl2cgen_rt = match (&opts.tl2cgen_runtime, &manifest.tl2cgen_runtime) {
        (Some(p), _) => Some(p.clone()),
        (None, Some(rt)) => match data::sha256_file(&rt.path) {
            Ok(got) if got == rt.sha256 => Some(rt.path.clone()),
            _ => {
                println!(
                    "libtl2cgen {} does not have its recorded hash",
                    rt.path.display()
                );
                ok = false;
                None
            }
        },
        (None, None) => None,
    };
    let mut checked = 0;
    for mc in &manifest.cells {
        let set = MethodSet::parse(&mc.methods)?;
        let dir = artifacts.join(&mc.dir);
        if mc.status != "ready" {
            println!("{}: not timed, status {}", mc.id, mc.status);
            continue;
        }
        let cell = match Cell::load(&dir) {
            Ok(c) => c,
            Err(e) => {
                ok = false;
                println!("{}: cannot load: {e:#}", mc.id);
                continue;
            }
        };
        let required: Vec<String> = set
            .required(&cell.framework)
            .iter()
            .map(ToString::to_string)
            .collect();
        let mut missing = Vec::new();
        let mut optional = Vec::new();
        let mut unsupported = Vec::new();
        for m in &required {
            let lib = match m.as_str() {
                "lightgbm_native" => Some((&natives.lightgbm, "lightgbm")),
                "xgboost_native" => Some((&natives.xgboost, "xgboost")),
                _ => None,
            };
            if let Some((lib, kind)) = lib
                && let Err(e) = probe_native(lib.as_ref(), kind)
            {
                missing.push(format!("{m} ({e})"));
            }
        }
        if set.baselines() {
            let mut opt = |name: &str, why: Option<String>| match why {
                None => optional.push(name.to_string()),
                Some(w) => unsupported.push(format!("{name}: {w}")),
            };
            let ext = (!cfg!(feature = "external-bench"))
                .then(|| "built without external-bench".to_string());
            opt(
                "tl2cgen",
                ext.clone().or_else(|| match (&cell.tl2cgen, &tl2cgen_rt) {
                    (None, _) => Some("not compiled".into()),
                    (_, None) => Some("no libtl2cgen runtime".into()),
                    _ => None,
                }),
            );
            if cell.framework == "lightgbm" {
                opt(
                    "lleaves",
                    ext.clone()
                        .or_else(|| cell.lleaves.is_none().then(|| "not compiled".into())),
                );
            }
            opt(
                "quickscorer",
                (!cfg!(feature = "quickscorer-bench"))
                    .then(|| "built without quickscorer-bench".to_string())
                    .or_else(|| cell.quickscorer.as_ref().err().cloned()),
            );
            if cell.max_group_rows() > crate::methods::CHUNK_ROWS {
                optional.push("treewalker_chunked128".into());
            }
        }
        // Variants that cannot load are recorded, never silently dropped.
        let mut seen = std::collections::BTreeSet::new();
        for v in &mc.variants {
            let mut load = v.load.clone();
            load.sort();
            if load.is_empty() || !seen.insert(load.clone()) {
                continue;
            }
            if let Err(e) = v.load_options().and_then(|o| cell.forest(&o)) {
                unsupported.push(format!("load {}: {e:#}", load.join("+")));
            }
        }
        if !missing.is_empty() {
            ok = false;
        }
        println!(
            "{}: required [{}]{}; optional [{}]{}",
            mc.id,
            required.join(", "),
            if missing.is_empty() {
                String::new()
            } else {
                format!(" MISSING [{}]", missing.join(", "))
            },
            optional.join(", "),
            if unsupported.is_empty() {
                String::new()
            } else {
                format!("; unsupported: {}", unsupported.join("; "))
            }
        );
        checked += 1;
    }
    println!(
        "preflight: {checked} cells, {}",
        if ok { "OK" } else { "FAILED" }
    );
    Ok(ok)
}
