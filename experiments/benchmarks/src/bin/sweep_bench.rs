//! `sweep_bench`: times prepared cells into a run directory.
//!
//! ```text
//! sweep_bench run <EXECUTION_MANIFEST> --output-dir experiments/data/runs/<run_id>
//! sweep_bench cell <CELL_DIR> --output-dir DIR [--set factorial] [--variant FLAGS]...
//! sweep_bench preflight <EXECUTION_MANIFEST>
//! sweep_bench validate <CELL_DIR>
//! ```
//!
//! `treewalker-exp manifest --suite NAME` writes the execution manifest, and
//! `treewalker-exp run` builds this binary and runs it pinned to one core.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context as _, Result};
use clap::{Args, Parser, Subcommand};
use serde_json::Value;
use treewalker_bench::artifacts::{Cell, ManifestCell, Natives, RunConfig, VariantSpec};
use treewalker_bench::driver;
use treewalker_bench::run::{self, Options};

#[derive(Parser)]
#[command(
    version,
    about = "Time TreeWalker and its baselines on prepared cells."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Execute an execution manifest into a run directory, resuming finished cells.
    Run {
        manifest: PathBuf,
        /// The run directory, e.g. experiments/data/runs/<run_id>.
        #[arg(long)]
        output_dir: PathBuf,
        /// Override the manifest's artifacts directory.
        #[arg(long)]
        artifacts_dir: Option<PathBuf>,
        /// Only cells whose ID matches this glob.
        #[arg(long)]
        cells: Option<String>,
        /// Only these methods, comma-separated (the layout check times TreeWalker).
        #[arg(long, value_delimiter = ',')]
        only: Option<Vec<String>>,
        #[command(flatten)]
        libs: Libs,
    },
    /// Time one cell into a run directory.
    Cell {
        cell_dir: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        /// Method set: factorial, ablation or treewalker.
        #[arg(long, default_value = "factorial")]
        set: String,
        /// A research variant: runtime flags joined by `,`, then optionally `|` and
        /// load flags; `all-on` for none. Repeatable.
        #[arg(long = "variant")]
        variants: Vec<String>,
        /// The modes to time, comma-separated: serving, batch.
        #[arg(long, value_delimiter = ',', default_value = "serving,batch")]
        modes: Vec<String>,
        /// Only these methods, comma-separated.
        #[arg(long, value_delimiter = ',')]
        only: Option<Vec<String>>,
        /// Replace the cell's tl2cgen library (the parallel_comp pilot).
        #[arg(long)]
        tl2cgen_lib: Option<PathBuf>,
        #[command(flatten)]
        config: ConfigArgs,
        #[command(flatten)]
        libs: Libs,
    },
    /// Report required and optional methods and unsupported variants; fail when a
    /// required native baseline or a valid PMU configuration is missing.
    Preflight {
        manifest: PathBuf,
        #[arg(long)]
        artifacts_dir: Option<PathBuf>,
        #[command(flatten)]
        libs: Libs,
    },
    /// One of XGBoost's extra processes, which `run` starts for each XGBoost cell:
    /// a JSON request in, a JSON result out.
    #[command(hide = true)]
    XgboostProcess { request: PathBuf, result: PathBuf },
    /// Validate one cell's TreeWalker outputs (and baselines, given their libraries)
    /// without timing.
    Validate {
        cell_dir: PathBuf,
        #[arg(long, default_value = "treewalker")]
        set: String,
        #[command(flatten)]
        libs: Libs,
    },
}

#[derive(Args, Clone, Default)]
struct Libs {
    /// Native LightGBM C library; default: the manifest's.
    #[arg(long)]
    lgb_lib: Option<PathBuf>,
    /// Native XGBoost C library; default: the manifest's.
    #[arg(long)]
    xgb_lib: Option<PathBuf>,
    /// tl2cgen's runtime, libtl2cgen; default: the manifest's.
    #[arg(long)]
    tl2cgen_runtime: Option<PathBuf>,
    /// Fail when hardware counters cannot be read.
    #[arg(long)]
    require_pmu: bool,
    /// JSON file of host facts the runner cannot see, recorded in run.json.
    #[arg(long)]
    system_info: Option<PathBuf>,
}

impl Libs {
    fn options(&self) -> Options {
        Options {
            lgb_lib: self.lgb_lib.clone(),
            xgb_lib: self.xgb_lib.clone(),
            tl2cgen_runtime: self.tl2cgen_runtime.clone(),
            require_pmu: self.require_pmu,
            system_info: self.system_info.clone(),
            ..Options::default()
        }
    }
}

#[derive(Args, Clone)]
struct ConfigArgs {
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    target_batches: Option<usize>,
    #[arg(long)]
    batch_rows: Option<usize>,
    #[arg(long)]
    min_rounds: Option<usize>,
    #[arg(long)]
    max_rounds: Option<usize>,
    #[arg(long)]
    precision_pct: Option<f64>,
    #[arg(long)]
    mode_budget_secs: Option<f64>,
    /// Rows a round times, at most; a larger pool times a stratified group sample,
    /// and validation covers the same groups.
    #[arg(long)]
    max_rows_per_round: Option<usize>,
    /// Do not read hardware counters.
    #[arg(long)]
    no_hardware_counters: bool,
}

impl ConfigArgs {
    const fn apply(&self, mut c: RunConfig) -> RunConfig {
        macro_rules! set {
            ($($f:ident),*) => { $(if let Some(v) = self.$f { c.$f = v; })* };
        }
        set!(
            seed,
            target_batches,
            batch_rows,
            min_rounds,
            max_rounds,
            precision_pct,
            mode_budget_secs,
            max_rows_per_round
        );
        c.hardware_counters &= !self.no_hardware_counters;
        c
    }
}

fn parse_variant(s: &str) -> VariantSpec {
    let (rt, ld) = s.split_once('|').unwrap_or((s, ""));
    let split = |x: &str| -> Vec<String> {
        x.split(',')
            .map(str::trim)
            .filter(|f| !f.is_empty() && *f != "all-on")
            .map(ToString::to_string)
            .collect()
    };
    VariantSpec {
        runtime: split(rt),
        load: split(ld),
    }
}

fn cell_entry(
    dir: &std::path::Path,
    set: &str,
    variants: &[String],
    modes: &[String],
) -> Result<ManifestCell> {
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("cell.json"))
            .with_context(|| format!("reading {}/cell.json", dir.display()))?,
    )?;
    Ok(ManifestCell {
        id: doc["id"].as_str().unwrap_or_default().to_string(),
        dir: dir.to_path_buf(),
        status: doc["status"].as_str().unwrap_or_default().to_string(),
        key: doc["key"].as_str().unwrap_or_default().to_string(),
        methods: set.to_string(),
        variants: variants.iter().map(|v| parse_variant(v)).collect(),
        modes: modes.to_vec(),
    })
}

fn main() -> ExitCode {
    match real_main() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn real_main() -> Result<bool> {
    match Cli::parse().command {
        Command::Run {
            manifest,
            output_dir,
            artifacts_dir,
            cells,
            only,
            libs,
        } => {
            let mut opts = libs.options();
            opts.only = only;
            run::run(
                &manifest,
                &output_dir,
                artifacts_dir.as_deref(),
                cells.as_deref(),
                &opts,
            )
        }
        Command::Preflight {
            manifest,
            artifacts_dir,
            libs,
        } => run::preflight(&manifest, artifacts_dir.as_deref(), &libs.options()),
        Command::Cell {
            cell_dir,
            output_dir,
            set,
            variants,
            modes,
            only,
            tl2cgen_lib,
            config,
            libs,
        } => {
            let mc = cell_entry(&cell_dir, &set, &variants, &modes)?;
            let mut opts = libs.options();
            opts.only = only;
            opts.tl2cgen_override = tl2cgen_lib;
            let mut ctx = run::context(
                "cell",
                config.apply(RunConfig::default()),
                &Natives::default(),
                None,
                &opts,
            )?;
            // The same output contract as run: run.json and timer.parquet.
            let (header, overhead) = run::header(&ctx, None, &opts, &cell_dir)?;
            let header = run::open_run(&output_dir, header, &overhead)?;
            let doc: Value =
                serde_json::from_str(&std::fs::read_to_string(cell_dir.join("cell.json"))?)?;
            let key = run::resume_key(&header, &mc, &cell_dir, &doc, &ctx);
            let o = driver::run_cell(&mut ctx, &mc, &cell_dir, &output_dir, &key, true)?;
            eprintln!(
                "{}: {} methods timed -> {}",
                o.id,
                o.timed_methods,
                o.dir.display()
            );
            Ok(true)
        }
        Command::XgboostProcess { request, result } => {
            let req: Value = serde_json::from_str(&std::fs::read_to_string(&request)?)?;
            let out = driver::xgboost_process(&req)?;
            std::fs::write(&result, out.to_string())?;
            Ok(true)
        }
        Command::Validate {
            cell_dir,
            set,
            libs,
        } => {
            let both = ["serving".to_string(), "batch".to_string()];
            let mc = cell_entry(&cell_dir, &set, &[], &both)?;
            let opts = libs.options();
            let config = RunConfig {
                hardware_counters: false,
                // No cap: the sample, and with it validation, covers every group.
                max_rows_per_round: 0,
                ..RunConfig::default()
            };
            let mut ctx = run::context("validate", config, &Natives::default(), None, &opts)?;
            let tmp =
                std::env::temp_dir().join(format!("sweep_bench-validate-{}", std::process::id()));
            let o = driver::run_cell(&mut ctx, &mc, &cell_dir, &tmp, "", false)?;
            let m: Value =
                serde_json::from_str(&std::fs::read_to_string(o.dir.join("manifest.json"))?)?;
            std::fs::remove_dir_all(&tmp)?;
            let cell = Cell::load(&cell_dir)?;
            let v = &m["validation"];
            let oracle_ok = v["oracle"]["status"] != "fail";
            let checks = v["checks"].as_array().cloned().unwrap_or_default();
            for c in &checks {
                println!(
                    "{} {}/{} [{}] vs {} ({}): max_abs {:.3e}, {} mismatches of {}",
                    if c["passed"] == true { "PASS" } else { "FAIL" },
                    c["method"].as_str().unwrap_or_default(),
                    c["variant"].as_str().unwrap_or_default(),
                    c["interface"].as_str().unwrap_or_default(),
                    c["against"].as_str().unwrap_or_default(),
                    c["mode"].as_str().unwrap_or_default(),
                    c["max_abs"].as_f64().unwrap_or(f64::NAN),
                    c["mismatches"],
                    c["rows"],
                );
            }
            println!("oracle: {}", v["oracle"]);
            for e in m["excluded"].as_array().into_iter().flatten() {
                println!("excluded: {e}");
            }
            let ok = oracle_ok && checks.iter().all(|c| c["passed"] == true);
            println!(
                "VALIDATE {}: {} ({} groups)",
                if ok { "PASS" } else { "FAIL" },
                cell.id,
                cell.n_groups()
            );
            Ok(ok)
        }
    }
}
