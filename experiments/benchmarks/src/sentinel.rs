//! The sentinel: one fixed cell, timed again and again through a run.
//!
//! It runs at the start of each invocation of a run and again after every
//! `sentinel_every` timed cells, so that drift in the machine over a long run shows
//! in the run's own data.
//!
//! Each invocation starts with a warm-up sentinel, then a measured one: a
//! process's first load of a library can read slow (LightGBM native 12-15% on the
//! M4), and the pair records that first-load effect. Its interfaces are fixed from
//! the run's first sentinel. Only cells whose timing began advance the cadence.
//! Each sentinel is a complete cell measurement under `sentinel/<k>/`, outside
//! `cells/`, and every attempt, failed ones included, goes to `run.json`'s
//! `sentinel.records` with each method's raw ticks and rows. The analysis derives
//! the drift against the run's first measured sentinel, flagged beyond
//! `sentinel_drift_pct`, and the first-load effect; the sentinel stops nothing. A
//! suite whose artifacts lack the cell records why it has no sentinel.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::{Value, json};

use crate::artifacts::ManifestCell;
use crate::driver::{self, Context};
use crate::output;

/// One sentinel attempt.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Record {
    pub index: usize,
    /// Cells of the run finished, reused ones included, when it ran.
    pub cells_done: usize,
    pub unix_time: u64,
    /// `ok`, or why it could not be measured.
    pub status: String,
    /// `warm-up`, the first sentinel of each invocation, or `measure`.
    pub role: String,
    /// Ticks and rows over every round, by `method/variant/mode`; empty when the
    /// attempt failed.
    pub totals: BTreeMap<String, [u64; 2]>,
}

/// The run's sentinel: its cell, or why there is none.
pub struct Sentinel {
    entry: Result<(ManifestCell, PathBuf), String>,
    every: usize,
    timed_since: usize,
}

impl Sentinel {
    /// From the manifest's `sentinel` entry, whose cell directory is under
    /// `artifacts`.
    pub fn new(entry: Option<&Value>, artifacts: &Path, every: usize) -> Self {
        let entry = match entry {
            None | Some(Value::Null) => Err("no sentinel cell in the manifest".to_string()),
            Some(e) if e["status"] != "ready" => Err(format!(
                "{}: {}",
                e["id"].as_str().unwrap_or("sentinel"),
                e["reason"]
                    .as_str()
                    .map_or_else(|| format!("status {}", e["status"]), ToString::to_string)
            )),
            Some(e) => serde_json::from_value::<ManifestCell>(e.clone())
                .map_err(|err| format!("sentinel entry: {err}"))
                .and_then(|mc| {
                    let dir = artifacts.join(&mc.dir);
                    if dir.join("cell.json").is_file() {
                        Ok((mc, dir))
                    } else {
                        Err(format!(
                            "{}: no cell.json under {}",
                            mc.id,
                            artifacts.display()
                        ))
                    }
                }),
        };
        Self {
            entry,
            every,
            timed_since: 0,
        }
    }

    /// Call after each cell that was measured (not reused); true when the sentinel
    /// is due.
    pub const fn after_timed_cell(&mut self) -> bool {
        self.timed_since += 1;
        self.every > 0 && self.timed_since >= self.every
    }

    /// The start of an invocation: a warm-up sentinel, then a measured one.
    pub fn start(&mut self, ctx: &mut Context, run_dir: &Path, header: &mut Value) -> Result<()> {
        self.run(ctx, run_dir, header, 0, true)?;
        if self.entry.is_ok() && self.every > 0 {
            self.run(ctx, run_dir, header, 0, false)?;
        }
        Ok(())
    }

    /// Measure the sentinel into `run_dir/sentinel/<k>/`, record it in `header`,
    /// and rewrite `run.json`. Failures are recorded, never returned: the
    /// sentinel stops nothing. Its interfaces are fixed from the run's first
    /// sentinel.
    pub fn run(
        &mut self,
        ctx: &mut Context,
        run_dir: &Path,
        header: &mut Value,
        cells_done: usize,
        warm_up: bool,
    ) -> Result<()> {
        self.timed_since = 0;
        if self.every == 0 {
            header["sentinel"]["skipped"] = json!("sentinel_every is 0");
            return Ok(());
        }
        let (mc, dir) = match &self.entry {
            Ok(e) => e.clone(),
            Err(reason) => {
                header["sentinel"]["skipped"] = json!(reason);
                return output::write_json(&run_dir.join("run.json"), header);
            }
        };
        let mut records: Vec<Record> =
            serde_json::from_value(header["sentinel"]["records"].clone()).unwrap_or_default();
        let interfaces: Option<BTreeMap<String, BTreeMap<String, String>>> =
            serde_json::from_value(header["sentinel"]["interfaces"].clone()).ok();
        let index = records.len();
        let role = if warm_up { "warm-up" } else { "measure" };
        eprintln!("[sentinel {index}, {role}] {}", mc.id);
        let out_dir = run_dir.join("sentinel").join(format!("{index:03}"));
        let saved = std::mem::replace(
            &mut ctx.interface_override,
            interfaces.clone().unwrap_or_default(),
        );
        let began = ctx.timing_began;
        let measured = (|| -> Result<driver::CellOutcome> {
            let doc: Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join("cell.json"))?)?;
            crate::run::check_entry(&mc, &doc)?;
            let key = crate::run::resume_key(header, &mc, &dir, &doc, ctx);
            driver::run_cell(ctx, &mc, &dir, &out_dir, &key, true)
        })();
        // The sentinel is not one of the run's cells.
        ctx.timing_began = began;
        ctx.interface_override = saved;
        let (status, totals, chosen) = match measured {
            Ok(o) => {
                let totals = o
                    .totals
                    .iter()
                    .flat_map(|(mode, t)| t.iter().map(move |(k, &v)| (format!("{k}/{mode}"), v)))
                    .collect();
                ("ok".to_string(), totals, Some(o.interfaces))
            }
            Err(e) => (format!("failed: {e:#}"), BTreeMap::new(), None),
        };
        records.push(Record {
            index,
            cells_done,
            unix_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            status,
            role: role.into(),
            totals,
        });
        header["sentinel"] = json!({
            "cell": mc.id,
            "every": self.every,
            "interfaces": interfaces.or(chosen),
            "records": records,
        });
        output::write_json(&run_dir.join("run.json"), header)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_sentinel_records_why() {
        let tmp = std::env::temp_dir();
        let s = Sentinel::new(None, &tmp, 25);
        assert!(s.entry.is_err());
        let skipped = json!({"id": "support/x", "status": "skipped", "reason": "not prepared"});
        let s = Sentinel::new(Some(&skipped), &tmp, 25);
        assert_eq!(s.entry.err().unwrap(), "support/x: not prepared");
        let ready = json!({"id": "support/x", "status": "ready", "dir": "no/such/dir",
                           "key": "k", "methods": "factorial", "modes": ["serving"]});
        let s = Sentinel::new(Some(&ready), &tmp, 25);
        assert!(s.entry.err().unwrap().contains("no cell.json"));
    }

    #[test]
    fn it_is_due_every_n_timed_cells() {
        let mut s = Sentinel::new(None, Path::new("."), 3);
        let due: Vec<bool> = (0..7).map(|_| s.after_timed_cell()).collect();
        assert_eq!(due, [false, false, true, true, true, true, true]);
        s.timed_since = 0;
        assert!(!s.after_timed_cell());
        assert!(!Sentinel::new(None, Path::new("."), 0).after_timed_cell());
    }
}
