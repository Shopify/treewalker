//! The f64 tree-ordering measurement, offline over prepared cells.
//!
//! With exact sums off, TreeWalker adds each row's leaf values in `f64` in its
//! loaded tree order, and tree ordering (on by default) permutes the trees, so the
//! sums move in the last bits. [`ordering`] loads a LightGBM cell's model twice,
//! exact sums off, tree ordering on and off, and predicts every row of the cell
//! with the research predictor's stages and with production `predict`.
//!
//! The primary comparison is ordering on against off, on the tree sums and raw
//! margins: both come from the same `f64` additions on any IEEE 754 machine, so it
//! does not depend on the platform. The secondary one compares the outputs, and
//! each ordering's outputs with the cell's prepared GTIL reference (`prepare`'s
//! predictions for LightGBM); a sigmoid-link model's outputs include the
//! platform's `exp`, and GTIL's ran where the cell was prepared.
//!
//! It runs on prepared artifacts, not in a run: the sums are deterministic, it
//! covers every row of the cell rather than the timed groups, and it adds nothing
//! to a run's time or its failure modes. Each cell's record names the files it
//! read with the hashes its `cell.json` records, verified before use.

use std::collections::HashMap;

use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};
use treewalker_gbdt::research::{Ablation, Stages};
use treewalker_gbdt::{Forest, LoadOptions};

use crate::artifacts::Cell;

/// How two per-row vectors differ: the rows not equal bit for bit, the largest
/// absolute difference (a NaN difference counts as infinite) and its row.
pub fn diff(a: &[f64], b: &[f64]) -> Value {
    assert_eq!(a.len(), b.len());
    let (mut differ, mut max_abs, mut at) = (0usize, 0.0f64, Value::Null);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() == y.to_bits() {
            continue;
        }
        differ += 1;
        let d = (x - y).abs();
        let d = if d.is_nan() { f64::INFINITY } else { d };
        if d > max_abs || at.is_null() {
            max_abs = d;
            at = json!(i);
        }
    }
    json!({"rows_differ": differ, "max_abs": max_abs, "at_row": at})
}

/// Each row's three stages and production `predict`'s output, in row order.
struct Predicted {
    tree_sum: Vec<f64>,
    raw_margin: Vec<f64>,
    output: Vec<f64>,
    production: Vec<f64>,
}

fn predict_all(cell: &Cell, forest: &Forest) -> Predicted {
    let mut r = forest.research_predictor(Ablation::default());
    let mut p = forest.predictor();
    let mut stages = Stages::default();
    let n = cell.n_rows;
    let mut out = Predicted {
        tree_sum: Vec::with_capacity(n),
        raw_margin: Vec::with_capacity(n),
        output: Vec::with_capacity(n),
        production: vec![0.0; n],
    };
    for g in 0..cell.n_groups() {
        let rows = cell.group(g);
        r.predict_group_stages(rows, &mut stages);
        out.tree_sum.extend_from_slice(&stages.tree_sum);
        out.raw_margin.extend_from_slice(&stages.raw_margin);
        out.output.extend_from_slice(&stages.output);
        p.predict_group(
            rows,
            &mut out.production[cell.offsets[g]..cell.offsets[g + 1]],
        );
    }
    out
}

/// Tree positions whose tree differs between two loads of one model: each tree's
/// node count and sorted leaf values, compared position by position.
fn trees_moved(a: &Forest, b: &Forest) -> usize {
    let print = |f: &Forest| -> Vec<(u32, Vec<u64>)> {
        f.trees()
            .iter()
            .map(|t| {
                let mut leaves: Vec<u64> = f
                    .tree_nodes(t)
                    .iter()
                    .filter(|n| n.skip == -1)
                    .map(|n| n.value.to_bits())
                    .collect();
                leaves.sort_unstable();
                (t.node_count, leaves)
            })
            .collect()
    };
    let (pa, pb) = (print(a), print(b));
    pa.iter().zip(&pb).filter(|(x, y)| x != y).count() + pa.len().abs_diff(pb.len())
}

/// The ordering measurement on one LightGBM cell.
pub fn ordering(cell: &Cell) -> Result<Value> {
    ensure!(
        cell.framework == "lightgbm",
        "{}: GTIL is the reference for LightGBM cells only",
        cell.id
    );
    let reference = cell
        .reference
        .as_ref()
        .with_context(|| format!("{}: no prepared reference", cell.id))?;
    ensure!(
        reference.len() == cell.n_rows,
        "{}: {} reference rows for {} rows",
        cell.id,
        reference.len(),
        cell.n_rows
    );
    let load = |disable_tree_ordering| -> Result<Forest> {
        let f = cell.forest(&LoadOptions {
            exact_sums: false,
            disable_tree_ordering,
            ..LoadOptions::default()
        })?;
        ensure!(!f.exact_sums(), "{}: exact sums still on", cell.id);
        Ok(f)
    };
    let (f_on, f_off) = (load(false)?, load(true)?);
    let (on, off) = (predict_all(cell, &f_on), predict_all(cell, &f_off));
    let mismatches = |p: &Predicted| {
        p.output
            .iter()
            .zip(&p.production)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    };
    let postprocessor = cell
        .oracle
        .as_ref()
        .map_or(Value::Null, |o| json!(o.postprocessor));
    Ok(json!({
        "id": cell.id,
        "dir": cell.dir,
        "files": cell.doc.get("files").cloned().unwrap_or(Value::Null),
        "rows": cell.n_rows,
        "groups": cell.n_groups(),
        "trees": f_on.trees().len(),
        "trees_moved_by_ordering": trees_moved(&f_on, &f_off),
        "postprocessor": postprocessor,
        "exact_sums": false,
        "primary": {
            "what": "tree ordering on against off, exact sums off",
            "tree_sum": diff(&on.tree_sum, &off.tree_sum),
            "raw_margin": diff(&on.raw_margin, &off.raw_margin),
        },
        "secondary": {
            "output": diff(&on.output, &off.output),
            "vs_reference": {
                "reference": "prepare's predictions (Treelite GTIL)",
                "ordering_on": diff(&on.output, reference),
                "ordering_off": diff(&off.output, reference),
            },
        },
        "production_vs_stages": {"ordering_on": mismatches(&on), "ordering_off": mismatches(&off)},
    }))
}

/// The measurement over `dirs`, one record per cell; a cell that is not LightGBM
/// or cannot be measured is listed under `skipped` with its reason.
pub fn run(dirs: &[std::path::PathBuf]) -> Result<Value> {
    let mut cells = Vec::new();
    let mut skipped = Vec::new();
    let mut worst: HashMap<&str, f64> = HashMap::new();
    for dir in dirs {
        let rec = Cell::load(dir).and_then(|c| {
            c.verify_files()?;
            ordering(&c)
        });
        match rec {
            Ok(r) => {
                for (key, v) in [
                    ("tree_sum", &r["primary"]["tree_sum"]),
                    ("raw_margin", &r["primary"]["raw_margin"]),
                    ("output", &r["secondary"]["output"]),
                    (
                        "ordering_on_vs_reference",
                        &r["secondary"]["vs_reference"]["ordering_on"],
                    ),
                    (
                        "ordering_off_vs_reference",
                        &r["secondary"]["vs_reference"]["ordering_off"],
                    ),
                ] {
                    let m = worst.entry(key).or_insert(0.0);
                    *m = m.max(v["max_abs"].as_f64().unwrap_or(f64::INFINITY));
                }
                eprintln!(
                    "{}: {} rows, tree sums differ on {} (max {:.3e}), raw margins max {:.3e}; \
                     vs GTIL on {:.3e}, off {:.3e}",
                    r["id"].as_str().unwrap_or_default(),
                    r["rows"],
                    r["primary"]["tree_sum"]["rows_differ"],
                    r["primary"]["tree_sum"]["max_abs"]
                        .as_f64()
                        .unwrap_or(f64::NAN),
                    r["primary"]["raw_margin"]["max_abs"]
                        .as_f64()
                        .unwrap_or(f64::NAN),
                    r["secondary"]["vs_reference"]["ordering_on"]["max_abs"]
                        .as_f64()
                        .unwrap_or(f64::NAN),
                    r["secondary"]["vs_reference"]["ordering_off"]["max_abs"]
                        .as_f64()
                        .unwrap_or(f64::NAN),
                );
                cells.push(r);
            }
            Err(e) => skipped.push(json!({"dir": dir, "reason": format!("{e:#}")})),
        }
    }
    let host = crate::system::host();
    Ok(json!({
        "command": "sweep_bench precision-order",
        "binary_sha256": crate::run::binary_sha256(),
        "source": crate::run::source_commit(),
        "host": {"os": host["os"], "arch": host["arch"], "cpu": host["cpu"]},
        "note": "primary: tree sums and raw margins, platform-independent f64 additions; \
                 secondary: outputs, which include the platform's exp under a sigmoid link, \
                 and GTIL's computed where the cell was prepared",
        "max_abs_over_cells": worst,
        "cells": cells,
        "skipped": skipped,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An import fixture as a LightGBM cell of 128-row groups, with its GTIL
    /// reference.
    fn fixture_cell(model: &str) -> Cell {
        let dir = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/import"
        ));
        let (data, n_rows, n_features) =
            crate::data::try_load_raw_f64(&dir.join("data.bin")).unwrap();
        let (reference, _, _) =
            crate::data::try_load_raw_f64(&dir.join(format!("{model}_reference.bin"))).unwrap();
        let offsets: Vec<usize> = (0..=n_rows / 128).map(|g| g * 128).collect();
        assert_eq!(offsets.last(), Some(&n_rows));
        Cell {
            id: format!("fixture/{model}/lightgbm/panel"),
            dir: dir.clone(),
            doc: json!({"files": {"model_treelite_bin": {"path": format!("{model}.bin")}}}),
            framework: "lightgbm".into(),
            status: "ready".into(),
            n_rows,
            n_features,
            entities: (0..offsets.len() as i64 - 1).collect(),
            offsets,
            walker_config: dir.join("walker_config.json"),
            model_treelite: dir.join(format!("{model}.bin")),
            model_native: None,
            reference: Some(reference),
            oracle: None,
            tl2cgen: None,
            lleaves: None,
            quickscorer: Err(String::new()),
            quickscorer_sha256: None,
            data,
        }
    }

    #[test]
    fn ordering_compares_both_loads_with_each_other_and_with_gtil() {
        for model in ["identity", "sigmoid_f64", "average_base"] {
            let cell = fixture_cell(model);
            let r = ordering(&cell).unwrap();
            assert_eq!(r["rows"], cell.n_rows, "{model}");
            assert_eq!(r["exact_sums"], false);
            assert_eq!(
                r["files"]["model_treelite_bin"]["path"],
                format!("{model}.bin")
            );
            for key in ["tree_sum", "raw_margin"] {
                let d = &r["primary"][key];
                assert!(
                    d["max_abs"].as_f64().unwrap() <= 1e-13,
                    "{model} {key}: {d}"
                );
            }
            // The fixtures' GTIL tolerance (tests/fixtures/import/manifest.json).
            for key in ["ordering_on", "ordering_off"] {
                let d = &r["secondary"]["vs_reference"][key];
                assert!(
                    d["max_abs"].as_f64().unwrap() <= 1e-14,
                    "{model} {key}: {d}"
                );
            }
            assert_eq!(r["production_vs_stages"]["ordering_on"], 0);
            assert_eq!(r["production_vs_stages"]["ordering_off"], 0);
        }
        let mut xgb = fixture_cell("identity");
        xgb.framework = "xgboost".into();
        assert!(ordering(&xgb).is_err());
        xgb.framework = "lightgbm".into();
        xgb.reference = None;
        assert!(ordering(&xgb).is_err());
    }

    #[test]
    fn diff_counts_bitwise_differences_and_finds_the_largest() {
        let a = [1.0, 2.0, f64::NAN, 4.0, -0.0];
        let b = [1.0, 2.5, f64::NAN, 3.0, 0.0];
        let d = diff(&a, &b);
        // NaN against the same NaN is equal; -0.0 against 0.0 is not bitwise.
        assert_eq!(d["rows_differ"], 3);
        assert_eq!(d["max_abs"], 1.0);
        assert_eq!(d["at_row"], 3);
        let d = diff(&[1.0, 0.0], &[f64::NAN, 0.0]);
        assert_eq!(d["max_abs"], Value::Null); // infinity serializes as null
        assert_eq!(d["at_row"], 0);
        assert_eq!(diff(&[1.0], &[1.0])["at_row"], Value::Null);
    }
}
