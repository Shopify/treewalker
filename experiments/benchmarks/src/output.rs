//! The run directory: `run.json`, and per cell `manifest.json` plus four Parquet
//! tables, zstd-compressed.
//!
//! ```text
//! experiments/data/runs/<run_id>/
//!   run.json
//!   timer.parquet          back-to-back read pairs, raw ticks
//!   cells/<cell>/          written under a temporary name, renamed when complete
//!     manifest.json  samples.parquet  groups.parquet  counters.parquet  hw.parquet
//! ```
//!
//! Rows are collected in memory and written when a cell completes, outside any timed
//! interval. Unsupported or missing values are null, with a status column. Once a run
//! is finished, `treewalker-exp pack` rewrites its cells as one file per table.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray, UInt32Array,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde_json::Value;

/// A column's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    U64,
    U32,
    I64,
    F64,
    Bool,
    Str,
    OptU64,
    OptF64,
}

enum Col {
    U64(Vec<u64>),
    U32(Vec<u32>),
    Bool(Vec<bool>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    /// Interned: an index into the table's strings.
    Str(Vec<u32>),
    OptU64(Vec<Option<u64>>),
    OptF64(Vec<Option<f64>>),
}

fn filter<T: Copy>(v: &mut Vec<T>, keep: &[bool]) {
    let mut i = 0;
    v.retain(|_| {
        i += 1;
        keep[i - 1]
    });
}

/// One cell value, for [`Table::push`].
#[derive(Debug, Clone, Copy)]
pub enum V<'a> {
    U64(u64),
    U32(u32),
    I64(i64),
    F64(f64),
    B(bool),
    S(&'a str),
    Opt(Option<u64>),
    OptF(Option<f64>),
}

/// A table with a fixed schema, collected row by row.
pub struct Table {
    names: Vec<&'static str>,
    cols: Vec<Col>,
    strings: Vec<String>,
    index: HashMap<String, u32>,
}

impl Table {
    pub fn new(schema: &[(&'static str, Kind)]) -> Self {
        Self {
            names: schema.iter().map(|(n, _)| *n).collect(),
            cols: schema
                .iter()
                .map(|(_, k)| match k {
                    Kind::U64 => Col::U64(Vec::new()),
                    Kind::U32 => Col::U32(Vec::new()),
                    Kind::I64 => Col::I64(Vec::new()),
                    Kind::F64 => Col::F64(Vec::new()),
                    Kind::Bool => Col::Bool(Vec::new()),
                    Kind::Str => Col::Str(Vec::new()),
                    Kind::OptU64 => Col::OptU64(Vec::new()),
                    Kind::OptF64 => Col::OptF64(Vec::new()),
                })
                .collect(),
            strings: Vec::new(),
            index: HashMap::new(),
        }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.index.get(s) {
            return i;
        }
        let i = self.strings.len() as u32;
        self.strings.push(s.to_string());
        self.index.insert(s.to_string(), i);
        i
    }

    /// Append one row; values must follow the schema.
    pub fn push(&mut self, row: &[V<'_>]) {
        assert_eq!(row.len(), self.cols.len(), "row width");
        for (i, v) in row.iter().enumerate() {
            let interned = if let V::S(s) = v {
                Some(self.intern(s))
            } else {
                None
            };
            match (&mut self.cols[i], v) {
                (Col::U64(c), V::U64(x)) => c.push(*x),
                (Col::U32(c), V::U32(x)) => c.push(*x),
                (Col::I64(c), V::I64(x)) => c.push(*x),
                (Col::F64(c), V::F64(x)) => c.push(*x),
                (Col::Bool(c), V::B(x)) => c.push(*x),
                (Col::Str(c), V::S(_)) => c.push(interned.unwrap_or_default()),
                (Col::OptU64(c), V::Opt(x)) => c.push(*x),
                (Col::OptU64(c), V::U64(x)) => c.push(Some(*x)),
                (Col::OptF64(c), V::OptF(x)) => c.push(*x),
                _ => panic!("column {} has another type", self.names[i]),
            }
        }
    }

    /// Drop the rows whose `(a, b)` string columns equal one of `pairs`.
    pub fn retain_pairs(&mut self, a: &str, b: &str, pairs: &[(String, String)]) {
        let col = |name: &str| self.names.iter().position(|n| *n == name).expect("column");
        let (ia, ib) = (col(a), col(b));
        let (Col::Str(va), Col::Str(vb)) = (&self.cols[ia], &self.cols[ib]) else {
            panic!("string columns");
        };
        let keep: Vec<bool> = va
            .iter()
            .zip(vb)
            .map(|(&x, &y)| {
                let (x, y) = (&self.strings[x as usize], &self.strings[y as usize]);
                !pairs.iter().any(|(p, q)| p == x && q == y)
            })
            .collect();
        for c in &mut self.cols {
            match c {
                Col::U64(v) => filter(v, &keep),
                Col::U32(v) | Col::Str(v) => filter(v, &keep),
                Col::I64(v) => filter(v, &keep),
                Col::F64(v) => filter(v, &keep),
                Col::Bool(v) => filter(v, &keep),
                Col::OptU64(v) => filter(v, &keep),
                Col::OptF64(v) => filter(v, &keep),
            }
        }
    }

    /// Every row as JSON values, in schema order: for a child process to hand its
    /// rows to the parent.
    pub fn rows_json(&self) -> Vec<Vec<Value>> {
        (0..self.len())
            .map(|r| {
                self.cols
                    .iter()
                    .map(|c| match c {
                        Col::U64(v) => Value::from(v[r]),
                        Col::U32(v) => Value::from(v[r]),
                        Col::I64(v) => Value::from(v[r]),
                        Col::F64(v) => Value::from(v[r]),
                        Col::Bool(v) => Value::from(v[r]),
                        Col::Str(v) => Value::from(self.strings[v[r] as usize].as_str()),
                        Col::OptU64(v) => v[r].map_or(Value::Null, Value::from),
                        Col::OptF64(v) => v[r].map_or(Value::Null, Value::from),
                    })
                    .collect()
            })
            .collect()
    }

    /// Append rows from [`Self::rows_json`] of a table with the same schema.
    pub fn push_json(&mut self, row: &[Value]) -> Result<()> {
        anyhow::ensure!(row.len() == self.cols.len(), "row width {}", row.len());
        let mut vals = Vec::with_capacity(row.len());
        for (v, c) in row.iter().zip(&self.cols) {
            let bad = || anyhow::anyhow!("value {v} does not fit its column");
            vals.push(match c {
                Col::U64(_) => V::U64(v.as_u64().ok_or_else(bad)?),
                Col::U32(_) => V::U32(
                    v.as_u64()
                        .and_then(|x| u32::try_from(x).ok())
                        .ok_or_else(bad)?,
                ),
                Col::I64(_) => V::I64(v.as_i64().ok_or_else(bad)?),
                Col::F64(_) => V::F64(v.as_f64().ok_or_else(bad)?),
                Col::Bool(_) => V::B(v.as_bool().ok_or_else(bad)?),
                Col::Str(_) => V::S(v.as_str().ok_or_else(bad)?),
                Col::OptU64(_) => V::Opt(v.as_u64()),
                Col::OptF64(_) => V::OptF(v.as_f64()),
            });
        }
        self.push(&vals);
        Ok(())
    }

    pub fn len(&self) -> usize {
        match self.cols.first() {
            Some(Col::U64(c)) => c.len(),
            Some(Col::U32(c) | Col::Str(c)) => c.len(),
            Some(Col::I64(c)) => c.len(),
            Some(Col::F64(c)) => c.len(),
            Some(Col::Bool(c)) => c.len(),
            Some(Col::OptU64(c)) => c.len(),
            Some(Col::OptF64(c)) => c.len(),
            None => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn batch(&self) -> Result<RecordBatch> {
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        for (name, col) in self.names.iter().zip(&self.cols) {
            let (dt, nullable, array): (DataType, bool, ArrayRef) = match col {
                Col::U64(c) => (
                    DataType::UInt64,
                    false,
                    Arc::new(UInt64Array::from(c.clone())),
                ),
                Col::U32(c) => (
                    DataType::UInt32,
                    false,
                    Arc::new(UInt32Array::from(c.clone())),
                ),
                Col::I64(c) => (
                    DataType::Int64,
                    false,
                    Arc::new(Int64Array::from(c.clone())),
                ),
                Col::F64(c) => (
                    DataType::Float64,
                    false,
                    Arc::new(Float64Array::from(c.clone())),
                ),
                Col::Bool(c) => (
                    DataType::Boolean,
                    false,
                    Arc::new(BooleanArray::from(c.clone())),
                ),
                Col::Str(c) => (
                    DataType::Utf8,
                    false,
                    Arc::new(StringArray::from_iter_values(
                        c.iter().map(|&i| self.strings[i as usize].as_str()),
                    )),
                ),
                Col::OptU64(c) => (
                    DataType::UInt64,
                    true,
                    Arc::new(UInt64Array::from(c.clone())),
                ),
                Col::OptF64(c) => (
                    DataType::Float64,
                    true,
                    Arc::new(Float64Array::from(c.clone())),
                ),
            };
            fields.push(Field::new(*name, dt, nullable));
            arrays.push(array);
        }
        Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?)
    }

    /// Write the table as one zstd-compressed Parquet file.
    pub fn write(&self, path: &Path) -> Result<()> {
        let batch = self.batch()?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .build();
        let file =
            std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }
}

/// The `samples` table: one row per timed interval. `process` is 0 for the run's
/// own process and k for XGBoost's extra process k.
pub fn samples_table() -> Table {
    Table::new(&[
        ("sample_id", Kind::U64),
        ("suite", Kind::Str),
        ("cell", Kind::Str),
        ("variant", Kind::Str),
        ("method", Kind::Str),
        ("mode", Kind::Str),
        ("interface", Kind::Str),
        ("block", Kind::U32),
        ("batch", Kind::U32),
        ("position", Kind::U32),
        ("repetition", Kind::U32),
        ("process", Kind::U32),
        ("group", Kind::U32),
        ("n_groups", Kind::U32),
        ("rows", Kind::U32),
        ("ticks", Kind::U64),
    ])
}

/// The `groups` table: each group's index, entity and row count, and whether it is
/// timed or left out of a group sample.
pub fn groups_table() -> Table {
    Table::new(&[
        ("suite", Kind::Str),
        ("cell", Kind::Str),
        ("group", Kind::U32),
        ("entity", Kind::I64),
        ("first_row", Kind::U64),
        ("rows", Kind::U32),
        ("timed", Kind::Bool),
    ])
}

/// The `counters` table: work counters per variant and group.
pub fn counters_table() -> Table {
    Table::new(&[
        ("suite", Kind::Str),
        ("cell", Kind::Str),
        ("variant", Kind::Str),
        ("group", Kind::U32),
        ("counters_version", Kind::U32),
        ("constant_steps", Kind::U64),
        ("varying_splits", Kind::U64),
        ("unsplit_skips", Kind::U64),
        ("recursive_calls", Kind::U64),
        ("leaf_hits", Kind::U64),
        ("leaf_adds", Kind::U64),
        ("partition_row_evals", Kind::U64),
        ("scan_row_evals", Kind::U64),
        ("scan_compares", Kind::U64),
        ("scan_missing_checks", Kind::U64),
        ("precompute_row_evals", Kind::U64),
        ("precompute_sort_compares", Kind::U64),
        ("precompute_sweep_compares", Kind::U64),
        ("precompute_mask_writes", Kind::U64),
    ])
}

/// The `hw` table: event deltas per cell, variant, method, mode and block.
pub fn hw_table() -> Table {
    let mut schema = vec![
        ("suite", Kind::Str),
        ("cell", Kind::Str),
        ("variant", Kind::Str),
        ("method", Kind::Str),
        ("mode", Kind::Str),
        ("block", Kind::U32),
        ("process", Kind::U32),
    ];
    schema.extend(crate::pmu::COLUMNS.iter().map(|c| (*c, Kind::OptU64)));
    schema.extend([
        ("time_enabled", Kind::OptU64),
        ("time_running", Kind::OptU64),
        ("rusage_nvcsw", Kind::OptU64),
        ("rusage_nivcsw", Kind::OptU64),
        ("status", Kind::Str),
    ]);
    Table::new(&schema)
}

/// Write JSON with a trailing newline, replacing the file atomically.
pub fn write_json(path: &Path, value: &Value) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(value)? + "\n")
        .with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// One cell's output directory, written under a temporary name.
pub struct CellDir {
    tmp: PathBuf,
    done: PathBuf,
    /// The cell ID this directory belongs to.
    id: String,
}

impl CellDir {
    /// `slug` must be one safe directory name ([`crate::artifacts::Cell::slug`]); the
    /// destination is always a child of `cells/`, never `cells/` itself.
    pub fn create(run_dir: &Path, slug: &str, id: &str) -> Result<Self> {
        anyhow::ensure!(
            !slug.is_empty()
                && !slug.starts_with('.')
                && Path::new(slug).components().count() == 1
                && matches!(
                    Path::new(slug).components().next(),
                    Some(std::path::Component::Normal(_))
                ),
            "unsafe output directory name {slug:?}"
        );
        let cells = run_dir.join("cells");
        let tmp = cells.join(format!(".tmp-{slug}"));
        if tmp.exists() {
            std::fs::remove_dir_all(&tmp)?;
        }
        std::fs::create_dir_all(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        Ok(Self {
            tmp,
            done: cells.join(slug),
            id: id.to_string(),
        })
    }

    pub fn path(&self, file: &str) -> PathBuf {
        self.tmp.join(file)
    }

    /// Rename the directory to its final name, replacing an older result of the
    /// same cell only: a directory whose manifest names another cell, or none, is
    /// never removed.
    pub fn finish(self) -> Result<PathBuf> {
        if self.done.exists() {
            let owner = std::fs::read_to_string(self.done.join("manifest.json"))
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                .and_then(|d| d.get("id").and_then(Value::as_str).map(ToString::to_string));
            anyhow::ensure!(
                owner.as_deref() == Some(self.id.as_str()),
                "{} holds {:?}, not {:?}; not replaced",
                self.done.display(),
                owner,
                self.id
            );
            std::fs::remove_dir_all(&self.done)?;
        }
        std::fs::rename(&self.tmp, &self.done)?;
        Ok(self.done)
    }
}

/// The resume key of a finished cell, from its `manifest.json`.
/// The tables a finished cell holds; resume requires every one.
pub const CELL_FILES: [&str; 5] = [
    "manifest.json",
    "samples.parquet",
    "groups.parquet",
    "counters.parquet",
    "hw.parquet",
];

pub fn finished_key(run_dir: &Path, slug: &str) -> Option<String> {
    let dir = run_dir.join("cells").join(slug);
    if !CELL_FILES.iter().all(|f| dir.join(f).is_file()) {
        return None;
    }
    let doc: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).ok()?).ok()?;
    doc.get("resume_key")?.as_str().map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsafe_names_never_touch_finished_cells() {
        let run = std::env::temp_dir().join(format!("tw-preserve-{}", std::process::id()));
        let done = run.join("cells").join("already-finished");
        std::fs::create_dir_all(&done).unwrap();
        std::fs::write(done.join("marker.txt"), "kept").unwrap();
        for bad in ["", ".", "..", "../x", "a/b", ".tmp-x"] {
            assert!(CellDir::create(&run, bad, "x").is_err(), "{bad:?}");
        }
        let slug = crate::artifacts::Cell::slug;
        for bad in ["", "a//b", "a/../b", "./a", "a/.hidden", "a\nb", "a\u{e9}"] {
            assert!(slug(bad).is_err(), "{bad:?}");
        }
        assert_eq!(
            slug("support/nt50_md4_h1/lightgbm/panel").unwrap(),
            "support_2fnt50_5fmd4_5fh1_2flightgbm_2fpanel"
        );
        // Injective: IDs that differ only around `/` and `_` stay distinct.
        assert_ne!(slug("a_/b").unwrap(), slug("a/_b").unwrap());
        assert_ne!(slug("a_2fb").unwrap(), slug("a/b").unwrap());
        // A finished directory of another cell is never replaced.
        let other = run.join("cells").join("theirs");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("manifest.json"), r#"{"id": "their/cell"}"#).unwrap();
        let mine = CellDir::create(&run, "theirs", "my/cell").unwrap();
        assert!(mine.finish().is_err());
        assert!(other.join("manifest.json").exists());
        let again = CellDir::create(&run, "theirs", "their/cell").unwrap();
        std::fs::write(again.path("manifest.json"), r#"{"id": "their/cell"}"#).unwrap();
        assert!(again.finish().is_ok());
        assert_eq!(
            std::fs::read_to_string(done.join("marker.txt")).unwrap(),
            "kept"
        );
        std::fs::remove_dir_all(&run).unwrap();
    }

    #[test]
    fn tables_round_trip_through_parquet() {
        use parquet::file::reader::FileReader as _;
        let dir = std::env::temp_dir().join(format!("tw-output-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut t = hw_table();
        let mut row = vec![
            V::S("s"),
            V::S("c"),
            V::S("all-on"),
            V::S("treewalker"),
            V::S("batch"),
            V::U32(0),
            V::U32(3),
        ];
        row.extend([V::Opt(None); 11]);
        row.push(V::Opt(Some(7)));
        row.push(V::S("unsupported"));
        t.push(&row);
        // Rows survive the JSON a child process hands its parent.
        let json = t.rows_json();
        t.push_json(&json[0]).unwrap();
        assert_eq!(t.rows_json(), [json[0].clone(), json[0].clone()]);
        assert!(t.push_json(&json[0][1..]).is_err());
        assert_eq!(t.len(), 2);
        let path = dir.join("hw.parquet");
        t.write(&path).unwrap();
        let reader =
            parquet::file::reader::SerializedFileReader::new(std::fs::File::open(&path).unwrap())
                .unwrap();
        assert_eq!(reader.metadata().file_metadata().num_rows(), 2);

        // groups.timed is a Parquet boolean, and survives the child's JSON too.
        let mut g = groups_table();
        for (i, timed) in [true, false].into_iter().enumerate() {
            g.push(&[
                V::S("s"),
                V::S("c"),
                V::U32(i as u32),
                V::I64(-1),
                V::U64(0),
                V::U32(1),
                V::B(timed),
            ]);
        }
        let json = g.rows_json();
        assert_eq!(json[1][6], Value::Bool(false));
        g.push_json(&json[0]).unwrap();
        let path = dir.join("groups.parquet");
        g.write(&path).unwrap();
        let reader =
            parquet::file::reader::SerializedFileReader::new(std::fs::File::open(&path).unwrap())
                .unwrap();
        let schema = reader.metadata().file_metadata().schema_descr();
        let timed = schema.column(6);
        assert_eq!(timed.name(), "timed");
        assert_eq!(timed.physical_type(), parquet::basic::Type::BOOLEAN);
        assert_eq!(reader.metadata().file_metadata().num_rows(), 3);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
