//! Incremental, crash-safe CSV writer.
//!
//! Opens in append mode and tracks which rows have already been written
//! (by a composite key) so the benchmark can resume after interruption.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Crash-safe incremental CSV writer.
///
/// On construction, reads any existing CSV to populate `written_keys`.
/// New rows are appended and flushed immediately.
pub struct CsvWriter {
    path: PathBuf,
    columns: Vec<String>,
    written_keys: HashSet<String>,
    key_indices: Vec<usize>,
    file: File,
    has_header: bool,
}

impl CsvWriter {
    /// Create or open a CSV file with the given columns.
    ///
    /// `key_columns` are a subset of `columns` used to determine whether
    /// a row has already been written (for resume support).
    pub fn new(path: &Path, columns: &[&str], key_columns: &[&str]) -> Self {
        let col_vec: Vec<String> = columns.iter().map(|s| (*s).to_string()).collect();
        let key_indices: Vec<usize> = key_columns
            .iter()
            .map(|k| {
                columns
                    .iter()
                    .position(|c| c == k)
                    .unwrap_or_else(|| panic!("key column {k:?} not found in columns"))
            })
            .collect();

        let mut written_keys = HashSet::new();
        let mut has_header = false;

        // Read existing file to populate written_keys.
        if path.exists()
            && let Ok(file) = File::open(path)
        {
            let reader = BufReader::new(file);
            for (i, line) in reader.lines().enumerate() {
                let Ok(line) = line else { continue };
                if i == 0 {
                    has_header = true;
                    continue; // skip header
                }
                let fields: Vec<&str> = line.split(',').collect();
                let key = Self::key_from_fields(&fields, &key_indices);
                written_keys.insert(key);
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| panic!("failed to open CSV {}: {e}", path.display()));

        let mut writer = Self {
            path: path.to_path_buf(),
            columns: col_vec,
            written_keys,
            key_indices,
            file,
            has_header,
        };

        if !writer.has_header {
            let header = columns.join(",");
            writeln!(writer.file, "{header}").expect("failed to write CSV header");
            writer.file.flush().expect("failed to flush CSV");
            writer.has_header = true;
        }

        writer
    }

    /// Check if a row with the given key values has already been written.
    pub fn is_done(&self, key_values: &[&str]) -> bool {
        let key = key_values.join("|");
        self.written_keys.contains(&key)
    }

    /// Write a row to the CSV. Values must be in column order.
    pub fn write_row(&mut self, values: &[&str]) {
        assert_eq!(
            values.len(),
            self.columns.len(),
            "row has {} values but {} columns",
            values.len(),
            self.columns.len()
        );
        let key = Self::key_from_fields(values, &self.key_indices);
        if self.written_keys.contains(&key) {
            return; // already written
        }
        let line = values.join(",");
        writeln!(self.file, "{line}").expect("failed to write CSV row");
        self.file.flush().expect("failed to flush CSV");
        self.written_keys.insert(key);
    }

    fn key_from_fields(fields: &[&str], indices: &[usize]) -> String {
        indices
            .iter()
            .map(|&i| *fields.get(i).unwrap_or(&""))
            .collect::<Vec<&str>>()
            .join("|")
    }

    /// Path to the CSV file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of data rows written (excluding header).
    pub fn n_rows(&self) -> usize {
        self.written_keys.len()
    }
}
