//! `TreeWalker`: high-performance tree ensemble inference for grouped prediction tasks.
//!
//! # Problem structure
//!
//! Grouped prediction tasks expand each entity into `max_group_width` rows and predict
//! a per-row probability. This creates a specific data structure that tree inference
//! can exploit:
//!
//! - **Constant features** (~78% of splits) are identical across all rows of an
//!   entity. Tree splits on these features are evaluated once, not per-row.
//!   This is the single biggest win: it turns O(trees × rows × depth) into
//!   O(trees × depth) for the constant portion of each tree.
//!
//! - **Varying features** may differ per row. Splits on these require
//!   partitioning the row set. When a varying feature is **monotonic** across the
//!   panel (e.g., elapsed time increases, remaining time decreases), the
//!   partition reduces to a prefix or suffix scan — no per-row evaluation.
//!   ~88% of varying splits are monotonic.
//!
//! `TreeWalker` implements *recursive partial evaluation* that walks constant
//! splits once, then partitions rows only at varying splits using generic
//! bitmasks (u32/u64/u128, supporting up to 128 rows per entity).
//!
//! # Usage
//!
//! ```ignore
//! use treewalker::forest::Forest;
//!
//! // Accepts .json or .bin (treelite binary v4, preferred for large models)
//! let forest = Forest::load("model_treelite.bin", "walker_config.json");
//!
//! // data: row-major f64 array, rows = entities × max_group_width, cols = features
//! // results: one probability per row
//! let mut results = vec![0.0f64; n_rows];
//! let h = forest.config.max_group_width;
//! for obs in 0..n_entities {
//!     forest.predict(&data, &mut results, obs * h, (obs + 1) * h);
//! }
//! ```

pub mod bench;
pub mod config;
pub mod mask;
pub mod forest;
pub mod parser;
pub mod predict;

pub use config::{AblationMode, ParseConfig};
pub use predict::PredictStats;

/// Load a raw f64 binary file: `u64 n_rows`, `u64 n_cols`, then `n_rows × n_cols` LE f64 values.
///
/// Returns `(data, n_rows, n_cols)`. Panics on I/O errors or malformed files.
/// For a non-panicking alternative, use [`try_load_raw_f64`].
pub fn load_raw_f64(path: impl AsRef<std::path::Path>) -> (Vec<f64>, usize, usize) {
    try_load_raw_f64(path.as_ref())
        .unwrap_or_else(|e| panic!("{}", e))
}

/// Load a raw f64 binary file, returning `Err` on I/O errors or malformed data.
///
/// Format: `u64 n_rows` (LE), `u64 n_cols` (LE), then `n_rows × n_cols` f64 values (LE).
pub fn try_load_raw_f64(path: &std::path::Path) -> Result<(Vec<f64>, usize, usize), String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    if bytes.len() < 16 {
        return Err(format!("{}: too short for header ({} bytes)", path.display(), bytes.len()));
    }

    let n_rows = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let n_cols = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;

    let expected = n_rows
        .checked_mul(n_cols)
        .and_then(|rc| rc.checked_mul(8))
        .and_then(|b| b.checked_add(16))
        .ok_or_else(|| format!(
            "{}: header overflow: {n_rows} rows × {n_cols} cols", path.display()
        ))?;
    if bytes.len() != expected {
        return Err(format!(
            "{}: size mismatch: got {}, expected {} ({n_rows} rows × {n_cols} cols)",
            path.display(), bytes.len(), expected,
        ));
    }

    let data: Vec<f64> = bytes[16..]
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    Ok((data, n_rows, n_cols))
}

/// Load variable-length group offsets from a binary file.
///
/// Format: `u64 n_groups`, then `(n_groups + 1)` u64 LE values representing
/// cumulative row offsets: `[0, end_of_group_0, end_of_group_1, ...]`.
/// Get current process RSS in kilobytes.
///
/// Linux: current RSS (`VmRSS`). macOS: peak RSS (`ru_maxrss`).
pub fn get_rss_kb() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    return rest.trim().trim_end_matches(" kB").trim()
                        .parse().unwrap_or(0);
                }
            }
        }
        0
    }
    #[cfg(target_os = "macos")]
    {
        unsafe {
            let mut info: libc::rusage = std::mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &raw mut info) == 0 {
                return info.ru_maxrss as usize / 1024;
            }
        }
        0
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { 0 }
}

pub fn load_group_offsets(path: &std::path::Path) -> Vec<usize> {
    let bytes = std::fs::read(path).expect("failed to read group_offsets file");
    assert!(bytes.len() >= 8, "group_offsets file too short for header");
    let n_groups = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let expected = 8 + (n_groups + 1) * 8;
    assert_eq!(
        bytes.len(), expected,
        "group_offsets size mismatch: got {}, expected {} ({n_groups} groups)",
        bytes.len(), expected,
    );
    bytes[8..]
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) as usize)
        .collect()
}
