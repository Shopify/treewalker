//! Binary data formats used by the benchmark artifacts.

/// Load a raw f64 binary file: `u64 n_rows`, `u64 n_cols`, then `n_rows × n_cols` LE f64 values.
///
/// Returns `(data, n_rows, n_cols)`. Panics on I/O errors or malformed files.
/// For a non-panicking alternative, use [`try_load_raw_f64`].
pub fn load_raw_f64(path: impl AsRef<std::path::Path>) -> (Vec<f64>, usize, usize) {
    try_load_raw_f64(path.as_ref()).unwrap_or_else(|e| panic!("{}", e))
}

/// Load a raw f64 binary file, returning `Err` on I/O errors or malformed data.
///
/// Format: `u64 n_rows` (LE), `u64 n_cols` (LE), then `n_rows × n_cols` f64 values (LE).
pub fn try_load_raw_f64(path: &std::path::Path) -> Result<(Vec<f64>, usize, usize), String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    if bytes.len() < 16 {
        return Err(format!(
            "{}: too short for header ({} bytes)",
            path.display(),
            bytes.len()
        ));
    }

    let n_rows = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let n_cols = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;

    let expected = n_rows
        .checked_mul(n_cols)
        .and_then(|rc| rc.checked_mul(8))
        .and_then(|b| b.checked_add(16))
        .ok_or_else(|| {
            format!(
                "{}: header overflow: {n_rows} rows × {n_cols} cols",
                path.display()
            )
        })?;
    if bytes.len() != expected {
        return Err(format!(
            "{}: size mismatch: got {}, expected {} ({n_rows} rows × {n_cols} cols)",
            path.display(),
            bytes.len(),
            expected,
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
/// Format: `u64 n_groups`, then `(n_groups + 1)` u64 LE offsets.
pub fn load_group_offsets(path: &std::path::Path) -> Vec<usize> {
    let bytes = std::fs::read(path).expect("failed to read group_offsets file");
    assert!(bytes.len() >= 8, "group_offsets file too short for header");
    let n_groups = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let expected = 8 + (n_groups + 1) * 8;
    assert_eq!(
        bytes.len(),
        expected,
        "group_offsets size mismatch: got {}, expected {} ({n_groups} groups)",
        bytes.len(),
        expected,
    );
    bytes[8..]
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) as usize)
        .collect()
}
