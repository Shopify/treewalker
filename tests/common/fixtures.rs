//! The import fixtures, in `tests/fixtures/import`.

use std::path::PathBuf;

/// The fixture `name`'s path.
pub fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/import")
        .join(name)
}

/// The fixture `name`'s bytes.
pub fn bytes(name: &str) -> Vec<u8> {
    let path = path(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A data or reference fixture's values: after a header of two `u64`s, the row and
/// column counts, the values row-major as little-endian `f64`s.
pub fn values(name: &str) -> Vec<f64> {
    bytes(name)[16..]
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().expect("8 bytes")))
        .collect()
}
