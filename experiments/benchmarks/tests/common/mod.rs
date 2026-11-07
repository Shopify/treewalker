use std::path::{Path, PathBuf};

/// Number of `"threshold": ,` entries in a Treelite JSON export, found without the
/// parser under test. Treelite writes an empty value for nonfinite thresholds, and the
/// loader rejects such dumps rather than repairing them (docs/treelite-loading.md).
pub(super) fn empty_thresholds(path: &Path) -> usize {
    const KEY: &[u8] = b"\"threshold\":";
    let bytes = std::fs::read(path).unwrap();
    (0..bytes.len().saturating_sub(KEY.len()))
        .filter(|&i| bytes[i..].starts_with(KEY))
        .filter(|&i| {
            bytes[i + KEY.len()..]
                .iter()
                .find(|b| !b.is_ascii_whitespace())
                .is_some_and(|&b| b == b',' || b == b'}')
        })
        .count()
}

/// A copy of a Treelite JSON export, under the target's temporary directory, with
/// every empty threshold filled with 0: what the export would be without them.
pub(super) fn with_empty_thresholds_filled(path: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COPIES: AtomicUsize = AtomicUsize::new(0);
    const KEY: &[u8] = b"\"threshold\":";
    let bytes = std::fs::read(path).unwrap();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(KEY) {
            out.extend_from_slice(KEY);
            i += KEY.len();
            let next = bytes[i..]
                .iter()
                .position(|b| !b.is_ascii_whitespace())
                .map_or(bytes.len(), |n| i + n);
            if matches!(bytes.get(next), Some(b',' | b'}')) {
                out.extend_from_slice(b" 0");
                i = next;
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("filled-thresholds");
    std::fs::create_dir_all(&dir).unwrap();
    let n = COPIES.fetch_add(1, Ordering::Relaxed);
    let filled = dir.join(format!("{}-{n}.json", std::process::id()));
    std::fs::write(&filled, out).unwrap();
    filled
}
