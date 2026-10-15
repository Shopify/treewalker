//! Binary data formats of the prepared artifacts, as `treewalker_exp.formats` writes them.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};

/// Load a raw f64 binary file: `u64 n_rows`, `u64 n_cols`, then `n_rows × n_cols` LE f64 values.
///
/// Returns `(data, n_rows, n_cols)`. Panics on I/O errors or malformed files.
/// For a non-panicking alternative, use [`try_load_raw_f64`].
pub fn load_raw_f64(path: impl AsRef<Path>) -> (Vec<f64>, usize, usize) {
    try_load_raw_f64(path.as_ref()).unwrap_or_else(|e| panic!("{e:#}"))
}

/// Load a raw f64 binary file, returning `Err` on I/O errors or malformed data.
///
/// Format: `u64 n_rows` (LE), `u64 n_cols` (LE), then `n_rows × n_cols` f64 values (LE).
pub fn try_load_raw_f64(path: &Path) -> Result<(Vec<f64>, usize, usize)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        bytes.len() >= 16,
        "{}: too short for header ({} bytes)",
        path.display(),
        bytes.len()
    );
    let n_rows = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
    let n_cols = u64::from_le_bytes(bytes[8..16].try_into()?) as usize;
    let expected = n_rows
        .checked_mul(n_cols)
        .and_then(|rc| rc.checked_mul(8))
        .and_then(|b| b.checked_add(16));
    ensure!(
        expected == Some(bytes.len()),
        "{}: {} bytes, header says {n_rows} rows × {n_cols} cols",
        path.display(),
        bytes.len()
    );
    let data = bytes[16..]
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    Ok((data, n_rows, n_cols))
}

/// Load variable-length group offsets: `u64 n_groups`, then `n_groups + 1` LE u64
/// offsets, starting at 0 and strictly increasing.
pub fn load_group_offsets(path: &Path) -> Result<Vec<usize>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len() >= 8, "{}: too short for header", path.display());
    let n_groups = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
    ensure!(
        bytes.len() == 8 + (n_groups + 1) * 8,
        "{}: {} bytes for {n_groups} groups",
        path.display(),
        bytes.len()
    );
    let offsets: Vec<usize> = bytes[8..]
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) as usize)
        .collect();
    ensure!(
        offsets[0] == 0 && offsets.windows(2).all(|w| w[0] < w[1]),
        "{}: offsets must start at 0 and strictly increase",
        path.display()
    );
    Ok(offsets)
}

/// A one-dimensional NumPy array, or a two-dimensional one with a single column.
#[derive(Debug, Clone, PartialEq)]
pub enum Npy {
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl Npy {
    pub fn into_f64(self) -> Vec<f64> {
        match self {
            Self::F64(v) => v,
            Self::I64(v) => v.into_iter().map(|x| x as f64).collect(),
        }
    }

    pub fn into_i64(self) -> Result<Vec<i64>> {
        match self {
            Self::I64(v) => Ok(v),
            Self::F64(_) => bail!("expected an integer array"),
        }
    }
}

/// Read a little-endian `<i8` or `<f8` `.npy` file of shape `(n,)` or `(n, 1)`, C order.
pub fn read_npy(path: &Path) -> Result<Npy> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    parse_npy(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn parse_npy(bytes: &[u8]) -> Result<Npy> {
    ensure!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "not a .npy file"
    );
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            ensure!(bytes.len() >= 12, "truncated header");
            (u32::from_le_bytes(bytes[8..12].try_into()?) as usize, 12)
        }
        v => bail!("unsupported .npy version {v}"),
    };
    ensure!(bytes.len() >= start + header_len, "truncated header");
    let header = std::str::from_utf8(&bytes[start..start + header_len])?;
    let field = |key: &str| -> Result<&str> {
        let at = header
            .find(&format!("'{key}':"))
            .with_context(|| format!("header has no {key}"))?;
        Ok(header[at + key.len() + 3..].trim_start())
    };
    ensure!(
        field("fortran_order")?.starts_with("False"),
        "Fortran order is not supported"
    );
    let descr = field("descr")?;
    let shape = field("shape")?;
    let shape = &shape[1..shape.find(')').context("malformed shape")?];
    let dims: Vec<usize> = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let n = match dims.as_slice() {
        [n] | [n, 1] => *n,
        _ => bail!("expected shape (n,) or (n, 1), got {dims:?}"),
    };
    let body = &bytes[start + header_len..];
    ensure!(
        body.len() == n * 8,
        "{} data bytes for {n} values",
        body.len()
    );
    let words = body.chunks_exact(8).map(|c| c.try_into().unwrap());
    if descr.starts_with("'<i8'") {
        Ok(Npy::I64(words.map(i64::from_le_bytes).collect()))
    } else if descr.starts_with("'<f8'") {
        Ok(Npy::F64(words.map(f64::from_le_bytes).collect()))
    } else {
        bail!("unsupported dtype {}", &descr[..descr.len().min(8)])
    }
}

/// SHA-256 of a file, as lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = std::io::Read::read(&mut file, &mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// SHA-256 of bytes, as lowercase hex.
pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(digest: &[u8]) -> String {
    use std::fmt::Write as _;
    digest.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(descr: &str, shape: &str, body: &[u8]) -> Vec<u8> {
        let header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn reads_integer_and_float_arrays() {
        let ints: Vec<u8> = [3i64, -1].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            parse_npy(&npy("<i8", "(2,)", &ints)).unwrap(),
            Npy::I64(vec![3, -1])
        );
        let floats = 0.5f64.to_le_bytes();
        assert_eq!(
            parse_npy(&npy("<f8", "(1, 1)", &floats)).unwrap(),
            Npy::F64(vec![0.5])
        );
    }

    #[test]
    fn rejects_other_shapes_and_types() {
        assert!(parse_npy(&npy("<f8", "(1, 2)", &[0; 16])).is_err());
        assert!(parse_npy(&npy("<f4", "(2,)", &[0; 8])).is_err());
        assert!(parse_npy(&npy("<i8", "(2,)", &[0; 8])).is_err());
    }

    #[test]
    fn hashes_match_sha256() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
