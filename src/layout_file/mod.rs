//! Experimental native-target layout files (`layout-file` feature).
//!
//! The schema is private and versioned, not a Rust memory dump or a portable model
//! interchange format. See `docs/layout-file.md` for compatibility and trust.

pub(crate) mod codec;

use crate::{Forest, LoadError};
use codec::{Codec, Encoder};
use std::io::{self, BufWriter, Write};
use std::path::Path;

fn header() -> Vec<u8> {
    format!(
        "TWLAYOUT\0v1\0{}\0{}\0{}\0{}\0{}\0links={}\0",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
        usize::BITS,
        cfg!(target_endian = "little"),
        cfg!(any(test, feature = "research")),
        crate::load::layout::LINK_STRIDE
    )
    .into_bytes()
}

fn fingerprint(config: &crate::WalkerConfig) -> io::Result<u64> {
    let mut w = Encoder::new(io::sink());
    config.put(&mut w)?;
    Ok(w.checksum.finish())
}

pub(crate) fn error(error: io::Error) -> LoadError {
    if error.kind() == io::ErrorKind::InvalidData || error.kind() == io::ErrorKind::UnexpectedEof {
        LoadError::Layout(error.to_string())
    } else {
        LoadError::Io(error)
    }
}

impl Forest {
    /// Save this forest's built prediction layout (`layout-file` feature).
    ///
    /// Experimental, versioned and target-specific; rebuild from the source model
    /// after a format or crate-version change. Includes the grouping configuration.
    /// Flushes buffered writes, but does not fsync. Publish a completed file by an
    /// atomic rename; never overwrite a file that may have live mapped readers.
    pub fn save_layout(&self, path: impl AsRef<Path>) -> Result<(), LoadError> {
        self.write_layout(path.as_ref()).map_err(error)
    }

    fn write_layout(&self, path: &Path) -> io::Result<()> {
        let mut w = Encoder::new(BufWriter::new(std::fs::File::create(path)?));
        w.bytes(&header())?;
        fingerprint(&self.model.config)?.put(&mut w)?;
        self.model.threshold_type.put(&mut w)?;
        self.model.put(&mut w)?;
        w.inner.write_all(&w.checksum.finish().to_le_bytes())?;
        w.inner.flush()
    }
}
