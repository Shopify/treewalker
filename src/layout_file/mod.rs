//! Experimental native-target layout files (`layout-file` feature).
//!
//! The schema is private and versioned, not a Rust memory dump or a portable model
//! interchange format. See `docs/layout-file.md` for compatibility and trust.

pub(crate) mod codec;
#[cfg(test)]
pub(crate) mod testsupport;
pub(crate) mod validate;

use crate::{Forest, LoadError};
use codec::{Checksum, Codec, Decoder, Encoder, invalid};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

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

fn open_layout(path: &Path) -> Result<(std::fs::File, usize), LoadError> {
    let file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    if size > 4u64 << 30 {
        return Err(LoadError::Limit("layout file exceeds 4 GiB".into()));
    }
    let len = usize::try_from(size)
        .ok()
        .filter(|&n| n >= 8)
        .ok_or_else(|| LoadError::Layout("layout file size".into()))?;
    Ok((file, len))
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
    /// Load a layout file into owned arrays (`layout-file` feature).
    ///
    /// Checks compatibility, corruption and the invariants of the unchecked kernel
    /// before returning. No source-model build or whole-file heap buffer is needed.
    /// Files are currently limited to 4 GiB and group widths to 2^20 rows.
    pub fn load_layout(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let (file, len) = open_layout(path.as_ref())?;
        let mut r = Decoder {
            inner: BufReader::new(file),
            at: 0,
            total: len - 8,
            checksum: Checksum::default(),
        };
        Self::read_layout(&mut r).map_err(error)
    }

    fn read_layout<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        let expected = header();
        let mut got = vec![0; expected.len()];
        r.bytes(&mut got)?;
        if got != expected {
            return Err(invalid("layout format/crate/target/research mismatch"));
        }
        let config_hash = u64::get(r)?;
        let threshold_type = crate::model::ThresholdType::get(r)?;
        let model = crate::model::Model::get(r)?;
        if r.at != r.total
            || fingerprint(&model.config)? != config_hash
            || model.threshold_type != threshold_type
        {
            return Err(invalid("layout extent/configuration/precision mismatch"));
        }
        let mut footer = [0; 8];
        let checksum = std::mem::take(&mut r.checksum).finish();
        r.total += 8;
        r.bytes(&mut footer)?;
        if u64::from_le_bytes(footer) != checksum {
            return Err(invalid("layout checksum mismatch"));
        }
        validate::model(&model)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

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
