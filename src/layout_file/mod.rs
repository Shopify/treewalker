//! Experimental native-target layout files (`layout-file` feature).
//!
//! The schema is private and versioned, not a Rust memory dump or a portable model
//! interchange format. See `docs/layout-file.md` for compatibility and trust.

pub(crate) mod codec;
#[cfg(all(unix, target_pointer_width = "64"))]
pub(crate) mod storage;
#[cfg(test)]
pub(crate) mod testsupport;
pub(crate) mod validate;

use crate::{Forest, LoadError};
use codec::{Checksum, Codec, Decoder, Encoder, invalid};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

/// The start of every layout file, whatever its schema or build.
const MAGIC: &str = "TWLAYOUT\0";

fn header() -> Vec<u8> {
    format!(
        "{MAGIC}v1\0{}\0build={}\0{}\0{}\0{}\0links={}\0",
        env!("CARGO_PKG_VERSION"),
        env!("TREEWALKER_BUILD_ID"),
        std::env::consts::ARCH,
        usize::BITS,
        cfg!(target_endian = "little"),
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

/// A layout file whose header, `found`, is not this build's: its fields and this
/// build's, for the caller that rebuilds it from its source model to report. The
/// file's bytes are untrusted, so all but printable ASCII are escaped.
fn mismatch(found: &[u8]) -> LoadError {
    let fields = |header: &[u8]| {
        header
            .split(|&b| b == 0)
            .filter(|field| !field.is_empty())
            .map(|field| field.escape_ascii().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    // The header's last whole field; a shorter header runs into the body.
    let found = &found[..found.iter().rposition(|&b| b == 0).map_or(0, |i| i + 1)];
    LoadError::LayoutMismatch(format!(
        "the file says {}, this build {}",
        fields(found),
        fields(&header())
    ))
}

fn fingerprint(config: &crate::WalkerConfig) -> io::Result<u64> {
    let mut w = Encoder::new(io::sink());
    config.put(&mut w)?;
    Ok(w.checksum.finish())
}

fn error(error: io::Error) -> LoadError {
    if error.kind() == io::ErrorKind::InvalidData || error.kind() == io::ErrorKind::UnexpectedEof {
        LoadError::Layout(error.to_string())
    } else if error.kind() == io::ErrorKind::FileTooLarge {
        LoadError::Limit(error.to_string())
    } else {
        LoadError::Io(error)
    }
}

fn production_only() -> LoadError {
    LoadError::Unsupported(
        "layout files are production-only; research builds rebuild from the source model".into(),
    )
}

impl Forest {
    /// Map a layout file, borrowing its largest arrays (`layout-file`, 64-bit Unix).
    ///
    /// Runs the same compatibility, corruption and structural checks as the owned
    /// loader. Other metadata remains owned; the source model is not decoded.
    /// Research builds return `LoadError::Unsupported` and rebuild from source.
    ///
    /// # Safety
    /// The underlying file must not be modified or truncated by any thread or
    /// process until the returned forest, all its clones, and all predictors that
    /// retain it have been dropped. Read-only/private mapping does not enforce
    /// this. Rename a new immutable artifact into place rather than overwriting
    /// the mapped file. This contract also applies while validation is running.
    #[cfg(all(unix, target_pointer_width = "64"))]
    pub unsafe fn map_layout(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        if cfg!(feature = "research") {
            return Err(production_only());
        }
        // SAFETY: the caller supplies the same immutable-file contract.
        unsafe { Self::map_layout_impl(path) }
    }

    /// Internal production-format reader, also exercised by white-box tests.
    /// The file must remain immutable until all returned mapping owners drop.
    #[cfg(all(unix, target_pointer_width = "64"))]
    pub(crate) unsafe fn map_layout_impl(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let (file, len) = open_layout(path.as_ref())?;
        // SAFETY: the caller guarantees immutability throughout all mapping owners.
        let mapping = Arc::new(unsafe { storage::Mapping::new(&file, len) }.map_err(error)?);
        let mut r = Decoder {
            inner: io::empty(),
            at: 0,
            total: mapping.len - 8,
            checksum: Checksum::default(),
            mapping: Some(mapping),
        };
        Self::read_layout(&mut r)
    }

    /// Load a layout file into owned arrays (`layout-file` feature).
    ///
    /// Checks compatibility, corruption and the invariants of the unchecked kernel
    /// before returning. No source-model build or whole-file heap buffer is needed.
    /// Files are currently limited to 4 GiB and group widths to 2^20 rows.
    /// Research builds return `LoadError::Unsupported` and rebuild from source.
    pub fn load_layout(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        if cfg!(feature = "research") {
            return Err(production_only());
        }
        Self::load_layout_impl(path)
    }

    pub(crate) fn load_layout_impl(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let (file, len) = open_layout(path.as_ref())?;
        let mut r = Decoder {
            inner: BufReader::new(file),
            at: 0,
            total: len - 8,
            checksum: Checksum::default(),
            #[cfg(all(unix, target_pointer_width = "64"))]
            mapping: None,
        };
        Self::read_layout(&mut r)
    }

    /// A layout file whose header, read first, is this build's. A header that starts
    /// as one but differs is from another schema, crate version, build or target.
    fn read_layout<R: Read>(r: &mut Decoder<R>) -> Result<Self, LoadError> {
        let expected = header();
        let mut got = vec![0; expected.len()];
        r.bytes(&mut got).map_err(error)?;
        if !got.starts_with(MAGIC.as_bytes()) {
            return Err(LoadError::Layout("not a layout file".into()));
        }
        if got != expected {
            return Err(mismatch(&got));
        }
        Self::read_body(r).map_err(error)
    }

    fn read_body<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
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
    /// after a source, format or crate-version change. Includes the grouping
    /// configuration.
    /// Flushes buffered writes, but does not fsync. Publish a completed file by an
    /// atomic rename; never overwrite a file that may have live mapped readers.
    /// Research builds return `LoadError::Unsupported` and rebuild from source.
    pub fn save_layout(&self, path: impl AsRef<Path>) -> Result<(), LoadError> {
        if cfg!(feature = "research") {
            return Err(production_only());
        }
        self.save_layout_impl(path)
    }

    pub(crate) fn save_layout_impl(&self, path: impl AsRef<Path>) -> Result<(), LoadError> {
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
