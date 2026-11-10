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

/// The fields that follow the magic, each ended by a NUL: the schema version; the
/// crate version and build ID of the build that wrote the file; and its target's
/// architecture, pointer width, endianness and link stride.
const FIELDS: usize = 7;

/// The fields that say which build wrote a file, for a mismatch's message. A reader
/// does not compare them: a file loads in any build of its schema version and target.
const WRITER: std::ops::Range<usize> = 1..3;

fn header() -> Vec<u8> {
    format!(
        "{MAGIC}v2\0{}\0build={}\0{}\0{}\0{}\0links={}\0",
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

/// Whether `found`, a file's header, has the schema version and the target of `ours`:
/// the same fields, the writer's aside.
fn same_format(found: &[u8], ours: &[u8]) -> bool {
    fn compared(header: &[u8]) -> impl Iterator<Item = (usize, &[u8])> {
        header[MAGIC.len()..]
            .split(|&b| b == 0)
            .enumerate()
            .filter(|(field, _)| !WRITER.contains(field))
    }
    compared(found).eq(compared(ours))
}

/// A layout file whose header, `found`, is of another schema version or target than
/// this build's: its fields and this build's, for the caller that rebuilds it from
/// its source model to report. The file's bytes are untrusted, so all but printable
/// ASCII are escaped.
fn mismatch(found: &[u8]) -> LoadError {
    let fields = |header: &[u8]| {
        header
            .split(|&b| b == 0)
            .filter(|field| !field.is_empty())
            .map(|field| field.escape_ascii().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    // The header's last whole field; a header of fewer fields runs into the body.
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

    /// A layout file whose header, read first, has this build's schema version and
    /// target. A header that starts as a layout file's but differs in them is a
    /// mismatch, whichever build wrote it.
    fn read_layout<R: Read>(r: &mut Decoder<R>) -> Result<Self, LoadError> {
        let ours = header();
        let mut found = vec![0; MAGIC.len()];
        r.bytes(&mut found).map_err(error)?;
        if found != MAGIC.as_bytes() {
            return Err(LoadError::Layout("not a layout file".into()));
        }
        // Its fields, within twice this build's bytes: the writer's may be longer.
        let mut fields = 0;
        while fields < FIELDS && found.len() < 2 * ours.len() {
            let mut byte = [0];
            r.bytes(&mut byte).map_err(error)?;
            fields += usize::from(byte[0] == 0);
            found.push(byte[0]);
        }
        if fields < FIELDS || !same_format(&found, &ours) {
            return Err(mismatch(&found));
        }
        Self::read_body(r).map_err(error)
    }

    fn read_body<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        let config_hash = u64::get(r)?;
        let threshold_type = crate::model::ThresholdType::get(r)?;
        let mut model = crate::model::Model::get(r)?;
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
        // The file stores no mask plan: validation builds it from what it checked.
        model.masks = validate::model(&model)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

    /// Save this forest's built prediction layout (`layout-file` feature).
    ///
    /// Experimental, versioned and target-specific: the file loads in a build of the
    /// same schema version and target, and is rebuilt from the source model for any
    /// other. Includes the grouping configuration.
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
        self.write_layout(path.as_ref(), &header()).map_err(error)
    }

    fn write_layout(&self, path: &Path, header: &[u8]) -> io::Result<()> {
        let mut w = Encoder::new(BufWriter::new(std::fs::File::create(path)?));
        w.bytes(header)?;
        fingerprint(&self.model.config)?.put(&mut w)?;
        self.model.threshold_type.put(&mut w)?;
        self.model.put(&mut w)?;
        w.inner.write_all(&w.checksum.finish().to_le_bytes())?;
        w.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::testsupport as support;
    use super::*;

    /// This build's header with its fields after the magic changed: `(field, value)`.
    fn header_with(changes: &[(usize, &str)]) -> Vec<u8> {
        let ours = header();
        let mut fields: Vec<&[u8]> = ours[MAGIC.len()..].split(|&b| b == 0).collect();
        for &(field, value) in changes {
            fields[field] = value.as_bytes();
        }
        [MAGIC.as_bytes(), &fields.join(&0)].concat()
    }

    /// Every loader's result for a `forest` saved under `header`.
    fn load(forest: &Forest, header: &[u8]) -> Vec<Result<Forest, LoadError>> {
        let file = support::File::new();
        forest.write_layout(&file.0, header).unwrap();
        let mut loaded = vec![Forest::load_layout_impl(&file.0)];
        // SAFETY: the test owns the file and does not write it while it is mapped;
        // on Unix the mapping outlives the file's name.
        #[cfg(all(unix, target_pointer_width = "64", not(miri)))]
        loaded.push(unsafe { Forest::map_layout_impl(&file.0) });
        loaded
    }

    #[test]
    fn a_file_of_another_build_of_the_schema_and_target_loads() {
        let forest = support::categorical(2);
        let rows = [1.0, 2.0, 0.0, 1.0, f64::NAN, 3.0, 2.0, f64::NAN];
        let mut expected = [0.0f64; 4];
        forest.predictor().predict_group(&rows, &mut expected);
        // Another crate version and build ID, longer than this build's and shorter,
        // which moves the body.
        for writer in [
            [(1, "999.999.999-other"), (2, "build=another")],
            [(1, "0"), (2, "")],
        ] {
            let theirs = header_with(&writer);
            assert_ne!(theirs.len(), header().len());
            for loaded in load(&forest, &theirs) {
                let mut actual = [0.0f64; 4];
                loaded
                    .unwrap()
                    .predictor()
                    .predict_group(&rows, &mut actual);
                assert_eq!(expected.map(f64::to_bits), actual.map(f64::to_bits));
            }
        }
    }

    #[test]
    fn another_schema_version_or_target_is_a_mismatch() {
        let forest = support::numeric(2, true);
        let mismatched = |header: &[u8], theirs: &str| {
            for loaded in load(&forest, header) {
                let error = loaded.unwrap_err();
                let LoadError::LayoutMismatch(message) = &error else {
                    panic!("{error}");
                };
                assert!(message.contains(theirs), "{message}");
            }
        };
        // The schema version, as 3.3.0's files have it, then the target's
        // architecture, pointer width, endianness and link stride.
        for (field, theirs) in [
            (0, "v1"),
            (0, "v0"),
            (3, "other"),
            (4, "0"),
            (5, "neither"),
            (6, "links=0"),
        ] {
            mismatched(&header_with(&[(field, theirs)]), theirs);
        }
        // A field more, a field less, and one that does not end within the reader's
        // bound, of which the message keeps none.
        mismatched(&header_with(&[(2, "build=a\0more")]), "more");
        let ours = header();
        let less = &ours[..ours.len() - 1 - ours.rsplit(|&b| b == 0).nth(1).unwrap().len()];
        mismatched(less, "TWLAYOUT");
        let long = "9".repeat(4 * ours.len());
        for loaded in load(&forest, &header_with(&[(1, &long)])) {
            let error = loaded.unwrap_err();
            assert!(matches!(error, LoadError::LayoutMismatch(_)), "{error}");
            assert!(error.to_string().len() < 4 * ours.len(), "{error}");
        }
    }
}
