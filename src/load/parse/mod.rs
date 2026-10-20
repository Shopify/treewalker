//! Decoding a Treelite model: the binary checkpoint ([`binary`]) or the JSON dump
//! ([`json`]) into the decoded forest ([`decoded`]), each tree validated and put in
//! heavy-first preorder as it is read ([`tree`]), with the model-level checks and
//! the input caps ([`validation`]).
//!
//! # Threshold semantics
//!
//! Treelite normalizes each framework's model (`treelite.frontend.load_lightgbm_model`
//! or `load_xgboost_model`, then `serialize_bytes()` or `dump_as_json()`) but keeps
//! its comparison: LightGBM splits on `val <= threshold` in f64, XGBoost on
//! `val < threshold` in f32. The decoder reads the threshold type from the header:
//!
//! - F64: a `<=` threshold is kept, and a `<` threshold becomes `t.next_down()`, so
//!   that every F64 split compares `<=`.
//! - F32: thresholds are kept, and splits compare `(val as f32) < (t as f32)`.

mod binary;
pub(crate) mod decoded;
mod json;
pub(crate) mod tree;
mod validation;
use crate::LoadError;
use std::io::Read;
pub(crate) use validation::{ParsedModel, validate_json_depth};

use std::path::Path;

use crate::config::{Choices, LoadOptions, WalkerConfig};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Explicit format for loading models from memory or readers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelFormat {
    /// Treelite 4.0 to 4.7 binary checkpoint, from `Model.serialize_bytes()`.
    TreeliteBinaryV4,
    /// Treelite JSON dump, from `Model.dump_as_json()`.
    TreeliteJson,
}

impl ModelFormat {
    pub(crate) fn from_path(path: &Path) -> Result<Self, LoadError> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("bin") => Ok(Self::TreeliteBinaryV4),
            Some("json") => Ok(Self::TreeliteJson),
            _ => Err(LoadError::Unsupported(format!(
                "model path {} must end in .bin or .json; use Forest::from_reader with an explicit ModelFormat",
                path.display()
            ))),
        }
    }
}

pub(crate) fn parse_reader(
    reader: impl Read,
    format: ModelFormat,
    config: &WalkerConfig,
    options: &LoadOptions,
    choices: &Choices,
) -> Result<ParsedModel, LoadError> {
    match format {
        ModelFormat::TreeliteBinaryV4 => binary::parse(reader, None, config, options, choices),
        ModelFormat::TreeliteJson => json::parse(reader, config, options, choices),
    }
}

/// [`parse_reader`] of a model file. Binary decoding first counts the nodes from the
/// trees' headers with positioned reads (6 per tree), so that they are allocated
/// once; a reader of unknown length cannot be read twice.
pub(crate) fn parse_file(
    file: std::fs::File,
    format: ModelFormat,
    config: &WalkerConfig,
    options: &LoadOptions,
    choices: &Choices,
) -> Result<ParsedModel, LoadError> {
    match format {
        ModelFormat::TreeliteBinaryV4 => {
            #[cfg(unix)]
            let nodes = binary::count_nodes(&file, options.max_binary_bytes);
            #[cfg(not(unix))]
            let nodes = None;
            binary::parse(file, nodes, config, options, choices)
        }
        ModelFormat::TreeliteJson => json::parse(file, config, options, choices),
    }
}

/// [`parse_reader`] of a model in memory. Binary decoding reads the fields in place;
/// JSON decoding parses a copy, which its parser rewrites.
pub(crate) fn parse_bytes(
    bytes: &[u8],
    format: ModelFormat,
    config: &WalkerConfig,
    options: &LoadOptions,
    choices: &Choices,
) -> Result<ParsedModel, LoadError> {
    match format {
        ModelFormat::TreeliteBinaryV4 => binary::parse_bytes(bytes, config, options, choices),
        ModelFormat::TreeliteJson => json::parse(bytes, config, options, choices),
    }
}
