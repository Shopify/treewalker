//! The public handle to a loaded model, [`Forest`], and its load entry points.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use crate::config::{Choices, LoadOptions, WalkerConfig};
use crate::load::parse;
use crate::model::Model;
use crate::{LoadError, ModelFormat};

/// A loaded model: a cheap, immutable handle that can be cloned and shared
/// across threads.
///
/// Predictions go through a [`Predictor`](crate::Predictor), which owns the
/// scratch space; create one per worker with [`Forest::predictor`].
#[derive(Clone)]
pub struct Forest {
    pub(crate) model: Arc<Model>,
}

impl std::fmt::Debug for Forest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = &*self.model;
        f.debug_struct("Forest")
            .field("trees", &m.tree_starts.len())
            .field("nodes", &(m.layout.nodes.len() - 1))
            .field("category_list_bytes", &(4 * m.categories.len()))
            .field("varying_predicates", &m.varying_predicates.len())
            .field("threshold_type", &m.threshold_type)
            .finish_non_exhaustive()
    }
}

impl Forest {
    /// Load a supported Treelite `.bin` or `.json` model and its grouping
    /// configuration file, with the default [`LoadOptions`].
    ///
    /// The model format follows the file extension. Unsupported model semantics, such
    /// as multiclass output, return [`LoadError::Unsupported`].
    pub fn load(
        model_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
    ) -> Result<Self, LoadError> {
        Self::load_with(model_path, config_path, &LoadOptions::default())
    }

    /// As [`Self::load`], with explicit load options.
    pub fn load_with(
        model_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<Self, LoadError> {
        Self::load_choosing(
            model_path.as_ref(),
            config_path.as_ref(),
            options,
            &Choices::default(),
        )
    }

    /// [`Self::load_with`] with the layout's `choices`. Binary decoding counts the
    /// model file's nodes first.
    pub(crate) fn load_choosing(
        model_path: &Path,
        config_path: &Path,
        options: &LoadOptions,
        choices: &Choices,
    ) -> Result<Self, LoadError> {
        let config = WalkerConfig::from_file_with(config_path, options)?;
        let format = ModelFormat::from_path(model_path)?;
        let file = std::fs::File::open(model_path)?;
        let parsed = parse::parse_file(file, format, &config, options, choices)?;
        Self::build(parsed, config, options, choices)
    }

    /// Load a model from memory with explicit format and validated grouping metadata.
    pub fn from_bytes(
        bytes: &[u8],
        format: ModelFormat,
        config: WalkerConfig,
        options: &LoadOptions,
    ) -> Result<Self, LoadError> {
        Self::from_bytes_with(bytes, format, config, options, &Choices::default())
    }

    /// Load a model from a reader. Binary decoding streams with reusable tree buffers;
    /// JSON decoding materializes a bounded document. Structural validation precedes
    /// optimized layout, and unsupported prediction semantics return an error.
    ///
    /// A binary reader's node count is not known before its last tree, so its nodes
    /// are decoded in segments and copied into place once, which holds them twice
    /// for a moment; [`Self::load`] and [`Self::from_bytes`] count them first and
    /// decode them in place.
    pub fn from_reader(
        reader: impl Read,
        format: ModelFormat,
        config: WalkerConfig,
        options: &LoadOptions,
    ) -> Result<Self, LoadError> {
        Self::from_reader_with(reader, format, config, options, &Choices::default())
    }

    /// [`Self::from_reader`] with the layout's `choices`.
    pub(crate) fn from_reader_with(
        reader: impl Read,
        format: ModelFormat,
        config: WalkerConfig,
        options: &LoadOptions,
        choices: &Choices,
    ) -> Result<Self, LoadError> {
        let parsed = parse::parse_reader(reader, format, &config, options, choices)?;
        Self::build(parsed, config, options, choices)
    }

    /// [`Self::from_bytes`] with the layout's `choices`.
    pub(crate) fn from_bytes_with(
        bytes: &[u8],
        format: ModelFormat,
        config: WalkerConfig,
        options: &LoadOptions,
        choices: &Choices,
    ) -> Result<Self, LoadError> {
        let parsed = parse::parse_bytes(bytes, format, &config, options, choices)?;
        Self::build(parsed, config, options, choices)
    }

    /// The feature classification the model was loaded with.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &WalkerConfig {
        &self.model.config
    }
}
