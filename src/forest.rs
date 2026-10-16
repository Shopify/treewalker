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
    /// Scaffolding until `src/legacy/` is deleted: today's model of the same input,
    /// which nothing reads.
    pub(crate) legacy: Option<Arc<crate::legacy::forest::Model>>,
}

impl std::fmt::Debug for Forest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = &*self.model;
        f.debug_struct("Forest")
            .field("trees", &m.trees.len())
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
        let config = WalkerConfig::from_file_with(config_path, options)?;
        let format = ModelFormat::from_path(model_path.as_ref())?;
        Self::from_file_with(
            std::fs::File::open(model_path)?,
            format,
            config,
            options,
            &Choices::default(),
        )
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
        // Scaffolding until `src/legacy/` is deleted: today's loader reads the same
        // input, so the input is read first, to its cap.
        let cap = match format {
            ModelFormat::TreeliteBinaryV4 => options.max_binary_bytes,
            ModelFormat::TreeliteJson => options.max_json_bytes,
        };
        let mut bytes = Vec::new();
        reader.take(cap.saturating_add(1)).read_to_end(&mut bytes)?;
        Self::from_bytes_with(&bytes, format, config, options, choices)
    }

    /// [`Self::from_reader_with`] of a model file, whose binary decoding counts the
    /// nodes first.
    pub(crate) fn from_file_with(
        file: std::fs::File,
        format: ModelFormat,
        config: WalkerConfig,
        options: &LoadOptions,
        choices: &Choices,
    ) -> Result<Self, LoadError> {
        // Scaffolding until `src/legacy/` is deleted.
        Self::from_reader_with(file, format, config, options, choices)
    }

    /// [`Self::from_bytes`] with the layout's `choices`.
    pub(crate) fn from_bytes_with(
        bytes: &[u8],
        format: ModelFormat,
        config: WalkerConfig,
        options: &LoadOptions,
        choices: &Choices,
    ) -> Result<Self, LoadError> {
        // Scaffolding until `src/legacy/` is deleted: today's model of the same bytes,
        // loaded first, so that a load fails where today's loader fails.
        let legacy = Self::legacy_model(bytes, format, &config, choices)?;
        let parsed = parse::parse_bytes(bytes, format, &config, options, choices)?;
        let mut forest = Self::build(parsed, config, options, choices)?;
        forest.legacy = legacy;
        Ok(forest)
    }

    /// Scaffolding until `src/legacy/` is deleted: today's model of `bytes`, with the
    /// layout `choices`. The new loader's unit tests load models that today's loader
    /// rejects, so there it is `None` instead of an error.
    #[cfg_attr(
        test,
        expect(
            clippy::unnecessary_wraps,
            reason = "fails only outside the unit tests"
        )
    )]
    fn legacy_model(
        bytes: &[u8],
        format: ModelFormat,
        config: &WalkerConfig,
        choices: &Choices,
    ) -> Result<Option<Arc<crate::legacy::forest::Model>>, LoadError> {
        let options = crate::legacy::Options::of(choices);
        let legacy = crate::legacy::forest::load(bytes, format, config.clone(), &options);
        #[cfg(test)]
        return Ok(legacy.ok());
        #[cfg(not(test))]
        legacy.map(Some)
    }

    /// The feature classification the model was loaded with.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &WalkerConfig {
        &self.model.config
    }
}
