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
    /// Scaffolding while `src/legacy/` serves the Predictor and the research API:
    /// today's model of the same input, which today's kernel predicts from.
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
        // Scaffolding while `src/legacy/` serves the Predictor and the research API:
        // today's loader reads the same input, so the input is read first, to its cap.
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
        // Scaffolding while `src/legacy/` serves the Predictor and the research API.
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
        // Scaffolding while `src/legacy/` serves the Predictor and the research API:
        // today's model of the same bytes, loaded first, so that a load fails where
        // today's loader fails.
        let legacy = Self::legacy_model(bytes, format, &config, options)?;
        let parsed = parse::parse_bytes(bytes, format, &config, options, choices)?;
        let mut forest = Self::build(parsed, config, options, choices)?;
        forest.legacy = legacy;
        Ok(forest)
    }

    /// Scaffolding while `src/legacy/` serves the Predictor and the research API:
    /// today's model of `bytes`, with the layout switches of `options`. The new
    /// loader's unit tests load models that today's loader rejects, so there it is
    /// `None` instead of an error.
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
        options: &LoadOptions,
    ) -> Result<Option<Arc<crate::legacy::forest::Model>>, LoadError> {
        let legacy = crate::legacy::forest::load(bytes, format, config.clone(), options);
        #[cfg(test)]
        return Ok(legacy.ok());
        #[cfg(not(test))]
        legacy.map(Some)
    }

    /// Scaffolding while `src/legacy/` serves the Predictor and the research API:
    /// today's model.
    pub(crate) const fn legacy(&self) -> &Arc<crate::legacy::forest::Model> {
        self.legacy.as_ref().expect("loaded with today's model")
    }

    /// The feature classification the model was loaded with.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &WalkerConfig {
        &self.model.config
    }

    /// Whether prediction sums leaf values exactly and rounds once, so each row's tree
    /// sum is the `f64` nearest to the true sum whatever the tree order. Averaging, the
    /// base score and the link function are applied afterwards in ordinary `f64`.
    /// False if the model was loaded with [`LoadOptions::exact_sums`] off, or if a leaf
    /// is not finite or the leaf exponents span too wide a range for a 126-bit fixed
    /// point; prediction then adds each row's leaves in `f64` in tree order, as the
    /// full walk does.
    // Scaffolding while `src/legacy/` serves the research API: its predictors' sums.
    // The Predictor sums every model exactly.
    #[inline]
    #[must_use]
    pub fn exact_sums(&self) -> bool {
        self.legacy().fixed_scale.is_some()
    }
}
