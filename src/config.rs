//! Walker configuration: feature classification for partial evaluation, and load
//! options.
//!
//! The training pipeline emits a `walker_config.json` alongside the model that
//! declares which features are constant within an entity, which are varying
//! across rows, and which varying features are monotonic. This allows the walker
//! to be model-agnostic — it works for any grouped prediction task (hazard,
//! ranking, CTR), not just a specific feature set.
//!
//! # Why bitmasks?
//!
//! Feature classification is stored as u128 bitmasks for O(1) lookup on the hot path.
//! A single `AND` + compare replaces what would otherwise be an array lookup or hash
//! check on every split node. Supports up to 64 features (validated at load time).

use crate::LoadError;
use serde::Deserialize;
use std::path::Path;

/// Raw JSON schema from the training pipeline.
/// Only used for deserialization — fields are immediately converted to bitmasks.
#[derive(Deserialize)]
struct WalkerConfigFile {
    n_features: usize,
    max_group_width: usize,
    varying_features: Vec<usize>,
    mono_inc_features: Vec<usize>,
    mono_dec_features: Vec<usize>,
}

/// Load-time options: optimizations baked into the model's layout.
///
/// Every option defaults to the optimized layout. The `disable_*` options and
/// `prefix_depth = 0` exist to measure what each optimization contributes; changing
/// one means loading the model again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadOptions {
    /// Keep the trees in the model's order instead of sorting trees with similar
    /// early splits next to each other.
    pub disable_tree_ordering: bool,
    /// Store every categorical bitset, even when an identical one is already pooled.
    pub disable_bitset_intern: bool,
    /// Constant-prefix depth K: trees sharing their first K heavy-path constant splits
    /// evaluate them once per group instead of once per tree. 0 disables it; the
    /// default is 2.
    pub prefix_depth: usize,
    /// Give every varying node its own predicate instead of sharing one between
    /// identical splits. This multiplies the precompute work, and large models can
    /// exceed the 65,535 predicate limit.
    pub disable_predicate_dedup: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            disable_tree_ordering: false,
            disable_bitset_intern: false,
            prefix_depth: 2,
            disable_predicate_dedup: false,
        }
    }
}

/// Feature classification for partial evaluation.
///
/// A feature is exactly one of:
/// - **Constant**: identical across all rows of a group. Its splits are evaluated
///   once per tree (the constant walk).
/// - **Varying, monotonic increasing**: values ascend across the group's rows, so a
///   split partitions them with a prefix scan.
/// - **Varying, monotonic decreasing**: values descend; a suffix scan.
/// - **Varying, non-monotonic**: each row is evaluated on its own.
///
/// Build one with [`WalkerConfig::builder`], or read the training pipeline's
/// `walker_config.json` with [`WalkerConfig::from_file`]. The equality and
/// monotonicity of each group's values are caller contracts, not checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalkerConfig {
    n_features: usize,
    max_group_width: usize,
    /// Bit `i` is set if feature `i` is varying across rows.
    varying_mask: u128,
    /// Bit `i` is set if feature `i` is monotonic increasing across the group.
    mono_inc_mask: u128,
    /// Bit `i` is set if feature `i` is monotonic decreasing across the group.
    mono_dec_mask: u128,
}

impl WalkerConfig {
    /// Start a configuration for `n_features` features, 1 to 64, in the model's
    /// trained column order.
    ///
    /// ```
    /// use treewalker_gbdt::WalkerConfig;
    ///
    /// // Features 0-2 are constant within a group; 3 and 4 vary, 3 ascending.
    /// let config = WalkerConfig::builder(5)
    ///     .max_group_width(128)
    ///     .varying([4])
    ///     .increasing([3])
    ///     .build()?;
    /// assert!(config.is_varying(3) && config.is_increasing(3) && !config.is_varying(0));
    /// # Ok::<(), treewalker_gbdt::LoadError>(())
    /// ```
    pub const fn builder(n_features: usize) -> WalkerConfigBuilder {
        WalkerConfigBuilder {
            n_features,
            max_group_width: None,
            varying: None,
            increasing: Vec::new(),
            decreasing: Vec::new(),
        }
    }

    /// Read and validate the five-field grouping JSON schema from a file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        use std::io::Read;
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(65_537)
            .read_to_end(&mut bytes)?;
        Self::from_json(&bytes)
    }

    /// Parse and validate grouping JSON, bounded to 64 KiB: `n_features`,
    /// `max_group_width`, `varying_features`, `mono_inc_features` and
    /// `mono_dec_features`, validated as the builder validates them.
    pub fn from_json(bytes: &[u8]) -> Result<Self, LoadError> {
        if bytes.len() > 65_536 {
            return Err(LoadError::Limit("configuration exceeds 64 KiB".into()));
        }
        crate::parser::validate_json_depth(bytes)
            .map_err(|e| LoadError::MalformedConfig(e.to_string()))?;
        let mut data = bytes.to_vec();
        let f: WalkerConfigFile = simd_json::serde::from_slice(&mut data)
            .map_err(|e| LoadError::MalformedConfig(e.to_string()))?;
        Self::builder(f.n_features)
            .max_group_width(f.max_group_width)
            .varying(f.varying_features)
            .increasing(f.mono_inc_features)
            .decreasing(f.mono_dec_features)
            .build()
    }

    /// Number of input features per row.
    #[inline]
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Maximum rows per group. It selects the row-mask width; groups wider than
    /// 1,024 rows run in pieces of 1,024.
    #[inline]
    #[must_use]
    pub const fn max_group_width(&self) -> usize {
        self.max_group_width
    }

    /// Whether `feature` varies across a group's rows (monotonic or not).
    #[inline]
    #[must_use]
    pub const fn is_varying(&self, feature: usize) -> bool {
        feature < self.n_features && self.varying_mask & (1u128 << feature) != 0
    }

    /// Whether `feature` ascends across a group's rows.
    #[inline]
    #[must_use]
    pub const fn is_increasing(&self, feature: usize) -> bool {
        feature < self.n_features && self.mono_inc_mask & (1u128 << feature) != 0
    }

    /// Whether `feature` descends across a group's rows.
    #[inline]
    #[must_use]
    pub const fn is_decreasing(&self, feature: usize) -> bool {
        feature < self.n_features && self.mono_dec_mask & (1u128 << feature) != 0
    }
}

/// Builds a [`WalkerConfig`]; see [`WalkerConfig::builder`].
///
/// [`Self::max_group_width`] and one of [`Self::varying`] or [`Self::all_varying`]
/// are required: a feature not declared varying is constant, so a forgotten
/// declaration would silently give wrong predictions for an ordinary matrix.
/// [`Self::increasing`] and [`Self::decreasing`] imply varying. Each setter replaces
/// what an earlier call to it set.
#[derive(Clone, Debug)]
#[must_use]
pub struct WalkerConfigBuilder {
    n_features: usize,
    max_group_width: Option<usize>,
    varying: Option<Varying>,
    increasing: Vec<usize>,
    decreasing: Vec<usize>,
}

#[derive(Clone, Debug)]
enum Varying {
    All,
    Only(Vec<usize>),
}

impl WalkerConfigBuilder {
    /// The widest group a predictor will accept, at least 1.
    pub const fn max_group_width(mut self, rows: usize) -> Self {
        self.max_group_width = Some(rows);
        self
    }

    /// The features that vary across a group's rows; every other feature, unless
    /// declared monotonic, is constant. An empty list declares every feature constant.
    pub fn varying(mut self, features: impl IntoIterator<Item = usize>) -> Self {
        self.varying = Some(Varying::Only(features.into_iter().collect()));
        self
    }

    /// Declare every feature varying: rows of a group may differ in any column.
    pub fn all_varying(mut self) -> Self {
        self.varying = Some(Varying::All);
        self
    }

    /// The features that ascend across a group's rows. Implies varying.
    pub fn increasing(mut self, features: impl IntoIterator<Item = usize>) -> Self {
        self.increasing = features.into_iter().collect();
        self
    }

    /// The features that descend across a group's rows. Implies varying.
    pub fn decreasing(mut self, features: impl IntoIterator<Item = usize>) -> Self {
        self.decreasing = features.into_iter().collect();
        self
    }

    /// Validate the declarations.
    ///
    /// Fails with [`LoadError::MalformedConfig`] if `n_features` is not 1 to 64,
    /// `max_group_width` is missing or 0, the varying features were not declared, an
    /// index is out of range or listed twice, or a feature is both increasing and
    /// decreasing.
    pub fn build(self) -> Result<WalkerConfig, LoadError> {
        let n_features = self.n_features;
        if !(1..=64).contains(&n_features) {
            return Err(LoadError::MalformedConfig(format!(
                "n_features must be 1..=64, got {n_features}"
            )));
        }
        let max_group_width = match self.max_group_width {
            Some(0) => {
                return Err(LoadError::MalformedConfig(
                    "max_group_width must be at least 1".into(),
                ));
            }
            Some(rows) => rows,
            None => {
                return Err(LoadError::MalformedConfig(
                    "max_group_width is required".into(),
                ));
            }
        };
        let to_mask = |indices: &[usize], label: &str| -> Result<u128, LoadError> {
            let mut mask = 0;
            for &i in indices {
                if i >= n_features {
                    return Err(LoadError::MalformedConfig(format!(
                        "{label}: index {i} >= n_features {n_features}"
                    )));
                }
                let bit = 1u128 << i;
                if mask & bit != 0 {
                    return Err(LoadError::MalformedConfig(format!(
                        "{label}: duplicate index {i}"
                    )));
                }
                mask |= bit;
            }
            Ok(mask)
        };
        let varying_mask = match &self.varying {
            Some(Varying::All) => (1u128 << n_features) - 1,
            Some(Varying::Only(features)) => to_mask(features, "varying features")?,
            None => {
                return Err(LoadError::MalformedConfig(
                    "declare the varying features with varying(..) or all_varying(); \
                     undeclared features are constant"
                        .into(),
                ));
            }
        };
        let mono_inc_mask = to_mask(&self.increasing, "increasing features")?;
        let mono_dec_mask = to_mask(&self.decreasing, "decreasing features")?;
        let both = mono_inc_mask & mono_dec_mask;
        if both != 0 {
            return Err(LoadError::MalformedConfig(format!(
                "feature {} is both increasing and decreasing",
                both.trailing_zeros()
            )));
        }
        Ok(WalkerConfig {
            n_features,
            max_group_width,
            varying_mask: varying_mask | mono_inc_mask | mono_dec_mask,
            mono_inc_mask,
            mono_dec_mask,
        })
    }
}
