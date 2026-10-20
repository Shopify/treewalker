//! Walker configuration: feature classification for partial evaluation, and load
//! options.
//!
//! The training pipeline emits a `walker_config.json` alongside the model that
//! declares which features are constant within an entity, which are varying
//! across rows, and which varying features are monotonic. This allows the walker
//! to be model-agnostic — it works for any grouped prediction task (hazard,
//! ranking, CTR), not just a specific feature set.
//!
//! # Feature sets
//!
//! Each role is the sorted list of the feature indices declared for it, or "every
//! feature" for [`WalkerConfigBuilder::all_varying`]. A lookup is a binary search,
//! and the sets take memory in proportion to the declarations, not to the largest
//! index or to `n_features`. A model has at most `i32::MAX` features, the range of
//! Treelite's `num_feature`.

use crate::LoadError;
use serde::Deserialize;
use std::path::Path;

/// Raw JSON schema from the training pipeline.
/// Only used for deserialization: fields are immediately converted to feature sets.
#[derive(Deserialize)]
struct WalkerConfigFile {
    n_features: usize,
    max_group_width: usize,
    varying_features: Vec<usize>,
    mono_inc_features: Vec<usize>,
    mono_dec_features: Vec<usize>,
}

/// Load-time options: the caps on untrusted input.
///
/// The caps bound what a hostile file can make the loader read, allocate and decode
/// before it is rejected with [`LoadError::Limit`]. Raise them to load larger trusted
/// inputs. Nothing else in the model's representation has a fixed limit: trees,
/// nodes, depth, features, predicates and category IDs are bounded by the input, by
/// memory, and by the widths documented in `docs/treelite-loading.md`. The layout's
/// choices (tree order, interning, prefix groups, predicate sharing) are the
/// library's; the `research` feature can turn each off to measure it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadOptions {
    /// Bytes a binary model may take in all; default 4 GiB.
    pub max_binary_bytes: u64,
    /// Bytes one array field of a binary model may take; default 64 MiB. It also
    /// bounds the trees (4 bytes each in `target_id`) and the nodes and category
    /// entries of each tree.
    pub max_binary_array_bytes: u64,
    /// Bytes of a JSON model; default 64 MiB. Large models should use binary.
    pub max_json_bytes: u64,
    /// Bytes of a model's `attributes` string; default 16 MiB.
    pub max_attributes_bytes: u64,
    /// Nesting of arrays and objects in JSON models, attributes and configurations;
    /// default 64.
    pub max_json_nesting: usize,
    /// Bytes of a configuration file; default 64 KiB.
    pub max_config_bytes: u64,
    /// Bytes of the categorical rank tables built at load; default 256 MiB. A
    /// categorical feature with C categories whose splits' sets need W chains takes
    /// 4 × (W + C + 3) bytes, and 8 more per exception, a code's rank in a chain
    /// where it differs from an unlisted category's: at most one per listed category
    /// (or missing sent left) of each distinct heavy set. A categorical feature the
    /// walker configuration declares varying also takes, for the masks its rows
    /// flip, 4 × (C + 3) bytes and at most 8 per exception in a chain that its splits
    /// test. The cap also bounds, per feature, what is
    /// built on the way: 4 bytes per listed category of each split's heavy side, and
    /// the inclusions between distinct heavy sets, which a small hostile file could
    /// otherwise make quadratic in its splits: 8 bytes per inclusion between two
    /// listed sets or two complements, and a bit for each pair of a listed set and a
    /// complement.
    pub max_rank_table_bytes: u64,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            max_binary_bytes: 4 << 30,
            max_binary_array_bytes: 64 << 20,
            max_json_bytes: 64 << 20,
            max_attributes_bytes: 16 << 20,
            max_json_nesting: 64,
            max_config_bytes: 64 << 10,
            max_rank_table_bytes: 256 << 20,
        }
    }
}

/// The layout choices a load makes, all on in production; the research module's
/// `LoadAblation` turns each off to measure it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Choices {
    /// Sort trees by their heavy paths.
    pub(crate) tree_ordering: bool,
    /// Pool identical category lists once.
    pub(crate) bitset_intern: bool,
    /// How prefix groups are formed.
    pub(crate) prefix: crate::rules::Prefix,
    /// Share one predicate between identical varying splits.
    pub(crate) predicate_dedup: bool,
}

impl Default for Choices {
    fn default() -> Self {
        Self {
            tree_ordering: true,
            bitset_intern: true,
            prefix: crate::rules::Prefix::Rule,
            predicate_dedup: true,
        }
    }
}

/// A set of feature indices: every feature, or `indices`, sorted and distinct, each
/// below the set's `n_features`. Canonical, so equal sets compare equal.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FeatureSet {
    all: bool,
    /// At most `i32::MAX` features, so each index fits.
    indices: Vec<u32>,
}

impl FeatureSet {
    const ALL: Self = Self {
        all: true,
        indices: Vec::new(),
    };

    /// Whether `feature` is in the set, for a feature below the set's `n_features`:
    /// a binary search of the indices.
    #[inline]
    const fn contains(&self, feature: usize) -> bool {
        if self.all {
            return true;
        }
        let indices = self.indices.as_slice();
        let (mut lo, mut hi) = (0, indices.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let at = indices[mid] as usize;
            if at == feature {
                return true;
            }
            if at < feature {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        false
    }

    /// The set of `indices`, each below `n_features` and listed once.
    fn of(indices: &[usize], n_features: usize, label: &str) -> Result<Self, LoadError> {
        let mut sorted = Vec::with_capacity(indices.len());
        for &i in indices {
            if i >= n_features {
                return Err(LoadError::MalformedConfig(format!(
                    "{label}: index {i} >= n_features {n_features}"
                )));
            }
            sorted.push(i as u32);
        }
        sorted.sort_unstable();
        let duplicate = sorted
            .windows(2)
            .find_map(|w| (w[0] == w[1]).then_some(w[0]));
        if let Some(i) = duplicate {
            return Err(LoadError::MalformedConfig(format!(
                "{label}: duplicate index {i}"
            )));
        }
        Ok(Self {
            all: false,
            indices: sorted,
        }
        .canonical(n_features))
    }

    /// The union of two sets over `n_features`: a merge of their indices.
    fn union(&self, other: &Self, n_features: usize) -> Self {
        if self.all || other.all {
            return Self::ALL;
        }
        let (ours, theirs) = (self.indices.as_slice(), other.indices.as_slice());
        let mut indices = Vec::with_capacity(ours.len() + theirs.len());
        let (mut i, mut j) = (0, 0);
        while i < ours.len() && j < theirs.len() {
            let (x, y) = (ours[i], theirs[j]);
            indices.push(x.min(y));
            i += usize::from(x <= y);
            j += usize::from(y <= x);
        }
        indices.extend_from_slice(&ours[i..]);
        indices.extend_from_slice(&theirs[j..]);
        Self {
            all: false,
            indices,
        }
        .canonical(n_features)
    }

    /// The lowest index in both sets, if any: a merge of their indices up to the
    /// first they share.
    fn first_common(&self, other: &Self, n_features: usize) -> Option<usize> {
        match (self.all, other.all) {
            (true, true) => Some(0),
            (true, false) => other.first(),
            (false, true) => self.first(),
            (false, false) => {
                let (ours, theirs) = (self.indices.as_slice(), other.indices.as_slice());
                let (mut i, mut j) = (0, 0);
                while i < ours.len() && j < theirs.len() && ours[i] != theirs[j] {
                    let (x, y) = (ours[i], theirs[j]);
                    i += usize::from(x < y);
                    j += usize::from(y < x);
                }
                (i < ours.len() && j < theirs.len()).then(|| ours[i] as usize)
            }
        }
        .filter(|&f| f < n_features)
    }

    fn first(&self) -> Option<usize> {
        self.indices.first().map(|&f| f as usize)
    }

    /// Every feature as `ALL`.
    fn canonical(self, n_features: usize) -> Self {
        if self.all || self.indices.len() == n_features {
            return Self::ALL;
        }
        self
    }
}

/// Feature classification for partial evaluation.
///
/// A feature is exactly one of:
/// - **Constant**: identical across all rows of a group. Its splits are evaluated
///   once per tree (the constant walk).
/// - **Varying, monotonic increasing**: values ascend across the group's rows.
/// - **Varying, monotonic decreasing**: values descend.
/// - **Varying, non-monotonic**: no order.
///
/// Prediction sorts each varying feature's rows to build its splits' row masks,
/// whatever the declared order; rows that arrive sorted take the sort one pass. Only
/// the research API's event scans (`disable_varying_precompute`) use the declared
/// order, to stop comparing at the first row past a threshold.
///
/// Build one with [`WalkerConfig::builder`], or read the training pipeline's
/// `walker_config.json` with [`WalkerConfig::from_file`]. The equality of each
/// group's constant values is a caller contract, not checked; production's outputs
/// do not depend on monotonicity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalkerConfig {
    n_features: usize,
    max_group_width: usize,
    /// The features that vary across rows, monotonic or not.
    varying: FeatureSet,
    /// The features that increase across a group's rows.
    increasing: FeatureSet,
    /// The features that decrease across a group's rows.
    decreasing: FeatureSet,
}

impl WalkerConfig {
    /// Start a configuration for `n_features` features, 1 to `i32::MAX`, in the
    /// model's trained column order.
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

    /// Read and validate the five-field grouping JSON schema from a file, within the
    /// default [`LoadOptions`] caps.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        Self::from_file_with(path, &LoadOptions::default())
    }

    /// As [`Self::from_file`], within the caps of `options`:
    /// [`LoadOptions::max_config_bytes`] and [`LoadOptions::max_json_nesting`].
    pub fn from_file_with(
        path: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<Self, LoadError> {
        use std::io::Read;
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(options.max_config_bytes.saturating_add(1))
            .read_to_end(&mut bytes)?;
        Self::from_json_with(&bytes, options)
    }

    /// Parse and validate grouping JSON, bounded to 64 KiB: `n_features`,
    /// `max_group_width`, `varying_features`, `mono_inc_features` and
    /// `mono_dec_features`, validated as the builder validates them.
    pub fn from_json(bytes: &[u8]) -> Result<Self, LoadError> {
        Self::from_json_with(bytes, &LoadOptions::default())
    }

    /// As [`Self::from_json`], within the caps of `options`:
    /// [`LoadOptions::max_config_bytes`] and [`LoadOptions::max_json_nesting`].
    pub fn from_json_with(bytes: &[u8], options: &LoadOptions) -> Result<Self, LoadError> {
        if bytes.len() as u64 > options.max_config_bytes {
            return Err(LoadError::Limit(format!(
                "configuration exceeds {} bytes (LoadOptions::max_config_bytes)",
                options.max_config_bytes
            )));
        }
        crate::load::parse::validate_json_depth(bytes, options.max_json_nesting)?;
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

    /// Maximum rows per group. A predictor's scratch grows linearly with it (see
    /// [`crate::Forest::predictor`]), so it bounds memory, not only the calls.
    #[inline]
    #[must_use]
    pub const fn max_group_width(&self) -> usize {
        self.max_group_width
    }

    /// Whether `feature` varies across a group's rows (monotonic or not).
    #[inline]
    #[must_use]
    pub const fn is_varying(&self, feature: usize) -> bool {
        feature < self.n_features && self.varying.contains(feature)
    }

    /// Whether `feature` ascends across a group's rows.
    #[inline]
    #[must_use]
    pub const fn is_increasing(&self, feature: usize) -> bool {
        feature < self.n_features && self.increasing.contains(feature)
    }

    /// Whether `feature` descends across a group's rows.
    #[inline]
    #[must_use]
    pub const fn is_decreasing(&self, feature: usize) -> bool {
        feature < self.n_features && self.decreasing.contains(feature)
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
    /// The widest group a predictor will accept, at least 1. Each predictor's
    /// scratch grows linearly with it; see [`crate::Forest::predictor`].
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
    /// Fails with [`LoadError::MalformedConfig`] if `n_features` is not 1 to
    /// `i32::MAX` (Treelite's `num_feature` is a 32-bit signed integer),
    /// `max_group_width` is missing or 0, the varying features were not declared, an
    /// index is out of range or listed twice, or a feature is both increasing and
    /// decreasing.
    pub fn build(self) -> Result<WalkerConfig, LoadError> {
        let n_features = self.n_features;
        if !(1..=i32::MAX as usize).contains(&n_features) {
            return Err(LoadError::MalformedConfig(format!(
                "n_features must be 1..={}, got {n_features}",
                i32::MAX
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
        let varying = match &self.varying {
            Some(Varying::All) => FeatureSet::ALL,
            Some(Varying::Only(features)) => {
                FeatureSet::of(features, n_features, "varying features")?
            }
            None => {
                return Err(LoadError::MalformedConfig(
                    "declare the varying features with varying(..) or all_varying(); \
                     undeclared features are constant"
                        .into(),
                ));
            }
        };
        let increasing = FeatureSet::of(&self.increasing, n_features, "increasing features")?;
        let decreasing = FeatureSet::of(&self.decreasing, n_features, "decreasing features")?;
        if let Some(f) = increasing.first_common(&decreasing, n_features) {
            return Err(LoadError::MalformedConfig(format!(
                "feature {f} is both increasing and decreasing"
            )));
        }
        Ok(WalkerConfig {
            n_features,
            max_group_width,
            varying: varying
                .union(&increasing, n_features)
                .union(&decreasing, n_features),
            increasing,
            decreasing,
        })
    }
}
