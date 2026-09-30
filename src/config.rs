//! Walker configuration: feature classification for partial evaluation.
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
//! check on every split node. Supports up to 64 features (enforced by assert at load time).

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

/// Ablation flags — disable individual optimizations for controlled experiments.
///
/// Used by the paper's ablation study to measure each trick's contribution.
/// Each flag becomes a const generic on `partial_eval` — the compiler monomorphizes
/// a separate version per combination, eliminating all runtime ablation branches.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct AblationMode {
    /// Treat all monotonic varying features as non-monotonic (disable early-break scans).
    /// Only effective when `disable_varying_precompute` is also set — the precompute path
    /// evaluates all predicates identically regardless of monotonicity.
    pub disable_monotonic: bool,
    /// Always recurse both children at varying splits (disable unsplit shortcut).
    pub disable_unsplit: bool,
    /// Fall back to per-row partition functions instead of precomputed varying masks.
    pub disable_varying_precompute: bool,
    /// Use brute-force O(P×n) precompute instead of the sorted-threshold sweep.
    /// Only effective when precompute is enabled (the default).
    pub disable_predicate_sweep: bool,
}

impl AblationMode {
    /// True when no ablation flags are set — selects the fast path.
    #[inline]
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Parse-time configuration — controls optimizations baked into the data layout.
///
/// These require a different `Forest` instance to ablate (model must be re-parsed).
#[derive(Clone, Copy)]
pub struct ParseConfig {
    /// Skip `auto_order_trees` — keep trees in LightGBM's original order.
    pub disable_tree_ordering: bool,
    /// Skip bitset interning — always append to pool, no deduplication.
    pub disable_bitset_intern: bool,
    /// Constant-prefix trie depth. Trees sharing the first K heavy-path constant
    /// splits are grouped so those splits are evaluated once per observation instead
    /// of once per tree. 0 = disabled. Default = 2.
    pub prefix_depth: usize,
    /// Skip predicate deduplication — assign a unique `varying_pred_id` to every
    /// varying node instead of sharing IDs for identical predicates. Inflates the
    /// predicates vec from ~5K to ~50K, making precompute proportionally slower.
    pub disable_predicate_dedup: bool,
}

impl Default for ParseConfig {
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
/// - **Constant**: not in `varying_mask`. Identical across all rows of an entity.
///   Splits evaluated once per tree (the constant walk).
/// - **Varying, monotonic increasing**: in both `varying_mask` and `mono_inc_mask`.
///   Values sorted ascending across the panel. Partition via prefix scan.
/// - **Varying, monotonic decreasing**: in both `varying_mask` and `mono_dec_mask`.
///   Values sorted descending. Partition via suffix scan.
/// - **Varying, non-monotonic**: in `varying_mask` but neither mono mask.
///   Each row evaluated independently (most expensive case, but rare).
pub struct WalkerConfig {
    pub n_features: usize,
    /// Maximum rows per group. Hard limit 128 (u128 bitmask).
    /// For fixed-width groups this equals the group width.
    /// For variable-width groups, this is the upper bound.
    pub max_group_width: usize,
    /// Bit `i` is set if feature `i` is varying across rows.
    pub varying_mask: u128,
    /// Bit `i` is set if feature `i` is monotonic increasing across the panel.
    pub mono_inc_mask: u128,
    /// Bit `i` is set if feature `i` is monotonic decreasing across the panel.
    pub mono_dec_mask: u128,
    /// Runtime ablation flags for controlled experiments.
    pub ablation: AblationMode,
}

impl WalkerConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Self {
        let mut data = std::fs::read(path).expect("failed to read walker config");
        let f: WalkerConfigFile =
            simd_json::serde::from_slice(&mut data).expect("failed to parse walker config");

        // Hard limits: feature bitmask width (u128) and row mask width (u128).
        assert!(
            f.n_features <= 128,
            "treewalker supports at most 128 features (got {})",
            f.n_features,
        );
        assert!(
            f.max_group_width <= 128,
            "treewalker supports at most 128 rows per entity (got {})",
            f.max_group_width,
        );

        // Convert feature index lists to bitmasks. Each feature index becomes
        // a set bit in the corresponding u128 mask.
        let to_mask = |indices: &[usize], label: &str| -> u128 {
            indices.iter().fold(0u128, |m, &i| {
                assert!(
                    i < f.n_features,
                    "{label} contains feature index {i}, but n_features is {}",
                    f.n_features,
                );
                m | (1u128 << i)
            })
        };

        Self {
            n_features: f.n_features,
            max_group_width: f.max_group_width,
            varying_mask: to_mask(&f.varying_features, "varying_features"),
            mono_inc_mask: to_mask(&f.mono_inc_features, "mono_inc_features"),
            mono_dec_mask: to_mask(&f.mono_dec_features, "mono_dec_features"),
            ablation: AblationMode::default(),
        }
    }

    /// Check if feature `feat` is varying. Single AND + compare.
    #[inline]
    #[must_use]
    pub const fn is_varying(&self, feat: usize) -> bool {
        self.varying_mask & (1u128 << feat) != 0
    }

    /// Check if feature `feat` is monotonic increasing.
    #[inline]
    #[must_use]
    pub const fn is_mono_inc(&self, feat: usize) -> bool {
        self.mono_inc_mask & (1u128 << feat) != 0
    }

    /// Check if feature `feat` is monotonic decreasing.
    #[inline]
    #[must_use]
    pub const fn is_mono_dec(&self, feat: usize) -> bool {
        self.mono_dec_mask & (1u128 << feat) != 0
    }
}
