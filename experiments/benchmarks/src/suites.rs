//! The two suite definitions, as method sets.
//!
//! `treewalker-exp manifest` decides which cells a suite has and which research
//! variants each cell times; the runner decides what each method set loads.
//!
//! - `factorial`: production `predict`, the full walk, `treewalker_chunked128` for
//!   groups above 128 rows, and every baseline the cell supports.
//! - `ablation`: production `predict` as the reference, plus the cell's research
//!   variants. Ablation timings are normalized to the research all-on variant.
//! - `treewalker`: TreeWalker's methods only, for cells with no native model.

use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodSet {
    Factorial,
    Ablation,
    TreeWalker,
}

impl MethodSet {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "factorial" => Self::Factorial,
            "ablation" => Self::Ablation,
            "treewalker" => Self::TreeWalker,
            other => bail!("unknown method set {other}"),
        })
    }

    pub const fn full_walk(self) -> bool {
        matches!(self, Self::Factorial | Self::TreeWalker)
    }

    pub const fn baselines(self) -> bool {
        matches!(self, Self::Factorial)
    }

    /// The methods a cell of this set requires: a run without them is rejected.
    pub const fn required(self, framework: &str) -> &'static [&'static str] {
        match self {
            Self::Factorial => match framework.as_bytes() {
                b"lightgbm" => &["treewalker", "treewalker_fullwalk", "lightgbm_native"],
                b"xgboost" => &["treewalker", "treewalker_fullwalk", "xgboost_native"],
                _ => &["treewalker", "treewalker_fullwalk"],
            },
            Self::Ablation => &["treewalker"],
            Self::TreeWalker => &["treewalker", "treewalker_fullwalk"],
        }
    }
}
