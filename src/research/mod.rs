//! Research API, behind the `research` feature: runtime ablations, work counters and
//! phase times, the stages of a prediction, the reference full walk and model
//! introspection.
//!
//! None of it is needed to serve predictions. It exists to measure and check the
//! production path:
//!
//! ```
//! # use treewalker_gbdt::{Forest, ModelFormat, LoadOptions, WalkerConfig};
//! use treewalker_gbdt::research::{Ablation, Stages};
//! # let model = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/import/identity.bin")).unwrap();
//! # let config = WalkerConfig::builder(2).max_group_width(4).varying([0]).build().unwrap();
//! # let forest = Forest::from_bytes(&model, ModelFormat::TreeliteBinaryV4, config, &LoadOptions::default()).unwrap();
//! let rows = [0.5, 1.0, 1.5, 1.0]; // two rows of two features
//! let mut out = [0.0; 2];
//!
//! let mut r = forest.research_predictor(Ablation { disable_unsplit: true, ..Default::default() });
//! r.predict_group(&rows, &mut out); // timed: the ablation, no counters
//! let counters = r.predict_group_counted(&rows, &mut out); // the same ablation, counted
//! assert_eq!(counters.groups, 1);
//! let phases = r.take_phases(); // time per phase of the counted calls
//!
//! let mut stages = Stages::default();
//! r.predict_group_stages(&rows, &mut stages); // tree_sum, raw_margin, output per row
//! assert_eq!(stages.output, out);
//!
//! let mut reference = [0.0; 2];
//! forest.predict_full_walk(&rows, &mut reference); // every tree, row by row
//! ```

mod ablation;
mod introspect;
mod predictor;
mod reference;

pub use ablation::{Ablation, LoadAblation};
pub use introspect::LayoutStats;
pub use predictor::{ResearchPredictor, Stages};

pub use crate::model::{Node, ThresholdType, Tree};
pub use crate::predict::{COUNTERS_VERSION, Phases, WorkCounters};
