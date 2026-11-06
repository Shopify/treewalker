//! Configuration schema, separate from the JSON import schema.
use super::{FeatureSet, WalkerConfig};
use crate::layout_file::codec::fields;

fields!(FeatureSet { all, indices });
fields!(WalkerConfig {
    n_features,
    max_group_width,
    varying,
    increasing,
    decreasing
});
