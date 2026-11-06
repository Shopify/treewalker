//! Layout schema: array fields live here, so rank-table and leaf-width changes
//! have one serialization site each. Bump the format version when changing them.
use super::exceptions::ExceptionSlots;
use super::{BinMap, CategoryBins, CategoryCodes, FeatureBins, Layout, Need};
use crate::layout_file::codec::fields;
use crate::layout_file::codec::tagged;

fields!(CategoryCodes { start, len, small });
fields!(CategoryBins {
    codes,
    slot,
    slots,
    offsets,
    held
});
fields!(
    FeatureBins {
        feature,
        thresholds,
        slot,
        categories
    },
    33
);
fields!(BinMap {
    f32,
    n_features,
    n_slots,
    thresholds,
    books,
    defaults,
    offsets,
    exception_slots,
    exception_words,
    shifts,
    n_categorical,
    constant,
    varying
});
fields!(Layout {
    nodes,
    bins,
    leaves,
    varying_share,
    chain_sets,
    chains
});
fields!(Need { leaves, slots }, 16);
// Width is explicit on disk, not inferred from Rust's enum representation.
tagged!(ExceptionSlots { 2 => Narrow(value), 4 => Wide(value) });
