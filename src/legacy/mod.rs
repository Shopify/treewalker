//! Today's loader, model and kernel, which the new pipeline has replaced: every Forest
//! still loads today's model beside the new one until they are deleted.

pub(crate) mod forest;
pub(crate) mod mask;
pub(crate) mod parser;
pub(crate) mod predict;

use crate::config::Choices;
use crate::rules::Prefix;

/// Today's load options for the new loader's layout choices; the rule's prefix
/// groups are today's fixed depth of 2.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Options {
    pub(crate) disable_tree_ordering: bool,
    pub(crate) disable_bitset_intern: bool,
    pub(crate) prefix_depth: usize,
    pub(crate) disable_predicate_dedup: bool,
}

impl Options {
    pub(crate) const fn of(choices: &Choices) -> Self {
        Self {
            disable_tree_ordering: !choices.tree_ordering,
            disable_bitset_intern: !choices.bitset_intern,
            prefix_depth: match choices.prefix {
                Prefix::Rule => 2,
                Prefix::Depth(k) => k,
            },
            disable_predicate_dedup: !choices.predicate_dedup,
        }
    }
}
