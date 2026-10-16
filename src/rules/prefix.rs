//! Prefix groups: runs of trees sharing their first heavy-path constant splits, which a
//! group evaluates once, and how deep each group goes.

/// How the load forms prefix groups: by [`prefix_groups`]' rule, or, for the research
/// ablations, every run of trees sharing exactly `k` heavy-path constant splits (0
/// for none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Prefix {
    Rule,
    Depth(usize),
}
