//! The kernel's constants, derived from the forest and the machine ([`machine`])
//! rather than tuned on benchmarks. Each rule states its inputs and its derivation
//! next to its code:
//!
//! - [`lanes()`]: walk lanes, from the target's registers and the walk's demand for them.
//! - [`chunk_budget`]: bytes of a chunk's working set, from the L2 share per CPU.
//! - [`distance`]: restart prefetch distance, from the latency of the level that
//!   holds the nodes and a lower bound on the time a lane spends on an item.
//! - [`prefix_groups`]: which trees share a constant prefix and how deep, from the
//!   forest's heavy paths and their training weights.

mod lanes;
pub(crate) mod machine;
mod prefetch;
mod prefix;
mod steps;

pub(crate) use lanes::{MAX_LANES, lanes};
pub(crate) use prefetch::{Distance, MAX_DISTANCE, distance, pass_distance};
pub(crate) use prefix::{
    PathNode, Paths, Prefix, PrefixStats, Token, compare_paths, prefix_groups,
};
pub(crate) use steps::{ItemLengths, StepSums, Steps};

use machine::Caches;

// ---------------------------------------------------------------------------
// Chunk budget
// ---------------------------------------------------------------------------

/// Bytes a chunk's passes may keep in cache: one CPU's share of L2.
///
/// Every pass of a chunk rewrites and rereads its scratch (walk queues, stops,
/// words, leaf entries and arena) and rereads the predicates' masks, so they should
/// stay in the cache nearest the core that holds them all: L1 holds a fraction of a
/// typical chunk, and L2 is the last level private to a core or a small cluster.
/// Each CPU gets its share, as when every core runs a predictor. The chunk's nodes
/// and leaf values do not count: a group reaches each node with one item at most
/// and reads each leaf value once, so they stream through once per group (counting
/// them, as a first form of this rule did, halved the M4's chunks and doubled the
/// passes of 500-tree depth-8 groups, which cost them 1–3%).
pub(crate) fn chunk_budget(caches: &Caches) -> usize {
    caches
        .level(2)
        .map_or(machine::ASSUMED[1].bytes, machine::Cache::per_cpu)
}
