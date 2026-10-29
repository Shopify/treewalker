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

/// What the rules read of a model besides its layout and steps, boxed in
/// [`crate::model::Model`]: its walks' item lengths, and its node pool's latency when
/// the pool passes the last cache, measured by the first predictor that needs it
/// ([`machine::latency`]). A cell inline would make `Model` interior-mutable, so that
/// LLVM could no longer treat the kernel's `&Model` as read-only (it changed the code
/// of every kernel function that takes one), and the item lengths inline would move
/// the kernel's fields.
#[derive(Debug, Default)]
pub(crate) struct RuleInputs {
    pub(crate) items: ItemLengths,
    pub(crate) pool_latency: std::sync::OnceLock<machine::Latency>,
}

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

// ---------------------------------------------------------------------------
// A predictor's choices
// ---------------------------------------------------------------------------

/// What a predictor's kernel runs with on this machine: the chunk budget and the
/// prefetch distances of the walk without varying stops (single rows) and with
/// them (groups).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tuning {
    pub(crate) chunk_bytes: usize,
    pub(crate) constant: Distance,
    pub(crate) varying: Distance,
}

impl Tuning {
    /// The rules applied to `model` on this machine. The first call probes the
    /// caches' latencies.
    pub(crate) fn new(model: &crate::model::Model) -> Self {
        let caches = machine::caches();
        let layout = &model.layout;
        let pool = size_of_val(layout.nodes.as_slice());
        // Prefix starts skip the grouped trees' shared splits.
        let skipped = model.prefix.saved / model.tree_roots.len().max(1) as f64;
        let (leaf, stop) = (model.steps.leaf - skipped, model.steps.stop - skipped);
        Self {
            chunk_bytes: chunk_budget(caches),
            constant: distance(
                pool,
                &layout.nodes,
                &model.rules.pool_latency,
                leaf,
                Some(&model.rules.items.leaf_short),
                lanes(false),
                caches,
            ),
            varying: distance(
                pool,
                &layout.nodes,
                &model.rules.pool_latency,
                stop,
                None,
                lanes(true),
                caches,
            ),
        }
    }
}
