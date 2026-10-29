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

mod prefix;

pub(crate) use prefix::Prefix;
