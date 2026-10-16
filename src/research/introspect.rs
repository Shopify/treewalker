//! Model introspection: the parsed nodes and trees, and what the kernel's rules chose
//! for a model on this machine.

use super::{Node, ThresholdType, Tree};
use crate::exact::Plan;
use crate::forest::Forest;
use crate::load::layout::LeafValues;
use crate::rules::machine;
use crate::rules::{self, Tuning};

/// What the kernel chose for a model on this machine, the inputs of each choice, and
/// the size of its layout: the per-cell model statistics of the benchmark harness.
///
/// See the `rules` module (`src/rules/`) for each rule.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct LayoutStats {
    /// Share of internal nodes that split on varying features.
    pub varying_share: f64,
    /// Walk lanes of the walks with varying stops (groups), for the model's kind of
    /// categorical split.
    pub lanes: usize,
    /// Walk lanes of the walks without them (single rows).
    pub lanes_constant: usize,
    /// General-purpose registers of the target the rule counts (compile time).
    pub registers: usize,
    /// Vector register bits of the build (compile time).
    pub vector_bits: usize,
    /// Bytes of the node pool, 16 per node including the dummy.
    pub pool_bytes: usize,
    /// Bytes of the leaf-value table.
    pub leaf_bytes: usize,
    /// Slots of a wide accumulator, 8 bytes each, or 0 when the model's sums fit the
    /// 126-bit rule and take an `i128`.
    pub wide_limbs: usize,
    /// Bytes of the categorical features' rank tables: an unlisted category's rank
    /// in each chain, and each code's exceptions, its ranks where they differ.
    pub rank_table_bytes: usize,
    /// Bytes of the varying categorical features' flip lists: per code, the
    /// predicates whose masks its rows flip, and their bounds.
    pub flip_table_bytes: usize,
    /// The categorical features' chain slots, which a row fills from the rank
    /// tables, and the distinct heavy sets of their splits, which the chains cover.
    pub chain_slots: usize,
    pub chain_sets: usize,
    /// Bytes per cache line, and the L1 data cache's and L2's share per CPU.
    pub line_bytes: usize,
    pub l1_bytes: usize,
    pub l2_share_bytes: usize,
    /// Where the cache sizes came from: `sysctl`, `sysfs`, `cpuid` or `assumed`.
    pub cache_source: &'static str,
    /// The chunk budget: bytes of a chunk's working set.
    pub chunk_bytes: usize,
    /// Expected lane steps per tree, to the leaf and to the first varying stop, for
    /// rows like the training data.
    pub steps_leaf: f64,
    pub steps_stop: f64,
    /// The cache level holding the nodes and the walk's restart prefetch distance in
    /// items, for single rows and for groups, and that level's latency in L1
    /// latencies as the rules read it (quantized), in nanoseconds, and in L1
    /// latencies as measured.
    pub pool_level: u8,
    pub distance_constant: usize,
    pub distance: usize,
    /// The rounds per prefetch of the walk for single rows (the walk for groups
    /// prefetches every round).
    pub rate_constant: usize,
    pub pool_latency_l1s: f64,
    pub pool_latency_ns: f64,
    pub pool_latency_ratio: f64,
    /// Prefix groups, the trees in them, their deepest and mean depth, and the walk
    /// steps they save and cost per call, as the rule counts them.
    pub prefix_groups: usize,
    pub prefix_trees: usize,
    pub prefix_max_depth: usize,
    pub prefix_mean_depth: f64,
    pub prefix_saved_steps: f64,
    pub prefix_cost_steps: f64,
}

impl Forest {
    /// All trees' nodes, in the parsed layout.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.model.nodes
    }

    /// Per-tree offsets into [`Self::nodes`].
    #[must_use]
    pub fn trees(&self) -> &[Tree] {
        &self.model.trees
    }

    /// One tree's nodes.
    #[must_use]
    pub fn tree_nodes(&self, tree: &Tree) -> &[Node] {
        &self.model.nodes[tree.node_start as usize..(tree.node_start + tree.node_count) as usize]
    }

    /// Bytes of the parsed categorical splits' category lists: 4 per category and
    /// 4 per list, for the splits whose categories are not all below 32.
    #[must_use]
    pub fn bitset_bytes(&self) -> usize {
        size_of_val(self.model.categories.as_slice())
    }

    /// Whether thresholds compare in `f32` (XGBoost) or `f64` (LightGBM).
    #[must_use]
    pub fn threshold_type(&self) -> ThresholdType {
        self.model.threshold_type
    }

    /// The kernel's choices for this model on this machine, their inputs, and its
    /// layout's size. The first call in a process probes the caches' latencies, as
    /// creating the first predictor does.
    #[must_use]
    pub fn layout_stats(&self) -> LayoutStats {
        let model = &self.model;
        let layout = &model.layout;
        let caches = machine::caches();
        let tuning = Tuning::new(model);
        let level = tuning.varying.level;
        LayoutStats {
            varying_share: layout.varying_share,
            lanes: rules::lanes(true),
            lanes_constant: rules::lanes(false),
            registers: machine::GENERAL_REGISTERS,
            vector_bits: machine::VECTOR_BITS,
            pool_bytes: size_of_val(layout.nodes.as_slice()),
            leaf_bytes: match &layout.leaves {
                LeafValues::Fixed(v) => size_of_val(v.as_slice()),
                LeafValues::Wide(v) => size_of_val(v.as_slice()),
            },
            wide_limbs: match model.plan {
                Plan::Fixed(_) => 0,
                Plan::Wide(wide) => wide.limbs(),
            },
            rank_table_bytes: layout.bins.rank_table_bytes(),
            flip_table_bytes: size_of_val(model.masks.flip_bounds.as_slice())
                + size_of_val(model.masks.flips.as_slice()),
            chain_slots: layout.bins.chain_slots(),
            chain_sets: layout.chain_sets,
            line_bytes: caches.line,
            l1_bytes: caches.level(1).map_or(0, |c| c.bytes),
            l2_share_bytes: caches.level(2).map_or(0, machine::Cache::per_cpu),
            cache_source: caches.source.name(),
            chunk_bytes: tuning.chunk_bytes,
            steps_leaf: model.steps.leaf,
            steps_stop: model.steps.stop,
            pool_level: level,
            distance_constant: tuning.constant.items,
            distance: tuning.varying.items,
            rate_constant: usize::from(tuning.constant.rate),
            pool_latency_l1s: tuning.varying.latency,
            pool_latency_ns: machine::latency(level, &layout.nodes, &model.rules.pool_latency).ns,
            pool_latency_ratio: machine::latency(level, &layout.nodes, &model.rules.pool_latency)
                .ratio,
            prefix_groups: model.prefix.groups,
            prefix_trees: model.prefix.trees,
            prefix_max_depth: model.prefix.max_depth,
            prefix_mean_depth: model.prefix.mean_depth,
            prefix_saved_steps: model.prefix.saved,
            prefix_cost_steps: model.prefix.cost,
        }
    }
}
