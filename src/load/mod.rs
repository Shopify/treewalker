//! Loading: from a Treelite model to the [`Model`] the kernel predicts with, in one
//! decode pass and one build pass.
//!
//! # Pipeline
//!
//! 1. **Decode** ([`parse`]): the binary checkpoint or the JSON dump is read and
//!    validated tree by tree. Each tree is put in heavy-first preorder, the heavier
//!    child (by training weight) right after its parent, so that a walk that goes
//!    heavy reads the next node, and kept in the 16-byte layout nodes they become,
//!    in input order, as packed words whose splits are interned in the forest's
//!    table of distinct splits ([`parse::decoded`]).
//! 2. **Tree order** ([`order`]): trees with the same first heavy-path splits sort
//!    next to each other, so that trees sharing a prefix form a run and adjacent walks
//!    take similar paths; the category lists are renumbered in that order.
//! 3. **Varying predicates** ([`predicates`]): one predicate per distinct varying
//!    split, sorted by feature, kind and threshold, so that the precompute evaluates
//!    each one once per row however many nodes test it.
//! 4. **Prefix groups** ([`crate::rules::prefix_groups`]): runs of trees sharing
//!    their first constant heavy-path splits, which a group evaluates once; each
//!    tree's walk starts where they lead.
//! 5. **Exact-sum plan** ([`crate::exact::Plan`]), from the leaf values' extent.
//! 6. **Layout** ([`layout`]): bins, codebooks, chains and rank tables from the
//!    distinct splits, then one pass per tree in final order rewrites its decoded
//!    nodes in place as layout nodes and writes the leaf values, with each tree's
//!    expected steps and scratch needs. The nodes stay in input order: tree order
//!    is the order of the tree starts, and so of the prefix groups and chunks.
//! 7. **Mask plan** ([`layout::MaskPlan`]): each varying feature's predicates, the
//!    categorical ones by chain, and per code the masks its rows flip, which the
//!    precompute reads.

// Hand-written Treelite JSON, for the layout's and the precompute's tests.
#[cfg(test)]
#[allow(dead_code, reason = "these tests use a part")]
#[path = "../../tests/common/json.rs"]
pub(crate) mod json;
pub(crate) mod layout;
mod order;
pub(crate) mod parse;
pub(crate) mod predicates;

use std::sync::Arc;

use crate::LoadError;
use crate::config::{Choices, LoadOptions, WalkerConfig};
use crate::exact::Plan;
use crate::forest::Forest;
use crate::model::{Model, Tree};

impl Forest {
    /// The model of a decoded forest: its trees ordered, its predicates, prefix
    /// groups and exact-sum plan, and its layout, written in one pass over the
    /// ordered trees, which rewrites the decoded nodes in place.
    pub(crate) fn build(
        parsed: parse::ParsedModel,
        config: WalkerConfig,
        options: &LoadOptions,
        choices: &Choices,
    ) -> Result<Self, LoadError> {
        let parse::ParsedModel {
            mut decoded,
            threshold_type,
            output,
        } = parsed;
        let order = order::tree_order(&decoded, choices.tree_ordering);
        let (categories, category_starts) = order::repack_categories(&mut decoded, &order);
        let (varying_predicates, predicate_ids) =
            predicates::varying_predicates(&decoded, &order, choices.predicate_dedup)?;
        let feature_ranges = predicates::build_feature_ranges(&varying_predicates);
        let mut tree_starts = Vec::with_capacity(order.len());
        let mut trees = Vec::with_capacity(order.len());
        for (&t, &category_start) in order.iter().zip(&category_starts) {
            let tree = &decoded.trees[t as usize];
            tree_starts.push(tree.start);
            trees.push(Tree {
                node_start: tree.start,
                node_count: tree.count,
                category_start,
            });
        }
        let paths = order::HeavyPaths {
            decoded: &decoded,
            order: Some(&order),
        };
        let (prefix_groups, prefix) =
            crate::rules::prefix_groups(&paths, &tree_starts, choices.prefix);
        let plan = Plan::of(decoded.extent, tree_starts.len());
        let mut prefix_depths = vec![0u32; tree_starts.len()];
        for g in &prefix_groups {
            prefix_depths[g.first as usize..g.end as usize].fill(g.depth);
        }
        let layout::Built {
            layout,
            steps,
            items,
            needs,
            points,
            #[cfg(any(test, feature = "research"))]
            nodes,
        } = layout::Layout::build(
            &layout::Input {
                order: &order,
                starts: &tree_starts,
                categories: &categories,
                predicates: &predicate_ids,
                n_predicates: varying_predicates.len(),
                config: &config,
                threshold_type,
                plan,
                max_rank_bytes: options.max_rank_table_bytes,
                category_bits_above: choices.category_bits_above,
                prefix_depths: &prefix_depths,
            },
            decoded.into_layout_source(),
        )?;
        drop(predicate_ids);
        // The walks start at the roots' links.
        let mut tree_roots = tree_starts;
        for root in &mut tree_roots {
            *root = layout::node_link(*root as usize);
        }
        let bounds = crate::predict::sweep_bounds(&varying_predicates, threshold_type);
        let masks = layout::MaskPlan::build(
            &layout.bins,
            &feature_ranges,
            &points,
            options.max_rank_table_bytes,
        )?;
        debug_assert!(
            (size_of::<u32>() * (masks.flip_bounds.len() + masks.flips.len())) as u64
                <= layout.bins.flip_bytes_bound(),
            "the layout bounds the masks' flips"
        );
        drop(points);

        // SAFETY: malloc_trim takes no pointers; it only returns free heap pages to
        // the system. Miri does not emulate it.
        #[cfg(all(target_os = "linux", not(miri)))]
        unsafe {
            libc::malloc_trim(0);
        }

        Ok(Self {
            model: Arc::new(Model {
                output,
                trees,
                config,
                categories,
                varying_predicates,
                masks,
                threshold_type,
                prefix_groups,
                prefix,
                tree_roots,
                steps,
                plan,
                layout,
                needs,
                rules: Box::new(crate::rules::RuleInputs {
                    items,
                    pool_latency: std::sync::OnceLock::new(),
                }),
                bounds,
                #[cfg(any(test, feature = "research"))]
                nodes,
            }),
        })
    }
}

/// The index of `len` entries appended to a table of `start` entries that 32-bit
/// indices address, the category pool or the codebooks: past `u32::MAX` entries in
/// all, a [`LoadError::Limit`], so that every index into the table, and its length,
/// fits in 32 bits.
pub(crate) fn u32_extent(start: usize, len: usize, table: &str) -> Result<u32, LoadError> {
    match start.checked_add(len).map(u32::try_from) {
        Some(Ok(_)) => Ok(start as u32),
        _ => Err(LoadError::Limit(format!(
            "{table} exceed {} entries: their indices are 32-bit",
            u32::MAX
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bound of the category pool and the codebooks, without 2^32 entries.
    #[test]
    fn u32_tables_end_at_u32_max() {
        let max = u32::MAX as usize;
        assert_eq!(u32_extent(0, 0, "lists").unwrap(), 0);
        assert_eq!(u32_extent(max - 10, 10, "lists").unwrap(), u32::MAX - 10);
        assert_eq!(u32_extent(max, 0, "lists").unwrap(), u32::MAX);
        for (start, len) in [(max - 10, 11), (max, 1), (1, max), (usize::MAX, 1)] {
            assert!(matches!(
                u32_extent(start, len, "lists"),
                Err(LoadError::Limit(_))
            ));
        }
    }
}
