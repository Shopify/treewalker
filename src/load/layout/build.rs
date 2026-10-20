//! Building the layout from the decoded forest, in one pass over the ordered trees.
//!
//! The bins, codebooks, chains and rank tables come from the forest's distinct
//! splits, and each split's node words for both heavy sides from those; the pass
//! then rewrites every decoded node in place as its layout node, and writes the leaf
//! values, in tree order, with each tree's expected steps and scratch needs. The
//! nodes stay where they were decoded, in input order: tree order only decides the
//! order of the tree starts, and so of the prefix groups and chunks, and of the leaf
//! values.

use super::chains::{self, Sets};
use super::{
    BinMap, CategoryBins, CategoryCodes, ChainPoint, DUMMY_SLOT, FIRST_CATEGORY, FLAG_LEAF,
    FeatureBins, Layout, LayoutNode, LeafValues, MASK_SHIFT, MAX_PREDICATES, MAX_SLOTS, MISSING,
    Need, SLOT_SHIFT,
};
use crate::LoadError;
use crate::config::WalkerConfig;
use crate::exact::Plan;
use crate::load::parse::decoded::{LayoutSource, Packed, Split, TreeNodes};
use crate::load::predicates::PredicateIds;
use crate::model::{FLAG_CATEGORICAL, FLAG_VARYING, ThresholdType, category_list};
use crate::rules::{ItemLengths, StepSums, Steps};

/// Both flags of a varying categorical split, tested at once: rarely true, where a
/// branch on whether a split varies first would be unpredictable.
const VARYING_CATEGORICAL: u8 = FLAG_VARYING | FLAG_CATEGORICAL;

/// What the build reads.
pub(crate) struct Input<'a> {
    /// The decoded trees in tree order, and each one's root, where it was decoded.
    pub(crate) order: &'a [u32],
    pub(crate) starts: &'a [u32],
    /// The lists of listed splits, at the splits' values.
    pub(crate) categories: &'a [u32],
    pub(crate) predicates: &'a PredicateIds,
    pub(crate) n_predicates: usize,
    pub(crate) config: &'a WalkerConfig,
    pub(crate) threshold_type: ThresholdType,
    pub(crate) plan: Plan,
    /// The cap on the rank tables' bytes.
    pub(crate) max_rank_bytes: u64,
    /// Each tree's prefix depth, in tree order: its group's, or 0.
    pub(crate) prefix_depths: &'a [u32],
}

/// The layout and what the pass over the nodes learns.
pub(crate) struct Built {
    pub(crate) layout: Layout,
    pub(crate) steps: Steps,
    pub(crate) items: ItemLengths,
    pub(crate) needs: Vec<Need>,
    /// Each varying predicate's test, for the categorical ones.
    pub(crate) points: Vec<ChainPoint>,
    /// The parsed nodes, for introspection and the full walk.
    #[cfg(any(test, feature = "research"))]
    pub(crate) nodes: Vec<crate::model::Node>,
}

/// A split's categories, from its inline word or its list in `categories`.
fn split_categories(split: &Split, categories: &[u32], out: &mut Vec<u32>) {
    out.clear();
    if split.inline_cat() {
        let mut w = split.value as u32;
        while w != 0 {
            out.push(w.trailing_zeros());
            w &= w - 1;
        }
    } else {
        out.extend_from_slice(category_list(categories, split.value as u32));
    }
}

/// What [`chain_slots`] reads of one categorical feature.
struct Feature<'a> {
    codes: CategoryCodes,
    books: &'a [u32],
    /// The feature's splits, some numeric, in `splits`.
    ids: &'a [u32],
    splits: &'a [Split],
    categories: &'a [u32],
    /// The feature's first chain slot, and its index among the categorical features.
    slot: usize,
    held: usize,
    /// The cap on the rank tables' bytes, which also bounds the heavy sets and their
    /// inclusion graph, built on the way.
    max_bytes: u64,
}

/// The categorical features' rank tables, flat: [`BinMap`]'s defaults, offsets and
/// exceptions, and the chain slots so far.
#[derive(Default)]
struct Tables {
    defaults: Vec<u32>,
    offsets: Vec<u32>,
    exceptions: Vec<[u32; 2]>,
    chains: usize,
}

/// A load that would exceed `max_bytes`.
fn too_big(what: &str, max_bytes: u64) -> LoadError {
    LoadError::Limit(format!(
        "{what} exceed {max_bytes} bytes (LoadOptions::max_rank_table_bytes)"
    ))
}

/// The chain slots of one categorical feature: its splits' distinct heavy sets, a
/// minimum chain cover of them, each categorical split's words by heavy side into
/// `words`, and the feature's defaults and exceptions appended to `tables`; with the
/// number of distinct heavy sets.
///
/// A heavy set is a split's list, its listed codes and with a default left missing,
/// or with its heavy side on the right the list's complement, which holds no
/// category. Everything is built from the lists, never from sets over the domain,
/// so the work and the bytes follow the model's lists.
fn chain_slots(
    f: &Feature<'_>,
    tables: &mut Tables,
    words: &mut [[u64; 2]],
    members: &mut Vec<u32>,
) -> Result<(CategoryBins, usize), LoadError> {
    let domain = f.codes.domain();
    // Each split side's heavy set: its list in `lists` and whether it is complemented.
    let owners: Vec<(u32, usize)> = f
        .ids
        .iter()
        .filter(|&&s| f.splits[s as usize].is_categorical())
        .flat_map(|&s| {
            (0..2)
                .filter(move |&side| f.splits[s as usize].sides >> side & 1 != 0)
                .map(move |side| (s, side))
        })
        .collect();
    let (mut lists, mut spans) = (Vec::new(), Vec::with_capacity(owners.len()));
    for &(s, side) in &owners {
        let split = &f.splits[s as usize];
        split_categories(split, f.categories, members);
        let at = lists.len();
        if (at + members.len() + 1) as u64 * 4 > f.max_bytes {
            return Err(too_big("a feature's categorical heavy sets", f.max_bytes));
        }
        if split.default_left() {
            lists.push(MISSING);
        }
        lists.extend(members.iter().map(|&c| {
            let code = f.codes.of(f.books, c);
            debug_assert!(
                code >= FIRST_CATEGORY,
                "a split's categories are in its codebook"
            );
            code
        }));
        // Already sorted and distinct: a split's categories ascend (an inline word's
        // bits from the lowest, a list as encoded), the codebook keeps their order,
        // and missing's code is below every category's.
        debug_assert!(
            lists[at..].is_sorted_by(|a, b| a < b),
            "a heavy set's codes ascend"
        );
        spans.push((at as u32, lists.len() as u32, side == 1));
    }
    let set_bytes = lists.len() as u64 * 4;
    // The distinct sets, by size, then listed before complemented, then by list.
    let list = |i: usize| &lists[spans[i].0 as usize..spans[i].1 as usize];
    // Sorted by a key of the size, the kind and the first code, then by the lists.
    let key = |i: usize| {
        let (list, complement) = (list(i), spans[i].2);
        let size = if complement {
            domain - list.len()
        } else {
            list.len()
        };
        (size as u64) << 33
            | u64::from(complement) << 32
            | u64::from(list.first().map_or(0, |&c| c))
    };
    let mut by_size: Vec<(u64, u32)> = (0..owners.len()).map(|i| (key(i), i as u32)).collect();
    by_size.sort_unstable_by(|&(ka, a), &(kb, b)| {
        ka.cmp(&kb)
            .then_with(|| list(a as usize).cmp(list(b as usize)))
    });
    let (mut starts, mut codes, mut complement) = (vec![0u32], Vec::new(), Vec::new());
    let mut set_of = vec![0u32; owners.len()];
    let mut last: Option<(u64, usize)> = None;
    for &(k, i) in &by_size {
        let i = i as usize;
        if last.is_none_or(|(kl, l)| kl != k || list(l) != list(i)) {
            codes.extend_from_slice(list(i));
            starts.push(codes.len() as u32);
            complement.push(spans[i].2);
            last = Some((k, i));
        }
        set_of[i] = (complement.len() - 1) as u32;
    }
    drop(lists);
    let sets = Sets {
        domain,
        starts: &starts,
        codes: &codes,
        complement: &complement,
    };
    let cover = chains::cover(&sets, f.max_bytes - set_bytes)?;

    // Each split side's chain slot and position.
    for (&(s, side), &i) in owners.iter().zip(&set_of) {
        let (chain, k) = cover.at[i as usize];
        words[s as usize][side] = u64::from(k) | ((f.slot as u64 + u64::from(chain)) << SLOT_SHIFT);
    }

    // A chain's listed sets come first, as a complement, which holds no category, is
    // in no listed set, so no category's rank is the number of listed sets: the
    // default. A code's rank differs from it only where a set of the chain lists the
    // code: at the first listed set that holds it, or past the last complement whose
    // list holds it. So the exceptions come from the lists, `(code, chain, rank)`.
    let m = cover.chains();
    let mut found: Vec<(u32, u32, u32)> = Vec::new();
    tables.defaults.resize(f.slot, 0);
    for c in 0..m {
        let chain = cover.chain(c);
        let listed = chain
            .iter()
            .take_while(|&&i| !complement[i as usize])
            .count();
        tables.defaults.push(listed as u32);
        for (k, &i) in chain.iter().enumerate() {
            let set = sets.list(i as usize);
            if k < listed {
                let before = if k == 0 {
                    &[][..]
                } else {
                    sets.list(chain[k - 1] as usize)
                };
                found.extend(chains::difference(set, before).map(|e| (e, c as u32, k as u32)));
            } else {
                let after = chain.get(k + 1).map_or(&[][..], |&j| sets.list(j as usize));
                found.extend(chains::difference(set, after).map(|e| (e, c as u32, k as u32 + 1)));
            }
        }
    }
    tables.chains += m;
    let bytes = 4 * (tables.chains + tables.offsets.len() + domain + 1) as u64
        + 8 * (tables.exceptions.len() + found.len()) as u64;
    if bytes > f.max_bytes {
        return Err(too_big("the categorical splits' rank tables", f.max_bytes));
    }
    let base = tables.exceptions.len();
    let end = u32::try_from(base + found.len())
        .map_err(|_| LoadError::Limit("the rank tables exceed u32 exceptions".into()))?;
    // By code, each code's in chain order: a counting sort.
    let offsets = tables.offsets.len();
    LoadError::reserve(tables.offsets.try_reserve_exact(domain + 1))?;
    tables.offsets.resize(offsets + domain + 1, 0);
    let bounds = &mut tables.offsets[offsets..];
    for &(e, _, _) in &found {
        bounds[e as usize + 1] += 1;
    }
    bounds[0] = base as u32;
    for e in 0..domain {
        bounds[e + 1] += bounds[e];
    }
    debug_assert_eq!(bounds[domain], end);
    LoadError::reserve(tables.exceptions.try_reserve_exact(found.len()))?;
    tables.exceptions.resize(end as usize, [0, 0]);
    let mut next = bounds[..domain].to_vec();
    for &(e, c, rank) in &found {
        let at = &mut next[e as usize];
        tables.exceptions[*at as usize] = [(f.slot + c as usize) as u32, rank];
        *at += 1;
    }
    Ok((
        CategoryBins {
            codes: f.codes,
            slot: f.slot,
            chains: m,
            offsets,
            held: f.held,
        },
        sets.len(),
    ))
}

impl Layout {
    /// Build the layout of the decoded forest `source` in `input.order`, with leaf
    /// values for the model's exact-sum [`Plan`], rewriting its nodes in place.
    pub(crate) fn build(input: &Input<'_>, source: LayoutSource) -> Result<Built, LoadError> {
        let n_predicates = input.n_predicates;
        if n_predicates > MAX_PREDICATES {
            return Err(LoadError::Limit(format!(
                "{n_predicates} varying predicates exceed {MAX_PREDICATES}"
            )));
        }
        // Children, leaf indices and the dummy's index are 32-bit, and the dummy's
        // heavy child, one past it, too.
        let total = source.nodes.len();
        let dummy = u32::try_from(total)
            .ok()
            .filter(|&d| d < u32::MAX)
            .ok_or_else(|| LoadError::Limit(format!("{total} nodes exceed u32")))?;
        let mut words = vec![[0u64; 2]; source.splits.len()];
        let (bins, chain_sets) = bin_map(input, &source.splits, &mut words)?;
        Ok(encode_trees(input, source, &words, dummy, bins, chain_sets))
    }
}

/// The bins, codebooks and rank tables of the forest's distinct `splits`, flat, each
/// feature's slots in feature order, and each split's word 0 by its heavy side
/// (left, right) into `words`; with the number of distinct heavy sets.
fn bin_map(
    input: &Input<'_>,
    splits: &[Split],
    words: &mut [[u64; 2]],
) -> Result<(BinMap, usize), LoadError> {
    let Input {
        categories,
        config,
        threshold_type,
        max_rank_bytes,
        ..
    } = *input;
    // Splits by feature, in feature order.
    let mut by_feature: Vec<u32> = (0..splits.len() as u32).collect();
    by_feature.sort_unstable_by_key(|&s| splits[s as usize].feature);

    let (mut thresholds, mut books, mut tables) = (Vec::new(), Vec::new(), Tables::default());
    let (mut constant, mut varying) = (Vec::new(), Vec::new());
    let (mut t, mut cats, mut members) = (Vec::new(), Vec::new(), Vec::new());
    let mut next_slot = DUMMY_SLOT + 1;
    let (mut chain_sets, mut n_categorical) = (0, 0);
    for ids in
        by_feature.chunk_by(|&a, &b| splits[a as usize].feature == splits[b as usize].feature)
    {
        let feature = splits[ids[0] as usize].feature;
        t.clear();
        cats.clear();
        let mut categorical = false;
        for &s in ids {
            let split = &splits[s as usize];
            if split.is_categorical() {
                categorical = true;
                split_categories(split, categories, &mut members);
                cats.extend_from_slice(&members);
            } else {
                // Float32 thresholds arrive rounded, so comparing in f64 is exact.
                debug_assert!(
                    threshold_type == ThresholdType::F64
                        || f64::from(split.value as f32).to_bits() == split.value.to_bits()
                );
                t.push(split.value);
            }
        }
        let numeric_splits = !t.is_empty();
        t.sort_by(f64::total_cmp);
        // Sorted, so `<=` is IEEE equality: signed zeros split alike and share a bin.
        t.dedup_by(|later, earlier| later <= earlier);
        let start = thresholds.len();
        thresholds.extend_from_slice(&t);
        let numeric = next_slot;
        if numeric_splits {
            next_slot += 4;
        }
        let category_bins = if !categorical {
            None
        } else {
            cats.sort_unstable();
            cats.dedup();
            let codes = CategoryCodes::new(&mut books, &cats)?;
            let (c, sets) = chain_slots(
                &Feature {
                    codes,
                    books: &books,
                    ids,
                    splits,
                    categories,
                    slot: next_slot,
                    held: n_categorical,
                    max_bytes: max_rank_bytes,
                },
                &mut tables,
                words,
                &mut members,
            )?;
            next_slot += c.chains;
            n_categorical += 1;
            chain_sets += sets;
            Some(c)
        };
        if next_slot > MAX_SLOTS {
            return Err(LoadError::Limit(format!(
                "the features' bins need more than {MAX_SLOTS} slots"
            )));
        }
        for &s in ids {
            let split = &splits[s as usize];
            if split.is_categorical() {
                continue;
            }
            let right = u64::from(!split.default_left());
            let k = t.partition_point(|&x| x < split.value);
            debug_assert!(t[k] <= split.value, "threshold is in its feature's bins");
            let left = (numeric as u64 + right) << SLOT_SHIFT;
            let inverted = (numeric as u64 + 2 + right) << SLOT_SHIFT;
            words[s as usize] = [
                k as u64 | left,
                u64::from(u32::MAX - 1 - k as u32) | inverted,
            ];
        }
        let b = FeatureBins {
            feature: feature as usize,
            thresholds: start..thresholds.len(),
            slot: numeric,
            categories: category_bins,
        };
        if config.is_varying(feature as usize) {
            varying.push(b);
        } else {
            constant.push(b);
        }
    }
    if thresholds.len() >= u32::MAX as usize {
        return Err(LoadError::Limit("threshold count exceeds u32".into()));
    }
    let Tables {
        mut defaults,
        offsets,
        exceptions,
        ..
    } = tables;
    defaults.resize(next_slot, 0);
    defaults[DUMMY_SLOT] = u32::MAX;
    let bins = BinMap {
        f32: threshold_type == ThresholdType::F32,
        n_features: config.n_features(),
        n_slots: next_slot,
        thresholds,
        books,
        defaults,
        offsets,
        exceptions,
        n_categorical,
        constant,
        varying,
    };
    Ok((bins, chain_sets))
}

/// The trees' layout nodes, in tree order, from the words of `bin_map`: each tree's
/// nodes rewritten in place, its leaf values in the model's exact-sum [`Plan`], its
/// expected steps and its scratch needs.
fn encode_trees(
    input: &Input<'_>,
    source: LayoutSource,
    words: &[[u64; 2]],
    dummy: u32,
    bins: BinMap,
    chain_sets: usize,
) -> Built {
    let Input {
        order,
        starts,
        predicates,
        n_predicates,
        plan,
        prefix_depths,
        ..
    } = *input;
    let LayoutSource {
        trees,
        splits,
        mut nodes,
    } = source;
    // Nodes, in tree order. Each tree takes passes that do not branch on a
    // node's kind, which is unpredictable: its light children and its leaves and
    // splits listed apart, its expected steps, then its leaves and its splits,
    // each rewritten in place after it is read.
    let leaf_total: usize = trees.iter().map(|t| t.leaves as usize).sum();
    let mut leaves = match plan {
        Plan::Fixed(_) => LeafValues::Fixed(Vec::with_capacity(leaf_total)),
        Plan::Wide(_) => LeafValues::Wide(Vec::with_capacity(leaf_total)),
    };
    #[cfg(any(test, feature = "research"))]
    let mut parsed = vec![
        crate::model::Node {
            value: 0.0,
            skip: crate::model::NONE,
            varying_pred_id: crate::model::NONE,
            feature: 0,
            flags: 0,
            heavy: 0,
        };
        nodes.len()
    ];
    let mut points = vec![ChainPoint::NONE; n_predicates];
    let (mut steps, mut needs) = (StepSums::default(), Vec::with_capacity(order.len()));
    let mut kinds = Kinds::default();
    // Whether each split is constant, and an entry for leaves.
    let dummy_split = splits.len();
    let constant: Vec<bool> = splits
        .iter()
        .map(|s| !s.is_varying())
        .chain([false])
        .collect();
    let (mut internal, mut on_varying, mut next_predicate) = (0usize, 0usize, 0usize);
    for ((&t, &start), &prefix) in order.iter().zip(starts).zip(prefix_depths) {
        let tree = &trees[t as usize];
        let start = start as usize;
        let tree_nodes = TreeNodes::new(&nodes[start..start + tree.count as usize]);
        kinds.of(tree_nodes);
        needs.push(Need {
            leaves: tree.leaves as usize,
            slots: if tree.varying == 0 {
                0
            } else {
                (tree.leaves as usize).min(tree.varying as usize + 1)
            },
        });
        steps.tree(tree_nodes.len(), prefix as usize, |local| {
            let node = tree_nodes.at(local);
            let leaf = node.is_leaf();
            let id = if leaf {
                dummy_split
            } else {
                node.id() as usize
            };
            let split = (
                node.heavy_probability(),
                kinds.light[local] as usize,
                constant[id],
            );
            (!leaf).then_some(split)
        });
        for &local in &kinds.leaves {
            let at = start + local as usize;
            let value = Packed::of(nodes[at]).value();
            let leaf = match &mut leaves {
                LeafValues::Fixed(values) => {
                    let Plan::Fixed(scale) = plan else {
                        unreachable!()
                    };
                    values.push(crate::exact::to_fixed(value, scale));
                    values.len() - 1
                }
                LeafValues::Wide(values) => {
                    let Plan::Wide(wide) = plan else {
                        unreachable!()
                    };
                    values.push(wide.leaf(value));
                    values.len() - 1
                }
            } as u64;
            nodes[at] = LayoutNode {
                w0: leaf,
                w1: leaf | FLAG_LEAF,
            };
            #[cfg(any(test, feature = "research"))]
            {
                parsed[at].value = value;
            }
        }
        internal += kinds.splits.len();
        for &local in &kinds.splits {
            let at = start + local as usize;
            let node = Packed::of(nodes[at]);
            let s = node.id() as usize;
            let split = &splits[s];
            let heavy_left = node.heavy_left();
            let w0 = words[s][usize::from(!heavy_left)];
            // The predicate's heavy-side mask, without a branch on whether the
            // split varies.
            let varying = split.is_varying();
            on_varying += usize::from(varying);
            let id = match predicates {
                PredicateIds::BySplit(ids) => ids[s],
                PredicateIds::ByNode(ids) => {
                    let id = ids.get(next_predicate).copied().unwrap_or(0);
                    next_predicate += usize::from(varying);
                    id
                }
            };
            let mask = 2 * u64::from(id) + 1 + u64::from(!heavy_left);
            let mask = if varying { mask } else { 0 };
            // A varying categorical predicate's left rows are its heavy set's,
            // or their complement's when its heavy side is the right.
            if split.flags & VARYING_CATEGORICAL == VARYING_CATEGORICAL {
                points[id as usize] = ChainPoint {
                    slot: (w0 >> SLOT_SHIFT) as u32,
                    k: w0 as u32,
                    invert: !heavy_left,
                };
            }
            nodes[at] = LayoutNode {
                w0,
                w1: (start as u64 + u64::from(kinds.light[local as usize])) | mask << MASK_SHIFT,
            };
            #[cfg(any(test, feature = "research"))]
            {
                parsed[at] = crate::model::Node {
                    value: split.value,
                    skip: kinds.light[local as usize],
                    varying_pred_id: if varying { id } else { crate::model::NONE },
                    feature: split.feature,
                    flags: split.flags
                        | if heavy_left {
                            crate::model::FLAG_HEAVY_IS_LEFT
                        } else {
                            0
                        },
                    heavy: node.heavy(),
                };
            }
        }
    }
    debug_assert_eq!(nodes.len(), dummy as usize);
    // The decoders leave room for it.
    debug_assert!(nodes.capacity() > nodes.len());
    nodes.push(LayoutNode::dummy(dummy));

    let varying_share = if internal == 0 {
        0.0
    } else {
        on_varying as f64 / internal as f64
    };
    Built {
        layout: Layout {
            nodes,
            bins,
            leaves,
            varying_share,
            chain_sets,
        },
        steps: steps.steps(order.len()),
        items: steps.items(),
        needs,
        points,
        #[cfg(any(test, feature = "research"))]
        nodes: parsed,
    }
}

/// A tree's nodes by kind: each split's light child, by the split's index, and its
/// leaves and splits, each in order.
#[derive(Default)]
struct Kinds {
    light: Vec<u32>,
    leaves: Vec<u32>,
    splits: Vec<u32>,
    stack: Vec<u32>,
}

impl Kinds {
    /// The kinds of a tree in heavy-first preorder. A split's light child follows its
    /// heavy subtree, which ends at the leaf where as many leaves as splits have
    /// been seen since it: a stack of splits, pushed at each split and popped at
    /// each leaf, without a branch on the kind (entry `n` takes the writes that do
    /// nothing).
    fn of(&mut self, nodes: TreeNodes<'_>) {
        let n = nodes.len();
        for v in [
            &mut self.light,
            &mut self.leaves,
            &mut self.splits,
            &mut self.stack,
        ] {
            v.clear();
            v.resize(n + 1, 0);
        }
        let (light, leaves, splits, stack) = (
            &mut self.light[..=n],
            &mut self.leaves[..=n],
            &mut self.splits[..=n],
            &mut self.stack[..=n],
        );
        let (mut depth, mut n_leaves, mut n_splits) = (0usize, 0usize, 0usize);
        for (i, node) in nodes.iter().enumerate() {
            let leaf = node.is_leaf();
            // A leaf pops the split whose light child is next; a split pushes itself.
            let top = stack[depth.saturating_sub(1).min(n)] as usize;
            light[if leaf && depth > 0 { top.min(n) } else { n }] = i as u32 + 1;
            stack[if leaf { n } else { depth.min(n) }] = i as u32;
            depth = if leaf {
                depth.saturating_sub(1)
            } else {
                depth + 1
            };
            leaves[n_leaves.min(n)] = i as u32;
            splits[n_splits.min(n)] = i as u32;
            n_leaves += usize::from(leaf);
            n_splits += usize::from(!leaf);
        }
        self.leaves.truncate(n_leaves);
        self.splits.truncate(n_splits);
    }
}
