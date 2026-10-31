//! Building the layout from the decoded forest, in one pass over the ordered trees.
//!
//! The bins, codebooks, chains and rank tables come from the forest's distinct
//! splits, and each split's node words for both heavy sides from those; the pass
//! then rewrites every decoded node in place as its layout node, and writes the leaf
//! values, in tree order, with each tree's expected steps and scratch needs. The
//! nodes stay where they were decoded, in input order: tree order only decides the
//! order of the tree starts, and so of the prefix groups and chunks, and of the leaf
//! values.

use std::collections::BTreeSet;

use super::chains::{self, Cover, Sets};
use super::{
    BOUND_BIAS, BinMap, CategoryBins, CategoryCodes, ChainPoint, DUMMY_SLOT, FIRST_CATEGORY,
    FLAG_LEAF, FeatureBins, Layout, LayoutNode, LeafValues, MASK_SHIFT, MAX_PREDICATES, MAX_SLOTS,
    MISSING, Need, SHIFT_AT, SLOT_SHIFT, node_link,
};
use crate::LoadError;
use crate::config::WalkerConfig;
use crate::exact::Plan;
use crate::load::parse::decoded::{LayoutSource, Packed, Split, TreeNodes};
use crate::load::parse::tree::check_node_total;
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
    /// A model whose chains take more slots packs their rank fields
    /// ([`super::PACK_ABOVE`] but in research).
    pub(crate) pack_above: usize,
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

/// What [`category_slots`] reads of one categorical feature.
struct Feature<'a> {
    codes: CategoryCodes,
    books: &'a [u32],
    /// The feature's splits, some numeric, in `splits`.
    ids: &'a [u32],
    splits: &'a [Split],
    categories: &'a [u32],
    /// The feature's first categorical slot, and its index among the categorical
    /// features.
    slot: usize,
    held: usize,
    /// The cap on the rank tables' bytes, which also bounds the heavy sets, built on
    /// the way.
    max_bytes: u64,
}

/// The categorical features' rank tables, flat: [`BinMap`]'s defaults, offsets and
/// exceptions, and the categorical slots so far.
#[derive(Default)]
struct Tables {
    defaults: Vec<u32>,
    offsets: Vec<u32>,
    exceptions: Vec<[u32; 2]>,
    slots: usize,
}

impl Tables {
    /// Their bytes so far, as [`BinMap::rank_table_bytes`] counts them.
    const fn bytes(&self) -> u64 {
        4 * (self.slots + self.offsets.len()) as u64 + 8 * self.exceptions.len() as u64
    }
}

/// A load that would exceed `max_bytes`.
fn too_big(what: &str, max_bytes: u64) -> LoadError {
    LoadError::Limit(format!(
        "{what} exceed {max_bytes} bytes (LoadOptions::max_rank_table_bytes)"
    ))
}

/// A categorical feature's distinct heavy sets, and which one each split side's is:
/// what [`category_slots`] covers with chains or holds in bits.
///
/// A heavy set is a split's list, its listed codes and with a default left missing,
/// or with its heavy side on the right the list's complement, which holds no
/// category. Everything is built from the lists, never from sets over the domain, so
/// the work and the bytes follow the model's lists.
struct HeavySets {
    /// Each categorical split side, `(split, side)`, and its set.
    owners: Vec<(u32, usize)>,
    set_of: Vec<u32>,
    /// The distinct sets' lists, set i's at `codes[starts[i]..starts[i + 1]]`, and
    /// whether each is complemented.
    starts: Vec<u32>,
    codes: Vec<u32>,
    complement: Vec<bool>,
    /// Bytes of every split side's list, built on the way.
    list_bytes: u64,
}

impl HeavySets {
    fn sets(&self, domain: usize) -> Sets<'_> {
        Sets {
            domain,
            starts: &self.starts,
            codes: &self.codes,
            complement: &self.complement,
        }
    }
}

/// The distinct heavy sets of the categorical splits of `f`.
fn heavy_sets(f: &Feature<'_>, members: &mut Vec<u32>) -> Result<HeavySets, LoadError> {
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
    let list_bytes = lists.len() as u64 * 4;
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
    Ok(HeavySets {
        owners,
        set_of,
        starts,
        codes,
        complement,
        list_bytes,
    })
}

/// A chain's rank field: its word among the feature's, its lowest bit and its width.
#[derive(Clone, Copy, Debug)]
struct Field {
    word: u32,
    low: u32,
    width: u32,
}

/// The fields of `cover`'s chains: with `pack`, a chain of m sets takes the bit length
/// of m, the ranks 0 to m, packed first-fit decreasing into the fewest 32-bit words,
/// each at the lowest free bit of the first word with room; else, and in a word of
/// one field, a chain takes a whole word, with no shift. With the number of words.
fn pack_fields(cover: &Cover, pack: bool) -> (Vec<Field>, usize) {
    let m = cover.chains();
    if !pack {
        let whole = |c: usize| Field {
            word: c as u32,
            low: 0,
            width: u32::BITS,
        };
        return ((0..m).map(whole).collect(), m);
    }
    let width = |c: usize| u32::BITS - (cover.chain(c).len() as u32).leading_zeros();
    let mut fields = vec![
        Field {
            word: 0,
            low: 0,
            width: 0,
        };
        m
    ];
    let mut order: Vec<u32> = (0..m as u32).collect();
    order.sort_by_key(|&c| std::cmp::Reverse(width(c as usize)));
    // The bits each word uses, its fields, and the words by their free bits.
    let (mut used, mut count): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    let mut by_free: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); u32::BITS as usize + 1];
    for &c in &order {
        let b = width(c as usize);
        let first = (b as usize..by_free.len())
            .filter_map(|free| by_free[free].first().copied())
            .min();
        let w = if let Some(w) = first {
            by_free[(u32::BITS - used[w as usize]) as usize].remove(&w);
            w
        } else {
            used.push(0);
            count.push(0);
            (used.len() - 1) as u32
        };
        fields[c as usize] = Field {
            word: w,
            low: used[w as usize],
            width: b,
        };
        used[w as usize] += b;
        count[w as usize] += 1;
        by_free[(u32::BITS - used[w as usize]) as usize].insert(w);
    }
    for f in &mut fields {
        if count[f.word as usize] == 1 {
            (f.low, f.width) = (0, u32::BITS);
        }
    }
    (fields, used.len())
}

/// The slots of one categorical feature: its splits' distinct heavy sets
/// ([`heavy_sets`]) in the chains of `cover`, each chain's rank field in the words of
/// [`pack_fields`], each categorical split's words by heavy side into `words`, and the
/// feature's word defaults and exceptions appended to `tables`.
fn category_slots(
    f: &Feature<'_>,
    heavy: &HeavySets,
    cover: &Cover,
    pack: bool,
    tables: &mut Tables,
    words: &mut [[u64; 2]],
) -> Result<CategoryBins, LoadError> {
    let domain = f.codes.domain();
    let sets = heavy.sets(domain);
    let complement = &heavy.complement;
    let (fields, n_words) = pack_fields(cover, pack);

    // Each split side's word, field and position: its bound puts the field at the top,
    // `rank <= k` exactly when the shifted word is at most ((k + 1) << (32 - b)) - 1,
    // whatever the fields below it hold.
    for (&(s, side), &i) in heavy.owners.iter().zip(&heavy.set_of) {
        let (chain, k) = cover.at[i as usize];
        let Field { word, low, width } = fields[chain as usize];
        let bound = ((u64::from(k) + 1) << (u32::BITS - width)) - 1 + u64::from(BOUND_BIAS);
        let shift = u32::BITS - width - low;
        words[s as usize][side] = bound
            | u64::from(shift) << SHIFT_AT
            | ((f.slot as u64 + u64::from(word)) << SLOT_SHIFT);
    }

    // A chain's listed sets come first, as a complement, which holds no category, is
    // in no listed set, so no category's rank is the number of listed sets: the
    // default, which each word packs for its fields. A code's rank differs from it
    // only where a set of the chain lists the code: at the first listed set that holds
    // it, or past the last complement whose list holds it. So the exceptions come from
    // the lists, `(code, chain, rank)`, taken chain by chain in word order, so that
    // each code's are in word order, then merged by word.
    let m = cover.chains();
    tables.defaults.resize(f.slot + n_words, 0);
    let mut by_word: Vec<u32> = (0..m as u32).collect();
    by_word.sort_by_key(|&c| (fields[c as usize].word, fields[c as usize].low));
    let mut found: Vec<(u32, u32, u32)> = Vec::new();
    for &c in &by_word {
        let chain = cover.chain(c as usize);
        let listed = chain
            .iter()
            .take_while(|&&i| !complement[i as usize])
            .count();
        let Field { word, low, .. } = fields[c as usize];
        tables.defaults[f.slot + word as usize] |= (listed as u32) << low;
        for (k, &i) in chain.iter().enumerate() {
            let set = sets.list(i as usize);
            if k < listed {
                let before = if k == 0 {
                    &[][..]
                } else {
                    sets.list(chain[k - 1] as usize)
                };
                found.extend(chains::difference(set, before).map(|e| (e, c, k as u32)));
            } else {
                let after = chain.get(k + 1).map_or(&[][..], |&j| sets.list(j as usize));
                found.extend(chains::difference(set, after).map(|e| (e, c, k as u32 + 1)));
            }
        }
    }
    tables.slots += n_words;
    // By code, each code's in word order: a counting sort, then a word per code and
    // word, its default with the code's ranks in its fields. The triples are at most
    // the heavy sets' listed codes, which the cap bounded; the tables are checked
    // once merged.
    let mut bounds = vec![0u32; domain + 1];
    for &(e, _, _) in &found {
        bounds[e as usize + 1] += 1;
    }
    for e in 0..domain {
        bounds[e + 1] += bounds[e];
    }
    let mut sorted = vec![(0u32, 0u32); found.len()];
    let mut next = bounds[..domain].to_vec();
    for &(e, c, rank) in &found {
        let at = &mut next[e as usize];
        sorted[*at as usize] = (c, rank);
        *at += 1;
    }
    drop(found);
    let offsets = tables.offsets.len();
    LoadError::reserve(tables.offsets.try_reserve_exact(domain + 1))?;
    for e in 0..domain {
        tables.offsets.push(
            u32::try_from(tables.exceptions.len())
                .map_err(|_| LoadError::Limit("the rank tables exceed u32 exceptions".into()))?,
        );
        let ranks = &sorted[bounds[e] as usize..bounds[e + 1] as usize];
        for run in ranks.chunk_by(|a, b| fields[a.0 as usize].word == fields[b.0 as usize].word) {
            let slot = f.slot + fields[run[0].0 as usize].word as usize;
            let mut value = tables.defaults[slot];
            for &(c, rank) in run {
                let Field { low, width, .. } = fields[c as usize];
                value = value & !((u32::MAX >> (u32::BITS - width)) << low) | rank << low;
            }
            tables.exceptions.push([slot as u32, value]);
        }
    }
    tables.offsets.push(
        u32::try_from(tables.exceptions.len())
            .map_err(|_| LoadError::Limit("the rank tables exceed u32 exceptions".into()))?,
    );
    if tables.bytes() > f.max_bytes {
        return Err(too_big("the categorical splits' rank tables", f.max_bytes));
    }
    Ok(CategoryBins {
        codes: f.codes,
        slot: f.slot,
        slots: n_words as u32,
        offsets,
        held: f.held,
    })
}

/// Whether the features' bins fit in `slots` slots: at most [`MAX_SLOTS`], what word 0's
/// slot field holds.
pub(super) fn slots_fit(slots: usize) -> Result<(), LoadError> {
    if slots > MAX_SLOTS {
        return Err(LoadError::Limit(format!(
            "the features' bins need more than {MAX_SLOTS} slots"
        )));
    }
    Ok(())
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
        // At most MAX_NODES with the dummy, on every target, which the decoders
        // checked as they appended trees.
        let total = source.nodes.len();
        check_node_total(total, 0)?;
        let dummy = node_link(total);
        let mut words = vec![[0u64; 2]; source.splits.len()];
        let (bins, sets) = bin_map(input, &source.splits, &mut words)?;
        Ok(encode_trees(input, source, &words, dummy, bins, sets))
    }
}

/// A categorical feature's codes, distinct heavy sets and their cover, between the
/// two passes of [`bin_map`].
struct Covered {
    codes: CategoryCodes,
    heavy: HeavySets,
    cover: Cover,
}

/// The bins, codebooks and rank tables of the forest's distinct `splits`, flat, each
/// feature's slots in feature order, and each split's word 0 by its heavy side
/// (left, right) into `words`; with the number of distinct heavy sets and of chains.
///
/// A first pass takes each feature's thresholds, codebook, heavy sets and cover; a
/// model whose chains then take more than `input.pack_above` slots packs their rank
/// fields ("Packed fields"), and the second pass lays the slots out.
fn bin_map(
    input: &Input<'_>,
    splits: &[Split],
    words: &mut [[u64; 2]],
) -> Result<(BinMap, [usize; 2]), LoadError> {
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
    let features =
        || by_feature.chunk_by(|&a, &b| splits[a as usize].feature == splits[b as usize].feature);

    let (mut thresholds, mut books, mut tables) = (Vec::new(), Vec::new(), Tables::default());
    let (mut constant, mut varying) = (Vec::new(), Vec::new());
    let (mut t, mut cats, mut members) = (Vec::new(), Vec::new(), Vec::new());
    // Each feature's thresholds and, if categorical, its cover; with the heavy sets'
    // lists so far, which the cap bounds.
    let mut first: Vec<(std::ops::Range<usize>, Option<Covered>)> = Vec::new();
    let (mut n_chains, mut n_sets, mut list_bytes) = (0, 0, 0);
    for ids in features() {
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
        t.sort_by(f64::total_cmp);
        // Sorted, so `<=` is IEEE equality: signed zeros split alike and share a bin.
        t.dedup_by(|later, earlier| later <= earlier);
        let start = thresholds.len();
        thresholds.extend_from_slice(&t);
        let covered = if categorical {
            cats.sort_unstable();
            cats.dedup();
            let codes = CategoryCodes::new(&mut books, &cats)?;
            let feature = Feature {
                codes,
                books: &books,
                ids,
                splits,
                categories,
                slot: 0,
                held: 0,
                max_bytes: max_rank_bytes,
            };
            let heavy = heavy_sets(&feature, &mut members)?;
            list_bytes += heavy.list_bytes;
            if list_bytes > max_rank_bytes {
                return Err(too_big(
                    "the features' categorical heavy sets",
                    max_rank_bytes,
                ));
            }
            let cover = chains::cover(&heavy.sets(codes.domain()));
            n_chains += cover.chains();
            n_sets += heavy.complement.len();
            Some(Covered {
                codes,
                heavy,
                cover,
            })
        } else {
            None
        };
        first.push((start..thresholds.len(), covered));
    }
    if thresholds.len() >= u32::MAX as usize {
        return Err(LoadError::Limit("threshold count exceeds u32".into()));
    }
    let pack = n_chains > input.pack_above;

    let mut next_slot = DUMMY_SLOT + 1;
    let mut n_categorical = 0;
    for (ids, (range, covered)) in features().zip(first) {
        let feature = splits[ids[0] as usize].feature;
        let t = &thresholds[range.clone()];
        let numeric = next_slot;
        if !t.is_empty() {
            next_slot += 4;
        }
        let category_bins = if let Some(Covered {
            codes,
            heavy,
            cover,
        }) = covered
        {
            let c = category_slots(
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
                &heavy,
                &cover,
                pack,
                &mut tables,
                words,
            )?;
            next_slot += c.slots();
            n_categorical += 1;
            Some(c)
        } else {
            None
        };
        slots_fit(next_slot)?;
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
            // The bounds of `bin <= k` and, inverted, of `u32::MAX - bin <= u32::MAX -
            // 1 - k` for `bin > k`.
            words[s as usize] = [
                (k as u64 + u64::from(BOUND_BIAS)) | left,
                u64::from(u32::MAX - 1 - k as u32 + BOUND_BIAS) | inverted,
            ];
        }
        let b = FeatureBins {
            feature: feature as usize,
            thresholds: range,
            slot: numeric,
            categories: category_bins,
        };
        if config.is_varying(feature as usize) {
            varying.push(b);
        } else {
            constant.push(b);
        }
    }
    let Tables {
        mut defaults,
        offsets,
        exceptions,
        ..
    } = tables;
    defaults.resize(next_slot, 0);
    defaults[DUMMY_SLOT] = u32::MAX;
    // A shift is in a split's word 0 only where a word holds two fields or more.
    let shifts = words.iter().flatten().any(|&w| w >> SHIFT_AT & 31 != 0);
    let bins = BinMap {
        f32: threshold_type == ThresholdType::F32,
        n_features: config.n_features(),
        n_slots: next_slot,
        thresholds,
        books,
        defaults,
        offsets,
        exceptions,
        shifts,
        n_categorical,
        constant,
        varying,
    };
    Ok((bins, [n_sets, n_chains]))
}

/// The trees' layout nodes, in tree order, from the words of `bin_map`, and the dummy
/// at link `dummy`: each tree's nodes rewritten in place, its leaf values in the
/// model's exact-sum [`Plan`], its expected steps and its scratch needs.
fn encode_trees(
    input: &Input<'_>,
    source: LayoutSource,
    words: &[[u64; 2]],
    dummy: u32,
    bins: BinMap,
    [chain_sets, chains]: [usize; 2],
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
                    // Word 0 holds the shift and the bound.
                    shift: (w0 >> SHIFT_AT) as u32 & 31,
                    bound: w0 as u32,
                    invert: !heavy_left,
                };
            }
            nodes[at] = LayoutNode {
                w0,
                w1: u64::from(node_link(start + kinds.light[local as usize] as usize))
                    | mask << MASK_SHIFT,
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
    // Predicates by node are taken in the order varying_predicates numbered them,
    // the trees' in tree order and each tree's in heavy-first preorder: all of them.
    debug_assert!(
        !matches!(predicates, PredicateIds::ByNode(ids) if ids.len() != next_predicate),
        "the nodes take every predicate assigned by node"
    );
    debug_assert_eq!(node_link(nodes.len()), dummy);
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
            chains,
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
