//! Building the layout from the decoded forest, in one pass over the ordered trees.
//!
//! The bins, codebooks, chains and rank tables come from the forest's distinct
//! splits, and each split's node words for both heavy sides from those; the pass
//! then rewrites every decoded node in place as its layout node, and writes the leaf
//! values, in tree order, with each tree's expected steps and scratch needs. The
//! nodes stay where they were decoded, in input order: tree order only decides the
//! order of the tree starts, and so of the prefix groups and chunks, and of the leaf
//! values.

use std::collections::{BTreeSet, HashMap};

use super::chains::{self, Cover, Sets};
use super::{
    BOUND_BIAS, BinMap, CategoryBins, CategoryCodes, ChainPoint, DUMMY_SLOT, ExceptionSlots,
    FIRST_CATEGORY, FLAG_LEAF, FeatureBins, Layout, LayoutNode, MASK_SHIFT, MAX_PREDICATES,
    MAX_SLOTS, MISSING, NO_CATEGORY, Need, SHIFT_AT, SLOT_SHIFT, node_link,
};
use crate::LoadError;
use crate::config::WalkerConfig;
use crate::exact::Plan;
use crate::load::parse::decoded::{LayoutSource, Packed, Split, TreeNodes};
use crate::load::parse::tree::check_node_total;
use crate::load::predicates::PredicateIds;
use crate::model::{FLAG_CATEGORICAL, FLAG_VARYING, ThresholdType, category_list};
use crate::rules::{GroupSums, ItemLengths, StepSums, Steps, lanes, pack};

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
    /// Whether the model packs its rank fields, in research and the tests; else the
    /// packing rule decides ([`crate::rules::pack`]).
    pub(crate) pack: Option<bool>,
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
    exception_slots: ExceptionSlots,
    exception_words: Vec<u32>,
    slots: usize,
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
/// the work and the bytes follow the model's lists. Each distinct list's codes are
/// held once, after missing's, and its sets, with or without missing, listed or
/// complemented, are ranges of them.
struct HeavySets {
    /// The set of each categorical split side, in [`sides`] order.
    set_of: Vec<u32>,
    /// The distinct sets' ranges in `codes`, and whether each is complemented.
    ranges: Vec<(u32, u32)>,
    codes: Vec<u32>,
    complement: Vec<bool>,
    /// The distinct sets' codes, each set's counted in full, which orders the covers.
    listed: usize,
    /// Bytes of every split side's list, built on the way.
    list_bytes: u64,
}

impl HeavySets {
    fn sets(&self, domain: usize) -> Sets<'_> {
        Sets {
            domain,
            ranges: &self.ranges,
            codes: &self.codes,
            complement: &self.complement,
        }
    }
}

/// Each categorical split side of `f` that is heavy somewhere, `(split, side)`.
fn sides<'a>(f: &'a Feature<'_>) -> impl Iterator<Item = (u32, usize)> + 'a {
    f.ids
        .iter()
        .filter(move |&&s| f.splits[s as usize].is_categorical())
        .flat_map(move |&s| {
            (0..2)
                .filter(move |&side| f.splits[s as usize].sides >> side & 1 != 0)
                .map(move |side| (s, side))
        })
}

/// A split's categories as decoded, ascending: an inline word's bits, or a list in
/// the pool.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum RawList<'a> {
    Inline(u32),
    Pool(&'a [u32]),
}

impl RawList<'_> {
    fn of<'a>(split: &Split, categories: &'a [u32]) -> RawList<'a> {
        if split.inline_cat() {
            RawList::Inline(split.value as u32)
        } else {
            RawList::Pool(category_list(categories, split.value as u32))
        }
    }

    fn iter(self) -> impl Iterator<Item = u32> {
        let (mut word, mut list) = match self {
            Self::Inline(word) => (word, &[][..]),
            Self::Pool(list) => (0, list),
        };
        std::iter::from_fn(move || {
            if word != 0 {
                let c = word.trailing_zeros();
                word &= word - 1;
                Some(c)
            } else {
                let (&first, rest) = list.split_first()?;
                list = rest;
                Some(first)
            }
        })
    }

    const fn len(self) -> usize {
        match self {
            Self::Inline(word) => word.count_ones() as usize,
            Self::Pool(list) => list.len(),
        }
    }

    /// The largest category, or 0.
    fn max(self) -> u32 {
        match self {
            Self::Inline(word) => word.checked_ilog2().unwrap_or(0),
            Self::Pool(list) => list.last().copied().unwrap_or(0),
        }
    }
}

/// A table by category, or the distinct categories' bits, is used only when it has
/// at most this many entries per list entry it replaces; else a search and a sort.
const DENSE_SPREAD: usize = 8;

/// The most entries of the category-to-code table: 8 MiB.
const MAX_DENSE_CODES: usize = 1 << 21;

/// The distinct heavy sets of the categorical splits of `f`.
fn heavy_sets(f: &Feature<'_>) -> Result<HeavySets, LoadError> {
    let domain = f.codes.domain();
    let book = f.codes.book(f.books);
    let table = book.last().and_then(|&max| {
        let max = max as usize;
        if max >= MAX_DENSE_CODES || max >= book.len().saturating_mul(DENSE_SPREAD) {
            return None;
        }
        let mut table = vec![NO_CATEGORY; max + 1];
        for (i, &c) in book.iter().enumerate() {
            table[c as usize] = i as u32 + FIRST_CATEGORY;
        }
        Some(table)
    });
    let code_of = |c: u32| {
        table.as_ref().map_or_else(
            || f.codes.of(f.books, c),
            |table| table.get(c as usize).copied().unwrap_or(NO_CATEGORY),
        )
    };
    // Each split side's list, whether its set takes missing, and whether it is
    // complemented.
    let sides: Vec<(RawList<'_>, bool, bool)> = sides(f)
        .map(|(s, side)| {
            let split = &f.splits[s as usize];
            let list = RawList::of(split, f.categories);
            (list, split.default_left(), side == 1)
        })
        .collect();
    // The cap counts each side's list as encoded, with room for missing's code.
    let mut list_bytes = 0u64;
    for &(list, missing, _) in &sides {
        if list_bytes + (list.len() as u64 + 1) * 4 > f.max_bytes {
            return Err(too_big("a feature's categorical heavy sets", f.max_bytes));
        }
        list_bytes += (list.len() + usize::from(missing)) as u64 * 4;
    }
    // The distinct sets, by size, then listed before complemented, then by list:
    // sorted by a key of the size, the kind and the first code, then by the lists.
    let key = |i: usize| {
        let (list, missing, complement) = sides[i];
        let len = list.len() + usize::from(missing);
        let size = if complement { domain - len } else { len };
        let first = if missing {
            MISSING
        } else {
            list.iter().next().map_or(0, code_of)
        };
        (size as u64) << 33 | u64::from(complement) << 32 | u64::from(first)
    };
    let mut by_size: Vec<(u64, u32)> = (0..sides.len()).map(|i| (key(i), i as u32)).collect();
    by_size.sort_unstable_by(|&(ka, a), &(kb, b)| {
        // Equal keys take missing alike; the codebook keeps the categories' order.
        ka.cmp(&kb)
            .then_with(|| sides[a as usize].0.iter().cmp(sides[b as usize].0.iter()))
    });
    let (mut ranges, mut codes, mut complement) = (Vec::new(), Vec::new(), Vec::new());
    let mut held: HashMap<RawList<'_>, (u32, u32)> = HashMap::default();
    let mut set_of = vec![0u32; sides.len()];
    let (mut last, mut listed) = (None, 0);
    for &(k, i) in &by_size {
        let i = i as usize;
        let (list, missing, inverted) = sides[i];
        if last != Some((k, list)) {
            let (start, end) = *held.entry(list).or_insert_with(|| {
                let start = codes.len();
                codes.push(MISSING);
                codes.extend(list.iter().map(|c| {
                    let code = code_of(c);
                    debug_assert!(
                        code >= FIRST_CATEGORY,
                        "a split's categories are in its codebook"
                    );
                    code
                }));
                debug_assert!(
                    codes[start..].is_sorted_by(|a, b| a < b),
                    "a heavy set's codes ascend"
                );
                (start as u32, codes.len() as u32)
            });
            ranges.push((start + u32::from(!missing), end));
            listed += list.len() + usize::from(missing);
            complement.push(inverted);
            last = Some((k, list));
        }
        set_of[i] = (complement.len() - 1) as u32;
    }
    // The sets stay until the second pass, so they are trimmed.
    drop(by_size);
    drop(held);
    drop(table);
    codes.shrink_to_fit();
    ranges.shrink_to_fit();
    Ok(HeavySets {
        set_of,
        ranges,
        codes,
        complement,
        listed,
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

/// The chains of `fields` in word order, each word's by their lowest bit.
fn by_word(fields: &[Field]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..fields.len() as u32).collect();
    order.sort_by_key(|&c| (fields[c as usize].word, fields[c as usize].low));
    order
}

/// The exceptions of `chain`, a chain of `sets`' listed sets then its complements:
/// each code whose rank differs from the default, the number of listed sets, with its
/// rank, to `found`. A complement, which holds no category, is in no listed set, so a
/// code's rank differs only where a set of the chain lists the code: at the first
/// listed set that holds it, or past the last complement whose list holds it. A code
/// has one rank a chain, so one exception. With the listed sets.
fn chain_exceptions(chain: &[u32], sets: &Sets<'_>, mut found: impl FnMut(u32, u32)) -> usize {
    let listed = chain
        .iter()
        .take_while(|&&i| !sets.complement[i as usize])
        .count();
    for (k, &i) in chain.iter().enumerate() {
        let set = sets.list(i as usize);
        let (other, rank) = if k < listed {
            let before = k.checked_sub(1).map(|j| chain[j]);
            (before, k as u32)
        } else {
            (chain.get(k + 1).copied(), k as u32 + 1)
        };
        let other = other.map_or(&[][..], |j| sets.list(j as usize));
        for e in chains::difference(set, other) {
            found(e, rank);
        }
    }
    listed
}

/// The exceptions a feature's fill saves a code when its chains of `cover` pack: a
/// code's exceptions, one a chain, less the packed words they fall in, weighted by the
/// code's exceptions ([`crate::rules::pack`]).
fn exceptions_saved(heavy: &HeavySets, cover: &Cover, domain: usize) -> f64 {
    let sets = heavy.sets(domain);
    let (fields, _) = pack_fields(cover, true);
    let (mut chains, mut words) = (vec![0u32; domain], vec![0u32; domain]);
    let mut last = vec![u32::MAX; domain];
    for c in by_word(&fields) {
        let word = fields[c as usize].word;
        chain_exceptions(cover.chain(c as usize), &sets, |e, _| {
            let e = e as usize;
            chains[e] += 1;
            words[e] += u32::from(last[e] != word);
            last[e] = word;
        });
    }
    let total: f64 = chains.iter().map(|&x| f64::from(x)).sum();
    if total == 0.0 {
        return 0.0;
    }
    let weighted: f64 = chains
        .iter()
        .zip(&words)
        .map(|(&x, &y)| f64::from(x) * f64::from(x - y))
        .sum();
    weighted / total
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
    let (fields, n_words) = pack_fields(cover, pack);

    // Each split side's word, field and position: its bound puts the field at the top,
    // `rank <= k` exactly when the shifted word is at most ((k + 1) << (32 - b)) - 1,
    // whatever the fields below it hold.
    for ((s, side), &i) in sides(f).zip(&heavy.set_of) {
        let (chain, k) = cover.at[i as usize];
        let Field { word, low, width } = fields[chain as usize];
        let bound = ((u64::from(k) + 1) << (u32::BITS - width)) - 1 + u64::from(BOUND_BIAS);
        let shift = u32::BITS - width - low;
        words[s as usize][side] = bound
            | u64::from(shift) << SHIFT_AT
            | ((f.slot as u64 + u64::from(word)) << SLOT_SHIFT);
    }

    // Chains are visited in word order, keeping each code's ranks for a word together.
    // Count one exception per (code, word), then write it in place by code,
    // starting from the word's default and replacing its exceptional rank fields.
    tables.defaults.resize(f.slot + n_words, 0);
    let order = by_word(&fields);
    let mut bounds = vec![0u32; domain + 1];
    let mut last = vec![u32::MAX; domain];
    for &c in &order {
        let Field { word, low, .. } = fields[c as usize];
        let listed = chain_exceptions(cover.chain(c as usize), &sets, |e, _| {
            let e = e as usize;
            if last[e] != word {
                last[e] = word;
                bounds[e + 1] += 1;
            }
        });
        tables.defaults[f.slot + word as usize] |= (listed as u32) << low;
    }
    tables.slots += n_words;
    // The no-category code is in no list, so it has no exceptions and fresh bins
    // hold its words, as [`BinMap::fill_with`] needs.
    debug_assert_eq!(bounds[NO_CATEGORY as usize + 1], 0);
    let too_many = || LoadError::Limit("the rank tables exceed u32 exceptions".into());
    let base = tables.exception_words.len();
    bounds[0] = base as u32;
    for e in 0..domain {
        bounds[e + 1] = bounds[e + 1].checked_add(bounds[e]).ok_or_else(too_many)?;
    }
    // The tables are checked before they grow. The exceptions are at most the heavy
    // sets' listed codes, which the cap bounded.
    let end = bounds[domain] as usize;
    if 4 * (tables.slots as u64 + tables.offsets.len() as u64 + domain as u64 + 1 + end as u64)
        + tables.exception_slots.bytes_after(end, n_words)
        > f.max_bytes
    {
        return Err(too_big("the categorical splits' rank tables", f.max_bytes));
    }
    let offsets = tables.offsets.len();
    LoadError::reserve(tables.offsets.try_reserve_exact(domain + 1))?;
    tables.offsets.extend_from_slice(&bounds);
    tables.exception_slots.resize(end, n_words)?;
    LoadError::reserve(tables.exception_words.try_reserve_exact(end - base))?;
    tables.exception_words.resize(end, 0);
    let defaults = &tables.defaults;
    let exception_slots = &mut tables.exception_slots;
    let exception_words = &mut tables.exception_words;
    let mut next = bounds;
    last.fill(u32::MAX);
    for &c in &order {
        let Field { word, low, width } = fields[c as usize];
        let slot = f.slot + word as usize;
        let mask = !((u32::MAX >> (u32::BITS - width)) << low);
        chain_exceptions(cover.chain(c as usize), &sets, |e, rank| {
            let e = e as usize;
            if last[e] != word {
                last[e] = word;
                exception_slots.set(next[e] as usize, word);
                exception_words[next[e] as usize] = defaults[slot];
                next[e] += 1;
            }
            let value = &mut exception_words[next[e] as usize - 1];
            *value = *value & mask | rank << low;
        });
    }
    Ok(CategoryBins {
        codes: f.codes,
        slot: f.slot,
        slots: n_words,
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
        let (bins, sets, words) = bin_map(input, &source.splits, |max| {
            group_lane_slots(input, &source, max)
        })?;
        Ok(encode_trees(input, source, &words, dummy, bins, sets))
    }
}

/// The expected lane slots of a group of the configuration's widest groups, for the
/// packing rule: a pass over the trees, which stops once their steps pass `max`, past
/// which the rule doesn't pack.
fn group_lane_slots(input: &Input<'_>, source: &LayoutSource, max: f64) -> f64 {
    let width = input.config.max_group_width();
    let constant: Vec<bool> = source.splits.iter().map(|s| !s.is_varying()).collect();
    let mut sums = GroupSums::default();
    for tree in &source.trees {
        let start = tree.start as usize;
        let nodes = TreeNodes::new(&source.nodes[start..start + tree.count as usize]);
        let splits = nodes.iter().map(|node| {
            (!node.is_leaf()).then(|| (node.heavy_probability(), constant[node.id() as usize]))
        });
        sums.tree(width, splits);
        if sums.group >= max {
            return sums.group;
        }
    }
    sums.lane_slots(lanes(width > 1, false))
}

/// Each split's word 0 by its heavy side (left, right).
type SplitWords = Vec<[u64; 2]>;

/// A categorical feature's codes, distinct heavy sets and their cover, between the
/// two passes of [`bin_map`].
struct Covered {
    codes: CategoryCodes,
    heavy: HeavySets,
    cover: Option<Cover>,
}

/// The bins, codebooks and rank tables of the forest's distinct `splits`, flat, each
/// feature's slots in feature order; with the number of distinct heavy sets and of
/// chains, and each split's word 0 by its heavy side (left, right).
///
/// A first pass takes each feature's thresholds, codebook, heavy sets and cover; the
/// packing rule ([`crate::rules::pack`]) then decides whether the chains' rank fields
/// pack ("Packed fields"), from the bins' size, the fills' exceptions, and the
/// groups' expected `lane_slots`, which it takes only when the first two leave the
/// choice open, and only up to the most at which packing pays; and the second pass
/// lays the slots out.
fn bin_map(
    input: &Input<'_>,
    splits: &[Split],
    lane_slots: impl FnOnce(f64) -> f64,
) -> Result<(BinMap, [usize; 2], SplitWords), LoadError> {
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
    let (mut t, mut cats) = (Vec::new(), Vec::new());
    // Each feature's thresholds and, if categorical, its cover; with the heavy sets'
    // lists so far, which the cap bounds.
    let mut first: Vec<(std::ops::Range<usize>, Option<Covered>)> = Vec::new();
    let (mut n_chains, mut n_sets, mut list_bytes) = (0, 0, 0);
    for ids in features() {
        t.clear();
        let mut categorical = false;
        for &s in ids {
            let split = &splits[s as usize];
            if split.is_categorical() {
                categorical = true;
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
            cats.clear();
            // Dense categories take bits up to the largest; sparse lists are
            // deduplicated as they grow, since splits can repeat their categories.
            let lists = || {
                ids.iter()
                    .map(|&s| &splits[s as usize])
                    .filter(|split| split.is_categorical())
                    .map(|split| RawList::of(split, categories))
            };
            let entries: usize = lists().map(RawList::len).sum();
            let words = lists()
                .map(RawList::max)
                .max()
                .map_or(0, |max| max as usize / 64 + 1);
            if words <= entries.saturating_mul(DENSE_SPREAD).div_ceil(64)
                && words <= MAX_DENSE_CODES / 2
            {
                let mut bits = vec![0u64; words];
                for c in lists().flat_map(RawList::iter) {
                    bits[c as usize / 64] |= 1 << (c % 64);
                }
                for (i, mut word) in bits.into_iter().enumerate() {
                    while word != 0 {
                        cats.push((i * 64) as u32 + word.trailing_zeros());
                        word &= word - 1;
                    }
                }
            } else {
                let mut distinct = 0;
                for list in lists() {
                    cats.extend(list.iter());
                    if cats.len() > 2 * distinct + 4096 {
                        cats.sort_unstable();
                        cats.dedup();
                        distinct = cats.len();
                    }
                }
                cats.sort_unstable();
                cats.dedup();
            }
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
            let heavy = heavy_sets(&feature)?;
            list_bytes += heavy.list_bytes;
            if list_bytes > max_rank_bytes {
                return Err(too_big(
                    "the features' categorical heavy sets",
                    max_rank_bytes,
                ));
            }
            Some(Covered {
                codes,
                heavy,
                cover: None,
            })
        } else {
            None
        };
        first.push((start..thresholds.len(), covered));
    }
    // The first pass's scratch, freed before the covers and the rank tables.
    drop(cats);
    if thresholds.len() >= u32::MAX as usize {
        return Err(LoadError::Limit("threshold count exceeds u32".into()));
    }
    // The covers, the features of the fewest list entries first, sharing their steps.
    let size = |c: &Covered| c.heavy.listed + c.heavy.complement.len();
    let mut by_size: Vec<&mut Covered> = first.iter_mut().filter_map(|f| f.1.as_mut()).collect();
    by_size.sort_by_key(|c| size(c));
    let (mut pool, mut minimum) = (chains::SHARED_WORK, true);
    for c in by_size {
        let cover = chains::cover(&c.heavy.sets(c.codes.domain()), &mut pool);
        n_chains += cover.chains();
        n_sets += c.heavy.complement.len();
        minimum &= cover.minimum;
        c.cover = Some(cover);
    }
    let pack = input.pack.unwrap_or_else(|| {
        let width = config.max_group_width();
        let numeric = first.iter().filter(|(range, _)| !range.is_empty()).count();
        let unpacked = DUMMY_SLOT + 1 + 4 * numeric + n_chains;
        if n_chains == 0 {
            return false;
        }
        // Covers cut short may keep extra chains, which packed take fields, not slots.
        if !minimum || !pack::resident_here(4 * unpacked) {
            return true;
        }
        // A group fills its constant features; a single row every one.
        let saved: f64 = features()
            .zip(&first)
            .filter_map(|(ids, (_, covered))| {
                let feature = splits[ids[0] as usize].feature as usize;
                let c = covered
                    .as_ref()
                    .filter(|_| width == 1 || !config.is_varying(feature))?;
                let cover = c.cover.as_ref().expect("every categorical feature's cover");
                Some(exceptions_saved(&c.heavy, cover, c.codes.domain()))
            })
            .sum();
        let max = pack::max_lane_slots(saved, width > 1);
        saved > 0.0 && lane_slots(max) < max
    });

    let mut words = vec![[0u64; 2]; splits.len()];
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
                cover.as_ref().expect("every categorical feature's cover"),
                pack,
                &mut tables,
                &mut words,
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
        exception_slots,
        exception_words,
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
        exception_slots,
        exception_words: super::array(exception_words),
        shifts,
        n_categorical,
        constant,
        varying,
    };
    Ok((bins, [n_sets, n_chains], words))
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
    let mut leaves = Vec::with_capacity(match plan {
        Plan::Fixed(_) => 0,
        Plan::Wide(_) => trees.iter().map(|t| t.leaves as usize).sum(),
    });
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
            nodes[at] = match plan {
                Plan::Fixed(scale) => LayoutNode::fixed(crate::exact::FixedLeaf::new(value, scale)),
                Plan::Wide(wide) => {
                    leaves.push(wide.leaf(value));
                    let leaf = (leaves.len() - 1) as u64;
                    LayoutNode {
                        w0: leaf,
                        w1: leaf | FLAG_LEAF,
                    }
                }
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
            nodes: super::array(nodes),
            bins,
            leaves,
            varying_share,
            chain_sets: chain_sets as u32,
            chains: chains as u32,
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
