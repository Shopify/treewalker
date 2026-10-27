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
    BIT_K, BOUND_BIAS, BinMap, CategoryBins, CategoryCodes, ChainPoint, DUMMY_SLOT, FIRST_CATEGORY,
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
    /// Constant categorical features whose splits have more distinct heavy sets take
    /// bits, when their tables fit ([`super::CATEGORY_BITS_ABOVE`] but in research).
    pub(crate) category_bits_above: usize,
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
    /// Whether it varies, and so keeps chains, and above how many distinct heavy sets
    /// it takes bits otherwise; and whether a constant feature of the model has more,
    /// beside which it takes bits that take no more bytes than its chains can.
    varying: bool,
    bits_above: usize,
    beside_bits: bool,
    /// The cap on the rank tables' bytes, which also bounds the heavy sets and their
    /// inclusion graph, built on the way.
    max_bytes: u64,
}

/// Whether the categorical splits among `ids` have more than `above` distinct heavy
/// sets, as [`heavy_sets`] counts them: the split sides bound them, and past `above`
/// each side's set, its categories, default and side, is hashed and the distinct
/// hashes counted. A collision can only count fewer, which chooses chains beside a
/// feature that still takes bits: it changes the layout, never a prediction.
fn more_sets_than(
    ids: &[u32],
    splits: &[Split],
    categories: &[u32],
    above: usize,
    members: &mut Vec<u32>,
) -> bool {
    let categorical = || {
        ids.iter()
            .map(|&s| &splits[s as usize])
            .filter(|s| s.is_categorical())
    };
    let sides: usize = categorical()
        .map(|s| (s.sides & 3).count_ones() as usize)
        .sum();
    if sides <= above {
        return false;
    }
    let mut hashes = Vec::with_capacity(sides);
    for split in categorical() {
        split_categories(split, categories, members);
        // FNV-1a over the length, the categories and the default.
        let list = [members.len() as u32, u32::from(split.default_left())];
        let h = list
            .iter()
            .chain(members.iter())
            .fold(0xcbf2_9ce4_8422_2325_u64, |h, &x| {
                (h ^ u64::from(x)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        for side in 0..2 {
            if split.sides >> side & 1 != 0 {
                hashes.push(h ^ side);
            }
        }
    }
    hashes.sort_unstable();
    hashes.dedup();
    hashes.len() > above
}

/// The categorical features' rank tables, flat: [`BinMap`]'s defaults, offsets,
/// exceptions and bits, and the chain slots so far.
#[derive(Default)]
struct Tables {
    defaults: Vec<u32>,
    offsets: Vec<u32>,
    exceptions: Vec<[u32; 2]>,
    bits: Vec<u32>,
    chains: usize,
}

impl Tables {
    /// Their bytes so far, as [`BinMap::rank_table_bytes`] counts them.
    const fn bytes(&self) -> u64 {
        4 * (self.chains + self.offsets.len() + self.bits.len()) as u64
            + 8 * self.exceptions.len() as u64
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

/// The slots of one categorical feature: a minimum chain cover of its splits'
/// distinct heavy sets ([`heavy_sets`]), each categorical split's words by heavy side
/// into `words`, and the feature's defaults and exceptions appended to `tables`; or
/// its bits ([`bit_slots`]) when it is constant and has more than `f.bits_above`
/// sets, or beside such a feature, bits that take no more bytes than its chains can;
/// with the number of distinct heavy sets.
fn category_slots(
    f: &Feature<'_>,
    tables: &mut Tables,
    words: &mut [[u64; 2]],
    members: &mut Vec<u32>,
) -> Result<(CategoryBins, usize), LoadError> {
    let domain = f.codes.domain();
    let heavy = heavy_sets(f, members)?;
    let (owners, set_of, complement) = (&heavy.owners, &heavy.set_of, &heavy.complement);
    let sets = heavy.sets(domain);
    // Chains take at most a slot and a default per set, a bound per code and one more,
    // and an exception per listed code of each set.
    let n = sets.len();
    let bit_bytes = 4 * domain as u64 * n.div_ceil(32) as u64;
    let chain_bytes = 4 * (n + domain + 1) as u64 + 8 * sets.codes.len() as u64;
    if !f.varying
        && (n > f.bits_above || f.beside_bits && bit_bytes <= chain_bytes)
        && let Some(c) = bit_slots(f, &sets, owners, set_of, tables, words)?
    {
        return Ok((c, n));
    }
    let cover = chains::cover(&sets, f.max_bytes - heavy.list_bytes)?;

    // Each split side's chain slot and position, as its bound.
    for (&(s, side), &i) in owners.iter().zip(set_of) {
        let (chain, k) = cover.at[i as usize];
        words[s as usize][side] =
            u64::from(k + BOUND_BIAS) | ((f.slot as u64 + u64::from(chain)) << SLOT_SHIFT);
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
    let bytes = tables.bytes() + 4 * (domain + 1) as u64 + 8 * found.len() as u64;
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
            slots: m,
            bits: false,
            offsets,
            held: f.held,
        },
        sets.len(),
    ))
}

/// The bit slots of one constant categorical feature whose distinct heavy sets are
/// `sets`, `set_of` each split side's of `owners`: a slot per 32 sets, each code's
/// row appended to `tables.bits`, set j's bit set where the code is not in it, no
/// category's row as the slots' defaults, and each split side's word into `words`.
/// `None`, with nothing written, when the rows would pass the cap with the tables
/// so far.
fn bit_slots(
    f: &Feature<'_>,
    sets: &Sets<'_>,
    owners: &[(u32, usize)],
    set_of: &[u32],
    tables: &mut Tables,
    words: &mut [[u64; 2]],
) -> Result<Option<CategoryBins>, LoadError> {
    let (domain, slots) = (f.codes.domain(), sets.len().div_ceil(32));
    let cells = match usize::try_from(domain as u64 * slots as u64) {
        Ok(cells) if tables.bytes() + 4 * cells as u64 <= f.max_bytes => cells,
        _ => return Ok(None),
    };
    // No category is in no list: out of a listed set and in a complement.
    let mut none = vec![0u32; slots];
    for (j, &complement) in sets.complement.iter().enumerate() {
        none[j / 32] |= u32::from(!complement) << (j % 32);
    }
    let offsets = tables.bits.len();
    LoadError::reserve(tables.bits.try_reserve_exact(cells))?;
    for _ in 0..domain {
        tables.bits.extend_from_slice(&none);
    }
    // A listed code is on the other side of each of its sets.
    let rows = &mut tables.bits[offsets..];
    for j in 0..sets.len() {
        for &e in sets.list(j) {
            rows[e as usize * slots + j / 32] ^= 1 << (j % 32);
        }
    }
    tables.defaults.resize(f.slot, 0);
    tables.defaults.extend_from_slice(&none);
    // Set j's bit, shifted to the top, is clear where the code is in it.
    for (&(s, side), &j) in owners.iter().zip(set_of) {
        let slot = (f.slot + j as usize / 32) as u64;
        words[s as usize][side] =
            u64::from(BIT_K + BOUND_BIAS) | u64::from(31 - j % 32) << SHIFT_AT | slot << SLOT_SHIFT;
    }
    Ok(Some(CategoryBins {
        codes: f.codes,
        slot: f.slot,
        slots,
        bits: true,
        offsets,
        held: f.held,
    }))
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
        let mut took = Took::default();
        let built = bin_map(input, &source.splits, &mut words, Bits::Beside, &mut took);
        let (bins, sets) = if took.bits {
            bits_or_chains(input, &source.splits, &mut words, built, took)?
        } else {
            built?
        };
        Ok(encode_trees(input, source, &words, dummy, bins, sets))
    }
}

/// Which constant categorical features may take bits ("Category bits").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bits {
    /// Those past the threshold, and, if one is, those whose bits take no more bytes
    /// than their chains can.
    Beside,
    /// Those past the threshold.
    Above,
    /// None: chains everywhere, as without bits.
    None,
}

/// What [`bin_map`] gave bits to, even if a later feature failed: any feature, and one
/// that took them only beside a feature past the threshold.
#[derive(Clone, Copy, Debug, Default)]
struct Took {
    bits: bool,
    beside: bool,
}

/// The bins when a feature took bits: `first`, the bins built with them, unless they
/// can leave too little of the cap to the features after the bits or to the varying
/// features' mask flips, which [`MaskPlan`](super::MaskPlan) counts after the layout.
/// Then the bins are built again with bits only past the threshold (if a feature took
/// them beside it) and with chains everywhere, as without bits, until a layout's
/// tables and the flips' bound fit; of the layouts built, the one whose tables take
/// the fewest bytes is kept, the earliest, with the most bits, on ties. The flips are
/// the same in every layout, so a model that loads with any of them loads.
fn bits_or_chains(
    input: &Input<'_>,
    splits: &[Split],
    words: &mut [[u64; 2]],
    first: Result<(BinMap, [usize; 2]), LoadError>,
    took: Took,
) -> Result<(BinMap, [usize; 2]), LoadError> {
    let tables = |b: &BinMap| b.rank_table_bytes() as u64;
    let fits = |b: &BinMap| tables(b) + b.flip_bytes_bound() <= input.max_rank_bytes;
    let mut errors = Vec::new();
    // The smallest layout so far, and its words unless they are `words`.
    let mut best = match first {
        Ok(b) if fits(&b.0) => return Ok(b),
        Ok(b) => Some((b, None)),
        Err(LoadError::Limit(e)) => {
            errors.push(format!("with category bits, {e}"));
            None
        }
        Err(e) => return Err(e),
    };
    let fallbacks = [
        (Bits::Above, "with bits only past the threshold"),
        (Bits::None, "with chains everywhere"),
    ];
    for (bits, what) in &fallbacks[usize::from(!took.beside)..] {
        let mut w = vec![[0u64; 2]; splits.len()];
        match bin_map(input, splits, &mut w, *bits, &mut Took::default()) {
            Ok(b) => {
                // A layout that fits takes fewer bytes than every one before it,
                // which did not.
                let done = fits(&b.0);
                if best
                    .as_ref()
                    .is_none_or(|(c, _)| tables(&b.0) < tables(&c.0))
                {
                    best = Some((b, Some(w)));
                }
                if done {
                    break;
                }
            }
            Err(LoadError::Limit(e)) => errors.push(format!("{what}, {e}")),
            Err(e) => return Err(e),
        }
    }
    match best {
        Some((b, w)) => {
            if let Some(w) = w {
                words.copy_from_slice(&w);
            }
            Ok(b)
        }
        None => Err(LoadError::Limit(errors.join("; "))),
    }
}

/// The bins, codebooks and rank tables of the forest's distinct `splits`, flat, each
/// feature's slots in feature order, and each split's word 0 by its heavy side
/// (left, right) into `words`; with the number of distinct heavy sets in chains and
/// in bits. The constant categorical features `bits` allows may take bits; `took`
/// records which did, even if a later feature fails.
fn bin_map(
    input: &Input<'_>,
    splits: &[Split],
    words: &mut [[u64; 2]],
    bits: Bits,
    took: &mut Took,
) -> Result<(BinMap, [usize; 2]), LoadError> {
    let Input {
        categories,
        config,
        threshold_type,
        max_rank_bytes,
        ..
    } = *input;
    let category_bits_above = if bits == Bits::None {
        usize::MAX
    } else {
        input.category_bits_above
    };
    // Splits by feature, in feature order.
    let mut by_feature: Vec<u32> = (0..splits.len() as u32).collect();
    by_feature.sort_unstable_by_key(|&s| splits[s as usize].feature);
    let features =
        || by_feature.chunk_by(|&a, &b| splits[a as usize].feature == splits[b as usize].feature);

    let (mut thresholds, mut books, mut tables) = (Vec::new(), Vec::new(), Tables::default());
    let (mut constant, mut varying) = (Vec::new(), Vec::new());
    let (mut t, mut cats, mut members) = (Vec::new(), Vec::new(), Vec::new());
    // Whether a constant feature has more sets than the threshold, known before any
    // feature is laid out, as the features beside it may take bits too.
    let beside_bits = bits == Bits::Beside
        && features().any(|ids| {
            let feature = splits[ids[0] as usize].feature as usize;
            !config.is_varying(feature)
                && more_sets_than(ids, splits, categories, category_bits_above, &mut members)
        });
    let mut next_slot = DUMMY_SLOT + 1;
    let (mut set_counts, mut n_categorical) = ([0, 0], 0);
    for ids in features() {
        let feature = splits[ids[0] as usize].feature;
        let varies = config.is_varying(feature as usize);
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
            let (c, sets) = category_slots(
                &Feature {
                    codes,
                    books: &books,
                    ids,
                    splits,
                    categories,
                    slot: next_slot,
                    held: n_categorical,
                    varying: varies,
                    bits_above: category_bits_above,
                    beside_bits,
                    max_bytes: max_rank_bytes,
                },
                &mut tables,
                words,
                &mut members,
            )?;
            next_slot += c.slots;
            n_categorical += 1;
            set_counts[usize::from(c.bits)] += sets;
            took.bits |= c.bits;
            took.beside |= c.bits && sets <= category_bits_above;
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
            // The bounds of `bin <= k` and, inverted, of `u32::MAX - bin <= u32::MAX -
            // 1 - k` for `bin > k`.
            words[s as usize] = [
                (k as u64 + u64::from(BOUND_BIAS)) | left,
                u64::from(u32::MAX - 1 - k as u32 + BOUND_BIAS) | inverted,
            ];
        }
        let b = FeatureBins {
            feature: feature as usize,
            thresholds: start..thresholds.len(),
            slot: numeric,
            categories: category_bins,
        };
        if varies {
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
        bits,
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
        bits,
        n_categorical,
        constant,
        varying,
    };
    Ok((bins, set_counts))
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
    [chain_sets, bit_sets]: [usize; 2],
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
                    // Word 0 holds the bound.
                    k: w0 as u32 - BOUND_BIAS,
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
            bit_sets,
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
