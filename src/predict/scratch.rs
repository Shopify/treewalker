//! A predictor's scratch, allocated once when it is created: the masks, sums and walk
//! buffers for the configuration's widest group, and the chunks of trees whose passes'
//! working set fits the chunk budget at each mask width.

use super::mask::rows::{self, Unit};
use super::mask::{Mask, Wide};
use super::precompute::SWEEP_TURNS;
use super::resolve::{Carrier, Handle, Slot};
use super::walk::{Item, MAX_PAD};
use crate::exact::{self, Plan, WIDE_SUMS};
use crate::load::layout::Need;
use crate::model::Model;
use crate::rules::Tuning;

/// Rows up to which groups wider than 64 rows take `Wide` masks, whose width is a
/// type: those of the widest, `Wide<16>`. Wider groups take runtime-width masks,
/// whose loops pay a counter, branches and scans that a width fixed at compile time
/// unrolls away; one per group replaces running wide groups in pieces of `Wide<16>`,
/// which walked every tree once per piece.
pub(super) const WIDE_ROWS: usize = 1024;

/// A mask width class, 2^k bits for k of 5 and up, narrowest first: the masks of a
/// group of up to that many rows. Scratch and chunks are sized per class; a
/// runtime-width mask is at most its class's bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Width(u32);

impl Width {
    pub(super) const W32: Self = Self(5);
    pub(super) const W64: Self = Self(6);
    pub(super) const W128: Self = Self(7);
    pub(super) const W256: Self = Self(8);
    pub(super) const W512: Self = Self(9);
    pub(super) const W1024: Self = Self(10);

    pub(super) const fn of(rows: usize) -> Self {
        if rows <= 32 {
            Self::W32
        } else {
            Self(usize::BITS - (rows - 1).leading_zeros())
        }
    }

    /// Every class up to `self`, narrowest first.
    pub(super) fn up_to(self) -> impl Iterator<Item = Self> {
        (Self::W32.0..=self.0).map(Self)
    }

    /// The class's position in [`Self::up_to`].
    pub(super) const fn index(self) -> usize {
        (self.0 - Self::W32.0) as usize
    }

    pub(super) const fn bytes(self) -> usize {
        1 << (self.0 - 3)
    }

    /// Arena bytes per slot: the mask, and for runtime-width masks a header unit,
    /// which holds the mask's row span. Chunks budget the class's mask bytes alone,
    /// which a header and mask pass by at most a sixteenth, at 2,048 rows.
    pub(super) const fn slot_bytes(self) -> usize {
        if self.runtime() {
            self.bytes() + size_of::<Unit>()
        } else {
            self.bytes()
        }
    }

    /// Whether items carry a handle into the arena rather than the mask.
    pub(super) const fn handles(self) -> bool {
        self.0 >= Self::W128.0
    }

    /// Whether the class's masks have a runtime number of units.
    pub(super) const fn runtime(self) -> bool {
        self.0 > WIDE_ROWS.trailing_zeros()
    }

    /// Scratch bytes per leaf at this width: two walk queues and the leaf entries of
    /// items, a stop's node and word 1, and with fixed-width handles the stop's place
    /// in the list of varying stops that resolve splits.
    pub(super) const fn leaf_bytes(self) -> usize {
        let item = if self.0 == Self::W64.0 {
            size_of::<Item<u64>>()
        } else {
            size_of::<Item<u32>>()
        };
        let listed = if self.handles() && !self.runtime() {
            size_of::<u32>()
        } else {
            0
        };
        3 * item + size_of::<u32>() + size_of::<u64>() + listed
    }
}

/// Bytes of each varying mask of a group of `rows` rows.
const fn mask_bytes(rows: usize) -> usize {
    if rows > WIDE_ROWS {
        rows::units(rows) * size_of::<Unit>()
    } else {
        Width::of(rows).bytes()
    }
}

/// Mask types a [`Unit`] buffer may be viewed as: words without padding, valid for
/// any bits, aligned to at most 16 bytes, and dividing or multiples of 16 bytes.
///
/// # Safety
///
/// Implementors must be such types.
pub(super) unsafe trait Words: Copy {}

// SAFETY: plain integers.
unsafe impl Words for u32 {}

// SAFETY: plain integers.
unsafe impl Words for u64 {}

// SAFETY: arrays of u64, 16-byte aligned, 16 to 128 bytes.
unsafe impl<const W: usize> Words for Wide<W> {}

/// `units` as a slice of `T`.
pub(super) const fn view<T: Words>(units: &mut [Unit]) -> &mut [T] {
    const {
        assert!(align_of::<T>() <= align_of::<Unit>());
        assert!(
            size_of::<Unit>().is_multiple_of(size_of::<T>())
                || size_of::<T>().is_multiple_of(size_of::<Unit>())
        );
    };
    let len = size_of_val(units) / size_of::<T>();
    // SAFETY: T is aligned to at most a unit and valid for any bits, and len values
    // of T fit the units' bytes.
    unsafe { std::slice::from_raw_parts_mut(units.as_mut_ptr().cast::<T>(), len) }
}

/// A chunk of trees and what its passes need at most.
#[derive(Clone, Copy, Debug)]
pub(super) struct Chunk {
    pub(super) trees: (usize, usize),
    /// Leaves in the chunk: a bound on the items of a pass and on its leaf entries.
    pub(super) leaves: usize,
    /// Arena slots the chunk's splits may take.
    pub(super) slots: usize,
}

/// The queues of one carrier type.
pub(super) struct Queues<C> {
    pub(super) items: Vec<Item<C>>,
    pub(super) next: Vec<Item<C>>,
    pub(super) leaves: Vec<Item<C>>,
    /// The positions of a pass's varying stops, which resolve lists for fixed-width
    /// handles (empty for other carriers).
    pub(super) listed: Vec<u32>,
}

impl<C: Copy> Queues<C> {
    fn new(none: C, items: usize, leaves: usize, listed: usize) -> Self {
        let item = Item {
            node: 0,
            carry: none,
        };
        Self {
            items: vec![item; items],
            next: vec![item; items],
            leaves: vec![item; leaves],
            listed: vec![0; listed],
        }
    }
}

/// Exact-sum buffers for a group: two difference arrays, which alternate leaf
/// entries, so an entry's adds never wait on the previous entry's stores to the same
/// row. Entries that share a run boundary would otherwise chain through
/// store-to-load forwarding, and every entry of a group's first pass covers all rows.
pub(super) enum Sums {
    /// Two arrays of one `i128` per row plus one.
    Fixed(Vec<i128>),
    /// Two arrays of one accumulator per row plus one; [`WIDE_SUMS`] running ones,
    /// the finish's first, which a single row's leaves take turns in; and as many
    /// for the leaf pass's sums of every row, which [`WideRuns`](crate::exact::WideRuns) keeps zeroed.
    Wide {
        wide: exact::Wide,
        diff: Vec<i64>,
        acc: Vec<i64>,
        sums: Vec<i64>,
    },
}

/// Scratch that every mask type shares.
pub(super) struct Common {
    pub(super) bins: Vec<u32>,
    /// Each tree's start node, then [`MAX_PAD`] dummies.
    pub(super) roots: Vec<u32>,
    /// Chunks per mask width class the configuration allows, narrowest first.
    pub(super) chunks: Vec<Vec<Chunk>>,
    /// Chunks of single rows, whose walks stop only at leaves: a tree is one stop,
    /// budgeted as one 32-bit leaf, so few chunks hold every tree.
    pub(super) constant: Vec<Chunk>,
    /// A constant chunk's roots and [`MAX_PAD`] dummies, for every chunk but the
    /// last, whose roots the walk reads in place; empty with a single chunk.
    pub(super) constant_roots: Vec<u32>,
    /// The configuration's widest group.
    pub(super) max_rows: usize,
    /// The all-rows mask, then every predicate's left and right masks, at the
    /// widest width.
    pub(super) masks: Vec<Unit>,
    pub(super) arena: Vec<Unit>,
    /// The sweep's copies of its running left mask at the wide widths, or at runtime
    /// widths those copies and the precompute's left and missing masks.
    pub(super) bits: Vec<Unit>,
    /// A mask for scanned sides.
    pub(super) scan: Vec<Unit>,
    /// Each item's stop node and the node's word 1, at the item's position.
    pub(super) stops: Vec<u32>,
    pub(super) words: Vec<u64>,
    /// A group's non-missing values of one feature with their rows, sorted for the
    /// sweep, and its categorical codes, one per row.
    pub(super) order: Vec<(f64, u32)>,
    pub(super) codes: Vec<u32>,
    pub(super) sums: Sums,
    /// The machine's rules applied to the model.
    pub(super) tuning: Tuning,
}

/// The kernel's scratch, sized for the model and its configuration.
pub(super) struct Scratch {
    pub(super) common: Common,
    pub(super) q32: Queues<u32>,
    pub(super) q64: Queues<u64>,
    pub(super) handles: Queues<Handle>,
    pub(super) slots: Queues<Slot>,
}

impl Scratch {
    pub(super) fn new(model: &Model, tuning: Tuning) -> Self {
        let layout = &model.layout;
        let rows = model.config.max_group_width();
        let n_masks = 2 * model.varying_predicates.len() + 1;
        let need = row_scratch_bytes(model, rows, n_masks);
        let widest = Width::of(rows);
        let dummy = layout.dummy();
        let chunk_bytes = tuning.chunk_bytes;

        let mut roots = model.tree_starts.clone();
        roots.resize(model.trees.len() + MAX_PAD, dummy);

        let needs = &model.needs;
        // Chunks of consecutive trees within the budget at each width class, less the
        // predicates' masks and the bins, which every chunk reads; and the most any
        // chunk needs, per carrier.
        let widths: Vec<Width> = widest.up_to().collect();
        let chunks: Vec<Vec<Chunk>> = widths
            .iter()
            .map(|&w| {
                let shared = n_masks * w.bytes() + 4 * layout.bins.n_slots();
                chunk_trees(needs, w, tree_budget(chunk_bytes, shared))
            })
            .collect();
        let most = |of: &dyn Fn(Width) -> bool, field: fn(&Chunk) -> usize| {
            widths
                .iter()
                .zip(&chunks)
                .filter(|(w, _)| of(**w))
                .flat_map(|(_, c)| c.iter().map(field))
                .max()
                .unwrap_or(0)
        };
        let constant = constant_chunks(model.trees.len(), chunk_bytes);
        let constant_leaves = constant.iter().map(|c| c.leaves).max().unwrap_or(0);
        let constant_roots = if constant.len() > 1 {
            vec![dummy; constant_leaves + MAX_PAD]
        } else {
            Vec::new()
        };
        let leaves_of = |w: Width| most(&|x| x == w, |c| c.leaves);
        let max_leaves = most(&|_| true, |c| c.leaves).max(constant_leaves);
        let handle_leaves = most(&|w| w.handles() && !w.runtime(), |c| c.leaves);
        let slot_leaves = most(&Width::runtime, |c| c.leaves);
        let arena_bytes = widths
            .iter()
            .zip(&chunks)
            .filter(|(w, _)| w.handles())
            .flat_map(|(w, c)| c.iter().map(move |c| (c.slots + 3) * w.slot_bytes()))
            .max()
            .unwrap_or(0);
        let runtime_units = if rows > WIDE_ROWS {
            rows::units(rows)
        } else {
            0
        };

        let units = |bytes: usize| bytes.div_ceil(size_of::<Unit>());
        let sums = match model.plan {
            Plan::Fixed(_) => Sums::Fixed(buffer(2 * (rows + 1), 0, rows, need)),
            Plan::Wide(wide) => Sums::Wide {
                wide,
                diff: buffer(2 * (rows + 1) * wide.limbs(), 0, rows, need),
                acc: vec![0; WIDE_SUMS * wide.limbs()],
                sums: vec![0; WIDE_SUMS * wide.limbs()],
            },
        };
        let queues = |leaves: usize| (leaves + MAX_PAD + 1, leaves + 1);
        let ((i32s, l32s), (i64s, l64s), (ihs, lhs), (iss, lss)) = (
            queues(leaves_of(Width::W32)),
            queues(leaves_of(Width::W64)),
            queues(handle_leaves),
            queues(slot_leaves),
        );
        let scratch = Self {
            common: Common {
                bins: layout.bins.new_bins(),
                roots,
                chunks,
                constant,
                constant_roots,
                max_rows: rows,
                masks: buffer(
                    units(n_masks * mask_bytes(rows)),
                    <Unit as Mask>::ZERO,
                    rows,
                    need,
                ),
                arena: vec![<Unit as Mask>::ZERO; units(arena_bytes)],
                bits: buffer(
                    sweep_units(rows, runtime_units),
                    <Unit as Mask>::ZERO,
                    rows,
                    need,
                ),
                scan: buffer(runtime_units, <Unit as Mask>::ZERO, rows, need),
                stops: vec![0; max_leaves + MAX_PAD + 1],
                words: vec![0; max_leaves + MAX_PAD + 1],
                order: buffer(rows, (0.0, 0), rows, need),
                codes: buffer(rows, 0, rows, need),
                sums,
                tuning,
            },
            q32: Queues::new(0, i32s, l32s, 0),
            q64: Queues::new(0, i64s, l64s, 0),
            handles: Queues::new(<Handle as Carrier<Wide<2>>>::NONE, ihs, lhs, lhs),
            slots: Queues::new(<Slot as Carrier<Unit>>::NONE, iss, lss, 0),
        };
        // Checked once the buffers are reserved: a width past it needs over 13 TiB of
        // them, which the allocator refuses first.
        assert!(
            rows <= WIDE_ROWS || u32::try_from(2 * runtime_units).is_ok(),
            "max_group_width {rows} has more mask words than 32-bit row-span indices count"
        );
        scratch
    }
}

/// Bytes of the scratch that grows with the widest group of `rows` rows, given
/// `n_masks` varying masks: per row, two difference-array entries (16-byte `i128`s, or
/// 8-byte slots per wide limb), a sort entry of 16 bytes and a code of 4, and a bit
/// of each mask; past [`WIDE_ROWS`], three more runtime-width masks: the
/// precompute's running left and missing rows and a scanned side. Panics, naming
/// `max_group_width`, if the bytes overflow `isize`.
fn row_scratch_bytes(model: &Model, rows: usize, n_masks: usize) -> usize {
    let sum_bytes = match model.plan {
        Plan::Fixed(_) => Some(size_of::<i128>()),
        Plan::Wide(wide) => wide.limbs().checked_mul(size_of::<i64>()),
    };
    let units = rows::units(rows);
    let bytes = (|| {
        let sums = rows
            .checked_add(1)?
            .checked_mul(2)?
            .checked_mul(sum_bytes?)?;
        let order = rows.checked_mul(size_of::<(f64, u32)>() + size_of::<u32>())?;
        let mask = if rows > WIDE_ROWS {
            units.checked_mul(size_of::<Unit>())?
        } else {
            Width::of(rows).bytes()
        };
        let masks = n_masks.checked_mul(mask)?;
        let runtime = if rows > WIDE_ROWS {
            units.checked_mul(size_of::<Unit>())?
        } else {
            0
        };
        let bits = sweep_units(rows, units).checked_mul(size_of::<Unit>())?;
        let total = sums
            .checked_add(order)?
            .checked_add(masks)?
            .checked_add(runtime)?
            .checked_add(bits)?;
        isize::try_from(total).ok().map(|_| total)
    })();
    bytes.unwrap_or_else(|| {
        panic!(
            "max_group_width {rows} needs more scratch than memory can address \
             (over {} bytes)",
            isize::MAX
        )
    })
}

/// A scratch buffer of `len` copies of `value`, reserved without aborting: panics,
/// naming `max_group_width` (`rows`) and the scratch's `bytes`, if the allocator
/// refuses.
fn buffer<T: Clone>(len: usize, value: T, rows: usize, bytes: usize) -> Vec<T> {
    let mut v = Vec::new();
    assert!(
        v.try_reserve_exact(len).is_ok(),
        "max_group_width {rows} needs {bytes} bytes of scratch, which cannot be allocated"
    );
    v.resize(len, value);
    v
}

/// Units of [`Common::bits`] for groups of up to `rows` rows, `units` per runtime-width
/// mask: the sweep's [`SWEEP_TURNS`] copies of the widest wide mask, or at runtime
/// widths those copies and the left and missing masks.
const fn sweep_units(rows: usize, units: usize) -> usize {
    if rows > WIDE_ROWS {
        (SWEEP_TURNS + 2) * units
    } else if rows > 64 {
        SWEEP_TURNS * Width::of(rows).bytes() / size_of::<Unit>()
    } else {
        0
    }
}

/// The budget left for a chunk's trees after `shared` bytes every chunk reads, but
/// at least half the budget: when the shared bytes do not fit their half, they
/// cannot stay in cache anyway, and smaller chunks would only add passes.
pub(super) const fn tree_budget(budget: usize, shared: usize) -> usize {
    let half = budget / 2;
    let left = budget.saturating_sub(shared);
    if left > half { left } else { half }
}

/// Chunks of single rows' walks, which stop only at leaves: a tree is one item, and
/// what a pass revisits is its word 1 and, for chunks but the last, its root in
/// `constant_roots`. The nodes are read once per call, so they do not count.
pub(super) fn constant_chunks(trees: usize, budget: usize) -> Vec<Chunk> {
    let per_chunk = (budget / (size_of::<u64>() + size_of::<u32>())).max(1);
    (0..trees)
        .step_by(per_chunk)
        .map(|t| {
            let end = (t + per_chunk).min(trees);
            Chunk {
                trees: (t, end),
                leaves: end - t,
                slots: 0,
            }
        })
        .collect()
}

/// Consecutive trees in chunks whose working set at width `w` stays within `budget`
/// bytes, or one tree if it alone passes it.
pub(super) fn chunk_trees(needs: &[Need], w: Width, budget: usize) -> Vec<Chunk> {
    let cost =
        |n: Need| n.leaves * w.leaf_bytes() + if w.handles() { n.slots * w.bytes() } else { 0 };
    let mut chunks = Vec::new();
    let mut chunk = Chunk {
        trees: (0, 0),
        leaves: 0,
        slots: 0,
    };
    let mut bytes = 0;
    for (t, &need) in needs.iter().enumerate() {
        if t > chunk.trees.0 && bytes + cost(need) > budget {
            chunks.push(chunk);
            chunk = Chunk {
                trees: (t, t),
                leaves: 0,
                slots: 0,
            };
            bytes = 0;
        }
        bytes += cost(need);
        chunk.trees.1 = t + 1;
        chunk.leaves += need.leaves;
        chunk.slots += need.slots;
    }
    if chunk.trees.1 > chunk.trees.0 {
        chunks.push(chunk);
    }
    chunks
}
