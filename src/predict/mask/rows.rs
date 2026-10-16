//! Row masks of a runtime number of 128-bit units, for groups wider than the
//! [`Wide`] masks: a group of `n` rows takes `units(n)` units ([`units`]), bit `r`
//! being bit `r % 64` of word `r / 64`. They live in arenas and are passed as slices or
//! pointers, with their unit count or their row span ([`Span`]), and each operation
//! loops over the units with one NEON or SSE2 vector per unit. Masks carried through
//! a group's splits keep their span, so that a split and a leaf entry touch only the
//! units and words that hold their rows.

use std::hint::select_unpredictable;

use super::ops::add_word_runs;
use super::{Wide, simd};
use crate::exact::RunSink;

/// The 16-byte unit of a runtime-width mask: two words, one vector.
pub type Unit = Wide<2>;

/// Units of a mask of `rows` rows.
#[must_use]
pub const fn units(rows: usize) -> usize {
    rows.div_ceil(128)
}

/// The words of `m`, lowest first.
#[must_use]
pub const fn words(m: &[Unit]) -> &[u64] {
    // SAFETY: a unit is two words without padding (`repr(C)`), so the units' bytes
    // hold twice as many words, at a word-aligned address.
    unsafe { std::slice::from_raw_parts(m.as_ptr().cast::<u64>(), 2 * m.len()) }
}

/// The words of `m`, lowest first, to write.
pub const fn words_mut(m: &mut [Unit]) -> &mut [u64] {
    // SAFETY: as in [`words`]; the borrow is exclusive.
    unsafe { std::slice::from_raw_parts_mut(m.as_mut_ptr().cast::<u64>(), 2 * m.len()) }
}

/// Sets `m` to rows `0..n`. Panics if `n` exceeds its rows.
pub fn first(m: &mut [Unit], n: usize) {
    let w = words_mut(m);
    assert!(
        n <= 64 * w.len(),
        "first({n}) exceeds {} rows",
        64 * w.len()
    );
    for (i, x) in w.iter_mut().enumerate() {
        let k = n.saturating_sub(64 * i).min(64);
        *x = u64::MAX.checked_shr(64 - k as u32).unwrap_or(0);
    }
}

/// The words of a runtime-width mask that hold its rows, `lo..hi`: from its first
/// nonzero word to its last, or `0..0` when it has none. The span's units
/// ([`Self::units`]) hold the mask, which is 0 in their words outside the span; units
/// outside them may hold anything, and the operations below neither read nor write
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub lo: usize,
    pub hi: usize,
}

impl Span {
    /// The span of a mask without rows.
    pub const EMPTY: Self = Self { lo: 0, hi: 0 };

    /// The span of rows `0..n`, as [`first`] writes them.
    #[must_use]
    pub const fn first(n: usize) -> Self {
        Self {
            lo: 0,
            hi: n.div_ceil(64),
        }
    }

    /// Whether the mask has no rows.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.lo == self.hi
    }

    /// The units that hold the span's words.
    #[must_use]
    pub const fn units(self) -> std::ops::Range<usize> {
        self.lo / 2..self.hi.div_ceil(2)
    }
}

/// Splits the mask at `rows` of span `span` by the mask at `side`: writes `rows & side`
/// at `heavy` and `rows & !side` at `light` over the span's units, and returns the
/// span of each, which lies within `span`. Each unit is read before its own unit is
/// written, so `heavy` may be `rows`. The sides' spans are tracked in vector lanes as
/// their units are written ([`simd::Spans`]), so that no branch depends on where the
/// rows lie.
///
/// # Safety
///
/// `rows` and `side` are valid for reading the span's units, `heavy` and `light` for
/// writing them; `light` overlaps none of the others, and `heavy` is `rows` or overlaps
/// none of the others. Word indices are below 2^32.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "resolve's split, once per varying stop"
)]
pub unsafe fn split(
    rows: *const Unit,
    side: *const Unit,
    heavy: *mut Unit,
    light: *mut Unit,
    span: Span,
) -> (Span, Span) {
    let units = span.units();
    let mut spans = simd::Spans::new(units.start);
    for i in units {
        // SAFETY: the caller guarantees the bounds and that unit i of rows is read
        // before unit i of heavy is written.
        unsafe {
            let (carried, split) = ((*rows.add(i)).0, (*side.add(i)).0);
            let heavy_rows = simd::and(&carried, &split);
            let light_rows = simd::and_not(&carried, &split);
            *heavy.add(i) = Wide(heavy_rows);
            *light.add(i) = Wide(light_rows);
            spans.add(&heavy_rows, &light_rows);
        }
    }
    let side_span = |(first, last): (u32, u32)| {
        select_unpredictable(
            first == u32::MAX,
            Span::EMPTY,
            Span {
                lo: first as usize,
                hi: last as usize + 1,
            },
        )
    };
    let [h, l] = spans.get();
    (side_span(h), side_span(l))
}

/// For each maximal run of rows `start..end` of the mask whose span's words are `w`,
/// from word `base` of the mask, adds `v` at `start` and subtracts it at `end` in
/// `sink`, as [`super::Mask::add_runs`] does. A mask that is one run writes it
/// directly; otherwise each nonzero word writes the runs that start or end in it.
/// Panics if a run ends at or past `sink.entries()`.
///
/// `w` is a span ([`Span`]): empty, or its first and last words are not 0. So an
/// entry reads only the words that hold its rows, and the one-run test reads words
/// from the first only until one is not full: a fragmented mask pays for a word or
/// two rather than a pass over its span.
#[expect(
    clippy::inline_always,
    reason = "the leaf pass calls this twice per loop, and outlined it costs a call per entry"
)]
#[inline(always)]
pub fn add_runs<S: RunSink + ?Sized>(w: &[u64], base: usize, v: S::Value, sink: &mut S) {
    let (Some(&low), Some(&high)) = (w.first(), w.last()) else {
        return;
    };
    debug_assert!(
        low != 0 && high != 0,
        "a span's first and last words have rows"
    );
    let last = w.len() - 1;
    let lo = 64 * base + low.trailing_zeros() as usize;
    let hi = 64 * (base + last) + 64 - high.leading_zeros() as usize;
    assert!(hi < sink.entries(), "a run ends past the sink");
    // One run: its first word is full from its first row up, its last word full up to
    // its last row, and the words between are full; or, in one word, its rows are
    // contiguous.
    let one_run = if last == 0 {
        let shifted = low >> low.trailing_zeros();
        shifted & shifted.wrapping_add(1) == 0
    } else {
        low == u64::MAX << low.trailing_zeros()
            && high & high.wrapping_add(1) == 0
            && w[1..last].iter().all(|&x| x == u64::MAX)
    };
    if one_run {
        // SAFETY: lo < hi < sink.entries().
        unsafe {
            sink.add(lo, v);
            sink.sub(hi, v);
        }
        return;
    }
    // Each word read once: the words next to the span are 0, so a word's
    // neighbours' edge rows come from the words carried, and the last word is peeled.
    let mut below = 0;
    let mut x = low;
    for i in 0..last {
        // SAFETY: i < last < w.len().
        let next = unsafe { *w.get_unchecked(i + 1) };
        if x != 0 {
            // SAFETY: every run starts at or after lo and ends at or before hi, and
            // hi < sink.entries().
            unsafe { add_word_runs(sink, 64 * (base + i), x, below, next << 63, v) };
        }
        below = x >> 63;
        x = next;
    }
    // SAFETY: as above.
    unsafe { add_word_runs(sink, 64 * (base + last), high, below, 0, v) };
}
