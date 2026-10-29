//! The fixed-width row masks ([`Mask`]): `u32` and `u64` by value, and the [`Wide`]
//! arrays of 2 to 16 words, with the runs of rows they add to a difference array.

use std::hint::select_unpredictable;
use std::marker::PhantomData;

use super::simd;
use crate::exact::RunSink;
use crate::predict::isa::{Isa, Native};

/// A row mask: bit `r` set means row `r` is active.
///
/// `u32` and `u64` are carried by value; [`Wide`] masks are meant to live in an arena
/// and be read by reference.
pub trait Mask: Copy + Eq + Send + Sync + 'static {
    /// The level its operations are compiled for ([`Isa`]): [`Wide`]'s own, and
    /// [`Native`] for the words, whose operations do not depend on one.
    type Level: Isa;

    /// Rows the mask holds.
    const WIDTH: usize;

    /// No rows.
    const ZERO: Self;

    /// Rows `0..n`, every row of a group of `n`. Panics if `n > WIDTH`.
    fn first(n: usize) -> Self;

    /// Every row if `c`, else none, without a branch.
    fn ones_if(c: bool) -> Self;

    /// Sets row `r < WIDTH`.
    fn set(&mut self, r: usize);

    /// Flips row `r < WIDTH`.
    fn flip(&mut self, r: usize);

    /// Rows in both.
    #[must_use]
    fn and(self, o: Self) -> Self;

    /// Rows in `self` but not in `o`.
    #[must_use]
    fn and_not(self, o: Self) -> Self;

    /// Rows in either.
    #[must_use]
    fn or(self, o: Self) -> Self;

    /// Rows in exactly one.
    #[must_use]
    fn xor(self, o: Self) -> Self;

    /// True if no row is set.
    fn is_zero(&self) -> bool;

    /// `a` if `c`, else `b`, without a branch.
    fn select(c: bool, a: Self, b: Self) -> Self;

    /// For each maximal run of rows `start..end`, adds `v` at `start` and subtracts it at
    /// `end` in `sink`. Panics if a run ends at or past `sink.entries()`, so a group of
    /// `n` rows needs `n + 1` entries and no row at or past `n`.
    fn add_runs<S: RunSink + ?Sized>(&self, v: S::Value, sink: &mut S);

    /// Calls `f` with each set row, in increasing order.
    fn for_each_row(&self, f: impl FnMut(usize));
}

macro_rules! impl_word_mask {
    ($ty:ty) => {
        impl Mask for $ty {
            type Level = Native;
            const WIDTH: usize = <$ty>::BITS as usize;
            const ZERO: Self = 0;

            #[inline]
            fn first(n: usize) -> Self {
                assert!(n <= Self::WIDTH, "first({n}) exceeds WIDTH={}", Self::WIDTH);
                Self::MAX.checked_shr((Self::WIDTH - n) as u32).unwrap_or(0)
            }
            #[inline]
            fn ones_if(c: bool) -> Self {
                Self::from(c).wrapping_neg()
            }
            #[inline]
            fn set(&mut self, r: usize) {
                debug_assert!(r < Self::WIDTH);
                *self |= 1 << r;
            }
            #[inline]
            fn flip(&mut self, r: usize) {
                debug_assert!(r < Self::WIDTH);
                *self ^= 1 << r;
            }
            #[inline]
            fn and(self, o: Self) -> Self {
                self & o
            }
            #[inline]
            fn and_not(self, o: Self) -> Self {
                self & !o
            }
            #[inline]
            fn or(self, o: Self) -> Self {
                self | o
            }
            #[inline]
            fn xor(self, o: Self) -> Self {
                self ^ o
            }
            #[inline]
            fn is_zero(&self) -> bool {
                *self == 0
            }
            #[inline]
            fn select(c: bool, a: Self, b: Self) -> Self {
                select_unpredictable(c, a, b)
            }
            #[inline]
            fn for_each_row(&self, mut f: impl FnMut(usize)) {
                let mut m = *self;
                while m != 0 {
                    f(m.trailing_zeros() as usize);
                    m &= m - 1;
                }
            }
            // Always: the leaf pass calls this twice per loop, and outlined it costs a
            // call per entry.
            #[inline(always)]
            fn add_runs<S: RunSink + ?Sized>(&self, v: S::Value, sink: &mut S) {
                let m = *self;
                let end = Self::WIDTH - m.leading_zeros() as usize;
                assert!(m == 0 || end < sink.entries(), "a run ends past the sink");
                let mut starts = m & !(m << 1);
                let mut ends = m & !(m >> 1);
                while starts != 0 {
                    let a = starts.trailing_zeros() as usize;
                    let b = ends.trailing_zeros() as usize + 1;
                    // SAFETY: a < b <= end < sink.entries().
                    unsafe {
                        sink.add(a, v);
                        sink.sub(b, v);
                    }
                    starts &= starts - 1;
                    ends &= ends - 1;
                }
            }
        }
    };
}

impl_word_mask!(u32);
impl_word_mask!(u64);

/// Row mask of `W` 64-bit words, `64 * W` rows: bit `r` is bit `r % 64` of word
/// `r / 64`, its operations those of level `X` ([`Isa`]). [`Mask`] is implemented for
/// `W` of 2, 4, 8 and 16. Aligned to a NEON or SSE2 vector. Where the build dispatches
/// to levels, its operations are `#[inline(always)]`, for the reason the x86-64
/// backend's are.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct Wide<const W: usize, X = Native>(pub [u64; W], pub PhantomData<X>);

impl<const W: usize, X> Wide<W, X> {
    /// The mask of words `w`.
    #[inline(always)]
    pub const fn new(w: [u64; W]) -> Self {
        Self(w, PhantomData)
    }
}

macro_rules! impl_wide_mask {
    ($($w:literal),*) => {$(
        impl<X: Isa> PartialEq for Wide<$w, X> {
            #[cfg_attr(dispatch, inline(always))]
            #[cfg_attr(not(dispatch), inline)]
            fn eq(&self, o: &Self) -> bool {
                simd::is_zero::<X, $w>(&simd::xor::<X, $w>(&self.0, &o.0))
            }
        }

        impl<X: Isa> Eq for Wide<$w, X> {}

        impl<X: Isa> Mask for Wide<$w, X> {
            type Level = X;
            const WIDTH: usize = 64 * $w;
            const ZERO: Self = Self::new([0; $w]);

            // Always: outlined, its result goes through a stack copy on x86.
            #[inline(always)]
            fn first(n: usize) -> Self {
                assert!(n <= Self::WIDTH, "first({n}) exceeds WIDTH={}", Self::WIDTH);
                Self::new(simd::first(n))
            }
            #[cfg_attr(dispatch, inline(always))]
            #[cfg_attr(not(dispatch), inline)]
            fn ones_if(c: bool) -> Self {
                Self::new([u64::from(c).wrapping_neg(); $w])
            }
            // Always: the precompute's NaN gather calls this per missing row, and
            // outlined it reloads the loop's state after each call.
            #[inline(always)]
            fn set(&mut self, r: usize) {
                self.0[r / 64] |= 1 << (r % 64);
            }
            // Always: the categorical masks call this per flip, and a wide mask's
            // flip is one word's.
            #[inline(always)]
            fn flip(&mut self, r: usize) {
                self.0[r / 64] ^= 1 << (r % 64);
            }
            #[cfg_attr(dispatch, inline(always))]
            #[cfg_attr(not(dispatch), inline)]
            fn and(self, o: Self) -> Self {
                Self::new(simd::and::<X, $w>(&self.0, &o.0))
            }
            #[cfg_attr(dispatch, inline(always))]
            #[cfg_attr(not(dispatch), inline)]
            fn and_not(self, o: Self) -> Self {
                Self::new(simd::and_not::<X, $w>(&self.0, &o.0))
            }
            #[cfg_attr(dispatch, inline(always))]
            #[cfg_attr(not(dispatch), inline)]
            fn or(self, o: Self) -> Self {
                Self::new(simd::or::<X, $w>(&self.0, &o.0))
            }
            #[cfg_attr(dispatch, inline(always))]
            #[cfg_attr(not(dispatch), inline)]
            fn xor(self, o: Self) -> Self {
                Self::new(simd::xor::<X, $w>(&self.0, &o.0))
            }
            #[cfg_attr(dispatch, inline(always))]
            #[cfg_attr(not(dispatch), inline)]
            fn is_zero(&self) -> bool {
                simd::is_zero::<X, $w>(&self.0)
            }
            #[cfg_attr(dispatch, inline(always))]
            #[cfg_attr(not(dispatch), inline)]
            fn select(c: bool, a: Self, b: Self) -> Self {
                Self::new(simd::select::<X, $w>(c, &a.0, &b.0))
            }
            #[cfg_attr(dispatch, inline(always))]
            #[cfg_attr(not(dispatch), inline)]
            fn for_each_row(&self, mut f: impl FnMut(usize)) {
                for (i, &w) in self.0.iter().enumerate() {
                    let mut m = w;
                    while m != 0 {
                        f(64 * i + m.trailing_zeros() as usize);
                        m &= m - 1;
                    }
                }
            }
            // Always: the leaf pass calls this twice per loop, and outlined it costs a
            // call per entry.
            #[inline(always)]
            fn add_runs<S: RunSink + ?Sized>(&self, v: S::Value, sink: &mut S) {
                add_wide_runs::<$w, simd::Maps<X>, S>(&self.0, v, sink);
            }
        }
    )*};
}

impl_wide_mask!(2, 4, 8, 16);

/// Facts about each word of a wide mask, as maps of `s = stride::<W>()` bits per word:
/// word `i` owns bits `s * i..s * (i + 1)`, all set or all clear. The vector versions
/// choose the stride their compares narrow to cheaply.
pub(super) trait WordMaps {
    /// Bits per word in a map of `W` words; `stride::<W>() * W <= 64`.
    fn stride<const W: usize>() -> u32;

    /// The words that are not 0.
    fn nonzero<const W: usize>(w: &[u64; W]) -> u64;
}

/// For each run of rows that starts or ends in a mask's word `x`, whose rows start at
/// `row`, adds `v` at its start or subtracts it past its end in `sink`: `below` holds
/// the row before the word in bit 0, and `above` the row after it in bit 63.
///
/// # Safety
///
/// Those runs start and end below `sink.entries()`.
#[expect(
    clippy::inline_always,
    reason = "the multi-run loops call this per word; outlined it costs a call per word"
)]
#[inline(always)]
pub(super) unsafe fn add_word_runs<S: RunSink + ?Sized>(
    sink: &mut S,
    row: usize,
    x: u64,
    below: u64,
    above: u64,
    v: S::Value,
) {
    let mut starts = x & !((x << 1) | below);
    let mut ends = x & !((x >> 1) | above);
    // SAFETY: the caller's.
    unsafe {
        while starts != 0 {
            sink.add(row + starts.trailing_zeros() as usize, v);
            starts &= starts - 1;
        }
        while ends != 0 {
            sink.sub(row + ends.trailing_zeros() as usize + 1, v);
            ends &= ends - 1;
        }
    }
}

/// Rows set in the words `w`. On AArch64, NEON's byte counts added across words before
/// one sum (`neon::popcount`): `count_ones` there reduces each vector of counts on its
/// own. Elsewhere `count_ones` per word, which LLVM lowers to the level's popcount:
/// POPCNT per word, VPOPCNTQ with AVX-512's VPOPCNTDQ, or its own byte lookup on AVX2
/// without it.
#[expect(
    clippy::inline_always,
    reason = "add_wide_runs calls this per mask, and outlined it costs a call per mask"
)]
#[inline(always)]
pub(super) fn count_rows<const W: usize>(w: &[u64; W]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        simd::popcount(w)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        w.iter().map(|x| x.count_ones()).sum()
    }
}

/// [`Mask::add_runs`] for the words `w` of a wide mask, with the word facts of `O`. A
/// mask that is one run (as many rows set as its span from its first to its last row
/// holds) writes it directly; otherwise each word from the first nonzero one to the
/// last writes the runs that start or end in it (words of 0 between them write none).
#[expect(
    clippy::inline_always,
    reason = "the leaf pass calls this twice per loop, and outlined it costs a call per entry"
)]
#[inline(always)]
pub(super) fn add_wide_runs<const W: usize, O: WordMaps, S: RunSink + ?Sized>(
    w: &[u64; W],
    v: S::Value,
    sink: &mut S,
) {
    let stride = O::stride::<W>();
    debug_assert!(stride >= 1 && stride as usize * W <= 64);
    let nonzero = O::nonzero(w);
    if nonzero == 0 {
        return;
    }
    let first = (nonzero.trailing_zeros() / stride) as usize;
    let last = (nonzero.ilog2() / stride) as usize;
    debug_assert!(first <= last && last < W);
    // SAFETY: a map's set bits are below `stride * W`, so its words are below `W`.
    let (head, tail) = unsafe { (*w.get_unchecked(first), *w.get_unchecked(last)) };
    let lo = 64 * first + head.trailing_zeros() as usize;
    let hi = 64 * last + 64 - tail.leading_zeros() as usize;
    assert!(hi < sink.entries(), "a run ends past the sink");
    // Every word's rows: the words outside the span are 0.
    if count_rows(w) as usize == hi - lo {
        // SAFETY: lo < hi < sink.entries().
        unsafe {
            sink.add(lo, v);
            sink.sub(hi, v);
        }
        return;
    }
    // Each word from the first nonzero one to the last, read once: the words next to
    // them are 0, so a word's neighbours' edge rows come from the words carried, and
    // the last word is peeled.
    let mut below = 0;
    let mut x = head;
    for i in first..last {
        // SAFETY: i < last < W.
        let next = unsafe { *w.get_unchecked(i + 1) };
        // SAFETY: every run starts at or after lo and ends at or before hi, and
        // hi < sink.entries().
        unsafe { add_word_runs(sink, 64 * i, x, below, next << 63, v) };
        below = x >> 63;
        x = next;
    }
    // SAFETY: as above.
    unsafe { add_word_runs(sink, 64 * last, x, below, 0, v) };
}
