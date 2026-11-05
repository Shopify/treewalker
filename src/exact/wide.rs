//! The wide accumulator: leaves past the 126-bit rule in 29-bit digits held in `i64`
//! slots, so that adding a leaf adds three digits without carries, and its difference
//! arrays.

use std::mem::MaybeUninit;

use super::RunSink;
use super::{decompose, scaled};

/// Bits of a wide accumulator's digit.
///
/// Three constraints fix it. A leaf's 53-bit mantissa, placed at any offset within
/// its first digit (0 to `DIGIT_BITS - 1`), spans at most `53 + DIGIT_BITS - 1`
/// bits, which three digits hold when `DIGIT_BITS >= 26`. A signed digit fits an
/// `i32`, so that a leaf takes 16 bytes as an `i128` does, when `DIGIT_BITS <= 31`.
/// And no `i64` slot may overflow for any tree count a model can have: the busiest
/// slot is the top digit while the finish adds a row's entries, which holds at most
/// `3 × MAX_TREES + 3` digits' worth (see [`WideRuns`]), so `(3 × MAX_TREES + 3) ×
/// 2^DIGIT_BITS < 2^63`, which holds up to 29 bits. The widest digit within all
/// three is 29 bits: fewer digits per accumulator than any narrower one.
const DIGIT_BITS: u32 = 29;
const DIGIT_MASK: i64 = (1 << DIGIT_BITS) - 1;
/// The digits a placed leaf covers.
const LEAF_DIGITS: usize = 3;
/// More trees than a model can have: every tree has a node, and node indices are
/// 32-bit.
const MAX_TREES: u64 = u32::MAX as u64;
/// Digits of the widest layout: finite `f64` values span 1024 + 1074 bits.
pub(super) const MAX_LIMBS: usize = (1024 + 1074usize).div_ceil(DIGIT_BITS as usize);

const _: () = assert!(53 + DIGIT_BITS - 1 <= LEAF_DIGITS as u32 * DIGIT_BITS);
const _: () = assert!(DIGIT_BITS < i32::BITS);
const _: () = assert!(((3 * MAX_TREES + 3) as u128) << DIGIT_BITS < i64::MAX as u128);

/// The layout of a wide accumulator: [`Self::limbs`] signed 64-bit slots, least
/// significant first, each holding a digit worth 2^(29 k) units of 2^[`Self::low`].
///
/// The value is the sum of the slots times their weights, so a slot may hold more
/// than a digit: adding a leaf adds its three digits to three slots, independently,
/// and no carry propagates. The slots' headroom ([`DIGIT_BITS`]) keeps every sum of
/// a model's leaves exact without carries, and [`Self::normalize`] carries them once,
/// before a conversion. An accumulator is a `&mut [i64]` of `limbs` slots, zeroed to
/// start, so a difference array is one flat buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wide {
    limbs: usize,
    low: i32,
}

/// A leaf value in a [`Wide`] layout's digits: `digits[j]` is added to slot
/// `first + j`. Each digit's magnitude is below 2^29, and all three take the value's
/// sign.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WideLeaf {
    first: u32,
    digits: [i32; LEAF_DIGITS],
}

impl Wide {
    /// A layout in which the sum of any of `values` over up to `n_trees` trees is
    /// exact, whatever the span of the values: as many digits as the span needs, at
    /// least three. 73 digits cover any finite `f64` values with any tree count.
    ///
    /// # Panics
    ///
    /// If a value is not finite.
    #[cfg(test)]
    pub fn new(values: impl IntoIterator<Item = f64>, n_trees: usize) -> Self {
        let super::Extent { min_exp, max_top } = super::extent(values).expect("finite values");
        Self::spanning(min_exp, max_top, n_trees)
    }

    pub(super) const fn spanning(min_exp: i32, max_top: i32, n_trees: usize) -> Self {
        assert!(n_trees as u64 <= MAX_TREES, "the slots' headroom");
        if min_exp == i32::MAX {
            return Self {
                limbs: LEAF_DIGITS,
                low: 0,
            };
        }
        // Every value's set bits lie in [min_exp, max_top), span bits in units of
        // 2^min_exp, so its digits fit the span's digits, and three of them from a
        // digit at most limbs - 3. A row's sum takes no more digits: the slots'
        // headroom holds its tree count in the top one.
        let span = (max_top - min_exp) as usize;
        let digits = span.div_ceil(DIGIT_BITS as usize);
        Self {
            limbs: if digits > LEAF_DIGITS {
                digits
            } else {
                LEAF_DIGITS
            },
            low: min_exp,
        }
    }

    /// Slots per accumulator.
    #[inline]
    pub const fn limbs(self) -> usize {
        self.limbs
    }

    /// The exponent of slot 0's lowest bit.
    #[inline]
    #[cfg(test)]
    pub const fn low(self) -> i32 {
        self.low
    }

    /// `v`'s digits, placed so that all three are slots of the layout. `v` must be
    /// finite, within the layout's span, with no set bit below 2^low.
    pub fn leaf(self, v: f64) -> WideLeaf {
        let (mut mantissa, ex) = decompose(v);
        if mantissa == 0 {
            return WideLeaf::default();
        }
        let mut at = ex - self.low;
        if at < 0 {
            debug_assert!(
                mantissa.trailing_zeros() as i32 >= -at,
                "{v} has bits below 2^{}",
                self.low
            );
            mantissa >>= -at;
            at = 0;
        }
        // The digit holding the lowest bit, or the last from which three digits fit:
        // the highest bit lies below digit limbs, so three digits from there hold it.
        let first = (at as usize / DIGIT_BITS as usize).min(self.limbs - LEAF_DIGITS);
        let placed = u128::from(mantissa) << (at as usize - first * DIGIT_BITS as usize);
        debug_assert!(
            placed >> (LEAF_DIGITS as u32 * DIGIT_BITS) == 0,
            "{v} past the span"
        );
        let sign = if v.is_sign_negative() { -1 } else { 1 };
        let digit = |j: u32| sign * ((placed >> (j * DIGIT_BITS)) as i64 & DIGIT_MASK) as i32;
        WideLeaf {
            first: first as u32,
            digits: [digit(0), digit(1), digit(2)],
        }
    }

    /// Add `v` to `acc` exactly. `v` must be finite, with no set bit below 2^low.
    #[inline]
    #[cfg(any(test, feature = "research"))]
    pub fn add(self, acc: &mut [i64], v: f64) {
        assert_eq!(acc.len(), self.limbs);
        let leaf = self.leaf(v);
        // SAFETY: `acc` holds the layout's slots, and the leaf is the layout's.
        unsafe { leaf.add_to(acc.as_mut_ptr()) };
    }

    /// Subtract `v` from `acc` exactly, as [`Self::add`] adds.
    #[inline]
    #[cfg(test)]
    pub fn sub(self, acc: &mut [i64], v: f64) {
        assert_eq!(acc.len(), self.limbs);
        let leaf = self.leaf(v);
        // SAFETY: as in `add`.
        unsafe { leaf.sub_from(acc.as_mut_ptr()) };
    }

    /// Add the accumulator `other` to `acc`, slot by slot.
    #[inline]
    pub fn add_acc(self, acc: &mut [i64], other: &[i64]) {
        debug_assert!(acc.len() == self.limbs && other.len() == self.limbs);
        for (a, &b) in acc.iter_mut().zip(other) {
            *a += b;
        }
    }

    /// Add the accumulators `a` and `b` to `acc`, slot by slot, clear them, and carry
    /// as [`Self::normalize`] does: one pass for two rows' difference entries, each
    /// slot's carry going into the next slot's sum as it is added. Each slot's sum is
    /// the same as adding `a`, then `b`, then the carry: i64 adds are associative
    /// modulo 2^64, and the slots' headroom bounds every partial sum.
    #[inline]
    pub fn add_clear_and_carry(self, acc: &mut [i64], a: &mut [i64], b: &mut [i64]) {
        debug_assert!(acc.len() == self.limbs && a.len() == self.limbs && b.len() == self.limbs);
        let (top, digits) = acc.split_last_mut().expect("three slots or more");
        let (a_top, a) = a.split_last_mut().expect("as many slots");
        let (b_top, b) = b.split_last_mut().expect("as many slots");
        let mut carry = 0;
        for ((d, x), y) in digits.iter_mut().zip(a).zip(b) {
            let v = *d + *x + *y + carry;
            carry = v >> DIGIT_BITS;
            *d = v & DIGIT_MASK;
            (*x, *y) = (0, 0);
        }
        *top += *a_top + *b_top + carry;
        (*a_top, *b_top) = (0, 0);
    }

    /// Carry every slot but the top one into its digit, `0..2^29`, keeping the value:
    /// the top slot takes the carries and the sign.
    #[inline]
    pub fn normalize(self, acc: &mut [i64]) {
        debug_assert_eq!(acc.len(), self.limbs);
        let mut carry = 0;
        let (top, digits) = acc.split_last_mut().expect("three slots or more");
        for d in digits {
            let v = *d + carry;
            carry = v >> DIGIT_BITS;
            *d = v & DIGIT_MASK;
        }
        *top += carry;
    }

    /// The `f64` nearest to `acc`, ties to even, as [`to_f64`](super::to_f64) rounds: zero is +0.0,
    /// results past `f64::MAX` round to infinity, and results in the subnormal range
    /// are exact. Not inlined: its copy of the slots would grow the caller's frame,
    /// which a single row's path shares with the `i128` sums.
    #[inline(never)]
    pub fn to_f64(self, acc: &[i64]) -> f64 {
        debug_assert_eq!(acc.len(), self.limbs);
        // Uninitialized past the slots, which nothing reads: no fill of all
        // MAX_LIMBS before the copy.
        let mut digits = [const { MaybeUninit::<i64>::uninit() }; MAX_LIMBS];
        let digits = digits[..self.limbs].write_copy_of_slice(acc);
        self.normalize(digits);
        self.normalized_to_f64(digits)
    }

    /// [`Self::to_f64`] of an accumulator that [`Self::normalize`] left.
    pub fn normalized_to_f64(self, acc: &[i64]) -> f64 {
        debug_assert_eq!(acc.len(), self.limbs);
        let (&top, digits) = acc.split_last().expect("three slots or more");
        debug_assert!(digits.iter().all(|d| (0..=DIGIT_MASK).contains(d)));
        let negative = top < 0;
        // The value is top × W plus digits below W, which are not negative, where W
        // is the top slot's weight. A negative value's magnitude is then
        // (-top - 1) × W + (W - lower) when lower is nonzero: digit `first` of the
        // lower part, its lowest nonzero one, becomes 2^29 - digit, those above it
        // 2^29 - 1 - digit, and those below stay zero.
        let first = digits.iter().position(|&d| d != 0);
        let h = digits.len();
        let magnitude = |i: usize| -> u64 {
            if !negative {
                return acc[i] as u64;
            }
            match first {
                None if i == h => top.unsigned_abs(),
                None => 0,
                Some(_) if i == h => (-(top + 1)) as u64,
                Some(f) if i < f => 0,
                Some(f) if i == f => (DIGIT_MASK + 1 - acc[i]) as u64,
                Some(_) => (DIGIT_MASK - acc[i]) as u64,
            }
        };
        let Some(high) = (0..=h).rev().find(|&i| magnitude(i) != 0) else {
            return 0.0;
        };
        // The top slot's magnitude may take 63 bits and each digit below 29, so three
        // slots from the highest nonzero one hold its top 64 bits and more. Any digit
        // below them is nonzero exactly when the lower part's lowest nonzero digit is:
        // a positive value's digits are themselves, and a negative one's magnitude is
        // nonzero at `first`.
        let low_slot = high.saturating_sub(LEAF_DIGITS - 1);
        let mut bits = 0u128;
        for i in (low_slot..=high).rev() {
            bits = bits << DIGIT_BITS | u128::from(magnitude(i));
        }
        let sticky = first.is_some_and(|f| f < low_slot);
        nearest(
            bits,
            sticky,
            low_slot as i32 * DIGIT_BITS as i32 + self.low,
            negative,
        )
    }
}

impl WideLeaf {
    /// Add the leaf to the accumulator at `acc`, without bounds checks: the leaf pass
    /// and a single row's sum add one per run end or leaf, where checks took about a
    /// third of the work.
    ///
    /// # Safety
    ///
    /// `acc` points to an accumulator of the leaf's layout, whose [`Wide::limbs`]
    /// slots hold the leaf's digits, `first..first + 3`, as [`Wide::leaf`] places them.
    #[expect(
        clippy::inline_always,
        reason = "the leaf pass adds one per run end, where a call would cost more than the add"
    )]
    #[inline(always)]
    pub unsafe fn add_to(self, acc: *mut i64) {
        for (j, &d) in self.digits.iter().enumerate() {
            // SAFETY: the caller guarantees the slot.
            unsafe { *acc.add(self.first as usize + j) += i64::from(d) };
        }
    }

    /// Subtract the leaf from the accumulator at `acc`, as [`Self::add_to`] adds.
    ///
    /// # Safety
    ///
    /// As for [`Self::add_to`].
    #[expect(
        clippy::inline_always,
        reason = "the leaf pass subtracts one per run end, as add_to adds"
    )]
    #[inline(always)]
    pub unsafe fn sub_from(self, acc: *mut i64) {
        for (j, &d) in self.digits.iter().enumerate() {
            // SAFETY: the caller guarantees the slot.
            unsafe { *acc.add(self.first as usize + j) -= i64::from(d) };
        }
    }
}

/// `magnitude` times 2^k, plus less than one unit of its lowest bit if `sticky`,
/// negated if `negative`, rounded to the nearest `f64`, ties to even.
fn nearest(magnitude: u128, sticky: bool, k: i32, negative: bool) -> f64 {
    let width = 128 - magnitude.leading_zeros() as i32;
    // Keep the top 64 bits and fold the rest and `sticky` into a sticky bit, as in
    // to_f64: it sits below the rounding position, since 64 bits exceed the 54 that
    // rounding reads.
    let (top, shift) = if width > 64 {
        let s = (width - 64) as u32;
        let rest = magnitude & ((1u128 << s) - 1) != 0;
        (
            (magnitude >> s) as u64 | u64::from(rest || sticky),
            s as i32,
        )
    } else {
        // Room below the value: shift it up so that the sticky bit stays below the
        // rounding position.
        let s = (64 - width) as u32;
        (((magnitude << s) as u64) | u64::from(sticky), -(s as i32))
    };
    scaled(top, shift + k, negative)
}

/// A difference array of [`Wide`] accumulators, one per entry, laid out flat: the
/// leaf pass's sink for models past the 126-bit rule.
///
/// Headroom: a tree's leaf entries cover each row once, so an entry receives at most
/// one run end and one run start per tree, across both of the leaf pass's arrays,
/// and entry 0 at most one start or one sum of every row: each slot holds at most
/// `2 × trees` digits, each below 2^29 in magnitude. The finish adds a row's two
/// entries to a running accumulator whose digits it has normalized, below 2^29, and
/// whose top slot holds the previous row's sum, at most `trees` leaves below the
/// layout's span, so at most `trees × 2^29`; with the carries, every slot stays
/// below `(3 × trees + 3) × 2^29`.
///
/// Sums of every row ([`RunSink::add_all_of`]) take turns among [`WIDE_SUMS`]
/// accumulators, which [`RunSink::flush`] adds to entry 0 and clears: consecutive
/// leaves mostly share their slots, and an add to the slots the previous add wrote
/// would wait for its stores.
///
/// The values it adds are leaves of its layout ([`Wide::leaf`]), so that each lands in
/// its entry's slots, which it adds without bounds checks.
pub struct WideRuns<'a> {
    /// Slots per accumulator, the layout's [`Wide::limbs`].
    n: usize,
    /// `limbs.len() / n`, which the leaf pass reads per entry. With `n` in place of
    /// the layout, the sink keeps the size it had without it, and so its callers'
    /// frames.
    entries: usize,
    limbs: &'a mut [i64],
    sums: &'a mut [i64],
}

/// Adds `values` to [`WIDE_SUMS`] accumulators of `n` slots each from `acc`, value
/// `i` to accumulator `(turn + i) % WIDE_SUMS`, and returns how many it added. The
/// accumulators' addresses are computed once, and the loop takes them in turn four
/// values at a time, with no multiply per value.
///
/// # Safety
///
/// `acc` holds `WIDE_SUMS * n` slots, and every value is a leaf of a layout of `n`
/// slots.
#[expect(
    clippy::inline_always,
    reason = "the leaf pass and a single row's sums call this once per chunk; inlined, its values come from the caller's loop"
)]
#[inline(always)]
pub(crate) unsafe fn add_in_turns(
    acc: *mut i64,
    n: usize,
    turn: usize,
    mut values: impl Iterator<Item = WideLeaf>,
) -> usize {
    // SAFETY: each accumulator is within the caller's WIDE_SUMS * n slots.
    let at = |k: usize| unsafe { acc.add((turn + k) % WIDE_SUMS * n) };
    let bases: [_; WIDE_SUMS] = [at(0), at(1), at(2), at(3)];
    let mut added = 0;
    loop {
        for &base in &bases {
            let Some(v) = values.next() else {
                return added;
            };
            // SAFETY: as above.
            unsafe { v.add_to(base) };
            added += 1;
        }
    }
}

/// Accumulators that take turns for a run of adds to the same sum. An add to slots
/// the previous add stored to waits for store-to-load forwarding, 4 to 5 cycles on
/// current cores, while the adds between them issue in 1 to 2 cycles each: four
/// turns keep the slots' loads off the critical path.
pub const WIDE_SUMS: usize = 4;

impl<'a> WideRuns<'a> {
    /// The sink over `limbs`, `wide.limbs()` per entry, whose sums of every row take
    /// turns in `sums`: zeroed, [`WIDE_SUMS`] accumulators or none, if the sink takes
    /// no such sums.
    pub fn new(wide: Wide, limbs: &'a mut [i64], sums: &'a mut [i64]) -> Self {
        debug_assert!(limbs.len().is_multiple_of(wide.limbs));
        assert!(
            sums.is_empty() || sums.len() == WIDE_SUMS * wide.limbs,
            "the sums of every row"
        );
        let n = wide.limbs;
        Self {
            n,
            entries: limbs.len() / n,
            limbs,
            sums,
        }
    }
}

impl RunSink for WideRuns<'_> {
    type Value = WideLeaf;
    /// The next turn among the sums.
    type Sum = usize;

    #[inline]
    fn entries(&self) -> usize {
        self.entries
    }

    #[inline]
    unsafe fn add(&mut self, at: usize, v: WideLeaf) {
        debug_assert!(at < self.entries && v.first as usize + LEAF_DIGITS <= self.n);
        // SAFETY: the caller keeps `at` below the entries, whose slots `limbs` holds,
        // and `v` is a leaf of the layout.
        unsafe { v.add_to(self.limbs.as_mut_ptr().add(at * self.n)) };
    }

    #[inline]
    unsafe fn sub(&mut self, at: usize, v: WideLeaf) {
        debug_assert!(at < self.entries && v.first as usize + LEAF_DIGITS <= self.n);
        // SAFETY: as above.
        unsafe { v.sub_from(self.limbs.as_mut_ptr().add(at * self.n)) };
    }

    #[cfg_attr(
        dispatch,
        expect(
            clippy::inline_always,
            reason = "each level's leaf pass compiles it with the level's features, not the baseline's"
        ),
        inline(always)
    )]
    #[cfg_attr(not(dispatch), inline)]
    fn add_all_of(&mut self, turn: &mut usize, values: impl Iterator<Item = WideLeaf>) {
        assert!(!self.sums.is_empty(), "a sink without sums of every row");
        // SAFETY: the sums are WIDE_SUMS accumulators of `n` slots, and the values
        // are leaves of the layout.
        *turn += unsafe { add_in_turns(self.sums.as_mut_ptr(), self.n, *turn, values) };
    }

    /// One pass over the slots ([`fold`]): the sums' slot sums are those of adding
    /// each sum in turn, as i64 adds are associative modulo 2^64 and the slots'
    /// headroom bounds every partial sum.
    #[cfg_attr(
        dispatch,
        expect(
            clippy::inline_always,
            reason = "each level's leaf pass compiles it with the level's features"
        ),
        inline(always)
    )]
    fn flush(&mut self, _: usize) {
        let n = self.n;
        if self.sums.is_empty() {
            return;
        }
        let (s0, rest) = self.sums.split_at_mut(n);
        let (s1, rest) = rest.split_at_mut(n);
        let (s2, s3) = rest.split_at_mut(n);
        const { assert!(WIDE_SUMS == 4, "one slice per sum") };
        fold(&mut self.limbs[..n], s0, s1, s2, s3);
    }
}

/// Adds the sums `s0` to `s3` to `acc` and clears them, slot by slot. The slices are
/// separate `&mut` arguments so that LLVM knows they do not overlap: read through
/// [`WideRuns`]' fields they might, and it checked their bounds against each other at
/// run time before taking the vector loop. `#[inline]` at every level: LLVM inlines it
/// into each leaf pass, keeping that knowledge, while its copies inlined by
/// `#[inline(always)]` into the levels' trampolines had the checks again.
#[inline]
fn fold(acc: &mut [i64], s0: &mut [i64], s1: &mut [i64], s2: &mut [i64], s3: &mut [i64]) {
    let sums = s0.iter_mut().zip(s1).zip(s2.iter_mut().zip(s3));
    for (a, ((x0, x1), (x2, x3))) in acc.iter_mut().zip(sums) {
        *a += *x0 + *x1 + (*x2 + *x3);
        (*x0, *x1, *x2, *x3) = (0, 0, 0, 0);
    }
}
