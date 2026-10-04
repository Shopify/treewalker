//! Generic row-mask abstraction for partial evaluation.
//!
//! [`RowMask`] abstracts the bitmask that tracks which rows in a group are active at
//! each split during tree traversal. Prediction uses `u16` (≤16 rows), `u32` (≤32),
//! `u64` (≤64) and [`Bits<W>`] of `W` 64-bit words above that: one native word keeps
//! the per-predicate mask table small for small groups, and 64-bit words keep the
//! per-word loops short for wide ones.
//!
//! The compiler monomorphizes `partial_eval<..., M: RowMask>` into separate
//! versions per mask width, eliminating all trait dispatch overhead.

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

/// Bitmask over rows in a prediction group.
///
/// Bit `r` is set if row `r` is active (still traversing this subtree).
/// The hot-path operations are AND (partition), OR (accumulate), NOT (complement),
/// `is_zero` (unsplit check), and `trailing_zeros` (iterate active rows).
pub trait RowMask:
    Copy
    + Eq
    + BitAnd<Output = Self>
    + BitAndAssign
    + BitOr<Output = Self>
    + BitOrAssign
    + Not<Output = Self>
    + Send
    + Sync
    + 'static
{
    /// Maximum number of rows this mask type supports.
    const WIDTH: usize;

    /// All bits zero (no active rows).
    const ZERO: Self;

    /// Mask with the first `n` bits set (rows `0..n` active).
    /// Panics if `n > WIDTH`.
    fn from_width(n: usize) -> Self;

    /// True if no bits are set.
    fn is_zero(self) -> bool;

    /// Index of the lowest set bit. Returns `WIDTH` if `self` is zero.
    fn trailing_zeros(self) -> u32;

    /// Clear the lowest set bit: `self & (self - 1)`.
    #[must_use]
    fn clear_lowest(self) -> Self;

    /// Number of set bits.
    fn count_ones(self) -> u32;

    /// Test whether bit `r` is set.
    #[cfg(test)]
    fn test_bit(self, r: usize) -> bool;

    /// Return `self` with bit `r` set.
    #[must_use]
    fn set_bit(self, r: usize) -> Self;

    /// Call `f(start, end)` for every maximal run of set bits `start..end`, in
    /// increasing order.
    fn for_each_run(self, f: impl FnMut(usize, usize));

    /// Call `f` with the index of every set bit, in increasing order.
    fn for_each_bit(self, f: impl FnMut(usize));
}

macro_rules! impl_row_mask {
    ($ty:ty, $width:literal) => {
        impl RowMask for $ty {
            const WIDTH: usize = $width;
            const ZERO: Self = 0;

            #[inline]
            fn from_width(n: usize) -> Self {
                debug_assert!(n <= Self::WIDTH, "from_width({n}) exceeds WIDTH={}", $width);
                if n == $width { Self::MAX } else { (1 << n) - 1 }
            }
            #[inline]
            fn is_zero(self) -> bool {
                self == 0
            }
            #[inline]
            fn trailing_zeros(self) -> u32 {
                self.trailing_zeros()
            }
            #[inline]
            fn clear_lowest(self) -> Self {
                self & self.wrapping_sub(1)
            }
            #[inline]
            fn count_ones(self) -> u32 {
                self.count_ones()
            }
            #[cfg(test)]
            #[inline]
            fn test_bit(self, r: usize) -> bool {
                debug_assert!(r < Self::WIDTH);
                self & (1 << r) != 0
            }
            #[inline]
            fn set_bit(self, r: usize) -> Self {
                debug_assert!(r < Self::WIDTH);
                self | (1 << r)
            }
            #[inline]
            fn for_each_run(self, mut f: impl FnMut(usize, usize)) {
                let mut starts = self & !(self << 1);
                let mut ends = self & !(self >> 1);
                while starts != 0 {
                    f(
                        starts.trailing_zeros() as usize,
                        ends.trailing_zeros() as usize + 1,
                    );
                    starts &= starts.wrapping_sub(1);
                    ends &= ends.wrapping_sub(1);
                }
            }
            #[inline]
            fn for_each_bit(self, mut f: impl FnMut(usize)) {
                let mut m = self;
                while m != 0 {
                    f(m.trailing_zeros() as usize);
                    m &= m.wrapping_sub(1);
                }
            }
        }
    };
}

impl_row_mask!(u16, 16);
impl_row_mask!(u32, 32);
impl_row_mask!(u64, 64);

/// Row mask of `W` 64-bit words (`64 * W` rows); bit `r` is bit `r % 64` of word `r / 64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bits<const W: usize>(pub [u64; W]);

impl<const W: usize> BitAnd for Bits<W> {
    type Output = Self;
    #[inline]
    fn bitand(mut self, rhs: Self) -> Self {
        self &= rhs;
        self
    }
}

impl<const W: usize> BitAndAssign for Bits<W> {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0) {
            *a &= b;
        }
    }
}

impl<const W: usize> BitOr for Bits<W> {
    type Output = Self;
    #[inline]
    fn bitor(mut self, rhs: Self) -> Self {
        self |= rhs;
        self
    }
}

impl<const W: usize> BitOrAssign for Bits<W> {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0) {
            *a |= b;
        }
    }
}

impl<const W: usize> Not for Bits<W> {
    type Output = Self;
    #[inline]
    fn not(mut self) -> Self {
        for a in &mut self.0 {
            *a = !*a;
        }
        self
    }
}

impl<const W: usize> RowMask for Bits<W> {
    const WIDTH: usize = 64 * W;
    const ZERO: Self = Self([0; W]);

    #[inline]
    fn from_width(n: usize) -> Self {
        debug_assert!(
            n <= Self::WIDTH,
            "from_width({n}) exceeds WIDTH={}",
            Self::WIDTH
        );
        let mut words = [0; W];
        for (i, w) in words.iter_mut().enumerate() {
            let below = n.saturating_sub(64 * i);
            *w = if below >= 64 {
                u64::MAX
            } else {
                (1 << below) - 1
            };
        }
        Self(words)
    }
    #[inline]
    fn is_zero(self) -> bool {
        self.0.iter().fold(0, |acc, &w| acc | w) == 0
    }
    #[inline]
    fn trailing_zeros(self) -> u32 {
        for (i, &w) in self.0.iter().enumerate() {
            if w != 0 {
                return (64 * i) as u32 + w.trailing_zeros();
            }
        }
        Self::WIDTH as u32
    }
    #[inline]
    fn clear_lowest(mut self) -> Self {
        if let Some(w) = self.0.iter_mut().find(|w| **w != 0) {
            *w &= *w - 1;
        }
        self
    }
    #[inline]
    fn count_ones(self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
    #[cfg(test)]
    #[inline]
    fn test_bit(self, r: usize) -> bool {
        debug_assert!(r < Self::WIDTH);
        self.0[r / 64] >> (r % 64) & 1 != 0
    }
    #[inline]
    fn set_bit(mut self, r: usize) -> Self {
        debug_assert!(r < Self::WIDTH);
        self.0[r / 64] |= 1 << (r % 64);
        self
    }
    /// Walks the set words only, so a run costs O(1) plus the words it spans.
    #[inline]
    fn for_each_run(self, mut f: impl FnMut(usize, usize)) {
        let mut i = 0;
        while i < W {
            let mut w = self.0[i];
            while w != 0 {
                let lo = w.trailing_zeros() as usize;
                let ones = (w >> lo).trailing_ones() as usize;
                let start = 64 * i + lo;
                if lo + ones < 64 {
                    f(start, start + ones);
                    w &= !(((1 << ones) - 1) << lo);
                    continue;
                }
                // The run reaches the top of word i: extend it through full words.
                i += 1;
                while i < W && self.0[i] == u64::MAX {
                    i += 1;
                }
                if i == W {
                    f(start, Self::WIDTH);
                    return;
                }
                let t = self.0[i].trailing_ones() as usize;
                f(start, 64 * i + t);
                w = self.0[i] & !((1 << t) - 1);
            }
            i += 1;
        }
    }
    #[inline]
    fn for_each_bit(self, mut f: impl FnMut(usize)) {
        for (i, &w) in self.0.iter().enumerate() {
            let mut w = w;
            while w != 0 {
                f(64 * i + w.trailing_zeros() as usize);
                w &= w - 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mask<M: RowMask>() {
        // Zero
        assert!(M::ZERO.is_zero());
        assert_eq!(M::ZERO.count_ones(), 0);

        // from_width
        let m3 = M::from_width(3);
        assert_eq!(m3.count_ones(), 3);
        assert!(m3.test_bit(0));
        assert!(m3.test_bit(1));
        assert!(m3.test_bit(2));
        assert!(!m3.test_bit(3));

        // trailing_zeros
        let m = M::ZERO.set_bit(5);
        assert_eq!(m.trailing_zeros(), 5);

        // clear_lowest
        let m = M::ZERO.set_bit(2).set_bit(5);
        let cleared = m.clear_lowest();
        assert!(!cleared.test_bit(2));
        assert!(cleared.test_bit(5));

        // AND / OR / NOT
        let a = M::from_width(4); // bits 0-3
        let b = M::ZERO.set_bit(2).set_bit(5); // bits 2, 5
        assert_eq!((a & b).count_ones(), 1); // bit 2
        assert_eq!((a | b).count_ones(), 5); // bits 0,1,2,3,5

        // from_width(WIDTH) = all bits set
        let full = M::from_width(M::WIDTH);
        assert_eq!(full.count_ones(), M::WIDTH as u32);

        // Runs: rows 0-1, 3, 5-(WIDTH-1).
        let runs = M::from_width(2) | M::ZERO.set_bit(3) | (full & !M::from_width(5));
        assert_eq!(run_list(runs), [(0, 2), (3, 4), (5, M::WIDTH)]);
        assert_eq!(run_list(full), [(0, M::WIDTH)]);
        assert_eq!(run_list(M::ZERO), []);
        assert_eq!(bits(b), [2, 5]);

        // Random masks against a bit-by-bit scan.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..2000 {
            let mut mask = M::ZERO;
            let density = state % 7;
            for row in 0..M::WIDTH {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if (state >> 33) % 7 <= density {
                    mask = mask.set_bit(row);
                }
            }
            let mut expected = Vec::new();
            let mut row = 0;
            while row < M::WIDTH {
                if mask.test_bit(row) {
                    let begin = row;
                    while row < M::WIDTH && mask.test_bit(row) {
                        row += 1;
                    }
                    expected.push((begin, row));
                } else {
                    row += 1;
                }
            }
            assert_eq!(run_list(mask), expected);
        }
    }

    fn run_list<M: RowMask>(m: M) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        m.for_each_run(|s, e| out.push((s, e)));
        out
    }

    fn bits<M: RowMask>(m: M) -> Vec<usize> {
        let mut out = Vec::new();
        m.for_each_bit(|r| out.push(r));
        out
    }

    #[test]
    fn test_u16_mask() {
        test_mask::<u16>();
    }

    #[test]
    fn test_u32_mask() {
        test_mask::<u32>();
    }

    #[test]
    fn test_u64_mask() {
        test_mask::<u64>();
    }

    #[test]
    fn test_bits_masks() {
        test_mask::<Bits<2>>();
        test_mask::<Bits<4>>();
        test_mask::<Bits<16>>();
    }

    #[test]
    fn test_bits_runs_cross_words() {
        // Rows 60-70 (crosses words 0-1), 127-128 (words 1-2), 191 (end of word 2).
        let mut m = Bits::<4>::ZERO;
        for r in (60..=70).chain([127, 128, 191]) {
            m = m.set_bit(r);
        }
        assert_eq!(run_list(m), [(60, 71), (127, 129), (191, 192)]);
        let long = Bits::<4>::from_width(256) & !Bits::<4>::from_width(3);
        assert_eq!(run_list(long), [(3, 256)]);
        let spans = Bits::<4>::from_width(200) & !Bits::<4>::from_width(10);
        assert_eq!(run_list(spans), [(10, 200)]);
        assert_eq!(bits(m).len(), 14);
        assert_eq!(m.trailing_zeros(), 60);
        assert_eq!(bits(m.clear_lowest())[0], 61);
        assert_eq!(Bits::<4>::from_width(130).count_ones(), 130);
        assert_eq!(run_list(Bits::<4>::from_width(130)), [(0, 130)]);
    }
}
