//! Generic row-mask abstraction for partial evaluation.
//!
//! [`RowMask`] abstracts the u32/u64/u128 bitmask that tracks which rows in a
//! group are active at each split during tree traversal. The trait is implemented
//! for `u32` (≤32 rows), `u64` (≤64 rows), and `u128` (≤128 rows).
//!
//! All methods are trivial one-liners that compile to single instructions.
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
    fn test_bit(self, r: usize) -> bool;

    /// Return `self` with bit `r` set.
    #[must_use]
    fn set_bit(self, r: usize) -> Self;
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
            #[inline] fn is_zero(self) -> bool { self == 0 }
            #[inline] fn trailing_zeros(self) -> u32 { self.trailing_zeros() }
            #[inline] fn clear_lowest(self) -> Self { self & self.wrapping_sub(1) }
            #[inline] fn count_ones(self) -> u32 { self.count_ones() }
            #[inline] fn test_bit(self, r: usize) -> bool { debug_assert!(r < Self::WIDTH); self & (1 << r) != 0 }
            #[inline] fn set_bit(self, r: usize) -> Self { debug_assert!(r < Self::WIDTH); self | (1 << r) }
        }
    };
}

impl_row_mask!(u32, 32);
impl_row_mask!(u64, 64);
impl_row_mask!(u128, 128);

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
        let a = M::from_width(4);  // bits 0-3
        let b = M::ZERO.set_bit(2).set_bit(5);  // bits 2, 5
        assert_eq!((a & b).count_ones(), 1); // bit 2
        assert_eq!((a | b).count_ones(), 5); // bits 0,1,2,3,5

        // from_width(WIDTH) = all bits set
        let full = M::from_width(M::WIDTH);
        assert_eq!(full.count_ones(), M::WIDTH as u32);
    }

    #[test]
    fn test_u32_mask() { test_mask::<u32>(); }

    #[test]
    fn test_u64_mask() { test_mask::<u64>(); }

    #[test]
    fn test_u128_mask() { test_mask::<u128>(); }
}
