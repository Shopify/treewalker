//! An exact sum of `f64` values, independent of the library's accumulators: two
//! unsigned big integers in units of 2^-1074, one for the positive and one for the
//! negative values, and a rounding that builds the result's bits by hand.
//!
//! The library's unit tests include this file too.

/// 32-bit limbs for 2,144 bits: a million values below 2^1024 in units of 2^-1074.
const LIMBS: usize = 67;

/// A running exact sum.
#[derive(Clone)]
pub struct Sum {
    positive: [u32; LIMBS],
    negative: [u32; LIMBS],
}

impl Sum {
    pub const fn new() -> Self {
        Self {
            positive: [0; LIMBS],
            negative: [0; LIMBS],
        }
    }

    /// Add a finite value.
    pub fn add(&mut self, v: f64) {
        assert!(v.is_finite(), "{v} is not finite");
        let bits = v.to_bits();
        let biased = (bits >> 52) & 0x7ff;
        let fraction = bits & ((1 << 52) - 1);
        // v = ±mantissa × 2^(position - 1074).
        let (mantissa, position) = if biased == 0 {
            (fraction, 0)
        } else {
            (fraction | 1 << 52, biased as usize - 1)
        };
        let limbs = if bits >> 63 == 0 {
            &mut self.positive
        } else {
            &mut self.negative
        };
        let mut carry = u128::from(mantissa) << (position % 32);
        for limb in &mut limbs[position / 32..] {
            if carry == 0 {
                break;
            }
            let s = u128::from(*limb) + (carry & 0xffff_ffff);
            *limb = s as u32;
            carry = (carry >> 32) + (s >> 32);
        }
        assert!(carry == 0, "sum out of range");
    }

    /// The `f64` nearest to the sum, ties to even; +0.0 for zero.
    pub fn round(&self) -> f64 {
        let (large, small, sign) = if less(&self.positive, &self.negative) {
            (&self.negative, &self.positive, 1u64 << 63)
        } else {
            (&self.positive, &self.negative, 0)
        };
        let mut n = [0u32; LIMBS];
        let mut borrow = 0i64;
        for i in 0..LIMBS {
            let d = i64::from(large[i]) - i64::from(small[i]) - borrow;
            borrow = i64::from(d < 0);
            n[i] = (d + (borrow << 32)) as u32;
        }
        let bit = |i: usize| u64::from(n[i / 32] >> (i % 32) & 1);
        let Some(top) = (0..LIMBS).rev().find(|&i| n[i] != 0) else {
            return 0.0;
        };
        let width = 32 * top + 32 - n[top].leading_zeros() as usize;
        // Below 2^53 units, the sum's bits are the f64's bits: subnormals and the
        // lowest binade share the unit 2^-1074.
        if width <= 53 {
            let q = (0..width).fold(0u64, |q, i| q | bit(i) << i);
            return f64::from_bits(sign | q);
        }
        // Keep 53 bits, then round half to even on the dropped bits.
        let mut s = width - 53;
        let mut q = (0..53).fold(0u64, |q, i| q | bit(s + i) << i);
        let half = bit(s - 1) == 1;
        // A set bit below s - 1: a whole limb, or the low bits of a partial one.
        let sticky = n[..(s - 1) / 32].iter().any(|&l| l != 0)
            || n[(s - 1) / 32] & ((1 << ((s - 1) % 32)) - 1) != 0;
        if half && (sticky || q & 1 == 1) {
            q += 1;
            if q == 1 << 53 {
                q >>= 1;
                s += 1;
            }
        }
        // q × 2^(s - 1074) with 2^52 <= q < 2^53 has biased exponent s + 1.
        let biased = s as u64 + 1;
        if biased >= 0x7ff {
            return f64::from_bits(sign | 0x7ff << 52);
        }
        f64::from_bits(sign | biased << 52 | (q & ((1 << 52) - 1)))
    }
}

fn less(a: &[u32; LIMBS], b: &[u32; LIMBS]) -> bool {
    a.iter().rev().cmp(b.iter().rev()).is_lt()
}

/// The correctly rounded sum of finite values.
pub fn sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut s = Sum::new();
    for v in values {
        s.add(v);
    }
    s.round()
}
