//! Exact leaf sums.
//!
//! A model has finitely many leaf values, so all of them are integer multiples of
//! 2^-e for some scale e. Times 2^e they are exact integers, their sum over the
//! trees is exact in `i128`, and one conversion back to `f64` rounds once. The
//! prediction is the correctly rounded sum of the leaf values whatever order the
//! leaves are added in, which lets a leaf be added to a run of rows with two writes
//! to a difference array.
//!
//! When the leaf exponents span too wide a range for `i128`, a [`Wide`] accumulator
//! holds the same fixed point in as many 29-bit digits as the span needs, each in an
//! `i64` slot with room for every carry, so that adding a leaf adds three digits and
//! carries nothing. A [`Plan`] picks one of the two per model.

#[cfg(test)]
#[path = "../../tests/common/oracle.rs"]
pub(crate) mod oracle;

mod runs;
#[cfg(test)]
mod tests;
mod wide;

pub use runs::RunSink;
pub(crate) use wide::add_in_turns;
pub use wide::{WIDE_SUMS, Wide, WideLeaf, WideRuns};

/// Smallest e such that every leaf value times 2^e is an integer, provided sums over
/// `n_trees` trees cannot overflow. `None` if a leaf is not finite or the leaf
/// exponents span too wide a range for `i128`.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "models pick their sums through Plan")
)]
pub fn scale(leaf_values: impl IntoIterator<Item = f64>, n_trees: usize) -> Option<i32> {
    let (min_exp, max_top) = extent(leaf_values)?;
    fixed_scale(min_exp, max_top, n_trees)
}

/// `v` times 2^e. Exact when `e` is the [`scale`] of a set of values containing `v`.
#[inline]
pub fn to_fixed(v: f64, e: i32) -> i128 {
    let (m, ex) = decompose(v);
    if m == 0 {
        return 0;
    }
    let shift = ex + e;
    debug_assert!(shift >= -(m.trailing_zeros() as i32), "inexact scale");
    let magnitude = if shift >= 0 {
        i128::from(m) << shift
    } else {
        i128::from(m >> -shift)
    };
    if v.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

/// `magnitude >> s` with a sticky bit for the bits shifted out, for `magnitude` below
/// 2^(64 + s) and `s` in `1..=64`, from its 64-bit halves: a 128-bit shift by a
/// variable amount compiles to a generic two-word shift. No shift below reaches 64:
/// the high half moves up by `64 - s < 64`, the low half down by 1 and then
/// `s - 1 < 64` (all of it gone at s = 64), and the sticky test moves the low half's
/// `s` shifted-out bits up by `64 - s`.
#[inline]
pub(crate) fn top_bits(magnitude: u128, s: u32) -> u64 {
    debug_assert!((1..=64).contains(&s) && magnitude.checked_shr(64 + s).unwrap_or(0) == 0);
    let (hi, lo) = ((magnitude >> 64) as u64, magnitude as u64);
    let top = (hi << (64 - s)) | (lo >> 1 >> (s - 1));
    top | u64::from(lo << (64 - s) != 0)
}

/// The `f64` nearest to `acc` times 2^-e, ties to even, for results in the normal range.
#[inline]
pub fn to_f64(acc: i128, e: i32) -> f64 {
    if acc == 0 {
        return 0.0;
    }
    let magnitude = acc.unsigned_abs();
    let width = 128 - magnitude.leading_zeros() as i32;
    // Keep the top 64 bits and fold the rest into a sticky bit: the u64 to f64
    // conversion then rounds as the full integer would, because the sticky bit sits
    // below the rounding position.
    let (top, shift) = if width > 64 {
        let s = (width - 64) as u32;
        (top_bits(magnitude, s), s as i32)
    } else {
        (magnitude as u64, 0)
    };
    scaled(top, shift - e, acc < 0)
}

/// `top` times 2^k, negated if `negative`, rounded once by the conversion of `top`.
///
/// Scaling by a power of two is exact while the result stays normal. A result in the
/// subnormal range is exact too, as long as the value's lowest possible bit is worth
/// at least 2^-1074, the lowest bit of any `f64`: below 2^-1022 it then has at most
/// 52 significant bits, so `top` converts exactly and the scaled result is
/// representable.
#[inline]
pub(super) fn scaled(top: u64, mut k: i32, negative: bool) -> f64 {
    let mut r = top as f64;
    while k != 0 {
        let step = k.clamp(-1000, 1000);
        r *= f64::from_bits(((1023 + step) as u64) << 52);
        k -= step;
    }
    if negative { -r } else { r }
}

/// `v` = ±m × 2^ex with an integer mantissa m.
#[inline]
pub(super) const fn decompose(v: f64) -> (u64, i32) {
    let bits = v.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1 << 52) - 1);
    if biased == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1 << 52), biased - 1075)
    }
}

/// The exponent of the lowest set bit of any nonzero value, and the exponent just
/// above the highest set bit of any. `(i32::MAX, i32::MIN)` if every value is zero,
/// `None` if one is not finite.
pub(super) fn extent(values: impl IntoIterator<Item = f64>) -> Option<(i32, i32)> {
    let mut extent = Extent::default();
    for v in values {
        if !v.is_finite() {
            return None;
        }
        extent.add(v);
    }
    Some((extent.min_exp, extent.max_top))
}

/// The [`extent`] of finite values, added one at a time.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Extent {
    min_exp: i32,
    max_top: i32,
}

impl Default for Extent {
    fn default() -> Self {
        Self {
            min_exp: i32::MAX,
            max_top: i32::MIN,
        }
    }
}

impl Extent {
    /// Add a finite value.
    #[inline]
    pub(crate) fn add(&mut self, v: f64) {
        debug_assert!(v.is_finite());
        if v == 0.0 {
            return;
        }
        let (m, ex) = decompose(v);
        self.min_exp = self.min_exp.min(ex + m.trailing_zeros() as i32);
        self.max_top = self.max_top.max(ex + 64 - m.leading_zeros() as i32);
    }
}

/// Bits for a count of `n_trees`: `n_trees < 2^tree_bits`.
const fn tree_bits(n_trees: usize) -> i32 {
    (usize::BITS - n_trees.leading_zeros()) as i32
}

/// The 126-bit rule of [`scale`], on an [`extent`].
fn fixed_scale(min_exp: i32, max_top: i32, n_trees: usize) -> Option<i32> {
    if min_exp == i32::MAX {
        return Some(0);
    }
    let e = -min_exp;
    // A scaled leaf is below 2^(max_top + e), a row's sum over the trees below
    // 2^(max_top + e + tree_bits), and a difference-array entry, which takes at most
    // one run start and one run end per tree, below twice that: at most 2^126.
    (max_top + e + tree_bits(n_trees) < 126).then_some(e)
}

/// How a model's leaf values are summed exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// In `i128` at the [`scale`] e, with [`to_fixed`] and [`to_f64`].
    Fixed(i32),
    /// In a [`Wide`] accumulator, for leaves past the 126-bit rule.
    Wide(Wide),
}

impl Plan {
    /// The plan for a model's leaf values and tree count: [`Plan::Fixed`] when
    /// [`scale`] finds a scale, otherwise [`Plan::Wide`].
    ///
    /// # Panics
    ///
    /// If a leaf is not finite; the loader rejects such models.
    #[cfg(test)]
    pub fn new(leaf_values: impl IntoIterator<Item = f64>, n_trees: usize) -> Self {
        let (min_exp, max_top) = extent(leaf_values).expect("finite leaf values");
        Self::of(Extent { min_exp, max_top }, n_trees)
    }

    /// The plan for leaf values of `extent` over `n_trees` trees.
    pub(crate) fn of(extent: Extent, n_trees: usize) -> Self {
        let Extent { min_exp, max_top } = extent;
        fixed_scale(min_exp, max_top, n_trees).map_or_else(
            || Self::Wide(Wide::spanning(min_exp, max_top, n_trees)),
            Self::Fixed,
        )
    }
}
