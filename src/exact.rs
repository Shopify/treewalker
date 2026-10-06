//! Exact leaf sums.
//!
//! A model has finitely many leaf values, so all of them are integer multiples of
//! 2^-e for some scale e. Times 2^e they are exact integers, their sum over the
//! trees is exact in `i128`, and one conversion back to `f64` rounds once. The
//! prediction is the correctly rounded sum of the leaf values whatever order the
//! leaves are added in, which lets a leaf be added to a run of rows with two writes
//! to a difference array.

/// Smallest e such that every leaf value times 2^e is an integer, provided sums over
/// `n_trees` trees cannot overflow. `None` if a leaf is not finite or the leaf
/// exponents span too wide a range; prediction then sums in `f64`.
pub fn scale(leaf_values: impl IntoIterator<Item = f64>, n_trees: usize) -> Option<i32> {
    let mut min_exp = i32::MAX; // exponent of the lowest set bit of any leaf
    let mut max_top = i32::MIN; // exponent just above the highest set bit of any leaf
    for v in leaf_values {
        if v == 0.0 {
            continue;
        }
        if !v.is_finite() {
            return None;
        }
        let (m, ex) = decompose(v);
        min_exp = min_exp.min(ex + m.trailing_zeros() as i32);
        max_top = max_top.max(ex + 64 - m.leading_zeros() as i32);
    }
    if min_exp == i32::MAX {
        return Some(0);
    }
    let e = -min_exp;
    // A scaled leaf is below 2^(max_top + e), a row's sum over the trees below
    // 2^(max_top + e + tree_bits), and a difference-array entry, which takes at most
    // one run start and one run end per tree, below twice that: at most 2^126.
    let tree_bits = (usize::BITS - n_trees.leading_zeros()) as i32;
    (max_top + e + tree_bits < 126).then_some(e)
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
        let sticky = u64::from(magnitude & ((1u128 << s) - 1) != 0);
        ((magnitude >> s) as u64 | sticky, s as i32)
    } else {
        (magnitude as u64, 0)
    };
    // Scaling by a power of two is exact while the result stays normal.
    let mut r = top as f64;
    let mut k = shift - e;
    while k != 0 {
        let step = k.clamp(-1000, 1000);
        r *= f64::from_bits(((1023 + step) as u64) << 52);
        k -= step;
    }
    if acc < 0 { -r } else { r }
}

/// `v` = ±m × 2^ex with an integer mantissa m.
#[inline]
const fn decompose(v: f64) -> (u64, i32) {
    let bits = v.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1 << 52) - 1);
    if biased == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1 << 52), biased - 1075)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_sum(values: &[f64]) -> f64 {
        let e = scale(values.iter().copied(), values.len()).expect("scale");
        to_f64(values.iter().map(|&v| to_fixed(v, e)).sum(), e)
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "exact results")]
    fn sums_are_correctly_rounded() {
        // Sequential f64 addition loses the 1.0 and doubles the residual.
        assert_eq!(exact_sum(&[1e16, 1.0, -1e16]), 1.0);
        assert_eq!(1e16 + 1.0 - 1e16, 0.0);
        assert_eq!(exact_sum(&[0.1, 0.2, -0.3]), 2f64.powi(-55));
        assert_eq!(0.1 + 0.2 - 0.3, 2f64.powi(-54));
        assert_eq!(exact_sum(&[0.0, -0.0]), 0.0);
        assert_eq!(exact_sum(&[3.0, -2.5]), 0.5);
    }

    #[test]
    fn sums_do_not_depend_on_order() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        let mut values: Vec<f64> = (0..500)
            .map(|_| {
                let unit = (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                unit * 2f64.powi((next() % 20) as i32 - 10)
            })
            .collect();
        let forward = exact_sum(&values);
        values.reverse();
        assert_eq!(exact_sum(&values).to_bits(), forward.to_bits());
        values.sort_by(f64::total_cmp);
        assert_eq!(exact_sum(&values).to_bits(), forward.to_bits());
    }

    #[test]
    fn conversion_matches_integer_cast() {
        let mut state = 1u64;
        for _ in 0..10_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let width = (state % 126) as u32 + 1;
            let raw =
                (i128::from(state) << 64 | i128::from(state.rotate_left(17))) >> (128 - width);
            let acc = if state & 1 == 0 { raw } else { -raw };
            for e in [0, 7, 52, 300] {
                let expected = acc as f64 * 2f64.powi(-e);
                assert_eq!(to_f64(acc, e).to_bits(), expected.to_bits(), "{acc} {e}");
            }
        }
    }

    #[test]
    fn fixed_point_round_trips_leaf_values() {
        let values = [0.75, -3.0e-5, 1.0e3, 2.5e-10, -0.0];
        let e = scale(values, values.len()).unwrap();
        for v in values {
            assert_eq!(to_f64(to_fixed(v, e), e).to_bits(), (v + 0.0).to_bits());
        }
    }

    #[test]
    fn scale_rejects_what_cannot_be_exact() {
        assert_eq!(scale([1.0, f64::INFINITY], 2), None);
        assert_eq!(scale([1.0, f64::NAN], 2), None);
        assert_eq!(scale([1e300, 1e-300], 2), None);
        assert_eq!(scale([0.0, -0.0], 2), Some(0));
        assert_eq!(scale([0.5, 3.0], 2), Some(1));
    }
}
