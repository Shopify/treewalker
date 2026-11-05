//! Exact sums against the oracle: correct rounding, order independence, the plans'
//! choice and the wide accumulator's edges.

use super::oracle;
use super::wide::MAX_LIMBS;
use super::*;

fn exact_sum(values: &[f64]) -> f64 {
    let e = scale(values.iter().copied(), values.len()).expect("scale");
    to_f64(values.iter().map(|&v| to_fixed(v, e)).sum(), e)
}

fn wide_sum(wide: Wide, values: &[f64]) -> f64 {
    let mut acc = vec![0; wide.limbs()];
    for &v in values {
        wide.add(&mut acc, v);
    }
    wide.to_f64(&acc)
}

/// The sum through a layout fitted to the values.
fn fitted_sum(values: &[f64]) -> f64 {
    wide_sum(Wide::new(values.iter().copied(), values.len()), values)
}

/// A small linear congruential generator, so the tests need no dependency.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 ^ (self.0 >> 29)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// A finite value with a random sign and mantissa and a biased exponent in
    /// `exponents` (0 for subnormals), a few of them zero.
    fn value(&mut self, exponents: std::ops::RangeInclusive<u64>) -> f64 {
        let span = exponents.end() - exponents.start() + 1;
        let biased = exponents.start() + self.below(span);
        let fraction = match self.below(8) {
            0 => 0,
            1 => 1,
            _ => self.next() & ((1 << 52) - 1),
        };
        let sign = self.below(2) << 63;
        let v = f64::from_bits(sign | biased << 52 | fraction);
        if self.below(50) == 0 { 0.0 * v } else { v }
    }
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
        let raw = (i128::from(state) << 64 | i128::from(state.rotate_left(17))) >> (128 - width);
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

#[test]
fn plans_use_i128_under_the_rule_and_wide_past_it() {
    assert_eq!(Plan::new([0.5, 3.0], 2), Plan::Fixed(1));
    assert_eq!(Plan::new([0.0, -0.0], 2), Plan::Fixed(0));
    // 2^-1 and 2^-124: 124 bits of span, and 1 bit of tree count for one tree,
    // 2 for two.
    let edge = [0.5, 2f64.powi(-124)];
    assert_eq!(Plan::new(edge, 1), Plan::Fixed(124));
    assert_eq!(scale(edge, 2), None);
    let Plan::Wide(wide) = Plan::new(edge, 2) else {
        panic!("past the rule");
    };
    // 124 bits of span: five 29-bit digits.
    assert_eq!((wide.limbs(), wide.low()), (5, -124));
    let Plan::Wide(wide) = Plan::new([1e300, 1e-300], 2) else {
        panic!("past the rule");
    };
    assert_eq!(wide.low(), -1049); // the lowest set bit of 1e-300
    // 1e300's highest bit is 2^996, so 2,046 bits of span.
    assert_eq!(wide.limbs(), 71);
    // Any finite values with any tree count fit in 73 digits.
    let extremes = [f64::MAX, -f64::MAX, f64::from_bits(1), -f64::from_bits(1)];
    let wide = Wide::new(extremes, 1_000_000);
    assert_eq!((wide.limbs(), wide.low()), (73, -1074));
    assert_eq!(Wide::new(extremes, u32::MAX as usize).limbs(), MAX_LIMBS);
}

#[test]
#[should_panic(expected = "finite leaf values")]
fn plans_need_finite_leaves() {
    let _ = Plan::new([1.0, f64::INFINITY], 2);
}

#[test]
fn wide_sums_match_the_oracle_past_the_rule() {
    let mut rng = Lcg(7);
    for case in 0..2_000 {
        let n = 1 + rng.below(40) as usize;
        // Exponent windows from a few bits wide to the whole range.
        let lo = rng.below(2047);
        let hi = (lo + rng.below(2047 - lo)).min(2046);
        let values: Vec<f64> = (0..n).map(|_| rng.value(lo..=hi)).collect();
        let expected = oracle::sum(values.iter().copied());
        assert_eq!(
            fitted_sum(&values).to_bits(),
            expected.to_bits(),
            "case {case}: {values:?}"
        );
        // The widest layout gives the same result.
        let widest = Wide::new([f64::MAX, f64::from_bits(1)], 1_000_000);
        assert_eq!(wide_sum(widest, &values).to_bits(), expected.to_bits());
    }
}

#[test]
fn wide_sums_agree_with_i128_where_both_apply() {
    let mut rng = Lcg(11);
    let mut fixed = 0;
    for _ in 0..2_000 {
        let n = 1 + rng.below(200) as usize;
        let lo = 900 + rng.below(200);
        let values: Vec<f64> = (0..n).map(|_| rng.value(lo..=lo + 40)).collect();
        if let Some(e) = scale(values.iter().copied(), n) {
            fixed += 1;
            let sum: i128 = values.iter().map(|&v| to_fixed(v, e)).sum();
            let expected = oracle::sum(values.iter().copied());
            assert_eq!(to_f64(sum, e).to_bits(), expected.to_bits(), "{values:?}");
            assert_eq!(fitted_sum(&values).to_bits(), expected.to_bits());
        }
    }
    assert!(fixed > 1_000, "{fixed}");
    // Subnormal results under the rule.
    let tiny = [
        f64::from_bits(3),
        -f64::from_bits(1),
        f64::from_bits(1 << 51),
    ];
    let e = scale(tiny, 3).unwrap();
    let sum: i128 = tiny.iter().map(|&v| to_fixed(v, e)).sum();
    assert_eq!(to_f64(sum, e).to_bits(), (1 << 51) + 2);
    assert_eq!(fitted_sum(&tiny).to_bits(), (1 << 51) + 2);
}

#[test]
fn wide_sums_round_at_the_edges() {
    let min = f64::from_bits(1);
    let half_ulp_of_max = 2f64.powi(970);
    let cases: Vec<(Vec<f64>, f64)> = vec![
        // Ties to even, decided by a bit 1,000 places below.
        (vec![1.0, 2f64.powi(-53)], 1.0),
        (vec![1.0, 2f64.powi(-53), min], 1.0 + f64::EPSILON),
        (vec![1.0, 2f64.powi(-53), -min], 1.0),
        (
            vec![1.0 + f64::EPSILON, 2f64.powi(-53)],
            f64::from_bits(1.0f64.to_bits() + 2),
        ),
        (vec![-1.0, -(2f64.powi(-53)), -min], -1.0 - f64::EPSILON),
        // Cancellation down to the smallest subnormal.
        (vec![1e300, min, -1e300], min),
        (vec![1e300, -min, -1e300, 1e-300], 1e-300),
        (vec![f64::MAX, -f64::MAX, -min], -min),
        (vec![1e16, 1.0, -1e16, 1e-320], 1.0),
        // Signed zeros sum to +0.0.
        (vec![-0.0, -0.0], 0.0),
        (vec![-0.0, 1e300, -1e300, 1e-300, -1e-300], 0.0),
        // Subnormal results are exact.
        (vec![2f64.powi(-1022), -min], f64::from_bits((1 << 52) - 1)),
        (vec![1e300, -1e300, min, min, min], f64::from_bits(3)),
        // Overflow to infinity at f64::MAX plus half an ulp.
        (vec![f64::MAX, half_ulp_of_max], f64::INFINITY),
        (vec![f64::MAX, half_ulp_of_max, -min], f64::MAX),
        (vec![-f64::MAX, -f64::MAX, 1.0], f64::NEG_INFINITY),
        (vec![f64::MAX, f64::MAX, -f64::MAX], f64::MAX),
    ];
    for (values, expected) in cases {
        let oracle = oracle::sum(values.iter().copied());
        assert_eq!(oracle.to_bits(), expected.to_bits(), "oracle {values:?}");
        assert_eq!(
            fitted_sum(&values).to_bits(),
            expected.to_bits(),
            "{values:?}"
        );
    }
    assert_eq!(fitted_sum(&[-0.0, -0.0]).to_bits(), 0.0f64.to_bits());
}

#[test]
fn wide_sums_do_not_depend_on_order() {
    let mut rng = Lcg(3);
    let mut values: Vec<f64> = (0..300).map(|_| rng.value(0..=2046)).collect();
    let wide = Wide::new(values.iter().copied(), values.len());
    let sum = |values: &[f64]| {
        let mut acc = vec![0; wide.limbs()];
        for &v in values {
            wide.add(&mut acc, v);
        }
        acc
    };
    let forward = sum(&values);
    values.reverse();
    assert_eq!(sum(&values), forward);
    values.sort_by(f64::total_cmp);
    assert_eq!(sum(&values), forward);
    for i in 1..values.len() {
        values.swap(i, rng.below(i as u64 + 1) as usize);
    }
    assert_eq!(sum(&values), forward);
    assert_eq!(
        wide.to_f64(&forward).to_bits(),
        oracle::sum(values.iter().copied()).to_bits()
    );
}

#[test]
fn wide_difference_arrays_sum_runs_exactly() {
    // Each leaf adds to a run of rows: + at its start and - at its end. Entries
    // hold digits that the prefix sums carry; the prefix sums are the rows' exact sums.
    let mut rng = Lcg(5);
    for _ in 0..200 {
        let (rows, trees) = (1 + rng.below(40) as usize, 1 + rng.below(30) as usize);
        let leaves: Vec<f64> = (0..trees * 4).map(|_| rng.value(0..=2046)).collect();
        let wide = Wide::new(leaves.iter().copied(), trees);
        let n = wide.limbs();
        let mut diff = vec![0i64; (rows + 1) * n];
        let mut expected = vec![Vec::new(); rows];
        for t in 0..trees {
            // Split the rows into runs, each with one of the tree's leaves.
            let mut a = 0;
            while a < rows {
                let b = (a + 1 + rng.below(rows as u64) as usize).min(rows);
                let v = leaves[t * 4 + rng.below(4) as usize];
                wide.add(&mut diff[a * n..(a + 1) * n], v);
                wide.sub(&mut diff[b * n..(b + 1) * n], v);
                for row in &mut expected[a..b] {
                    row.push(v);
                }
                a = b;
            }
        }
        let mut acc = vec![0; n];
        for (r, row) in expected.iter().enumerate() {
            wide.add_acc(&mut acc, &diff[r * n..(r + 1) * n]);
            let expected = oracle::sum(row.iter().copied());
            assert_eq!(wide.to_f64(&acc).to_bits(), expected.to_bits(), "{row:?}");
        }
        wide.add_acc(&mut acc, &diff[rows * n..]);
        assert!(acc.iter().all(|&l| l == 0));
    }
}

#[test]
fn wide_sums_hold_a_million_extremes() {
    let wide = Wide::new([f64::MAX, f64::from_bits(1)], 1_000_000);
    let mut acc = vec![0; wide.limbs()];
    let mut oracle = oracle::Sum::new();
    for i in 0..1_000_000 {
        let v = if i % 3 == 2 { -f64::MAX } else { f64::MAX };
        wide.add(&mut acc, v);
        oracle.add(v);
    }
    wide.add(&mut acc, f64::from_bits(1));
    oracle.add(f64::from_bits(1));
    assert_eq!(oracle.round().to_bits(), f64::INFINITY.to_bits());
    assert_eq!(wide.to_f64(&acc).to_bits(), f64::INFINITY.to_bits());
    // Subtracting it all again leaves the smallest subnormal, exactly.
    for i in 0..1_000_000 {
        wide.sub(&mut acc, if i % 3 == 2 { -f64::MAX } else { f64::MAX });
    }
    assert_eq!(wide.to_f64(&acc).to_bits(), 1);
}

#[test]
fn top_bits_shift_by_halves() {
    // Against the 128-bit shift and mask, at every shift with magnitudes at the
    // width's edges and random below them.
    let mut rng = Lcg(41);
    for s in 1..=64u32 {
        let width = 64 + s;
        let top = 1u128 << (width - 1);
        let ones = u128::MAX >> (128 - width);
        let random = |rng: &mut Lcg| {
            let m = (u128::from(rng.next()) << 64 | u128::from(rng.next())) & ones;
            m | top
        };
        let mut cases = vec![
            top,
            ones,
            top | 1,
            top | (1u128 << (s - 1)),
            top | ((1u128 << s) - 1),
        ];
        cases.extend((0..200).map(|_| random(&mut rng)));
        for m in cases {
            let want = (m >> s) as u64 | u64::from(m & ((1u128 << s) - 1) != 0);
            assert_eq!(top_bits(m, s), want, "{m:#x} >> {s}");
        }
    }
}

#[test]
fn wide_sums_in_turns_and_their_flush_sum_exactly() {
    // Leaves added to every row through the sums' turns, then flushed into entry 0,
    // give entry 0 the slot sums of adding each leaf there, from any first turn.
    let mut rng = Lcg(43);
    for count in [0usize, 1, 3, 4, 5, 8, 13, 100] {
        let values: Vec<f64> = (0..count).map(|_| rng.value(0..=2046)).collect();
        let wide = Wide::new(values.iter().copied(), count.max(1));
        let n = wide.limbs();
        let mut want = vec![0i64; 2 * n];
        for &v in &values {
            wide.add(&mut want[..n], v);
        }
        for first in 0..WIDE_SUMS {
            let (mut limbs, mut sums) = (vec![0i64; 2 * n], vec![0i64; WIDE_SUMS * n]);
            let mut sink = WideRuns::new(wide, &mut limbs, &mut sums);
            let mut turn = first;
            sink.add_all_of(&mut turn, values.iter().map(|&v| wide.leaf(v)));
            assert_eq!(turn, first + count);
            sink.flush(turn);
            assert_eq!(limbs, want, "{count} values from turn {first}");
            assert!(sums.iter().all(|&x| x == 0));
        }
    }
}

#[test]
fn wide_adds_carry_in_the_same_pass() {
    // add_clear_and_carry against adding both rows' slots and then normalizing.
    let mut rng = Lcg(47);
    for _ in 0..200 {
        let values: Vec<f64> = (0..30).map(|_| rng.value(0..=2046)).collect();
        let wide = Wide::new(values.iter().copied(), values.len());
        let n = wide.limbs();
        let mut acc = vec![0i64; n];
        let (mut a, mut b) = (vec![0i64; n], vec![0i64; n]);
        for (i, &v) in values.iter().enumerate() {
            if i % 3 == 0 {
                wide.sub(&mut a, v);
            } else {
                wide.add(if i % 2 == 0 { &mut a } else { &mut b }, v);
            }
        }
        let mut want = acc.clone();
        for ((s, x), y) in want.iter_mut().zip(&a).zip(&b) {
            *s += *x + *y;
        }
        wide.normalize(&mut want);
        wide.add_clear_and_carry(&mut acc, &mut a, &mut b);
        assert_eq!(acc, want);
        assert!(a.iter().chain(&b).all(|&x| x == 0));
        assert_eq!(
            wide.to_f64(&acc).to_bits(),
            wide.normalized_to_f64(&want).to_bits()
        );
    }
}
