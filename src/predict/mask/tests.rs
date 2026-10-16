//! The new masks against the plain-word reference in `scalar` and a row-by-row scan.

use super::ops::{WordMaps, add_wide_runs};
use super::rows::{self, Unit};
use super::{Mask, Wide, scalar};
use crate::exact::RunSink;

/// Group sizes at and around every word and width boundary.
const SIZES: [usize; 19] = [
    1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024,
];

/// Miri runs a subset: the listed group sizes, each fixed mask once and a few random
/// masks.
const MIRI: bool = cfg!(miri);

/// A mask's words, padded with zeros to 16 words, so every width shares the reference.
trait Words: Mask + std::fmt::Debug {
    fn words(&self) -> [u64; 16];
    fn from_words(w: &[u64; 16]) -> Self;
    /// [`Mask::add_runs`] through the reference word maps, for wide masks.
    fn add_runs_reference<S: RunSink + ?Sized>(&self, v: S::Value, sink: &mut S) {
        self.add_runs(v, sink);
    }
    /// For wide masks, checks the vector word maps against the reference.
    fn check_maps(&self) {}
    /// For wide masks, checks the vector `first(n)` against the reference.
    fn check_first(&self, _n: usize) {}
}

impl Words for u32 {
    fn words(&self) -> [u64; 16] {
        let mut w = [0; 16];
        w[0] = u64::from(*self);
        w
    }
    fn from_words(w: &[u64; 16]) -> Self {
        w[0] as Self
    }
}

impl Words for u64 {
    fn words(&self) -> [u64; 16] {
        let mut w = [0; 16];
        w[0] = *self;
        w
    }
    fn from_words(w: &[u64; 16]) -> Self {
        w[0]
    }
}

macro_rules! impl_words {
    ($($w:literal),*) => {$(
        impl Words for Wide<$w> {
            fn words(&self) -> [u64; 16] {
                let mut w = [0; 16];
                w[..$w].copy_from_slice(&self.0);
                w
            }
            fn from_words(w: &[u64; 16]) -> Self {
                Self(w[..$w].try_into().unwrap())
            }
            fn add_runs_reference<S: RunSink + ?Sized>(&self, v: S::Value, sink: &mut S) {
                add_wide_runs::<$w, scalar::Maps, S>(&self.0, v, sink);
            }
            fn check_maps(&self) {
                check_maps::<$w>(&self.0);
            }
            fn check_first(&self, n: usize) {
                assert_eq!(super::simd::first::<$w>(n), scalar::first::<$w>(n));
            }
        }
    )*};
}

impl_words!(2, 4, 8, 16);

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn bit(w: &[u64; 16], r: usize) -> bool {
    w[r / 64] >> (r % 64) & 1 != 0
}

fn set_range(w: &mut [u64; 16], start: usize, end: usize) {
    for r in start..end {
        w[r / 64] |= 1 << (r % 64);
    }
}

/// Rows `0..n`.
fn rows(n: usize) -> [u64; 16] {
    let mut w = [0; 16];
    set_range(&mut w, 0, n);
    w
}

/// A random mask over rows `0..n`, one of several shapes: random bits of a random
/// density, runs of short, word-sized or long lengths, which cross word boundaries, or
/// one run.
fn random(rng: &mut Rng, n: usize) -> [u64; 16] {
    let mut w = [0; 16];
    match rng.below(6) {
        0..=1 => {
            for x in &mut w {
                *x = match rng.below(4) {
                    0 => rng.next() & rng.next() & rng.next(),
                    1 => rng.next(),
                    2 => rng.next() | rng.next(),
                    _ => !(rng.next() & rng.next() & rng.next() & rng.next()),
                };
            }
        }
        2..=4 => {
            let scale = [4, 80, 400][rng.below(3)];
            let mut r = rng.below(scale.min(n));
            while r < n {
                let len = 1 + rng.below(scale);
                set_range(&mut w, r, (r + len).min(n));
                r += len + 1 + rng.below(scale);
            }
        }
        _ => {
            let start = rng.below(n);
            let end = start + 1 + rng.below(n - start);
            set_range(&mut w, start, end);
        }
    }
    scalar::and(&w, &rows(n))
}

/// Fixed masks over rows `0..n`: empty, full, single rows at both ends and at word
/// boundaries, both ends, and full but for one row.
fn fixed(n: usize) -> Vec<[u64; 16]> {
    let mut out = vec![[0; 16], rows(n)];
    let mut edges = vec![0, n - 1];
    for b in (64..n).step_by(64) {
        edges.extend([b - 1, b]);
    }
    for &r in &edges {
        let mut one = [0; 16];
        set_range(&mut one, r, r + 1);
        out.push(one);
        out.push(scalar::xor(&rows(n), &one));
    }
    let mut ends = [0; 16];
    set_range(&mut ends, 0, 1);
    set_range(&mut ends, n - 1, n);
    out.push(ends);
    out
}

/// Maximal runs of set rows, by a row-by-row scan.
fn runs(w: &[u64; 16]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut r = 0;
    while r < 1024 {
        if bit(w, r) {
            let start = r;
            while r < 1024 && bit(w, r) {
                r += 1;
            }
            out.push((start, r));
        } else {
            r += 1;
        }
    }
    out
}

/// Records every write, with the entry count it checks against.
struct Record {
    entries: usize,
    writes: Vec<(usize, i64)>,
}

impl RunSink for Record {
    type Value = i64;
    type Sum = ();

    fn entries(&self) -> usize {
        self.entries
    }

    unsafe fn add(&mut self, at: usize, v: i64) {
        assert!(at < self.entries);
        self.writes.push((at, v));
    }

    unsafe fn sub(&mut self, at: usize, v: i64) {
        assert!(at < self.entries);
        self.writes.push((at, -v));
    }

    fn add_all_of(&mut self, (): &mut (), values: impl Iterator<Item = i64>) {
        self.writes.extend(values.map(|v| (0, v)));
    }

    fn flush(&mut self, (): ()) {}
}

fn check_runs<M: Words>(m: M, n: usize, v: i64) {
    let w = m.words();
    let expected = {
        let mut writes: Vec<_> = runs(&w)
            .into_iter()
            .flat_map(|(start, end)| [(start, v), (end, -v)])
            .collect();
        writes.sort_unstable();
        writes
    };
    let mut record = Record {
        entries: n + 1,
        writes: Vec::new(),
    };
    m.add_runs(v, &mut record);
    record.writes.sort_unstable();
    assert_eq!(record.writes, expected, "{m:?}");

    let mut reference = Record {
        entries: n + 1,
        writes: Vec::new(),
    };
    m.add_runs_reference(v, &mut reference);
    reference.writes.sort_unstable();
    assert_eq!(reference.writes, expected, "{m:?}");
    m.check_maps();

    // The difference array sums back to the mask.
    let mut diff = vec![0i128; n + 1];
    m.add_runs(i128::from(v), diff.as_mut_slice());
    let mut sum = 0;
    for (r, d) in diff.iter().enumerate() {
        sum += d;
        let want = if r < n && bit(&w, r) {
            i128::from(v)
        } else {
            0
        };
        assert_eq!(sum, want, "row {r} of {m:?}");
    }
}

/// The vector nonzero-word map of `w` against the reference's, word by word: each
/// word's `stride` bits all set or all clear, and no bits past the last word.
fn check_maps<const W: usize>(w: &[u64; W]) {
    type Fast = super::simd::Maps;
    type Reference = scalar::Maps;
    let stride = Fast::stride::<W>();
    assert!(stride >= 1 && stride as usize * W <= 64);
    let group = u64::MAX >> (64 - stride);
    let reference = Reference::nonzero(w);
    let mut want = 0;
    for i in 0..W {
        if reference >> i & 1 != 0 {
            want |= group << (stride as usize * i);
        }
    }
    assert_eq!(Fast::nonzero(w), want, "{w:x?}");
    assert_eq!(Fast::popcount(w), Reference::popcount(w));
}

fn check_pair<M: Words>(a: [u64; 16], b: [u64; 16], n: usize, rng: &mut Rng) {
    let (ma, mb) = (M::from_words(&a), M::from_words(&b));
    assert_eq!(ma.words(), a);
    assert_eq!(ma.and(mb).words(), scalar::and(&a, &b));
    assert_eq!(ma.and_not(mb).words(), scalar::and_not(&a, &b));
    assert_eq!(ma.or(mb).words(), scalar::or(&a, &b));
    assert_eq!(ma.xor(mb).words(), scalar::xor(&a, &b));
    assert_eq!(ma.is_zero(), scalar::is_zero(&a));
    assert_eq!(ma.popcount(), scalar::popcount(&a));
    assert_eq!(ma == mb, a == b);
    assert_eq!(ma, M::from_words(&a));
    for c in [false, true] {
        assert_eq!(M::select(c, ma, mb).words(), scalar::select(c, &a, &b));
    }
    if n > 0 {
        let r = rng.below(n);
        let mut flipped = a;
        flipped[r / 64] ^= 1 << (r % 64);
        assert_ne!(ma, M::from_words(&flipped));
        let mut once = ma;
        once.flip(r);
        assert_eq!(once.words(), flipped, "flip({r})");
        once.flip(r);
        assert_eq!(once, ma, "flip({r}) twice");
    }
    ma.check_first(n);
    check_runs(ma, n, 1 + rng.below(1000) as i64);
}

fn check<M: Words>(seed: u64) {
    let mut rng = Rng(seed);
    let width = M::WIDTH;

    // Construction.
    assert_eq!(M::ZERO.words(), [0; 16]);
    assert!(M::ZERO.is_zero());
    assert_eq!(M::ones_if(false), M::ZERO);
    assert_eq!(M::ones_if(true).words(), rows(width));
    assert_eq!(M::first(0), M::ZERO);
    for n in (1..=width).filter(|n| !MIRI || SIZES.contains(n)) {
        assert_eq!(M::first(n).words(), rows(n), "first({n})");
        let mut m = M::ZERO;
        m.set(n - 1);
        let mut one = [0; 16];
        set_range(&mut one, n - 1, n);
        assert_eq!(m.words(), one, "set({})", n - 1);
        m.set(n - 1);
        assert_eq!(m.words(), one, "set twice");
    }
    let mut m = M::ZERO;
    for r in (0..width).rev().step_by(3) {
        m.set(r);
    }
    assert!((0..width).all(|r| bit(&m.words(), r) == ((width - 1 - r) % 3 == 0)));

    // Operations on every group size this width holds.
    let sizes = SIZES
        .iter()
        .copied()
        .chain((0..if MIRI { 0 } else { 8 }).map(|_| 1 + rng.below(width)))
        .filter(|&n| n <= width)
        .collect::<Vec<_>>();
    for n in sizes {
        let fixed = fixed(n);
        for (i, &a) in fixed.iter().enumerate() {
            if MIRI {
                check_pair::<M>(a, fixed[(i + 1) % fixed.len()], n, &mut rng);
                continue;
            }
            for &b in &fixed {
                check_pair::<M>(a, b, n, &mut rng);
            }
        }
        for _ in 0..if MIRI { 4 } else { 300 } {
            let a = random(&mut rng, n);
            let b = match rng.below(3) {
                0 => a,
                1 => fixed[rng.below(fixed.len())],
                _ => random(&mut rng, n),
            };
            check_pair::<M>(a, b, n, &mut rng);
        }
    }
}

#[test]
fn mask_32() {
    check::<u32>(1);
}

#[test]
fn mask_64() {
    check::<u64>(2);
}

#[test]
fn mask_128() {
    check::<Wide<2>>(3);
}

#[test]
fn mask_256() {
    check::<Wide<4>>(4);
}

#[test]
fn mask_512() {
    check::<Wide<8>>(5);
}

#[test]
fn mask_1024() {
    check::<Wide<16>>(6);
}

/// The reference itself on known words.
#[test]
fn scalar_reference() {
    let a = [0b1100, u64::MAX];
    let b = [0b1010, 1];
    assert_eq!(scalar::and(&a, &b), [0b1000, 1]);
    assert_eq!(scalar::and_not(&a, &b), [0b0100, u64::MAX - 1]);
    assert_eq!(scalar::or(&a, &b), [0b1110, u64::MAX]);
    assert_eq!(scalar::xor(&a, &b), [0b0110, u64::MAX - 1]);
    assert_eq!(scalar::popcount(&a), 66);
    assert!(scalar::is_zero(&[0u64; 4]));
    assert!(!scalar::is_zero(&[0, 0, 0, 1 << 63]));
    assert_eq!(scalar::select(true, &a, &b), a);
    assert_eq!(scalar::select(false, &a, &b), b);
}

#[test]
fn runs_cross_words() {
    // Rows 60-70 (words 0-1), 127-128 (words 1-2), 191 (end of word 2), and 192-255.
    let mut m = Wide::<4>::ZERO;
    for r in (60..=70).chain([127, 128, 191]).chain(192..256) {
        m.set(r);
    }
    let mut diff = [0i128; 257];
    m.add_runs(1, &mut diff[..]);
    let nonzero: Vec<_> = (0..257)
        .filter(|&i| diff[i] != 0)
        .map(|i| (i, diff[i]))
        .collect();
    assert_eq!(
        nonzero,
        [(60, 1), (71, -1), (127, 1), (129, -1), (191, 1), (256, -1)]
    );
}

#[test]
#[should_panic(expected = "a run ends past the sink")]
fn word_run_past_sink() {
    let mut diff = [0i128; 64];
    u64::MAX.add_runs(1, &mut diff[..]);
}

#[test]
#[should_panic(expected = "a run ends past the sink")]
fn wide_single_run_past_sink() {
    let mut diff = [0i128; 200];
    Wide::<4>::first(200).add_runs(1, &mut diff[..]);
}

#[test]
#[should_panic(expected = "a run ends past the sink")]
fn wide_runs_past_sink() {
    let mut diff = [0i128; 200];
    let mut m = Wide::<4>::ZERO;
    m.set(0);
    m.set(199);
    m.add_runs(1, &mut diff[..]);
}

#[test]
fn empty_mask_needs_no_entries() {
    let mut diff: [i128; 0] = [];
    0u32.add_runs(1, &mut diff[..]);
    Wide::<2>::ZERO.add_runs(1, &mut diff[..]);
}

/// A random runtime-width mask over rows `0..n` in `units(n)` units: random bits of a
/// random density, runs of short to long lengths, or one run.
fn random_rows(rng: &mut Rng, n: usize) -> Vec<Unit> {
    let mut m = vec![Unit::ZERO; rows::units(n)];
    let w = rows::words_mut(&mut m);
    let set = |w: &mut [u64], start: usize, end: usize| {
        for r in start..end {
            w[r / 64] |= 1 << (r % 64);
        }
    };
    match rng.below(5) {
        0..=1 => {
            for x in w.iter_mut() {
                *x = match rng.below(5) {
                    0 => 0,
                    1 => rng.next() & rng.next() & rng.next(),
                    2 => rng.next(),
                    3 => u64::MAX,
                    _ => rng.next() | rng.next(),
                };
            }
        }
        2..=3 => {
            let scale = [4, 80, 400, 3000][rng.below(4)];
            let mut r = rng.below(scale.min(n));
            while r < n {
                let len = 1 + rng.below(scale);
                set(w, r, (r + len).min(n));
                r += len + 1 + rng.below(scale);
            }
        }
        _ => {
            let start = rng.below(n);
            let end = start + 1 + rng.below(n - start);
            set(w, start, end);
        }
    }
    for (r, x) in w.iter_mut().enumerate() {
        let k = n.saturating_sub(64 * r).min(64);
        *x &= u64::MAX.checked_shr(64 - k as u32).unwrap_or(0);
    }
    m
}

/// The span of `m` by a scan of its words.
fn span_of(m: &[Unit]) -> rows::Span {
    let w = rows::words(m);
    match (
        w.iter().position(|&x| x != 0),
        w.iter().rposition(|&x| x != 0),
    ) {
        (Some(lo), Some(last)) => rows::Span { lo, hi: last + 1 },
        _ => rows::Span::EMPTY,
    }
}

fn row_bit(m: &[Unit], r: usize) -> bool {
    rows::words(m)[r / 64] >> (r % 64) & 1 != 0
}

/// Runtime-width masks against a row-by-row scan, at and around word, unit and
/// 1,024-row boundaries and beyond.
#[test]
#[expect(clippy::many_single_char_names, reason = "masks and their sides")]
fn runtime_width() {
    let mut rng = Rng(7);
    let sizes: &[usize] = if MIRI {
        &[1, 65, 128, 129, 1025]
    } else {
        &[
            1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 1100, 2047, 2048,
            2049, 4096, 5000,
        ]
    };
    for &n in sizes {
        let u = rows::units(n);
        assert_eq!(u, n.div_ceil(128));
        let mut all = vec![Unit::ZERO; u];
        rows::first(&mut all, n);
        assert!(
            (0..128 * u).all(|r| row_bit(&all, r) == (r < n)),
            "first({n})"
        );
        assert_eq!(rows::Span::first(n), span_of(&all), "first({n})'s span");
        for _ in 0..if MIRI { 3 } else { 200 } {
            let a = random_rows(&mut rng, n);
            let b = match rng.below(4) {
                0 => a.clone(),
                1 => all.clone(),
                _ => random_rows(&mut rng, n),
            };
            // The split, to separate slots and with the heavy side over the rows, of the
            // whole mask and of its span alone, with junk in the units outside it.
            let span = span_of(&a);
            let (mut heavy, mut light) = (vec![Unit::ZERO; u], vec![Unit::ZERO; u]);
            // SAFETY: each buffer holds u units, and none overlap.
            let (hs, ls) = unsafe {
                rows::split(
                    a.as_ptr(),
                    b.as_ptr(),
                    heavy.as_mut_ptr(),
                    light.as_mut_ptr(),
                    rows::Span { lo: 0, hi: 2 * u },
                )
            };
            for r in 0..128 * u {
                let (x, y) = (row_bit(&a, r), row_bit(&b, r));
                assert_eq!(row_bit(&heavy, r), x && y);
                assert_eq!(row_bit(&light, r), x && !y);
            }
            assert_eq!((hs, ls), (span_of(&heavy), span_of(&light)), "split spans");
            let units = span.units();
            let junk = |m: &mut Vec<Unit>| {
                for (i, x) in rows::words_mut(m).iter_mut().enumerate() {
                    if !units.contains(&(i / 2)) {
                        *x = 0x5555_5555_5555_5555 << (i % 2);
                    }
                }
            };
            let (mut in_place, mut light2) = (a.clone(), vec![Unit::ZERO; u]);
            junk(&mut in_place);
            junk(&mut light2);
            let p = in_place.as_mut_ptr();
            // SAFETY: heavy is the rows; light overlaps neither.
            let spans = unsafe { rows::split(p, b.as_ptr(), p, light2.as_mut_ptr(), span) };
            assert_eq!(spans, (hs, ls), "split of the span");
            for (side, whole) in [(&in_place, &heavy), (&light2, &light)] {
                let (x, y) = (rows::words(side), rows::words(whole));
                assert!(
                    units
                        .clone()
                        .all(|i| x[2 * i..2 * i + 2] == y[2 * i..2 * i + 2]),
                    "split in place"
                );
            }

            // Runs against a scan, and the difference array summing back to the mask.
            let mut expected = Vec::new();
            let mut r = 0;
            while r < n {
                if row_bit(&a, r) {
                    let start = r;
                    while r < n && row_bit(&a, r) {
                        r += 1;
                    }
                    expected.extend([(start, 5), (r, -5)]);
                } else {
                    r += 1;
                }
            }
            expected.sort_unstable();
            let mut record = Record {
                entries: n + 1,
                writes: Vec::new(),
            };
            let w = &rows::words(&a)[span.lo..span.hi];
            rows::add_runs(w, span.lo, 5, &mut record);
            record.writes.sort_unstable();
            assert_eq!(record.writes, expected, "runs of {n} rows");
            let mut diff = vec![0i128; n + 1];
            rows::add_runs(w, span.lo, 3, diff.as_mut_slice());
            let mut sum = 0;
            for (r, d) in diff.iter().enumerate() {
                sum += d;
                assert_eq!(sum, if r < n && row_bit(&a, r) { 3 } else { 0 });
            }
        }
    }
}

/// The vector span tracker against the plain-word one and a scan, from random first
/// units, on masks of random words, sparse words and none.
#[test]
fn span_tracker() {
    let mut rng = Rng(11);
    for _ in 0..if MIRI { 20 } else { 2000 } {
        let start = rng.below(1 << 20);
        let n = rng.below(40);
        let word = |rng: &mut Rng| match rng.below(4) {
            0 | 1 => 0,
            2 => 1 << rng.below(64),
            _ => rng.next(),
        };
        let units: Vec<[[u64; 2]; 2]> = (0..n)
            .map(|_| {
                [
                    [word(&mut rng), word(&mut rng)],
                    [word(&mut rng), word(&mut rng)],
                ]
            })
            .collect();
        let (mut fast, mut reference) = (super::simd::Spans::new(start), scalar::Spans::new(start));
        for [a, b] in &units {
            fast.add(a, b);
            reference.add(a, b);
        }
        let scan = |m: usize| {
            let w: Vec<u64> = units.iter().flat_map(|u| u[m]).collect();
            let first = w.iter().position(|&x| x != 0);
            let last = w.iter().rposition(|&x| x != 0);
            first.map(|f| {
                (
                    (2 * start + f) as u32,
                    (2 * start + last.unwrap_or(f)) as u32,
                )
            })
        };
        let got = fast.get();
        assert_eq!(got, reference.get(), "tracker against the reference");
        for (m, &(first, last)) in got.iter().enumerate() {
            match scan(m) {
                Some(span) => assert_eq!((first, last), span, "tracker against a scan"),
                None => assert_eq!(first, u32::MAX, "a mask without rows"),
            }
        }
    }
}

#[test]
#[should_panic(expected = "a run ends past the sink")]
fn runtime_width_run_past_sink() {
    let mut m = vec![Unit::ZERO; rows::units(1500)];
    rows::first(&mut m, 1500);
    let s = rows::Span::first(1500);
    let mut diff = vec![0i128; 1500];
    rows::add_runs(&rows::words(&m)[s.lo..s.hi], 0, 1, diff.as_mut_slice());
}

#[test]
fn runtime_width_empty_mask_needs_no_entries() {
    let mut diff: [i128; 0] = [];
    rows::add_runs(&[], 0, 1, &mut diff[..]);
}
