//! Wide-mask operations on x86-64: SSE2, part of the baseline, two words per vector;
//! at a level with AVX2 ([`Isa`]), four words per vector for masks of 256 bits and up.
//!
//! Where the build dispatches to levels (`isa::DISPATCH`), the operations and the
//! helpers that levels with SSE4.1, AVX2 or AVX-512 call are `#[inline(always)]`:
//! compiled out of line they would lack the level's features, which only its
//! trampolines enable, and call each intrinsic. Builds that enable those features keep
//! `#[inline]`.

use std::marker::PhantomData;

use super::ops::WordMaps;
use crate::predict::isa::Isa;

macro_rules! dispatch {
    ($f:ident($($arg:expr),*)) => {
        if X::AVX2 && W.is_multiple_of(4) { avx2::$f($($arg),*) } else { sse2::$f($($arg),*) }
    };
}

#[cfg_attr(dispatch, inline(always))]
#[cfg_attr(not(dispatch), inline)]
pub fn and<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    dispatch!(and(a, b))
}

#[cfg_attr(dispatch, inline(always))]
#[cfg_attr(not(dispatch), inline)]
pub fn and_not<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    dispatch!(and_not(a, b))
}

#[cfg_attr(dispatch, inline(always))]
#[cfg_attr(not(dispatch), inline)]
pub fn or<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    dispatch!(or(a, b))
}

#[cfg_attr(dispatch, inline(always))]
#[cfg_attr(not(dispatch), inline)]
pub fn xor<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    dispatch!(xor(a, b))
}

#[cfg_attr(dispatch, inline(always))]
#[cfg_attr(not(dispatch), inline)]
pub fn is_zero<X: Isa, const W: usize>(a: &[u64; W]) -> bool {
    dispatch!(is_zero(a))
}

/// Word maps of one bit per word: AVX-512 tests into a mask register with AVX-512F and
/// VL; else SSE4.1 compares (AVX2 for 4 words at a time) and a move mask; else one word
/// at a time.
pub struct Maps<X>(PhantomData<X>);

impl<X: Isa> WordMaps for Maps<X> {
    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    fn stride<const W: usize>() -> u32 {
        1
    }

    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    fn nonzero<const W: usize>(w: &[u64; W]) -> u64 {
        maps::nonzero::<X, W>(w)
    }
}

mod maps {
    use std::arch::x86_64::{
        _mm_castsi128_pd, _mm_cmpeq_epi64, _mm_loadu_si128, _mm_movemask_pd, _mm_setzero_si128,
        _mm_test_epi64_mask, _mm256_loadu_si256, _mm256_test_epi64_mask,
    };

    use crate::predict::isa::Isa;

    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    pub fn nonzero<X: Isa, const W: usize>(w: &[u64; W]) -> u64 {
        if X::AVX512 {
            nonzero_avx512(w)
        } else if X::SSE41 {
            nonzero_sse41::<X, W>(w)
        } else {
            nonzero_words(w)
        }
    }

    /// One word at a time.
    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    fn nonzero_words<const W: usize>(w: &[u64; W]) -> u64 {
        w.iter()
            .enumerate()
            .fold(0, |m, (i, &x)| m | u64::from(x != 0) << i)
    }

    /// The mask register's bits of a test of each word against itself, 4 words at a
    /// time, or 2 when `W` is 2, gathered into one map. Only levels with AVX-512F and VL
    /// call it.
    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    fn nonzero_avx512<const W: usize>(w: &[u64; W]) -> u64 {
        if W.is_multiple_of(4) {
            let (q, _) = w.as_chunks::<4>();
            let mut m = 0;
            for (j, q) in q.iter().enumerate() {
                // SAFETY: the level has AVX-512F and VL; `q` holds the 32 bytes read,
                // and the load is unaligned.
                let bits = unsafe {
                    let v = _mm256_loadu_si256(q.as_ptr().cast());
                    _mm256_test_epi64_mask(v, v)
                };
                m |= u64::from(bits) << (4 * j);
            }
            m
        } else {
            let (p, _) = w.as_chunks::<2>();
            let mut m = 0;
            for (j, p) in p.iter().enumerate() {
                // SAFETY: the level has AVX-512F and VL; `p` holds the 16 bytes read,
                // and the load is unaligned.
                let bits = unsafe {
                    let v = _mm_loadu_si128(p.as_ptr().cast());
                    _mm_test_epi64_mask(v, v)
                };
                m |= u64::from(bits) << (2 * j);
            }
            m
        }
    }

    /// The words that are 0, by a compare and a move mask per 2 words, or per 4 with
    /// AVX2. Only levels with SSE4.1 call it.
    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    fn zero<X: Isa, const W: usize>(w: &[u64; W]) -> u64 {
        if X::AVX2 && W.is_multiple_of(4) {
            use std::arch::x86_64::{
                _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_loadu_si256, _mm256_movemask_pd,
                _mm256_setzero_si256,
            };
            let (q, _) = w.as_chunks::<4>();
            let mut m = 0;
            for (j, q) in q.iter().enumerate() {
                // SAFETY: the level has AVX2; `q` holds the 32 bytes read, and the load
                // is unaligned.
                let bits = unsafe {
                    let vec = _mm256_loadu_si256(q.as_ptr().cast());
                    let eq = _mm256_cmpeq_epi64(vec, _mm256_setzero_si256());
                    _mm256_movemask_pd(_mm256_castsi256_pd(eq))
                };
                m |= u64::from(bits as u32) << (4 * j);
            }
            return m;
        }
        let (p, _) = w.as_chunks::<2>();
        let mut m = 0;
        for (j, p) in p.iter().enumerate() {
            // SAFETY: the level has SSE4.1; `p` holds the 16 bytes read, and the load is
            // unaligned.
            let bits = unsafe {
                let vec = _mm_loadu_si128(p.as_ptr().cast());
                let eq = _mm_cmpeq_epi64(vec, _mm_setzero_si128());
                _mm_movemask_pd(_mm_castsi128_pd(eq))
            };
            m |= u64::from(bits as u32) << (2 * j);
        }
        m
    }

    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    fn nonzero_sse41<X: Isa, const W: usize>(w: &[u64; W]) -> u64 {
        !zero::<X, W>(w) & (u64::MAX >> (64 - W))
    }
}

/// Rows `0..n` of `64 * W` as words, `n <= 64 * W`.
#[inline]
pub fn first<const W: usize>(n: usize) -> [u64; W] {
    std::array::from_fn(|i| {
        let k = n.saturating_sub(64 * i).min(64);
        u64::MAX.checked_shr(64 - k as u32).unwrap_or(0)
    })
}
#[cfg_attr(dispatch, inline(always))]
#[cfg_attr(not(dispatch), inline)]
pub fn select<X: Isa, const W: usize>(c: bool, a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    dispatch!(select(c, a, b))
}

pub use sse2::Spans;

mod sse2 {
    use std::marker::PhantomData;

    use crate::predict::isa::Isa;

    use std::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_and_si128, _mm_andnot_si128, _mm_castps_si128,
        _mm_castsi128_ps, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128,
        _mm_set1_epi32, _mm_set1_epi64x, _mm_setr_epi32, _mm_setzero_si128, _mm_shuffle_ps,
        _mm_storeu_si128, _mm_xor_si128,
    };

    /// The mask's words in vector-sized pairs.
    #[inline]
    fn pairs<const W: usize>(a: &[u64; W]) -> &[[u64; 2]] {
        const { assert!(W.is_multiple_of(2) && W <= 16) };
        a.as_chunks().0
    }

    #[inline]
    fn load(p: &[u64; 2]) -> __m128i {
        // SAFETY: `p` holds the 16 bytes read, and the load is unaligned.
        unsafe { _mm_loadu_si128(p.as_ptr().cast()) }
    }

    /// Applies `f` to each pair of vectors of `a` and `b`.
    #[inline]
    fn map<const W: usize>(
        a: &[u64; W],
        b: &[u64; W],
        f: impl Fn(__m128i, __m128i) -> __m128i,
    ) -> [u64; W] {
        let mut out = [0; W];
        let (o, _) = out.as_chunks_mut::<2>();
        for ((o, a), b) in o.iter_mut().zip(pairs(a)).zip(pairs(b)) {
            // SAFETY: `o` holds the 16 bytes written, and the store is unaligned.
            unsafe { _mm_storeu_si128(o.as_mut_ptr().cast(), f(load(a), load(b))) };
        }
        out
    }

    #[inline]
    pub fn and<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
        // SAFETY: SSE2 is always available on x86-64.
        map(a, b, |x, y| unsafe { _mm_and_si128(x, y) })
    }

    #[inline]
    pub fn and_not<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
        // SAFETY: SSE2 is always available on x86-64.
        map(a, b, |x, y| unsafe { _mm_andnot_si128(y, x) })
    }

    #[inline]
    pub fn or<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
        // SAFETY: SSE2 is always available on x86-64.
        map(a, b, |x, y| unsafe { _mm_or_si128(x, y) })
    }

    #[inline]
    pub fn xor<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
        // SAFETY: SSE2 is always available on x86-64.
        map(a, b, |x, y| unsafe { _mm_xor_si128(x, y) })
    }

    #[inline]
    pub fn is_zero<const W: usize>(a: &[u64; W]) -> bool {
        // SAFETY: SSE2 is always available on x86-64.
        unsafe {
            let acc = pairs(a)
                .iter()
                .fold(_mm_setzero_si128(), |acc, p| _mm_or_si128(acc, load(p)));
            _mm_movemask_epi8(_mm_cmpeq_epi8(acc, _mm_setzero_si128())) == 0xFFFF
        }
    }

    #[inline]
    pub fn select<const W: usize>(c: bool, a: &[u64; W], b: &[u64; W]) -> [u64; W] {
        // SAFETY: SSE2 is always available on x86-64.
        let m = unsafe { _mm_set1_epi64x(i64::from(c).wrapping_neg()) };
        // SAFETY: as above.
        map(a, b, |x, y| unsafe {
            _mm_or_si128(_mm_and_si128(m, x), _mm_andnot_si128(m, y))
        })
    }

    /// Lanes of `a` where `m` is set, else of `b`.
    #[inline]
    fn blend(m: __m128i, a: __m128i, b: __m128i) -> __m128i {
        // SAFETY: SSE2 is always available on x86-64.
        unsafe { _mm_or_si128(_mm_and_si128(m, a), _mm_andnot_si128(m, b)) }
    }

    /// Four lanes, all ones where a word of `a` then `b` is 0: with SSE4.1 each word
    /// compared whole and gathered by one shuffle, else its halves.
    #[inline]
    fn zero_words<X: Isa>(a: __m128i, b: __m128i) -> __m128i {
        if X::SSE41 {
            zero_words_sse41(a, b)
        } else {
            zero_words_sse2(a, b)
        }
    }

    /// [`zero_words`] with SSE4.1: each word compared whole, gathered by one shuffle.
    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    fn zero_words_sse41(a: __m128i, b: __m128i) -> __m128i {
        use std::arch::x86_64::_mm_cmpeq_epi64;
        // SAFETY: the level has SSE4.1.
        unsafe {
            let (a, b) = (
                _mm_cmpeq_epi64(a, _mm_setzero_si128()),
                _mm_cmpeq_epi64(b, _mm_setzero_si128()),
            );
            let (a, b) = (_mm_castsi128_ps(a), _mm_castsi128_ps(b));
            _mm_castps_si128(_mm_shuffle_ps::<0b10_00_10_00>(a, b))
        }
    }

    /// [`zero_words`] with SSE2: each word's low halves or'ed with its high halves,
    /// gathered by two shuffles, then compared.
    #[inline]
    fn zero_words_sse2(a: __m128i, b: __m128i) -> __m128i {
        use std::arch::x86_64::_mm_cmpeq_epi32;
        // SAFETY: SSE2 is always available on x86-64.
        unsafe {
            let (a, b) = (_mm_castsi128_ps(a), _mm_castsi128_ps(b));
            let low = _mm_castps_si128(_mm_shuffle_ps::<0b10_00_10_00>(a, b));
            let high = _mm_castps_si128(_mm_shuffle_ps::<0b11_01_11_01>(a, b));
            _mm_cmpeq_epi32(_mm_or_si128(low, high), _mm_setzero_si128())
        }
    }

    /// The first and last nonzero words of two runtime-width masks, given unit by unit
    /// in increasing order, without a branch: four lanes, one per word of a unit of
    /// each mask, hold the next word index, the last index whose word had rows, the
    /// first such index, and whether no word had rows yet. Word indices are below
    /// 2^32.
    #[derive(Clone, Copy)]
    pub struct Spans<X> {
        index: __m128i,
        last: __m128i,
        first: __m128i,
        unseen: __m128i,
        level: PhantomData<X>,
    }

    impl<X: Isa> Spans<X> {
        /// Spans of masks given from unit `unit` on.
        #[inline]
        pub fn new(unit: usize) -> Self {
            let w = (2 * unit) as u32 as i32;
            // SAFETY: SSE2 is always available on x86-64.
            unsafe {
                Self {
                    index: _mm_setr_epi32(w, w.wrapping_add(1), w, w.wrapping_add(1)),
                    last: _mm_setzero_si128(),
                    first: _mm_set1_epi32(-1),
                    unseen: _mm_set1_epi32(-1),
                    level: PhantomData,
                }
            }
        }

        /// The next unit of each mask.
        #[inline]
        pub fn add(&mut self, a: &[u64; 2], b: &[u64; 2]) {
            // SAFETY: SSE2 is always available on x86-64.
            unsafe {
                let zero = zero_words::<X>(load(a), load(b));
                self.last = blend(zero, self.last, self.index);
                self.first = blend(self.unseen, self.index, self.first);
                self.unseen = _mm_and_si128(self.unseen, zero);
                self.index = _mm_add_epi32(self.index, _mm_set1_epi32(2));
            }
        }

        /// Each mask's first and last nonzero word; the first is `u32::MAX` for a mask
        /// without rows.
        #[inline]
        pub fn get(self) -> [(u32, u32); 2] {
            let (mut first, mut last) = ([0u32; 4], [0u32; 4]);
            // SAFETY: SSE2 is always available on x86-64; each array holds the 16 bytes
            // written, and the stores are unaligned.
            unsafe {
                _mm_storeu_si128(
                    first.as_mut_ptr().cast(),
                    _mm_or_si128(self.first, self.unseen),
                );
                _mm_storeu_si128(last.as_mut_ptr().cast(), self.last);
            }
            [
                (first[0].min(first[1]), last[0].max(last[1])),
                (first[2].min(first[3]), last[2].max(last[3])),
            ]
        }
    }
}

/// Only levels with AVX2 call these.
mod avx2 {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_andnot_si256, _mm256_loadu_si256, _mm256_or_si256,
        _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_testz_si256,
        _mm256_xor_si256,
    };

    /// The mask's words in vector-sized quads. Callers pass `W` divisible by 4.
    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    fn quads<const W: usize>(a: &[u64; W]) -> &[[u64; 4]] {
        let (q, rest) = a.as_chunks();
        debug_assert!(rest.is_empty() && W <= 16);
        q
    }

    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    fn load(q: &[u64; 4]) -> __m256i {
        // SAFETY: the level has AVX2; `q` holds the 32 bytes read, and the
        // load is unaligned.
        unsafe { _mm256_loadu_si256(q.as_ptr().cast()) }
    }

    /// Applies `f` to each pair of vectors of `a` and `b`.
    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    fn map<const W: usize>(
        a: &[u64; W],
        b: &[u64; W],
        f: impl Fn(__m256i, __m256i) -> __m256i,
    ) -> [u64; W] {
        let mut out = [0; W];
        let (o, _) = out.as_chunks_mut::<4>();
        for ((o, a), b) in o.iter_mut().zip(quads(a)).zip(quads(b)) {
            // SAFETY: the level has AVX2; `o` holds the 32 bytes written, and
            // the store is unaligned.
            unsafe { _mm256_storeu_si256(o.as_mut_ptr().cast(), f(load(a), load(b))) };
        }
        out
    }

    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    pub fn and<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
        // SAFETY: the level has AVX2.
        map(a, b, |x, y| unsafe { _mm256_and_si256(x, y) })
    }

    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    pub fn and_not<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
        // SAFETY: the level has AVX2.
        map(a, b, |x, y| unsafe { _mm256_andnot_si256(y, x) })
    }

    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    pub fn or<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
        // SAFETY: the level has AVX2.
        map(a, b, |x, y| unsafe { _mm256_or_si256(x, y) })
    }

    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    pub fn xor<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
        // SAFETY: the level has AVX2.
        map(a, b, |x, y| unsafe { _mm256_xor_si256(x, y) })
    }

    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    pub fn is_zero<const W: usize>(a: &[u64; W]) -> bool {
        // SAFETY: the level has AVX2.
        unsafe {
            let mut acc = _mm256_setzero_si256();
            for q in quads(a) {
                acc = _mm256_or_si256(acc, load(q));
            }
            _mm256_testz_si256(acc, acc) == 1
        }
    }

    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    pub fn select<const W: usize>(c: bool, a: &[u64; W], b: &[u64; W]) -> [u64; W] {
        // SAFETY: the level has AVX2.
        let m = unsafe { _mm256_set1_epi64x(i64::from(c).wrapping_neg()) };
        // SAFETY: as above.
        map(a, b, |x, y| unsafe {
            _mm256_or_si256(_mm256_and_si256(m, x), _mm256_andnot_si256(m, y))
        })
    }
}
