//! Wide-mask operations on NEON, two words per vector. NEON is part of the AArch64
//! baseline, so the intrinsics below are always available.

use std::arch::aarch64::{
    uint8x8_t, uint32x4_t, uint64x2_t, vaddlvq_u8, vaddq_u8, vaddq_u32, vaddq_u64, vandq_u32,
    vandq_u64, vbicq_u64, vbslq_u32, vbslq_u64, vceqzq_u32, vcntq_u8, vcombine_u32, vcombine_u64,
    vcreate_u32, vcreate_u64, vdupq_n_u8, vdupq_n_u32, vdupq_n_u64, veorq_u64, vget_high_u32,
    vget_lane_u32, vget_lane_u64, vget_low_u32, vld1q_u64, vmaxvq_u32, vmovn_u16, vmovn_u32,
    vmovn_u64, vnegq_s64, vorrq_u32, vorrq_u64, vpmax_u32, vpmaxq_u32, vpmin_u32, vqsubq_u64,
    vreinterpret_u8_u16, vreinterpret_u8_u32, vreinterpret_u64_u8, vreinterpretq_s64_u64,
    vreinterpretq_u8_u16, vreinterpretq_u8_u64, vreinterpretq_u16_u8, vreinterpretq_u16_u32,
    vreinterpretq_u32_u64, vshlq_u64, vshrn_n_u16, vst1q_u64, vtstq_u64, vuzp1q_u8, vuzp1q_u16,
    vuzp1q_u32,
};

use std::marker::PhantomData;

use super::ops::WordMaps;
use crate::predict::isa::Isa;

/// The mask's words in vector-sized pairs.
#[inline]
fn pairs<const W: usize>(a: &[u64; W]) -> &[[u64; 2]] {
    const { assert!(W.is_multiple_of(2) && W <= 16) };
    a.as_chunks().0
}

#[inline]
fn load(p: &[u64; 2]) -> uint64x2_t {
    // SAFETY: `p` holds the two words read.
    unsafe { vld1q_u64(p.as_ptr()) }
}

/// Applies `f` to each pair of vectors of `a` and `b`.
#[inline]
fn map<const W: usize>(
    a: &[u64; W],
    b: &[u64; W],
    f: impl Fn(uint64x2_t, uint64x2_t) -> uint64x2_t,
) -> [u64; W] {
    let mut out = [0; W];
    let (o, _) = out.as_chunks_mut::<2>();
    for ((o, a), b) in o.iter_mut().zip(pairs(a)).zip(pairs(b)) {
        // SAFETY: `o` holds the two words written.
        unsafe { vst1q_u64(o.as_mut_ptr(), f(load(a), load(b))) };
    }
    out
}

#[inline]
pub fn and<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    // SAFETY: NEON is always available on AArch64.
    map(a, b, |x, y| unsafe { vandq_u64(x, y) })
}

#[inline]
pub fn and_not<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    // SAFETY: NEON is always available on AArch64.
    map(a, b, |x, y| unsafe { vbicq_u64(x, y) })
}

#[inline]
pub fn or<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    // SAFETY: NEON is always available on AArch64.
    map(a, b, |x, y| unsafe { vorrq_u64(x, y) })
}

#[inline]
pub fn xor<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    // SAFETY: NEON is always available on AArch64.
    map(a, b, |x, y| unsafe { veorq_u64(x, y) })
}

#[inline]
pub fn is_zero<X: Isa, const W: usize>(a: &[u64; W]) -> bool {
    // SAFETY: NEON is always available on AArch64.
    unsafe {
        let acc = pairs(a)
            .iter()
            .fold(vdupq_n_u64(0), |acc, p| vorrq_u64(acc, load(p)));
        vmaxvq_u32(vreinterpretq_u32_u64(acc)) == 0
    }
}

/// Rows set in `a`: byte counts summed in one vector, at most 8 per vector and 64 in
/// all, then summed across the vector once.
#[inline]
pub fn popcount<const W: usize>(a: &[u64; W]) -> u32 {
    // SAFETY: NEON is always available on AArch64.
    unsafe {
        let acc = pairs(a).iter().fold(vdupq_n_u8(0), |acc, p| {
            vaddq_u8(acc, vcntq_u8(vreinterpretq_u8_u64(load(p))))
        });
        u32::from(vaddlvq_u8(acc))
    }
}

/// A per-word test `f` (all ones or all zeros in each 64-bit lane) of `a`'s words,
/// narrowed by halving each lane until the `W` words fill 64 bits: `64 / W` bits per
/// word, the last step, for 16 words, taking a nibble from each byte.
#[inline]
fn word_map<const W: usize>(a: &[u64; W], f: impl Fn(uint64x2_t) -> uint64x2_t) -> u64 {
    let p = pairs(a);
    // SAFETY: NEON is always available on AArch64.
    unsafe {
        // Pair `i`'s words as 32-bit lanes, then pairs `i` and `i + 1` as four 32-bit
        // words, then pairs `i` to `i + 3` as eight 16-bit words.
        let pair = |i: usize| vreinterpretq_u32_u64(f(load(&p[i])));
        let quad = |i: usize| vreinterpretq_u16_u32(vuzp1q_u32(pair(i), pair(i + 1)));
        let oct = |i: usize| vreinterpretq_u8_u16(vuzp1q_u16(quad(i), quad(i + 2)));
        let map: uint8x8_t = match W {
            2 => vreinterpret_u8_u32(vmovn_u64(f(load(&p[0])))),
            4 => vreinterpret_u8_u16(vmovn_u32(vuzp1q_u32(pair(0), pair(1)))),
            8 => vmovn_u16(vuzp1q_u16(quad(0), quad(2))),
            16 => vshrn_n_u16::<4>(vreinterpretq_u16_u8(vuzp1q_u8(oct(0), oct(4)))),
            _ => unreachable!("wide masks have 2, 4, 8 or 16 words"),
        };
        vget_lane_u64::<0>(vreinterpret_u64_u8(map))
    }
}

/// Word maps by vector compares, narrowed to `64 / W` bits per word.
pub struct Maps<X>(PhantomData<X>);

impl<X: Isa> WordMaps for Maps<X> {
    #[inline]
    fn stride<const W: usize>() -> u32 {
        64 / W as u32
    }

    #[inline]
    fn nonzero<const W: usize>(w: &[u64; W]) -> u64 {
        // SAFETY: NEON is always available on AArch64.
        word_map(w, |x| unsafe { vtstq_u64(x, x) })
    }
}

/// Rows `0..n` of `64 * W` as words, `n <= 64 * W`: per word at `64 * i`, the rows
/// it lacks are `64 - (n - 64 * i)`, both differences saturating, in `0..=64`, and
/// USHL by minus that many shifts all ones right, to 0 at 64.
#[inline]
pub fn first<const W: usize>(n: usize) -> [u64; W] {
    let mut out = [0; W];
    let (o, _) = out.as_chunks_mut::<2>();
    // SAFETY: NEON is always available on AArch64; `o` holds the two words written.
    unsafe {
        let n = vdupq_n_u64(n as u64);
        let mut at = vcombine_u64(vcreate_u64(0), vcreate_u64(64));
        for o in o {
            let lacks = vqsubq_u64(vdupq_n_u64(64), vqsubq_u64(n, at));
            let words = vshlq_u64(
                vdupq_n_u64(u64::MAX),
                vnegq_s64(vreinterpretq_s64_u64(lacks)),
            );
            vst1q_u64(o.as_mut_ptr(), words);
            at = vaddq_u64(at, vdupq_n_u64(128));
        }
    }
    out
}
#[inline]
pub fn select<X: Isa, const W: usize>(c: bool, a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    // SAFETY: NEON is always available on AArch64.
    let m = unsafe { vdupq_n_u64(u64::from(c).wrapping_neg()) };
    // SAFETY: as above.
    map(a, b, |x, y| unsafe { vbslq_u64(m, x, y) })
}

/// The first and last nonzero words of two runtime-width masks, given unit by unit in
/// increasing order, without a branch: four lanes, one per word of a unit of each
/// mask, hold the next word index, the last index whose word had rows, the first such
/// index, and whether no word had rows yet. Word indices are below 2^32.
#[derive(Clone, Copy)]
pub struct Spans<X> {
    index: uint32x4_t,
    last: uint32x4_t,
    first: uint32x4_t,
    unseen: uint32x4_t,
    level: PhantomData<X>,
}

impl<X> Spans<X> {
    /// Spans of masks given from unit `unit` on.
    #[inline]
    pub fn new(unit: usize) -> Self {
        let w = 2 * unit as u64;
        // SAFETY: NEON is always available on AArch64.
        unsafe {
            let pair = vcreate_u32(((w + 1) << 32) | w);
            Self {
                index: vcombine_u32(pair, pair),
                last: vdupq_n_u32(0),
                first: vdupq_n_u32(u32::MAX),
                unseen: vdupq_n_u32(u32::MAX),
                level: PhantomData,
            }
        }
    }

    /// The next unit of each mask.
    #[inline]
    pub fn add(&mut self, a: &[u64; 2], b: &[u64; 2]) {
        // SAFETY: NEON is always available on AArch64.
        unsafe {
            // Each word's larger half: [a's words, b's words], 0 where a word is.
            let words = vpmaxq_u32(
                vreinterpretq_u32_u64(load(a)),
                vreinterpretq_u32_u64(load(b)),
            );
            let zero = vceqzq_u32(words);
            self.last = vbslq_u32(zero, self.last, self.index);
            self.first = vbslq_u32(self.unseen, self.index, self.first);
            self.unseen = vandq_u32(self.unseen, zero);
            self.index = vaddq_u32(self.index, vdupq_n_u32(2));
        }
    }

    /// Each mask's first and last nonzero word; the first is `u32::MAX` for a mask
    /// without rows.
    #[inline]
    pub fn get(self) -> [(u32, u32); 2] {
        // SAFETY: NEON is always available on AArch64.
        unsafe {
            let first = vorrq_u32(self.first, self.unseen);
            let first = vpmin_u32(vget_low_u32(first), vget_high_u32(first));
            let last = vpmax_u32(vget_low_u32(self.last), vget_high_u32(self.last));
            [
                (vget_lane_u32::<0>(first), vget_lane_u32::<0>(last)),
                (vget_lane_u32::<1>(first), vget_lane_u32::<1>(last)),
            ]
        }
    }
}
