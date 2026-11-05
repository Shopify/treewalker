//! Wide-mask operations on plain words: the fallback for targets without a vector
//! version, and the reference the tests check the vector versions against.

use std::array::from_fn;
use std::hint::select_unpredictable;
use std::marker::PhantomData;

use super::ops::WordMaps;
use crate::predict::isa::Isa;

#[inline]
pub fn and<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    from_fn(|i| a[i] & b[i])
}

#[inline]
pub fn and_not<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    from_fn(|i| a[i] & !b[i])
}

#[inline]
pub fn or<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    from_fn(|i| a[i] | b[i])
}

#[inline]
pub fn xor<X: Isa, const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    from_fn(|i| a[i] ^ b[i])
}

#[inline]
pub fn is_zero<X: Isa, const W: usize>(a: &[u64; W]) -> bool {
    a.iter().fold(0, |acc, &x| acc | x) == 0
}

/// Word maps of one bit per word.
pub struct Maps<X>(PhantomData<X>);

impl<X: Isa> WordMaps for Maps<X> {
    #[inline]
    fn stride<const W: usize>() -> u32 {
        1
    }

    #[inline]
    fn nonzero<const W: usize>(w: &[u64; W]) -> u64 {
        w.iter()
            .enumerate()
            .fold(0, |m, (i, &x)| m | u64::from(x != 0) << i)
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
#[inline]
pub fn select<X: Isa, const W: usize>(c: bool, a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    from_fn(|i| select_unpredictable(c, a[i], b[i]))
}

/// The first and last nonzero words of two runtime-width masks, given unit by unit in
/// increasing order: per word of a unit of each mask, the last index whose word had
/// rows, the first such index, and whether no word had rows yet.
#[derive(Clone, Copy)]
pub struct Spans<X> {
    index: u32,
    last: [u32; 4],
    first: [u32; 4],
    unseen: [bool; 4],
    level: PhantomData<X>,
}

impl<X> Spans<X> {
    /// Spans of masks given from unit `unit` on; word indices are below 2^32.
    #[inline]
    pub const fn new(unit: usize) -> Self {
        Self {
            index: 2 * unit as u32,
            last: [0; 4],
            first: [u32::MAX; 4],
            unseen: [true; 4],
            level: PhantomData,
        }
    }

    /// The next unit of each mask.
    #[inline]
    pub fn add(&mut self, a: &[u64; 2], b: &[u64; 2]) {
        for (k, w) in [a[0], a[1], b[0], b[1]].into_iter().enumerate() {
            let i = self.index + (k as u32 & 1);
            self.last[k] = select_unpredictable(w == 0, self.last[k], i);
            self.first[k] = select_unpredictable(self.unseen[k], i, self.first[k]);
            self.unseen[k] &= w == 0;
        }
        self.index += 2;
    }

    /// Each mask's first and last nonzero word; the first is `u32::MAX` for a mask
    /// without rows.
    #[inline]
    pub fn get(self) -> [(u32, u32); 2] {
        let first: [u32; 4] =
            from_fn(|k| select_unpredictable(self.unseen[k], u32::MAX, self.first[k]));
        [
            (first[0].min(first[1]), self.last[0].max(self.last[1])),
            (first[2].min(first[3]), self.last[2].max(self.last[3])),
        ]
    }
}
