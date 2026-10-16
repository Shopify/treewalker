//! Difference arrays the leaf pass adds runs of rows to ([`RunSink`]), and the `i128` one
//! of models under the 126-bit rule.

/// A difference array the leaf pass writes runs into: rows `start..end` add a value at
/// entry `start` and subtract it at entry `end`, so a group of `n` rows needs `n + 1`
/// entries.
pub trait RunSink {
    /// The value one leaf adds to its rows.
    type Value: Copy;
    /// A running sum of values that every row gets, kept apart from the entries.
    type Sum: Copy + Default;

    /// Number of entries.
    fn entries(&self) -> usize;

    /// Adds `v` at entry `at`.
    ///
    /// # Safety
    ///
    /// `at < self.entries()`.
    unsafe fn add(&mut self, at: usize, v: Self::Value);

    /// Subtracts `v` at entry `at`.
    ///
    /// # Safety
    ///
    /// `at < self.entries()`.
    unsafe fn sub(&mut self, at: usize, v: Self::Value);

    /// Adds each of `values` to every row through `sum`.
    fn add_all_of(&mut self, sum: &mut Self::Sum, values: impl Iterator<Item = Self::Value>);

    /// Adds `sum` to every row, at the first entry. No subtraction at the last entry
    /// is needed: the prefix sum reads entries `0..n` only.
    fn flush(&mut self, sum: Self::Sum);
}

/// Fixed-point leaf values under the 126-bit rule, which cannot overflow.
impl RunSink for [i128] {
    type Value = i128;
    type Sum = i128;

    #[inline]
    fn entries(&self) -> usize {
        self.len()
    }

    #[inline]
    unsafe fn add(&mut self, at: usize, v: i128) {
        // SAFETY: the caller keeps `at` below `self.len()`.
        unsafe { *self.get_unchecked_mut(at) += v };
    }

    #[inline]
    unsafe fn sub(&mut self, at: usize, v: i128) {
        // SAFETY: the caller keeps `at` below `self.len()`.
        unsafe { *self.get_unchecked_mut(at) -= v };
    }

    #[inline]
    fn add_all_of(&mut self, sum: &mut i128, values: impl Iterator<Item = i128>) {
        for v in values {
            *sum += v;
        }
    }

    #[inline]
    fn flush(&mut self, sum: i128) {
        if let Some(first) = self.first_mut() {
            *first += sum;
        }
    }
}
