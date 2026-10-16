//! The public [`Predictor`]: its calls, their input checks, and how a call's rows divide
//! into groups.

#![expect(
    clippy::inline_always,
    reason = "keeps the production call path independent of the research builds"
)]

use std::sync::Arc;

use crate::model::Model;
use crate::predict::{Options, Workspace};

/// How the rows of one call divide into groups.
#[derive(Clone, Copy)]
pub(crate) enum Groups<'a> {
    /// All rows form one group.
    One,
    /// Consecutive groups of this many rows; the last may be shorter.
    Fixed(usize),
    /// Group `g` is rows `offsets[g]..offsets[g + 1]`.
    Offsets(&'a [usize]),
}

impl Groups<'_> {
    pub(crate) const fn count(self, n_rows: usize) -> usize {
        match self {
            Self::One => (n_rows > 0) as usize,
            Self::Fixed(width) => n_rows.div_ceil(width),
            Self::Offsets(offsets) => offsets.len() - 1,
        }
    }

    /// Rows of group `g < count`. Never overflows: every bound is at most `n_rows`.
    #[inline]
    pub(crate) fn bounds(self, g: usize, n_rows: usize) -> (usize, usize) {
        match self {
            Self::One => (0, n_rows),
            Self::Fixed(width) => {
                let start = g * width;
                (start, start + width.min(n_rows - start))
            }
            Self::Offsets(offsets) => (offsets[g], offsets[g + 1]),
        }
    }
}

/// Predicts groups of rows with a [`Forest`]'s model and its own scratch space.
///
/// Create one with [`Forest::predictor`] and reuse it, one per worker thread: creating
/// a predictor allocates the scratch space for the widest group, which grows with
/// `max_group_width`, and calls never allocate. A predictor is `Send`, so it can move to the thread that uses it, and it
/// keeps its model alive.
///
/// Every call takes row-major `f64` input, `n_features` values per row in the trained
/// column order, and writes one output per row. Within a group, constant features
/// must be equal in every row; this contract is not checked. Outputs do not depend
/// on whether monotonic features follow their declared direction. Empty input is a
/// no-op.
///
/// # Panics
///
/// Each call panics, before writing any output, if the input length is not a multiple
/// of `n_features`, if `out` does not hold one value per row, or if the grouping is
/// invalid as documented on the call. Groups may have any width up to the configured
/// `max_group_width`.
pub struct Predictor {
    pub(crate) model: Arc<Model>,
    pub(crate) workspace: Workspace,
}

impl std::fmt::Debug for Predictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Predictor")
            .field("max_group_width", &self.model.config.max_group_width())
            .finish_non_exhaustive()
    }
}

impl Predictor {
    /// Predict one group: every row of `rows` belongs to the same entity.
    ///
    /// # Panics
    ///
    /// If `rows.len()` is not a multiple of `n_features`, if `out.len()` is not the row
    /// count, or if the group has more than `max_group_width` rows.
    #[inline]
    pub fn predict_group(&mut self, rows: &[f64], out: &mut [f64]) {
        self.check(rows, out.len(), Groups::One);
        self.production(rows, out, Groups::One);
    }

    /// Predict consecutive groups of varying width: group `g` is rows
    /// `offsets[g]..offsets[g + 1]` of `data`.
    ///
    /// `offsets` starts at 0, ends at the row count and strictly increases, so it holds
    /// one more entry than there are groups: `[0]` for no rows.
    ///
    /// # Panics
    ///
    /// If `data.len()` is not a multiple of `n_features`, if `out.len()` is not the row
    /// count, if `offsets` is empty, does not start at 0, does not end at the row count
    /// or does not strictly increase, or if a group has more than `max_group_width`
    /// rows.
    #[inline]
    pub fn predict_groups(&mut self, data: &[f64], offsets: &[usize], out: &mut [f64]) {
        let groups = Groups::Offsets(offsets);
        self.check(data, out.len(), groups);
        self.production(data, out, groups);
    }

    /// Predict consecutive groups of `width` rows. The last group is shorter when the
    /// row count is not a multiple of `width`.
    ///
    /// # Panics
    ///
    /// If `data.len()` is not a multiple of `n_features`, if `out.len()` is not the row
    /// count, or if `width` is 0 or more than `max_group_width`.
    #[inline]
    pub fn predict_fixed(&mut self, data: &[f64], width: usize, out: &mut [f64]) {
        let groups = Groups::Fixed(width);
        self.check(data, out.len(), groups);
        self.production(data, out, groups);
    }

    /// Run a call's checked groups through the production kernel.
    #[inline(always)]
    fn production(&mut self, data: &[f64], out: &mut [f64], groups: Groups<'_>) {
        self.kernel::<false, false>(data, out, groups, Options::default(), None, true);
    }

    /// Panic unless the call's input and output lengths match the model and the
    /// grouping is valid.
    #[inline(always)]
    pub(crate) fn check(&self, data: &[f64], out_len: usize, groups: Groups<'_>) {
        let config = &self.model.config;
        let (nf, max_width) = (config.n_features(), config.max_group_width());
        assert!(
            data.len().is_multiple_of(nf),
            "input length {} is not a multiple of n_features {nf}",
            data.len()
        );
        let n_rows = data.len() / nf;
        assert!(
            out_len == n_rows,
            "output length {out_len} differs from the row count {n_rows}"
        );
        match groups {
            Groups::One => assert!(
                n_rows <= max_width,
                "group of {n_rows} rows exceeds max_group_width {max_width}"
            ),
            Groups::Fixed(width) => {
                assert!(width > 0, "group width must be positive");
                assert!(
                    width <= max_width,
                    "group width {width} exceeds max_group_width {max_width}"
                );
            }
            Groups::Offsets(offsets) => {
                assert!(offsets.first() == Some(&0), "offsets must start at 0");
                assert!(
                    offsets.last() == Some(&n_rows),
                    "offsets must end at the row count {n_rows}"
                );
                for pair in offsets.windows(2) {
                    assert!(pair[0] < pair[1], "offsets must strictly increase");
                    assert!(
                        pair[1] - pair[0] <= max_width,
                        "group of {} rows exceeds max_group_width {max_width}",
                        pair[1] - pair[0]
                    );
                }
            }
        }
    }
}
