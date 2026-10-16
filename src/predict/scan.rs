//! Varying events that scan their rows instead of reading the precomputed masks: the
//! research ablation without precompute, and production's [`NoScan`], which never
//! scans.

use super::WorkCounters;
use super::mask::Mask;
use super::mask::rows::{self, Unit};
use crate::model::{Model, VaryingPredicate};

/// How a varying event finds its rows when precompute is off: by scanning them.
pub(super) trait Scan<M> {
    /// The left rows of varying predicate `pred` among `rows`.
    fn left(&mut self, pred: usize, rows: M) -> M;

    /// The left rows of varying predicate `pred` among `rows`, the units of a
    /// runtime-width mask from its unit `unit` on, into `left`, which has as many
    /// units.
    fn left_rows(&mut self, pred: usize, rows: &[Unit], left: &mut [Unit], unit: usize);
}

/// Production's [`Scan`], which it never calls: a type without data, unlike
/// research's scanner, so that their passes cannot be merged.
#[derive(Clone, Copy)]
pub(super) struct NoScan;

impl<M> Scan<M> for NoScan {
    #[inline(always)]
    fn left(&mut self, _: usize, rows: M) -> M {
        rows
    }

    #[inline(always)]
    fn left_rows(&mut self, _: usize, _: &[Unit], _: &mut [Unit], _: usize) {}
}

/// The left rows of a varying predicate among an event's rows, by scanning them: the
/// mask resolve uses when precompute is off.
pub(super) struct Scanner<'a> {
    pub(super) model: &'a Model,
    /// The group's rows.
    pub(super) rows: &'a [f64],
    /// Whether scans of monotonic features stop comparing at the first row past the
    /// threshold.
    pub(super) monotonic: bool,
    pub(super) counters: WorkCounters,
}

impl Scanner<'_> {
    /// Whether each row, given in increasing order, goes left at varying predicate
    /// `pred`.
    fn router<const F32: bool, const STATS: bool>(
        &mut self,
        pred: usize,
    ) -> impl FnMut(usize) -> bool + '_ {
        let model = self.model;
        let p = model.varying_predicates[pred];
        let (f, nf) = (p.feature() as usize, model.config.n_features());
        // Numeric predicates on monotonic features: once a row's value passes the
        // threshold (increasing) or comes below it (decreasing), every later value
        // does too, so only missing values remain to route.
        let (direction, default_left) = match p {
            VaryingPredicate::Num { default_left, .. } if self.monotonic => {
                if model.config.is_increasing(f) {
                    (Some(false), default_left)
                } else if model.config.is_decreasing(f) {
                    (Some(true), default_left)
                } else {
                    (None, default_left)
                }
            }
            _ => (None, false),
        };
        let mut past = false;
        let (rows, counters) = (self.rows, &mut self.counters);
        move |r| {
            let v = rows[r * nf + f];
            if STATS {
                counters.scan_rows += 1;
            }
            if past {
                if v.is_nan() {
                    default_left
                } else {
                    direction == Some(true)
                }
            } else {
                if STATS {
                    counters.scan_evals += 1;
                }
                let goes = p.goes_left::<F32>(v, &model.categories);
                past = direction.is_some() && !v.is_nan() && goes == direction.unwrap_or(false);
                goes
            }
        }
    }
}

/// Research's [`Scan`]: the scanner, with the threshold type and counting as consts.
pub(super) struct Scans<'s, 'a, const F32: bool, const STATS: bool>(pub(super) &'s mut Scanner<'a>);

impl<M: Mask, const F32: bool, const STATS: bool> Scan<M> for Scans<'_, '_, F32, STATS> {
    fn left(&mut self, pred: usize, rows: M) -> M {
        let mut route = self.0.router::<F32, STATS>(pred);
        let mut left = M::ZERO;
        rows.for_each_row(|r| {
            if route(r) {
                left.set(r);
            }
        });
        left
    }

    fn left_rows(&mut self, pred: usize, rows: &[Unit], left: &mut [Unit], unit: usize) {
        let mut route = self.0.router::<F32, STATS>(pred);
        let left = rows::words_mut(left);
        left.fill(0);
        for (i, (&w, out)) in rows::words(rows).iter().zip(left).enumerate() {
            let mut m = w;
            let row = 128 * unit + 64 * i;
            while m != 0 {
                let b = m.trailing_zeros();
                *out |= u64::from(route(row + b as usize)) << b;
                m &= m - 1;
            }
        }
    }
}
