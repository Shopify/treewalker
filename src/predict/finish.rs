//! The finish: each row's exact sum rounded once from the difference arrays, then the
//! model's averaging, base score and link function.

use super::scratch::Sums;
use crate::exact::{self, Plan};
use crate::model::{Model, Postprocessor};

impl Model {
    /// Prefix-sum both difference arrays into `results`, clearing them.
    pub(super) fn finish(&self, results: &mut [f64], sums: &mut Sums) {
        let n = results.len();
        match (sums, self.plan) {
            (Sums::Fixed(diff), Plan::Fixed(e)) => {
                let half = diff.len() / 2;
                let (even, odd) = diff.split_at_mut(half);
                let mut acc = 0i128;
                for ((out, a), b) in results.iter_mut().zip(&mut even[..n]).zip(&mut odd[..n]) {
                    acc += *a + *b;
                    (*a, *b) = (0, 0);
                    *out = exact::to_f64(acc, e);
                }
                (even[n], odd[n]) = (0, 0);
            }
            (
                Sums::Wide {
                    wide, diff, acc, ..
                },
                _,
            ) => {
                let l = wide.limbs();
                let half = diff.len() / 2;
                let (even, odd) = diff.split_at_mut(half);
                let acc = &mut acc[..l];
                acc.fill(0);
                let rows = even.chunks_exact_mut(l).zip(odd.chunks_exact_mut(l));
                for (out, (a, b)) in results.iter_mut().zip(rows) {
                    // Carry once per row, which keeps the slots within their headroom.
                    wide.add_clear_and_carry(acc, a, b);
                    *out = wide.normalized_to_f64(acc);
                }
                even[n * l..(n + 1) * l].fill(0);
                odd[n * l..(n + 1) * l].fill(0);
            }
            _ => unreachable!("sums follow the plan"),
        }
    }
}

/// `1 / (1 + exp(-x))` with `x = pre(v)` for every value `v`, two at a time.
#[inline(always)]
fn sigmoid_inplace(slice: &mut [f64], pre: impl Fn(f64) -> f64) {
    let (chunks, remainder) = slice.split_at_mut(slice.len() & !1);
    for pair in chunks.chunks_exact_mut(2) {
        let (a, b) = ((-pre(pair[0])).exp(), (-pre(pair[1])).exp());
        pair[0] = 1.0 / (1.0 + a);
        pair[1] = 1.0 / (1.0 + b);
    }
    for v in remainder {
        *v = 1.0 / (1.0 + (-pre(*v)).exp());
    }
}

impl Model {
    /// Each tree sum divided by the averaging divisor, plus the base score, rounded at
    /// each step; skipped when both are identities.
    #[expect(clippy::float_cmp, reason = "exact fast-path identities")]
    #[inline(always)]
    pub(crate) fn apply_margin(&self, values: &mut [f64]) {
        let out = self.output;
        if out.divisor != 1.0 || out.base_score != 0.0 {
            for v in &mut *values {
                *v = *v / out.divisor + out.base_score;
            }
        }
    }

    /// The link function: identity, or the sigmoid of `alpha` times the margin.
    #[expect(clippy::float_cmp, reason = "exact fast-path identity")]
    #[inline(always)]
    pub(crate) fn apply_link(&self, values: &mut [f64]) {
        match self.output.postprocessor {
            Postprocessor::Identity => {}
            Postprocessor::Sigmoid(alpha) => {
                if alpha != 1.0 {
                    for v in &mut *values {
                        *v *= alpha;
                    }
                }
                sigmoid_inplace(values, |v| v);
            }
        }
    }

    /// Turn tree sums into outputs: [`Self::apply_margin`] then [`Self::apply_link`],
    /// in one pass, each value through the same operations in the same order; the
    /// identities are skipped per call, as there.
    #[expect(clippy::float_cmp, reason = "exact fast-path identities")]
    #[inline]
    pub(super) fn finalize(&self, values: &mut [f64]) {
        let out = self.output;
        let (divisor, base) = (out.divisor, out.base_score);
        let margin = divisor != 1.0 || base != 0.0;
        match out.postprocessor {
            Postprocessor::Identity => self.apply_margin(values),
            Postprocessor::Sigmoid(alpha) => match (margin, alpha != 1.0) {
                (false, false) => sigmoid_inplace(values, |v| v),
                (true, false) => sigmoid_inplace(values, |v| v / divisor + base),
                (false, true) => sigmoid_inplace(values, |v| v * alpha),
                (true, true) => sigmoid_inplace(values, |v| (v / divisor + base) * alpha),
            },
        }
    }
}
