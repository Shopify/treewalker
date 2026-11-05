//! The finish: each row's exact sum rounded once from the difference arrays, then the
//! model's averaging, base score and link function.

use super::isa::Isa;
use super::scratch::Sums;
use super::sigmoid::sigmoid_inplace;
use crate::exact::{self, Plan};
use crate::model::{Model, Postprocessor};

impl Model {
    /// [`Self::finish`] at level `X` ([`Isa`]): where the build dispatches, out of line,
    /// with the body compiled inside the level's trampoline; elsewhere inlined.
    #[cfg_attr(not(dispatch), inline(always))]
    pub(super) fn finish_at<X: Isa>(&self, results: &mut [f64], sums: &mut Sums) {
        if X::NATIVE {
            return self.finish(results, sums);
        }
        X::with3(
            #[inline(always)]
            |m: &Self, results, sums| m.finish(results, sums),
            self,
            results,
            sums,
        );
    }

    /// Prefix-sum both difference arrays into `results`, clearing them.
    #[cfg_attr(dispatch, inline(always))]
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

impl Model {
    /// Each tree sum divided by the averaging divisor, plus the base score, rounded at
    /// each step; skipped when both are identities, and the division by a divisor of 1,
    /// whose quotient is the sum exactly.
    #[expect(clippy::float_cmp, reason = "exact fast-path identities")]
    #[inline(always)]
    pub(crate) fn apply_margin(&self, values: &mut [f64]) {
        let out = self.output;
        if out.divisor != 1.0 {
            for v in &mut *values {
                *v = *v / out.divisor + out.base_score;
            }
        } else if out.base_score != 0.0 {
            for v in &mut *values {
                *v += out.base_score;
            }
        }
    }

    /// The link function: identity, or the sigmoid of `alpha` times the margin.
    #[cfg(feature = "research")]
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

    /// [`Self::finalize`] at level `X` ([`Isa`]): where the build dispatches, out of line,
    /// with the body compiled inside the level's trampoline; elsewhere inlined.
    #[cfg_attr(not(dispatch), inline(always))]
    #[cfg_attr(dispatch, inline)]
    pub(super) fn finalize_at<X: Isa>(&self, values: &mut [f64]) {
        if X::NATIVE {
            return self.finalize(values);
        }
        X::with2(
            #[inline(always)]
            |m: &Self, values| m.finalize(values),
            self,
            values,
        );
    }

    /// Turn tree sums into outputs: [`Self::apply_margin`] then [`Self::apply_link`],
    /// in one pass, each value through the same operations in the same order; the
    /// identities and the division by 1 are skipped per call, as there.
    #[expect(clippy::float_cmp, reason = "exact fast-path identities")]
    #[cfg_attr(dispatch, inline(always))]
    #[cfg_attr(not(dispatch), inline)]
    pub(super) fn finalize(&self, values: &mut [f64]) {
        let out = self.output;
        let (divisor, base) = (out.divisor, out.base_score);
        match out.postprocessor {
            Postprocessor::Identity => self.apply_margin(values),
            Postprocessor::Sigmoid(alpha) => match (divisor != 1.0, base != 0.0, alpha != 1.0) {
                (false, false, false) => sigmoid_inplace(values, |v| v),
                (false, false, true) => sigmoid_inplace(values, |v| v * alpha),
                (false, true, false) => sigmoid_inplace(values, |v| v + base),
                (false, true, true) => sigmoid_inplace(values, |v| (v + base) * alpha),
                (true, _, false) => sigmoid_inplace(values, |v| v / divisor + base),
                (true, _, true) => sigmoid_inplace(values, |v| (v / divisor + base) * alpha),
            },
        }
    }
}
