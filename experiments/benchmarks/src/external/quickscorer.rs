//! QuickScorer, from its Rust crate: the one baseline that needs no C library.

use crate::external::{ExternalMethod, Precision, check_call, timed};

/// The model's finalization, which QuickScorer leaves to the caller: it returns the
/// raw leaf sum.
#[derive(Debug, Clone, Copy)]
pub struct Link {
    pub divisor: f64,
    pub base_score: f64,
    /// `None` for identity.
    pub sigmoid_alpha: Option<f64>,
}

impl Link {
    #[inline]
    fn apply(self, tree_sum: f64) -> f64 {
        let margin = tree_sum / self.divisor + self.base_score;
        self.sigmoid_alpha
            .map_or(margin, |alpha| 1.0 / (1.0 + (-(margin * alpha)).exp()))
    }
}

/// QuickScorer: `score_fast` for one row, `score_collection` for many.
///
/// Both take `f32` rows, so the copy from `f64` is timed, and the link is applied
/// inside the timed call. Thresholds and leaves are `f32`, and missing values go left
/// at every split; validation decides per cell whether that matches the model.
pub struct QuickScorerBench {
    qs: quickscorer::QuickScorer,
    link: Link,
    n_features: usize,
    row_f32: Vec<f32>,
    rows_f32: Vec<f32>,
    out: Vec<f64>,
    /// Multi-row calls loop `score_fast` over the rows.
    row_loop: bool,
}

const QUICKSCORER_CANDIDATES: [&str; 2] = ["score_collection", "score_fast loop"];

impl QuickScorerBench {
    /// Load a LightGBM text model or a QuickScorer XML model. Trees with more than
    /// 128 leaves are rejected: the crate asserts on them.
    pub fn load(
        model: &std::path::Path,
        link: Link,
        n_features: usize,
        max_rows: usize,
    ) -> Result<Self, String> {
        let qs = std::panic::catch_unwind(|| quickscorer::QuickScorer::from_model_file(model))
            .map_err(|_| "QuickScorer rejected the model (more than 128 leaves in a tree)")?
            .map_err(|e| format!("QuickScorer could not load the model: {e}"))?;
        Ok(Self {
            qs,
            link,
            n_features,
            row_f32: vec![0.0; n_features],
            rows_f32: vec![0.0; n_features * max_rows],
            out: vec![0.0; max_rows],
            row_loop: false,
        })
    }
}

impl ExternalMethod for QuickScorerBench {
    fn interface(&self, n_rows: usize, multi: bool) -> &'static str {
        if n_rows == 1 && !multi {
            "score_fast"
        } else {
            QUICKSCORER_CANDIDATES[usize::from(self.row_loop)]
        }
    }

    fn candidates(&self) -> Vec<&'static str> {
        QUICKSCORER_CANDIDATES.to_vec()
    }

    fn select(&mut self, i: usize) {
        self.row_loop = i == 1;
    }

    fn precision(&self) -> Precision {
        Precision::F32
    }

    fn predict_timed(&mut self, rows: &[f64], n_rows: usize, multi: bool) -> Result<u64, String> {
        let nf = self.n_features;
        check_call(rows, n_rows, nf, self.out.len())?;
        let link = self.link;
        let ((), ticks) = if n_rows == 1 && !multi {
            let (qs, buf, out) = (&mut self.qs, &mut self.row_f32, &mut self.out);
            timed(|| {
                for (d, &s) in buf.iter_mut().zip(rows) {
                    *d = s as f32;
                }
                out[0] = link.apply(qs.score_fast(buf));
            })
        } else if self.row_loop {
            let (qs, buf, out) = (&mut self.qs, &mut self.row_f32, &mut self.out);
            timed(|| {
                for (row, o) in rows.chunks_exact(nf).zip(out.iter_mut()) {
                    for (d, &s) in buf.iter_mut().zip(row) {
                        *d = s as f32;
                    }
                    *o = link.apply(qs.score_fast(buf));
                }
            })
        } else {
            let (qs, buf, out) = (&mut self.qs, &mut self.rows_f32, &mut self.out);
            timed(|| {
                for (d, &s) in buf[..n_rows * nf].iter_mut().zip(rows) {
                    *d = s as f32;
                }
                let docs: Vec<&[f32]> = buf[..n_rows * nf].chunks_exact(nf).collect();
                let scores = qs.score_collection(&docs);
                for (o, s) in out.iter_mut().zip(scores) {
                    *o = link.apply(s);
                }
            })
        };
        std::hint::black_box(&self.out[..n_rows]);
        Ok(ticks)
    }

    fn output(&self, n_rows: usize) -> Vec<f64> {
        self.out[..n_rows].to_vec()
    }

    fn convert_only(&mut self, rows: &[f64], n_rows: usize, multi: bool) -> bool {
        let nf = self.n_features;
        if check_call(rows, n_rows, nf, self.out.len()).is_err() {
            return false;
        }
        if (n_rows == 1 && !multi) || self.row_loop {
            for row in rows.chunks_exact(nf) {
                for (d, &s) in self.row_f32.iter_mut().zip(row) {
                    *d = s as f32;
                }
                std::hint::black_box(&self.row_f32);
            }
        } else {
            for (d, &s) in self.rows_f32[..n_rows * nf].iter_mut().zip(rows) {
                *d = s as f32;
            }
            std::hint::black_box(&self.rows_f32[..n_rows * nf]);
        }
        true
    }
}
