//! Baseline adapters: compiled libraries through `libloading`, and QuickScorer.
//!
//! Every adapter takes the same row-major `f64` rows and returns the model's output,
//! the link applied, so every method computes the same quantity and is validated the
//! same way. Each call times its own work between [`timer::start`] and
//! [`timer::stop`]: whatever the library does to consume `f64` rows is inside, setup
//! (load, compile, proxy objects, buffers) is not.
//!
//! A one-row serving call uses the library's single-row interface. A multi-row call,
//! and every call in batch mode, uses one of the adapter's candidates: its
//! multi-row call, or a loop of its single-row interface over the rows. The runner
//! probes the candidates on each cell and mode and times the faster
//! ([`ExternalMethod::candidates`]):
//!
//! | Adapter | One row | Many rows: candidates |
//! |---|---|---|
//! | LightGBM | `PredictForMatSingleRowFast` | `PredictForMat`; a `PredictForMatSingleRowFast` loop |
//! | XGBoost | `PredictFromDense` | one `PredictFromDense` call; a loop of 1-row calls |
//! | lleaves | `forest_root` | `forest_root` |
//! | tl2cgen | the generated `predict` | runtime `DMatrixCreateFromMat` + `PredictBatch`; a `predict` loop |
//! | QuickScorer | `score_fast` | `score_collection`; a `score_fast` loop |

#![cfg_attr(
    feature = "quickscorer-bench",
    expect(
        clippy::inline_always,
        reason = "the counter reads must sit right around each library call"
    )
)]

#[cfg(feature = "quickscorer-bench")]
use crate::timer;

/// Output precision, which sets the validation tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Precision {
    /// Thresholds, leaves and sums in `f64`: 1e-13.
    F64,
    /// Some stage rounds to `f32`: 1e-5.
    F32,
}

impl Precision {
    pub const fn tolerance(self) -> f64 {
        match self {
            Self::F64 => 1e-13,
            Self::F32 => 1e-5,
        }
    }
}

/// The name of tl2cgen's float64 missing-value defect, as validation records it.
pub const TL2CGEN_F64_MISSING_ALIAS: &str =
    "tl2cgen-f64-missing-alias: Entry union int missing vs double fvalue";

/// Whether tl2cgen reads `value` as missing in a float64-threshold model.
///
/// Its `Entry` union is `{ int missing; double fvalue; }`, and the generated code
/// tests `data[i].missing != -1` before using a value (tl2cgen 1.0
/// `main_node.cc`, `condition_node.cc`). A non-NaN double whose low 32 bits are all
/// ones aliases `missing == -1`, so tl2cgen takes the default direction for it. In
/// float32 models the same bits would be a NaN, so nothing aliases there.
pub const fn tl2cgen_f64_missing_alias(value: f64) -> bool {
    !value.is_nan() && value.to_bits() as u32 == u32::MAX
}

/// Why QuickScorer cannot take a model with categorical splits.
pub const QUICKSCORER_CATEGORICAL: &str = "unsupported: categorical splits";

/// Whether a LightGBM text model has a categorical split: a `decision_type` value
/// with bit 0 set, LightGBM's `kCategoricalMask`.
///
/// QuickScorer has no categorical split: it would read one as a threshold and
/// predict wrongly, so the runner excludes it up front. TreeWalker's
/// `bitset_bytes` counts only out-of-line bitsets and misses small categories.
pub fn lightgbm_has_categorical(model_text: &str) -> bool {
    model_text
        .lines()
        .filter_map(|l| l.strip_prefix("decision_type="))
        .flat_map(str::split_whitespace)
        .any(|d| d.parse::<u32>().is_ok_and(|d| d & 1 == 1))
}

/// Why QuickScorer cannot take a cell whose missing values the model routes
/// otherwise.
pub const QUICKSCORER_MISSING: &str = "unsupported: missing values";

/// The features on which a LightGBM text model has a split that may not send a
/// missing value left.
///
/// QuickScorer has no missing-value handling: a NaN never passes `th < x`, so it
/// always goes left. `decision_type`'s bits 2-3 are the missing type: NaN sends
/// missing values the default way (bit 1, default left), while None (NaN read as
/// 0) and Zero may send them right. The rule is conservative, not exact: None with
/// a nonnegative threshold and Zero with default left send a NaN left too, but are
/// counted. No valid real cell is lost to it (on the Expedia cells it agrees with
/// validation), and validation stays the backstop.
pub fn lightgbm_missing_right_features(model_text: &str) -> std::collections::BTreeSet<usize> {
    let mut out = std::collections::BTreeSet::new();
    let mut features: Vec<usize> = Vec::new();
    for line in model_text.lines() {
        if let Some(v) = line.strip_prefix("split_feature=") {
            features = v
                .split_whitespace()
                .filter_map(|f| f.parse().ok())
                .collect();
        } else if let Some(v) = line.strip_prefix("decision_type=") {
            for (d, &f) in v.split_whitespace().zip(&features) {
                let Ok(d) = d.parse::<u32>() else { continue };
                let (default_left, missing) = (d & 2 != 0, (d >> 2) & 3);
                if !(missing == 2 && default_left) {
                    out.insert(f);
                }
            }
        }
    }
    out
}

/// The features a QuickScorer XML model splits on (its `<feature>` numbers are
/// 1-based).
pub fn quickscorer_xml_features(xml: &str) -> std::collections::BTreeSet<usize> {
    xml.split("<feature>")
        .skip(1)
        .filter_map(|s| s.split('<').next()?.trim().parse::<usize>().ok())
        .filter_map(|f| f.checked_sub(1))
        .collect()
}

/// A known defect: its name, and whether a row's inputs make it show.
pub type KnownDefect = (&'static str, fn(&[f64]) -> bool);

/// A baseline that predicts rows through one library.
pub trait ExternalMethod {
    /// The interface a call over `n_rows` rows uses; `multi` forces the multi-row
    /// interface, as batch mode does. A multi-row call uses the selected candidate.
    fn interface(&self, n_rows: usize, multi: bool) -> &'static str;

    /// The interfaces a multi-row call can use, in probe order: the library's
    /// multi-row call first, then a loop over its single-row interface where it has
    /// one.
    fn candidates(&self) -> Vec<&'static str> {
        vec![self.interface(2, true)]
    }

    /// Use candidate `i` of [`Self::candidates`] for multi-row calls.
    fn select(&mut self, i: usize) {
        let _ = i;
    }

    fn precision(&self) -> Precision;

    /// The library's effective settings after load, recorded in the cell manifest;
    /// `Null` when the adapter has none to report.
    fn settings(&self) -> serde_json::Value {
        serde_json::Value::Null
    }

    /// Predict `rows` (`n_rows × n_features`, row-major) and return the ticks the
    /// call took. The output stays in the adapter until the next call.
    fn predict_timed(&mut self, rows: &[f64], n_rows: usize, multi: bool) -> Result<u64, String>;

    /// The last call's output as `f64`, for validation.
    fn output(&self, n_rows: usize) -> Vec<f64>;

    /// Only the input conversion of the call [`Self::predict_timed`] would make, into
    /// the same buffers, untimed: the conversion pass times many of these in one
    /// interval. False for adapters whose conversion is inside the library.
    fn convert_only(&mut self, rows: &[f64], n_rows: usize, multi: bool) -> bool {
        let _ = (rows, n_rows, multi);
        false
    }

    /// A known defect of the library that makes some rows' outputs differ, found
    /// from the input alone: its name, and whether a row is affected. Validation
    /// checks the unaffected rows as usual and records the affected ones.
    fn known_defect(&self) -> Option<crate::external::KnownDefect> {
        None
    }
}

/// Check a call's shape against an adapter's buffers, outside the timed region:
/// `rows` holds exactly `n_rows` rows of `n_features`, and the output has room.
pub fn check_call(
    rows: &[f64],
    n_rows: usize,
    n_features: usize,
    capacity: usize,
) -> Result<(), String> {
    if n_rows == 0 || n_rows > capacity {
        return Err(format!("{n_rows} rows for an output of {capacity}"));
    }
    if Some(rows.len()) != n_rows.checked_mul(n_features) {
        return Err(format!(
            "{} values for {n_rows} rows of {n_features}",
            rows.len()
        ));
    }
    Ok(())
}

/// Run `f` between two counter reads.
#[cfg(feature = "quickscorer-bench")]
#[inline(always)]
fn timed<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let t0 = timer::start();
    let out = f();
    let t1 = timer::stop();
    (out, t1.wrapping_sub(t0))
}

#[cfg(feature = "quickscorer-bench")]
mod quickscorer;
#[cfg(feature = "quickscorer-bench")]
pub use quickscorer::{Link, QuickScorerBench};

#[cfg(feature = "external-bench")]
mod native;
#[cfg(feature = "external-bench")]
pub use native::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_values_the_model_sends_right_are_found() {
        // decision_type: bit 1 default left, bits 2-3 missing type (0 None, 1
        // Zero, 2 NaN). 10 = NaN, default left: QuickScorer agrees. 8 = NaN,
        // default right; 0 = None; 6 = Zero, default left: all may differ.
        let text = "Tree=0\nsplit_feature=3 4 5 6\ndecision_type=10 8 0 6\n";
        let got: Vec<usize> = lightgbm_missing_right_features(text).into_iter().collect();
        assert_eq!(got, [4, 5, 6]);
        let xml = "<tree><split><feature>1</feature><threshold>0.5</threshold>\
                   <split pos=\"left\"><feature> 12 </feature></split></split></tree>";
        let got: Vec<usize> = quickscorer_xml_features(xml).into_iter().collect();
        assert_eq!(got, [0, 11]);
    }

    #[test]
    fn categorical_splits_are_found_from_decision_type_bit_0() {
        // LightGBM's decision_type: bit 0 categorical, bit 1 default left, bits 2-3
        // the missing type. 2 and 10 are numerical splits, 1 and 9 categorical.
        let numerical = "Tree=0\nnum_leaves=3\ndecision_type=2 10 8\nthreshold=1 2\n";
        assert!(!lightgbm_has_categorical(numerical));
        let categorical = "Tree=0\ndecision_type=2 2\n\nTree=1\ndecision_type=10 9\n";
        assert!(lightgbm_has_categorical(categorical));
        assert!(lightgbm_has_categorical("decision_type=1"));
        // A one-leaf tree has no decision_type line; other keys never count.
        assert!(!lightgbm_has_categorical(
            "Tree=0\nnum_leaves=1\ncat_threshold=1\n"
        ));
    }
}
