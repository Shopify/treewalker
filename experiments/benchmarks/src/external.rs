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

// ---------------------------------------------------------------------------
// QuickScorer
// ---------------------------------------------------------------------------

/// The model's finalization, which QuickScorer leaves to the caller: it returns the
/// raw leaf sum.
#[cfg(feature = "quickscorer-bench")]
#[derive(Debug, Clone, Copy)]
pub struct Link {
    pub divisor: f64,
    pub base_score: f64,
    /// `None` for identity.
    pub sigmoid_alpha: Option<f64>,
}

#[cfg(feature = "quickscorer-bench")]
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
#[cfg(feature = "quickscorer-bench")]
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

#[cfg(feature = "quickscorer-bench")]
const QUICKSCORER_CANDIDATES: [&str; 2] = ["score_collection", "score_fast loop"];

#[cfg(feature = "quickscorer-bench")]
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

#[cfg(feature = "quickscorer-bench")]
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

#[cfg(feature = "external-bench")]
pub use native::*;

#[cfg(feature = "external-bench")]
mod native {
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_double, c_int, c_void};
    use std::path::Path;

    use libloading::Library;

    use super::{ExternalMethod, Precision, timed};

    /// A libloading error with the system's `dlerror` message, which libloading
    /// reports as the error's source.
    fn dl_error(context: &str, e: &libloading::Error) -> String {
        use std::error::Error as _;
        e.source().map_or_else(
            || format!("{context}: {e}"),
            |cause| format!("{context}: {e}: {cause}"),
        )
    }

    /// Resolve `name` in `lib` as a value of type `T` (a function pointer).
    ///
    /// # Safety
    /// `T` must be the symbol's true type.
    unsafe fn sym<T: Copy>(lib: &Library, name: &CStr) -> Result<T, String> {
        // SAFETY: the caller guarantees the type.
        unsafe { lib.get::<T>(name.to_bytes_with_nul()) }
            .map(|s| *s)
            .map_err(|e| dl_error(&format!("symbol {}", name.to_string_lossy()), &e))
    }

    fn cstr(s: &str) -> CString {
        CString::new(s).expect("no interior NUL")
    }

    type GetStr = unsafe extern "C" fn() -> *const c_char;

    /// The text of a library's last-error function, which returns a C string.
    fn error_text(f: GetStr) -> String {
        // SAFETY: the library returns a valid NUL-terminated string.
        unsafe { CStr::from_ptr(f()) }
            .to_string_lossy()
            .into_owned()
    }

    /// Load a native library and resolve every C call its adapter makes.
    pub fn probe(path: &Path, kind: &str) -> Result<(), String> {
        let symbols: &[&CStr] = match kind {
            "lightgbm" => &[
                c"LGBM_BoosterCreateFromModelfile",
                c"LGBM_BoosterPredictForMat",
                c"LGBM_BoosterPredictForMatSingleRowFastInit",
                c"LGBM_BoosterPredictForMatSingleRowFast",
                c"LGBM_BoosterGetNumClasses",
                c"LGBM_BoosterGetNumFeature",
                c"LGBM_FastConfigFree",
                c"LGBM_BoosterFree",
                c"LGBM_GetLastError",
            ],
            "xgboost" => &[
                c"XGBoosterCreate",
                c"XGBoosterLoadModel",
                c"XGBoosterSetParam",
                c"XGProxyDMatrixCreate",
                c"XGBoosterPredictFromDense",
                c"XGBoosterGetNumFeature",
                c"XGDMatrixFree",
                c"XGBoosterFree",
                c"XGBGetLastError",
            ],
            other => return Err(format!("unknown native library {other}")),
        };
        // SAFETY: loading runs the library's initializers; the symbols are only
        // looked up, never called.
        unsafe {
            let lib = Library::new(path)
                .map_err(|e| dl_error(&format!("loading {}", path.display()), &e))?;
            for name in symbols {
                lib.get::<*const c_void>(name.to_bytes_with_nul())
                    .map_err(|e| dl_error(&format!("symbol {}", name.to_string_lossy()), &e))?;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // lleaves
    // -----------------------------------------------------------------------

    type ForestRoot = unsafe extern "C" fn(*const c_double, *mut c_double, c_int, c_int);

    /// lleaves: `forest_root(data, out, start, end)` over the call's rows.
    pub struct LleavesBench {
        _lib: Library,
        forest_root: ForestRoot,
        n_features: usize,
        out: Vec<f64>,
    }

    impl LleavesBench {
        pub fn load(path: &Path, n_features: usize, max_rows: usize) -> Result<Self, String> {
            // SAFETY: an lleaves library exports forest_root with this signature.
            unsafe {
                let lib = Library::new(path)
                    .map_err(|e| dl_error(&format!("loading {}", path.display()), &e))?;
                let forest_root = sym::<ForestRoot>(&lib, c"forest_root")?;
                Ok(Self {
                    _lib: lib,
                    forest_root,
                    n_features,
                    out: vec![0.0; max_rows],
                })
            }
        }
    }

    impl ExternalMethod for LleavesBench {
        fn interface(&self, _: usize, _: bool) -> &'static str {
            "forest_root"
        }
        fn precision(&self) -> Precision {
            Precision::F64
        }
        fn predict_timed(&mut self, rows: &[f64], n: usize, _: bool) -> Result<u64, String> {
            super::check_call(rows, n, self.n_features, self.out.len())?;
            let (f, out) = (self.forest_root, self.out.as_mut_ptr());
            // SAFETY: rows holds n rows and out at least n values.
            let ((), ticks) = timed(|| unsafe { f(rows.as_ptr(), out, 0, n as c_int) });
            std::hint::black_box(&self.out[..n]);
            Ok(ticks)
        }
        fn output(&self, n: usize) -> Vec<f64> {
            self.out[..n].to_vec()
        }
    }

    // -----------------------------------------------------------------------
    // tl2cgen
    // -----------------------------------------------------------------------

    /// tl2cgen's `Entry` union for float32 thresholds: `{ int missing; float fvalue; }`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    union Entry32 {
        missing: i32,
        fvalue: f32,
    }

    /// tl2cgen's `Entry` union for float64 thresholds: `{ int missing; double fvalue; }`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    union Entry64 {
        missing: i32,
        fvalue: f64,
    }

    /// Zero `n` outputs of tl2cgen's output type.
    ///
    /// # Safety
    /// `out` must hold at least `n` values of the type.
    #[inline(always)]
    const unsafe fn zero(out: *mut c_void, f32_out: bool, n: usize) {
        // SAFETY: the caller guarantees the size.
        unsafe {
            if f32_out {
                std::ptr::write_bytes(out.cast::<f32>(), 0, n);
            } else {
                std::ptr::write_bytes(out.cast::<f64>(), 0, n);
            }
        }
    }

    /// Row `r` of an output buffer of tl2cgen's output type.
    ///
    /// # Safety
    /// `out` must hold more than `r` values of the type.
    #[inline(always)]
    const unsafe fn out_at(out: *mut c_void, f32_out: bool, r: usize) -> *mut c_void {
        // SAFETY: the caller guarantees the size.
        unsafe {
            if f32_out {
                out.cast::<f32>().add(r).cast()
            } else {
                out.cast::<f64>().add(r).cast()
            }
        }
    }

    const TL2CGEN_CANDIDATES: [&str; 2] = ["DMatrixCreateFromMat+PredictBatch", "predict loop"];

    type Predict32 = unsafe extern "C" fn(*mut Entry32, c_int, *mut c_void);
    type Predict64 = unsafe extern "C" fn(*mut Entry64, c_int, *mut c_void);
    type Handle = *mut c_void;

    struct Tl2cgenRuntime {
        _lib: Library,
        create: unsafe extern "C" fn(
            *const c_void,
            *const c_char,
            u64,
            u64,
            *const c_void,
            *mut Handle,
        ) -> c_int,
        free_dmat: unsafe extern "C" fn(Handle) -> c_int,
        predict_batch: unsafe extern "C" fn(Handle, Handle, c_int, c_int, *mut c_void) -> c_int,
        free_predictor: unsafe extern "C" fn(Handle) -> c_int,
        last_error: unsafe extern "C" fn() -> *const c_char,
        predictor: Handle,
    }

    impl Drop for Tl2cgenRuntime {
        fn drop(&mut self) {
            // SAFETY: the predictor was created by TL2cgenPredictorLoad.
            unsafe { (self.free_predictor)(self.predictor) };
        }
    }

    /// tl2cgen: the generated `predict` for one row, the runtime library for many.
    ///
    /// Float64 models misread some values as missing; see
    /// [`super::tl2cgen_f64_missing_alias`]. The adapter passes values as tl2cgen's
    /// own runtime does, so its outputs keep the defect, and validation attributes
    /// it.
    ///
    /// The entry and output types follow the library's `get_threshold_type()` and
    /// `get_leaf_output_type()` (finding 0.14). The runtime's matrix creation copies
    /// the rows and is timed with the prediction; it casts values to the threshold
    /// type itself. The generated code adds leaves into the output, so each call
    /// zeroes it first, as tl2cgen's Python predictor does by allocating zeros.
    pub struct Tl2cgenBench {
        _lib: Library,
        predict32: Option<Predict32>,
        predict64: Option<Predict64>,
        entries32: Vec<Entry32>,
        entries64: Vec<Entry64>,
        out_f32: bool,
        out32: Vec<f32>,
        out64: Vec<f64>,
        runtime: Tl2cgenRuntime,
        n_features: usize,
        dtype: CString,
        missing: f64,
        /// Multi-row calls loop the generated `predict` over the rows.
        row_loop: bool,
    }

    impl Tl2cgenBench {
        /// Load a compiled model and the tl2cgen runtime (`libtl2cgen`).
        pub fn load(
            model: &Path,
            runtime: &Path,
            n_features: usize,
            max_rows: usize,
        ) -> Result<Self, String> {
            // SAFETY: the model library exports tl2cgen's generated API, and the
            // runtime its C API, with the signatures in tl2cgen 1.0's main_node.cc and
            // include/tl2cgen/c_api.h.
            unsafe {
                let lib = Library::new(model)
                    .map_err(|e| dl_error(&format!("loading {}", model.display()), &e))?;
                let threshold =
                    CStr::from_ptr(sym::<GetStr>(&lib, c"get_threshold_type")?()).to_owned();
                let leaf =
                    CStr::from_ptr(sym::<GetStr>(&lib, c"get_leaf_output_type")?()).to_owned();
                let nf = sym::<unsafe extern "C" fn() -> i32>(&lib, c"get_num_feature")?();
                // One output per row: a scalar model, one target and one class.
                let targets = sym::<unsafe extern "C" fn() -> i32>(&lib, c"get_num_target")?();
                let mut classes = [0i32; 1];
                if targets == 1 {
                    sym::<unsafe extern "C" fn(*mut i32)>(&lib, c"get_num_class")?(
                        classes.as_mut_ptr(),
                    );
                }
                if targets != 1 || classes[0] != 1 {
                    return Err(format!(
                        "{targets} targets, {} classes; only scalar models",
                        classes[0]
                    ));
                }
                if nf as usize > n_features {
                    return Err(format!("model has {nf} features, data {n_features}"));
                }
                let f32_thresholds = match threshold.to_bytes() {
                    b"float32" => true,
                    b"float64" => false,
                    t => return Err(format!("threshold type {}", String::from_utf8_lossy(t))),
                };
                let out_f32 = match leaf.to_bytes() {
                    b"float32" => true,
                    b"float64" => false,
                    t => return Err(format!("leaf output type {}", String::from_utf8_lossy(t))),
                };
                let (predict32, predict64) = if f32_thresholds {
                    (Some(sym::<Predict32>(&lib, c"predict")?), None)
                } else {
                    (None, Some(sym::<Predict64>(&lib, c"predict")?))
                };

                let rt = Library::new(runtime)
                    .map_err(|e| dl_error(&format!("loading {}", runtime.display()), &e))?;
                let load = sym::<unsafe extern "C" fn(*const c_char, c_int, *mut Handle) -> c_int>(
                    &rt,
                    c"TL2cgenPredictorLoad",
                )?;
                let last_error = sym::<GetStr>(&rt, c"TL2cgenGetLastError")?;
                let path = cstr(&model.to_string_lossy());
                let mut predictor: Handle = std::ptr::null_mut();
                if load(path.as_ptr(), 1, &raw mut predictor) != 0 {
                    return Err(format!("TL2cgenPredictorLoad: {}", error_text(last_error)));
                }
                let runtime = Tl2cgenRuntime {
                    create: sym(&rt, c"TL2cgenDMatrixCreateFromMat")?,
                    free_dmat: sym(&rt, c"TL2cgenDMatrixFree")?,
                    predict_batch: sym(&rt, c"TL2cgenPredictorPredictBatch")?,
                    free_predictor: sym(&rt, c"TL2cgenPredictorFree")?,
                    last_error,
                    predictor,
                    _lib: rt,
                };
                Ok(Self {
                    _lib: lib,
                    predict32,
                    predict64,
                    entries32: vec![Entry32 { missing: -1 }; n_features],
                    entries64: vec![Entry64 { missing: -1 }; n_features],
                    out_f32,
                    out32: vec![0.0; max_rows],
                    out64: vec![0.0; max_rows],
                    runtime,
                    n_features,
                    dtype: cstr("float64"),
                    missing: f64::NAN,
                    row_loop: false,
                })
            }
        }

        const fn out_ptr(&mut self) -> *mut c_void {
            if self.out_f32 {
                self.out32.as_mut_ptr().cast()
            } else {
                self.out64.as_mut_ptr().cast()
            }
        }
    }

    impl ExternalMethod for Tl2cgenBench {
        fn known_defect(&self) -> Option<crate::external::KnownDefect> {
            fn affected(row: &[f64]) -> bool {
                row.iter().any(|&v| super::tl2cgen_f64_missing_alias(v))
            }
            self.predict64.is_some().then_some((
                super::TL2CGEN_F64_MISSING_ALIAS,
                affected as fn(&[f64]) -> bool,
            ))
        }
        fn interface(&self, n: usize, multi: bool) -> &'static str {
            if n == 1 && !multi {
                "predict"
            } else {
                TL2CGEN_CANDIDATES[usize::from(self.row_loop)]
            }
        }
        fn candidates(&self) -> Vec<&'static str> {
            TL2CGEN_CANDIDATES.to_vec()
        }
        fn select(&mut self, i: usize) {
            self.row_loop = i == 1;
        }
        fn precision(&self) -> Precision {
            if self.out_f32 || self.predict32.is_some() {
                Precision::F32
            } else {
                Precision::F64
            }
        }
        fn predict_timed(&mut self, rows: &[f64], n: usize, multi: bool) -> Result<u64, String> {
            super::check_call(rows, n, self.n_features, self.out64.len())?;
            let (out, out_f32) = (self.out_ptr(), self.out_f32);
            let ticks = if n == 1 && !multi {
                // As tl2cgen's deployment guide does: fill the entries, call predict.
                if let Some(predict) = self.predict32 {
                    let entries = &mut self.entries32;
                    timed(|| {
                        for (e, &v) in entries.iter_mut().zip(rows) {
                            *e = if v.is_nan() {
                                Entry32 { missing: -1 }
                            } else {
                                Entry32 { fvalue: v as f32 }
                            };
                        }
                        // SAFETY: entries has n_features values; out has room for
                        // one value of the output type, zeroed first.
                        unsafe {
                            zero(out, out_f32, 1);
                            predict(entries.as_mut_ptr(), 0, out);
                        }
                    })
                    .1
                } else {
                    let predict = self.predict64.expect("one predict symbol");
                    let entries = &mut self.entries64;
                    timed(|| {
                        for (e, &v) in entries.iter_mut().zip(rows) {
                            *e = if v.is_nan() {
                                Entry64 { missing: -1 }
                            } else {
                                Entry64 { fvalue: v }
                            };
                        }
                        // SAFETY: as above.
                        unsafe {
                            zero(out, out_f32, 1);
                            predict(entries.as_mut_ptr(), 0, out);
                        }
                    })
                    .1
                }
            } else if self.row_loop {
                // The deployment guide's single-row call, once per row.
                let nf = self.n_features;
                if let Some(predict) = self.predict32 {
                    let entries = &mut self.entries32;
                    timed(|| {
                        for (r, row) in rows.chunks_exact(nf).enumerate() {
                            for (e, &v) in entries.iter_mut().zip(row) {
                                *e = if v.is_nan() {
                                    Entry32 { missing: -1 }
                                } else {
                                    Entry32 { fvalue: v as f32 }
                                };
                            }
                            // SAFETY: entries has n_features values; out has room
                            // for n values of the output type, and r < n.
                            unsafe {
                                let o = out_at(out, out_f32, r);
                                zero(o, out_f32, 1);
                                predict(entries.as_mut_ptr(), 0, o);
                            }
                        }
                    })
                    .1
                } else {
                    let predict = self.predict64.expect("one predict symbol");
                    let entries = &mut self.entries64;
                    timed(|| {
                        for (r, row) in rows.chunks_exact(nf).enumerate() {
                            for (e, &v) in entries.iter_mut().zip(row) {
                                *e = if v.is_nan() {
                                    Entry64 { missing: -1 }
                                } else {
                                    Entry64 { fvalue: v }
                                };
                            }
                            // SAFETY: as above.
                            unsafe {
                                let o = out_at(out, out_f32, r);
                                zero(o, out_f32, 1);
                                predict(entries.as_mut_ptr(), 0, o);
                            }
                        }
                    })
                    .1
                }
            } else {
                let rt = &self.runtime;
                let (dtype, missing, nf) = (
                    self.dtype.as_ptr(),
                    &raw const self.missing,
                    self.n_features,
                );
                let (rc, ticks) = timed(|| {
                    let mut dmat: Handle = std::ptr::null_mut();
                    // SAFETY: rows holds n × nf f64 values; dmat receives a new matrix,
                    // freed below; out has room for n values of the output type.
                    unsafe {
                        zero(out, out_f32, n);
                        let rc = (rt.create)(
                            rows.as_ptr().cast(),
                            dtype,
                            n as u64,
                            nf as u64,
                            missing.cast(),
                            &raw mut dmat,
                        );
                        if rc != 0 {
                            return rc;
                        }
                        let rc = (rt.predict_batch)(rt.predictor, dmat, 0, 0, out);
                        (rt.free_dmat)(dmat);
                        rc
                    }
                });
                if rc != 0 {
                    return Err(format!("tl2cgen runtime: {}", error_text(rt.last_error)));
                }
                ticks
            };
            if self.out_f32 {
                std::hint::black_box(&self.out32[..n]);
            } else {
                std::hint::black_box(&self.out64[..n]);
            }
            Ok(ticks)
        }
        fn output(&self, n: usize) -> Vec<f64> {
            if self.out_f32 {
                self.out32[..n].iter().map(|&v| f64::from(v)).collect()
            } else {
                self.out64[..n].to_vec()
            }
        }
    }

    // -----------------------------------------------------------------------
    // LightGBM
    // -----------------------------------------------------------------------

    type Booster = *mut c_void;
    type FastConfig = *mut c_void;
    type PredictForMat = unsafe extern "C" fn(
        Booster,
        *const c_void,
        c_int,
        i32,
        i32,
        c_int,
        c_int,
        c_int,
        c_int,
        *const c_char,
        *mut i64,
        *mut c_double,
    ) -> c_int;
    type FastInit = unsafe extern "C" fn(
        Booster,
        c_int,
        c_int,
        c_int,
        c_int,
        i32,
        *const c_char,
        *mut FastConfig,
    ) -> c_int;
    type FastPredict =
        unsafe extern "C" fn(FastConfig, *const c_void, *mut i64, *mut c_double) -> c_int;

    const C_API_DTYPE_FLOAT64: c_int = 1;
    const C_API_PREDICT_NORMAL: c_int = 0;

    /// LightGBM's C API: `PredictForMatSingleRowFast` for one row, `PredictForMat`
    /// for many.
    ///
    /// The `FastInit` configuration is created once per model as untimed setup.
    /// The row-loop candidate calls `PredictForMatSingleRowFast` once per row.
    pub struct LightGbmBench {
        _lib: Library,
        handle: Booster,
        fast: FastConfig,
        predict_mat: PredictForMat,
        predict_fast: FastPredict,
        free_booster: unsafe extern "C" fn(Booster) -> c_int,
        free_fast: unsafe extern "C" fn(FastConfig) -> c_int,
        last_error: unsafe extern "C" fn() -> *const c_char,
        params: CString,
        n_features: usize,
        row_loop: bool,
        out: Vec<f64>,
    }

    const LIGHTGBM_CANDIDATES: [&str; 2] = ["PredictForMat", "PredictForMatSingleRowFast loop"];

    impl LightGbmBench {
        pub fn load(
            lib_path: &Path,
            model: &Path,
            n_features: usize,
            max_rows: usize,
        ) -> Result<Self, String> {
            // SAFETY: lib_path is LightGBM's C library (v4.7.0 c_api.h signatures).
            unsafe {
                let lib = Library::new(lib_path)
                    .map_err(|e| dl_error(&format!("loading {}", lib_path.display()), &e))?;
                let create = sym::<
                    unsafe extern "C" fn(*const c_char, *mut c_int, *mut Booster) -> c_int,
                >(&lib, c"LGBM_BoosterCreateFromModelfile")?;
                let last_error =
                    sym::<unsafe extern "C" fn() -> *const c_char>(&lib, c"LGBM_GetLastError")?;
                let init = sym::<FastInit>(&lib, c"LGBM_BoosterPredictForMatSingleRowFastInit")?;
                let path = cstr(&model.to_string_lossy());
                let (mut iters, mut handle) = (0, std::ptr::null_mut());
                if create(path.as_ptr(), &raw mut iters, &raw mut handle) != 0 {
                    return Err(format!("LightGBM: {}", error_text(last_error)));
                }
                let (mut classes, mut features) = (0 as c_int, 0 as c_int);
                let num_classes = sym::<unsafe extern "C" fn(Booster, *mut c_int) -> c_int>(
                    &lib,
                    c"LGBM_BoosterGetNumClasses",
                )?;
                let num_features = sym::<unsafe extern "C" fn(Booster, *mut c_int) -> c_int>(
                    &lib,
                    c"LGBM_BoosterGetNumFeature",
                )?;
                if num_classes(handle, &raw mut classes) != 0
                    || num_features(handle, &raw mut features) != 0
                {
                    return Err(format!("LightGBM: {}", error_text(last_error)));
                }
                if classes != 1 || features as usize != n_features {
                    return Err(format!(
                        "LightGBM model has {classes} classes and {features} features; \
                         expected 1 and {n_features}"
                    ));
                }
                let params = cstr("num_threads=1");
                let mut fast = std::ptr::null_mut();
                if init(
                    handle,
                    C_API_PREDICT_NORMAL,
                    0,
                    0,
                    C_API_DTYPE_FLOAT64,
                    n_features as i32,
                    params.as_ptr(),
                    &raw mut fast,
                ) != 0
                {
                    return Err(format!("LightGBM FastInit: {}", error_text(last_error)));
                }
                Ok(Self {
                    predict_mat: sym(&lib, c"LGBM_BoosterPredictForMat")?,
                    predict_fast: sym(&lib, c"LGBM_BoosterPredictForMatSingleRowFast")?,
                    free_booster: sym(&lib, c"LGBM_BoosterFree")?,
                    free_fast: sym(&lib, c"LGBM_FastConfigFree")?,
                    last_error,
                    _lib: lib,
                    handle,
                    fast,
                    params,
                    n_features,
                    row_loop: false,
                    out: vec![0.0; max_rows],
                })
            }
        }
    }

    impl Drop for LightGbmBench {
        fn drop(&mut self) {
            // SAFETY: both handles were created at load.
            unsafe {
                (self.free_fast)(self.fast);
                (self.free_booster)(self.handle);
            }
        }
    }

    impl ExternalMethod for LightGbmBench {
        fn interface(&self, n: usize, multi: bool) -> &'static str {
            if n == 1 && !multi {
                "PredictForMatSingleRowFast"
            } else {
                LIGHTGBM_CANDIDATES[usize::from(self.row_loop)]
            }
        }
        fn candidates(&self) -> Vec<&'static str> {
            LIGHTGBM_CANDIDATES.to_vec()
        }
        fn select(&mut self, i: usize) {
            self.row_loop = i == 1;
        }
        fn precision(&self) -> Precision {
            Precision::F64
        }
        fn predict_timed(&mut self, rows: &[f64], n: usize, multi: bool) -> Result<u64, String> {
            let nf = self.n_features;
            super::check_call(rows, n, nf, self.out.len())?;
            let out = self.out.as_mut_ptr();
            let (rc, ticks) = if self.row_loop || (n == 1 && !multi) {
                let (fast, f) = (self.fast, self.predict_fast);
                timed(|| {
                    let mut len = 0i64;
                    for r in 0..n {
                        // SAFETY: row r has nf values; out has room for n.
                        let rc = unsafe {
                            f(
                                fast,
                                rows.as_ptr().add(r * nf).cast(),
                                &raw mut len,
                                out.add(r),
                            )
                        };
                        if rc != 0 {
                            return rc;
                        }
                    }
                    0
                })
            } else {
                let (h, f, p) = (self.handle, self.predict_mat, self.params.as_ptr());
                timed(|| {
                    let mut len = 0i64;
                    // SAFETY: rows holds n × nf f64 values, row-major; out room for n.
                    unsafe {
                        f(
                            h,
                            rows.as_ptr().cast(),
                            C_API_DTYPE_FLOAT64,
                            n as i32,
                            nf as i32,
                            1,
                            C_API_PREDICT_NORMAL,
                            0,
                            0,
                            p,
                            &raw mut len,
                            out,
                        )
                    }
                })
            };
            if rc != 0 {
                return Err(format!("LightGBM: {}", error_text(self.last_error)));
            }
            std::hint::black_box(&self.out[..n]);
            Ok(ticks)
        }
        fn output(&self, n: usize) -> Vec<f64> {
            self.out[..n].to_vec()
        }
    }

    // -----------------------------------------------------------------------
    // XGBoost
    // -----------------------------------------------------------------------

    type PredictFromDense = unsafe extern "C" fn(
        Handle,
        *const c_char,
        *const c_char,
        Handle,
        *mut *const u64,
        *mut u64,
        *mut *const f32,
    ) -> c_int;

    /// XGBoost's in-place prediction, `XGBoosterPredictFromDense`: one call for any
    /// row count, or the row-loop candidate's one call per row.
    ///
    /// It reads the `f64` rows through an array interface and converts them itself
    /// (finding 0.5); its output is `f32`, owned by XGBoost until the next call. The
    /// row loop copies each row's output into the adapter, inside the timed call.
    pub struct XgBoostBench {
        _lib: Library,
        handle: Handle,
        proxy: Handle,
        predict: PredictFromDense,
        free_proxy: unsafe extern "C" fn(Handle) -> c_int,
        free_booster: unsafe extern "C" fn(Handle) -> c_int,
        last_error: unsafe extern "C" fn() -> *const c_char,
        config: CString,
        n_features: usize,
        last: *const f32,
        last_len: usize,
        settings: serde_json::Value,
        row_loop: bool,
        loop_out: Vec<f32>,
    }

    const XGBOOST_CANDIDATES: [&str; 2] = ["PredictFromDense", "PredictFromDense 1-row loop"];

    impl XgBoostBench {
        pub fn load(
            lib_path: &Path,
            model: &Path,
            n_features: usize,
            max_rows: usize,
        ) -> Result<Self, String> {
            // SAFETY: lib_path is XGBoost's C library (3.x c_api.h signatures).
            unsafe {
                let lib = Library::new(lib_path)
                    .map_err(|e| dl_error(&format!("loading {}", lib_path.display()), &e))?;
                let last_error =
                    sym::<unsafe extern "C" fn() -> *const c_char>(&lib, c"XGBGetLastError")?;
                let err = || format!("XGBoost: {}", error_text(last_error));
                let create = sym::<unsafe extern "C" fn(*const Handle, u64, *mut Handle) -> c_int>(
                    &lib,
                    c"XGBoosterCreate",
                )?;
                let load = sym::<unsafe extern "C" fn(Handle, *const c_char) -> c_int>(
                    &lib,
                    c"XGBoosterLoadModel",
                )?;
                let set_param = sym::<
                    unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> c_int,
                >(&lib, c"XGBoosterSetParam")?;
                let proxy_create = sym::<unsafe extern "C" fn(*mut Handle) -> c_int>(
                    &lib,
                    c"XGProxyDMatrixCreate",
                )?;
                let mut handle = std::ptr::null_mut();
                if create(std::ptr::null(), 0, &raw mut handle) != 0 {
                    return Err(err());
                }
                let path = cstr(&model.to_string_lossy());
                if load(handle, path.as_ptr()) != 0 {
                    return Err(err());
                }
                if set_param(handle, c"nthread".as_ptr(), c"1".as_ptr()) != 0 {
                    return Err(err());
                }
                let mut proxy = std::ptr::null_mut();
                if proxy_create(&raw mut proxy) != 0 {
                    return Err(err());
                }
                let num_features = sym::<unsafe extern "C" fn(Handle, *mut u64) -> c_int>(
                    &lib,
                    c"XGBoosterGetNumFeature",
                )?;
                let mut features = 0u64;
                if num_features(handle, &raw mut features) != 0 {
                    return Err(err());
                }
                if features as usize != n_features {
                    return Err(format!(
                        "XGBoost model has {features} features, data {n_features}"
                    ));
                }
                // The booster's effective configuration (nthread included), read
                // once, untimed, for the cell manifest.
                let save_config = sym::<
                    unsafe extern "C" fn(Handle, *mut u64, *mut *const c_char) -> c_int,
                >(&lib, c"XGBoosterSaveJsonConfig")?;
                let (mut len, mut out): (u64, *const c_char) = (0, std::ptr::null());
                if save_config(handle, &raw mut len, &raw mut out) != 0 || out.is_null() {
                    return Err(err());
                }
                let mut settings: serde_json::Value =
                    serde_json::from_str(&CStr::from_ptr(out).to_string_lossy())
                        .map_err(|e| format!("XGBoost: its JSON config does not parse: {e}"))?;
                // The training RNG's 624-word state says nothing about prediction.
                if let Some(g) = settings
                    .pointer_mut("/learner/generic_param")
                    .and_then(serde_json::Value::as_object_mut)
                {
                    g.remove("rng_state");
                }
                Ok(Self {
                    predict: sym(&lib, c"XGBoosterPredictFromDense")?,
                    free_proxy: sym(&lib, c"XGDMatrixFree")?,
                    free_booster: sym(&lib, c"XGBoosterFree")?,
                    last_error,
                    _lib: lib,
                    handle,
                    proxy,
                    config: cstr(
                        r#"{"type":0,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":false,"missing":NaN}"#,
                    ),
                    n_features,
                    last: std::ptr::null(),
                    last_len: 0,
                    settings,
                    row_loop: false,
                    loop_out: vec![0.0; max_rows],
                })
            }
        }
    }

    impl XgBoostBench {
        /// One 1-row `PredictFromDense` call per row, each output copied out.
        fn predict_loop(&mut self, rows: &[f64], n: usize) -> Result<u64, String> {
            let nf = self.n_features;
            super::check_call(rows, n, nf, self.loop_out.len())?;
            let (h, proxy, f, config) =
                (self.handle, self.proxy, self.predict, self.config.as_ptr());
            let out = self.loop_out.as_mut_ptr();
            let (mut shape, mut dim): (*const u64, u64) = (std::ptr::null(), 0);
            let (rc, ticks) = timed(|| {
                for r in 0..n {
                    let array = format!(
                        r#"{{"data":[{},true],"shape":[1,{nf}],"typestr":"<f8","version":3}}"#,
                        rows.as_ptr() as usize + r * nf * std::mem::size_of::<f64>()
                    );
                    let array = CString::new(array).expect("no NUL");
                    let mut result: *const f32 = std::ptr::null();
                    // SAFETY: the interface points at row r's nf f64 values; XGBoost
                    // writes its output pointer to result, valid until the next call;
                    // out has room for n values.
                    unsafe {
                        let rc = f(
                            h,
                            array.as_ptr(),
                            config,
                            proxy,
                            &raw mut shape,
                            &raw mut dim,
                            &raw mut result,
                        );
                        if rc != 0 {
                            return rc;
                        }
                        if result.is_null() {
                            return -1;
                        }
                        *out.add(r) = *result;
                    }
                }
                0
            });
            if rc != 0 {
                self.last = std::ptr::null();
                self.last_len = 0;
                return Err(format!("XGBoost: {}", error_text(self.last_error)));
            }
            // The last call's shape: one output for its one row.
            // SAFETY: XGBoost's shape holds dim values until the next call.
            let len: u64 = if shape.is_null() {
                0
            } else {
                unsafe { std::slice::from_raw_parts(shape, dim as usize) }
                    .iter()
                    .product()
            };
            if len != 1 {
                self.last = std::ptr::null();
                self.last_len = 0;
                return Err(format!("XGBoost returned {len} outputs for one row"));
            }
            self.last = std::hint::black_box(self.loop_out.as_ptr());
            self.last_len = n;
            Ok(ticks)
        }
    }

    impl Drop for XgBoostBench {
        fn drop(&mut self) {
            // SAFETY: both handles were created at load.
            unsafe {
                (self.free_proxy)(self.proxy);
                (self.free_booster)(self.handle);
            }
        }
    }

    impl ExternalMethod for XgBoostBench {
        fn interface(&self, n: usize, multi: bool) -> &'static str {
            if n == 1 && !multi {
                "PredictFromDense"
            } else {
                XGBOOST_CANDIDATES[usize::from(self.row_loop)]
            }
        }
        fn candidates(&self) -> Vec<&'static str> {
            XGBOOST_CANDIDATES.to_vec()
        }
        fn select(&mut self, i: usize) {
            self.row_loop = i == 1;
        }
        fn precision(&self) -> Precision {
            Precision::F32
        }
        fn settings(&self) -> serde_json::Value {
            self.settings.clone()
        }
        fn predict_timed(&mut self, rows: &[f64], n: usize, multi: bool) -> Result<u64, String> {
            if self.row_loop && (n > 1 || multi) {
                return self.predict_loop(rows, n);
            }
            super::check_call(rows, n, self.n_features, usize::MAX)?;
            let (h, proxy, f, config, nf) = (
                self.handle,
                self.proxy,
                self.predict,
                self.config.as_ptr(),
                self.n_features,
            );
            let mut result: *const f32 = std::ptr::null();
            let (mut shape, mut dim): (*const u64, u64) = (std::ptr::null(), 0);
            let (rc, ticks) = timed(|| {
                // The array interface is how in-place prediction consumes rows.
                let array = format!(
                    r#"{{"data":[{},true],"shape":[{n},{nf}],"typestr":"<f8","version":3}}"#,
                    rows.as_ptr() as usize
                );
                let array = CString::new(array).expect("no NUL");
                // SAFETY: the interface points at n × nf f64 values that outlive the
                // call; XGBoost writes its output pointer to result.
                unsafe {
                    f(
                        h,
                        array.as_ptr(),
                        config,
                        proxy,
                        &raw mut shape,
                        &raw mut dim,
                        &raw mut result,
                    )
                }
            });
            if rc != 0 {
                return Err(format!("XGBoost: {}", error_text(self.last_error)));
            }
            self.last = std::hint::black_box(result);
            // One output per row, or the slice in output() would be wrong.
            // SAFETY: XGBoost's shape holds dim values until the next call.
            let len: u64 = if shape.is_null() {
                0
            } else {
                unsafe { std::slice::from_raw_parts(shape, dim as usize) }
                    .iter()
                    .product()
            };
            if result.is_null() || len != n as u64 {
                self.last = std::ptr::null();
                self.last_len = 0;
                return Err(format!("XGBoost returned {len} outputs for {n} rows"));
            }
            self.last_len = n;
            Ok(ticks)
        }
        fn output(&self, n: usize) -> Vec<f64> {
            if self.last.is_null() {
                return Vec::new();
            }
            // SAFETY: XGBoost's last output holds last_len values until the next call.
            unsafe { std::slice::from_raw_parts(self.last, n.min(self.last_len)) }
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        }
    }
}

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
