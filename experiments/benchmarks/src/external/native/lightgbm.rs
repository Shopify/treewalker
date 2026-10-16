//! LightGBM's C API, from the library `build-native` builds.

use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::path::Path;

use libloading::Library;

use super::{cstr, dl_error, error_text, sym};
use crate::external::{ExternalMethod, Precision, check_call, timed};

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
        check_call(rows, n, nf, self.out.len())?;
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
