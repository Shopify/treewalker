//! XGBoost's C API, from the library `build-native` builds.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;

use libloading::Library;

use super::tl2cgen::Handle;
use super::{cstr, dl_error, error_text, sym};
use crate::external::{ExternalMethod, Precision, check_call, timed};

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
            let proxy_create =
                sym::<unsafe extern "C" fn(*mut Handle) -> c_int>(&lib, c"XGProxyDMatrixCreate")?;
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
        check_call(rows, n, nf, self.loop_out.len())?;
        let (h, proxy, f, config) = (self.handle, self.proxy, self.predict, self.config.as_ptr());
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
        check_call(rows, n, self.n_features, usize::MAX)?;
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
