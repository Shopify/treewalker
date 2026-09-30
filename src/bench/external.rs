//! FFI wrappers for external prediction methods via `libloading` (dlopen).
//!
//! Each method implements [`ExternalMethod`], providing `predict_group(start, end)`
//! with the same interface so the timing harness can treat them uniformly.

use libloading::{Library, Symbol};
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::path::Path;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A benchmark-able external prediction method.
///
/// Implementations hold a loaded model/library and can predict one group at a time.
/// The `data` slice is the full test dataset (row-major f64); `start..end` are
/// absolute row indices for the current group.
pub trait ExternalMethod {
    /// Predict one group of rows. Results are written somewhere internal
    /// (we don't check correctness in the timing loop — that's a separate validation step).
    fn predict_group(&mut self, data: &[f64], n_cols: usize, start: usize, end: usize);

    /// Human-readable method name for CSV output.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// lleaves
// ---------------------------------------------------------------------------

/// lleaves compiled `.so` — calls `forest_root(data*, results*, start, end)`.
pub struct LleavesBench {
    _lib: Library,
    forest_root: unsafe extern "C" fn(*const c_double, *mut c_double, c_int, c_int),
    results: Vec<f64>,
}

impl LleavesBench {
    /// Load an lleaves `.so` from `so_path`. Returns `None` if the file doesn't exist.
    ///
    /// # Safety
    /// The `.so` must export a valid `forest_root` symbol with the expected signature.
    pub fn load(so_path: &Path, n_rows: usize) -> Option<Self> {
        if !so_path.exists() {
            return None;
        }
        // SAFETY: The .so is a model-specific lleaves compiled library that
        // exports `forest_root` with signature (f64*, f64*, i32, i32) -> void.
        // The caller guarantees so_path points to a valid lleaves .so.
        unsafe {
            let lib = Library::new(so_path)
                .map_err(|e| eprintln!("  lleaves: failed to load {}: {e}", so_path.display()))
                .ok()?;
            let forest_root: Symbol<unsafe extern "C" fn(*const c_double, *mut c_double, c_int, c_int)> =
                lib.get(b"forest_root")
                    .map_err(|e| eprintln!("  lleaves: symbol error: {e}"))
                    .ok()?;
            let forest_root = *forest_root;
            Some(Self {
                _lib: lib,
                forest_root,
                results: vec![0.0; n_rows],
            })
        }
    }
}

impl ExternalMethod for LleavesBench {
    fn predict_group(&mut self, data: &[f64], _n_cols: usize, start: usize, end: usize) {
        // SAFETY: data pointer is valid for [0..n_rows*n_cols], start..end are
        // within bounds, results buffer is pre-allocated to n_rows.
        unsafe {
            (self.forest_root)(
                data.as_ptr(),
                self.results.as_mut_ptr(),
                start as c_int,
                end as c_int,
            );
        }
    }

    fn name(&self) -> &'static str {
        "lleaves"
    }
}

// ---------------------------------------------------------------------------
// tl2cgen
// ---------------------------------------------------------------------------

/// tl2cgen compiled `.so` — calls `predict(Entry*, pred_margin, result*)` per row.
///
/// The `Entry` union has the same size as `f64` for f64-threshold models:
/// `{ fvalue: f64 }` or `{ missing: i64 }` (sentinel = -1).
pub struct Tl2cgenBench {
    _lib: Library,
    predict_fn: unsafe extern "C" fn(*mut Tl2cgenEntry, c_int, *mut c_double),
    n_features: usize,
    entries: Vec<Tl2cgenEntry>,
    result_buf: Vec<f64>,
}

/// tl2cgen Entry union for f64-threshold models.
#[repr(C)]
#[derive(Copy, Clone)]
pub union Tl2cgenEntry {
    pub fvalue: f64,
    pub missing: i64,
}

impl Tl2cgenBench {
    /// Load a tl2cgen `.so`. Returns `None` if the file doesn't exist or symbols are missing.
    ///
    /// # Safety
    /// The `.so` must be a valid tl2cgen compiled model.
    pub fn load(so_path: &Path) -> Option<Self> {
        if !so_path.exists() {
            return None;
        }
        // SAFETY: The .so is a tl2cgen compiled model that exports `predict`,
        // `get_num_feature`, etc. with the signatures documented in tl2cgen's
        // codegen (main_node.cc). Entry is repr(C) matching the generated union.
        unsafe {
            let lib = Library::new(so_path)
                .map_err(|e| eprintln!("  tl2cgen: failed to load {}: {e}", so_path.display()))
                .ok()?;

            let get_num_feature: Symbol<unsafe extern "C" fn() -> c_int> = lib
                .get(b"get_num_feature")
                .map_err(|e| eprintln!("  tl2cgen: symbol error (get_num_feature): {e}"))
                .ok()?;
            let n_features = get_num_feature() as usize;

            let predict_fn: Symbol<unsafe extern "C" fn(*mut Tl2cgenEntry, c_int, *mut c_double)> =
                lib.get(b"predict")
                    .map_err(|e| eprintln!("  tl2cgen: symbol error (predict): {e}"))
                    .ok()?;
            let predict_fn = *predict_fn;

            let entries = vec![Tl2cgenEntry { missing: -1 }; n_features];
            eprintln!("  tl2cgen: loaded ({n_features} features)");

            Some(Self {
                _lib: lib,
                predict_fn,
                n_features,
                entries,
                result_buf: vec![0.0; 1],
            })
        }
    }
}

impl ExternalMethod for Tl2cgenBench {
    fn predict_group(&mut self, data: &[f64], n_cols: usize, start: usize, end: usize) {
        // tl2cgen's predict() is per-row: we pay per-row FFI call overhead.
        // This is the real cost of using tl2cgen in a serving scenario.
        for row_idx in start..end {
            // Populate Entry array from test data.
            let row_offset = row_idx * n_cols;
            for col in 0..self.n_features {
                let val = data[row_offset + col];
                if val.is_nan() {
                    self.entries[col] = Tl2cgenEntry { missing: -1 };
                } else {
                    self.entries[col] = Tl2cgenEntry { fvalue: val };
                }
            }
            unsafe {
                (self.predict_fn)(
                    self.entries.as_mut_ptr(),
                    0, // pred_margin = 0 (apply postprocessing)
                    self.result_buf.as_mut_ptr(),
                );
            }
        }
    }

    fn name(&self) -> &'static str {
        "tl2cgen"
    }
}

// ---------------------------------------------------------------------------
// LightGBM native
// ---------------------------------------------------------------------------

/// Opaque handle type for LightGBM C API.
type BoosterHandle = *mut c_void;

/// Function pointer type for `LGBM_BoosterPredictForMat`.
type LgbmPredictFn = unsafe extern "C" fn(
    BoosterHandle,
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

/// LightGBM native prediction via C API.
pub struct LightGBMBench {
    _lib: Library,
    handle: BoosterHandle,
    predict_fn: LgbmPredictFn,
    free_fn: unsafe extern "C" fn(BoosterHandle) -> c_int,
    n_cols: usize,
    param_cstr: CString,
    result_buf: Vec<f64>,
}

impl LightGBMBench {
    /// Load LightGBM from a shared library and a model file.
    ///
    /// `max_group_width` controls the result buffer size (one prediction per row).
    ///
    /// # Safety
    /// `lib_path` must be a valid LightGBM shared library.
    pub fn load(lib_path: &Path, model_path: &Path, n_cols: usize, max_group_width: usize) -> Option<Self> {
        if !lib_path.exists() || !model_path.exists() {
            return None;
        }
        // SAFETY: lib_path must be a valid LightGBM shared library exporting
        // the C API symbols (LGBM_BoosterCreateFromModelfile, etc.) with the
        // signatures documented at https://lightgbm.readthedocs.io/en/latest/C-API.html
        unsafe {
            let lib = Library::new(lib_path)
                .map_err(|e| eprintln!("  lightgbm: failed to load {}: {e}", lib_path.display()))
                .ok()?;

            // Resolve symbols.
            let create_fn: Symbol<
                unsafe extern "C" fn(*const c_char, *mut c_int, *mut BoosterHandle) -> c_int,
            > = lib
                .get(b"LGBM_BoosterCreateFromModelfile")
                .map_err(|e| eprintln!("  lightgbm: symbol error: {e}"))
                .ok()?;
            let predict_fn: Symbol<LgbmPredictFn> = lib
                .get(b"LGBM_BoosterPredictForMat")
                .map_err(|e| eprintln!("  lightgbm: symbol error: {e}"))
                .ok()?;
            let free_fn: Symbol<unsafe extern "C" fn(BoosterHandle) -> c_int> = lib
                .get(b"LGBM_BoosterFree")
                .map_err(|e| eprintln!("  lightgbm: symbol error: {e}"))
                .ok()?;

            // Load model.
            let model_cstr = CString::new(model_path.to_str().unwrap()).unwrap();
            let mut num_iters: c_int = 0;
            let mut handle: BoosterHandle = std::ptr::null_mut();
            let rc = create_fn(model_cstr.as_ptr(), &raw mut num_iters, &raw mut handle);
            if rc != 0 || handle.is_null() {
                eprintln!("  lightgbm: failed to load model (rc={rc})");
                return None;
            }
            eprintln!("  lightgbm: loaded ({num_iters} iterations)");

            let predict_fn = *predict_fn;
            let free_fn = *free_fn;
            let param_cstr = CString::new("num_threads=1").unwrap();
            // Allocate result buffer for the largest group (one prediction per row).
            let result_buf = vec![0.0; max_group_width];

            Some(Self {
                _lib: lib,
                handle,
                predict_fn,
                free_fn,
                n_cols,
                param_cstr,
                result_buf,
            })
        }
    }
}

impl ExternalMethod for LightGBMBench {
    fn predict_group(&mut self, data: &[f64], n_cols: usize, start: usize, end: usize) {
        let nrow = (end - start) as i32;
        // SAFETY: data[start*n_cols..end*n_cols] is within bounds (checked by
        // the caller). We pass a pointer to the group's first row and nrow.
        // result_buf is pre-allocated to max_group_width >= nrow.
        let data_ptr = unsafe { data.as_ptr().add(start * n_cols) };
        let mut out_len: i64 = 0;
        unsafe {
            (self.predict_fn)(
                self.handle,
                data_ptr.cast::<c_void>(),
                1,    // C_API_DTYPE_FLOAT64
                nrow, // nrow
                self.n_cols as i32,
                1, // is_row_major
                0, // C_API_PREDICT_NORMAL
                0, // start_iteration
                0, // num_iteration (all)
                self.param_cstr.as_ptr(),
                &raw mut out_len,
                self.result_buf.as_mut_ptr(),
            );
        }
    }

    fn name(&self) -> &'static str {
        "lightgbm_native"
    }
}

impl Drop for LightGBMBench {
    fn drop(&mut self) {
        unsafe {
            (self.free_fn)(self.handle);
        }
    }
}

// ---------------------------------------------------------------------------
// XGBoost native
// ---------------------------------------------------------------------------

/// Opaque handle types for XGBoost C API.
type XGBoosterHandle = *mut c_void;
type DMatrixHandle = *mut c_void;

/// XGBoost native prediction via C API using `XGBoosterPredictFromDense`
/// (inplace predict — no DMatrix construction per group).
pub struct XGBoostBench {
    _lib: Library,
    handle: XGBoosterHandle,
    proxy: DMatrixHandle,
    // Inplace prediction from dense array
    predict_from_dense_fn: unsafe extern "C" fn(
        XGBoosterHandle,
        *const c_char, // JSON array interface
        *const c_char, // JSON config
        DMatrixHandle, // proxy
        *mut *const u64, // out_shape
        *mut u64,        // out_dim
        *mut *const f32, // out_result
    ) -> c_int,
    free_proxy_fn: unsafe extern "C" fn(DMatrixHandle) -> c_int,
    free_booster_fn: unsafe extern "C" fn(XGBoosterHandle) -> c_int,
    n_cols: usize,
    // Pre-allocated f32 conversion buffer (max 128 rows × n_cols).
    f32_buf: Vec<f32>,
    // Pre-allocated JSON config string (reused across calls).
    predict_config: CString,
}

impl XGBoostBench {
    /// Load XGBoost from a shared library and a model file.
    ///
    /// # Safety
    /// `lib_path` must be a valid XGBoost shared library (>= 2.0 for inplace predict).
    pub fn load(lib_path: &Path, model_path: &Path, n_cols: usize) -> Option<Self> {
        if !lib_path.exists() || !model_path.exists() {
            return None;
        }
        // SAFETY: lib_path must be a valid XGBoost shared library exporting
        // the C API symbols (XGBoosterCreate, XGBoosterPredictFromDense, etc.)
        // with signatures per https://xgboost.readthedocs.io/en/stable/c.html
        unsafe {
            let lib = Library::new(lib_path)
                .map_err(|e| eprintln!("  xgboost: failed to load {}: {e}", lib_path.display()))
                .ok()?;

            // Resolve symbols.
            let booster_create: Symbol<
                unsafe extern "C" fn(*const DMatrixHandle, u64, *mut XGBoosterHandle) -> c_int,
            > = lib.get(b"XGBoosterCreate").ok()?;
            let booster_load: Symbol<
                unsafe extern "C" fn(XGBoosterHandle, *const c_char) -> c_int,
            > = lib.get(b"XGBoosterLoadModel").ok()?;
            let booster_set_param: Symbol<
                unsafe extern "C" fn(XGBoosterHandle, *const c_char, *const c_char) -> c_int,
            > = lib.get(b"XGBoosterSetParam").ok()?;
            let proxy_create: Symbol<
                unsafe extern "C" fn(*mut DMatrixHandle) -> c_int,
            > = lib.get(b"XGProxyDMatrixCreate").ok()?;
            let predict_from_dense_fn: Symbol<
                unsafe extern "C" fn(
                    XGBoosterHandle,
                    *const c_char, *const c_char, DMatrixHandle,
                    *mut *const u64, *mut u64, *mut *const f32,
                ) -> c_int,
            > = lib.get(b"XGBoosterPredictFromDense").ok()?;
            let free_proxy_fn: Symbol<unsafe extern "C" fn(DMatrixHandle) -> c_int> =
                lib.get(b"XGDMatrixFree").ok()?;
            let free_booster_fn: Symbol<unsafe extern "C" fn(XGBoosterHandle) -> c_int> =
                lib.get(b"XGBoosterFree").ok()?;

            // Create booster.
            let mut handle: XGBoosterHandle = std::ptr::null_mut();
            let rc = booster_create(std::ptr::null(), 0, &raw mut handle);
            if rc != 0 || handle.is_null() {
                eprintln!("  xgboost: XGBoosterCreate failed (rc={rc})");
                return None;
            }

            // Load model.
            let model_cstr = CString::new(model_path.to_str().unwrap()).unwrap();
            let rc = booster_load(handle, model_cstr.as_ptr());
            if rc != 0 {
                eprintln!("  xgboost: XGBoosterLoadModel failed (rc={rc})");
                return None;
            }

            // Set nthread=1.
            let key = CString::new("nthread").unwrap();
            let val = CString::new("1").unwrap();
            booster_set_param(handle, key.as_ptr(), val.as_ptr());

            // Create proxy DMatrix (reused across all predict calls).
            let mut proxy: DMatrixHandle = std::ptr::null_mut();
            let rc = proxy_create(&raw mut proxy);
            if rc != 0 || proxy.is_null() {
                eprintln!("  xgboost: XGProxyDMatrixCreate failed (rc={rc})");
                return None;
            }

            eprintln!("  xgboost: loaded (inplace predict via XGBoosterPredictFromDense)");

            let predict_from_dense_fn = *predict_from_dense_fn;
            let free_proxy_fn = *free_proxy_fn;
            let free_booster_fn = *free_booster_fn;

            // Pre-allocate f32 buffer for max group size (128 rows).
            let f32_buf = vec![0.0f32; 128 * n_cols];
            // Prediction config: normal prediction, no iteration limit.
            let predict_config = CString::new(
                r#"{"type":0,"training":false,"iteration_range":[0,0],"strict_shape":false}"#
            ).unwrap();

            Some(Self {
                _lib: lib,
                handle,
                proxy,
                predict_from_dense_fn,
                free_proxy_fn,
                free_booster_fn,
                n_cols,
                f32_buf,
                predict_config,
            })
        }
    }
}

impl ExternalMethod for XGBoostBench {
    fn predict_group(&mut self, data: &[f64], n_cols: usize, start: usize, end: usize) {
        let nrow = end - start;
        let n_elems = nrow * n_cols;

        // Convert f64 → f32 (XGBoost uses f32 internally).
        let src = &data[start * n_cols..start * n_cols + n_elems];
        for (dst, &s) in self.f32_buf[..n_elems].iter_mut().zip(src) {
            *dst = s as f32;
        }

        // Build JSON array interface pointing to f32_buf.
        // This is the same mechanism Python's inplace_predict uses under the hood.
        let array_json = format!(
            r#"{{"data":[{},false],"shape":[{},{}],"typestr":"<f4","version":3}}"#,
            self.f32_buf.as_ptr() as usize, nrow, n_cols,
        );
        let array_cstr = CString::new(array_json).unwrap();

        // SAFETY: f32_buf is pre-allocated and populated with valid f32 data.
        // The JSON array interface points to f32_buf's memory, which remains
        // valid for the duration of the predict call. The proxy DMatrix is
        // created once at load time and reused. XGBoost writes results to an
        // internal buffer returned via out_result (valid until next predict call).
        unsafe {
            let mut out_shape: *const u64 = std::ptr::null();
            let mut out_dim: u64 = 0;
            let mut out_result: *const f32 = std::ptr::null();
            (self.predict_from_dense_fn)(
                self.handle,
                array_cstr.as_ptr(),
                self.predict_config.as_ptr(),
                self.proxy,
                &raw mut out_shape,
                &raw mut out_dim,
                &raw mut out_result,
            );
        }
    }

    fn name(&self) -> &'static str {
        "xgboost_native"
    }
}

impl Drop for XGBoostBench {
    fn drop(&mut self) {
        unsafe {
            (self.free_proxy_fn)(self.proxy);
            (self.free_booster_fn)(self.handle);
        }
    }
}

// ---------------------------------------------------------------------------
// QuickScorer
// ---------------------------------------------------------------------------

/// QuickScorer baseline — per-row `score_fast` within each group.
#[cfg(feature = "external-bench")]
pub struct QuickScorerBench {
    qs: quickscorer::QuickScorer,
    data_f32: Vec<f32>,
    n_cols: usize,
}

#[cfg(feature = "external-bench")]
impl QuickScorerBench {
    /// Load a QuickScorer model from a treelite JSON file.
    ///
    /// Returns `None` if the model has >128 leaves per tree (QuickScorer limitation)
    /// or if loading fails for any other reason.
    pub fn load(model_path: &Path, data: &[f64], n_cols: usize) -> Option<Self> {
        if !model_path.exists() {
            return None;
        }
        let result = std::panic::catch_unwind(|| {
            quickscorer::QuickScorer::from_model_file(model_path)
        });
        match result {
            Ok(Ok(qs)) => {
                // Pre-convert all data to f32 (QuickScorer uses f32).
                let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                eprintln!("  quickscorer: loaded");
                Some(Self { qs, data_f32, n_cols })
            }
            Ok(Err(e)) => {
                eprintln!("  quickscorer: load error: {e}");
                None
            }
            Err(_) => {
                eprintln!("  quickscorer: panicked during load (likely >128 leaves per tree)");
                None
            }
        }
    }
}

#[cfg(feature = "external-bench")]
impl ExternalMethod for QuickScorerBench {
    fn predict_group(&mut self, _data: &[f64], _n_cols: usize, start: usize, end: usize) {
        // Per-row scoring within the group, using pre-converted f32 data.
        for row_idx in start..end {
            let row = &self.data_f32[row_idx * self.n_cols..(row_idx + 1) * self.n_cols];
            let _ = self.qs.score_fast(row);
        }
    }

    fn name(&self) -> &'static str {
        "quickscorer"
    }
}
