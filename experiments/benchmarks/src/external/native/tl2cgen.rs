//! tl2cgen: the model compiled to C, with tl2cgen's runtime for multi-row calls.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

use libloading::Library;

use super::{GetStr, cstr, dl_error, error_text, sym};
use crate::external::{ExternalMethod, Precision, check_call, timed};

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
pub(super) type Handle = *mut c_void;

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
/// [`crate::external::tl2cgen_f64_missing_alias`]. The adapter passes values as tl2cgen's
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
            let leaf = CStr::from_ptr(sym::<GetStr>(&lib, c"get_leaf_output_type")?()).to_owned();
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
            row.iter()
                .any(|&v| crate::external::tl2cgen_f64_missing_alias(v))
        }
        self.predict64.is_some().then_some((
            crate::external::TL2CGEN_F64_MISSING_ALIAS,
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
        check_call(rows, n, self.n_features, self.out64.len())?;
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
