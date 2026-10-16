//! The baselines behind C libraries, loaded with `libloading`: lleaves, tl2cgen,
//! LightGBM and XGBoost.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;

use libloading::Library;

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
        let lib =
            Library::new(path).map_err(|e| dl_error(&format!("loading {}", path.display()), &e))?;
        for name in symbols {
            lib.get::<*const c_void>(name.to_bytes_with_nul())
                .map_err(|e| dl_error(&format!("symbol {}", name.to_string_lossy()), &e))?;
        }
    }
    Ok(())
}

mod lightgbm;
mod lleaves;
mod tl2cgen;
mod xgboost;

pub use lightgbm::LightGbmBench;
pub use lleaves::LleavesBench;
pub use tl2cgen::Tl2cgenBench;
pub use xgboost::XgBoostBench;
