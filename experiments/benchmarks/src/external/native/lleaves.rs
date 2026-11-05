//! lleaves: the model compiled by LLVM into one `forest_root` function.

use std::os::raw::{c_double, c_int};
use std::path::Path;

use libloading::Library;

use super::{dl_error, sym};
use crate::external::{ExternalMethod, Precision, check_call, timed};

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
        check_call(rows, n, self.n_features, self.out.len())?;
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
