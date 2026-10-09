//! The timed methods of a cell: TreeWalker's builds and the baseline adapters.
//!
//! Each call reads the counter around exactly the method's own call; the output is
//! consumed through `black_box` after the second read, and kept for validation.

#![expect(
    clippy::inline_always,
    reason = "the counter reads must sit right around each call"
)]

use treewalker_gbdt::research::ResearchPredictor;
use treewalker_gbdt::{Forest, Predictor};

use crate::artifacts::VariantSpec;
use crate::external::{ExternalMethod, Precision};
use crate::timer;

/// TreeWalker's row-count threshold for the old caller-side workaround.
pub const CHUNK_ROWS: usize = 128;

/// What runs a method's calls.
pub enum Engine {
    /// Production `predict`: `predict_group`, or `predict_groups` in batch mode.
    Production(Predictor),
    /// The research timed build with one variant's runtime flags.
    Research(ResearchPredictor),
    /// The research per-row reference walk.
    FullWalk(Forest),
    /// Production `predict` on a model loaded at width 128, each group split into
    /// pieces of at most 128 rows by the caller: `treewalker_chunked128`.
    Chunked(Predictor),
    External(Box<dyn ExternalMethod>),
}

/// One method (and variant) of a cell.
pub struct TimedMethod {
    /// The `method` column.
    pub method: String,
    /// The `variant` column: `all-on`, a variant ID, or `-` for baselines.
    pub variant: String,
    pub engine: Engine,
    /// The candidate interfaces that passed validation, by index into
    /// [`Self::candidates`]; probing chooses among them.
    pub allowed: Vec<usize>,
    /// The multi-row interface each mode timed, by mode name.
    pub chosen: std::collections::BTreeMap<String, &'static str>,
    /// A research variant's spec; `None` for every other method.
    pub spec: Option<VariantSpec>,
    out: Vec<f64>,
    /// Chunk offsets of the current batch call, for `Chunked`.
    chunks: Vec<usize>,
}

#[inline(always)]
fn timed(f: impl FnOnce()) -> u64 {
    let t0 = timer::start();
    f();
    timer::stop().wrapping_sub(t0)
}

impl TimedMethod {
    pub fn new(method: &str, variant: &str, engine: Engine, max_rows: usize) -> Self {
        let mut m = Self {
            method: method.to_string(),
            variant: variant.to_string(),
            engine,
            allowed: Vec::new(),
            chosen: std::collections::BTreeMap::new(),
            spec: None,
            out: vec![0.0; max_rows],
            chunks: Vec::new(),
        };
        m.allowed = (0..m.candidates(false).len()).collect();
        m
    }

    /// The interfaces a multi-row call can use: a baseline's multi-row call and,
    /// where it has a single-row interface, a loop over it. TreeWalker has one.
    pub fn candidates(&self, multi: bool) -> Vec<&'static str> {
        match &self.engine {
            Engine::External(e) => e.candidates(),
            _ => vec![self.interface(2, multi)],
        }
    }

    /// Use candidate `i` for multi-row calls.
    pub fn select(&mut self, i: usize) {
        if let Engine::External(e) = &mut self.engine {
            e.select(i);
        }
    }

    pub const fn is_treewalker(&self) -> bool {
        !matches!(self.engine, Engine::External(_))
    }

    /// The output precision, for the validation tolerance.
    pub fn precision(&self) -> Precision {
        match &self.engine {
            Engine::External(e) => e.precision(),
            _ => Precision::F64,
        }
    }

    /// A baseline's effective settings after load (`Null` for TreeWalker).
    pub fn settings(&self) -> serde_json::Value {
        match &self.engine {
            Engine::External(e) => e.settings(),
            _ => serde_json::Value::Null,
        }
    }

    /// The interface a call uses: `multi` is batch mode.
    pub fn interface(&self, n_rows: usize, multi: bool) -> &'static str {
        match &self.engine {
            Engine::Production(_) | Engine::Research(_) => {
                if multi {
                    "predict_groups"
                } else {
                    "predict_group"
                }
            }
            Engine::FullWalk(_) => "predict_full_walk",
            Engine::Chunked(_) => {
                if multi {
                    "predict_groups (chunks of 128)"
                } else {
                    "predict_fixed (128)"
                }
            }
            Engine::External(e) => e.interface(n_rows, multi),
        }
    }

    /// Time one group, as a serving call.
    pub fn serve(&mut self, rows: &[f64], n_rows: usize) -> Result<u64, String> {
        if n_rows > self.out.len() {
            return Err(format!("{n_rows} rows for an output of {}", self.out.len()));
        }
        // Opaque inputs: the call cannot be hoisted above the first counter read.
        let rows = std::hint::black_box(rows);
        let out = std::hint::black_box(&mut self.out[..n_rows]);
        let ticks = match &mut self.engine {
            Engine::Production(p) => timed(|| p.predict_group(rows, out)),
            Engine::Research(r) => timed(|| r.predict_group(rows, out)),
            Engine::FullWalk(f) => timed(|| f.predict_full_walk(rows, out)),
            Engine::Chunked(p) => timed(|| p.predict_fixed(rows, CHUNK_ROWS, out)),
            Engine::External(e) => return e.predict_timed(rows, n_rows, false),
        };
        std::hint::black_box(&self.out[..n_rows]);
        Ok(ticks)
    }

    /// Time one batch call over consecutive groups: `offsets` are local, from 0 to
    /// the call's row count.
    pub fn batch(&mut self, rows: &[f64], offsets: &[usize]) -> Result<u64, String> {
        let n_rows = *offsets.last().expect("offsets end at the row count");
        if n_rows > self.out.len() {
            return Err(format!("{n_rows} rows for an output of {}", self.out.len()));
        }
        let (rows, offsets) = std::hint::black_box((rows, offsets));
        if let Engine::Chunked(_) = self.engine {
            // The caller splits groups into pieces of at most 128 rows, untimed.
            self.chunks.clear();
            self.chunks.push(0);
            for w in offsets.windows(2) {
                let mut s = w[0];
                while s < w[1] {
                    s = (s + CHUNK_ROWS).min(w[1]);
                    self.chunks.push(s);
                }
            }
        }
        let out = std::hint::black_box(&mut self.out[..n_rows]);
        let ticks = match &mut self.engine {
            Engine::Production(p) => timed(|| p.predict_groups(rows, offsets, out)),
            Engine::Research(r) => timed(|| r.predict_groups(rows, offsets, out)),
            Engine::FullWalk(f) => timed(|| f.predict_full_walk(rows, out)),
            Engine::Chunked(p) => {
                let chunks = &self.chunks;
                timed(|| p.predict_groups(rows, chunks, out))
            }
            Engine::External(e) => return e.predict_timed(rows, n_rows, true),
        };
        std::hint::black_box(&self.out[..n_rows]);
        Ok(ticks)
    }

    /// Only the input conversion of a call, for the adapters that convert outside
    /// the library (QuickScorer); false for the others.
    pub fn convert(&mut self, rows: &[f64], n_rows: usize, multi: bool) -> bool {
        match &mut self.engine {
            Engine::External(e) => e.convert_only(rows, n_rows, multi),
            _ => false,
        }
    }

    /// A known library defect: its name and the rows it affects.
    pub fn known_defect(&self) -> Option<crate::external::KnownDefect> {
        match &self.engine {
            Engine::External(e) => e.known_defect(),
            _ => None,
        }
    }

    /// The last call's output.
    pub fn output(&self, n_rows: usize) -> Vec<f64> {
        match &self.engine {
            Engine::External(e) => e.output(n_rows),
            _ => self.out[..n_rows].to_vec(),
        }
    }
}
