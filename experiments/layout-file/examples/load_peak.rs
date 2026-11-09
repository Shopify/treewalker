//! The heap and time `Forest::load` takes, to compare a model's formats:
//!
//! ```text
//! cargo run --release --manifest-path experiments/layout-file/Cargo.toml \
//!     --example load_peak -- MODEL CONFIG
//! ```
//!
//! prints the most heap bytes the load held at once, the bytes the loaded forest
//! holds, and the load's seconds. The model's format follows its extension. The
//! text and array caps are raised to 1 GiB, as converting a large model needs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering::Relaxed};
use std::time::Instant;
use treewalker_gbdt::{Forest, LoadOptions};

/// The system allocator, counting the bytes allocated less those freed in
/// [`LIVE`], and their most in [`PEAK`].
struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

fn count(delta: isize) {
    let live = LIVE.fetch_add(delta, Relaxed) + delta;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: each call passes its arguments to `System` unchanged and returns its result;
// counting touches only atomics, which allocate nothing.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size().cast_signed());
        // SAFETY: the caller's contract for `alloc`, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size().cast_signed());
        // SAFETY: the caller's contract for `alloc_zeroed`, passed on.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(-layout.size().cast_signed());
        // SAFETY: the caller's contract for `dealloc`, passed on.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size.cast_signed() - layout.size().cast_signed());
        // SAFETY: the caller's contract for `realloc`, passed on.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, config] = args.as_slice() else {
        eprintln!("usage: load_peak MODEL CONFIG");
        std::process::exit(2);
    };
    let options = LoadOptions {
        max_json_bytes: 1 << 30,
        max_binary_array_bytes: 1 << 30,
        ..LoadOptions::default()
    };
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let start = Instant::now();
    let forest = Forest::load_with(model, config, &options).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    let (peak, held) = (PEAK.load(Relaxed) - base, LIVE.load(Relaxed) - base);
    println!("{peak} {held} {seconds:.4}");
    drop(forest);
}
