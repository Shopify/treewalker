//! The memory a small input, or an input past a small limit, can make the loader
//! take: a counting allocator measures the heap bytes a call holds at its peak, on
//! the calling thread, so that the other tests' allocations on their threads do not
//! count.
#![expect(clippy::float_cmp, reason = "the generator's exact category test")]

mod common;

use common::model::{Model, Node, Op, Precision, Task, Tree};
use common::rng::Rng;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use treewalker_gbdt::{Forest, LoadError, LoadOptions, ModelFormat, WalkerConfig};

/// The system allocator, counting the calling thread's bytes in [`BYTES`].
struct Counting;

thread_local! {
    /// The thread's heap bytes allocated less those freed since [`peak`] began, and
    /// their peak.
    static BYTES: Cell<(isize, isize)> = const { Cell::new((0, 0)) };
}

/// Count `delta` bytes on this thread; nothing once its locals are gone, at exit.
fn count(delta: isize) {
    let _ = BYTES.try_with(|bytes| {
        let (live, peak) = bytes.get();
        bytes.set((live + delta, peak.max(live + delta)));
    });
}

// SAFETY: each call passes its arguments to `System` unchanged and returns its result;
// counting touches only a thread-local `Cell`, which allocates nothing.
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

/// The result of `f`, and the most heap bytes it held at once on this thread beyond
/// what the thread held before.
fn peak<T>(f: impl FnOnce() -> T) -> (T, usize) {
    BYTES.with(|bytes| bytes.set((0, 0)));
    let result = f();
    let (_, peak) = BYTES.with(Cell::get);
    (result, peak.cast_unsigned())
}

/// A configuration takes memory by the features it declares, not by their largest
/// index: a bitset up to it would take 256 MiB for this 120-byte file, and the
/// unions of the roles would clone it twice, about 768 MiB at once.
#[test]
fn configurations_take_memory_by_their_declarations() {
    let json = br#"{"n_features":2147483647,"max_group_width":1,"varying_features":[2147483646],"mono_inc_features":[],"mono_dec_features":[]}"#;
    let (config, bytes) = peak(|| WalkerConfig::from_json(json).unwrap());
    assert!(config.is_varying(2_147_483_646) && !config.is_varying(2_147_483_645));
    assert!(bytes < 64 << 10, "{bytes} bytes at the peak");
    let last = i32::MAX as usize - 1;
    let (config, bytes) = peak(|| {
        WalkerConfig::builder(last + 1)
            .max_group_width(1)
            .varying([last])
            .increasing([last - 1])
            .decreasing([0])
            .build()
            .unwrap()
    });
    assert!(config.is_varying(last) && config.is_increasing(last - 1));
    assert!(config.is_varying(0) && config.is_decreasing(0) && !config.is_varying(1));
    assert!(bytes < 64 << 10, "{bytes} bytes at the peak");
}

/// A binary model longer than `LoadOptions::max_binary_bytes` fails at the limit
/// without reserving its nodes. The guess of the node count read a tree that began
/// below the limit to the input's end, and the decoder reserved 16 bytes a node
/// before it read past the limit: 4 MiB for this tree of 262,143 nodes under a
/// 4 KiB limit.
#[test]
fn binary_models_past_the_byte_limit_reserve_no_nodes() {
    let internal = (1 << 17) - 1;
    let nodes = (0..=2 * internal)
        .map(|i| {
            if i < internal {
                Node::Numeric {
                    feature: 0,
                    op: Op::Le,
                    threshold: i as f64,
                    default_left: true,
                    left: 2 * i + 1,
                    right: 2 * i + 2,
                    count: Some(1),
                }
            } else {
                Node::Leaf {
                    value: 1.0,
                    count: Some(1),
                }
            }
        })
        .collect();
    let model = Model {
        precision: Precision::F64,
        n_features: 1,
        trees: vec![Tree { nodes }],
        average: false,
        base_score: 0.0,
        sigmoid: None,
        task: Task::Regressor,
    };
    let binary = model.to_binary(&mut Rng::new(1));
    let config = WalkerConfig::builder(1)
        .max_group_width(1)
        .varying([0])
        .build()
        .unwrap();
    let options = LoadOptions {
        max_binary_bytes: 4096,
        ..LoadOptions::default()
    };
    let (result, bytes) =
        peak(|| Forest::from_bytes(&binary, ModelFormat::TreeliteBinaryV4, config, &options));
    assert!(matches!(result, Err(LoadError::Limit(_))));
    assert!(bytes < 64 << 10, "{bytes} bytes at the peak");
}
