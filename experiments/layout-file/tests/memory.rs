//! The heap a production load takes at its peak: a counting allocator measures the
//! bytes a call holds at once on the calling thread, so that the other tests'
//! allocations on their threads do not count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write as _;
use std::path::PathBuf;
use treewalker_gbdt::{Forest, LoadOptions, ModelFormat, WalkerConfig};

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

/// The result of `f`, the most heap bytes it held at once on this thread beyond
/// what the thread held before, and those it still holds, in its result.
fn peak<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    BYTES.with(|bytes| bytes.set((0, 0)));
    let result = f();
    let (live, peak) = BYTES.with(Cell::get);
    (result, peak.cast_unsigned(), live.cast_unsigned())
}

/// A file in the temporary directory, removed when dropped.
struct Temp(PathBuf);
impl Temp {
    fn new(name: &str, contents: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("treewalker-memory-{}-{name}", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// LightGBM text from a file is counted first and decoded where its layout keeps
/// its nodes, so the load holds them once: at its peak, little per node beyond the
/// loaded forest. A reader of unknown length decodes them in segments and copies
/// them, which takes at least 16 bytes per node more.
#[test]
fn lightgbm_text_from_a_file_holds_its_nodes_once() {
    // Balanced trees of few distinct splits, so that the nodes dominate.
    const TREES: usize = 128;
    let splits = 511;
    let nodes = TREES * (2 * splits + 1);
    let entries =
        |n, entry: &dyn Fn(usize) -> String| (0..n).map(entry).collect::<Vec<_>>().join(" ");
    let child = |c: usize| {
        if c < splits {
            c.to_string()
        } else {
            (!(c - splits).cast_signed()).to_string()
        }
    };
    let tree = format!(
        "num_leaves={}\nnum_cat=0\nsplit_feature={}\nthreshold={}\ndecision_type={}\n\
         left_child={}\nright_child={}\nleaf_value={}\nleaf_count={}\ninternal_count={}\n",
        splits + 1,
        entries(splits, &|i| (i % 4).to_string()),
        entries(splits, &|i| format!("{}.5", i % 8)),
        entries(splits, &|_| "2".into()),
        entries(splits, &|i| child(2 * i + 1)),
        entries(splits, &|i| child(2 * i + 2)),
        entries(splits + 1, &|i| format!("0.{i}")),
        entries(splits + 1, &|_| "1".into()),
        entries(splits, &|_| "2".into()),
    );
    let mut text = String::from(
        "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nmax_feature_idx=3\n\
         objective=regression\n\n",
    );
    for i in 0..TREES {
        write!(text, "Tree={i}\n{tree}\n\n").unwrap();
    }
    text.push_str("end of trees\n");
    let model = Temp::new("model.txt", &text);
    let config = Temp::new(
        "config.json",
        r#"{"n_features":4,"max_group_width":64,"varying_features":[0,1],"mono_inc_features":[],"mono_dec_features":[]}"#,
    );
    let options = LoadOptions::default();
    let (forest, loaded, held) = peak(|| Forest::load_with(&model.0, &config.0, &options).unwrap());
    drop(forest);
    let walker = WalkerConfig::from_file_with(&config.0, &options).unwrap();
    let file = std::fs::File::open(&model.0).unwrap();
    let (forest, read, _) =
        peak(|| Forest::from_reader(file, ModelFormat::LightGbmText, walker, &options).unwrap());
    drop(forest);
    assert!(
        loaded <= held + 4 * nodes,
        "{loaded} bytes at the peak from a file, {held} held"
    );
    assert!(
        read >= loaded + 16 * nodes,
        "{read} bytes at the peak from a reader, {loaded} from a file"
    );
}
