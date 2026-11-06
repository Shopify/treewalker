//! Sets `cfg(dispatch)` where the kernel is compiled at more levels than the build's
//! own: on x86-64 without AVX2 and FMA (`src/predict/isa.rs`). With `layout-file`, sets
//! `TREEWALKER_BUILD_ID`, a hash of the crate's sources that layout files record.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_LAYOUT_FILE").is_some() {
        println!("cargo::rerun-if-changed=src");
        println!("cargo::rustc-env=TREEWALKER_BUILD_ID={:016x}", build_id());
    }
    println!("cargo::rustc-check-cfg=cfg(dispatch)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has = |feature: &str| features.split(',').any(|f| f == feature);
    if arch == "x86_64" && !(has("avx2") && has("fma")) {
        println!("cargo::rustc-cfg=dispatch");
    }
}

fn build_id() -> u64 {
    use std::path::{Path, PathBuf};
    fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("read source directory") {
            let entry = entry.expect("read source entry");
            if entry.file_type().expect("source entry type").is_dir() {
                collect(&entry.path(), files);
            } else {
                files.push(entry.path());
            }
        }
    }
    // Same streaming multiply-rotate checksum as layout_file::codec::Checksum.
    // Retain partial words across path/content boundaries, just as the writer does.
    let (mut state, mut word, mut used) = (0u64, 0u64, 0u32);
    let mut feed = |bytes: &[u8]| {
        for &byte in bytes {
            word |= u64::from(byte) << (8 * used);
            used += 1;
            if used == 8 {
                state = (state.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
                word = 0;
                used = 0;
            }
        }
    };
    let mut files = vec![PathBuf::from("build.rs")];
    collect(Path::new("src"), &mut files);
    files.sort_unstable();
    for path in files {
        let relative = path.to_str().expect("UTF-8 source path").replace('\\', "/");
        feed(relative.as_bytes());
        feed(&std::fs::read(&path).expect("read source file"));
    }
    state = (state.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    (state.rotate_left(5) ^ u64::from(used)).wrapping_mul(0x517c_c1b7_2722_0a95)
}
