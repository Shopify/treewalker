//! Sets `cfg(dispatch)` where the kernel is compiled at more levels than the build's
//! own: on x86-64 without AVX2 and FMA (`src/predict/isa.rs`).

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(dispatch)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has = |feature: &str| features.split(',').any(|f| f == feature);
    if arch == "x86_64" && !(has("avx2") && has("fma")) {
        println!("cargo::rustc-cfg=dispatch");
    }
}
