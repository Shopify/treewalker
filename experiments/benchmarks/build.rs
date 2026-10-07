//! Records the compiler and flags the runner was built with, for `run.json`.

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    println!("cargo:rustc-env=TW_RUSTC_VERSION={}", version.trim());
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .replace('\u{1f}', " ");
    println!("cargo:rustc-env=TW_RUSTFLAGS={flags}");
    let profile = std::env::var("PROFILE").unwrap_or_default();
    println!("cargo:rustc-env=TW_PROFILE={profile}");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
}
