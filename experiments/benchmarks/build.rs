//! Records the compiler and flags the runner was built with, and the source it
//! was built from, for `run.json`.

use std::path::Path;
use std::process::Command;

/// What the executable compiles, relative to the repository: the library, this
/// package, and the files that set how they build.
const COMPILED: [&str; 8] = [
    "src",
    "Cargo.toml",
    "Cargo.lock",
    ".cargo",
    "rust-toolchain.toml",
    "experiments/benchmarks/src",
    "experiments/benchmarks/build.rs",
    "experiments/benchmarks/Cargo.toml",
];

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(rustc)
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
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let repo = Path::new(&manifest_dir).join("../..");
    let (commit, dirty, from) = source(&repo);
    println!("cargo:rustc-env=TW_SOURCE_COMMIT={commit}");
    println!("cargo:rustc-env=TW_SOURCE_DIRTY={dirty}");
    println!("cargo:rustc-env=TW_SOURCE_FROM={from}");
}

/// The commit of the tree being built, whether what it compiles differs from
/// that commit, and where the commit came from: a git archive's
/// `experiments/SOURCE_COMMIT` (export-subst), else git. The commit is empty when
/// neither knows it.
fn source(repo: &Path) -> (String, bool, &'static str) {
    let stamp = repo.join("experiments/SOURCE_COMMIT");
    rerun_if_changed(&stamp);
    if let Ok(s) = std::fs::read_to_string(&stamp) {
        let s = s.trim();
        if !s.starts_with("$Format") && !s.is_empty() {
            return (
                s.to_string(),
                false,
                "experiments/SOURCE_COMMIT (git archive)",
            );
        }
    }
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(repo)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let Some(commit) = git(&["rev-parse", "HEAD"]) else {
        return (String::new(), false, "git");
    };
    // Build again on a new commit, a checkout, or a change to what is compiled. Not on
    // the index: any `git status` rewrites it, this script's own included, so every
    // build in another target directory would rebuild this one; staging a change
    // leaves what is compiled as it was, and the files themselves are watched below.
    let mut watched = vec!["HEAD".to_string(), "packed-refs".to_string()];
    watched.extend(git(&["symbolic-ref", "-q", "HEAD"]));
    for name in &watched {
        if let Some(p) = git(&["rev-parse", "--path-format=absolute", "--git-path", name]) {
            rerun_if_changed(Path::new(&p));
        }
    }
    for p in COMPILED {
        rerun_if_changed(&repo.join(p));
    }
    let mut status = vec!["status", "--porcelain", "--"];
    status.extend(COMPILED);
    let dirty = git(&status).is_some_and(|s| !s.is_empty());
    (commit, dirty, "git")
}

/// A path that does not exist would make Cargo run the script on every build.
fn rerun_if_changed(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}
