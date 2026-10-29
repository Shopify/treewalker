#!/usr/bin/env bash
# The development checks, as concurrent lanes. `fast` is every commit's: formatting,
# clippy, the debug tests and the Python checks. `full` adds the release, research and
# bench tests, the property tests at their full counts, cross clippy, Miri, the spill
# checks and the bench crate on the core artifacts (TEST_ARTIFACTS_BASE, as its tests
# read it).
#
# Usage: experiments/scripts/checks.sh [fast|full], from the repository root. Each lane
# builds in its own target directory under target/checks, since cargo locks one; each
# step's log goes to target/checks/logs, and the summary lists every step's status,
# test counts and seconds. Exits 1 if any step failed. The cross checks compile C with
# clang and the macOS SDK's headers, as the README describes; Miri needs a nightly
# toolchain with the miri component; the Python checks take PYTHON (default python3)
# with the project's dependencies.
set -u
MODE=${1:-fast}
case $MODE in fast | full) ;; *) echo "usage: $0 [fast|full]" >&2; exit 2 ;; esac
ROOT=$(git rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2
OUT=$ROOT/target/checks
L=$OUT/logs
PY=${PYTHON:-python3}
rm -rf "$L"; mkdir -p "$L/res"
export PYTHONPATH=$ROOT/experiments
T0=$(date +%s)

# step N NAME CMD...: run CMD into its log; record its status, test counts and seconds.
step() {
  local n=$1 name="$2" f t0 rc t r py; shift 2
  f="$L/$(printf %02d "$n")-$(echo "$name" | tr ' /=,' '____').log"
  t0=$(date +%s)
  "$@" > "$f" 2>&1
  rc=$? t=$(( $(date +%s) - t0 ))
  r=$(grep -E '^test result' "$f" | awk '{p+=$4; fl+=$6} END {if (NR) printf " (%d test binaries: %d passed, %d failed)", NR, p, fl}')
  py=$(grep -oE '^[0-9]+ passed.* in [0-9.]+s' "$f" | tail -1)
  [ -n "$py" ] && r=" ($py)"
  echo "rc=$rc $name$r [${t}s]" > "$L/res/$(printf %02d "$n")"
}

lane_native() {
  export CARGO_TARGET_DIR=$OUT/native
  step 1 fmt cargo fmt --all --check
  step 2 clippy cargo clippy --workspace --all-targets -- -D warnings
  step 3 clippy-research cargo clippy --workspace --all-targets --features treewalker-bench/research -- -D warnings
  step 4 test cargo test --workspace
  [ "$MODE" = full ] || return 0
  step 5 test-release cargo test --release --features research
  step 6 test-bench cargo test -p treewalker-bench
}

lane_property() {
  export CARGO_TARGET_DIR=$OUT/property
  step 7 property-full env TW_PROPTEST_FULL=1 cargo test --release --features research --test property
}

lane_cross() {
  export CARGO_TARGET_DIR=$OUT/cross
  local sdk n=7
  sdk=$(xcrun --show-sdk-path)
  export CC_x86_64_unknown_linux_gnu=clang \
    CFLAGS_x86_64_unknown_linux_gnu="--target=x86_64-unknown-linux-gnu -isystem $sdk/usr/include" \
    CC_aarch64_unknown_linux_gnu=clang \
    CFLAGS_aarch64_unknown_linux_gnu="--target=aarch64-unknown-linux-gnu -isystem $sdk/usr/include -D__arm64__"
  for t in "aarch64-apple-darwin apple-m4" "x86_64-unknown-linux-gnu emeraldrapids" \
           "x86_64-unknown-linux-gnu x86-64-v4" "x86_64-unknown-linux-gnu x86-64" \
           "aarch64-unknown-linux-gnu neoverse-v2"; do
    set -- $t
    for f in "" "--features treewalker-bench/external-bench,treewalker-bench/pmu" "--features treewalker-bench/research"; do
      n=$((n + 1))
      step $n "cross $1 $2 $f" env RUSTFLAGS="-C target-cpu=$2" cargo clippy --target "$1" --workspace --all-targets $f -- -D warnings
    done
  done
}

lane_python() {
  if [ "$MODE" = full ]; then
    step 23 spills-m4 "$PY" experiments/scripts/walk_spills.py
    step 24 spills-x86 "$PY" experiments/scripts/walk_spills.py --target x86_64-unknown-linux-gnu --cpu emeraldrapids
    step 25 spills-x86-v4 "$PY" experiments/scripts/walk_spills.py --target x86_64-unknown-linux-gnu --cpu x86-64-v4
    step 26 spills-v2 "$PY" experiments/scripts/walk_spills.py --target aarch64-unknown-linux-gnu --cpu neoverse-v2
  fi
  step 27 pytest "$PY" -m pytest -q
  step 28 ruff "$PY" -m ruff check
  step 29 ruff-format "$PY" -m ruff format --check
  step 30 mypy "$PY" -m mypy
}

# miri N KIND CPU TARGET FLAGS: the lib's prefetching_walks (KIND lib) or tests/miri.rs
# (KIND test) under Miri, in its own target directory.
miri() {
  local n=$1 kind=$2 cpu=$3 target=$4 flags=$5 sel
  if [ "$kind" = lib ]; then sel="--lib -- prefetching_walks"; else sel="--test miri"; fi
  step "$n" "miri $kind $cpu" env CARGO_TARGET_DIR="$OUT/miri-$kind-$cpu" \
    RUSTFLAGS="$flags" cargo +nightly miri test --target "$target" $sel
}

lane_artifacts() {
  export CARGO_TARGET_DIR=$OUT/artifacts
  step 37 artifacts-core cargo test --manifest-path experiments/benchmarks/Cargo.toml --release --features research
}

echo "checks ($MODE) of $(git rev-parse --short HEAD), $(date '+%F %T')" | tee "$L/summary.txt"
lane_native & lane_python &
if [ "$MODE" = full ]; then
  lane_property & lane_cross & lane_artifacts &
  n=30
  for t in "x86-64 x86_64-unknown-linux-gnu -Ctarget-cpu=x86-64" \
           "x86-64-v3 x86_64-unknown-linux-gnu -Ctarget-cpu=x86-64-v3" "riscv64 riscv64gc-unknown-linux-gnu"; do
    set -- $t
    miri $((n + 1)) lib "$1" "$2" "${3:-}" & miri $((n + 2)) test "$1" "$2" "${3:-}" &
    n=$((n + 2))
  done
fi
wait
cat "$L"/res/* | tee -a "$L/summary.txt"
fails=$(cat "$L"/res/* | grep -cv '^rc=0 ')
echo "$(ls "$L/res" | wc -l | tr -d ' ') steps, $fails failed, $(( $(date +%s) - T0 )) s" | tee -a "$L/summary.txt"
[ "$fails" = 0 ]
