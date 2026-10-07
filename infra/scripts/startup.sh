#!/usr/bin/env bash
# TreeWalker reproducible benchmark — GCE startup script.
#
# Runs end to end with the repository's own commands: system packages →
# source → Rust (rust-toolchain.toml) and Python (.python-version) → native
# LightGBM and XGBoost (treewalker-exp build-native) → data and models →
# compiled baselines → sweep_bench → results.
#
# Templatefile variables (interpolated by Terraform):
#   gcs_source_uri    — gs:// path to the source archive (git archive of git_ref)
#   gcs_results_base  — gs:// prefix for uploading results
#   gcs_artifacts_uri — gs:// prefix where the trainer hands its fresh models over
#   gcs_expedia_uri   — gs:// path to expedia.parquet, empty when no suite needs it
#   gcs_cache_uri     — gs:// bucket that keeps prepared models and compiled
#                       baselines across deployments; empty for none
#   git_ref           — the ref archived; the commit is in experiments/SOURCE_COMMIT
#   role              — "trainer" (prepares and uploads models) or "benchmarker"
#                       (downloads the trainer's models, checked by hash)
#   suites            — space-separated suites from experiments/grids.toml
#   layout_check      — "true": also time TreeWalker on the acceptance cells in a
#                       build with 64-byte function alignment
#   apt_snapshot      — Ubuntu archive snapshot every apt operation uses; empty
#                       for the live archive
#   machine_type, image, turbo_mode, pmu_level — recorded in run.json
#
# Results land in /home/bench/results/. Progress is logged to
# /var/log/treewalker-bench.log.
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GCS_SOURCE_URI="${gcs_source_uri}"
GCS_RESULTS_BASE="${gcs_results_base}"
GCS_ARTIFACTS_URI="${gcs_artifacts_uri}"
GCS_EXPEDIA_URI="${gcs_expedia_uri}"
GCS_CACHE_URI="${gcs_cache_uri}"
ROLE="${role}"
GIT_REF="${git_ref}"
SUITES="${suites}"
LAYOUT_CHECK="${layout_check}"
APT_SNAPSHOT="${apt_snapshot}"
MACHINE_TYPE="${machine_type}"
IMAGE="${image}"
TURBO_MODE="${turbo_mode}"
PMU_LEVEL="${pmu_level}"

BENCH_USER=bench
BENCH_HOME=/home/$BENCH_USER
LOG=/var/log/treewalker-bench.log
RESULTS_DIR=$BENCH_HOME/results
REPO_DIR=$BENCH_HOME/treewalker
# The trainer's hand-over marker names this ref and these suites, so a marker
# left in the bucket by an earlier deployment cannot release the benchmarker.
UPLOAD_MARKER="UPLOAD_DONE-$GIT_REF-$(echo "$SUITES" | tr ' ' '+')"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date -Iseconds)] $*"; }

fail() {
  log "ERROR: $*"
  mkdir -p "$RESULTS_DIR"
  echo "$*" > "$RESULTS_DIR/ERROR"
  chown -R $BENCH_USER:$BENCH_USER "$RESULTS_DIR"
  exit 1
}

phase() { log "===== Phase $1: $2 ====="; }

# Catch any unhandled error (set -e exit) and write the ERROR sentinel so
# Terraform's check_done output never reports RUNNING for a crashed VM.
trap 'fail "Unhandled error on line $LINENO (exit code $?)"' ERR

# Run a command as the bench user in the repository, with cargo and uv on PATH.
# Every uv run passes --group baselines, because uv run removes packages
# outside the groups it syncs.
as_bench() {
  su - $BENCH_USER -c "source \"\$HOME/.cargo/env\" 2>/dev/null; source \"\$HOME/.local/bin/env\"; cd '$REPO_DIR' && $*"
}
twx() { as_bench "uv run --group baselines treewalker-exp $*"; }

# The cache: prepared/ holds what prep writes, compiled/<machine type>/ the
# baselines compiled on that machine. A pull only offers files: prep reuses a
# model or cell only when its manifest's key and hashes match, and
# compile-baselines a library only when its compile identity and hash match;
# anything else is rebuilt. A failed pull or push costs time, never the run.
# Bulk copies print nothing per file: the serial console drains at about 10 KB/s,
# and a 47 GB pull's file list kept it an hour behind the VM. Errors still print.
COMPILED_RE='.*\.(so|dylib)$|.*/(tl2cgen|lleaves|quickscorer)\.(json|xml)$'
ONLY_COMPILED_RE='^(?!.*\.(so|dylib)$)(?!.*/(tl2cgen|lleaves|quickscorer)\.(json|xml)$)'
cache_pull() {
  [ -n "$GCS_CACHE_URI" ] || return 0
  as_bench "mkdir -p experiments/artifacts"
  if gcloud storage ls "$GCS_CACHE_URI/$1/" &>/dev/null; then
    gcloud storage rsync --no-user-output-enabled -r "$GCS_CACHE_URI/$1/" "$REPO_DIR/experiments/artifacts/" \
      || log "Cache: pulling $1 failed; building here"
    chown -R $BENCH_USER:$BENCH_USER "$REPO_DIR/experiments/artifacts"
  fi
}
cache_push() {
  [ -n "$GCS_CACHE_URI" ] || return 0
  gcloud storage rsync --no-user-output-enabled -r --exclude="$2" "$REPO_DIR/experiments/artifacts/" "$GCS_CACHE_URI/$1/" \
    || log "Cache: pushing $1 failed"
}

metadata() {
  curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null || true
}

# Single redirect — all stdout/stderr goes to log + console (no double lines)
exec > >(tee -a "$LOG") 2>&1

# ---------------------------------------------------------------------------
# Phase 0: Bench user
# ---------------------------------------------------------------------------
phase 0 "Bench user"
if ! id $BENCH_USER &>/dev/null; then
  useradd -m -s /bin/bash $BENCH_USER
fi
mkdir -p "$RESULTS_DIR"
chown $BENCH_USER:$BENCH_USER "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Phase 1: System packages: a C/C++ compiler, CMake, git and curl. No LLVM:
# lleaves compiles through the LLVM that llvmlite bundles.
# ---------------------------------------------------------------------------
phase 1 "System packages"
export DEBIAN_FRONTEND=noninteractive
# The image is pinned, and so is the archive: with a snapshot, update, upgrade and
# install see the archive as it was then (Ubuntu 24.04 and later need no source
# changes), so GCC and the other tools match across deployments and the cached
# compiled baselines stay valid.
if [ -n "$APT_SNAPSHOT" ]; then
  echo "APT::Snapshot \"$APT_SNAPSHOT\";" > /etc/apt/apt.conf.d/50snapshot
fi
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq build-essential cmake git curl pkg-config libgomp1
log "cmake: $(cmake --version | head -1); gcc: $(gcc --version | head -1)"
log "apt: $(apt-cache policy gcc | grep -m1 -E 'https?://' | sed -E 's/^ +//')"

# ---------------------------------------------------------------------------
# Phase 2: Rust and uv. rustup installs no toolchain, so rust-toolchain.toml
# (1.97.1) governs; uv follows .python-version (3.14).
# ---------------------------------------------------------------------------
phase 2 "Rust and uv"
su - $BENCH_USER -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none'
su - $BENCH_USER -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

# ---------------------------------------------------------------------------
# Phase 3: Source
# ---------------------------------------------------------------------------
phase 3 "Source ($GIT_REF)"
mkdir -p "$REPO_DIR"
chown $BENCH_USER:$BENCH_USER "$REPO_DIR"
gcloud storage cp "$GCS_SOURCE_URI" /tmp/source.tar.gz
su - $BENCH_USER -c "tar xzf /tmp/source.tar.gz -C '$REPO_DIR'"
rm -f /tmp/source.tar.gz
log "Source commit: $(cat "$REPO_DIR/experiments/SOURCE_COMMIT")"
log "Rust: $(as_bench 'rustc --version')"

# ---------------------------------------------------------------------------
# Phase 4: Python dependencies and native libraries
# ---------------------------------------------------------------------------
phase 4 "Python dependencies and native LightGBM, XGBoost"
# tl2cgen has no aarch64 Linux wheel: there it builds from source through the
# tracked <cstdint> wrapper.
as_bench "CXX='$REPO_DIR/infra/scripts/cxx-cstdint' uv sync --locked --group baselines"
twx build-native

# ---------------------------------------------------------------------------
# Phase 5: Data and models. The trainer starts from the cache, if any, and
# prep rebuilds anything its manifest does not vouch for. The benchmarker takes
# the trainer's models; sweep_bench checks every file against the hashes in
# cell.json. Each machine starts its compiles from its own cached baselines.
# ---------------------------------------------------------------------------
phase 5 "Data and models ($ROLE)"
if [ "$ROLE" = "trainer" ]; then
  twx fetch
  if [ -n "$GCS_EXPEDIA_URI" ]; then
    gcloud storage cp "$GCS_EXPEDIA_URI" "$REPO_DIR/experiments/data/expedia.parquet"
    chown $BENCH_USER:$BENCH_USER "$REPO_DIR/experiments/data/expedia.parquet"
    twx fetch-expedia --check-only
  fi
  cache_pull prepared
  for suite in $SUITES; do
    twx prepare --suite "$suite"
  done
  cache_push prepared "$COMPILED_RE"
  # Compiled baselines are per machine; the benchmarker compiles its own.
  gcloud storage rsync --no-user-output-enabled -r --exclude="$COMPILED_RE" \
    "$REPO_DIR/experiments/artifacts/" "$GCS_ARTIFACTS_URI/"
  echo "$(date -Iseconds)" | gcloud storage cp - "$GCS_ARTIFACTS_URI/$UPLOAD_MARKER"
  log "Models uploaded to $GCS_ARTIFACTS_URI"
else
  WAIT_START=$(date +%s)
  MAX_WAIT=86400  # 24 hours: the factorial suite trains 640 models first
  until gcloud storage ls "$GCS_ARTIFACTS_URI/$UPLOAD_MARKER" &>/dev/null; do
    ELAPSED=$(( $(date +%s) - WAIT_START ))
    [ $ELAPSED -ge $MAX_WAIT ] && fail "Timed out waiting for the trainer after $${MAX_WAIT}s"
    log "Waiting for the trainer's models... ($((ELAPSED/60))m)"
    sleep 60
  done
  mkdir -p "$REPO_DIR/experiments/artifacts"
  gcloud storage rsync --no-user-output-enabled -r "$GCS_ARTIFACTS_URI/" "$REPO_DIR/experiments/artifacts/"
  rm -f "$REPO_DIR/experiments/artifacts/"UPLOAD_DONE*
  chown -R $BENCH_USER:$BENCH_USER "$REPO_DIR/experiments/artifacts"
fi
# Push the compiled baselines even when a compile fails, so a rerun keeps them.
cache_pull "compiled/$MACHINE_TYPE"
COMPILED_OK=true
for suite in $SUITES; do
  twx compile-baselines --suite "$suite" || COMPILED_OK=false
done
cache_push "compiled/$MACHINE_TYPE" "$ONLY_COMPILED_RE"
$COMPILED_OK || fail "Compiling the baselines failed"

# ---------------------------------------------------------------------------
# Phase 6: The machine, tuned and described
# ---------------------------------------------------------------------------
phase 6 "Machine settings"
# Per-thread, user-space counting of the PMU events.
sysctl -w kernel.perf_event_paranoid=2
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [ -f "$gov" ] && echo performance > "$gov"
done
echo 0 > /proc/sys/kernel/randomize_va_space
sync && echo 3 > /proc/sys/vm/drop_caches

# Facts sweep_bench cannot see from inside the guest, recorded in run.json.
SYSTEM_INFO=$BENCH_HOME/system_info.json
cat > "$SYSTEM_INFO" <<SYSINFO
{
  "git_ref": "$GIT_REF",
  "machine_type": "$MACHINE_TYPE",
  "zone": "$(metadata zone | awk -F/ '{print $NF}')",
  "image_requested": "$IMAGE",
  "image": "$(metadata image)",
  "apt_snapshot": "$APT_SNAPSHOT",
  "turbo_mode": "$TURBO_MODE",
  "pmu_level": "$PMU_LEVEL",
  "os_release": "$(. /etc/os-release && echo "$PRETTY_NAME")",
  "kernel_cmdline": "$(cat /proc/cmdline)",
  "cc": "$(cc --version | head -1)",
  "cxx": "$(c++ --version | head -1)",
  "cmake": "$(cmake --version | head -1)",
  "git": "$(git --version)",
  "uv": "$(as_bench 'uv --version')",
  "rustup": "$(as_bench 'rustup --version 2>/dev/null' | head -1)",
  "page_size": "$(getconf PAGESIZE)",
  "thp_enabled": "$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)",
  "thp_defrag": "$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)"
}
SYSINFO
chown $BENCH_USER:$BENCH_USER "$SYSTEM_INFO"
cat "$SYSTEM_INFO"

# ---------------------------------------------------------------------------
# Phase 7: Preflight and runs, pinned to core 0
# ---------------------------------------------------------------------------
phase 7 "Runs: $SUITES"
for suite in $SUITES; do
  twx preflight --suite "$suite" --require-pmu
done
ARCH_LABEL=$(uname -m)
# Copy finished cells to the bucket every 30 minutes while the suites run, so a
# long run's results are visible, and survive the VM, before it ends. Cells
# being written live under temporary names, which the copy skips.
# As the bench user: the runs write into it.
as_bench "mkdir -p experiments/data/runs"
# The copy runs niced on the last CPU, away from the pinned benchmark on core 0.
( while sleep 1800; do
    taskset -c "$(( $(nproc) - 1 ))" nice -n 19 gcloud storage rsync --no-user-output-enabled -r --exclude='.*/\.tmp-.*' \
      "$REPO_DIR/experiments/data/runs" "$GCS_RESULTS_BASE/runs" >/dev/null 2>&1 || true
  done ) &
SYNC_PID=$!
# A run exits nonzero when a cell fails; the cell's diagnostics are in failed/.
# Keep going, so one cell cannot cost the later suites or the results upload.
for suite in $SUITES; do
  twx run --suite "$suite" --run-id "$suite-$ARCH_LABEL" --core 0 \
    --system-info "$SYSTEM_INFO" --require-pmu \
    || log "Run $suite: some cells failed; see runs/$suite-$ARCH_LABEL/failed/"
done
kill "$SYNC_PID" 2>/dev/null || true
if [ "$LAYOUT_CHECK" = "true" ]; then
  # Layout sensitivity: the same kernel, instruction-identical, in a build whose
  # functions are aligned to 64 bytes. The spread is reported with the results.
  twx run --suite acceptance --run-id "layout-default-$ARCH_LABEL" --core 0 \
    --only treewalker,treewalker_research --system-info "$SYSTEM_INFO" --require-pmu \
    || log "Layout check (default): some cells failed"
  twx run --suite acceptance --run-id "layout-align64-$ARCH_LABEL" --core 0 \
    --only treewalker,treewalker_research --system-info "$SYSTEM_INFO" --require-pmu \
    --rustflags "'-C target-cpu=native -C llvm-args=-align-all-functions=6'" \
    || log "Layout check (align64): some cells failed"
fi

# ---------------------------------------------------------------------------
# Phase 8: Results
# ---------------------------------------------------------------------------
phase 8 "Results"
cp -r "$REPO_DIR/experiments/data/runs" "$RESULTS_DIR/"
cp "$SYSTEM_INFO" "$RESULTS_DIR/"
chown -R $BENCH_USER:$BENCH_USER "$RESULTS_DIR"
echo "$(date -Iseconds)" > "$RESULTS_DIR/DONE"
gcloud storage cp -r "$RESULTS_DIR/*" "$GCS_RESULTS_BASE/"
log "Results uploaded to $GCS_RESULTS_BASE/"
log "===== ALL PHASES COMPLETE ====="
