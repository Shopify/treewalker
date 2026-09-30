#!/usr/bin/env bash
# TreeWalker reproducible benchmark — GCE startup script.
#
# Runs end-to-end: install deps → download source → build native LGB/XGB →
# build TreeWalker → generate artifacts → run sweep → collect CSVs.
#
# Templatefile variables (interpolated by Terraform):
#   gcs_source_uri    — gs:// path to source tarball
#   gcs_results_base  — gs:// prefix for uploading results
#   gcs_artifacts_uri — gs:// prefix for shared training artifacts
#   git_ref           — branch/SHA label for metadata
#   role              — "trainer" (trains models, uploads artifacts)
#                       or "benchmarker" (downloads artifacts, compiles, benchmarks)
#
# Results land in /home/bench/results/.
# Progress logged to /var/log/treewalker-bench.log.
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GCS_SOURCE_URI="${gcs_source_uri}"
GCS_RESULTS_BASE="${gcs_results_base}"
GCS_ARTIFACTS_URI="${gcs_artifacts_uri}"
GCS_EXPEDIA_URI="${gcs_expedia_uri}"  # empty unless bench_suite = "paper"
ROLE="${role}"
GIT_REF="${git_ref}"
BENCH_SUITE="${bench_suite}"  # "paper" (full sweep) or "rebuttal" (E1 scenario + E2 chunked-G)
LLVM_VERSION=20
CMAKE_VER=3.31.6
LIGHTGBM_VER=v4.6.0
XGBOOST_VER=v3.2.0

BENCH_USER=bench
BENCH_HOME=/home/$BENCH_USER
LOG=/var/log/treewalker-bench.log
RESULTS_DIR=$BENCH_HOME/results

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

run_as_bench() {
  su - $BENCH_USER -c "$*"
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "ERROR: command failed with exit code $rc"
    return $rc
  fi
}

# Single redirect — all stdout/stderr goes to log + console (no double lines)
exec > >(tee -a "$LOG") 2>&1

# ---------------------------------------------------------------------------
# Phase 0: Create bench user
# ---------------------------------------------------------------------------
phase 0 "Create bench user"
if ! id $BENCH_USER &>/dev/null; then
  useradd -m -s /bin/bash $BENCH_USER
fi
mkdir -p "$RESULTS_DIR"
chown $BENCH_USER:$BENCH_USER "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Phase 1: System deps
# ---------------------------------------------------------------------------
phase 1 "System dependencies"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
  build-essential cmake git curl wget pkg-config \
  libssl-dev libgomp1 \
  lsb-release software-properties-common gnupg

# CMake (LightGBM needs >= 3.28; Ubuntu 24.04 ships 3.28.3 which is fine)
log "cmake: $(cmake --version | head -1)"

# ---------------------------------------------------------------------------
# Phase 2: LLVM (for lleaves compile)
# ---------------------------------------------------------------------------
phase 2 "LLVM $LLVM_VERSION"
wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh
bash /tmp/llvm.sh "$LLVM_VERSION" || log "WARNING: LLVM apt install failed"
# Create symlinks (idempotent)
for tool in llc clang opt; do
  versioned="/usr/bin/$${tool}-$${LLVM_VERSION}"
  [ -f "$versioned" ] && ln -sf "$versioned" "/usr/bin/$${tool}"
done
if command -v llc &>/dev/null; then
  log "LLVM: $(llc --version 2>&1 | head -1)"
else
  log "WARNING: llc not found after LLVM install"
fi

# ---------------------------------------------------------------------------
# Phase 3: Rust (as bench user)
# ---------------------------------------------------------------------------
phase 3 "Rust toolchain"
run_as_bench 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable'
log "Rust: $(run_as_bench 'source "$HOME/.cargo/env" && rustc --version')"

# ---------------------------------------------------------------------------
# Phase 4: Python + uv (as bench user)
# ---------------------------------------------------------------------------
phase 4 "Python + uv"
run_as_bench 'curl -LsSf https://astral.sh/uv/install.sh | sh'
run_as_bench 'source "$HOME/.local/bin/env" && uv python install 3.12'
log "uv: $(run_as_bench 'source "$HOME/.local/bin/env" && uv --version')"

# ---------------------------------------------------------------------------
# Phase 5: Download source from GCS
# ---------------------------------------------------------------------------
phase 5 "Download source ($GIT_REF)"
REPO_DIR=$BENCH_HOME/treewalker
mkdir -p "$REPO_DIR"
chown $BENCH_USER:$BENCH_USER "$REPO_DIR"
gcloud storage cp "$GCS_SOURCE_URI" /tmp/source.tar.gz
su - $BENCH_USER -c "tar xzf /tmp/source.tar.gz -C '$REPO_DIR'"
rm -f /tmp/source.tar.gz
log "Source extracted to $REPO_DIR"

# ---------------------------------------------------------------------------
# Phase 6: Build LightGBM C++ from source (native)
# ---------------------------------------------------------------------------
phase 6 "Build LightGBM $LIGHTGBM_VER (native C++)"
LGB_SRC=/tmp/LightGBM
if [ ! -d "$LGB_SRC/.git" ]; then
  rm -rf "$LGB_SRC"
  git clone --depth 1 --branch "$LIGHTGBM_VER" --recurse-submodules \
    https://github.com/microsoft/LightGBM.git "$LGB_SRC"
fi
mkdir -p "$LGB_SRC/build" && cd "$LGB_SRC/build"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-march=native" \
  -DCMAKE_CXX_FLAGS="-march=native" \
  -DUSE_OPENMP=ON
make -j"$(nproc)"
make install && ldconfig
log "LightGBM C++ built and installed"

# ---------------------------------------------------------------------------
# Phase 7: Build XGBoost C++ from source (native)
# ---------------------------------------------------------------------------
phase 7 "Build XGBoost $XGBOOST_VER (native C++)"
XGB_SRC=/tmp/xgboost
if [ ! -d "$XGB_SRC/.git" ]; then
  rm -rf "$XGB_SRC"
  git clone --depth 1 --branch "$XGBOOST_VER" --recurse-submodules \
    https://github.com/dmlc/xgboost.git "$XGB_SRC"
fi
mkdir -p "$XGB_SRC/build" && cd "$XGB_SRC/build"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-march=native" \
  -DCMAKE_CXX_FLAGS="-march=native"
make -j"$(nproc)"
log "XGBoost C++ built"

chmod -R a+rX "$LGB_SRC" "$XGB_SRC"

# ---------------------------------------------------------------------------
# Phase 8: Python deps (uv sync + native LGB/XGB override)
# ---------------------------------------------------------------------------
phase 8 "Python dependencies"

run_as_bench "
  source \"\$HOME/.local/bin/env\"
  cd '$REPO_DIR'
  uv sync
"

# Install tl2cgen separately (optional dep — needs g++ wrapper on aarch64
# because treelite's postprocessor.h is missing #include <cstdint>)
cat > /usr/local/bin/g++-fix << 'GCCEOF'
#!/bin/bash
exec /usr/bin/g++ -include cstdint "$@"
GCCEOF
chmod +x /usr/local/bin/g++-fix
run_as_bench "source \"\$HOME/.local/bin/env\" && cd '$REPO_DIR' && CXX=/usr/local/bin/g++-fix uv pip install 'tl2cgen>=1.0.0'" || log "WARNING: tl2cgen install failed"
run_as_bench "source \"\$HOME/.local/bin/env\" && cd '$REPO_DIR' && uv pip install 'lleaves @ git+https://github.com/siboehm/lleaves.git@v1.4.1'" || log "WARNING: lleaves install failed"
log "uv sync complete"

# Override LightGBM .so with native-built version.
LGB_SO=$(find "$REPO_DIR/.venv" -name "lib_lightgbm.so" -print -quit 2>/dev/null)
if [ -n "$LGB_SO" ] && [ -f /usr/local/lib/lib_lightgbm.so ]; then
  cp /usr/local/lib/lib_lightgbm.so "$LGB_SO"
  chown $BENCH_USER:$BENCH_USER "$LGB_SO"
  log "LightGBM .so replaced with native build"
else
  log "WARNING: Could not replace LightGBM .so"
fi

# Override XGBoost .so with native-built version.
XGB_SO=$(find "$REPO_DIR/.venv" -name "libxgboost.so" -print -quit 2>/dev/null)
if [ -n "$XGB_SO" ] && [ -f "$XGB_SRC/lib/libxgboost.so" ]; then
  cp "$XGB_SRC/lib/libxgboost.so" "$XGB_SO"
  chown $BENCH_USER:$BENCH_USER "$XGB_SO"
  log "XGBoost .so replaced with native build"
else
  log "WARNING: Could not replace XGBoost .so"
fi

# Verify imports and version match
run_as_bench "source \"\$HOME/.local/bin/env\" && cd '$REPO_DIR' && uv run python3 -c 'import lightgbm; print(f\"lightgbm {lightgbm.__version__}\")'"
run_as_bench "source \"\$HOME/.local/bin/env\" && cd '$REPO_DIR' && uv run python3 -c 'import xgboost; print(f\"xgboost {xgboost.__version__}\")'"

# ---------------------------------------------------------------------------
# Phase 9: Generate / download artifacts
# ---------------------------------------------------------------------------
if [ "$ROLE" = "trainer" ]; then
  phase 9 "Generate artifacts + upload (trainer role)"

  # Pull any existing artifacts from GCS first so prepare.py skips them.
  mkdir -p "$REPO_DIR/paper/experiments/artifacts"
  chown -R $BENCH_USER:$BENCH_USER "$REPO_DIR/paper/experiments/artifacts"
  if gcloud storage ls "$GCS_ARTIFACTS_URI/" &>/dev/null; then
    gcloud storage rsync -r \
      "$GCS_ARTIFACTS_URI/" "$REPO_DIR/paper/experiments/artifacts/"
    # Remove stale sentinel so we write a fresh one after upload.
    rm -f "$REPO_DIR/paper/experiments/artifacts/UPLOAD_DONE"
    chown -R $BENCH_USER:$BENCH_USER "$REPO_DIR/paper/experiments/artifacts"
    log "Pulled existing artifacts from GCS (prepare.py will skip them)"
  else
    log "No existing artifacts in GCS — training from scratch"
  fi

  # Train models, export treelite, generate test data, group distributions.
  # Skip compiled baselines — those are architecture-specific.
  # Idempotent: skips any config where artifacts already exist on disk.
  if [ "$BENCH_SUITE" = "rebuttal" ]; then
    # Rebuttal suite: SUPPORT reference combo + E1 scenario artifacts (train +
    # validate) + E2 chunked artifacts (prepare + correctness gate).
    run_as_bench "
      source \"\$HOME/.local/bin/env\"
      source \"\$HOME/.cargo/env\"
      cd '$REPO_DIR'
      uv run python3 paper/experiments/scripts/prepare.py --datasets support --combos 500,8,16 --skip-compiled
      uv run python3 paper/experiments/scripts/prepare_scenario.py
      uv run python3 paper/experiments/scripts/prepare_chunked.py --stage prepare
      uv run python3 paper/experiments/scripts/prepare_chunked.py --stage correctness
    "
    log "Rebuttal artifacts generated (SUPPORT + scenario + chunked), correctness gates passed"
  else
    # Expedia is not redistributable: Terraform uploads the locally built
    # parquet; check its fingerprint before training on it.
    [ -n "$GCS_EXPEDIA_URI" ] || fail "bench_suite=paper needs expedia.parquet (see fetch_expedia.py)"
    gcloud storage cp "$GCS_EXPEDIA_URI" "$REPO_DIR/paper/experiments/data/expedia.parquet"
    chown $BENCH_USER:$BENCH_USER "$REPO_DIR/paper/experiments/data/expedia.parquet"
    run_as_bench "
      source \"\$HOME/.local/bin/env\"
      source \"\$HOME/.cargo/env\"
      cd '$REPO_DIR'
      uv run python3 paper/experiments/scripts/fetch_expedia.py --check-only
      uv run python3 paper/experiments/scripts/prepare.py --grid all --prepare-groups --skip-compiled
    "
    log "Artifacts generated (models + data)"
  fi

  # Upload shared artifacts to GCS for the benchmarker.
  # Excludes .so files (architecture-specific).
  gcloud storage rsync -r --exclude=".*\.so$" \
    "$REPO_DIR/paper/experiments/artifacts/" "$GCS_ARTIFACTS_URI/"
  # Write sentinel AFTER upload completes so benchmarker knows it's safe.
  echo "$(date -Iseconds)" | gcloud storage cp - "$GCS_ARTIFACTS_URI/UPLOAD_DONE"
  log "Shared artifacts uploaded to $GCS_ARTIFACTS_URI"

  # Force recompilation of tl2cgen .so (model source changed from native to treelite binary)
  find "$REPO_DIR/paper/experiments/artifacts" -name "tl2cgen.so" -delete
  log "Cleared stale tl2cgen .so files"

  # Now compile tl2cgen/lleaves locally for this architecture.
  # Rebuttal suite runs only treewalker + fullwalk — compiled baselines not needed.
  if [ "$BENCH_SUITE" != "rebuttal" ]; then
    run_as_bench "
      source \"\$HOME/.local/bin/env\"
      source \"\$HOME/.cargo/env\"
      cd '$REPO_DIR'
      uv run python3 paper/experiments/scripts/prepare.py --grid all --compile-only
    "
    log "Local .so compilation complete"
  fi

else
  phase 9 "Download artifacts + compile locally (benchmarker role)"

  # Wait for trainer to upload artifacts (poll every 60s, max 3h).
  WAIT_START=$(date +%s)
  MAX_WAIT=10800  # 3 hours
  while true; do
    # Check for any walker_config.json in the artifacts bucket.
    if gcloud storage ls "$GCS_ARTIFACTS_URI/UPLOAD_DONE" &>/dev/null; then
      log "Trainer upload sentinel detected — artifacts complete"
      break
    fi
    ELAPSED=$(( $(date +%s) - WAIT_START ))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
      fail "Timed out waiting for trainer artifacts after $${MAX_WAIT}s"
    fi
    log "Waiting for UPLOAD_DONE sentinel... ($((ELAPSED/60))m elapsed)"
    sleep 60
  done

  # Download shared artifacts (models, data, configs — no .so files).
  mkdir -p "$REPO_DIR/paper/experiments/artifacts"
  chown -R $BENCH_USER:$BENCH_USER "$REPO_DIR/paper/experiments/artifacts"
  gcloud storage rsync -r \
    "$GCS_ARTIFACTS_URI/" "$REPO_DIR/paper/experiments/artifacts/"
  chown -R $BENCH_USER:$BENCH_USER "$REPO_DIR/paper/experiments/artifacts"
  log "Shared artifacts downloaded"

  # Force recompilation of tl2cgen .so (model source changed from native to treelite binary)
  find "$REPO_DIR/paper/experiments/artifacts" -name "tl2cgen.so" -delete
  log "Cleared stale tl2cgen .so files"

  if [ "$BENCH_SUITE" = "rebuttal" ]; then
    # Rebuttal suite: artifacts (incl. scenario cells, chunked dirs, references)
    # came from GCS. prepare_scenario skips existing artifacts but re-validates
    # (builds sweep_bench as a side effect); chunked correctness re-runs on this arch.
    run_as_bench "
      source \"\$HOME/.local/bin/env\"
      source \"\$HOME/.cargo/env\"
      cd '$REPO_DIR'
      uv run python3 paper/experiments/scripts/prepare_scenario.py
      uv run python3 paper/experiments/scripts/prepare_chunked.py --stage correctness
    "
    log "Rebuttal correctness gates passed on this architecture"
  else
    # Build sweep_bench + compile tl2cgen/lleaves locally.
    run_as_bench "
      source \"\$HOME/.local/bin/env\"
      source \"\$HOME/.cargo/env\"
      cd '$REPO_DIR'
      uv run python3 paper/experiments/scripts/prepare.py --grid all --compile-only
    "
    log "Local .so compilation complete"
  fi
fi

# ---------------------------------------------------------------------------
# Phase 10: Benchmark sweep
# ---------------------------------------------------------------------------
phase 10 "Benchmark sweep"

# Kernel tuning for stable benchmarks
log "Tuning kernel for benchmark stability"

# Set CPU governor to performance (fixed frequency)
# Note: turbo boost is managed by the GCE hypervisor, not the guest.
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [ -f "$gov" ] && echo performance > "$gov"
done
log "CPU governor: performance"

# Disable ASLR
echo 0 > /proc/sys/kernel/randomize_va_space
log "ASLR disabled"

# Drop caches
sync && echo 3 > /proc/sys/vm/drop_caches
log "Caches dropped"

# Log system info
log "--- System Info ---"
log "CPUs: $(nproc)"
lscpu 2>/dev/null | grep -iE "(model name|socket|core|thread|cache)" | while read -r line; do
  log "  $line"
done
free -h 2>/dev/null | head -2 | while read -r line; do
  log "  $line"
done

# Detect architecture label for output files.
ARCH_LABEL=$(uname -m)
case "$ARCH_LABEL" in
  x86_64|amd64) ARCH_LABEL="intel" ;;
  aarch64|arm64) ARCH_LABEL="arm" ;;
esac
log "Architecture label: $ARCH_LABEL"

# Locate native-compiled LightGBM and XGBoost shared libraries.
# Phase 6 installs LightGBM to /usr/local/lib via make install.
# Phase 7 builds XGBoost to $XGB_SRC/lib/.
# Both compiled with -march=native for fair comparison.
LGB_LIB=/usr/local/lib/lib_lightgbm.so
XGB_LIB=$XGB_SRC/lib/libxgboost.so
[ -f "$LGB_LIB" ] && log "LightGBM lib: $LGB_LIB (-march=native)" || log "WARNING: $LGB_LIB not found"
[ -f "$XGB_LIB" ] && log "XGBoost lib: $XGB_LIB (-march=native)" || log "WARNING: $XGB_LIB not found"

# Build optional flags for sweep_bench.
EXT_FLAGS=""
[ -f "$LGB_LIB" ] && EXT_FLAGS="$EXT_FLAGS --lgb-lib $LGB_LIB"
[ -f "$XGB_LIB" ] && EXT_FLAGS="$EXT_FLAGS --xgb-lib $XGB_LIB"

if [ "$BENCH_SUITE" = "rebuttal" ]; then
  # Clear the released CSVs so this run writes fresh ones.
  find "$REPO_DIR/paper/experiments/data" \( -name "scenario_credit_*" -o -name "chunked_g_*" \) -delete 2>/dev/null
  log "Cleared stale rebuttal CSVs/summaries"

  # E1: scenario grid (treewalker + fullwalk, timing + stats + inline correctness).
  run_as_bench "
    source \"\$HOME/.cargo/env\"
    cd '$REPO_DIR'
    taskset -c 0 ./target/release/sweep_bench \
      '$REPO_DIR/paper/experiments/artifacts' \
      --grid scen \
      --output-dir '$REPO_DIR/paper/experiments/data' \
      --warmup 3 --iters 21 --min-iters 11 \
      --max-time-secs 10
  "
  log "E1 scenario sweep complete"

  # E2: chunked-G timing pass.
  run_as_bench "
    source \"\$HOME/.local/bin/env\"
    source \"\$HOME/.cargo/env\"
    cd '$REPO_DIR'
    taskset -c 0 uv run python3 paper/experiments/scripts/prepare_chunked.py --stage timing
  "
  log "E2 chunked-G timing complete"

  # Summaries (read the CSVs this VM just wrote) with explicit provenance.
  MACHINE_TYPE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/machine-type 2>/dev/null | awk -F/ '{print $NF}')
  run_as_bench "
    source \"\$HOME/.local/bin/env\"
    cd '$REPO_DIR'
    uv run python3 paper/experiments/scripts/summarize_scenario.py --arch '$ARCH_LABEL' \
      --platform-note 'GCE $MACHINE_TYPE, Ubuntu 24.04 LTS, SMT off, core-pinned (taskset -c 0)'
  "
  log "Rebuttal summaries written"
else
# Clear stale CSVs from prior runs so sweep_bench starts fresh.
find "$REPO_DIR/paper/experiments/data" -name "grid*_results*.csv" -delete 2>/dev/null
log "Cleared stale result CSVs"

# Run sweep pinned to core 0 — all methods in one Rust binary.
run_as_bench "
  source \"\$HOME/.cargo/env\"
  cd '$REPO_DIR'
  taskset -c 0 ./target/release/sweep_bench \
    '$REPO_DIR/paper/experiments/artifacts' \
    --grid all \
    --output-dir '$REPO_DIR/paper/experiments/data' \
    --warmup 3 --iters 21 --min-iters 11 \
    --max-time-secs 30 \
    $EXT_FLAGS
"
log "Sweep complete (all methods via sweep_bench)"

# Copy with architecture-free name for backward compatibility.
for csv in grid1_results; do
  SRC="$REPO_DIR/paper/experiments/data/$${csv}_$${ARCH_LABEL}.csv"
  DST="$REPO_DIR/paper/experiments/data/$${csv}.csv"
  [ -f "$SRC" ] && cp "$SRC" "$DST" && log "  $${csv}_$${ARCH_LABEL} → $csv"
done

fi

# ---------------------------------------------------------------------------
# Phase 12: Collect results
# ---------------------------------------------------------------------------
phase 12 "Collect results"
cp "$REPO_DIR/paper/experiments/data/"*.csv "$RESULTS_DIR/" 2>/dev/null || true
cp "$REPO_DIR/paper/experiments/data/"*.md "$RESULTS_DIR/" 2>/dev/null || true
cp "$REPO_DIR/paper/experiments/data/"*.log "$RESULTS_DIR/" 2>/dev/null || true

cat > "$RESULTS_DIR/system_info.txt" <<SYSINFO
git_ref: $GIT_REF
timestamp: $(date -Iseconds)
machine_type: $(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/machine-type 2>/dev/null | awk -F/ '{print $NF}' || echo unknown)
zone: $(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | awk -F/ '{print $NF}' || echo unknown)
cpu: $(lscpu 2>/dev/null | grep "Model name" | sed 's/.*: *//' || echo unknown)
rust: $(run_as_bench "source \"\$HOME/.cargo/env\" && cd '$REPO_DIR' && rustc --version")
llvm: $(llc --version 2>&1 | head -1 || echo "not installed")
lightgbm: $LIGHTGBM_VER (source, cmake Release -march=native)
xgboost: $XGBOOST_VER (source, cmake Release -march=native)
arch_label: $ARCH_LABEL
bench_suite: $BENCH_SUITE
protocol: $([ "$BENCH_SUITE" = "rebuttal" ] && echo "sweep_bench --grid scen + prepare_chunked.py --stage timing (E1+E2 rebuttal)" || echo "sweep_bench --grid all (per-group Instant timing, all methods in Rust)")
seed: 42
SYSINFO

chown -R $BENCH_USER:$BENCH_USER "$RESULTS_DIR"
echo "$(date -Iseconds)" > "$RESULTS_DIR/DONE"
log "Results in $RESULTS_DIR"
ls -la "$RESULTS_DIR/"

# ---------------------------------------------------------------------------
# Phase 13: Upload results to GCS
# ---------------------------------------------------------------------------
phase 13 "Upload results to GCS"
gcloud storage cp -r "$RESULTS_DIR/*" "$GCS_RESULTS_BASE/"
log "Results uploaded to $GCS_RESULTS_BASE/"
log "===== ALL PHASES COMPLETE ====="
