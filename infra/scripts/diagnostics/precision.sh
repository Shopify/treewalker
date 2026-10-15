# Diagnostic: App. A.3's precision measurements for the camera-ready campaign, on
# the campaign's machine types, so that every value compared was produced on
# them. Deploy with suites = ["camera-ready"] and diagnostic_input set to the
# campaign's results (gs://BUCKET/results). Sourced by startup.sh in Phase 7,
# after preflight, with its helpers and variables; it writes to precision/.
#
# 1. sweep_bench precision-order over every LightGBM cell of the camera-ready
#    manifest: exact sums off, tree ordering on against off, and both orderings'
#    outputs against the GTIL reference that prepare computed on the trainer.
# 2. audit-f32 over the camera-ready suite: native XGBoost's raw margins from the
#    library this VM built the way the campaign's VM did, against the stage
#    oracle, beside the campaign's TreeWalker stages from both machines.

OUT="$RESULTS_DIR/precision"
INPUT="$BENCH_HOME/precision-input/runs"
[ -n "$DIAGNOSTIC_INPUT" ] \
  || fail "The precision diagnostic needs diagnostic_input: the campaign's results (gs://BUCKET/results)"
mkdir -p "$OUT" "$INPUT"
for machine in intel arm; do
  arch=x86_64
  [ "$machine" = arm ] && arch=aarch64
  run="camera-ready-$arch"
  mkdir -p "$INPUT/$run"
  for f in run.json cells.json stages.parquet; do
    gcloud storage cp --no-user-output-enabled "$DIAGNOSTIC_INPUT/$machine/runs/$run/$f" "$INPUT/$run/" \
      || failed_step "precision: fetching $run/$f failed"
  done
done
chown -R $BENCH_USER:$BENCH_USER "$OUT" "$BENCH_HOME/precision-input"

log "precision: tree ordering over the camera-ready LightGBM cells"
as_bench "target/release/sweep_bench precision-order \
  --manifest experiments/artifacts/manifests/camera-ready.json --output '$OUT/precision_order.json'" \
  > "$OUT/precision_order.log" 2>&1 || failed_step "precision: precision-order failed"

# The audit's native margins come from the XGBoost the runs timed, built here
# as on the campaign's VM, in place of the Python package's wheel library.
natives=$(ls "$REPO_DIR"/experiments/.cache/native/xgboost-*/libxgboost.so 2>/dev/null || true)
wheel=$(as_bench "uv run --group baselines python -c 'import os, xgboost; \
print(os.path.join(os.path.dirname(xgboost.__file__), \"lib\", \"libxgboost.so\"))'" | tail -1 || true)
if [ "$(echo "$natives" | wc -w)" -ne 1 ] || [ ! -f "$wheel" ]; then
  failed_step "precision: no single native XGBoost library ($natives) or wheel library ($wheel)"
else
  campaign=$(as_bench "uv run --group baselines python -c 'import json; \
print(json.load(open(\"$INPUT/camera-ready-$ARCH_LABEL/run.json\"))[\"natives\"][\"xgboost\"][\"sha256\"])'" | tail -1 || true)
  {
    echo "native (used): $natives $(sha256sum "$natives" | cut -d' ' -f1)"
    echo "campaign's native on this machine type: $campaign"
    echo "wheel (replaced for the audit): $wheel $(sha256sum "$wheel" | cut -d' ' -f1)"
  } > "$OUT/xgboost_library.txt"
  log "precision: $(head -2 "$OUT/xgboost_library.txt" | tr '\n' ' ')"
  cp "$wheel" "$wheel.wheel"
  cp "$natives" "$wheel"
  twx audit-f32 --suite camera-ready --runs "$INPUT" --out "$OUT" \
    > "$OUT/audit_f32.log" 2>&1 || failed_step "precision: audit-f32 failed"
  cp "$wheel.wheel" "$wheel"
fi
chown -R $BENCH_USER:$BENCH_USER "$OUT"
