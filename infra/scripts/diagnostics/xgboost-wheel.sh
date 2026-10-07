# Diagnostic: XGBoost native from our source build (-march=native) against the
# official xgboost-cpu wheel's library (generic x86-64 or aarch64), each timed
# alone, in alternating processes (A B B A, four rounds), on the XGBoost cells of
# the acceptance and ablation suites. Sourced by startup.sh in Phase 7, after
# preflight, with its helpers and variables.

WHEEL_LIB=$(as_bench "uv run --group baselines python -c 'import os, xgboost; \
print(os.path.join(os.path.dirname(xgboost.__file__), \"lib\", \"libxgboost.so\"))'" | tail -1)
log "wheel library: $WHEEL_LIB ($(sha256sum "$WHEEL_LIB" | cut -c1-12))"
for suite in acceptance ablation; do
  as_bench "uv run --group baselines python infra/scripts/diagnostics/swap_xgboost_lib.py \
    experiments/artifacts/manifests/$suite.json experiments/artifacts/manifests/$suite-wheel.json \
    '$WHEEL_LIB' 'xgboost-cpu wheel'"
done

set -f
specs="acceptance:credit/nt500_md4/xgboost/* acceptance:support/nt500_md4_h16/xgboost/*
  ablation:flchain/nt500_md8_h16/xgboost/* ablation:expedia/nt500_md8/xgboost/*"
for round in 1 2 3 4; do
  pos=0
  for build in native wheel wheel native; do
    pos=$((pos + 1))
    i=0
    for spec in $specs; do
      i=$((i + 1))
      suite=${spec%%:*}
      glob=${spec#*:}
      manifest=experiments/artifacts/manifests/$suite.json
      [ "$build" = wheel ] && manifest=experiments/artifacts/manifests/$suite-wheel.json
      as_bench "taskset -c 0 target/release/sweep_bench run $manifest \
        --output-dir experiments/data/runs/xgbwheel-$round-$pos-$build-$i-$ARCH_LABEL \
        --cells '$glob' --only xgboost_native --system-info '$SYSTEM_INFO' --require-pmu" \
        > /dev/null || log "xgbwheel $round $pos $build $i: failed"
    done
  done
done
set +f
