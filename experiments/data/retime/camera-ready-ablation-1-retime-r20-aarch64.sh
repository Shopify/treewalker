#!/bin/sh
# Re-time 4 cells of camera-ready-ablation-1-aarch64 with 20 rounds, as the campaign runs: core 0, the PMU required, the system facts recorded.
set -e
exec uv run --group baselines treewalker-exp retime-run /home/bench/treewalker/experiments/data/retime/camera-ready-ablation-1-retime-r20-aarch64.json --run-id camera-ready-ablation-1-retime-r20-aarch64 --parent camera-ready-ablation-1-aarch64 --core 0 --require-pmu ${SYSTEM_INFO:+--system-info "$SYSTEM_INFO"}
