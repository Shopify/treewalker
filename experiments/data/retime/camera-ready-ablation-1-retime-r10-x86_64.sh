#!/bin/sh
# Re-time 1 cells of camera-ready-ablation-1-x86_64 with 10 rounds, as the campaign runs: core 0, the PMU required, the system facts recorded.
set -e
exec uv run --group baselines treewalker-exp retime-run /home/bench/treewalker/experiments/data/retime/camera-ready-ablation-1-retime-r10-x86_64.json --run-id camera-ready-ablation-1-retime-r10-x86_64 --parent camera-ready-ablation-1-x86_64 --core 0 --require-pmu ${SYSTEM_INFO:+--system-info "$SYSTEM_INFO"}
