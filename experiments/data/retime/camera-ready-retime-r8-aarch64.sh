#!/bin/sh
# Re-time 100 cells of camera-ready-aarch64 with 8 rounds, as the campaign runs: core 0, the PMU required, the system facts recorded.
set -e
exec uv run --group baselines treewalker-exp retime-run /home/bench/treewalker/experiments/data/retime/camera-ready-retime-r8-aarch64.json --run-id camera-ready-retime-r8-aarch64 --parent camera-ready-aarch64 --core 0 --require-pmu ${SYSTEM_INFO:+--system-info "$SYSTEM_INFO"}
