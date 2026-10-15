#!/bin/sh
# Re-time 2 cells of camera-ready-x86_64 with 14 rounds, as the campaign runs: core 0, the PMU required, the system facts recorded.
set -e
exec uv run --group baselines treewalker-exp retime-run /home/bench/treewalker/experiments/data/retime/camera-ready-retime-r14-x86_64.json --run-id camera-ready-retime-r14-x86_64 --parent camera-ready-x86_64 --core 0 --require-pmu ${SYSTEM_INFO:+--system-info "$SYSTEM_INFO"}
