#!/bin/sh
# Re-time 1 cells of scenario-v1-x86_64 with 8 rounds, as the campaign runs: core 0, the PMU required, the system facts recorded.
set -e
exec uv run --group baselines treewalker-exp retime-run /home/bench/treewalker/experiments/data/retime/scenario-v1-retime-r8-x86_64.json --run-id scenario-v1-retime-r8-x86_64 --parent scenario-v1-x86_64 --core 0 --require-pmu ${SYSTEM_INFO:+--system-info "$SYSTEM_INFO"}
