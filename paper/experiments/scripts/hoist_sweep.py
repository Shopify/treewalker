#!/usr/bin/env python3
"""Run hoist_bench serially on prepared artifacts; keep inputs read-only.

No Python dependencies. Legacy configuration names are translated into copies
under the output directory. Every subprocess must pass hoist_bench's full-walk
correctness gate before it reports timings. Raw JSON/logs and an incremental CSV
are retained, including failed cells. No models or input rows are copied.
"""

import argparse
import csv
import fnmatch
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


ALIASES = {
    "max_group_width": "horizon",
    "varying_features": "time_varying_features",
    "mono_inc_features": "monotonic_increasing_features",
    "mono_dec_features": "monotonic_decreasing_features",
}


def normalize_config(source):
    config = json.loads(source.read_text())
    for current, legacy in ALIASES.items():
        if legacy in config:
            old_value = config.pop(legacy)
            if current in config and config[current] != old_value:
                raise ValueError(f"{source}: conflicting {current}/{legacy}")
            config[current] = old_value
        if current not in config:
            raise ValueError(f"{source}: missing {current}")
    return config


def discover(args):
    for source in sorted(args.artifacts.glob("*/*/**/walker_config.json")):
        directory = source.parent
        relative = directory.relative_to(args.artifacts)
        if args.dataset and relative.parts[0] not in args.dataset:
            continue
        if args.cell and not any(fnmatch.fnmatch(str(relative), p) for p in args.cell):
            continue
        # The ranking preparation keeps the same empirical inputs in both places.
        if (directory / "empirical/walker_config.json").exists():
            continue
        data = directory / "test_data.bin"
        if not data.is_file():
            continue
        for framework in args.framework:
            model = next(
                (
                    parent / framework / f"model_treelite.{suffix}"
                    for parent in [directory, directory.parent]
                    for suffix in ["bin", "json"]
                    if (parent / framework / f"model_treelite.{suffix}").is_file()
                ),
                None,
            )
            if model is not None:
                yield relative, framework, model, source, data


def flatten(report):
    row = {k: report[k] for k in [
        "groups", "rows", "trees", "prefix_depth", "tree_ordering",
        "max_prediction_delta", "speedup",
    ]}
    row.update(report["hoisting"])
    for mode in report["modes"]:
        prefix = mode["mode"] + "_"
        row.update({prefix + k: v for k, v in mode.items() if k not in ["mode", "work"]})
        row.update({prefix + k: v for k, v in mode["work"].items()})
    return row


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path, help="new directory for this run")
    parser.add_argument("--binary", type=Path, default=repo / "benchmarks/target/release/hoist_bench")
    parser.add_argument("--dataset", action="append", help="repeat to select datasets")
    parser.add_argument("--cell", action="append", help="repeatable glob, e.g. '*/nt500_md8_h16'")
    parser.add_argument("--framework", action="append", choices=["lightgbm", "xgboost"])
    parser.add_argument("--mode", action="append", choices=["default", "isolated"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--min-blocks", type=int, default=11)
    parser.add_argument("--max-blocks", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    args.artifacts = args.artifacts.resolve()
    args.binary = args.binary.resolve(strict=True)
    args.framework = args.framework or ["lightgbm", "xgboost"]
    args.mode = args.mode or ["default", "isolated"]
    if args.repeats < 1 or args.min_blocks < 11 or args.max_blocks < args.min_blocks or args.warmup < 0:
        parser.error("require repeats >= 1, max-blocks >= min-blocks >= 11, warmup >= 0")
    cells = list(discover(args))
    if not cells:
        parser.error("no matching artifact cells")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "configs").mkdir()
    manifest = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "binary": str(args.binary),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "argv": sys.argv,
        "cells": len(cells),
        "runs": [],
    }
    manifest_path = args.output / "manifest.json"
    with (args.output / "summary.csv").open("w", newline="") as csv_file:
        writer = None
        # Separate rounds reduce the chance all repetitions see the same transient load.
        for repeat in range(1, args.repeats + 1):
            for relative, framework, model, source, data in cells:
                stem = "__".join(relative.parts)
                config = args.output / "configs" / f"{stem}.json"
                config.write_text(json.dumps(normalize_config(source), indent=2) + "\n")
                for mode in args.mode:
                    name = f"{stem}__{framework}__{mode}__r{repeat}"
                    command = [str(args.binary), str(model), str(config), str(data)]
                    offsets = data.parent / "group_offsets.bin"
                    if offsets.exists():
                        command.extend(["--group-offsets", str(offsets)])
                    if mode == "isolated":
                        command.extend(["--no-tree-ordering", "--prefix-depth", "0"])
                    command.extend([
                        "--min-blocks", str(args.min_blocks), "--max-blocks", str(args.max_blocks),
                        "--warmup", str(args.warmup),
                    ])
                    started = time.monotonic()
                    raw_path = args.output / f"{name}.json"
                    print(f"RUN {name}", flush=True)
                    with raw_path.open("w") as stdout, (args.output / f"{name}.log").open("w") as stderr:
                        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
                    run = {
                        "name": name, "command": command, "source_config": str(source),
                        "returncode": result.returncode, "wall_seconds": time.monotonic() - started,
                    }
                    manifest["runs"].append(run)
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    if result.returncode:
                        print(f"FAIL {name}: exit {result.returncode}; see log", flush=True)
                        continue
                    report = json.loads(raw_path.read_text())
                    row = {"cell": str(relative), "framework": framework, "mode": mode, "repeat": repeat}
                    row.update(flatten(report))
                    if writer is None:
                        writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    csv_file.flush()
                    h = report["hoisting"]
                    print(
                        f"{name}: {report['speedup']:.3f}x, swaps={h['swaps']}, "
                        f"roots={h['varying_roots_before']}->{h['varying_roots_after']}, "
                        f"delta={report['max_prediction_delta']:.2e}", flush=True,
                    )
    failures = sum(run["returncode"] != 0 for run in manifest["runs"])
    print(f"{len(manifest['runs'])} runs, {failures} failures; {args.output / 'summary.csv'}")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
