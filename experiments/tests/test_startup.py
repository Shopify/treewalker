"""The campaign VM's startup script: its terminal markers, restarts and first boot."""

import getpass
import re
import shutil
import subprocess
import sys
from pathlib import Path

from treewalker_exp import pack

INFRA = Path(__file__).parents[2] / "infra"


def startup() -> str:
    return (INFRA / "scripts" / "startup.sh").read_text()


def function(name: str) -> str:
    """One of startup.sh's functions, as it runs on the VM (the template's $$
    escapes undone)."""
    text = startup()
    start = text.index(f"{name}() {{")
    return text[start : text.index("\n}\n", start) + 3].replace("$${", "${")


def check_done(results):
    tf = (INFRA / "outputs.tf").read_text()
    m = re.search(r'output "check_done".*?--command \'([^\']*)\'', tf, re.S)
    assert m is not None
    cmd = m[1]
    cmd = cmd.replace("/home/bench/results", str(results))
    return subprocess.run(["sh", "-c", cmd], capture_output=True, text=True).stdout.strip()


def test_check_done_reports_an_error_before_done(tmp_path):
    """outputs.tf's check_done, run against a results directory: ERROR wins, so a
    failure reads as an error even beside a stale DONE."""
    assert check_done(tmp_path) == "RUNNING"
    (tmp_path / "DONE").write_text("2026-10-11T00:00:00\n")
    assert check_done(tmp_path) == "2026-10-11T00:00:00"
    (tmp_path / "ERROR").write_text("Compiling the baselines failed\n")
    assert check_done(tmp_path) == "Compiling the baselines failed"


def test_a_failure_replaces_done(tmp_path):
    (tmp_path / "DONE").write_text("2026-10-11T00:00:00\n")
    script = (
        f"RESULTS_DIR={tmp_path}\nBENCH_USER={getpass.getuser()}\n"
        'log() { echo "$*"; }\n' + function("fail") + 'fail "Compiling the baselines failed"\n'
    )
    assert subprocess.run(["bash", "-c", script], capture_output=True).returncode == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ERROR"]
    assert check_done(tmp_path) == "Compiling the baselines failed"


def test_a_new_boot_retires_the_previous_marker_into_history(tmp_path):
    (tmp_path / "ERROR").write_text("an earlier failure\n")
    script = (
        f"RESULTS_DIR={tmp_path}\nGCS_RESULTS_BASE=gs://b/results/x\n"
        f"BENCH_USER={getpass.getuser()}\n"
        'gcloud() { echo "gcloud $*" >> "$RESULTS_DIR/calls"; }\n'
        + function("retire_markers")
        + "retire_markers\n"
    )
    assert subprocess.run(["bash", "-c", script]).returncode == 0
    assert check_done(tmp_path) == "RUNNING"
    assert "ERROR, retired at a boot on" in (tmp_path / "MARKER_HISTORY").read_text()
    assert "an earlier failure" in (tmp_path / "MARKER_HISTORY").read_text()
    calls = (tmp_path / "calls").read_text()
    assert "storage rm --quiet gs://b/results/x/DONE gs://b/results/x/ERROR" in calls


def claim_script(tmp_path):
    """startup.sh's claim_run, with log and fail stubbed and run_state answered by
    `pack.run_state`."""
    state = (
        f'run_state() {{ "{sys.executable}" -c "import sys, pathlib; '
        'from treewalker_exp import pack; print(pack.run_state(pathlib.Path(sys.argv[1])))" '
        '"$REPO_DIR/experiments/data/runs/$1"; }\n'
    )
    return (
        f'REPO_DIR={tmp_path}/repo\nCLAIMS={tmp_path}/claims.txt\ntouch "$CLAIMS"\n'
        'log() { echo "$*"; }\nfail() { echo "FAIL: $*"; exit 1; }\n'
        + state
        + function("claim_run")
    )


def packed(run):
    run.mkdir(parents=True, exist_ok=True)
    (run / "cells.json").write_text("{}")
    (run / "samples.parquet").write_bytes(b"x")


def raw(run):
    (run / "cells" / "a_cell").mkdir(parents=True, exist_ok=True)


def test_claim_run_fresh_resumed_and_refused(tmp_path):
    runs = tmp_path / "repo" / "experiments" / "data" / "runs"
    runs.mkdir(parents=True)
    prelude = claim_script(tmp_path)

    def boot(*calls):
        return subprocess.run(
            ["bash", "-c", prelude + "\n".join(calls)], capture_output=True, text=True
        )

    r = boot("claim_run camera-ready-x86_64", "claim_run layout-align64-x86_64")
    assert r.returncode == 0, r.stdout
    (runs / "camera-ready-x86_64").mkdir()
    (runs / "layout-align64-x86_64").mkdir()
    # A restart resumes both, without claiming them again.
    r = boot("claim_run camera-ready-x86_64", "claim_run layout-align64-x86_64")
    assert r.returncode == 0 and r.stdout.count("resuming") == 2
    assert (tmp_path / "claims.txt").read_text() == "camera-ready-x86_64\nlayout-align64-x86_64\n"
    # An existing directory this deployment never claimed is refused.
    (runs / "acceptance-x86_64").mkdir()
    r = boot("claim_run acceptance-x86_64")
    assert r.returncode == 1 and "was not claimed by this deployment" in r.stdout
    assert "acceptance" not in (tmp_path / "claims.txt").read_text()


def test_a_restart_skips_packed_runs_resumes_raw_ones_and_refuses_mixed(tmp_path):
    runs = tmp_path / "repo" / "experiments" / "data" / "runs"
    prelude = claim_script(tmp_path)
    parent, attempt = runs / "camera-ready-x86_64", runs / "camera-ready-retime-r8-x86_64"
    (tmp_path / "claims.txt").write_text(f"{parent.name}\n{attempt.name}\n")
    packed(parent)  # packed before the re-time phase
    raw(attempt)  # the re-time run was cut short

    def launch(rid):
        script = prelude + f"if claim_run {rid}; then echo LAUNCH; else echo SKIP; fi\n"
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    r = launch(parent.name)
    assert r.returncode == 0 and r.stdout.strip().endswith("SKIP") and "skipping" in r.stdout
    r = launch(attempt.name)
    assert r.returncode == 0 and r.stdout.strip().endswith("LAUNCH") and "resuming" in r.stdout
    # Packed after it finishes, the re-time run is skipped too on a later restart.
    shutil.rmtree(attempt / "cells")
    packed(attempt)
    assert launch(attempt.name).stdout.strip().endswith("SKIP")
    # Raw cells of a later run inside a packed directory: refused, nothing deleted.
    raw(parent)
    r = launch(parent.name)
    assert r.returncode == 1 and "holds packed tables beside raw cells" in r.stdout
    assert (parent / "cells" / "a_cell").exists() and (parent / "cells.json").exists()
    assert pack.run_state(parent) == "mixed"


def test_nothing_under_the_bench_home_exists_before_the_bench_user():
    """First boot: no top-level command before useradd touches the bench home (a
    directory created there as root makes useradd keep a root-owned home, and the
    bench user cannot install its tools), nor calls a helper that does; the
    markers are retired only after the results directory is created and chowned."""
    lines = startup().replace("$${", "${").split("\n")
    bodies: dict[str, list[str]] = {}
    top: list[tuple[int, str]] = []
    name = None
    for i, line in enumerate(lines):
        one = re.match(r"^([A-Za-z_]+)\(\) \{(.*)\}\s*$", line)
        m = re.match(r"^([A-Za-z_]+)\(\) \{\s*$", line)
        if one and name is None:
            bodies[one[1]] = [one[2]]
        elif m:
            name = m[1]
            bodies[name] = []
        elif name is not None:
            if line.startswith("}"):
                name = None
            else:
                bodies[name].append(line)
        else:
            top.append((i, line))
    home = re.compile(r"\$\{?(BENCH_HOME|RESULTS_DIR|REPO_DIR|SYSTEM_INFO|CLAIMS)\b")
    touching = {n for n, body in bodies.items() if any(home.search(b) for b in body)}
    # A helper that calls one that touches the home touches it too.
    for _ in range(3):
        touching |= {
            n
            for n, body in bodies.items()
            if any(set(re.findall(r"[A-Za-z_]+", b)) & touching for b in body)
        }
    assert {"fail", "retire_markers", "as_bench", "twx", "claim_run"} <= touching
    useradd = next(i for i, line in top if "useradd" in line)
    for i, line in top:
        code = line.split("#", 1)[0].strip()
        if i >= useradd or not code or code.startswith("trap "):
            continue
        assignment = re.match(r"^[A-Z_]+=", code)
        assert assignment or not home.search(code), f"line {i + 1}: {line}"
        called = set(re.findall(r"[A-Za-z_]+", code)) & touching
        assert not called, f"line {i + 1} calls {called} before the bench user exists"
    chown = next(
        i for i, line in top if line.strip() == 'chown $BENCH_USER:$BENCH_USER "$RESULTS_DIR"'
    )
    retire = next(i for i, line in top if line.strip() == "retire_markers")
    assert useradd < chown < retire
    home_chown = next(
        i for i, line in top if line.strip() == 'chown $BENCH_USER:$BENCH_USER "$BENCH_HOME"'
    )
    assert useradd < home_chown < retire


def block(start: str, end: str) -> str:
    """The startup.sh text from ``start`` to the first ``end`` after it, as it runs
    on the VM (the template's $$ escapes undone)."""
    text = startup()
    i = text.index(start)
    return text[i : text.index(end, i) + len(end)].replace("$${", "${")


def state_stub() -> str:
    return (
        f'run_state() {{ "{sys.executable}" -c "import sys, pathlib; '
        'from treewalker_exp import pack; print(pack.run_state(pathlib.Path(sys.argv[1])))" '
        '"$REPO_DIR/experiments/data/runs/$1"; }\n'
    )


def test_every_failure_in_the_runs_phase_is_recorded():
    text = startup()
    runs = text[text.index('phase 7 "') : text.index('phase 8 "')]
    assert "|| log" not in runs  # each failure goes through failed_step
    # The diagnostics are sourced into the runs phase, so they record theirs too.
    for script in sorted((INFRA / "scripts" / "diagnostics").glob("*.sh")):
        assert "|| log" not in script.read_text(), script.name


def test_the_boot_ends_with_done_only_without_failures(tmp_path):
    marker = block('if [ -z "$FAILURES" ]; then', "\nfi\n")
    for failures, expect in (("", "DONE"), ("Run x-aarch64: some cells failed; ", "ERROR")):
        d = tmp_path / expect
        d.mkdir()
        script = f"RESULTS_DIR={d}\nFAILURES='{failures}'\n{marker}"
        assert subprocess.run(["bash", "-c", script]).returncode == 0
        assert sorted(p.name for p in d.iterdir()) == [expect]
    assert "Run x-aarch64: some cells failed" in check_done(tmp_path / "ERROR")


def test_an_interrupted_run_is_not_packed_before_it_is_resumed(tmp_path):
    """The pack loop before the re-time phase packs a run an invocation finished,
    leaves raw one that an earlier boot cut short, and leaves every re-time run to
    the re-time phase, which resumes it first: an older invocation of a re-time
    run that was cut short on its retry would otherwise read as finished."""
    runs = tmp_path / "repo" / "experiments" / "data" / "runs"
    done = runs / "camera-ready-aarch64"
    cut = runs / "camera-ready-ablation-1-aarch64"
    retime = runs / "camera-ready-retime-r8-aarch64"
    for run in (done, cut, retime):
        (run / "cells" / "a_cell").mkdir(parents=True)
    (done / "run.json").write_text('{"invocations": [{"finished": 1}]}')
    (cut / "run.json").write_text("{}")  # the runner writes an invocation only at its end
    (retime / "run.json").write_text('{"invocations": [{"finished": 1}]}')  # an older one
    loop = block('for run in "$REPO_DIR"/experiments/data/runs/*/; do', "\ndone\n")
    script = (
        f"REPO_DIR={tmp_path}/repo\n"
        + state_stub()
        + f'twx() {{ echo "$*" >> {tmp_path}/calls; }}\nfailed_step() {{ :; }}\n'
        + loop
    )
    assert subprocess.run(["bash", "-c", script]).returncode == 0
    calls = (tmp_path / "calls").read_text()
    assert str(done) in calls and str(cut) not in calls and str(retime) not in calls
    assert pack.run_state(cut) == "partial" and pack.run_state(retime) == "raw"


def test_a_claimed_run_not_complete_at_the_end_is_a_failure(tmp_path):
    """A restart that skips an earlier boot's packed run must still see its
    unresolved failure: every claimed run is checked with run-complete."""
    runs = tmp_path / "repo" / "experiments" / "data" / "runs"
    ok, failing = runs / "camera-ready-aarch64", runs / "camera-ready-ablation-1-aarch64"
    for run in (ok, failing):
        packed(run)
        (run / "cells.json").write_text('{"c0": {}}')
        (run / "run.json").write_text('{"manifest": {"cells": 1}}')
    (failing / "failed").mkdir()
    (failing / "failed" / "c.json").write_text('{"id": "c"}')  # never finished
    claims = "camera-ready-aarch64\ncamera-ready-ablation-1-aarch64\nscenario-v1-aarch64\n"
    (tmp_path / "claims.txt").write_text(claims)
    check = block("while read -r rid; do", '\ndone < "$CLAIMS"\n')
    complete = (
        f'run_complete() {{ "{sys.executable}" -c "import sys, pathlib; '
        "from treewalker_exp import check_run as cr; "
        "p = cr.completeness(pathlib.Path(sys.argv[1])); print('; '.join(p)); "
        'sys.exit(1 if p else 0)" "$REPO_DIR/experiments/data/runs/$1"; }\n'
    )
    script = (
        f"REPO_DIR={tmp_path}/repo\nCLAIMS={tmp_path}/claims.txt\nFAILURES=''\n"
        + complete
        + 'failed_step() { FAILURES="$FAILURES$*; "; }\n'
        + check
        + 'echo "$FAILURES"\n'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout.strip()
    assert out == (
        "Run camera-ready-ablation-1-aarch64: incomplete: failed cell c.json ; "
        "Run scenario-v1-aarch64: incomplete: absent, not packed ;"
    )


def test_the_precision_diagnostic_reads_both_campaign_runs_and_restores_the_wheel(tmp_path):
    repo, home, results = tmp_path / "repo", tmp_path / "home", tmp_path / "results"
    native = repo / "experiments/.cache/native/xgboost-3.4.1-x/libxgboost.so"
    wheel = tmp_path / "venv/xgboost/lib/libxgboost.so"
    for f, text in ((native, "native"), (wheel, "wheel")):
        f.parent.mkdir(parents=True)
        f.write_text(text)
    calls = tmp_path / "calls"
    stubs = f"""
set -Eeuo pipefail
REPO_DIR={repo}; BENCH_HOME={home}; RESULTS_DIR={results}; BENCH_USER={getpass.getuser()}
ARCH_LABEL=x86_64; FAILURES=""
log() {{ :; }}
fail() {{ echo "FAIL: $*"; exit 1; }}
failed_step() {{ FAILURES="$FAILURES$*; "; }}
sha256sum() {{ echo "sha  $1"; }}
chown() {{ :; }}
gcloud() {{ echo "gcloud $4" >> {calls}; touch "$5/$(basename "$4")"; }}
as_bench() {{
  echo "as_bench $*" >> {calls}
  case "$*" in *xgboost.__file__*) echo {wheel} ;; *natives*) echo campaignsha ;; esac
}}
twx() {{ echo "twx $* (lib: $(cat {wheel}))" >> {calls}; }}
"""
    script = (INFRA / "scripts/diagnostics/precision.sh").read_text()
    r = subprocess.run(
        ["bash", "-c", stubs + "DIAGNOSTIC_INPUT=\n" + script], capture_output=True, text=True
    )
    assert r.returncode == 1 and "needs diagnostic_input" in r.stdout
    tail = 'echo "FAILURES=$FAILURES"\n'
    r = subprocess.run(
        ["bash", "-c", stubs + "DIAGNOSTIC_INPUT=gs://b/results\n" + script + tail],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "FAILURES="
    log = calls.read_text()
    flat = " ".join(log.split())
    for m, a in (("intel", "x86_64"), ("arm", "aarch64")):
        for f in ("run.json", "cells.json", "stages.parquet"):
            assert f"gcloud gs://b/results/{m}/runs/camera-ready-{a}/{f}" in log
            assert (home / f"precision-input/runs/camera-ready-{a}/{f}").exists()
    assert "precision-order --manifest experiments/artifacts/manifests/camera-ready.json" in flat
    assert f"twx audit-f32 --suite camera-ready --runs {home}/precision-input/runs" in log
    assert "(lib: native)" in log  # the audit ran on the campaign's library
    assert wheel.read_text() == "wheel"  # and the wheel's is back
    assert "campaignsha" in (results / "precision/xgboost_library.txt").read_text()
