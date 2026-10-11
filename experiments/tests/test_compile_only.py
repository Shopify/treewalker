"""Compile only fills the cache, or fails: the startup script's cache push and its
bucket check, run as they are on the VM (the template's $$ escapes undone)."""

import subprocess
from pathlib import Path

STARTUP = Path(__file__).parents[2] / "infra" / "scripts" / "startup.sh"
STUBS = 'log() { echo "$*"; }\nfail() { echo "FAIL: $*"; exit 1; }\n'


def text() -> str:
    return STARTUP.read_text().replace("$${", "${")


def run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_a_required_cache_push_that_fails_fails_the_boot():
    t = text()
    start = t.index("cache_push() {")
    push = t[start : t.index("\n}\n", start) + 3]
    for ok, required, code in ((False, "required", 1), (False, "", 0), (True, "required", 0)):
        gcloud = "gcloud() { return 0; }\n" if ok else "gcloud() { return 1; }\n"
        script = (
            "set -Eeuo pipefail\nGCS_CACHE_URI=gs://c\nREPO_DIR=/r\n"
            + STUBS
            + gcloud
            + push
            + f"cache_push compiled/m x {required}\n"
        )
        r = run(script)
        assert r.returncode == code, (ok, required, r.stdout)
        if not ok:
            assert ("FAIL: Cache: pushing" in r.stdout) == bool(required)


def test_compile_only_needs_a_cache_bucket():
    t = text()
    start = t.index('[ "$COMPILE_ONLY" != "true" ] || [ -n "$GCS_CACHE_URI" ]')
    check = t[start : t.index('lost with the VM"\n', start) + len('lost with the VM"\n')]
    for compile_only, bucket, code in (("true", "", 1), ("true", "gs://c", 0), ("false", "", 0)):
        script = f"set -Eeuo pipefail\nCOMPILE_ONLY={compile_only}\nGCS_CACHE_URI={bucket}\n"
        r = run(script + STUBS + check)
        assert r.returncode == code, (compile_only, bucket, r.stdout)
