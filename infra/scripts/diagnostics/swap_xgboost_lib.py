"""Copy an execution manifest with its XGBoost library replaced, by path and hash.

Usage: swap_xgboost_lib.py <manifest> <out> <libxgboost.so> <label>

sweep_bench loads the XGBoost library the manifest names and checks its hash, so
a copy that names another build times that build with everything else unchanged.
"""

import hashlib
import json
import sys
from pathlib import Path

src, dst, lib, label = sys.argv[1:5]
doc = json.loads(Path(src).read_text())
digest = hashlib.sha256(Path(lib).read_bytes()).hexdigest()
doc["native"]["xgboost"] = {
    **doc["native"]["xgboost"],
    "path": lib,
    "sha256": digest,
    "build": {"source": label},
}
Path(dst).write_text(json.dumps(doc, indent=1))
print(f"{dst}: xgboost {lib} ({digest[:12]})")
