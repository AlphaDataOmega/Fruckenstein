from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFO = json.loads((ROOT / "BUILD-INFO.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


release = ROOT / INFO["release"]["path"]
actual = sha256(release)
expected = INFO["release"]["sha256"]
if actual != expected:
    raise SystemExit(f"FAILED: {actual} != {expected}")

print(f"OK: {release.name}")
print(actual)
