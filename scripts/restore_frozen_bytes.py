from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "seat" / "crown-seat-candidate-0003-handoff"
PATHS = (
    "frozen/crown-seat-mathematical-foundation-v1/source/crown_seat.py",
    "frozen/crown-seat-mathematical-foundation-v1/parameters-development.json",
    "frozen/crown-cell-axial-spindle-foundation-v1/phase0-geometry/geometry-manifest.json",
)


for relative in PATHS:
    repository_path = (Path("seat") / "crown-seat-candidate-0003-handoff" / relative).as_posix()
    content = subprocess.check_output(["git", "cat-file", "blob", f"HEAD:{repository_path}"], cwd=ROOT)
    destination = HANDOFF / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    print(f"{hashlib.sha256(content).hexdigest()}  {relative}")
