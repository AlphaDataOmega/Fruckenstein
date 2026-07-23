"""Frozen entry point for the open-floor relational flock."""

from pathlib import Path
import os
import sys


bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
os.environ.setdefault("DUCK_OPEN_DUCK", str(bundle / "Open_Duck_Playground"))
# This artifact is specifically the visible, fully crowned flock.  Do not let
# a diagnostic environment inherited from a prior headless test redefine it.
os.environ["WATCH"] = "1"
os.environ["GROUP_CROWNS"] = "1"
os.environ["GROUP_CROWN_PERIOD"] = "5"
os.environ["GROUP_MEMORY"] = "1"

import group_ducks  # noqa: E402

group_ducks.main()
