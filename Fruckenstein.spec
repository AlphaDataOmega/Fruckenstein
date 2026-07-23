# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

ROOT = Path.cwd()
SRC = ROOT / "src" / "fruckenstein"
HANDOFF = ROOT / "seat" / "crown-seat-candidate-0003-handoff"
SEAT_DEST = Path("seat") / "crown-seat-candidate-0003-handoff"

datas = [
    (str(SRC / "instinct_repertoire.npz"), "."),
    (
        str(ROOT / "duck" / "Open_Duck_Playground" / "playground" / "open_duck_mini_v2" / "xmls"),
        str(Path("Open_Duck_Playground") / "playground" / "open_duck_mini_v2" / "xmls"),
    ),
    (str(HANDOFF / "handoff" / "SHA256SUMS"), str(SEAT_DEST / "handoff")),
    (
        str(HANDOFF / "frozen" / "crown-seat-mathematical-foundation-v1" / "source" / "crown_seat.py"),
        str(SEAT_DEST / "frozen" / "crown-seat-mathematical-foundation-v1" / "source"),
    ),
    (
        str(HANDOFF / "frozen" / "crown-seat-mathematical-foundation-v1" / "parameters-development.json"),
        str(SEAT_DEST / "frozen" / "crown-seat-mathematical-foundation-v1"),
    ),
    (
        str(HANDOFF / "frozen" / "crown-cell-axial-spindle-foundation-v1" / "phase0-geometry" / "geometry-manifest.json"),
        str(SEAT_DEST / "frozen" / "crown-cell-axial-spindle-foundation-v1" / "phase0-geometry"),
    ),
]

binaries = []
hiddenimports = [
    "mujoco.experimental.studio.native_viewer",
    "group_ducks",
    "relational_duck",
    "full_body_udi",
    "seat",
    "voxels",
    "constants",
    "crown_seat",
]

mujoco_bundle = collect_all("mujoco")
datas += mujoco_bundle[0]
binaries += mujoco_bundle[1]
hiddenimports += mujoco_bundle[2]

a = Analysis(
    [str(SRC / "group_ducks_standalone.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["jax"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Fruckenstein-Windows-x64",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
