# Build and reproduction

## Requirements

- Windows 10/11 or another platform supported by MuJoCo
- Python 3.11+
- a GPU is optional

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Visible flock

```powershell
.\.venv\Scripts\python src\fruckenstein\group_ducks_standalone.py
```

Memory defaults to `src/fruckenstein/RelationalDuckGroupData/` for a source run.

## Headless lifetime

```powershell
$env:WATCH = "0"
$env:N = "3600"
.\.venv\Scripts\python src\fruckenstein\group_ducks.py
```

Useful environment variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `WATCH` | `1` | Open the viewer when `1`; run headless when `0`. |
| `N` | `1200` | Headless control-step count. |
| `GROUP_CROWNS` | `1` | Enable Candidate-0003 crowns. |
| `GROUP_CROWN_PERIOD` | `5` | Fast-loop steps per crown update. |
| `GROUP_MEMORY` | `1` | Load and save body memory. |
| `GROUP_LEDGER_DIR` | automatic | Override the ledger directory. |

## Generational bake

```powershell
New-Item -ItemType Directory -Force work\ledger,work\run | Out-Null
.\.venv\Scripts\python src\fruckenstein\group_duck_generations.py `
  --generations 3 `
  --candidates 3 `
  --steps 3600 `
  --seed 1701 `
  --crown-period 5 `
  --ledger work\ledger `
  --run-dir work\run
```

The generational runner uses separate processes for candidate flocks and promotes only an eligible winner.

## Build the Windows executable

```powershell
.\.venv\Scripts\python -m pip install pyinstaller
.\.venv\Scripts\pyinstaller --noconfirm --clean Fruckenstein.spec
```

Expected output:

`dist/Fruckenstein-Windows-x64.exe`

The executable is windowed and intentionally contains the MuJoCo runtime, body XML/meshes, frozen crown source/parameters/geometry, voxel controller, and gait repertoire. It is not code-signed.

## Verify the published build

```powershell
.\.venv\Scripts\python scripts\verify_release.py
```
