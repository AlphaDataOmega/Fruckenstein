# Fruckenstein

Fruckenstein is a public research build of three physically embodied “crown ducks” learning and acting together in one MuJoCo world.

The ducks walk with a coordinated inherited gait, but steering, confidence, appetite, nociception, vitality, memory, and target choice are coupled through relational state. They collide with one another, can knock one another down, consume replenishing apples, preserve body-specific memory between runs, and can be tested through gated generations.

This repository contains:

- a reproducible Windows build published with each release;
- the controller source and body assets required to reproduce it;
- the frozen Candidate-0003 crown geometry and checksum boundary;
- the generational bake report;
- architecture, build, provenance, and safety notes.

## Run the Windows build

Download `Fruckenstein-Windows-x64.exe` from the
[latest GitHub release](https://github.com/AlphaDataOmega/Fruckenstein/releases/latest).

The application opens a non-headless MuJoCo window. Windows may warn because the executable is not code-signed. The build writes learned body memory beside the executable in `RelationalDuckGroupData/`.

Standard viewer interaction:

- drag to orbit the camera;
- right-drag to pan;
- use the mouse wheel to zoom;
- close the window to stop the simulation.

## What is creature and what is authored?

Authored:

- the articulated duck morphology and actuator map;
- a whole-gait repertoire;
- bounded safety ranges;
- the crown, voxel, ledger, and developmental update rules;
- appetite, nociception, vitality, and eligibility relations.

Developed through embodied consequence:

- target-relative steering trim;
- confidence and support for relational effects;
- which whole-gait perturbations work in a given body/context;
- persistent body-specific effect memory;
- which eligible candidate ledger is promoted between generations.

The build does **not** claim consciousness, emotion, subjective pain, biological life, or autonomous general intelligence. Terms such as hunger, pain, death, DNA, will, crown, and babble name operational relations in the simulation.

## Architecture

```text
body sensors and contact
        ↓
relational pressure + reliability
        ↓
null-preserving 27-voxel evidence field
        ↓
Candidate-0003 commit-or-refuse crown
        ↓
whole-gait enactment and bounded trim
        ↓
measured consequence → persistent body ledger
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the technical breakdown.

## Source run

Python 3.11+:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python src\fruckenstein\group_ducks_standalone.py
```

See [docs/BUILD.md](docs/BUILD.md) for headless tests, generational baking, and executable packaging.

## Current recorded bake

The included report records:

- 3 generations;
- 3 candidate flocks per generation;
- 27 duck lifetimes;
- 3 promoted winners;
- 3 selected apple consumptions;
- 0 selected falls;
- 0 deaths among selected winners.

These are project-reported simulation results under the recorded conditions, not independent certification or a general performance guarantee.

## Provenance and rights

Human conception and project direction: James Sterling Tuttle (“V”), Alpha Data Omega.

AI systems assisted with implementation, testing, analysis, packaging, and documentation as tools. They are not asserted as inventors or owners.

See [docs/PROVENANCE.md](docs/PROVENANCE.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[SHA256SUMS.txt](SHA256SUMS.txt).

Copyright © 2026 James Sterling Tuttle / Alpha Data Omega. All rights reserved
for original Fruckenstein materials. Public visibility does not grant a patent
license, commercial license, or permission to redistribute modified binaries.
Third-party components retain their own licenses.

## License

Apache License 2.0. **Attribution to AlphaDataOmega is required** for any use of the AlphaDataOmega portions, including research and commercial — see [NOTICE](NOTICE). Third-party components retain their own licenses (see THIRD_PARTY_NOTICES.md).
