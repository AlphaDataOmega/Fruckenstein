# Crown Seat Candidate-0003 — Independent Conformance Report

Reproduced by Claude (Fable), 2026-07-16, per handoff README required order.
Handoff: `crown-seat-candidate-0003-handoff` (sealed conformance handoff;
NOT duck-control authorization; candidate's own status remains
public-pre-freeze-review-not-frozen).

## Environment

- Declared: Python 3.12.13, NumPy 2.3.5, Windows, no network/GPU/SciPy.
- Used: Python 3.12.11 (closest available; patch-level delta disclosed),
  NumPy 2.3.5 exact, Linux (WSL2). Cross-OS, cross-patch reproduction.
- Deviation handled: archive transfer stripped exec bits on the three
  handoff scripts; restored to the manifest's own declared mode 0755
  (`deterministicMetadata.directoryAndShellScriptMode`). Zero content bytes
  changed anywhere; all hashes verified before and during every run.

## 1. Package verification — PASS

- Archive: `crown-seat-candidate-0003-handoff.tar.gz` sha256
  `62e888163b699bb70472d09c4566dda1652fc1758fd6428f069dd0e487ddfccf` — OK
  against its .sha256 record.
- Handoff checksum root (sha256 of SHA256SUMS):
  `3d4fcad7f0eb24ada04cda8c4ab2cf3523bb5d2d28e87bfc95fd0939a0cbee97` —
  matches HANDOFF-ROOT-SHA256.txt.
- Handoff files verified: 1480/1480.
- Candidate package files verified: **1448/1448** (path, size, sha256; file
  set exact — no extras, no missing).
- Package content hash unchanged:
  `a52433bcd17ae670b05ea2753007050ba49e4856c7100cd69bbda0e7d2f2e3fb`
- Reference implementation hash unchanged:
  `897204605fe26cb0917503a4756c3a41701ff68424f238678d6a88e58e2928fa`
- Candidate law + lineage hash unchanged:
  `826d8ab098d069732e63b40eebfc3de1354f3bee68116f0abd3b791e4cbc5dcf`
- Frozen Phase 0 geometry manifest unchanged:
  `b1046340489a76a8ea48c6d3c8c83c16d12720664630d49d1784431e0ed7b31f`
- Frozen Phase 0 freeze manifest unchanged:
  `d37f929a142717b68ea90a47d31dbf0edead06d5a089bb9c075a17814c81ec71`

## 2. Nine tests — 9/9 PASS (16.4 s)

deterministic_vectors_reproduce · energy_ledger_closes ·
frozen_phase0_byte_hashes · impulsive_disturbance_is_explicitly_ledgered ·
overlap_transport_is_frozen_and_once_only · rotation_equivariance ·
state_is_ten_funnels_plus_three_axial_nodes · three_axis_dc_gain ·
zero_input_is_dissipative

## 3. Deterministic conformance vectors — 5/5 PASS

Declared absolute tolerance 2e-5. Measured max error on every field of
every vector (disposition, coherence, confidence, threeComponentState):
**0.0 — bit-identical reproduction across OS (Windows→Linux) and Python
patch version.** axis-x / axis-y / axis-z / oblique: resolved=true;
zero: resolved=false with disposition exactly [0,0,0] (resolved-gate
behavior confirmed).

## Restrictions honored

No byte below frozen/ modified; proposed interface untouched; raw/
three_component_state read only via the declared vector-harness diagnostic
comparison; no duck actuation or control wiring; no hidden-seed access;
coherence/confidence not reinterpreted as upstream evidence quality.
