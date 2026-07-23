"""UDI organ 5: frozen Candidate-0003 commitment seat, hosted read-only.
Byte-verifies the frozen implementation, geometry, and parameters against
the sealed handoff checksums at import. Control surface: resolved,
disposition, coherence, confidence, resolutionStrength — nothing else.
Refusal ([0,0,0]) passes through untouched. RK4 dt 0.02, pressure held per
step, energy ledger via the frozen simulate() stage-integration pattern.
Cold start is an explicit counted lifecycle event."""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # never deposit .pyc inside the frozen tree

_HOME_HANDOFF = Path.home() / "ado/seat/crown-seat-candidate-0003-handoff"
_REPO_HANDOFF = Path(__file__).resolve().parents[2] / "seat/crown-seat-candidate-0003-handoff"
_BUNDLE_HANDOFF = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "seat/crown-seat-candidate-0003-handoff"
HANDOFF = next(
    path for path in (_HOME_HANDOFF, _REPO_HANDOFF, _BUNDLE_HANDOFF)
    if path.exists()
)
PACKAGE = HANDOFF / "frozen/crown-seat-mathematical-foundation-v1"
PHASE0 = HANDOFF / "frozen/crown-cell-axial-spindle-foundation-v1/phase0-geometry"
SEAT_DT = 0.02
CLIP_NORM = 1.5

_VERIFY = [
    "frozen/crown-seat-mathematical-foundation-v1/source/crown_seat.py",
    "frozen/crown-seat-mathematical-foundation-v1/parameters-development.json",
    "frozen/crown-cell-axial-spindle-foundation-v1/phase0-geometry/geometry-manifest.json",
]


def _verify_frozen_bytes():
    sums = {}
    for line in (HANDOFF / "handoff/SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, rel = line.split(None, 1)
            sums[rel.strip()] = h
    for rel in _VERIFY:
        want = sums.get(rel)
        if want is None:
            raise RuntimeError(f"handoff checksum set lacks {rel}")
        got = hashlib.sha256((HANDOFF / rel).read_bytes()).hexdigest()
        if got != want:
            raise RuntimeError(f"FROZEN BYTES CHANGED: {rel}: {got} != {want}")


_verify_frozen_bytes()
sys.path.insert(0, str(PACKAGE / "source"))
from crown_seat import CrownGeometry, CrownSeat, SeatParameters  # noqa: E402

_GEOMETRY = CrownGeometry.load(PHASE0 / "geometry-manifest.json")
_PARAMS = SeatParameters.from_json(PACKAGE / "parameters-development.json")


class CommitmentSeat:
    def __init__(self):
        self.seat = CrownSeat(_GEOMETRY, _PARAMS, "coherent-local-frame", seed=0)
        self.state = self.seat.zeros()
        self.lifecycle_events = 0
        self.clip_events = 0
        self.steps = 0
        self.resolved_ticks = 0
        self.coherence_sum = 0.0
        self.confidence_sum = 0.0
        self.energy_ledger = {
            "externalWork": 0.0, "dampingLoss": 0.0, "couplingTransferLoss": 0.0,
            "turbulenceInput": 0.0, "impulsiveDisturbanceWork": 0.0,
        }
        self.energy_start = 0.0
        self.reset()

    def reset(self):
        self.state = self.seat.zeros()
        self.lifecycle_events += 1
        self.energy_start = self.seat.energy(self.state)

    def step(self, pressure3):
        p = np.asarray(pressure3, dtype=float)
        if not np.all(np.isfinite(p)):
            p = np.zeros(3)
        n = float(np.linalg.norm(p))
        clipped = n > CLIP_NORM
        if clipped:
            p = p * (CLIP_NORM / n)
            self.clip_events += 1
        s = self.state
        k1 = self.seat.rhs(s, p)
        p1 = self.seat.power_terms(s, p)
        s2 = s + 0.5 * SEAT_DT * k1
        k2 = self.seat.rhs(s2, p)
        p2 = self.seat.power_terms(s2, p)
        s3 = s + 0.5 * SEAT_DT * k2
        k3 = self.seat.rhs(s3, p)
        p3 = self.seat.power_terms(s3, p)
        s4 = s + SEAT_DT * k3
        k4 = self.seat.rhs(s4, p)
        p4 = self.seat.power_terms(s4, p)
        for key, out in (("externalWorkRate", "externalWork"),
                         ("dampingLossRate", "dampingLoss"),
                         ("couplingTransferLossRate", "couplingTransferLoss"),
                         ("turbulenceInputRate", "turbulenceInput")):
            self.energy_ledger[out] += SEAT_DT * (
                p1[key] + 2.0 * p2[key] + 2.0 * p3[key] + p4[key]) / 6.0
        self.state = s + SEAT_DT * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        out = self.seat.output(self.state)
        self.steps += 1
        if out["resolved"]:
            self.resolved_ticks += 1
        self.coherence_sum += out["coherence"]
        self.confidence_sum += out["confidence"]
        return {
            "resolved": bool(out["resolved"]),
            "disposition": [float(v) for v in out["disposition"]],
            "coherence": float(out["coherence"]),
            "confidence": float(out["confidence"]),
            "resolutionStrength": float(out["resolutionStrength"]),
            "clipped": clipped,
            "pressureNorm": min(n, CLIP_NORM),
        }

    def report(self):
        energy_now = self.seat.energy(self.state)
        led = dict(self.energy_ledger)
        led["storedEnergyChange"] = energy_now - self.energy_start
        led["numericalResidual"] = (
            led["storedEnergyChange"]
            - led["externalWork"] + led["dampingLoss"] + led["couplingTransferLoss"]
            - led["turbulenceInput"] - led["impulsiveDisturbanceWork"])
        return {
            "steps": self.steps,
            "lifecycleEvents": self.lifecycle_events,
            "clipEvents": self.clip_events,
            "clipShare": self.clip_events / self.steps if self.steps else None,
            "resolvedShare": self.resolved_ticks / self.steps if self.steps else None,
            "meanCoherence": self.coherence_sum / self.steps if self.steps else None,
            "meanConfidence": self.confidence_sum / self.steps if self.steps else None,
            "energyLedger": {k: float(v) for k, v in led.items()},
        }
