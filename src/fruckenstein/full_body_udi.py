"""Embodied crown-column and voxel pressure for the relational duck.

This organ has no actuator names or authored joint directions.  The duck host
serializes its morphology into four axial regions; local velocity, load, and
command residual drive frozen Candidate-0003 seats.  Parent-relative
dispositions become voxel evidence.  Food enters only as a spatial token.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


UDI_CORE = Path(__file__).resolve().parents[2] / "udi" / "udi-core"
if str(UDI_CORE) not in sys.path:
    sys.path.insert(0, str(UDI_CORE))

UDI_JAX = Path(__file__).resolve().parents[2] / "duck" / "udi-hourglass" / "sim"
if str(UDI_JAX) not in sys.path:
    sys.path.insert(0, str(UDI_JAX))

from voxels import VoxelField  # noqa: E402
try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from crown_decider_jax import crown_decide, zeros_state  # noqa: E402
except (ImportError, OSError):
    jax = None
    from seat import CommitmentSeat  # noqa: E402


REGIONS = (
    (5, 6, 7, 8),          # crown
    (0, 1, 2, 9, 10, 11), # hips
    (3, 12),               # knees
    (4, 13),               # ankles / root
)
G_SCALE = np.array([6.0, 2.0, 8.0])
C_ENV = 1.5
LAM_AX = 0.25
LAM_AURA = 0.35
LAM_BACK = 0.15
AURA_EMA = 0.02
S_UDI = 8.0


def radial_cap(value, cap=C_ENV):
    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    return value * min(1.0, cap / max(norm, 1e-12))


class FullBodyUDI:
    def __init__(self):
        self.fast_backend = jax is not None
        if self.fast_backend:
            self._batch_decide = jax.jit(jax.vmap(crown_decide))
            self._single_decide = jax.jit(crown_decide)
            self.region_states = jnp.stack([zeros_state() for _ in REGIONS])
            self.aura_state = zeros_state()
            self.apex_state = zeros_state()
        else:
            self.regions = [CommitmentSeat() for _ in REGIONS]
            self.aura = CommitmentSeat()
            self.apex = CommitmentSeat()
        self.field = VoxelField()
        self.dispositions = [np.zeros(3) for _ in REGIONS]
        self.aura_disposition = np.zeros(3)
        self.consensus = np.zeros(3)
        self.steps = 0
        self.resolved = 0
        self.refused = 0
        self.coherence_sum = 0.0
        self.null_sum = 0.0
        self.pressure_sum = 0.0
        self.last = {
            "resolved": False, "disposition": np.zeros(3),
            "coherence": 0.0, "pressure": np.zeros(3), "nullMass": 1.0,
        }

    @staticmethod
    def _fast_outputs(states, dispositions, coherences):
        """Restore the richer CommitmentSeat output contract from JAX state."""
        states = np.asarray(states)
        dispositions = np.asarray(dispositions)
        coherences = np.asarray(coherences)
        outputs = []
        for state, disposition, coherence in zip(states, dispositions, coherences):
            axial = state[-9:].reshape(3, 3)
            strength = float(np.linalg.norm(np.mean(axial, axis=0)))
            resolved = bool(np.linalg.norm(disposition) > 0.5)
            outputs.append({
                "resolved": resolved,
                "disposition": disposition,
                "coherence": float(coherence),
                "resolutionStrength": strength,
            })
        return outputs

    def _step_regions(self, pressures):
        if not self.fast_backend:
            return [seat.step(radial_cap(p)) for seat, p in zip(self.regions, pressures)]
        clipped = np.stack([radial_cap(p) for p in pressures])
        self.region_states, dispositions, coherences = self._batch_decide(
            self.region_states, jnp.asarray(clipped)
        )
        return self._fast_outputs(self.region_states, dispositions, coherences)

    def _step_single(self, kind, pressure):
        if not self.fast_backend:
            return getattr(self, kind).step(radial_cap(pressure))
        state_name = f"{kind}_state"
        state, disposition, coherence = self._single_decide(
            getattr(self, state_name), jnp.asarray(radial_cap(pressure))
        )
        setattr(self, state_name, state)
        return self._fast_outputs(
            np.asarray(state)[None, :],
            np.asarray(disposition)[None, :],
            np.asarray(coherence)[None],
        )[0]

    def step(self, velocity, load, residual, apple_direction, hunger, reliability):
        velocity = np.asarray(velocity, dtype=float)
        load = np.asarray(load, dtype=float)
        residual = np.asarray(residual, dtype=float)
        gradients = []
        for indices in REGIONS:
            idx = np.asarray(indices, dtype=int)
            gradients.append(np.array([
                float(np.mean(velocity[idx])),
                float(np.mean(load[idx])),
                float(np.mean(residual[idx])),
            ]) * G_SCALE)

        pressures = []
        for index, gradient in enumerate(gradients):
            pressure = gradient.copy()
            if index > 0:
                pressure += LAM_AX * self.dispositions[index - 1]
            if index < len(REGIONS) - 1:
                pressure += LAM_AX * self.dispositions[index + 1]
            pressure += LAM_AURA * self.aura_disposition
            pressures.append(pressure)
        region_outputs = self._step_regions(pressures)
        new_dispositions = [
            np.asarray(output["disposition"], dtype=float) for output in region_outputs
        ]

        aura_pressure = self.consensus + LAM_BACK * np.mean(self.dispositions, axis=0)
        aura_output = self._step_single("aura", aura_pressure)
        self.aura_disposition = np.asarray(aura_output["disposition"], dtype=float)
        self.dispositions = new_dispositions
        self.consensus = (
            (1.0 - AURA_EMA) * self.consensus
            + AURA_EMA * np.mean(new_dispositions, axis=0)
        )

        weights, directions, reliabilities = [], [], []
        for index, output in enumerate(region_outputs):
            parent = (
                new_dispositions[index + 1]
                if index < len(new_dispositions) - 1 else self.aura_disposition
            )
            relation = new_dispositions[index] - parent
            magnitude = float(np.linalg.norm(relation))
            direction = relation / magnitude if output["resolved"] and magnitude > 1e-8 else None
            weights.append(float(output["resolutionStrength"]))
            directions.append(direction)
            reliabilities.append(float(output["coherence"]))

        apple = np.asarray(apple_direction, dtype=float)
        apple_norm = float(np.linalg.norm(apple))
        directions.append(apple / apple_norm if apple_norm > 1e-8 else None)
        weights.append(float(np.clip(hunger, 0.0, 1.0)))
        reliabilities.append(float(np.clip(reliability, 0.0, 1.0)))

        proposed = self.field.propose(weights, directions, reliabilities)
        believed = self.field.integrate(proposed)
        pressure = S_UDI * self.field.readout(believed)
        apex_output = self._step_single("apex", pressure)
        disposition = np.asarray(apex_output["disposition"], dtype=float)
        if apex_output["resolved"]:
            self.field.choice_feedback(disposition, eta_scale=apex_output["coherence"])

        metrics = self.field.metrics(believed)
        self.steps += 1
        self.resolved += int(apex_output["resolved"])
        self.refused += int(not apex_output["resolved"])
        self.coherence_sum += float(apex_output["coherence"])
        self.null_sum += metrics["nullMassMean"]
        self.pressure_sum += float(np.linalg.norm(pressure))
        self.last = {
            "resolved": bool(apex_output["resolved"]),
            "disposition": disposition,
            "coherence": float(apex_output["coherence"]),
            "pressure": pressure,
            "nullMass": metrics["nullMassMean"],
        }
        return self.last

    def report(self):
        n = max(1, self.steps)
        return {
            "backend": "jax-batched" if self.fast_backend else "numpy-reference",
            "steps": self.steps,
            "resolvedShare": self.resolved / n,
            "refusalShare": self.refused / n,
            "meanCoherence": self.coherence_sum / n,
            "meanNullMass": self.null_sum / n,
            "meanPressureNorm": self.pressure_sum / n,
            "lastDisposition": self.last["disposition"].tolist(),
        }
