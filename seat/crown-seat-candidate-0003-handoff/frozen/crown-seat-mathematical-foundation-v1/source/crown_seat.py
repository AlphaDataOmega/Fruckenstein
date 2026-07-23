#!/usr/bin/env python3
"""Reduced Phase 1 crown-seat mathematics.

This module consumes the frozen Phase 0 geometry read-only.  It contains no
Duck, actuator, task, or embodiment semantics.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


Array = np.ndarray
EPS = 1.0e-12


def unit(v: Array) -> Array:
    n = float(np.linalg.norm(v))
    return np.zeros_like(v) if n < EPS else v / n


def random_rotation(rng: np.random.Generator) -> Array:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


@dataclass(frozen=True)
class SeatParameters:
    decay: float = 0.82
    axial_decay: float = 0.46
    saturation: float = 0.34
    axial_saturation: float = 0.22
    ring_coupling: float = 1.05
    overlap_coupling: float = 0.72
    eye_coupling: float = 0.88
    spine_coupling: float = 0.66
    bridge_coupling: float = 0.22
    input_gain: float = 1.45
    transverse_input_fraction: float = 0.08
    handed_precession: float = 0.18
    dense_pair_weight: float = 0.16
    coherence_threshold: float = 0.54
    confidence_threshold: float = 0.40
    strength_threshold: float = 0.035
    dt: float = 0.02

    @classmethod
    def from_json(cls, path: Path) -> "SeatParameters":
        return cls(**json.loads(path.read_text(encoding="utf-8"))["parameters"])

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


class CrownGeometry:
    """Read-only reduced view of the frozen Phase 0 manifest."""

    def __init__(self, manifest: dict, frame_override: Array | None = None):
        self.manifest = copy.deepcopy(manifest)
        self.ids = [f["id"] for f in manifest["peripheralFunnels"]]
        self.index = {name: i for i, name in enumerate(self.ids)}
        self.funnels = {f["id"]: f for f in manifest["peripheralFunnels"]}
        frames = []
        for name in self.ids:
            fr = self.funnels[name]["localFrame"]
            frames.append(
                np.column_stack(
                    [fr["tangent"], fr["normal"], fr["binormal"]]
                ).astype(float)
            )
        self.frames = np.stack(frames) if frame_override is None else frame_override.copy()
        self.handedness = np.asarray(
            [self.funnels[name]["handedness"] for name in self.ids], dtype=float
        )
        self.cells = {
            "upper": [self.index[name] for name in self.ids if name.startswith("U")],
            "lower": [self.index[name] for name in self.ids if name.startswith("L")],
        }
        self.ring_edges = []
        for edge in manifest["graphEdges"]:
            if edge["kind"] == "peripheral-crown-adjacency":
                self.ring_edges.append(
                    (self.index[edge["from"]], self.index[edge["to"]], float(edge["weight"]))
                )
        self.overlaps = []
        overlap_by_pair = {
            (o["flowA"], o["flowB"]): o for o in manifest["overlaps"]
        }
        for edge in manifest["graphEdges"]:
            if edge["kind"] != "cross-cell-overlap":
                continue
            a, b = edge["from"], edge["to"]
            rec = overlap_by_pair[(a, b)]
            self.overlaps.append(
                (
                    self.index[a],
                    self.index[b],
                    float(edge["weight"]),
                    np.asarray(rec["overlapFrameTransformAtoB"], dtype=float),
                )
            )
        self.mobile_cores = {
            c["id"]: np.asarray(c["position"], dtype=float)
            for c in manifest["mobileCores"]
        }
        self.spine_nodes = {
            n["id"]: np.asarray(n["position"], dtype=float)
            for n in manifest["spine"]["nodes"]
        }
        self.geometry_hash = manifest["geometryHash"]

    @classmethod
    def load(cls, path: Path) -> "CrownGeometry":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def rotated(self, q: Array) -> "CrownGeometry":
        return CrownGeometry(self.manifest, frame_override=np.einsum("ab,ibc->iac", q, self.frames))

    def transport(self, receiver: int, sender: int) -> Array:
        return self.frames[receiver].T @ self.frames[sender]


@dataclass
class CouplingEdge:
    i: int
    j: int
    weight: float
    receiver_i_from_j: Array
    receiver_j_from_i: Array
    kind: str


class CrownSeat:
    """Ten local funnel vectors plus a three-node axial state.

    The upper and lower axial nodes are the two mobile-eye member states.  The
    middle axial node is the free junction and is not counted as a thirteenth
    crown member.
    """

    J = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    def __init__(
        self,
        geometry: CrownGeometry,
        parameters: SeatParameters,
        condition: str = "coherent-local-frame",
        seed: int = 0,
    ):
        self.geometry = geometry
        self.p = parameters
        self.condition = condition
        self.rng = np.random.default_rng(seed)
        self.n = len(geometry.ids)
        self.state_size = 3 * self.n + 9
        self.active = np.ones(self.n, dtype=bool)
        if condition == "missing-funnel":
            self.active[seed % self.n] = False
        self.edges = self._build_edges(condition)

    def zeros(self) -> Array:
        return np.zeros(self.state_size, dtype=float)

    def random_state(self, scale: float = 0.08) -> Array:
        s = self.rng.normal(scale=scale, size=self.state_size)
        for i, active in enumerate(self.active):
            if not active:
                s[3 * i : 3 * i + 3] = 0.0
        return s

    def unpack(self, state: Array) -> tuple[Array, Array]:
        return state[: 3 * self.n].reshape(self.n, 3), state[3 * self.n :].reshape(3, 3)

    def pack(self, funnels: Array, axial: Array) -> Array:
        return np.concatenate([funnels.reshape(-1), axial.reshape(-1)])

    def _consistent_edge(self, i: int, j: int, w: float, kind: str) -> CouplingEdge:
        m_i_j = self.geometry.transport(i, j)
        return CouplingEdge(i, j, w, m_i_j, m_i_j.T, kind)

    def _build_edges(self, condition: str) -> list[CouplingEdge]:
        if condition == "dense-phase-consistent":
            return [
                self._consistent_edge(i, j, self.p.dense_pair_weight, "dense")
                for i in range(self.n)
                for j in range(i + 1, self.n)
            ]

        ring = list(self.geometry.ring_edges)
        overlaps = list(self.geometry.overlaps)
        if condition == "degree-weight-matched-random-rewiring":
            ring = []
            for cell in (self.geometry.cells["upper"], self.geometry.cells["lower"]):
                order = list(self.rng.permutation(cell))
                ring.extend((order[k], order[(k + 1) % len(order)], 1.0) for k in range(len(order)))
            upper = list(self.rng.permutation(self.geometry.cells["upper"]))
            lower = list(self.rng.permutation(self.geometry.cells["lower"]))
            overlaps = [(a, b, self.geometry.overlaps[k][2], None) for k, (a, b) in enumerate(zip(upper, lower))]

        edges = [self._consistent_edge(i, j, w, "ring") for i, j, w in ring]
        if condition != "no-bleed":
            for i, j, w, frozen_m_j_i in overlaps:
                if frozen_m_j_i is None:
                    edge = self._consistent_edge(i, j, w, "overlap-rewired")
                else:
                    # The frozen record maps i/A local coordinates into j/B.
                    edge = CouplingEdge(i, j, w, frozen_m_j_i.T, frozen_m_j_i, "overlap")
                edges.append(edge)

        if condition == "randomized-phase-transport":
            changed = []
            for edge in edges:
                q = random_rotation(self.rng)
                m_i_j = q @ edge.receiver_i_from_j
                changed.append(
                    CouplingEdge(edge.i, edge.j, edge.weight, m_i_j, m_i_j.T, edge.kind)
                )
            edges = changed
        elif condition == "geometry-scrambled-coupling":
            perm = self.rng.permutation(self.n)
            changed = []
            for edge in edges:
                m_i_j = self.geometry.frames[perm[edge.i]].T @ self.geometry.frames[perm[edge.j]]
                changed.append(
                    CouplingEdge(edge.i, edge.j, edge.weight, m_i_j, m_i_j.T, edge.kind)
                )
            edges = changed
        return edges

    def input_drive(self, pressure: Array) -> Array:
        u = np.zeros((self.n, 3), dtype=float)
        fraction = self.p.transverse_input_fraction
        for i in range(self.n):
            if not self.active[i]:
                continue
            local = self.geometry.frames[i].T @ pressure
            u[i] = self.p.input_gain * np.asarray([local[0], fraction * local[1], fraction * local[2]])
        return u

    def rhs(self, state: Array, pressure: Array) -> Array:
        x, a = self.unpack(state)
        dx = -self.p.decay * x - self.p.saturation * np.sum(x * x, axis=1)[:, None] * x
        da = -self.p.axial_decay * a - self.p.axial_saturation * np.sum(a * a, axis=1)[:, None] * a
        dx += self.input_drive(pressure)

        for i in range(self.n):
            if self.active[i]:
                dx[i] += self.p.handed_precession * self.geometry.handedness[i] * (self.J @ x[i])
            else:
                dx[i] = 0.0

        for edge in self.edges:
            if not (self.active[edge.i] and self.active[edge.j]):
                continue
            gain = self.p.ring_coupling if edge.kind == "ring" else self.p.overlap_coupling
            if edge.kind == "dense":
                gain = self.p.ring_coupling
            dx[edge.i] += gain * edge.weight * (edge.receiver_i_from_j @ x[edge.j] - x[edge.i])
            dx[edge.j] += gain * edge.weight * (edge.receiver_j_from_i @ x[edge.i] - x[edge.j])

        global_x = np.einsum("iab,ib->ia", self.geometry.frames, x)
        for axial_index, cell_name in ((0, "upper"), (2, "lower")):
            active_members = [i for i in self.geometry.cells[cell_name] if self.active[i]]
            if not active_members:
                continue
            gain = self.p.eye_coupling / len(active_members)
            for i in active_members:
                delta = a[axial_index] - global_x[i]
                dx[i] += gain * (self.geometry.frames[i].T @ delta)
                da[axial_index] -= gain * delta

        for left, right in ((0, 1), (1, 2)):
            delta = a[right] - a[left]
            da[left] += self.p.spine_coupling * delta
            da[right] -= self.p.spine_coupling * delta

        overlap_weight = sum(o[2] for o in self.geometry.overlaps) or 1.0
        for i, j, w, _ in self.geometry.overlaps:
            for member in (i, j):
                if not self.active[member]:
                    continue
                gain = self.p.bridge_coupling * w / overlap_weight
                delta = a[1] - global_x[member]
                dx[member] += gain * (self.geometry.frames[member].T @ delta)
                da[1] -= gain * delta

        return self.pack(dx, da)

    def output(self, state: Array) -> dict:
        x, axial = self.unpack(state)
        global_x = np.einsum("iab,ib->ia", self.geometry.frames, x)
        members = [global_x[i] for i in range(self.n) if self.active[i]] + [axial[0], axial[2]]
        member_norm_sum = sum(float(np.linalg.norm(v)) for v in members)
        member_coherence = float(np.linalg.norm(np.sum(members, axis=0)) / (member_norm_sum + EPS))
        axial_norm_sum = sum(float(np.linalg.norm(v)) for v in axial)
        axial_coherence = float(np.linalg.norm(np.sum(axial, axis=0)) / (axial_norm_sum + EPS))
        coherence = float(np.clip(0.65 * member_coherence + 0.35 * axial_coherence, 0.0, 1.0))
        three_component_state = np.mean(axial, axis=0)
        strength = float(np.linalg.norm(three_component_state))
        confidence = float(coherence * strength / (strength + 0.08))
        resolved = bool(
            coherence >= self.p.coherence_threshold
            and confidence >= self.p.confidence_threshold
            and strength >= self.p.strength_threshold
        )
        disposition = unit(three_component_state) if resolved else np.zeros(3)
        return {
            "disposition": disposition,
            "threeComponentState": three_component_state,
            "axialNodeStates": axial.copy(),
            "coherence": coherence,
            "memberCoherence": member_coherence,
            "axialCoherence": axial_coherence,
            "confidence": confidence,
            "resolutionStrength": strength,
            "resolved": resolved,
        }

    def energy(self, state: Array) -> float:
        return 0.5 * float(np.dot(state, state))

    def power_terms(self, state: Array, pressure: Array) -> dict[str, float]:
        x, a = self.unpack(state)
        u = self.input_drive(pressure)
        external = float(np.sum(x * u))
        damping = float(
            self.p.decay * np.sum(x * x)
            + self.p.saturation * np.sum(np.sum(x * x, axis=1) ** 2)
            + self.p.axial_decay * np.sum(a * a)
            + self.p.axial_saturation * np.sum(np.sum(a * a, axis=1) ** 2)
        )
        coupling = 0.0
        for edge in self.edges:
            if not (self.active[edge.i] and self.active[edge.j]):
                continue
            gain = self.p.ring_coupling if edge.kind == "ring" else self.p.overlap_coupling
            if edge.kind == "dense":
                gain = self.p.ring_coupling
            delta = x[edge.i] - edge.receiver_i_from_j @ x[edge.j]
            coupling += gain * edge.weight * float(np.dot(delta, delta))
        gx = np.einsum("iab,ib->ia", self.geometry.frames, x)
        for axial_index, cell_name in ((0, "upper"), (2, "lower")):
            active_members = [i for i in self.geometry.cells[cell_name] if self.active[i]]
            gain = self.p.eye_coupling / max(1, len(active_members))
            coupling += sum(gain * float(np.dot(gx[i] - a[axial_index], gx[i] - a[axial_index])) for i in active_members)
        coupling += self.p.spine_coupling * float(np.dot(a[0] - a[1], a[0] - a[1]))
        coupling += self.p.spine_coupling * float(np.dot(a[1] - a[2], a[1] - a[2]))
        overlap_weight = sum(o[2] for o in self.geometry.overlaps) or 1.0
        for i, j, w, _ in self.geometry.overlaps:
            for member in (i, j):
                if self.active[member]:
                    gain = self.p.bridge_coupling * w / overlap_weight
                    coupling += gain * float(np.dot(gx[member] - a[1], gx[member] - a[1]))
        return {
            "externalWorkRate": external,
            "dampingLossRate": damping,
            "couplingTransferLossRate": coupling,
            "turbulenceInputRate": 0.0,
            "skewPower": 0.0,
        }

    def step(self, state: Array, pressure: Array, dt: float, method: str = "rk4") -> Array:
        if method == "rk4":
            k1 = self.rhs(state, pressure)
            k2 = self.rhs(state + 0.5 * dt * k1, pressure)
            k3 = self.rhs(state + 0.5 * dt * k2, pressure)
            k4 = self.rhs(state + dt * k3, pressure)
            return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if method == "heun":
            k1 = self.rhs(state, pressure)
            predictor = state + dt * k1
            return state + 0.5 * dt * (k1 + self.rhs(predictor, pressure))
        raise ValueError(f"unknown integration method: {method}")

    def simulate(
        self,
        duration: float,
        pressure_fn: Callable[[float], Array],
        state: Array | None = None,
        dt: float | None = None,
        method: str = "rk4",
        perturbations: dict[int, Callable[[Array], Array]] | None = None,
    ) -> dict:
        dt = self.p.dt if dt is None else dt
        steps = int(round(duration / dt))
        state = self.random_state() if state is None else state.copy()
        perturbations = perturbations or {}
        times = np.arange(steps + 1, dtype=float) * dt
        states = np.empty((steps + 1, self.state_size), dtype=float)
        coherence = np.empty(steps + 1)
        confidence = np.empty(steps + 1)
        strength = np.empty(steps + 1)
        resolved = np.empty(steps + 1, dtype=bool)
        disposition = np.empty((steps + 1, 3), dtype=float)
        pressures = np.empty((steps + 1, 3), dtype=float)
        energies = np.empty(steps + 1)
        powers = {key: np.empty(steps + 1) for key in self.power_terms(state, pressure_fn(0.0))}
        integrated = {key: 0.0 for key in powers}
        impulse_energy = 0.0
        for k, t in enumerate(times):
            if k in perturbations:
                energy_before = self.energy(state)
                state = perturbations[k](state.copy())
                impulse_energy += self.energy(state) - energy_before
            p_now = np.asarray(pressure_fn(float(t)), dtype=float)
            out = self.output(state)
            states[k] = state
            pressures[k] = p_now
            coherence[k] = out["coherence"]
            confidence[k] = out["confidence"]
            strength[k] = out["resolutionStrength"]
            resolved[k] = out["resolved"]
            disposition[k] = out["disposition"]
            energies[k] = self.energy(state)
            for key, value in self.power_terms(state, p_now).items():
                powers[key][k] = value
            if k < steps:
                if method == "rk4":
                    k1 = self.rhs(state, p_now)
                    s2 = state + 0.5 * dt * k1
                    k2 = self.rhs(s2, p_now)
                    s3 = state + 0.5 * dt * k2
                    k3 = self.rhs(s3, p_now)
                    s4 = state + dt * k3
                    k4 = self.rhs(s4, p_now)
                    stage_powers = [
                        self.power_terms(state, p_now),
                        self.power_terms(s2, p_now),
                        self.power_terms(s3, p_now),
                        self.power_terms(s4, p_now),
                    ]
                    for key in integrated:
                        integrated[key] += dt * (
                            stage_powers[0][key]
                            + 2.0 * stage_powers[1][key]
                            + 2.0 * stage_powers[2][key]
                            + stage_powers[3][key]
                        ) / 6.0
                    state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                elif method == "heun":
                    k1 = self.rhs(state, p_now)
                    predictor = state + dt * k1
                    k2 = self.rhs(predictor, p_now)
                    p1 = self.power_terms(state, p_now)
                    p2 = self.power_terms(predictor, p_now)
                    for key in integrated:
                        integrated[key] += 0.5 * dt * (p1[key] + p2[key])
                    state = state + 0.5 * dt * (k1 + k2)
                else:
                    raise ValueError(f"unknown integration method: {method}")
        integrals = {key: float(value) for key, value in integrated.items()}
        integrals["turbulenceInputRate"] += impulse_energy
        predicted_delta = (
            integrals["externalWorkRate"]
            + integrals["turbulenceInputRate"]
            - integrals["dampingLossRate"]
            - integrals["couplingTransferLossRate"]
        )
        numerical_error = float((energies[-1] - energies[0]) - predicted_delta)
        return {
            "time": times,
            "state": states,
            "pressure": pressures,
            "coherence": coherence,
            "confidence": confidence,
            "strength": strength,
            "resolved": resolved,
            "disposition": disposition,
            "energy": energies,
            "power": powers,
            "integratedEnergyLedger": {
                "externalWork": integrals["externalWorkRate"],
                "internalStoredEnergyChange": float(energies[-1] - energies[0]),
                "couplingTransferLoss": integrals["couplingTransferLossRate"],
                "dampingLoss": integrals["dampingLossRate"],
                "turbulenceInput": integrals["turbulenceInputRate"],
                "impulsiveDisturbanceWork": impulse_energy,
                "numericalError": numerical_error,
            },
        }

    def linear_matrices(self) -> tuple[Array, Array, Array]:
        zero = self.zeros()
        h = 1.0e-7
        a_mat = np.empty((self.state_size, self.state_size))
        for k in range(self.state_size):
            delta = np.zeros(self.state_size)
            delta[k] = h
            a_mat[:, k] = (self.rhs(delta, np.zeros(3)) - self.rhs(-delta, np.zeros(3))) / (2.0 * h)
        b_mat = np.column_stack([self.rhs(zero, np.eye(3)[k]) for k in range(3)])
        c_mat = np.zeros((3, self.state_size))
        axial_offset = 3 * self.n
        for node in range(3):
            c_mat[:, axial_offset + 3 * node : axial_offset + 3 * node + 3] = np.eye(3) / 3.0
        return a_mat, b_mat, c_mat

    def jacobian(self, state: Array, pressure: Array) -> Array:
        h = 2.0e-6
        jac = np.empty((self.state_size, self.state_size))
        for k in range(self.state_size):
            delta = np.zeros(self.state_size)
            delta[k] = h
            jac[:, k] = (self.rhs(state + delta, pressure) - self.rhs(state - delta, pressure)) / (2.0 * h)
        return jac

    def fixed_point(self, pressure: Array, duration: float = 80.0) -> Array:
        result = self.simulate(duration, lambda _t: pressure, state=self.zeros(), dt=0.03)
        return result["state"][-1]


class FixedCrownEyeField:
    """A mobile eye sampling the fixed five-funnel Phase 0 pressure field."""

    def __init__(self, geometry: CrownGeometry, cell: str = "upper", samples: int = 121):
        self.geometry = geometry
        self.cell = cell
        self.indices = geometry.cells[cell]
        self.samples = []
        s_values = np.linspace(0.0, 1.0, samples)
        for i in self.indices:
            f = geometry.funnels[geometry.ids[i]]
            p0, p1, p2 = [np.asarray(p, dtype=float) for p in f["centerline"]["controlPoints"]]
            points = np.asarray([(1-s)**2*p0 + 2*(1-s)*s*p1 + s*s*p2 for s in s_values])
            derivatives = np.asarray([2*(1-s)*(p1-p0) + 2*s*(p2-p1) for s in s_values])
            tangents = np.asarray([unit(v) for v in derivatives])
            pars = f["funnelRadius"]["parameters"]
            radii = pars["R_eye"] + (pars["R_outer"] - pars["R_eye"]) * (1-s_values) ** pars["alpha"]
            self.samples.append((s_values, points, derivatives, tangents, radii, f))

    def field(self, position: Array) -> tuple[float, Array, Array]:
        total_potential = 0.0
        total_force = np.zeros(3)
        component_force = []
        for s_values, points, derivatives, tangents, radii, f in self.samples:
            k = int(np.argmin(np.sum((points - position) ** 2, axis=1)))
            s = float(s_values[k])
            center = points[k]
            tangent = tangents[k]
            delta = position - center
            if k == 0 or k == len(s_values) - 1:
                radial = delta
            else:
                radial = delta - tangent * float(np.dot(delta, tangent))
            rho = float(np.linalg.norm(radial))
            radius = float(radii[k])
            pressure = f["pressurePotential"]["parameters"]
            delta_p = float(pressure["DeltaP"])
            k_rho = float(pressure["k_rho"])
            potential = delta_p * (1.0 - s) + 0.5 * k_rho * (rho / radius) ** 2
            radial_force = -k_rho * radial / (radius * radius)
            if 0 < k < len(s_values) - 1:
                speed = max(float(np.linalg.norm(derivatives[k])), 1.0e-6)
                longitudinal_force = delta_p * tangent / speed
            else:
                longitudinal_force = np.zeros(3)
            force = radial_force + longitudinal_force
            total_potential += potential
            total_force += force
            component_force.append(force)
        return total_potential, total_force, np.asarray(component_force)

    def simulate(
        self,
        initial_position: Array,
        initial_velocity: Array,
        duration: float = 10.0,
        dt: float = 0.005,
        mass: float = 1.0,
        damping: float = 1.15,
        force_scale: float = 0.11,
        force_cap: float = 4.0,
        enabled: bool = True,
    ) -> dict:
        steps = int(round(duration / dt))
        q = initial_position.astype(float).copy()
        v = initial_velocity.astype(float).copy()
        positions = np.empty((steps + 1, 3))
        velocities = np.empty((steps + 1, 3))
        potentials = np.empty(steps + 1)
        forces = np.empty((steps + 1, 3))
        deformations = np.empty(steps + 1)
        for k in range(steps + 1):
            potential, raw_force, components = self.field(q)
            force = force_scale * force_cap * np.tanh(raw_force / force_cap) if enabled else np.zeros(3)
            positions[k], velocities[k], potentials[k], forces[k] = q, v, potential, force
            norms = np.linalg.norm(components, axis=1)
            deformations[k] = float(np.std(norms) / (np.mean(norms) + EPS))
            if k == steps:
                break
            acceleration = (force - damping * v) / mass
            v = v + dt * acceleration
            q = q + dt * v
        return {
            "time": np.arange(steps + 1) * dt,
            "position": positions,
            "velocity": velocities,
            "potential": potentials,
            "force": forces,
            "crownStateDeformation": deformations,
        }


class ArticulatedSpine:
    """Two fixed links and one free hinge, with no angle-restoring term."""

    def __init__(self, link_length: float = 0.72):
        self.link_length = link_length

    def simulate(
        self,
        initial_positions: Array,
        direction: Array,
        handedness_sign: float,
        duration: float = 12.0,
        dt: float = 0.004,
        pull: float = 0.72,
        precession: float = 0.05,
        damping: float = 0.34,
    ) -> dict:
        q = initial_positions.astype(float).copy()
        v = np.zeros_like(q)
        d = unit(direction)
        steps = int(round(duration / dt))
        positions = np.empty((steps + 1, 3, 3))
        velocities = np.empty_like(positions)
        straightness = np.empty(steps + 1)
        hinge_angle = np.empty(steps + 1)
        angular_momentum = np.empty((steps + 1, 3))
        precession_rate = np.empty(steps + 1)
        previous_axis = unit(q[0] - q[2])
        for k in range(steps + 1):
            l1 = unit(q[0] - q[1])
            l2 = unit(q[2] - q[1])
            axis = unit(q[0] - q[2])
            positions[k], velocities[k] = q, v
            straightness[k] = float(np.linalg.norm(q[0] - q[2]) / (2.0 * self.link_length))
            hinge_angle[k] = float(math.acos(np.clip(np.dot(l1, l2), -1.0, 1.0)))
            com = np.mean(q, axis=0)
            angular_momentum[k] = np.sum(np.cross(q - com, v), axis=0)
            if k:
                precession_rate[k] = float(np.dot(np.cross(previous_axis, axis), d) / dt)
            else:
                precession_rate[k] = 0.0
            previous_axis = axis
            if k == steps:
                break
            # Equal-and-opposite tangential endpoint forces create a pure
            # handed torque about the current end-to-end axis.  They neither
            # choose a world direction nor change the net applied force.
            end_axis = q[0] - q[2]
            tangential = handedness_sign * precession * np.cross(d, end_axis)
            f_upper = pull * d + tangential
            f_lower = -pull * d - tangential
            forces = np.asarray([f_upper, -f_upper - f_lower, f_lower]) - damping * v
            v += dt * forces
            q += dt * v
            for _ in range(6):
                for i, j in ((0, 1), (2, 1)):
                    delta = q[i] - q[j]
                    dist = float(np.linalg.norm(delta))
                    if dist < EPS:
                        continue
                    correction = 0.5 * (dist - self.link_length) * delta / dist
                    q[i] -= correction
                    q[j] += correction
            for i, j in ((0, 1), (2, 1)):
                delta = unit(q[i] - q[j])
                rel = float(np.dot(v[i] - v[j], delta))
                v[i] -= 0.5 * rel * delta
                v[j] += 0.5 * rel * delta
        return {
            "time": np.arange(steps + 1) * dt,
            "position": positions,
            "velocity": velocities,
            "straightness": straightness,
            "hingeAngle": hinge_angle,
            "angularMomentum": angular_momentum,
            "precessionRate": precession_rate,
        }


def save_npz(path: Path, result: dict, keys: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: result[key] for key in keys})


def json_ready(value):
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(value.real), "imaginary": float(value.imag)}
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value
