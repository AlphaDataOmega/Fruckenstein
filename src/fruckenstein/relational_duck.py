"""Relational gait-development duck.

This is a development creature, not a cognition or sentience claim.  An
inherited repertoire proposes whole-body motions.  A bounded contextual
ledger predicts their felt consequences; the realized body writes the
relationship back.  No ledger value is sent to a joint as a command.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import os
import sys
import time

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
OPEN_DUCK = Path(os.environ.get(
    "DUCK_OPEN_DUCK", HERE.parents[1] / "duck" / "Open_Duck_Playground"
)).resolve()
MODEL_PATH = OPEN_DUCK / "playground" / "open_duck_mini_v2" / "xmls" / "scene_room.xml"
WATCH = os.environ.get("WATCH", "1") == "1"
CONTROL_STEPS = int(os.environ.get("N", "6000"))
SEED = int(os.environ.get("SEED", "11"))
PHI = (1.0 + math.sqrt(5.0)) / 2.0
RESOLUTION_FLOOR = 0.08
INITIAL_BANDWIDTH = 0.55
RESOLUTION_EXPONENT = 4.0
EAT_RADIUS = 0.20
ROOM_HALF = 1.55
APPLE_WALL_MARGIN = 0.10
APPLE_START_DISTANCE = 0.34
APPLE_MAX_DISTANCE = 1.30
APPLE_GROWTH_SPAN = 3.0
NSUB = 10
# The inherited recording travels opposite the beak frame.  Reverse-time
# playback is therefore retained, but the former full hip reflection made
# the feet fight the knees and ankles for traction.  These relations were
# selected by a forward/comfort Pareto search, not displacement alone.
FORWARD_LEG_RELATION = np.ones(14)
FORWARD_LEG_RELATION[[2, 11]] = -0.80
FORWARD_LEG_RELATION[[3, 12]] = 1.00
FORWARD_LEG_RELATION[[4, 13]] = 1.00
STEERING_TRIM_DEFAULT = -0.05
STEERING_TRIM_RANGE = (-0.45, 0.15)


def bandwidth_from_resolution(resolution: float) -> float:
    """Golden-ratio sharpening with a reversible, finite resolution floor."""
    r = float(np.clip(resolution, 0.0, 1.0))
    return max(RESOLUTION_FLOOR, INITIAL_BANDWIDTH / (PHI ** (RESOLUTION_EXPONENT * r)))


def resolution_from_bandwidth(bandwidth: float) -> float:
    h = float(np.clip(bandwidth, RESOLUTION_FLOOR, INITIAL_BANDWIDTH))
    return float(np.clip(
        math.log(INITIAL_BANDWIDTH / h, PHI) / RESOLUTION_EXPONENT, 0.0, 1.0
    ))


def follow_relation(current: float, target: float) -> float:
    """Sharpen promptly under pressure and relax more slowly when it passes."""
    rate = 1.0 / (PHI ** (3 if target > current else 5))
    return float(np.clip(current + rate * (target - current), 0.0, 1.0))


def ledger_path() -> Path:
    explicit = os.environ.get("DUCK_LEDGER_PATH")
    if explicit:
        return Path(explicit).resolve()
    # Keep the creature's memory beside the executable.  It remains visible,
    # portable with the duck, and writable in restricted desktop launches.
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent / "RelationalDuckData"
    else:
        root = HERE / "RelationalDuckData"
    return root / "gait-ledger.json"


@dataclass(frozen=True)
class GaitAction:
    name: str
    pace: float
    amplitude: float
    hip_reflection: bool
    steering: float


ACTIONS = (
    GaitAction("rest", 0.0, 0.0, False, 0.0),
    GaitAction("inherited-backward", 1.0, 0.38, False, 0.08),
    GaitAction("forward-soft", -1.0, 0.30, True, 0.06),
    GaitAction("forward-open", -1.0, 0.40, True, 0.08),
    GaitAction("forward-quick", -1.30, 0.38, True, 0.10),
)


class RelationalLedger:
    """Bounded value, separate support/resolution, persistent local kernels."""

    def __init__(self, path: Path):
        self.path = path
        self.arm = os.environ.get("RELATIONAL_ARM", "joined")
        self.kernels: list[dict] = []
        self.decisions = 0
        self.sleep_cycles = 0
        self.total_apples = 0
        self.gait_successes = [0.0] * len(ACTIONS)
        self.body_traits = {"steeringTrim": STEERING_TRIM_DEFAULT}
        self.wake_buffer: list[dict] = []
        self.load()

    def load(self):
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            schema = payload.get("schema")
            if schema not in ("relational-duck-ledger-v1", "relational-duck-ledger-v2"):
                return
            self.kernels = payload.get("kernels", [])
            self.decisions = int(payload.get("decisions", 0))
            self.sleep_cycles = int(payload.get("sleepCycles", 0))
            self.total_apples = int(payload.get("totalApples", 0))
            saved_successes = payload.get("gaitSuccesses", [])
            for i, value in enumerate(saved_successes[:len(ACTIONS)]):
                raw = max(0.0, float(value))
                self.gait_successes[i] = (
                    float(np.clip(raw, 0.0, 1.0)) if schema.endswith("v2")
                    else raw / (raw + PHI)
                )
            for row in self.kernels:
                if "resolution" not in row:
                    row["resolution"] = resolution_from_bandwidth(
                        float(row.get("bandwidth", INITIAL_BANDWIDTH))
                    )
                row["bandwidth"] = bandwidth_from_resolution(row["resolution"])
            traits = payload.get("bodyTraits", {})
            # steeringBias was an unstable per-tick intercept and is not
            # migrated.  A missing steeringTrim begins at the measured
            # neutral relation of the lower-conflict gait.
            self.body_traits["steeringTrim"] = float(np.clip(
                traits.get("steeringTrim", STEERING_TRIM_DEFAULT),
                *STEERING_TRIM_RANGE,
            ))
        except (OSError, ValueError, TypeError):
            self.kernels = []

    @staticmethod
    def _distance(a, b):
        # Older generations carried six-dimensional contexts.  Compare the
        # dimensions they actually lived; new full-body kernels become more
        # specific without invalidating inherited memory.
        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        shared = min(len(aa), len(bb))
        return float(np.linalg.norm(aa[:shared] - bb[:shared]))

    def query(self, context: np.ndarray, action: int) -> dict:
        rows = [k for k in self.kernels if int(k["action"]) == action]
        if not rows:
            prior = np.array([0.0, 0.80, -0.05, 0.0]) if action == 0 else np.zeros(4)
            return {"prediction": prior, "support": 0.0, "resolution": INITIAL_BANDWIDTH,
                    "accuracy": 0.50, "confidence": 0.0}
        weights = []
        for row in rows:
            d = self._distance(context, row["context"])
            h = max(RESOLUTION_FLOOR, float(row["bandwidth"]))
            coverage = min(1.0, len(row["context"]) / max(1, len(context)))
            specificity = 0.35 + 0.65 * coverage
            weights.append(float(row["support"]) * specificity * math.exp(-0.5 * (d / h) ** 2))
        weights = np.asarray(weights)
        total = float(weights.sum())
        if total < 1e-9:
            return {"prediction": np.zeros(4), "support": 0.0,
                    "resolution": INITIAL_BANDWIDTH, "accuracy": 0.50, "confidence": 0.0}
        normalized = weights / total
        prediction = np.sum(
            normalized[:, None] * np.asarray([r["outcome"] for r in rows]), axis=0
        )
        prediction = np.clip(prediction, -1.0, 1.0)
        accuracy = float(np.sum(normalized * np.asarray([r["accuracy"] for r in rows])))
        resolution = float(np.sum(normalized * np.asarray([r["bandwidth"] for r in rows])))
        confidence = float(1.0 - math.exp(-total / 3.0))
        return {"prediction": prediction, "support": total, "resolution": resolution,
                "accuracy": accuracy, "confidence": confidence}

    def observe(self, context, action, prediction, outcome):
        prediction = np.asarray(prediction, dtype=float)
        outcome = np.clip(np.asarray(outcome, dtype=float), -1.0, 1.0)
        error = float(np.mean(np.abs(prediction - outcome)))
        coherence = float(np.clip(1.0 - error / 2.0, 0.0, 1.0))
        if self.arm != "frozen-ledger":
            self.wake_buffer.append({
                "context": np.asarray(context, dtype=float).tolist(), "action": int(action),
                "outcome": outcome.tolist(), "coherence": coherence,
            })
        self.decisions += 1
        return coherence, error

    def sleep(self):
        """Consolidate lived bindings; repeated replay never mints evidence."""
        if not self.wake_buffer:
            return
        episode = self.wake_buffer
        self.wake_buffer = []
        # Unvisited relationships lose present authority without erasing their
        # historical outcome.  Their support and resolution can broaden again.
        for kernel in self.kernels:
            kernel["support"] = max(0.50, float(kernel["support"]) / PHI ** 0.08)
            kernel["resolution"] = follow_relation(
                float(kernel.get("resolution", 0.0)), 0.0
            )
            kernel["bandwidth"] = bandwidth_from_resolution(kernel["resolution"])
        if self.arm == "shuffled-binding" and len(episode) > 1:
            shifted = episode[1:] + episode[:1]
            episode = [
                {**event, "outcome": shifted[i]["outcome"], "coherence": shifted[i]["coherence"]}
                for i, event in enumerate(episode)
            ]
        order = episode if self.sleep_cycles % 2 == 0 else list(reversed(episode))
        for event in order:
            context = np.asarray(event["context"], dtype=float)
            candidates = [
                k for k in self.kernels
                if int(k["action"]) == event["action"]
                and len(k["context"]) == len(event["context"])
            ]
            nearest = min(candidates, key=lambda k: self._distance(context, k["context"])) if candidates else None
            if nearest is None or self._distance(context, nearest["context"]) > float(nearest["bandwidth"]):
                self.kernels.append({
                    "action": event["action"], "context": context.tolist(),
                    "outcome": event["outcome"], "support": 1.0,
                    "accuracy": event["coherence"], "resolution": 0.0,
                    "bandwidth": INITIAL_BANDWIDTH,
                })
                continue
            support = float(nearest["support"])
            event_weight = max(0.10, float(event["coherence"]))
            # At the evidence ceiling the oldest effective weight yields to
            # the new relation; the row cannot become an irreversible fossil.
            support = min(support, PHI ** 6 - event_weight)
            new_support = support + event_weight
            old_context = np.asarray(nearest["context"], dtype=float)
            old_outcome = np.asarray(nearest["outcome"], dtype=float)
            context_gradient = float(np.clip(
                self._distance(context, old_context)
                / max(RESOLUTION_FLOOR, float(nearest["bandwidth"])), 0.0, 1.0
            ))
            outcome_gradient = float(np.mean(np.abs(
                np.asarray(event["outcome"], dtype=float) - old_outcome
            )))
            nearest["context"] = (
                (support * old_context + event_weight * context) / new_support
            ).tolist()
            nearest["outcome"] = np.clip(
                (support * old_outcome + event_weight * np.asarray(event["outcome"]))
                / new_support, -1.0, 1.0
            ).tolist()
            nearest["accuracy"] = float(
                (support * float(nearest["accuracy"])
                 + event_weight * event["coherence"]) / new_support
            )
            nearest["support"] = new_support
            # Crowded coherent evidence and contradictory local outcomes both
            # demand finer distinctions.  Once pressure falls, resolution and
            # bandwidth move back toward the broad relation above.
            crowding = 1.0 - math.exp(-new_support / (PHI ** 3))
            pressure = float(np.clip(
                0.50 * crowding * float(event["coherence"])
                + 0.30 * outcome_gradient + 0.20 * context_gradient,
                0.0, 1.0,
            ))
            u = float(np.clip((pressure - 0.18) / 0.68, 0.0, 1.0))
            target_resolution = u * u * (3.0 - 2.0 * u)
            nearest["resolution"] = follow_relation(
                float(nearest.get("resolution", 0.0)), target_resolution
            )
            nearest["bandwidth"] = bandwidth_from_resolution(nearest["resolution"])
        self.sleep_cycles += 1
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "relational-duck-ledger-v2", "decisions": self.decisions,
            "sleepCycles": self.sleep_cycles, "totalApples": self.total_apples,
            "gaitSuccesses": self.gait_successes,
            "bodyTraits": self.body_traits,
            "kernels": self.kernels,
        }
        temporary = self.path.with_name(
            f".{self.path.stem}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class RelationalDuck:
    def __init__(self):
        self.rng = np.random.default_rng(SEED)
        self.ledger = RelationalLedger(ledger_path())
        z = np.load(HERE / "instinct_repertoire.npz")
        self.g0 = z["gait_0"].copy()
        self.gl = z["gait_L"].copy()
        self.gr = z["gait_R"].copy()
        self.period = int(z["P"])
        self.home_q = z["HOME_Q"].copy()
        self.home = self.home_q[7:].copy()
        self.lo, self.hi = z["ctrlrange"][:, 0], z["ctrlrange"][:, 1]
        self.forward_repertoires = []
        for source in (self.g0, self.gl, self.gr):
            transformed = self.home + (source - self.home) * FORWARD_LEG_RELATION
            self.forward_repertoires.append(transformed)

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.apple_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "apple")
        self.apple_mocap = int(self.model.body_mocapid[self.apple_id])
        self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.full_body = os.environ.get("FULL_BODY", "1") == "1"
        self.full_body_mode = os.environ.get("FULL_BODY_MODE", "udi")
        self.udi_authority = os.environ.get("UDI_AUTHORITY", "1") == "1"
        self.body_udi = None
        if self.full_body and self.full_body_mode == "udi":
            from full_body_udi import FullBodyUDI
            self.body_udi = FullBodyUDI()
        self.sensor_slices = {}
        for name in (
            "gyro", "local_linvel", "accelerometer", "upvector",
            "right_foot_global_linvel", "left_foot_global_linvel",
            "left_foot_pos", "right_foot_pos",
        ):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = int(self.model.sensor_adr[sid])
            self.sensor_slices[name] = slice(adr, adr + int(self.model.sensor_dim[sid]))
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)

        self.phase = 0.0
        self.yaw_rate_memory = 0.0
        self.steering_trim = float(self.ledger.body_traits["steeringTrim"])
        initial_rotation = self.data.xmat[self.base_id].reshape(3, 3)
        self.previous_body_yaw = float(math.atan2(initial_rotation[1, 0], initial_rotation[0, 0]))
        self.steering_cycle_phase = 0.0
        self.steering_cycle_yaw = 0.0
        self.steering_cycle_intent = 0.0
        self.steering_cycle_stance = 0.0
        self.steering_cycle_ticks = 0
        self.hunger = 0.12
        self.eaten = 0
        self.falls = 0
        self.current_action = 0
        self.current_query = self.ledger.query(self.context(), 0)
        self.action_start = None
        self.window_step = 0
        self.following = 0.65
        self.relations = []
        self.apple_distances = []
        self.body_ticks = 0
        self.body_totals = {
            "support": 0.0, "slip": 0.0, "tilt": 0.0,
            "gazeError": 0.0, "turnDemand": 0.0, "steering": 0.0,
            "feltStrain": 0.0,
        }
        self.action_counts = np.zeros(len(ACTIONS), dtype=int)
        self.new_apple(initial=True)
        self.begin_relation()

    def new_apple(self, initial=False):
        base = self.data.xpos[self.base_id].copy()
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        yaw = float(math.atan2(rotation[1, 0], rotation[0, 0]))
        yaw_rotation = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw)],
        ])
        stage = max(0, self.ledger.total_apples)
        # Every apple makes the next relation longer.  The remaining gap to
        # the room-scale challenge contracts by phi, so growth never reverses
        # or abruptly jumps into an unreachable task.
        radius = APPLE_MAX_DISTANCE - (
            (APPLE_MAX_DISTANCE - APPLE_START_DISTANCE)
            / (PHI ** (stage / APPLE_GROWTH_SPAN))
        )
        preferred = 0.12 if initial and stage == 0 else float(self.rng.uniform(-1.15, 1.15))
        # Keep the exact curriculum distance.  Near a wall, rotate the target
        # to a reachable direction instead of clipping it silently closer.
        offsets = [0.0]
        for step in range(1, 91):
            offset = math.pi * step / 90.0
            offsets.extend((offset, -offset))
        world = None
        angle = preferred
        for offset in offsets:
            candidate_angle = preferred + offset
            local = np.array([
                radius * math.cos(candidate_angle),
                radius * math.sin(candidate_angle),
            ])
            candidate = base.copy()
            candidate[:2] += yaw_rotation @ local
            if np.all(np.abs(candidate[:2]) <= ROOM_HALF - APPLE_WALL_MARGIN):
                angle, world = candidate_angle, candidate
                break
        if world is None:
            raise RuntimeError("no reachable apple direction at curriculum distance")
        self.data.mocap_pos[self.apple_mocap] = [world[0], world[1], 0.08]
        actual = float(np.linalg.norm(world[:2] - base[:2]))
        self.apple_distances.append(actual)
        self.current_apple_target_distance = actual

    def felt_state(self):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        to_apple = np.asarray(self.data.mocap_pos[self.apple_mocap]) - self.data.xpos[self.base_id]
        body = rotation.T @ to_apple
        distance = float(np.linalg.norm(to_apple[:2]))
        bearing = float(math.atan2(body[1], body[0]))
        upright = float(rotation[2, 2])
        local_velocity = rotation.T @ self.data.qvel[:3]
        return distance, bearing, upright, float(local_velocity[0])

    def sensor(self, name):
        return np.asarray(self.data.sensordata[self.sensor_slices[name]], dtype=float)

    def body_field(self, bearing=0.0):
        """Whole-body relation used as perception and local negotiation."""
        up = self.sensor("upvector")
        gyro = self.sensor("gyro")
        velocity = self.sensor("local_linvel")
        left_pos = self.sensor("left_foot_pos")
        right_pos = self.sensor("right_foot_pos")
        left_vel = self.sensor("left_foot_global_linvel")
        right_vel = self.sensor("right_foot_global_linvel")
        left_support = float(np.clip((0.055 - left_pos[2]) / 0.045, 0.0, 1.0))
        right_support = float(np.clip((0.055 - right_pos[2]) / 0.045, 0.0, 1.0))
        support = max(left_support, right_support)
        left_slip = float(np.linalg.norm(left_vel[:2])) * left_support
        right_slip = float(np.linalg.norm(right_vel[:2])) * right_support
        slip = float(np.clip(max(left_slip, right_slip) / 0.45, 0.0, 1.0))
        roll = float(math.atan2(up[1], max(1e-6, up[2])))
        pitch = float(-math.atan2(up[0], max(1e-6, up[2])))
        tilt = float(np.clip(math.hypot(roll, pitch) / 0.65, 0.0, 1.0))
        bearing_error = float(np.clip(bearing / (math.pi / 2), -1.0, 1.0))
        yaw_rate = float(getattr(self, "yaw_rate_memory", gyro[2]))
        turn_demand = float(np.clip(bearing_error - 0.60 * yaw_rate, -1.0, 1.0))
        return {
            "roll": roll, "pitch": pitch, "yawRate": yaw_rate,
            "forwardSpeed": float(velocity[0]), "lateralSpeed": float(velocity[1]),
            "leftSupport": left_support, "rightSupport": right_support,
            "support": support, "slip": slip, "tilt": tilt,
            "turnDemand": turn_demand,
        }

    def context(self):
        distance, bearing, upright, forward_speed = self.felt_state()
        context = [
            self.hunger,
            np.clip(distance / 1.2, 0.0, 1.0),
            math.cos(bearing), math.sin(bearing),
            np.clip(upright, -1.0, 1.0),
            np.clip(forward_speed / 0.4, -1.0, 1.0),
        ]
        if self.full_body:
            body = self.body_field(bearing)
            context.extend([
                np.clip(body["roll"] / 0.5, -1.0, 1.0),
                np.clip(body["pitch"] / 0.5, -1.0, 1.0),
                np.clip(body["yawRate"] / 2.0, -1.0, 1.0),
                np.clip(body["lateralSpeed"] / 0.35, -1.0, 1.0),
                body["leftSupport"], body["rightSupport"], body["slip"],
            ])
        return np.asarray(context, dtype=float)

    def choose_action(self):
        context = self.context()
        queries = [self.ledger.query(context, i) for i in range(len(ACTIONS))]
        if self.hunger < 0.28:
            return 0, queries[0]
        # Reaching food is stronger evidence than novelty.  Once the body has
        # carried a gait all the way through a relation, preserve that gait as
        # a learned habit while hungry; increasing distance is a new task for
        # the same successful coordination, not a reason to replace it.
        proven = np.asarray(self.ledger.gait_successes[1:], dtype=float)
        if proven.size and float(proven.max()) >= 0.58:
            chosen = int(np.argmax(proven) + 1)
            return chosen, queries[chosen]
        # An unresolved relation cannot be declared bad before the body has
        # lived it.  Under-supported moving gaits take turns becoming real.
        unfamiliar = [i for i in range(1, len(ACTIONS)) if queries[i]["support"] < 0.25]
        if unfamiliar:
            chosen = int(self.rng.choice(unfamiliar))
            return chosen, queries[chosen]
        scores = []
        safety_need = float(np.clip(1.0 - max(context[4], 0.0), 0.0, 1.0))
        for i, query in enumerate(queries):
            outcome = np.asarray(query["prediction"])
            # Desired consequence is relational and state-dependent: progress
            # matters with hunger; stability always matters; effort matters most
            # when satiated.  This score selects a whole gait, never a joint.
            score = (
                self.hunger * (1.00 * outcome[0] + 0.25 * outcome[3])
                + (0.08 + 0.45 * safety_need) * outcome[1]
                + (0.10 + 0.16 * (1.0 - self.hunger)) * outcome[2]
            )
            novelty = 0.16 * (1.0 - query["confidence"])
            if i == 0:
                score += 0.28 * (1.0 - self.hunger) - 0.52 * self.hunger
                novelty *= 0.25
            scores.append(float(score + novelty))
        moving = np.asarray(scores[1:])
        order = np.argsort(moving)[::-1]
        # Refusal is preserved: unresolved moving relations return to rest.
        if moving[order[0]] < -0.25:
            return 0, queries[0]
        best = int(order[0] + 1)
        return best, queries[best]

    def begin_relation(self):
        action, query = self.choose_action()
        self.current_action = action
        self.current_query = query
        self.action_counts[action] += 1
        distance, bearing, upright, _speed = self.felt_state()
        self.action_start = {
            "context": self.context(), "distance": distance, "bearing": bearing,
            "upright": upright, "position": self.data.xpos[self.base_id, :2].copy(),
            "effort": 0.0, "slip": 0.0, "tilt": 0.0, "ticks": 0,
        }
        self.window_step = 0

    def finish_relation(self, fell=False, ate=False):
        start = self.action_start
        distance, bearing, upright, _speed = self.felt_state()
        ticks = max(1, int(start["ticks"]))
        progress = 1.0 if ate else np.clip(
            (float(start["distance"]) - distance) / 0.18, -1.0, 1.0
        )
        stability = -1.0 if fell else np.clip(2.0 * upright - 1.0, -1.0, 1.0)
        tracking_strain = np.clip(float(start["effort"]) / ticks / 0.55, 0.0, 1.0)
        slip_strain = np.clip(float(start.get("slip", 0.0)) / ticks, 0.0, 1.0)
        tilt_strain = np.clip(float(start.get("tilt", 0.0)) / ticks, 0.0, 1.0)
        # Progress without bodily ease is not a successful gait.  This value
        # is felt through tracking, planted-foot slip, and loss of posture.
        effort = -np.clip(
            0.45 * tracking_strain + 0.40 * slip_strain + 0.15 * tilt_strain,
            0.0, 1.0,
        )
        heading = np.clip((abs(float(start["bearing"])) - abs(bearing)) / 0.8, -1.0, 1.0)
        outcome = np.array([progress, stability, effort, heading])
        coherence, error = self.ledger.observe(
            start["context"], self.current_action, self.current_query["prediction"], outcome
        )
        # A gait's authority is a reversible relationship to its lived
        # consequences, never a win counter.  Unplayed gaits soften slowly;
        # the played gait follows present progress, balance, and strain.
        for index in range(1, len(self.ledger.gait_successes)):
            self.ledger.gait_successes[index] += (
                0.0 - self.ledger.gait_successes[index]
            ) / (PHI ** 9)
        if self.current_action != 0:
            felt_success = float(np.clip(
                0.45 * (progress + 1.0) / 2.0
                + 0.35 * (stability + 1.0) / 2.0
                + 0.20 * (effort + 1.0) / 2.0,
                0.0, 1.0,
            ))
            if ate:
                felt_success = 1.0
            self.ledger.gait_successes[self.current_action] = follow_relation(
                self.ledger.gait_successes[self.current_action], felt_success
            )
        self.relations.append({
            "action": ACTIONS[self.current_action].name, "prediction": self.current_query["prediction"].tolist(),
            "outcome": outcome.tolist(), "coherence": coherence, "error": error,
            "ate": bool(ate),
        })
        self.hunger = float(np.clip(self.hunger + 0.045 + (0.08 if fell else 0.0), 0.0, 1.0))
        if len(self.ledger.wake_buffer) >= 18:
            self.ledger.sleep()

    def recover(self):
        xy = self.data.qpos[:2].copy()
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        yaw = float(math.atan2(rotation[1, 0], rotation[0, 0]))
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        self.data.qpos[:2] = xy
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        self.phase = 0.0
        self.previous_body_yaw = yaw
        self.steering_cycle_phase = 0.0
        self.steering_cycle_yaw = 0.0
        self.steering_cycle_intent = 0.0
        self.steering_cycle_stance = 0.0
        self.steering_cycle_ticks = 0
        self.falls += 1

    def control_tick(self):
        action = ACTIONS[self.current_action]
        distance, bearing, upright, _speed = self.felt_state()
        if self.full_body:
            raw_yaw_rate = float(self.sensor("gyro")[2])
            # Gait oscillation is not a turn.  Only the temporally coherent
            # remainder is allowed to oppose or reinforce steering.
            self.yaw_rate_memory = 0.98 * self.yaw_rate_memory + 0.02 * raw_yaw_rate
        body = self.body_field(bearing) if self.full_body else None
        actual_joints = self.data.qpos[7:21].copy()
        if body is not None and self.body_udi is not None:
            joint_velocity = np.asarray(self.data.qvel[6:20], dtype=float)
            actuator_load = np.clip(
                np.asarray(self.data.actuator_force, dtype=float) / 3.23, -1.0, 1.0
            )
            command_residual = np.clip(
                (np.asarray(self.data.ctrl, dtype=float) - actual_joints)
                / np.maximum(self.hi - self.lo, 1e-6),
                -1.0, 1.0,
            )
            udi = self.body_udi.step(
                joint_velocity, actuator_load, command_residual,
                np.array([math.cos(bearing), math.sin(bearing), 0.0]),
                self.hunger,
                max(0.0, upright) * body["support"] * (1.0 - body["slip"]),
            )
            body["udiResolved"] = float(udi["resolved"])
            body["udiCoherence"] = float(udi["coherence"])
            if self.udi_authority:
                # Refusal is literal: no apex commitment means no authored
                # turn is substituted.  The proven gait continues unchanged.
                body["turnDemand"] = (
                    float(udi["disposition"][1]) if udi["resolved"] else 0.0
                )
        if self.current_action == 0:
            template = self.home
            phase_rate = 0.0
        else:
            if action.hip_reflection:
                g0, gl, gr = self.forward_repertoires
            else:
                g0, gl, gr = self.g0, self.gl, self.gr
            bearing_error = float(np.clip(bearing / (math.pi / 2), -1.0, 1.0))
            crown_has_authority = self.body_udi is not None and self.udi_authority
            steering_intent = body["turnDemand"] if crown_has_authority else bearing_error
            steer = action.steering * steering_intent
            stance = 0.0
            if body is not None:
                stance = (
                    body["support"] * (1.0 - body["slip"])
                    * (1.0 - body["tilt"]) ** 2
                )
                # No per-tick intercept may overpower the stride.  The crown
                # supplies only a bounded turn relation around the body's
                # slowly learned neutral trim; trim itself is updated from
                # realized yaw over a complete gait cycle below.
                steer = float(np.clip(
                    self.steering_trim + 0.35 * steering_intent * stance,
                    *STEERING_TRIM_RANGE,
                ))
            left, right = max(steer, 0.0), max(-steer, 0.0)
            ip = int(self.phase) % self.period
            gait = g0[ip] + left * (gl[ip] - g0[ip]) + right * (gr[ip] - g0[ip])
            template = self.home + action.amplitude * (gait - self.home)
            phase_rate = action.pace

        if body is not None:
            # The head is observed but not separately commanded.  Moving it
            # as a decree produced an equal-and-opposite base rotation; the
            # body must turn through supported gait contact as one relation.
            pass

        tracking_error = float(np.mean(np.abs(template - actual_joints)))
        contact_coherence = float(np.clip(math.exp(-tracking_error / 0.35) * max(upright, 0.0), 0.0, 1.0))
        predicted_accuracy = float(self.current_query["accuracy"])
        predicted_confidence = float(self.current_query["confidence"])
        felt_availability = 0.42 + 0.58 * (
            predicted_confidence * predicted_accuracy + (1.0 - predicted_confidence) * 0.45
        )
        self.following = 0.90 * self.following + 0.10 * contact_coherence
        embodiment = np.clip(
            0.62 + 0.38 * felt_availability * (0.35 + 0.65 * self.following),
            0.62, 1.0,
        )
        if self.ledger.arm == "body-bypass":
            embodiment = 1.0
        # The template passes through the body's present joint state.  It is a
        # negotiated attractor, not a pose decree.
        target = actual_joints + embodiment * (template - actual_joints)
        self.data.ctrl[:] = np.clip(target, self.lo, self.hi)
        phase_consent = 1.0
        if body is not None:
            phase_consent = 0.88 + 0.12 * (
                body["support"] * (1.0 - body["slip"]) * (1.0 - body["tilt"])
            )
            self.body_ticks += 1
            self.body_totals["support"] += body["support"]
            self.body_totals["slip"] += body["slip"]
            self.body_totals["tilt"] += body["tilt"]
            self.body_totals["gazeError"] += abs(bearing - actual_joints[7])
            self.body_totals["turnDemand"] += abs(body["turnDemand"])
            self.body_totals["steering"] += abs(steer) if self.current_action else 0.0
            self.body_totals["feltStrain"] += float(np.clip(
                0.45 * tracking_error / 0.55
                + 0.40 * body["slip"] + 0.15 * body["tilt"],
                0.0, 1.0,
            ))
        self.phase += phase_rate * (0.45 + 0.55 * self.following) * phase_consent
        self.action_start["effort"] += float(np.mean(np.abs(target - actual_joints)))
        if body is not None:
            self.action_start["slip"] += body["slip"]
            self.action_start["tilt"] += body["tilt"]
        self.action_start["ticks"] += 1

        for _ in range(NSUB):
            mujoco.mj_step(self.model, self.data)
        rotation_after = self.data.xmat[self.base_id].reshape(3, 3)
        yaw_after = float(math.atan2(rotation_after[1, 0], rotation_after[0, 0]))
        yaw_delta = float(math.atan2(
            math.sin(yaw_after - self.previous_body_yaw),
            math.cos(yaw_after - self.previous_body_yaw),
        ))
        self.previous_body_yaw = yaw_after
        if body is not None and self.current_action != 0:
            phase_advance = abs(
                phase_rate * (0.45 + 0.55 * self.following) * phase_consent
            )
            self.steering_cycle_phase += phase_advance
            self.steering_cycle_yaw += yaw_delta
            self.steering_cycle_intent += steering_intent
            self.steering_cycle_stance += stance
            self.steering_cycle_ticks += 1
            if self.steering_cycle_phase >= self.period:
                n_cycle = max(1, self.steering_cycle_ticks)
                mean_intent = self.steering_cycle_intent / n_cycle
                mean_stance = self.steering_cycle_stance / n_cycle
                # Turning experience must not redefine neutral.  Only a
                # near-straight apex disposition can teach the gait's trim;
                # otherwise intentional yaw is mistaken for body drift.
                if abs(mean_intent) < 0.15:
                    relation_error = self.steering_cycle_yaw
                    target_trim = float(np.clip(
                        STEERING_TRIM_DEFAULT - 0.20 * relation_error * mean_stance,
                        *STEERING_TRIM_RANGE,
                    ))
                    self.steering_trim += (
                        target_trim - self.steering_trim
                    ) / (PHI ** 4)
                self.ledger.body_traits["steeringTrim"] = self.steering_trim
                self.steering_cycle_phase %= self.period
                self.steering_cycle_yaw = 0.0
                self.steering_cycle_intent = 0.0
                self.steering_cycle_stance = 0.0
                self.steering_cycle_ticks = 0
        self.window_step += 1

        new_distance, _bearing, new_upright, _speed = self.felt_state()
        if new_distance < EAT_RADIUS:
            # Close the relation against the apple that was actually reached
            # before a new context exists.  Otherwise the next apple would be
            # mis-bound as the consequence of the preceding gait.
            self.finish_relation(fell=False, ate=True)
            self.eaten += 1
            self.ledger.total_apples += 1
            self.ledger.save()
            self.hunger = max(0.04, self.hunger - 0.72)
            self.new_apple()
            self.begin_relation()
            return
        if new_upright < 0.5:
            self.finish_relation(fell=True)
            self.recover()
            self.begin_relation()
        elif self.window_step >= self.period:
            self.finish_relation(fell=False)
            self.begin_relation()

    def close(self):
        if self.action_start and self.action_start["ticks"]:
            self.finish_relation(fell=False)
        self.ledger.sleep()

    def report(self):
        half = max(1, len(self.relations) // 2)
        early = self.relations[:half]
        late = self.relations[half:]
        mean = lambda rows, key, index=None: float(np.mean([
            r[key] if index is None else r[key][index] for r in rows
        ])) if rows else 0.0
        by_action = {}
        for action in ACTIONS:
            rows = [r for r in self.relations if r["action"] == action.name]
            by_action[action.name] = {
                "n": len(rows),
                "meanProgress": mean(rows, "outcome", 0),
                "meanStability": mean(rows, "outcome", 1),
                "meanEffortValue": mean(rows, "outcome", 2),
                "meanCoherence": mean(rows, "coherence"),
                "apples": int(sum(bool(r.get("ate")) for r in rows)),
            }
        return {
            "arm": self.ledger.arm, "fullBody": self.full_body,
            "relations": len(self.relations), "eaten": self.eaten, "falls": self.falls,
            "totalApples": self.ledger.total_apples,
            "gaitSuccesses": {
                ACTIONS[i].name: float(n) for i, n in enumerate(self.ledger.gait_successes)
            },
            "provenGait": (
                ACTIONS[int(np.argmax(self.ledger.gait_successes[1:])) + 1].name
                if max(self.ledger.gait_successes[1:], default=0.0) >= 0.58 else None
            ),
            "appleTargetDistance": self.current_apple_target_distance,
            "appleDistanceHistory": self.apple_distances,
            "hunger": self.hunger, "actionCounts": {
                ACTIONS[i].name: int(n) for i, n in enumerate(self.action_counts)
            },
            "earlyProgress": mean(early, "outcome", 0), "lateProgress": mean(late, "outcome", 0),
            "earlyPredictionError": mean(early, "error"), "latePredictionError": mean(late, "error"),
            "byAction": by_action,
            "bodyField": {
                key: (value / self.body_ticks if self.body_ticks else 0.0)
                for key, value in self.body_totals.items()
            },
            "bodyTraits": {"steeringTrim": self.steering_trim},
            "udi": self.body_udi.report() if self.body_udi is not None else None,
            "kernels": len(self.ledger.kernels), "sleepCycles": self.ledger.sleep_cycles,
            "ledger": str(self.ledger.path),
        }


def main():
    duck = RelationalDuck()
    if not WATCH:
        try:
            for _ in range(CONTROL_STEPS):
                duck.control_tick()
        finally:
            duck.close()
        print(json.dumps(duck.report(), indent=2), flush=True)
        return

    from mujoco.experimental.studio.native_viewer import NativeViewer
    viewer = NativeViewer(duck.model, title="Relational Duck - Learning Through Its Body", width=1200, height=800)
    camera = viewer.camera
    camera.distance = 2.4
    camera.elevation = -24
    camera.azimuth = 130
    last_report = time.time()
    try:
        while viewer.is_running():
            started = time.perf_counter()
            duck.control_tick()
            camera.lookat[:] = duck.data.xpos[duck.base_id]
            viewer.sync(duck.model, duck.data)
            if time.time() - last_report > 5.0:
                q = duck.current_query
                print(
                    f"[relation] action={ACTIONS[duck.current_action].name} hunger={duck.hunger:.2f} "
                    f"felt={q['accuracy']:.2f} support={q['support']:.1f} apples={duck.eaten} "
                    f"next-distance={duck.current_apple_target_distance:.2f}m "
                    f"relations={len(duck.relations)} kernels={len(duck.ledger.kernels)}",
                    flush=True,
                )
                last_report = time.time()
            delay = NSUB * duck.model.opt.timestep - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
    finally:
        duck.close()
        viewer.stop()


if __name__ == "__main__":
    main()
