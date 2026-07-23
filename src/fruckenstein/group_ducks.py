"""Open-floor group dynamics for three embodied crown ducks.

Each duck has its own sensors, Candidate-003/voxel organ, gait phase, and
actuator slice inside one MuJoCo world.  A replenishing apple field creates
independent and shared choices.  Invisible torso hulls make contact physical;
contact forces are attributed before a fall is classified as a knockdown.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from full_body_udi import FullBodyUDI
from relational_duck import FORWARD_LEG_RELATION


HERE = Path(__file__).resolve().parent
OPEN_DUCK = Path(os.environ.get(
    "DUCK_OPEN_DUCK", HERE.parents[1] / "duck" / "Open_Duck_Playground"
)).resolve()
ROBOT_XML = OPEN_DUCK / "playground" / "open_duck_mini_v2" / "xmls" / "open_duck_mini_v2.xml"
WATCH = os.environ.get("WATCH", "1") == "1"
CONTROL_STEPS = int(os.environ.get("N", "1200"))
USE_CROWNS = os.environ.get("GROUP_CROWNS", "1") == "1"
CROWN_PERIOD = max(1, int(os.environ.get("GROUP_CROWN_PERIOD", "5")))
NSUB = 10
N_DUCKS = 3
APPLE_COUNT = 7
EAT_RADIUS = 0.20
TARGET_SWITCH_ADVANTAGE = 0.12
# The repertoire is not linear below about 0.35 enactment: under that point
# the feet move but the body scarcely travels.  This wider physical range is
# still bounded below the observed fall boundary (about 0.50).
GROUP_STEERING_RANGE = (-0.60, 0.45)
GROUP_STEERING_TRIM_DEFAULT = -0.20
ACTUATOR_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
SENSOR_NAMES = (
    "gyro", "local_linvel", "upvector",
    "right_foot_global_linvel", "left_foot_global_linvel",
    "left_foot_pos", "right_foot_pos",
)
# Give each creature personal space at birth.  Shared choices can still bring
# them together later, but they do not begin as an artificial huddle.
SPAWNS = ((-0.32, 0.0), (0.16, 0.277), (0.16, -0.277))
INITIAL_APPLES = (
    (0.0, 0.0), (0.58, 0.0), (-0.58, 0.0),
    (0.0, 0.58), (0.0, -0.58), (0.41, 0.41), (-0.41, -0.41),
)
COLORS = ((0.95, 0.25, 0.20, 1.0), (0.20, 0.55, 0.95, 1.0), (0.25, 0.80, 0.35, 1.0))
PHI = (1.0 + math.sqrt(5.0)) / 2.0
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
STARVATION_THRESHOLD = 0.92
STARVATION_DAMAGE = 0.00010
PAIN_DAMAGE = 0.00005
REST_HEALING = 0.00004
FALL_DAMAGE = 0.06
KNOCKDOWN_DAMAGE = 0.04
BABBLE_MAX = 0.14
BODY_ADAPTER_VERSION = "relational-group-body-v15-motivation-mortality"


def group_memory_dir() -> Path | None:
    if os.environ.get("GROUP_MEMORY", "1") != "1":
        return None
    explicit = os.environ.get("GROUP_LEDGER_DIR")
    if explicit:
        return Path(explicit).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "RelationalDuckGroupData"
    return HERE / "RelationalDuckGroupData"


def inheritance_fingerprint() -> str:
    """Bind lived memory to the morphology, repertoire, and body adapter."""
    digest = hashlib.sha256()
    digest.update(BODY_ADAPTER_VERSION.encode("utf-8"))
    digest.update(np.asarray(FORWARD_LEG_RELATION, dtype=np.float64).tobytes())
    digest.update(np.asarray(GROUP_STEERING_RANGE, dtype=np.float64).tobytes())
    digest.update((HERE / "instinct_repertoire.npz").read_bytes())
    for path in sorted(ROBOT_XML.parent.rglob("*.xml")):
        digest.update(path.relative_to(ROBOT_XML.parent).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def build_model() -> mujoco.MjModel:
    """Attach three complete robots to an unbounded shared floor."""
    apple_bodies = "\n".join(
        f'''<body name="apple_{index}" mocap="true" pos="{x} {y} 0.08">
          <geom name="apple_{index}_fruit" type="sphere" size="0.06" rgba="0.92 0.13 0.11 1" contype="0" conaffinity="0"/>
          <geom name="apple_{index}_stem" type="capsule" fromto="0 0 0.05 0.01 0 0.085" size="0.006" rgba="0.3 0.18 0.06 1" contype="0" conaffinity="0"/>
        </body>'''
        for index, (x, y) in enumerate(INITIAL_APPLES)
    )
    base = mujoco.MjSpec.from_string(f"""
    <mujoco model="Relational Duck Group">
      <option timestep="0.002" iterations="60" ls_iterations="20"/>
      <visual>
        <headlight diffuse="0.65 0.65 0.65" ambient="0.35 0.35 0.35" specular="0 0 0"/>
        <global azimuth="145" elevation="-28"/>
      </visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.95 0.97 1" rgb2="0.72 0.79 0.88" width="512" height="512"/>
        <texture type="2d" name="ground_tex" builtin="checker" rgb1="0.88 0.89 0.91" rgb2="0.73 0.75 0.79" width="256" height="256"/>
        <material name="ground_mat" texture="ground_tex" texrepeat="12 12" texuniform="true" reflectance="0.05"/>
      </asset>
      <worldbody>
        <light pos="0 0 4" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
        <geom name="open_floor" type="plane" size="0 0 0.01" material="ground_mat"
              contype="1" conaffinity="1" condim="3" friction="0.65 0.01 0.001"/>
        {apple_bodies}
      </worldbody>
    </mujoco>
    """)
    for index, (x, y) in enumerate(SPAWNS):
        yaw = math.atan2(-y, -x)
        child = mujoco.MjSpec.from_file(str(ROBOT_XML))
        frame = base.worldbody.add_frame(
            name=f"spawn_{index}", pos=[x, y, 0.0], quat=yaw_quat(yaw)
        )
        base.attach(child, prefix=f"d{index}_", frame=frame)

        # The source robot collides only at its soles.  These massless hulls
        # follow the existing inertial bodies and transfer real contact force
        # without changing their mass distribution.
        trunk = base.body(f"d{index}_trunk_assembly")
        trunk.add_geom(
            name=f"d{index}_torso_hull", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-0.04, 0.0, 0.065], size=[0.12, 0.075, 0.08],
            contype=1, conaffinity=1, condim=3, friction=[0.7, 0.01, 0.001],
            rgba=[0.0, 0.0, 0.0, 0.0], group=3, mass=0.0,
        )
        # A small non-colliding color mark lets the viewer follow individuals.
        trunk.add_geom(
            name=f"d{index}_identity", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[-0.04, 0.0, 0.16], size=[0.025], contype=0, conaffinity=0,
            rgba=list(COLORS[index]), group=0, mass=0.0,
        )
    return base.compile()


@dataclass
class DuckStats:
    apples: int = 0
    falls: int = 0
    knockdowns: int = 0
    contacts: int = 0
    max_contact_force: float = 0.0
    deaths: int = 0


class GroupDuck:
    def __init__(self, world: "GroupWorld", index: int, repertoire: dict):
        self.world = world
        self.index = index
        self.prefix = f"d{index}_"
        m = world.model
        self.base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.prefix + "base")
        self.identity_geom_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_GEOM, self.prefix + "identity"
        )
        self.freejoint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, self.prefix + "floating_base")
        self.free_qpos = int(m.jnt_qposadr[self.freejoint_id])
        self.free_dof = int(m.jnt_dofadr[self.freejoint_id])
        self.actuator_ids = np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, self.prefix + name)
            for name in ACTUATOR_NAMES
        ], dtype=int)
        joint_ids = np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, self.prefix + name)
            for name in ACTUATOR_NAMES
        ], dtype=int)
        self.joint_qpos = np.asarray(m.jnt_qposadr[joint_ids], dtype=int)
        self.joint_dof = np.asarray(m.jnt_dofadr[joint_ids], dtype=int)
        self.sensors = {}
        for name in SENSOR_NAMES:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, self.prefix + name)
            adr = int(m.sensor_adr[sid])
            self.sensors[name] = slice(adr, adr + int(m.sensor_dim[sid]))
        self.home = np.asarray(repertoire["HOME_Q"])[7:].copy()
        self.period = int(repertoire["P"])
        self.gaits = []
        for name in ("gait_0", "gait_L", "gait_R"):
            source = np.asarray(repertoire[name])
            self.gaits.append(self.home + (source - self.home) * FORWARD_LEG_RELATION)
        self.phase = 0.0
        self.age = 0
        self.memory_path = (
            world.memory_dir / f"duck-{index + 1}.json"
            if world.memory_dir is not None else None
        )
        memory = self.load_memory()
        self.trim = float(np.clip(
            memory.get("steeringTrim", GROUP_STEERING_TRIM_DEFAULT),
            *GROUP_STEERING_RANGE,
        ))
        self.crown = FullBodyUDI() if USE_CROWNS else None
        self.last_disposition = np.zeros(3)
        self.last_crown_resolved = False
        self.crown_clock = index % CROWN_PERIOD
        # Metabolic need is a relation between reserve, food distance,
        # realized progress, and bodily strain.  Resolution is reversible:
        # pressure sharpens it; successful/easy motion broadens it again.
        self.reserve = 0.58
        self.hunger = 1.0 - self.reserve
        self.drive_pressure = 0.0
        self.drive_resolution = 0.0
        self.gait_pace = 0.92
        self.gait_enactment = float(np.clip(
            memory.get("gaitEnactment", 0.44), 0.36, 0.44
        ))
        self.gait_amplitude = self.gait_enactment
        self.appetite = 1.0
        self.pain = 0.0
        self.vitality = 1.0
        self.impact_load = 0.0
        self.babble = 0.0
        self.babble_ticks = 0
        self.dead = False
        self.death_cause = None
        self.previous_target = -1
        self.previous_distance = None
        self.previous_bearing = None
        self.target_apple = -1
        # The antipode of a sphere has no unique tangent.  Once the creature
        # begins resolving a behind-target relation, temporal continuity keeps
        # that side until the target enters its forward hemisphere.
        self.turn_commitment = 0.0
        self.turn_authority = 0.35
        self.turn_resolution = 0.0
        saved_effects = memory.get("effects", {})
        saved_support = memory.get("support", {})
        self.effect_prediction = {
            "forward": float(saved_effects.get("forward", 0.0)),
            "leftDirect": float(saved_effects.get(
                "leftDirect", saved_effects.get("left", 0.0)
            )),
            "leftReverse": float(saved_effects.get("leftReverse", 0.0)),
            "rightDirect": float(saved_effects.get(
                "rightDirect", saved_effects.get("right", 0.0)
            )),
            "rightReverse": float(saved_effects.get("rightReverse", 0.0)),
        }
        self.effect_support = {
            "forward": float(saved_support.get("forward", 0.0)),
            "leftDirect": float(saved_support.get(
                "leftDirect", saved_support.get("left", 0.0)
            )),
            "leftReverse": float(saved_support.get("leftReverse", 0.0)),
            "rightDirect": float(saved_support.get(
                "rightDirect", saved_support.get("right", 0.0)
            )),
            "rightReverse": float(saved_support.get("rightReverse", 0.0)),
        }
        self.lifetime_cycles = int(memory.get("cycles", 0))
        self.lifetime_apples = int(memory.get("apples", 0))
        self.lifetime_falls = int(memory.get("falls", 0))
        self.lifetime_deaths = int(memory.get("deaths", 0))
        self.cycle_phase = 0.0
        self.cycle_progress = 0.0
        self.cycle_heading = 0.0
        self.cycle_yaw = 0.0
        self.cycle_strain = 0.0
        self.cycle_intent = 0.0
        self.cycle_mapping = {}
        self.cycle_ticks = 0
        self.last_control_relation = None
        self.stats = DuckStats()
        self.was_upright = True
        self.fallen_ticks = 0
        self.last_contact_tick = -10000
        self.last_opponent = None

    def load_memory(self) -> dict:
        if self.memory_path is None or not self.memory_path.exists():
            return {}
        try:
            payload = json.loads(self.memory_path.read_text(encoding="utf-8"))
            compatible = (
                payload.get("schema") == "duck-body-effects-v1"
                and payload.get("bodyFingerprint") == self.world.body_fingerprint
            )
            return payload if compatible else {}
        except (OSError, ValueError, TypeError):
            return {}

    def save_memory(self):
        if self.memory_path is None:
            return
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "duck-body-effects-v1",
            "bodyFingerprint": self.world.body_fingerprint,
            "duck": self.index,
            "cycles": self.lifetime_cycles,
            "apples": self.lifetime_apples,
            "falls": self.lifetime_falls,
            "deaths": self.lifetime_deaths,
            "gaitEnactment": self.gait_enactment,
            "steeringTrim": self.trim,
            "effects": self.effect_prediction,
            "support": self.effect_support,
        }
        temporary = self.memory_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.memory_path)

    def _learn_effect(self, name: str, observation: float):
        support = float(np.clip(self.effect_support[name], 0.0, 1.0))
        rate = (1.0 / PHI ** 4) * (1.0 - 0.72 * support)
        self.effect_prediction[name] += rate * (
            float(observation) - self.effect_prediction[name]
        )
        self.effect_support[name] += (
            1.0 - self.effect_support[name]
        ) / (PHI ** 6)

    def turn_mapping(self, intent: float) -> tuple[float, str]:
        side = "left" if intent >= 0.0 else "right"
        direct = side + "Direct"
        # Handedness belongs to the shared body/world geometry.  Earlier code
        # let noisy approach evidence reverse this mapping, so a duck could
        # learn that visible left meant physical right.  Experience may learn
        # authority and neutral drift, but it may not silently mirror space.
        return 1.0, direct

    def appetite_relation(self) -> float:
        """Translate lived depletion into permission to spend the inherited gait."""
        normalized = float(np.clip((self.hunger - 0.08) / 0.34, 0.0, 1.0))
        return normalized * normalized * (3.0 - 2.0 * normalized)

    def die(self, cause: str):
        if self.dead:
            return
        self.dead = True
        self.death_cause = cause
        self.vitality = 0.0
        self.pain = 1.0
        self.babble = 0.0
        self.last_control_relation = None
        self.stats.deaths = 1
        self.lifetime_deaths += 1
        self.world.model.geom_rgba[self.identity_geom_id] = [0.12, 0.12, 0.12, 1.0]
        self.world.death_events.append({
            "tick": self.world.tick,
            "duck": self.index,
            "cause": cause,
        })
        self.save_memory()

    def injure(self, amount: float, cause: str):
        """Apply a discrete bodily consequence without turning pain into a command."""
        if self.dead:
            return
        self.vitality = float(np.clip(self.vitality - max(0.0, amount), 0.0, 1.0))
        self.impact_load = max(self.impact_load, float(np.clip(amount / 0.10, 0.0, 1.0)))
        if self.vitality <= 0.0:
            self.die(cause)

    def update_life(self, strain: float, upright: float):
        """Integrate nociception, starvation, healing, and irreversible mortality."""
        if self.dead:
            return
        fall_relation = float(np.clip((0.65 - upright) / 0.65, 0.0, 1.0))
        strain_alarm = float(np.clip((strain - 0.32) / 0.68, 0.0, 1.0))
        nociception = float(np.clip(
            0.45 * strain_alarm + 0.75 * self.impact_load + 0.45 * fall_relation,
            0.0, 1.0,
        ))
        pain_rate = 1.0 / (PHI ** (2 if nociception > self.pain else 6))
        self.pain += pain_rate * (nociception - self.pain)
        self.pain = float(np.clip(self.pain, 0.0, 1.0))

        starvation = float(np.clip(
            (self.hunger - STARVATION_THRESHOLD) / (1.0 - STARVATION_THRESHOLD),
            0.0, 1.0,
        ))
        starvation_cost = STARVATION_DAMAGE * starvation
        bodily_cost = PAIN_DAMAGE * self.pain * self.pain
        healing = (
            REST_HEALING * (1.0 - self.vitality)
            if self.pain < 0.12 and self.hunger < 0.30
            else 0.0
        )
        self.vitality = float(np.clip(
            self.vitality + healing - starvation_cost - bodily_cost,
            0.0, 1.0,
        ))
        self.impact_load *= 0.78
        if self.vitality <= 0.0:
            self.die("starvation" if starvation_cost >= bodily_cost else "injury")

    def observe_step(self):
        relation = self.last_control_relation
        if self.dead or relation is None:
            return
        after = self.body_state()
        if after["targetApple"] != relation["targetApple"]:
            self.cycle_phase = 0.0
            self.cycle_progress = 0.0
            self.cycle_heading = 0.0
            self.cycle_yaw = 0.0
            self.cycle_strain = 0.0
            self.cycle_intent = 0.0
            self.cycle_mapping = {}
            self.cycle_ticks = 0
            return
        self.cycle_phase += relation["phaseAdvance"]
        self.cycle_progress += relation["distance"] - after["distance"]
        self.cycle_heading += abs(relation["bearing"]) - abs(after["bearing"])
        after_yaw = float(math.atan2(
            after["rotation"][1, 0], after["rotation"][0, 0]
        ))
        self.cycle_yaw += math.atan2(
            math.sin(after_yaw - relation["yaw"]),
            math.cos(after_yaw - relation["yaw"]),
        )
        self.cycle_strain += relation["strain"]
        self.cycle_intent += relation["intent"]
        mapping = relation["mapping"]
        if mapping is not None:
            self.cycle_mapping[mapping] = self.cycle_mapping.get(mapping, 0) + 1
        self.cycle_ticks += 1
        if self.cycle_phase < self.period:
            return

        ticks = max(1, self.cycle_ticks)
        strain = self.cycle_strain / ticks
        mean_intent = self.cycle_intent / ticks
        self._learn_effect("forward", float(np.clip(
            self.cycle_progress / 0.04, -1.0, 1.0
        )))
        if abs(mean_intent) > 0.20:
            if self.cycle_mapping:
                mapping = max(self.cycle_mapping, key=self.cycle_mapping.get)
                self._learn_effect(mapping, float(np.clip(
                    self.cycle_heading / 0.12, -1.0, 1.0
                )))
        else:
            # A near-straight intention exposes the gait's own curvature.
            # Neutral steering is learned from that realized whole-cycle yaw,
            # never from an authored joint correction.
            self.trim = float(np.clip(
                self.trim - 0.35 * self.cycle_yaw,
                *GROUP_STEERING_RANGE,
            ))

        # The body discovers the smallest enactment that actually carries the
        # inherited gait.  Failure can strengthen it; strain weakens it;
        # successful motion leaves it alone.
        target_enactment = self.gait_enactment
        if strain > 0.50:
            target_enactment = 0.36
        elif self.hunger > 0.25 and self.cycle_progress < 0.004:
            target_enactment = 0.44
        elif self.hunger < 0.18:
            target_enactment = 0.40
        enactment_rate = 1.0 / (
            PHI ** (4 if target_enactment > self.gait_enactment else 7)
        )
        self.gait_enactment += enactment_rate * (
            target_enactment - self.gait_enactment
        )
        self.gait_enactment = float(np.clip(self.gait_enactment, 0.36, 0.44))
        self.lifetime_cycles += 1
        if self.lifetime_cycles % 12 == 0:
            self.save_memory()
        self.cycle_phase %= self.period
        self.cycle_progress = 0.0
        self.cycle_heading = 0.0
        self.cycle_yaw = 0.0
        self.cycle_strain = 0.0
        self.cycle_intent = 0.0
        self.cycle_mapping = {}
        self.cycle_ticks = 0

    def sensor(self, name: str) -> np.ndarray:
        return np.asarray(self.world.data.sensordata[self.sensors[name]], dtype=float)

    def pose(self):
        rotation = self.world.data.xmat[self.base_id].reshape(3, 3)
        position = self.world.data.xpos[self.base_id].copy()
        return position, rotation

    def body_state(self):
        position, rotation = self.pose()
        target_index, target_position, target_distance = self.world.related_apple(
            position, self.target_apple
        )
        self.target_apple = target_index
        to_food = target_position - position
        local_food = rotation.T @ to_food
        bearing = float(math.atan2(local_food[1], local_food[0]))
        up = self.sensor("upvector")
        gyro = self.sensor("gyro")
        velocity = self.sensor("local_linvel")
        lp, rp = self.sensor("left_foot_pos"), self.sensor("right_foot_pos")
        lv, rv = self.sensor("left_foot_global_linvel"), self.sensor("right_foot_global_linvel")
        left_support = float(np.clip((0.055 - lp[2]) / 0.045, 0.0, 1.0))
        right_support = float(np.clip((0.055 - rp[2]) / 0.045, 0.0, 1.0))
        support = max(left_support, right_support)
        slip = float(np.clip(
            (np.linalg.norm(lv[:2]) * left_support + np.linalg.norm(rv[:2]) * right_support) / 0.45,
            0.0, 1.0,
        ))
        roll = float(math.atan2(up[1], max(1e-6, up[2])))
        pitch = float(-math.atan2(up[0], max(1e-6, up[2])))
        tilt = float(np.clip(math.hypot(roll, pitch) / 0.65, 0.0, 1.0))
        return {
            "position": position, "rotation": rotation, "bearing": bearing,
            "targetApple": target_index, "distance": target_distance,
            "upright": float(rotation[2, 2]),
            "support": support, "slip": slip, "tilt": tilt,
            "yawRate": float(gyro[2]), "velocity": velocity,
        }

    def control(self):
        d = self.world.data
        state = self.body_state()
        self.age += 1
        actual = np.asarray(d.qpos[self.joint_qpos], dtype=float)
        if self.dead:
            # A dead creature remains a physical body in the shared world, but
            # its controller follows the joints instead of producing a gait.
            d.ctrl[self.actuator_ids] = np.clip(
                actual,
                self.world.ctrl_lo[self.actuator_ids],
                self.world.ctrl_hi[self.actuator_ids],
            )
            self.last_control_relation = None
            return state

        motion = float(np.clip(np.linalg.norm(state["velocity"][:2]) / 0.35, 0.0, 1.0))
        strain = float(np.clip(0.55 * state["slip"] + 0.45 * state["tilt"], 0.0, 1.0))
        self.reserve = float(np.clip(
            self.reserve - (0.00020 + 0.00015 * motion + 0.00025 * strain), 0.0, 1.0
        ))
        self.hunger = 1.0 - self.reserve
        self.appetite = self.appetite_relation()
        self.update_life(strain, state["upright"])
        if self.dead:
            d.ctrl[self.actuator_ids] = np.clip(
                actual,
                self.world.ctrl_lo[self.actuator_ids],
                self.world.ctrl_hi[self.actuator_ids],
            )
            return state
        if state["upright"] < 0.5:
            d.ctrl[self.actuator_ids] = self.home
            self.fallen_ticks += 1
            self.last_control_relation = None
            return state
        self.fallen_ticks = 0
        target_changed = state["targetApple"] != self.previous_target
        if target_changed:
            self.turn_commitment = 0.0
        progress = 0.0 if target_changed or self.previous_distance is None else (
            self.previous_distance - state["distance"]
        )
        heading_progress = 0.0 if target_changed or self.previous_bearing is None else (
            abs(self.previous_bearing) - abs(state["bearing"])
        )
        self.previous_target = state["targetApple"]
        self.previous_distance = state["distance"]
        self.previous_bearing = state["bearing"]
        distance_relation = float(np.clip(state["distance"] / 0.75, 0.0, 1.0))
        blocked_relation = float(np.clip((0.0005 - progress) / 0.0015, 0.0, 1.0))
        pressure = float(np.clip(
            self.hunger * (0.45 * distance_relation + 0.55 * blocked_relation)
            + 0.25 * self.pain + 0.35 * strain,
            0.0, 1.0,
        ))
        self.drive_pressure += (pressure - self.drive_pressure) / (PHI ** 3)
        normalized = float(np.clip((self.drive_pressure - 0.10) / (0.70 - 0.10), 0.0, 1.0))
        target_resolution = normalized * normalized * (3.0 - 2.0 * normalized)
        relation_rate = 1.0 / (PHI ** (3 if target_resolution > self.drive_resolution else 5))
        self.drive_resolution += relation_rate * (target_resolution - self.drive_resolution)
        joint_velocity = np.asarray(d.qvel[self.joint_dof], dtype=float)
        load = np.clip(np.asarray(d.actuator_force[self.actuator_ids], dtype=float) / 3.23, -1.0, 1.0)
        residual = np.clip(
            (np.asarray(d.ctrl[self.actuator_ids], dtype=float) - actual)
            / np.maximum(self.world.ctrl_hi[self.actuator_ids] - self.world.ctrl_lo[self.actuator_ids], 1e-6),
            -1.0, 1.0,
        )
        if self.crown is not None:
            direction = np.array([math.cos(state["bearing"]), math.sin(state["bearing"]), 0.0])
            reliability = (
                max(0.0, state["upright"])
                * state["support"]
                * (1.0 - state["slip"])
                * (1.0 - 0.50 * self.pain)
            )
            # The crown decides at its own slower rhythm; the embodied loop
            # carries that settled relation between decisions.  This is a
            # genuine two-timescale organism, not skipped physics.
            if self.crown_clock % CROWN_PERIOD == 0:
                out = self.crown.step(
                    joint_velocity, load, residual, direction, self.hunger, reliability
                )
                self.last_disposition = np.asarray(out["disposition"], dtype=float)
                self.last_crown_resolved = bool(out["resolved"])
            self.crown_clock += 1
            horizontal = float(np.linalg.norm(self.last_disposition[:2]))
            # The disposition is a direction, not a lateral command.  Reading
            # only y erased the difference between "forward" and "behind".
            # Its full signed angle now becomes the bounded turn relation.
            intent = float(np.clip(
                math.atan2(self.last_disposition[1], self.last_disposition[0])
                / (math.pi / 2.0), -1.0, 1.0
            )) if self.last_crown_resolved and horizontal > 1e-6 else 0.0
        else:
            intent = float(np.clip(state["bearing"] / (math.pi / 2.0), -1.0, 1.0))
            self.last_disposition = np.array([math.sqrt(max(0.0, 1.0 - intent * intent)), intent, 0.0])
        if abs(state["bearing"]) > math.pi * 0.72 and abs(intent) > 0.35:
            if self.turn_commitment == 0.0:
                self.turn_commitment = math.copysign(1.0, intent)
            intent = self.turn_commitment * max(0.45, abs(intent))
        elif self.turn_commitment != 0.0:
            if abs(state["bearing"]) < math.pi * 0.40:
                self.turn_commitment = 0.0
            else:
                intent = self.turn_commitment * max(0.35, abs(intent))

        # Babble is not random joint noise.  When a whole-gait relation lacks
        # support, the body makes a small bounded variation in that relation
        # and lets the normal effect ledger judge what actually happened.
        exploration_sign = intent if abs(intent) > 0.05 else state["bearing"]
        exploration_key = "leftDirect" if exploration_sign >= 0.0 else "rightDirect"
        evidence = float(np.clip(self.effect_support[exploration_key], 0.0, 1.0))
        under_supported = float(np.clip((0.65 - evidence) / 0.65, 0.0, 1.0))
        crown_open = 0.35 if self.crown is not None and not self.last_crown_resolved else 0.0
        exploration_need = max(under_supported, crown_open)
        self.babble = float(
            BABBLE_MAX
            * exploration_need
            * (1.0 - self.pain)
            * math.sin(GOLDEN_ANGLE * (self.age / self.period + self.index * PHI))
        )
        if abs(self.babble) > 0.01:
            self.babble_ticks += 1
        intent = float(np.clip(intent + self.babble, -1.0, 1.0))

        heading_blocked = float(np.clip(
            (0.0004 - heading_progress) / 0.0012, 0.0, 1.0
        ))
        target_turn_resolution = abs(intent) * heading_blocked
        turn_rate = 1.0 / (
            PHI ** (3 if target_turn_resolution > self.turn_resolution else 6)
        )
        self.turn_resolution += turn_rate * (
            target_turn_resolution - self.turn_resolution
        )
        # Authority is learned from the relation between intended and realized
        # heading change.  It grows when turning fails and relaxes when the
        # body is already carrying the turn.
        self.turn_authority = 0.30 + 0.35 * self.turn_resolution
        stance = state["support"] * (1.0 - state["slip"]) * (1.0 - state["tilt"]) ** 2
        # Resolution changes what the whole gait can attempt, while stance
        # preserves the body's right to refuse urgency it cannot carry.
        self.gait_pace = 1.0
        movement_consent = float(np.clip(
            (0.06 + 0.94 * self.appetite) * (1.0 - 0.75 * self.pain),
            0.03, 1.0,
        ))
        self.gait_amplitude = self.gait_enactment * movement_consent
        turn_polarity, mapping = self.turn_mapping(intent)
        mapped_intent = turn_polarity * intent
        turn_scale = 1.80 if mapped_intent > 0.0 else 1.0
        steer = float(np.clip(
            self.trim + turn_scale * self.turn_authority * mapped_intent * stance,
            *GROUP_STEERING_RANGE,
        ))
        left, right = max(steer, 0.0), max(-steer, 0.0)
        ip = int(self.phase) % self.period
        gait = self.gaits[0][ip] + left * (self.gaits[1][ip] - self.gaits[0][ip]) \
            + right * (self.gaits[2][ip] - self.gaits[0][ip])
        birth_consent = min(1.0, self.age / 120.0)
        template = self.home + (self.gait_amplitude * birth_consent) * (gait - self.home)
        tracking = float(np.mean(np.abs(template - actual)))
        following = float(np.clip(math.exp(-tracking / 0.35) * max(state["upright"], 0.0), 0.0, 1.0))
        embodiment = 0.98 + 0.02 * following
        d.ctrl[self.actuator_ids] = np.clip(
            actual + embodiment * (template - actual),
            self.world.ctrl_lo[self.actuator_ids], self.world.ctrl_hi[self.actuator_ids],
        )
        phase_advance = self.gait_pace
        self.phase -= phase_advance
        self.last_control_relation = {
            "targetApple": state["targetApple"],
            "distance": state["distance"],
            "bearing": state["bearing"],
            "yaw": float(math.atan2(
                state["rotation"][1, 0], state["rotation"][0, 0]
            )),
            "strain": strain,
            "intent": intent,
            "mapping": mapping if abs(intent) > 0.20 else None,
            "phaseAdvance": phase_advance,
        }
        return state

    def feed(self):
        if self.dead:
            return
        self.reserve = float(np.clip(self.reserve + 0.68, 0.0, 1.0))
        self.hunger = 1.0 - self.reserve
        self.appetite = self.appetite_relation()
        self.lifetime_apples += 1
        self.save_memory()

    def recover(self):
        if self.dead:
            return
        d = self.world.data
        position, rotation = self.pose()
        yaw = float(math.atan2(rotation[1, 0], rotation[0, 0]))
        x, y = float(position[0]), float(position[1])
        d.qpos[self.free_qpos:self.free_qpos + 7] = [x, y, 0.15, *yaw_quat(yaw)]
        d.qpos[self.joint_qpos] = self.home
        d.qvel[self.free_dof:self.free_dof + 6] = 0.0
        d.qvel[self.joint_dof] = 0.0
        d.ctrl[self.actuator_ids] = self.home
        self.phase = 0.0
        self.age = 0
        self.previous_target = -1
        self.previous_distance = None
        self.previous_bearing = None
        self.turn_commitment = 0.0
        self.last_control_relation = None
        self.lifetime_falls += 1
        self.save_memory()
        mujoco.mj_forward(self.world.model, d)
        self.was_upright = True
        self.fallen_ticks = 0


class GroupWorld:
    def __init__(self):
        self.memory_dir = group_memory_dir()
        self.body_fingerprint = inheritance_fingerprint()
        self.model = build_model()
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.ctrl_lo = np.asarray(self.model.actuator_ctrlrange[:, 0])
        self.ctrl_hi = np.asarray(self.model.actuator_ctrlrange[:, 1])
        self.apple_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"apple_{index}")
            for index in range(APPLE_COUNT)
        ], dtype=int)
        self.apple_mocaps = np.asarray(self.model.body_mocapid[self.apple_ids], dtype=int)
        self.apple_positions = np.array([[x, y, 0.08] for x, y in INITIAL_APPLES], dtype=float)
        self.data.mocap_pos[self.apple_mocaps] = self.apple_positions
        self.apple_pending = np.zeros(APPLE_COUNT, dtype=bool)
        self.apple_cooldowns = np.zeros(APPLE_COUNT, dtype=int)
        self.death_events = []
        repertoire = dict(np.load(HERE / "instinct_repertoire.npz"))
        self.ducks = [GroupDuck(self, index, repertoire) for index in range(N_DUCKS)]
        for duck in self.ducks:
            self.data.qpos[duck.free_qpos + 2] = 0.15
            self.data.qpos[duck.joint_qpos] = duck.home
            self.data.ctrl[duck.actuator_ids] = duck.home
        mujoco.mj_forward(self.model, self.data)
        self.tick = 0
        self.rng = np.random.default_rng(1701)
        self.active_pairs = set()
        self.contact_events = 0
        self.apples_eaten = 0
        self.knockdowns = []
        self.geom_owner = np.full(self.model.ngeom, -1, dtype=int)
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            while body > 0:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                for index in range(N_DUCKS):
                    if name.startswith(f"d{index}_"):
                        self.geom_owner[geom] = index
                        body = 0
                        break
                else:
                    body = int(self.model.body_parentid[body])

    def contacts(self):
        pairs = set()
        for ci in range(self.data.ncon):
            contact = self.data.contact[ci]
            a, b = int(self.geom_owner[contact.geom1]), int(self.geom_owner[contact.geom2])
            if a < 0 or b < 0 or a == b:
                continue
            pair = tuple(sorted((a, b)))
            pairs.add(pair)
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, ci, force)
            magnitude = float(abs(force[0]))
            for actor, opponent in ((a, b), (b, a)):
                duck = self.ducks[actor]
                duck.last_contact_tick = self.tick
                duck.last_opponent = opponent
                duck.stats.max_contact_force = max(duck.stats.max_contact_force, magnitude)
                if not duck.dead:
                    duck.impact_load = max(
                        duck.impact_load,
                        float(np.clip((magnitude - 4.0) / 32.0, 0.0, 1.0)),
                    )
        for pair in pairs - self.active_pairs:
            self.contact_events += 1
            self.ducks[pair[0]].stats.contacts += 1
            self.ducks[pair[1]].stats.contacts += 1
        self.active_pairs = pairs

    def related_apple(self, position: np.ndarray, current: int = -1):
        available = np.flatnonzero(~self.apple_pending)
        if len(available) == 0:
            return -1, np.asarray(position, dtype=float).copy(), 0.0
        offsets = self.apple_positions[available, :2] - position[:2]
        local = int(np.argmin(np.linalg.norm(offsets, axis=1)))
        nearest = int(available[local])
        nearest_distance = float(np.linalg.norm(offsets[local]))
        index = nearest
        distance = nearest_distance
        if 0 <= current < APPLE_COUNT and not self.apple_pending[current]:
            current_distance = float(np.linalg.norm(
                self.apple_positions[current, :2] - position[:2]
            ))
            # A relationship persists until the alternative is decisively
            # closer.  Voronoi-boundary noise may not continually rewrite it.
            if nearest == current or nearest_distance + TARGET_SWITCH_ADVANTAGE >= current_distance:
                index = current
                distance = current_distance
        return index, self.apple_positions[index].copy(), distance

    def nearest_apple(self, position: np.ndarray):
        return self.related_apple(position, -1)

    def consume_apple(self, index: int):
        self.apples_eaten += 1
        self.apple_pending[index] = True
        self.apple_cooldowns[index] = 100
        # Remove the eaten fruit from both sight and the visible field while it
        # replenishes; its remembered location is retained in apple_positions.
        self.data.mocap_pos[self.apple_mocaps[index]] = [0.0, 0.0, -1.0]

    def replenish_apple(self, index: int):
        candidate = None
        # Each successful relation stretches the next one.  The field begins
        # within the proven gait's reach and grows toward open-floor travel
        # instead of immediately teleporting food to the hardest radius.
        radius_min = min(0.72, 0.32 + 0.015 * self.apples_eaten)
        radius_max = min(1.30, 0.55 + 0.040 * self.apples_eaten)
        for _ in range(64):
            angle = float(self.rng.uniform(-math.pi, math.pi))
            radius = float(self.rng.uniform(radius_min, radius_max))
            trial = np.array([radius * math.cos(angle), radius * math.sin(angle), 0.08])
            other = np.flatnonzero((~self.apple_pending) & (np.arange(APPLE_COUNT) != index))
            clear_of_food = len(other) == 0 or np.all(
                np.linalg.norm(self.apple_positions[other, :2] - trial[:2], axis=1) > 0.22
            )
            clear_of_bodies = all(
                np.linalg.norm(duck.pose()[0][:2] - trial[:2]) > 0.24 for duck in self.ducks
            )
            if clear_of_food and clear_of_bodies:
                candidate = trial
                break
        if candidate is None:
            candidate = trial
        self.apple_positions[index] = candidate
        self.data.mocap_pos[self.apple_mocaps[index]] = candidate
        self.apple_pending[index] = False
        self.apple_cooldowns[index] = 0

    def step(self):
        states = [duck.control() for duck in self.ducks]
        for _ in range(NSUB):
            mujoco.mj_step(self.model, self.data)
            self.contacts()
        self.tick += 1
        for duck in self.ducks:
            duck.observe_step()
        for index in range(APPLE_COUNT):
            if self.apple_pending[index]:
                self.apple_cooldowns[index] -= 1
                if self.apple_cooldowns[index] <= 0:
                    self.replenish_apple(index)
                continue
            living = [duck for duck in self.ducks if not duck.dead]
            if not living:
                continue
            distances = [
                float(np.linalg.norm(duck.pose()[0][:2] - self.apple_positions[index, :2]))
                for duck in living
            ]
            winner = int(np.argmin(distances))
            if distances[winner] < EAT_RADIUS:
                winner_duck = living[winner]
                winner_duck.stats.apples += 1
                winner_duck.feed()
                self.consume_apple(index)
        for duck, before in zip(self.ducks, states):
            upright = duck.body_state()["upright"] >= 0.5
            if not duck.dead and duck.was_upright and not upright:
                duck.stats.falls += 1
                injury = FALL_DAMAGE
                if self.tick - duck.last_contact_tick <= 12 and duck.last_opponent is not None:
                    duck.stats.knockdowns += 1
                    injury += KNOCKDOWN_DAMAGE
                    event = {"tick": self.tick, "fallen": duck.index, "by": duck.last_opponent}
                    self.knockdowns.append(event)
                    duck.injure(injury, "knockdown")
                else:
                    duck.injure(injury, "fall")
            duck.was_upright = upright
            if not duck.dead and not upright and duck.fallen_ticks >= 100:
                duck.recover()

    def report(self):
        return {
            "schema": "relational-duck-group-v4",
            "openFloor": True,
            "ducks": N_DUCKS,
            "living": sum(not duck.dead for duck in self.ducks),
            "crownAuthority": USE_CROWNS,
            "bodyFingerprint": self.body_fingerprint,
            "ticks": self.tick,
            "applesTotal": APPLE_COUNT,
            "applesEaten": self.apples_eaten,
            "applesAvailable": int(np.count_nonzero(~self.apple_pending)),
            "applesReplenishing": int(np.count_nonzero(self.apple_pending)),
            "contactEvents": self.contact_events,
            "knockdownEvents": self.knockdowns,
            "deathEvents": self.death_events,
            "individuals": [
                {
                    "duck": duck.index, "apples": duck.stats.apples,
                    "falls": duck.stats.falls, "knockdowns": duck.stats.knockdowns,
                    "contacts": duck.stats.contacts,
                    "maxContactForce": duck.stats.max_contact_force,
                    "alive": not duck.dead,
                    "deathCause": duck.death_cause,
                    "vitality": duck.vitality,
                    "pain": duck.pain,
                    "impactLoad": duck.impact_load,
                    "distanceToApple": duck.body_state()["distance"],
                    "hunger": duck.hunger, "reserve": duck.reserve,
                    "appetite": duck.appetite,
                    "drivePressure": duck.drive_pressure,
                    "driveResolution": duck.drive_resolution,
                    "gaitPace": duck.gait_pace, "gaitAmplitude": duck.gait_amplitude,
                    "babble": duck.babble, "babbleTicks": duck.babble_ticks,
                    "turnAuthority": duck.turn_authority,
                    "turnResolution": duck.turn_resolution,
                    "bodyMemory": {
                        "cycles": duck.lifetime_cycles,
                        "apples": duck.lifetime_apples,
                        "falls": duck.lifetime_falls,
                        "deaths": duck.lifetime_deaths,
                        "gaitEnactment": duck.gait_enactment,
                        "steeringTrim": duck.trim,
                        "effects": duck.effect_prediction,
                        "support": duck.effect_support,
                    },
                    "lastDisposition": duck.last_disposition.tolist(),
                }
                for duck in self.ducks
            ],
        }


def collision_capability_test() -> dict:
    """Falsifiable impulse test: two uncontrolled bodies collide head-on."""
    world = GroupWorld()
    d = world.data
    left, right = world.ducks[0], world.ducks[1]
    d.qpos[left.free_qpos:left.free_qpos + 7] = [-0.42, 0.0, 0.15, *yaw_quat(0.0)]
    d.qpos[right.free_qpos:right.free_qpos + 7] = [0.0, 0.0, 0.15, *yaw_quat(math.pi / 2.0)]
    d.qpos[world.ducks[2].free_qpos:world.ducks[2].free_qpos + 7] = [0.0, 1.5, 0.15, *yaw_quat(-math.pi / 2)]
    d.qvel[left.free_dof] = 3.0
    d.qvel[right.free_dof] = 0.0
    for duck in world.ducks:
        d.ctrl[duck.actuator_ids] = duck.home
    mujoco.mj_forward(world.model, d)
    minimum = [1.0, 1.0]
    for _ in range(500):
        mujoco.mj_step(world.model, d)
        world.contacts()
        for i, duck in enumerate((left, right)):
            minimum[i] = min(minimum[i], float(d.xmat[duck.base_id].reshape(3, 3)[2, 2]))
    for duck in (left, right):
        duck.update_life(0.0, float(d.xmat[duck.base_id].reshape(3, 3)[2, 2]))
    return {
        "contactEvents": world.contact_events,
        "minimumUpright": minimum,
        "knockable": world.contact_events > 0 and min(minimum) < 0.5,
        "maxContactForce": max(left.stats.max_contact_force, right.stats.max_contact_force),
        "painAfterImpact": [left.pain, right.pain],
        "vitalityAfterImpact": [left.vitality, right.vitality],
    }


def main():
    if "--collision-test" in sys.argv:
        print(json.dumps(collision_capability_test(), indent=2), flush=True)
        return
    world = GroupWorld()
    if not WATCH:
        for _ in range(CONTROL_STEPS):
            world.step()
        print(json.dumps(world.report(), indent=2), flush=True)
        return
    from mujoco.experimental.studio.native_viewer import NativeViewer
    viewer = NativeViewer(world.model, title="Relational Duck Group - Open Floor", width=1280, height=820)
    viewer.camera.distance = 3.0
    viewer.camera.elevation = -31
    viewer.camera.azimuth = 145
    last_report = time.time()
    runtime_log = None
    if world.memory_dir is not None:
        world.memory_dir.mkdir(parents=True, exist_ok=True)
        runtime_log = world.memory_dir / "flock-runtime.log"

    def emit_status(message: str):
        # Windowed PyInstaller applications intentionally have no stdout.
        # Printing to that absent stream used to kill an otherwise healthy
        # viewer at the first five-second status pulse.
        if sys.stdout is not None:
            print(message, flush=True)
        if runtime_log is not None:
            with runtime_log.open("a", encoding="utf-8") as stream:
                stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")

    emit_status(
        f"[launch] fingerprint={world.body_fingerprint[:12]} "
        f"motivation=hunger,pain,babble,death"
    )
    try:
        while viewer.is_running():
            started = time.perf_counter()
            world.step()
            visible_apples = list(world.apple_positions[~world.apple_pending])
            center = np.mean([duck.pose()[0] for duck in world.ducks] + visible_apples, axis=0)
            viewer.camera.lookat[:] = center
            viewer.sync(world.model, world.data)
            if time.time() - last_report > 5.0:
                r = world.report()
                emit_status(
                    f"[group] living={r['living']}/{r['ducks']} "
                    f"contacts={r['contactEvents']} knockdowns={len(r['knockdownEvents'])} "
                    f"apples={[d['apples'] for d in r['individuals']]} "
                    f"hunger={[round(d['hunger'], 2) for d in r['individuals']]} "
                    f"pain={[round(d['pain'], 2) for d in r['individuals']]}"
                )
                last_report = time.time()
            delay = NSUB * world.model.opt.timestep - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
    finally:
        viewer.stop()


if __name__ == "__main__":
    main()
