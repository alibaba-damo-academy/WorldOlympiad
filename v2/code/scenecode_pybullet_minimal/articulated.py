"""Generic DROID articulated-object execution in PyBullet.

This module is deliberately independent from the cup/tray policy scene.  It
loads the official SceneCode SDF after the visual elements have been removed by
``prepare_physics_sdf`` and lets the robot move the certified child link by
contact.  No task-specific dimensions or attachment constraints are used.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .droid_pick_place import OFFICIAL_DROID_FORCES, OFFICIAL_DROID_RESET
from .sdf import BridgeArtifact, _collision_points, prepare_physics_sdf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_gripper_urdf(source: Path, destination: Path, scale: float) -> tuple[str, int]:
    """Create a deterministic URDF derivative that scales gripper geometry only."""

    tree = ET.parse(source)
    changed_links: list[str] = []
    for link in tree.getroot().iter("link"):
        name = str(link.get("name") or "").lower()
        if not any(token in name for token in ("finger", "knuckle", "gripper")):
            continue
        changed = False
        for shape in link.iter():
            tag = shape.tag.rsplit("}", 1)[-1]
            if tag == "mesh":
                existing = [1.0, 1.0, 1.0]
                if shape.get("scale"):
                    existing = [float(value) for value in shape.get("scale").split()]
                    if len(existing) != 3:
                        raise ValueError("gripper mesh scale must contain three values")
                shape.set("scale", " ".join(f"{value * scale:.12g}" for value in existing))
                changed = True
            elif tag in {"box", "cylinder", "sphere"}:
                if tag == "box":
                    key = "size"
                    count = 3
                elif tag == "cylinder":
                    key = "radius"
                    count = 1
                    if shape.get("length"):
                        shape.set("length", f"{float(shape.get('length')) * scale:.12g}")
                else:
                    key = "radius"
                    count = 1
                if shape.get(key):
                    values = [float(value) for value in shape.get(key).split()]
                    if len(values) != count:
                        raise ValueError(f"gripper {tag} geometry has invalid dimensions")
                    shape.set(key, " ".join(f"{value * scale:.12g}" for value in values))
                    changed = True
        if changed:
            changed_links.append(str(link.get("name")))
    if not changed_links:
        raise ValueError("DROID URDF has no scalable gripper geometry links")
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.get("filename")
        if filename and not Path(filename).is_absolute() and not filename.startswith("package://"):
            mesh.set("filename", str((source.parent / filename).resolve()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return _sha256(destination), len(changed_links)


def configure_contact_dynamics(p: Any, body: int, link: int, friction: float) -> None:
    # This server's Bullet build ignores the friction update when collisionMargin
    # is supplied in the same call for a multibody link. Read back the applied value.
    p.changeDynamics(body, link, collisionMargin=0.0)
    p.changeDynamics(body, link, lateralFriction=friction)
    if not math.isclose(float(p.getDynamicsInfo(body, link)[1]), friction, abs_tol=1e-8):
        raise RuntimeError("PyBullet did not apply the requested contact friction")


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _qualified_tag(element: ET.Element, name: str) -> str:
    if element.tag.startswith("{"):
        namespace = element.tag[1:].split("}", 1)[0]
        return f"{{{namespace}}}{name}"
    return name


def _sdf_pose(link: ET.Element) -> tuple[tuple[float, float, float], tuple[float, float, float], str | None]:
    node = next((child for child in link if _local_tag(child.tag) == "pose"), None)
    if node is None or not (node.text or "").strip():
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None
    values = [float(value) for value in node.text.split()]
    if len(values) != 6:
        raise ValueError("SDF link pose must contain six values")
    return tuple(values[:3]), tuple(values[3:]), node.get("relative_to")


def _compose_sdf_pose(parent, child):
    parent_xyz, parent_rpy = parent
    child_xyz, child_rpy = child
    parent_rotation = Rotation.from_euler("xyz", parent_rpy)
    rotation = parent_rotation * Rotation.from_euler("xyz", child_rpy)
    xyz = np.asarray(parent_xyz, dtype=np.float64) + parent_rotation.apply(child_xyz)
    return tuple(float(value) for value in xyz), tuple(float(value) for value in rotation.as_euler("xyz"))


def flatten_sdf_link_poses(artifact: BridgeArtifact) -> BridgeArtifact:
    """Write every link pose in the PyBullet view as a model-frame pose.

    This server's SDF importer resolves a link ``<pose relative_to="...">``
    against the model root rather than the named parent frame.  A certified
    moving subtree (for example a handle fixed to a door) therefore loads at
    the model origin instead of on its parent.  SDF semantics place the pose
    relative to the referenced frame, so the executor resolves the chain once
    and writes the equivalent absolute pose; the official source SDF and every
    collision mesh stay untouched.
    """

    tree = ET.parse(artifact.physics_sdf)
    root = tree.getroot()
    model = next((node for node in root.iter() if _local_tag(node.tag) == "model"), None)
    if model is None:
        return artifact
    links = {
        node.get("name"): node
        for node in model
        if _local_tag(node.tag) == "link" and node.get("name")
    }
    identity = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    resolved: dict[str, tuple] = {}

    def resolve(name: str, stack: frozenset[str]):
        if name in resolved:
            return resolved[name]
        link = links.get(name)
        if link is None or name in stack:
            return identity
        xyz, rpy, relative_to = _sdf_pose(link)
        parent = identity
        if relative_to and relative_to != "__model__" and relative_to != name:
            parent = resolve(relative_to, stack | {name})
        resolved[name] = _compose_sdf_pose(parent, (xyz, rpy))
        return resolved[name]

    for name in links:
        resolve(name, frozenset())
    for name, link in links.items():
        xyz, rpy = resolved[name]
        node = next((child for child in link if _local_tag(child.tag) == "pose"), None)
        if node is None:
            node = ET.Element(_qualified_tag(link, "pose"))
            link.insert(0, node)
        node.text = " ".join(f"{value:.12g}" for value in (*xyz, *rpy))
        node.attrib.pop("relative_to", None)
    temporary = Path(artifact.physics_sdf).with_name(f".{Path(artifact.physics_sdf).name}.flat.tmp")
    try:
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(artifact.physics_sdf)
    finally:
        if temporary.exists():
            temporary.unlink()
    return replace(artifact, physics_sha256=_sha256(Path(artifact.physics_sdf)))


def _unit(value: Sequence[float], *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-9:
        raise ValueError(f"{label} must be non-zero")
    return vector / norm


@dataclass(frozen=True)
class ArticulationBinding:
    """The immutable SceneCode joint binding consumed by the simulator."""

    case_id: str
    component_id: str
    host_object_id: str
    host_sdf: str
    host_sdf_sha256: str
    joint_name: str
    joint_type: str
    parent_link: str
    child_link: str
    axis_xyz: tuple[float, float, float]
    lower_limit: float
    upper_limit: float
    closed_position: float
    open_position: float
    travel: float
    open_delta_xyz: tuple[float, float, float]
    contact_target_local_xyz: tuple[float, float, float] | None = None
    contact_normal_local_xyz: tuple[float, float, float] | None = None
    handle_axis_local_xyz: tuple[float, float, float] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ArticulationBinding":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {
            "case_id",
            "component_id",
            "host_object_id",
            "host_sdf",
            "host_sdf_sha256",
            "joint_name",
            "joint_type",
            "parent_link",
            "child_link",
            "axis_xyz",
            "lower_limit",
            "upper_limit",
            "closed_position",
            "open_position",
            "travel",
            "open_delta_xyz",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"articulation binding is missing fields: {missing}")
        binding = cls(
            case_id=str(payload["case_id"]),
            component_id=str(payload["component_id"]),
            host_object_id=str(payload["host_object_id"]),
            host_sdf=str(payload["host_sdf"]),
            host_sdf_sha256=str(payload["host_sdf_sha256"]),
            joint_name=str(payload["joint_name"]),
            joint_type=str(payload["joint_type"]),
            parent_link=str(payload["parent_link"]),
            child_link=str(payload["child_link"]),
            axis_xyz=tuple(float(value) for value in payload["axis_xyz"]),
            lower_limit=float(payload["lower_limit"]),
            upper_limit=float(payload["upper_limit"]),
            closed_position=float(payload["closed_position"]),
            open_position=float(payload["open_position"]),
            travel=float(payload["travel"]),
            open_delta_xyz=tuple(float(value) for value in payload["open_delta_xyz"]),
            contact_normal_local_xyz=(tuple(payload["contact_normal_local_xyz"]) if payload.get("contact_normal_local_xyz") is not None else None),
            handle_axis_local_xyz=(tuple(payload["handle_axis_local_xyz"]) if payload.get("handle_axis_local_xyz") is not None else None),
            contact_target_local_xyz=(
                tuple(float(value) for value in payload["contact_target_local_xyz"])
                if payload.get("contact_target_local_xyz") is not None else None
            ),
        )
        binding.validate()
        return binding

    def validate(self) -> None:
        if self.joint_type not in {"prismatic", "revolute"}:
            raise ValueError(f"unsupported articulated joint type: {self.joint_type}")
        _unit(self.axis_xyz, label="articulation axis")
        if self.contact_normal_local_xyz is not None or self.handle_axis_local_xyz is not None:
            if self.contact_target_local_xyz is None or self.contact_normal_local_xyz is None or self.handle_axis_local_xyz is None:
                raise ValueError("contact frame requires target, normal, and handle axis")
            normal = _unit(self.contact_normal_local_xyz, label="contact normal")
            handle_axis = _unit(self.handle_axis_local_xyz, label="handle axis")
            if abs(float(np.dot(normal, handle_axis))) > .99:
                raise ValueError("contact normal and handle axis must not be parallel")
        if self.contact_target_local_xyz is not None and len(self.contact_target_local_xyz) != 3:
            raise ValueError("contact target must be a three-vector")
        if self.contact_target_local_xyz is not None and not all(
            math.isfinite(value) for value in self.contact_target_local_xyz
        ):
            raise ValueError("contact target contains non-finite values")
        if not all(
            math.isfinite(value)
            for value in (
                self.lower_limit,
                self.upper_limit,
                self.closed_position,
                self.open_position,
                self.travel,
                *self.open_delta_xyz,
            )
        ):
            raise ValueError("articulation binding contains non-finite limits")
        if self.lower_limit >= self.upper_limit:
            raise ValueError("articulation limits are not ordered")
        if not self.lower_limit <= self.closed_position <= self.upper_limit:
            raise ValueError("closed position lies outside articulation limits")
        if not self.lower_limit <= self.open_position <= self.upper_limit:
            raise ValueError("open position lies outside articulation limits")
        if self.travel <= 1.0e-6 or abs(self.open_position - self.closed_position) <= 1.0e-6:
            raise ValueError("articulation has no usable travel")
        if float(np.linalg.norm(self.open_delta_xyz)) <= 1.0e-6:
            raise ValueError("articulation open delta is degenerate")


@dataclass(frozen=True)
class ArticulatedSceneInputs:
    table_sdf: Path | None
    host_sdf: Path
    panda_urdf: Path
    output_dir: Path
    binding: ArticulationBinding
    host_center_xy: tuple[float, float] = (0.0, 0.01)
    robot_base_xy: tuple[float, float] | None = None
    robot_embodiment: str = "robolab_droid"
    operator: str = "pull"
    success_predicate: str = "joint_at_open_limit"
    initial_arm_joint_positions: tuple[float, float, float, float, float, float, float] = OFFICIAL_DROID_RESET
    arm_joint_forces: tuple[float, float, float, float, float, float, float] = OFFICIAL_DROID_FORCES
    # Arm-side torque scale.  Pulling a handle loads joints 1-4 close to the
    # registered DROID force ceiling, so the wrist can lag its committed arc by
    # more than the grasp can absorb.
    arm_force_scale: float = 1.0
    # Contact-only pulling is best served by a compliant arm: a stiff position
    # loop converts the small handle-arc tracking lag into contact forces large
    # enough to pry the finger pads off a shallow handle.  The planned joint
    # velocity is fed forward (below) so compensation does not cost tracking.
    arm_position_gain: float = 0.1
    # Feed-forward scale for the planned arm joint velocity during actuation.
    # 1.0 commands exactly the reference joint velocity; the position servo's
    # tracking loss at this gain needs a larger value to hold the arc.
    arm_velocity_feedforward: float = 2.0
    # Hold the grasp at the reached configuration for the final few actuation
    # actions instead of driving the handle into its hard joint limit.
    actuation_hold_frames: int = 2
    # Hold a closed grasp after actuation until the articulation is at rest
    # before opening the fingers, so a rebounding handle is not launched away.
    release_settle_velocity_rad_s: float = 0.01
    disable_robot_gravity: bool = True
    action_authority: str = "deterministic articulated contact executor in PyBullet"
    action_counts: tuple[tuple[str, int], ...] = (
        ("settle", 15),
        ("approach", 45),
        ("contact", 24),
        ("actuate", 90),
        ("retreat", 24),
        ("final_settle", 15),
    )
    approach_clearance_m: float = 0.08
    contact_clearance_m: float = 0.012
    open_fraction_required: float = 0.90
    # Experimental reference-video mode.  It keeps the official SDF/joint and
    # records the same trajectory format, but drives the articulated joint with
    # a PyBullet position actuator instead of claiming a DROID contact grasp.
    reference_joint_actuation: bool = False
    # Explicit PyBullet-only calibration variant. Official SceneCode assets
    # remain immutable and these values are recorded in the rollout artifacts.
    gripper_geometry_scale: float = 1.0
    handle_collision_scale: float = 1.0
    contact_force_scale: float = 1.0
    contact_pull_force: float = 0.0
    asset_origin: str = "scenecode"
    gripper_contact_offset_m: float = 0.155
    # The Robotiq finger bodies overhang the pad contact plane towards the
    # wrist.  A handle that stands only a couple of centimetres off its panel
    # would otherwise let that overhang press into the panel as the fingers
    # close.  The grasp is planned this far in front of the certified contact
    # target along the outward normal, so the pads still close on the handle
    # while the finger bodies stay clear of the mounting surface.
    grasp_standoff_m: float = 0.010
    # Optional deeper standoff used only for the actuation arc, so the pads keep
    # a normal load while the pull tries to pry them off the handle.  ``None``
    # keeps the closing standoff.
    actuation_standoff_m: float | None = None
    # Maximum angular lead the commanded grasp pose may run ahead of the
    # measured furniture joint.  The pull force comes from this bounded lead;
    # letting the whole reference-front-to-measured gap accumulate instead
    # walks the pads off a short handle.  The bound is a safety ceiling: a
    # freely moving door keeps the commanded arc as the reference.
    grasp_lead_rad: float = 0.050
    robot_base_z: float = 0.0
    robot_base_yaw: float | None = None
    preserve_host_orientation: bool = False
    # Number of actuation actions for a full travel sweep.  A slower pull keeps
    # the commanded grasp lead over the passive furniture joint small, so the
    # fingers stay engaged on the handle instead of being strained by the whole
    # reference-versus-measured gap.
    actuate_steps: int | None = None
    automatic_robot_placement: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "automatic_robot_placement",
                           self.robot_base_xy is None and self.robot_base_yaw is None
                           and self.robot_base_z == 0.)
        if self.actuate_steps is not None:
            object.__setattr__(self, "action_counts", tuple(
                (name, self.actuate_steps if name == "actuate" else count)
                for name, count in self.action_counts
            ))
        # The registered default workcell has host (0, .01) and robot
        # (.36, -.43). Translate both together when a scene moves the host;
        # an explicitly supplied robot pose remains in world coordinates.
        if self.robot_base_xy is None:
            object.__setattr__(self, "robot_base_xy", (
                self.host_center_xy[0] + .36,
                self.host_center_xy[1] - .44,
            ))
        if self.robot_base_yaw is None:
            object.__setattr__(self, "robot_base_yaw", math.pi / 2.)

    def validate(self) -> None:
        self.binding.validate()
        if self.robot_embodiment != "robolab_droid":
            raise ValueError("articulated reference currently requires robolab_droid")
        if self.operator not in {"pull", "press", "close"}:
            raise ValueError(f"unsupported articulated operator: {self.operator}")
        expected_predicates = {
            "pull": "joint_at_open_limit",
            "press": "joint_at_open_limit",
            "close": "joint_at_closed_limit",
        }
        if self.success_predicate != expected_predicates[self.operator]:
            raise ValueError("articulated operator and success predicate disagree")
        if len(self.initial_arm_joint_positions) != 7 or not np.isfinite(
            np.asarray(self.initial_arm_joint_positions, dtype=np.float64)
        ).all():
            raise ValueError("initial DROID arm position must contain seven finite values")
        if len(self.arm_joint_forces) != 7 or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in self.arm_joint_forces
        ):
            raise ValueError("DROID arm forces must contain seven positive values")
        if not math.isfinite(self.robot_base_z) or self.robot_base_z < 0:
            raise ValueError("robot base elevation must be finite and nonnegative")
        if not math.isfinite(self.robot_base_yaw):
            raise ValueError("robot base yaw must be finite")
        if not .10 <= self.gripper_contact_offset_m <= .20:
            raise ValueError("gripper contact offset is outside the registered pad range")
        if not 0.0 <= self.grasp_standoff_m <= .03:
            raise ValueError("grasp standoff must lie in [0, 0.03] m")
        if self.actuation_standoff_m is not None and not 0.0 <= self.actuation_standoff_m <= self.grasp_standoff_m:
            raise ValueError("actuation standoff must lie in [0, grasp standoff] m")
        if not .002 <= self.grasp_lead_rad <= .06:
            raise ValueError("grasp lead must lie in [0.002, 0.06] rad")
        if self.binding.contact_normal_local_xyz is not None and (
            self.reference_joint_actuation or self.gripper_geometry_scale != 1 or self.handle_collision_scale != 1 or self.contact_pull_force != 0
        ):
            raise ValueError("contact-frame execution requires unassisted native geometry")
        if not 0 < self.arm_position_gain <= 1:
            raise ValueError("arm position gain must lie in (0, 1]")
        if not 0.0 <= self.arm_velocity_feedforward <= 4.0:
            raise ValueError("arm velocity feedforward must lie in [0, 4]")
        if not 0 <= self.actuation_hold_frames <= 30:
            raise ValueError("actuation hold frames must lie in [0, 30]")
        if not 0.0 <= self.release_settle_velocity_rad_s <= 1.0:
            raise ValueError("release settle velocity must lie in [0, 1] rad/s")
        if not 0.5 <= self.gripper_geometry_scale <= 1.0:
            raise ValueError("gripper geometry scale must lie in [0.5, 1.0]")
        if not 1.0 <= self.handle_collision_scale <= 3.0:
            raise ValueError("handle collision scale must lie in [1.0, 3.0]")
        if not 1.0 <= self.contact_force_scale <= 4.0:
            raise ValueError("contact force scale must lie in [1.0, 4.0]")
        if not 1.0 <= self.arm_force_scale <= 4.0:
            raise ValueError("arm force scale must lie in [1.0, 4.0]")
        if not 0.0 <= self.contact_pull_force <= 200.0:
            raise ValueError("contact pull force must lie in [0, 200] N")
        if len(self.robot_base_xy) != 2 or not np.isfinite(np.asarray(self.robot_base_xy, dtype=np.float64)).all():
            raise ValueError("robot base position must contain two finite values")
        if not (0 <= self.contact_clearance_m < self.approach_clearance_m):
            raise ValueError("articulated approach clearances are invalid")
        if not 0.5 <= self.open_fraction_required <= 1.0:
            raise ValueError("open fraction requirement must lie in [0.5, 1]")
        if not self.action_counts or any(count <= 0 for _, count in self.action_counts):
            raise ValueError("articulated action counts must be positive")
        if self.actuate_steps is not None and not 30 <= self.actuate_steps <= 900:
            raise ValueError("articulated actuation step count must lie in [30, 900]")


@dataclass(frozen=True)
class ArticulatedSceneResult:
    trajectory_path: str
    result_path: str
    frame_count: int
    simulation_seconds: float
    joint_name: str
    initial_joint_position: float
    final_joint_position: float
    open_fraction: float
    contact_frame_count: int
    attachment_constraint_count: int
    gripper_geometry_scale: float
    handle_collision_scale: float
    scaled_handle_links: tuple[str, ...]
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PyBulletDroidArticulatedScene:
    """Run one certified articulated operator in a persistent PyBullet scene."""

    physics_hz = 240
    control_hz = 15
    physics_steps_per_action = physics_hz // control_hz
    droid_gripper_closed_rad = math.pi / 4.0
    droid_gripper_multipliers = {
        "finger_joint": 1.0,
        "right_outer_knuckle_joint": -1.0,
        "left_inner_finger_knuckle_joint": 1.0,
        "right_inner_finger_knuckle_joint": -1.0,
        "left_inner_finger_joint": -1.0,
        "right_inner_finger_joint": 1.0,
    }

    def __init__(self, inputs: ArticulatedSceneInputs) -> None:
        inputs.validate()
        try:
            import pybullet as bullet
            import pybullet_data
        except ImportError as exc:  # pragma: no cover - server-only dependency
            raise RuntimeError("PyBullet is required for articulated execution") from exc
        self.p = bullet
        self.pybullet_data = pybullet_data
        self.inputs = inputs
        self.output_dir = inputs.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if _sha256(inputs.host_sdf.resolve()) != inputs.binding.host_sdf_sha256:
            raise ValueError("articulated host SDF digest differs from binding certificate")
        self.scaled_handle_links = self._resolve_handle_links()
        physics_dir = self.output_dir / "physics_sdf"
        self.prepared_table = (
            flatten_sdf_link_poses(prepare_physics_sdf(inputs.table_sdf.resolve(), physics_dir / "table.sdf"))
            if inputs.table_sdf is not None
            else None
        )
        self.prepared_host = flatten_sdf_link_poses(prepare_physics_sdf(
            inputs.host_sdf.resolve(),
            physics_dir / "host.sdf",
            collision_scales_by_link={
                name: inputs.handle_collision_scale
                for name in self.scaled_handle_links
            },
        ))
        self.gripper_urdf = inputs.panda_urdf.resolve()
        self.gripper_urdf_sha256 = _sha256(self.gripper_urdf)
        self.gripper_scaled_link_count = 0
        if inputs.gripper_geometry_scale != 1.0:
            self.gripper_urdf = physics_dir / "droid_gripper_variant.urdf"
            self.gripper_urdf_sha256, self.gripper_scaled_link_count = _prepare_gripper_urdf(
                inputs.panda_urdf.resolve(),
                self.gripper_urdf,
                inputs.gripper_geometry_scale,
            )
        self.client = self.p.connect(self.p.DIRECT)
        if self.client < 0:
            raise RuntimeError("failed to connect to PyBullet in DIRECT mode")
        self.frames: list[dict[str, Any]] = []
        self.physics_step = 0
        self.contact_frame_count = 0
        self._reference_joint_target: float | None = None
        self.external_assist_step_count = 0
        self.planning_evidence: dict[str, Any] = {}
        self._closed = False
        try:
            self._build_scene()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "PyBulletDroidArticulatedScene":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.p.disconnect(self.client)
            self._closed = True

    def _resolve_handle_links(self) -> tuple[str, ...]:
        """Select handle-bearing links from the certified SDF structure."""

        root = ET.parse(self.inputs.host_sdf.resolve()).getroot()
        names = tuple(
            sorted(
                str(link.get("name"))
                for link in root.iter()
                if link.tag.rsplit("}", 1)[-1] == "link"
                and link.get("name")
                and "handle" in str(link.get("name")).lower()
            )
        )
        exposed = tuple(
            name
            for name in names
            if name != self.inputs.binding.child_link and "body" not in name.lower()
        )
        if exposed:
            names = exposed
        else:
            names = (self.inputs.binding.child_link,)
        # Some assets model the handle as collision geometry on the moving
        # child itself. In that case the certified child link is the only
        # deterministic fallback; no case name or dimensions are consulted.
        return names

    def _build_scene(self) -> None:
        p = self.p
        p.resetSimulation()
        p.setAdditionalSearchPath(self.pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.physics_hz)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / self.physics_hz,
            numSolverIterations=200,
            deterministicOverlappingPairs=1,
        )
        self.ground = p.loadURDF("plane.urdf")
        self.table = (
            p.loadSDF(self.prepared_table.physics_sdf)[0]
            if self.prepared_table is not None
            else self.ground
        )
        self.host = p.loadSDF(self.prepared_host.physics_sdf)[0]
        self._bind_joint()
        # Articulated furniture is a fixed support; only its declared child
        # links may move.  A dynamic base would let the robot push the whole
        # cabinet during approach and corrupt the reference trajectory.
        p.changeDynamics(self.host, -1, mass=0.0)
        p.changeDynamics(self.table, -1, mass=0.0, lateralFriction=0.8)
        self._place_host()
        self._load_robot()
        for _ in range(self.physics_hz):
            self._step_physics_once()
        # The warm-up settles the imported SDF contacts.  Re-apply the
        # certified DROID conditioning pose afterwards so collision impulses
        # from the warm-up cannot silently change the controller's reference
        # configuration before the first trajectory frame.
        for joint, position in zip(
            self.arm_joints, self.inputs.initial_arm_joint_positions, strict=True
        ):
            p.resetJointState(self.panda, joint, float(position))
            self._command_arm_joint(joint, float(position))
        for joint in self.gripper_joints.values():
            p.resetJointState(self.panda, joint, 0.0)
        self._command_droid_gripper(0.0)
        self.physics_step = 0
        initial = (
            self.inputs.binding.closed_position
            if self.inputs.operator != "close"
            else self.inputs.binding.open_position
        )
        self._set_joint_state(initial)
        p.setJointMotorControl2(self.host, self.articulation_joint, p.VELOCITY_CONTROL, force=0.0)
        self._record("conditioning", None)

    def _bind_joint(self) -> None:
        p = self.p
        binding = self.inputs.binding
        self.host_joints: dict[str, int] = {}
        link_names: dict[str, int] = {}
        for index in range(p.getNumJoints(self.host)):
            info = p.getJointInfo(self.host, index)
            name = info[1].decode("utf-8")
            child = info[12].decode("utf-8")
            self.host_joints[name] = index
            link_names[child] = index
        if binding.joint_name not in self.host_joints:
            raise ValueError(f"binding joint is absent from PyBullet SDF: {binding.joint_name}")
        if binding.child_link not in link_names:
            raise ValueError(f"binding child link is absent from PyBullet SDF: {binding.child_link}")
        self.articulation_joint = self.host_joints[binding.joint_name]
        info = p.getJointInfo(self.host, self.articulation_joint)
        expected = {"prismatic": p.JOINT_PRISMATIC, "revolute": p.JOINT_REVOLUTE}[binding.joint_type]
        if int(info[2]) != expected:
            raise ValueError("PyBullet joint type differs from articulation binding")
        self.articulation_child_link = link_names[binding.child_link]
        moving_links = {self.articulation_child_link}
        # Handles can be separate links rigidly attached to the selected door.
        # Do not include another movable joint or an unrelated cabinet handle.
        while True:
            descendants = {index for index in range(p.getNumJoints(self.host))
                           if p.getJointInfo(self.host, index)[2] == p.JOINT_FIXED
                           and p.getJointInfo(self.host, index)[16] in moving_links}
            if descendants <= moving_links:
                break
            moving_links.update(descendants)
        self.contact_target_links = {
            link_names[name] for name in self.scaled_handle_links
            if name in link_names and link_names[name] in moving_links
        } or {self.articulation_child_link}
        self.articulation_parent_link = next(
            (
                index
                for index in range(p.getNumJoints(self.host))
                if p.getJointInfo(self.host, index)[12].decode("utf-8") == binding.parent_link
            ),
            -1,
        )
        self.joint_name = binding.joint_name
        self.joint_lower = float(info[8])
        self.joint_upper = float(info[9])
        if not (self.joint_lower <= binding.closed_position <= self.joint_upper):
            raise ValueError("closed binding position is outside PyBullet joint limits")
        if not (self.joint_lower <= binding.open_position <= self.joint_upper):
            raise ValueError("open binding position is outside PyBullet joint limits")

    def _place_host(self) -> None:
        p = self.p
        # Align the certified opening direction with the robot's approach side
        # while preserving the generated asset's shape and joint semantics.  A
        # SceneCode model root may contain an arbitrary yaw (the official drawer
        # does); leaving it untouched can place the active handle behind an
        # unreachable IK branch.  This is an instance transform, not a collision
        # or articulation edit.
        initial_q = (
            self.inputs.binding.closed_position
            if self.inputs.operator != "close"
            else self.inputs.binding.open_position
        )
        self._set_joint_state(initial_q)
        closed_pose = np.asarray(self._child_link_pose()[0], dtype=np.float64)
        self._set_joint_state(self.inputs.binding.open_position)
        open_pose = np.asarray(self._child_link_pose()[0], dtype=np.float64)
        self._set_joint_state(initial_q)
        travel_xy = open_pose[:2] - closed_pose[:2]
        if self.inputs.binding.contact_normal_local_xyz is not None:
            # A hinge arc chord is not the approach normal. Use the certified
            # handle frame, transformed by the actual imported child pose.
            child_rotation = np.asarray(p.getMatrixFromQuaternion(self._child_link_pose()[1])).reshape(3, 3)
            travel_xy = (child_rotation @ np.asarray(self.inputs.binding.contact_normal_local_xyz))[:2]
        elif np.linalg.norm(travel_xy) <= 1.0e-6 and self.inputs.binding.joint_type == "revolute":
            travel_xy = np.asarray(self.inputs.binding.open_delta_xyz[:2], dtype=np.float64)
        approach_xy = np.asarray(self.inputs.robot_base_xy, dtype=np.float64) - np.asarray(
            self.inputs.host_center_xy, dtype=np.float64
        )
        if np.linalg.norm(travel_xy) > 1.0e-6 and np.linalg.norm(approach_xy) > 1.0e-6:
            current_orientation = p.getBasePositionAndOrientation(self.host)[1]
            desired_angle = math.atan2(float(approach_xy[1]), float(approach_xy[0]))
            travel_angle = math.atan2(float(travel_xy[1]), float(travel_xy[0]))
            yaw_delta = desired_angle - travel_angle
            yaw_offset = p.getQuaternionFromEuler((0.0, 0.0, yaw_delta))
            aligned_orientation = p.multiplyTransforms(
                (0.0, 0.0, 0.0), yaw_offset,
                (0.0, 0.0, 0.0), current_orientation,
            )[1]
        else:
            aligned_orientation = p.getBasePositionAndOrientation(self.host)[1]
        if self.inputs.preserve_host_orientation:
            aligned_orientation = p.getBasePositionAndOrientation(self.host)[1]
        current = p.getBasePositionAndOrientation(self.host)
        p.resetBasePositionAndOrientation(self.host, current[0], aligned_orientation)
        bounds = [p.getAABB(self.host, index) for index in range(-1, p.getNumJoints(self.host))
                  if p.getCollisionShapeData(self.host, index)]
        lower = np.min([bound[0] for bound in bounds], axis=0)
        upper = np.max([bound[1] for bound in bounds], axis=0)
        table_top = p.getAABB(self.table)[1][2]
        center = ((lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2)
        delta = (
            self.inputs.host_center_xy[0] - center[0],
            self.inputs.host_center_xy[1] - center[1],
            table_top + 0.002 - lower[2],
        )
        p.resetBasePositionAndOrientation(
            self.host,
            tuple(current[0][axis] + delta[axis] for axis in range(3)),
            aligned_orientation,
        )
        for link_index in range(-1, p.getNumJoints(self.host)):
            configure_contact_dynamics(p, self.host, link_index, 0.9)

    def _load_robot(self) -> None:
        p = self.p
        table_top = p.getAABB(self.table)[1][2]
        self.panda_base = (*self.inputs.robot_base_xy, table_top + 0.002 + self.inputs.robot_base_z)
        self.robot_mount = None
        if self.inputs.robot_base_z > 0:
            half = (0.11, 0.11, self.inputs.robot_base_z / 2)
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=(.24, .26, .28, 1))
            self.robot_mount = p.createMultiBody(
                0, collision, visual,
                basePosition=(*self.inputs.robot_base_xy, table_top + .002 + half[2]),
            )
        self.panda = p.loadURDF(
            str(self.gripper_urdf),
            basePosition=self.panda_base,
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, self.inputs.robot_base_yaw)),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        joint_by_name = {
            p.getJointInfo(self.panda, index)[1].decode("utf-8"): index
            for index in range(p.getNumJoints(self.panda))
        }
        self.arm_joints = [joint_by_name[f"panda_joint{index}"] for index in range(1, 8)]
        self.gripper_joints = {
            name: joint_by_name[name] for name in self.droid_gripper_multipliers
        }
        self.gripper_actuator_forces = {
            name: float(p.getJointInfo(self.panda, joint)[10])
            for name, joint in self.gripper_joints.items()
        }
        if any(force <= 0.0 for force in self.gripper_actuator_forces.values()):
            raise ValueError("gripper asset declares a non-positive joint effort")
        link_by_name = {
            p.getJointInfo(self.panda, index)[12].decode("utf-8"): index
            for index in range(p.getNumJoints(self.panda))
        }
        self.finger_contact_links = [
            link_by_name["left_inner_finger"],
            link_by_name["right_inner_finger"],
        ]
        self.end_effector = link_by_name["base_link"]
        self.arm_force_by_joint = {
            joint: force * self.inputs.arm_force_scale * self.inputs.contact_force_scale
            for joint, force in zip(self.arm_joints, self.inputs.arm_joint_forces, strict=True)
        }
        for joint, position in zip(self.arm_joints, self.inputs.initial_arm_joint_positions, strict=True):
            p.resetJointState(self.panda, joint, float(position))
            self._command_arm_joint(joint, float(position))
        for joint in self.gripper_joints.values():
            p.resetJointState(self.panda, joint, 0.0)
        self._command_droid_gripper(0.0)
        for link in self.finger_contact_links:
            configure_contact_dynamics(p, self.panda, link, 4.0)

    def _command_arm_joint(self, joint: int, position: float, velocity: float = 0.0) -> None:
        self.p.setJointMotorControl2(
            self.panda,
            joint,
            self.p.POSITION_CONTROL,
            targetPosition=position,
            targetVelocity=velocity,
            force=self.arm_force_by_joint[joint],
            positionGain=self.inputs.arm_position_gain,
            velocityGain=1.0,
        )

    def _command_droid_gripper(self, closed_fraction: float) -> None:
        target = float(np.clip(closed_fraction, 0.0, 1.0) * self.droid_gripper_closed_rad)
        for name, multiplier in self.droid_gripper_multipliers.items():
            # Drive each finger joint at the effort its own robot asset declares.
            # The four-bar Robotiq fingers are back-drivable, so an arbitrary
            # lower actuator cap lets the pull reaction open the grasp.
            self.p.setJointMotorControl2(
                self.panda,
                self.gripper_joints[name],
                self.p.POSITION_CONTROL,
                targetPosition=multiplier * target,
                force=self.gripper_actuator_forces[name] * self.inputs.contact_force_scale,
                positionGain=0.3,
                velocityGain=1.0,
            )

    def _step_physics_once(self) -> None:
        if self.inputs.disable_robot_gravity:
            for link_index in range(self.p.getNumJoints(self.panda)):
                mass = float(self.p.getDynamicsInfo(self.panda, link_index)[0])
                if mass <= 0:
                    continue
                com = self.p.getLinkState(self.panda, link_index)[0]
                self.p.applyExternalForce(
                    self.panda,
                    link_index,
                    (0.0, 0.0, mass * 9.81),
                    com,
                    self.p.WORLD_FRAME,
                )
        self.p.stepSimulation()

    def _set_joint_state(self, position: float) -> None:
        self.p.resetJointState(self.host, self.articulation_joint, float(position))

    def _joint_position(self) -> float:
        return float(self.p.getJointState(self.host, self.articulation_joint)[0])

    def _droid_gripper_position(self) -> float:
        return float(
            np.clip(
                self.p.getJointState(self.panda, self.gripper_joints["finger_joint"])[0]
                / self.droid_gripper_closed_rad,
                0.0,
                1.0,
            )
        )

    def _child_link_pose(self) -> tuple[Sequence[float], Sequence[float]]:
        state = self.p.getLinkState(
            self.host, self.articulation_child_link, computeForwardKinematics=True
        )
        return state[4], state[5]

    def _collision_surface_target(
        self, direction_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Use the outermost official SDF collision geom as contact surface."""

        root = ET.parse(self.prepared_host.physics_sdf).getroot()
        child = next(
            (
                node
                for node in root.iter()
                if node.tag.rsplit("}", 1)[-1] == "link"
                and node.get("name") == self.inputs.binding.child_link
            ),
            None,
        )
        if child is None:
            raise RuntimeError("articulation child link is absent from physics SDF")
        link_position, link_orientation = self._child_link_pose()
        if self.inputs.binding.contact_target_local_xyz is not None:
            world_target = np.asarray(
                self.p.multiplyTransforms(
                    link_position,
                    link_orientation,
                    self.inputs.binding.contact_target_local_xyz,
                    (0.0, 0.0, 0.0, 1.0),
                )[0],
                dtype=np.float64,
            )
            return world_target + direction_world * self.inputs.contact_clearance_m, np.ones(3), {
                "collision_name": "declared_contact_target",
                "collision_component_count": 0,
                "contact_target_local_xyz": list(self.inputs.binding.contact_target_local_xyz),
            }
        inverse_orientation = self.p.invertTransform(
            (0.0, 0.0, 0.0), link_orientation
        )[1]
        direction_local = np.asarray(
            self.p.multiplyTransforms(
                (0.0, 0.0, 0.0), inverse_orientation,
                direction_world.tolist(), (0.0, 0.0, 0.0, 1.0)
            )[0], dtype=np.float64
        )
        direction_local /= np.linalg.norm(direction_local)
        candidates = []
        for collision in list(child):
            if collision.tag.rsplit("}", 1)[-1] != "collision":
                continue
            points = np.asarray(_collision_points(collision), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
                continue
            low, high = points.min(axis=0), points.max(axis=0)
            half = (high - low) / 2.0
            if not np.isfinite(half).all() or np.any(half <= 1.0e-7):
                continue
            center = (low + high) / 2.0
            outer = float(np.dot(center, direction_local) + np.dot(np.abs(direction_local), half))
            candidates.append((outer, float(np.prod(2.0 * half)), collision.get("name", "collision"), center, half))
        if not candidates:
            raise RuntimeError("articulation child link has no usable collision geometry")
        outer, volume, name, center, half = max(candidates, key=lambda item: (item[0], -item[1], item[2]))
        local_surface = center + direction_local * float(np.dot(np.abs(direction_local), half))
        world_surface = np.asarray(
            self.p.multiplyTransforms(
                link_position, link_orientation, local_surface.tolist(),
                (0.0, 0.0, 0.0, 1.0)
            )[0], dtype=np.float64
        )
        return world_surface + direction_world * self.inputs.contact_clearance_m, half, {
            "collision_name": name,
            "collision_component_count": len(candidates),
            "collision_aabb_center_link": center.tolist(),
            "collision_aabb_half_link": half.tolist(),
            "outer_projection_link_m": outer,
            "collision_aabb_volume_m3": volume,
        }

    def _contact_evidence(self) -> list[dict[str, Any]]:
        contacts = self.p.getContactPoints(
            bodyA=self.panda,
            bodyB=self.host,
        )
        return [
            {
                "robot_link_index": int(contact[3]),
                "host_link_index": int(contact[4]),
                "position_on_robot_world": list(contact[5]),
                "position_on_host_world": list(contact[6]),
                "normal_on_host_world": list(contact[7]),
                "contact_distance_m": float(contact[8]),
                "normal_force_n": float(contact[9]),
            }
            for contact in contacts
            if int(contact[3]) in self.finger_contact_links
            and int(contact[4]) in self.contact_target_links
        ]

    def _robot_base_link_pose(self) -> tuple[Sequence[float], Sequence[float]]:
        # Bullet reports the floating-base COM; URDF loading and Drake welding
        # consume the root-link frame. Preserve both rather than adding its
        # inertial offset to every replay.
        world_com = self.p.getBasePositionAndOrientation(self.panda)
        dynamics = self.p.getDynamicsInfo(self.panda, -1)
        com_from_link = self.p.invertTransform(dynamics[3], dynamics[4])
        return self.p.multiplyTransforms(*world_com, *com_from_link)

    def _record(self, phase: str, commanded_action: list[float] | None) -> None:
        p = self.p
        ee = p.getLinkState(self.panda, self.end_effector, computeForwardKinematics=True)
        contacts = self._contact_evidence()
        if contacts:
            self.contact_frame_count += 1
        robot_link_positions = {
            p.getJointInfo(self.panda, link)[12].decode("utf-8"): list(
                p.getLinkState(self.panda, link, computeForwardKinematics=True)[4]
            )
            for link in self.finger_contact_links
        }
        host_links: dict[str, dict[str, Any]] = {}
        for link_index in range(-1, p.getNumJoints(self.host)):
            if link_index == -1:
                position, orientation = p.getBasePositionAndOrientation(self.host)
                name = p.getBodyInfo(self.host)[0].decode("utf-8") or "base_link"
            else:
                info = p.getJointInfo(self.host, link_index)
                name = info[12].decode("utf-8")
                state = p.getLinkState(self.host, link_index, computeForwardKinematics=True)
                position, orientation = state[4], state[5]
            host_links[name] = {
                "position": list(position),
                "orientation_xyzw": list(orientation),
                "link_index": link_index,
            }
        self.frames.append(
            {
                "frame": len(self.frames),
                "sim_step": self.physics_step,
                "time_s": len(self.frames) / self.control_hz,
                "phase": phase,
                "operator_step_index": 0,
                "active_subject_id": self.inputs.binding.component_id,
                "active_target_id": self.inputs.binding.host_object_id,
                "active_operator": self.inputs.operator,
                "active_success_predicate": self.inputs.success_predicate,
                "policy_source": {"commanded_action": commanded_action},
                "robot": {
                    "base_position": list(self._robot_base_link_pose()[0]),
                    "base_orientation_xyzw": list(self._robot_base_link_pose()[1]),
                    "base_pose_frame": "urdf_root_link",
                    "base_com_position": list(p.getBasePositionAndOrientation(self.panda)[0]),
                    "base_com_orientation_xyzw": list(p.getBasePositionAndOrientation(self.panda)[1]),
                    "joint_names": [f"panda_joint{index}" for index in range(1, 8)]
                    + list(self.droid_gripper_multipliers),
                    "joint_positions": [
                        float(p.getJointState(self.panda, joint)[0])
                        for joint in self.arm_joints
                    ]
                    + [float(p.getJointState(self.panda, self.gripper_joints[name])[0]) for name in self.droid_gripper_multipliers],
                    "end_effector_position": list(ee[4]),
                    "end_effector_orientation_xyzw": list(ee[5]),
                    "finger_link_positions": robot_link_positions,
                    "droid_gripper_position": self._droid_gripper_position(),
                },
                "objects": {
                    self.inputs.binding.host_object_id: {
                        "position": list(p.getBasePositionAndOrientation(self.host)[0]),
                        "orientation_xyzw": list(p.getBasePositionAndOrientation(self.host)[1]),
                        "link_position": list(p.getBasePositionAndOrientation(self.host)[0]),
                        "link_orientation_xyzw": list(p.getBasePositionAndOrientation(self.host)[1]),
                        "role": "support",
                        "link_states": host_links,
                    }
                },
                "articulated": {
                    "component_id": self.inputs.binding.component_id,
                    "host_object_id": self.inputs.binding.host_object_id,
                    "joint_name": self.joint_name,
                    "joint_type": self.inputs.binding.joint_type,
                    "parent_link": self.inputs.binding.parent_link,
                    "child_link": self.inputs.binding.child_link,
                    "position": self._joint_position(),
                    "velocity": float(self.p.getJointState(self.host, self.articulation_joint)[1]),
                    "closed_position": self.inputs.binding.closed_position,
                    "open_position": self.inputs.binding.open_position,
                    "contact_evidence": contacts,
                },
                "robot_articulation_contact_evidence": contacts,
            }
        )

    def _execute_action(self, arm: Sequence[float], gripper: float, phase: str,
                        arm_velocity: Sequence[float] | None = None) -> None:
        values = np.asarray(arm, dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("articulated arm action must contain seven finite values")
        velocities = None if arm_velocity is None else np.asarray(arm_velocity, dtype=np.float64)
        for index, (joint, position) in enumerate(zip(self.arm_joints, values, strict=True)):
            info = self.p.getJointInfo(self.panda, joint)
            position = float(np.clip(position, info[8], info[9]))
            self._command_arm_joint(
                joint, position, 0.0 if velocities is None else float(velocities[index])
            )
        if self._reference_joint_target is not None:
            self.p.setJointMotorControl2(
                self.host,
                self.articulation_joint,
                self.p.POSITION_CONTROL,
                targetPosition=float(self._reference_joint_target),
                force=250.0,
                positionGain=1.0,
                velocityGain=0.5,
            )
        self._command_droid_gripper(gripper)
        for _ in range(self.physics_steps_per_action):
            if phase == "actuate" and self.inputs.contact_pull_force > 0 and self._reference_joint_target is None:
                contacts = self.p.getContactPoints(
                    bodyA=self.panda, bodyB=self.host, linkIndexB=self.articulation_child_link
                )
                if contacts:
                    self.external_assist_step_count += 1
                    direction = np.asarray(self._travel_direction_world, dtype=np.float64)
                    point = np.asarray(contacts[0][6], dtype=np.float64)
                    self.p.applyExternalForce(
                        self.host,
                        self.articulation_child_link,
                        (direction * self.inputs.contact_pull_force).tolist(),
                        point.tolist(),
                        self.p.WORLD_FRAME,
                    )
            self._step_physics_once()
            self.physics_step += 1
        self._record(phase, [*values.tolist(), float(gripper)])

    def _move_to(self, target: np.ndarray, count: int, orientation: Sequence[float], phase: str, gripper: float) -> None:
        start = np.asarray(
            self.p.getLinkState(self.panda, self.end_effector, computeForwardKinematics=True)[4],
            dtype=np.float64,
        )
        for index in range(count):
            alpha = (index + 1) / count
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            point = start + (target - start) * smooth
            solution = self.p.calculateInverseKinematics(
                self.panda,
                self.end_effector,
                point.tolist(),
                orientation,
                maxNumIterations=300,
                residualThreshold=1e-6,
            )
            if len(solution) < 7 or not np.isfinite(solution[:7]).all():
                raise RuntimeError(f"PyBullet IK failed for articulated phase {phase}")
            self._execute_action(solution[:7], gripper, phase)

    def _contact_configuration(
        self,
        desired_finger_midpoint: np.ndarray,
        orientation: Sequence[float],
        *,
        prefer_contact: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Find a reachable arm configuration whose real fingers approach a target.

        PyBullet's IK target is the Robotiq base link, not the midpoint between
        the two contact fingers.  A single Cartesian target can therefore select
        a distant IK branch for a portable DROID URDF.  The deterministic search
        below evaluates a fixed offset lattice against forward-kinematic finger
        positions and returns the best actual configuration.  It is only a
        planner: final motion still goes through position motors and contact
        evidence is collected from the live simulation.
        """

        p = self.p
        desired = np.asarray(desired_finger_midpoint, dtype=np.float64)
        if desired.shape != (3,) or not np.isfinite(desired).all():
            raise ValueError("articulated contact target must be a finite 3-vector")
        current = self._arm_position()
        lower = [float(p.getJointInfo(self.panda, joint)[8]) for joint in self.arm_joints]
        upper = [float(p.getJointInfo(self.panda, joint)[9]) for joint in self.arm_joints]
        ranges = [max(upper[index] - lower[index], 1.0) for index in range(7)]
        state_id = p.saveState()
        best_solution: np.ndarray | None = None
        best_error = float("inf")
        best_score = float("inf")
        try:
            offsets = (
                -0.45,
                -0.30,
                -0.15,
                0.0,
                0.15,
                0.30,
                0.45,
            )
            for offset_x in offsets:
                for offset_y in offsets:
                    for offset_z in offsets:
                        target = desired + np.asarray(
                            (offset_x, offset_y, offset_z), dtype=np.float64
                        )
                        solution = p.calculateInverseKinematics(
                            self.panda,
                            self.end_effector,
                            target.tolist(),
                            orientation,
                            lowerLimits=lower,
                            upperLimits=upper,
                            jointRanges=ranges,
                            restPoses=current.tolist(),
                            maxNumIterations=300,
                            residualThreshold=1e-5,
                        )
                        if len(solution) < 7 or not np.isfinite(solution[:7]).all():
                            continue
                        clipped_solution = np.clip(
                            np.asarray(solution[:7], dtype=np.float64),
                            np.asarray(lower, dtype=np.float64),
                            np.asarray(upper, dtype=np.float64),
                        )
                        for index, joint in enumerate(self.arm_joints):
                            p.resetJointState(
                                self.panda,
                                joint,
                                float(clipped_solution[index]),
                            )
                        midpoint = np.mean(
                            np.asarray(
                                [
                                    p.getLinkState(
                                        self.panda,
                                        link,
                                        computeForwardKinematics=True,
                                    )[4]
                                    for link in self.finger_contact_links
                                ],
                                dtype=np.float64,
                            ),
                            axis=0,
                        )
                        error = float(np.linalg.norm(midpoint - desired))
                        distances = [
                            min(
                                [
                                    float(point[8])
                                    for point in p.getClosestPoints(
                                        self.panda,
                                        self.host,
                                        0.20,
                                        linkIndexA=link,
                                        linkIndexB=self.articulation_child_link,
                                    )
                                ],
                                default=0.20,
                            )
                            for link in self.finger_contact_links
                        ]
                        if prefer_contact:
                            # Both fingers must be close to the certified child
                            # collision geometry.  A midpoint-only objective can
                            # place the midpoint correctly while leaving one or
                            # both physical finger pads floating in space.
                            max_gap = max(max(distance, 0.0) for distance in distances)
                            score = max_gap + 0.10 * min(error, 0.50)
                        else:
                            score = error
                        if score < best_score:
                            best_score = score
                            best_error = error
                            best_solution = clipped_solution.copy()
        finally:
            p.restoreState(stateId=state_id)
            p.removeState(stateUniqueId=state_id)
        if best_solution is None or not math.isfinite(best_error):
            raise RuntimeError("PyBullet IK found no finite articulated contact configuration")
        return best_solution, best_error

    def _move_to_configuration(
        self, target: np.ndarray, count: int, phase: str, gripper: float
    ) -> None:
        """Interpolate a planned arm configuration through real position motors."""

        start = self._arm_position()
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("articulated arm configuration must contain seven finite values")
        for index in range(count):
            alpha = (index + 1) / count
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            self._execute_action(start + (target - start) * smooth, gripper, phase)

    def _move_to_configuration_until_contact(
        self, target: np.ndarray, count: int, phase: str, gripper: float
    ) -> bool:
        """Move a planned contact phase and require contact at its endpoint.

        A transient collision while crossing an approach corridor is not a
        grasp.  The full phase is therefore always executed; only the final
        consecutive frames determine whether the planned contact is stable.
        """

        start = self._arm_position()
        target = np.asarray(target, dtype=np.float64)
        recent_contact: list[bool] = []
        for index in range(count):
            alpha = (index + 1) / count
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            self._execute_action(start + (target - start) * smooth, gripper, phase)
            recent_contact.append(bool(self._contact_evidence()))
        stable_count = min(3, count)
        return all(recent_contact[-stable_count:])

    def run(self) -> ArticulatedSceneResult:
        if self.inputs.binding.contact_normal_local_xyz is not None:
            from .articulated_contact import run_contact_trajectory
            return run_contact_trajectory(self)
        p = self.p
        binding = self.inputs.binding
        initial_q = binding.closed_position if self.inputs.operator != "close" else binding.open_position
        self._set_joint_state(initial_q)
        closed_child = np.asarray(self._child_link_pose()[0], dtype=np.float64)
        self._set_joint_state(binding.open_position)
        open_child = np.asarray(self._child_link_pose()[0], dtype=np.float64)
        self._set_joint_state(initial_q)
        travel_vector = open_child - closed_child
        if np.linalg.norm(travel_vector) <= 1.0e-6 and binding.joint_type == "revolute":
            travel_vector = np.asarray(binding.open_delta_xyz, dtype=np.float64)
        if np.linalg.norm(travel_vector) <= 1.0e-6:
            raise RuntimeError("PyBullet articulated joint produces no world-space travel")
        direction = travel_vector / np.linalg.norm(travel_vector)
        self._travel_direction_world = direction
        contact_target, _, _ = self._collision_surface_target(direction)
        approach_target = contact_target + direction * self.inputs.approach_clearance_m
        pull_target = contact_target + travel_vector * (
            self.inputs.open_fraction_required
            if self.inputs.operator != "close"
            else -self.inputs.open_fraction_required
        )
        initial_orientation = p.getLinkState(
            self.panda, self.end_effector, computeForwardKinematics=True
        )[5]
        finger_positions = [
            np.asarray(
                p.getLinkState(self.panda, link, computeForwardKinematics=True)[4],
                dtype=np.float64,
            )
            for link in self.finger_contact_links
        ]
        finger_axis = finger_positions[1][:2] - finger_positions[0][:2]
        if float(np.linalg.norm(finger_axis)) <= 1.0e-6:
            raise RuntimeError("DROID finger contact links have a degenerate separation axis")
        # Keep the finger separation perpendicular to the opening travel so
        # both pads can contact the moving face, matching the registered DROID
        # grasp convention used by the official expert.
        grasp_yaw = (
            math.atan2(float(direction[1]), float(direction[0]))
            - math.atan2(float(finger_axis[1]), float(finger_axis[0]))
            - math.pi / 2.0
        )
        orientation = p.multiplyTransforms(
            (0.0, 0.0, 0.0),
            p.getQuaternionFromEuler((0.0, 0.0, grasp_yaw)),
            (0.0, 0.0, 0.0),
            initial_orientation,
        )[1]
        counts = dict(self.inputs.action_counts)
        approach_solution, approach_error = self._contact_configuration(
            approach_target, orientation
        )
        contact_solution, contact_error = self._contact_configuration(
            contact_target, orientation, prefer_contact=True
        )
        pull_solution, pull_error = self._contact_configuration(
            pull_target, orientation, prefer_contact=True
        )
        if self.inputs.reference_joint_actuation:
            return self._run_reference_joint_actuation(
                initial_q=initial_q,
                target_q=(
                    binding.open_position
                    if self.inputs.operator != "close"
                    else binding.closed_position
                ),
                direction=direction,
                contact_target=contact_target,
                approach_solution=approach_solution,
                contact_solution=contact_solution,
                pull_solution=pull_solution,
                counts=counts,
            )
        self._move_to_configuration(
            approach_solution, counts["approach"], "approach", 0.0
        )
        contact_found = self._move_to_configuration_until_contact(
            contact_solution, counts["contact"], "contact", 0.0
        )
        if not contact_found or not self._contact_evidence():
            return self._finish(False, initial_q, direction, contact_target)
        # Use the same full-close grasp actuator as the official DROID expert.
        # Whether the handle is actually held is decided only by live contact
        # evidence below.
        held = self._arm_position()
        for _ in range(counts["contact"]):
            self._execute_action(held, 1.0, "contact")
        if not self._contact_evidence():
            return self._finish(False, initial_q, direction, contact_target)
        self._move_to_configuration(
            pull_solution, counts["actuate"], "actuate", 1.0
        )
        self._move_to_configuration(
            contact_solution, counts["retreat"], "retreat", 0.0
        )
        for _ in range(counts["final_settle"]):
            self._execute_action(self._arm_position(), 0.0, "final_settle")
        final_q = self._joint_position()
        target_q = binding.open_position if self.inputs.operator != "close" else binding.closed_position
        span = abs(binding.open_position - binding.closed_position)
        progress = abs(final_q - initial_q) / span
        success = bool(
            progress >= self.inputs.open_fraction_required
            and (
                (self.inputs.operator != "close" and abs(final_q - binding.open_position) <= (1. - self.inputs.open_fraction_required + 1e-9) * span)
                or (self.inputs.operator == "close" and abs(final_q - binding.closed_position) <= (1. - self.inputs.open_fraction_required + 1e-9) * span)
            )
            and self.contact_frame_count > 0
            and self._attachment_constraint_count() == 0
        )
        return self._finish(success, initial_q, direction, contact_target, final_q=final_q, target_q=target_q)

    def _run_reference_joint_actuation(
        self,
        *,
        initial_q: float,
        target_q: float,
        direction: np.ndarray,
        contact_target: np.ndarray,
        approach_solution: np.ndarray,
        contact_solution: np.ndarray,
        pull_solution: np.ndarray,
        counts: dict[str, int],
    ) -> ArticulatedSceneResult:
        """Produce a visual reference trajectory with an explicit joint actuator.

        This is intentionally opt-in and separately labelled in the result.  It
        is useful for validating the official SceneCode replay while the DROID
        contact policy is still being calibrated; it is not a contact-success
        certificate and is never used by the normal materializer path.
        """

        self._reference_joint_target = initial_q
        self._move_to_configuration(
            approach_solution, counts["approach"], "approach", 0.0
        )
        self._move_to_configuration(
            contact_solution, counts["contact"], "contact", 1.0
        )
        if self.contact_frame_count <= 0:
            return self._finish(False, initial_q, direction, contact_target)
        start = self._arm_position()
        steps = counts["actuate"]
        for index in range(steps):
            alpha = (index + 1) / steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            arm = start + (pull_solution - start) * smooth
            joint_alpha = (index + 1) / steps
            self._reference_joint_target = initial_q + (
                target_q - initial_q
            ) * joint_alpha
            self._execute_action(arm, 1.0, "actuate")
        self._reference_joint_target = target_q
        self._move_to_configuration(
            contact_solution, counts["retreat"], "retreat", 0.0
        )
        for _ in range(counts["final_settle"]):
            self._execute_action(self._arm_position(), 0.0, "final_settle")
        final_q = self._joint_position()
        self._reference_joint_target = None
        # The actuator is the explicit terminal authority in this opt-in mode;
        # no contact evidence is promoted to a robot-success claim.
        return self._finish(
            True,
            initial_q,
            direction,
            contact_target,
            final_q=final_q,
            target_q=target_q,
        )

    def _arm_position(self) -> np.ndarray:
        return np.asarray([self.p.getJointState(self.panda, joint)[0] for joint in self.arm_joints], dtype=np.float64)

    def _attachment_constraint_count(self) -> int:
        count = 0
        for index in range(self.p.getNumConstraints()):
            info = self.p.getConstraintInfo(self.p.getConstraintUniqueId(index))
            if self.host in {int(info[0]), int(info[2])}:
                count += 1
        return count

    def _finish(
        self,
        success: bool,
        initial_q: float,
        direction: np.ndarray,
        contact_target: np.ndarray,
        *,
        final_q: float | None = None,
        target_q: float | None = None,
    ) -> ArticulatedSceneResult:
        binding = self.inputs.binding
        final_q = self._joint_position() if final_q is None else final_q
        target_q = binding.open_position if target_q is None else target_q
        span = abs(binding.open_position - binding.closed_position)
        progress = abs(final_q - initial_q) / span
        trajectory_path = self.output_dir / "trajectory.json"
        trajectory_path.write_text(
            json.dumps(
                {
                    "schema_version": "scenecode_pybullet_articulated_rollout.v1",
                    "authority": {
                        "physics": "PyBullet",
                        "visual_assets": self.inputs.asset_origin,
                        "execution_mode": ("joint_actuator_diagnostic" if self.inputs.reference_joint_actuation else ("externally_assisted_diagnostic" if self.inputs.contact_pull_force else "robot_contact")),
                        "articulation": "SceneCode articulation binding certificate",
                        "robot": "RoboLab DROID Franka+Robotiq 2F-85 portable URDF",
                        "actions": self.inputs.action_authority,
                        "gripper_geometry_scale": self.inputs.gripper_geometry_scale,
                        "gripper_urdf_sha256": self.gripper_urdf_sha256,
                        "gripper_scaled_link_count": self.gripper_scaled_link_count,
                        "handle_collision_scale": self.inputs.handle_collision_scale,
                        "scaled_handle_links": list(self.scaled_handle_links),
                    },
                    "physics_hz": self.physics_hz,
                    "sample_hz": self.control_hz,
                    "control_hz": self.control_hz,
                    "binding": asdict(binding),
                    "reference_acceptance": {"required_joint_travel_fraction": self.inputs.open_fraction_required},
                    "prepared_assets": {
                        "table": (
                            self.prepared_table.to_dict()
                            if self.prepared_table is not None
                            else None
                        ),
                        "ground_plane": {
                            "body_id": int(self.ground),
                            "source": "PyBullet plane.urdf",
                            "fixed": True,
                        },
                        "host": self.prepared_host.to_dict(),
                    },
                    "robot_mount": {"height_m": self.inputs.robot_base_z, "size_xy_m": [.22, .22]},
                    "geometry_variant": {
                        "kind": "pybullet_collision_calibration",
                        "gripper_geometry_scale": self.inputs.gripper_geometry_scale,
                        "handle_collision_scale": self.inputs.handle_collision_scale,
                        "scaled_handle_links": list(self.scaled_handle_links),
                    },
                    "frames": self.frames,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result_path = self.output_dir / "result.json"
        result = {
            "schema_version": "scenecode_pybullet_articulated_result.v1",
            "robot_embodiment": "robolab_droid",
            "asset_origin": self.inputs.asset_origin,
            "operator": self.inputs.operator,
            "success_predicate": self.inputs.success_predicate,
            "binding": asdict(binding),
            "initial_joint_position": initial_q,
            "final_joint_position": final_q,
            "target_joint_position": target_q,
            "open_fraction": progress,
            "contact_frame_count": self.contact_frame_count,
            "gripper_geometry_scale": self.inputs.gripper_geometry_scale,
            "gripper_urdf_sha256": self.gripper_urdf_sha256,
            "handle_collision_scale": self.inputs.handle_collision_scale,
            "scaled_handle_links": list(self.scaled_handle_links),
            "object_attachment_constraint_count": self._attachment_constraint_count(),
            "gripper_coupling_constraint_count": 0,
            "expected_gripper_coupling_constraint_count": 0,
            "attachment_constraint_count": self._attachment_constraint_count(),
            "contact_target_world": contact_target.tolist(),
            "travel_direction_world": direction.tolist(),
            "reference_acceptance": {"required_joint_travel_fraction": self.inputs.open_fraction_required},
            "predicate_evidence": {
                "joint_at_target": abs(final_q - target_q) <= (1. - self.inputs.open_fraction_required + 1e-9) * span,
                "progress_fraction": progress,
                "contact_frame_count": self.contact_frame_count,
            },
            "success": bool(success),
            "physical_success": bool(success and not self.inputs.reference_joint_actuation and self.external_assist_step_count == 0),
            "execution_mode": ("joint_actuator_diagnostic" if self.inputs.reference_joint_actuation else ("externally_assisted_diagnostic" if self.inputs.contact_pull_force else "robot_contact")),
            "external_assist_step_count": self.external_assist_step_count,
            "contact_pull_force_n": self.inputs.contact_pull_force,
            "planning_evidence": self.planning_evidence,
            "contact_execution_evidence": getattr(self, "contact_execution_evidence", {}),
            "measured_contact_dynamics": {
                "host_lateral_friction": self.p.getDynamicsInfo(self.host, self.articulation_child_link)[1],
                "finger_lateral_friction": [self.p.getDynamicsInfo(self.panda, link)[1] for link in self.finger_contact_links],
            },
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return ArticulatedSceneResult(
            trajectory_path=str(trajectory_path),
            result_path=str(result_path),
            frame_count=len(self.frames),
            simulation_seconds=len(self.frames) / self.control_hz,
            joint_name=self.joint_name,
            initial_joint_position=initial_q,
            final_joint_position=final_q,
            open_fraction=progress,
            contact_frame_count=self.contact_frame_count,
            attachment_constraint_count=self._attachment_constraint_count(),
            gripper_geometry_scale=self.inputs.gripper_geometry_scale,
            handle_collision_scale=self.inputs.handle_collision_scale,
            scaled_handle_links=self.scaled_handle_links,
            success=bool(success),
        )


def run_droid_articulated(inputs: ArticulatedSceneInputs) -> ArticulatedSceneResult:
    output = inputs.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite articulated rollout: {output}")
    with PyBulletDroidArticulatedScene(inputs) as scene:
        return scene.run()
