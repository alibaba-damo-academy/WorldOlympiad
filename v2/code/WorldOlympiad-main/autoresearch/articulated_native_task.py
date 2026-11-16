"""Measured, simulator-independent contract for native articulated policy tasks.

The reference is a closed link pose, never an expert future trajectory. Joint
coordinates come from the current simulator link frame, not a target command.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class JointTask:
    joint_type: str
    axis_world: tuple[float, float, float]
    closed_link_xyz: tuple[float, float, float]
    closed_link_xyzw: tuple[float, float, float, float]
    pivot_world: tuple[float, float, float]
    closed_position: float
    target_position: float
    lower: float
    upper: float
    tolerance: float
    max_speed: float
    hold_steps: int
    max_steps: int

    def __post_init__(self):
        values = [*self.axis_world, *self.closed_link_xyz, *self.closed_link_xyzw,
                  *self.pivot_world, self.closed_position, self.target_position,
                  self.lower, self.upper, self.tolerance, self.max_speed]
        if not np.isfinite(values).all():
            raise ValueError("non-finite joint task")
        if self.joint_type not in {"prismatic", "revolute"}:
            raise ValueError("joint task requires prismatic or revolute motion")
        if abs(np.linalg.norm(self.axis_world) - 1.0) > 1e-6:
            raise ValueError("joint axis must be normalized")
        if abs(np.linalg.norm(self.closed_link_xyzw) - 1.0) > 1e-6:
            raise ValueError("closed link quaternion must be normalized")
        if not self.lower <= self.closed_position <= self.upper:
            raise ValueError("closed state outside joint limits")
        if not self.lower <= self.target_position <= self.upper:
            raise ValueError("target outside joint limits")
        if not 0 < self.tolerance < abs(self.target_position-self.closed_position):
            raise ValueError("tolerance must distinguish closed and target states")
        if self.max_speed <= 0 or self.hold_steps < 1 or self.max_steps < self.hold_steps:
            raise ValueError("invalid task termination contract")
        if self.joint_type == "revolute" and self.upper-self.lower >= math.pi:
            raise ValueError("pose-based revolute measurement requires travel below pi")

    @classmethod
    def from_dict(cls, raw):
        return cls(**raw)

    def pose_at(self, coordinate):
        """Forward kinematics, for isolated native runtime validation only."""
        delta = float(coordinate)-self.closed_position
        axis = np.asarray(self.axis_world)
        position = np.asarray(self.closed_link_xyz)
        rotation = Rotation.from_quat(self.closed_link_xyzw)
        if self.joint_type == "prismatic":
            return (position+axis*delta).tolist(), rotation.as_quat().tolist()
        motion = Rotation.from_rotvec(axis*delta)
        pivot = np.asarray(self.pivot_world)
        return (pivot+motion.apply(position-pivot)).tolist(), (motion*rotation).as_quat().tolist()

    def measure(self, position, quaternion, linear_velocity, angular_velocity):
        arrays = [np.asarray(v,dtype=float) for v in
                  (position,quaternion,linear_velocity,angular_velocity)]
        if [a.shape for a in arrays] != [(3,),(4,),(3,),(3,)] or not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("invalid measured link state")
        pos, quat, linear, angular = arrays
        if abs(np.linalg.norm(quat)-1) > 1e-3:
            raise ValueError("invalid measured quaternion")
        axis = np.asarray(self.axis_world)
        r0 = Rotation.from_quat(self.closed_link_xyzw)
        rotation = Rotation.from_quat(quat)
        if self.joint_type == "prismatic":
            delta = float(np.dot(pos-np.asarray(self.closed_link_xyz),axis))
            speed = float(np.dot(linear,axis))
        else:
            delta = float(np.dot((rotation*r0.inv()).as_rotvec(),axis))
            speed = float(np.dot(angular,axis))
        coordinate = self.closed_position+delta
        expected_xyz,expected_quat = self.pose_at(coordinate)
        translation_residual = float(np.linalg.norm(pos-np.asarray(expected_xyz)))
        rotation_residual = float((rotation*Rotation.from_quat(expected_quat).inv()).magnitude())
        return {"position":coordinate,"velocity":speed,
                "target_error":abs(coordinate-self.target_position),
                "translation_residual_m":translation_residual,
                "rotation_residual_rad":rotation_residual,
                "within_limits": self.lower-0.005 <= coordinate <= self.upper+0.005,
                "at_target":abs(coordinate-self.target_position)<=self.tolerance}

def reference_actions_for_native_replay(payload, joint_name):
    """Extract a measured reference joint path without changing its time axis."""
    frames=payload.get('frames',[])
    if payload.get('sample_hz')!=15 or len(frames)<2:
        raise ValueError('reference replay needs at least two frames at 15 Hz')
    if not np.allclose(np.diff([f['time_s'] for f in frames]),1/15,atol=1e-6):
        raise ValueError('reference timestamps are not consecutive')
    rows=[]
    for frame in frames:
        if frame['articulated']['joint_name']!=joint_name:
            raise ValueError('reference joint differs from native task')
        robot=frame['robot']; names=robot['joint_names'];positions=robot['joint_positions']
        if len(names)!=len(positions) or len(set(names))!=len(names):
            raise ValueError('ambiguous reference joint names')
        values=dict(zip(names,positions))
        row=[values[f'panda_joint{i}'] for i in range(1,8)]+[robot['droid_gripper_position']]
        if not np.isfinite(row).all() or not 0<=row[-1]<=1:
            raise ValueError('invalid reference joint state')
        rows.append(row)
    return np.asarray(rows,dtype=np.float32)


class JointTermination:
    def __init__(self, task: JointTask, *, evaluation_max_steps: int | None = None):
        self.task=task
        self.stable_steps=0
        if evaluation_max_steps is not None and (isinstance(evaluation_max_steps, bool)
                or not isinstance(evaluation_max_steps, int) or evaluation_max_steps < task.hold_steps):
            raise ValueError("shared joint-task window must cover its stability check")
        self.evaluation_max_steps = evaluation_max_steps
        self.completion_frame = None

    def step(self, state, *, contacts: bool, gripper: float, step: int):
        if not state["within_limits"] or state["translation_residual_m"] > 0.01 or state["rotation_residual_rad"] > 0.03:
            raise RuntimeError("native joint constraint violated")
        released = not contacts and gripper < 0.2
        stable = state["at_target"] and abs(state["velocity"]) <= self.task.max_speed and released
        self.stable_steps = self.stable_steps+1 if stable else 0
        success = self.stable_steps >= self.task.hold_steps
        if success and self.completion_frame is None:
            self.completion_frame = step
        if self.evaluation_max_steps is not None:
            done = step >= self.evaluation_max_steps
            return {"task_success": success, "released": released,
                    "stable_steps": self.stable_steps, "completion_frame": self.completion_frame,
                    "done": done, "reason": "evaluation_window_complete" if done else None}
        return {"task_success": success, "released": released,
                "stable_steps":self.stable_steps,
                "done":success or step>=self.task.max_steps,
                "reason":"task_completed" if success else "project_task_timeout" if step>=self.task.max_steps else None}
