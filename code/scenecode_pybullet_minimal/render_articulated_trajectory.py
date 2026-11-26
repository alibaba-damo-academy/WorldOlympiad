#!/usr/bin/env python3
"""Render a recorded PyBullet articulated trajectory for visual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--host-sdf", type=Path, required=True)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()

    import pybullet as p
    import pybullet_data

    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    frames = trajectory["frames"][args.start_frame:]
    if not frames:
        raise ValueError("trajectory contains no frames")
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    ground = p.loadURDF("plane.urdf")
    host = p.loadSDF(str(args.host_sdf))[0]
    robot_frame = frames[0]["robot"]
    robot = p.loadURDF(
        str(args.panda_urdf),
        basePosition=robot_frame["base_position"],
        baseOrientation=robot_frame["base_orientation_xyzw"],
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )
    host_id = trajectory["binding"]["host_object_id"]
    joint_name = trajectory["binding"]["joint_name"]
    host_joint_by_name = {
        p.getJointInfo(host, i)[1].decode("utf-8"): i
        for i in range(p.getNumJoints(host))
    }
    if joint_name not in host_joint_by_name:
        raise ValueError(f"trajectory joint is absent from replay SDF: {joint_name}")
    host_joint = host_joint_by_name[joint_name]
    robot_joint_by_name = {
        p.getJointInfo(robot, i)[1].decode("utf-8"): i
        for i in range(p.getNumJoints(robot))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(trajectory.get("sample_hz", 15)),
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {args.output}")
    try:
        for frame in frames:
            obj = frame["objects"][host_id]
            p.resetBasePositionAndOrientation(
                host, obj["position"], obj["orientation_xyzw"]
            )
            articulated = frame["articulated"]
            p.resetJointState(host, host_joint, articulated["position"])
            names = frame["robot"]["joint_names"]
            positions = frame["robot"]["joint_positions"]
            for name, position in zip(names, positions, strict=True):
                if name in robot_joint_by_name:
                    p.resetJointState(robot, robot_joint_by_name[name], position)
            view = p.computeViewMatrix(
                cameraEyePosition=(0.75, 3.0, 1.45),
                cameraTargetPosition=(0.0, 0.0, 0.92),
                cameraUpVector=(0.0, 0.0, 1.0),
            )
            projection = p.computeProjectionMatrixFOV(
                fov=52.0, aspect=args.width / args.height, nearVal=0.02, farVal=4.0
            )
            _, _, rgba, _, _ = p.getCameraImage(
                args.width,
                args.height,
                viewMatrix=view,
                projectionMatrix=projection,
                renderer=p.ER_TINY_RENDERER,
            )
            image = np.asarray(rgba, dtype=np.uint8).reshape(args.height, args.width, 4)[:, :, :3]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            contacts = len(frame["articulated"].get("contact_evidence", []))
            label = (
                f"frame {frame['frame']:03d}  phase={frame['phase']}  "
                f"q={articulated['position']:.3f}  contacts={contacts}"
            )
            cv2.rectangle(image, (10, 10), (args.width - 10, 45), (0, 0, 0), -1)
            cv2.putText(image, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()
        p.disconnect()
    print(args.output)


if __name__ == "__main__":
    main()
