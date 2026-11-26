"""Authoritative measured passive object-joint snapshots."""
import math


def measured_object_joints(physics, body, client):
    result = {}
    for index in range(physics.getNumJoints(body, physicsClientId=client)):
        info = physics.getJointInfo(body, index, physicsClientId=client)
        if info[2] not in (physics.JOINT_REVOLUTE, physics.JOINT_PRISMATIC):
            continue
        name = info[1].decode()
        value = float(physics.getJointState(body, index, physicsClientId=client)[0])
        if not name or name in result or not math.isfinite(value):
            raise ValueError('invalid measured object joint identity/state')
        result[name] = value
    return result
