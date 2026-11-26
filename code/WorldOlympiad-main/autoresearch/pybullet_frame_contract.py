"""Record Bullet base poses in the link frame used by native scene importers."""
from __future__ import annotations


def measured_base_link_pose(bullet, body):
    """Undo Bullet's local inertial transform, including its rotation."""
    world_com = bullet.getBasePositionAndOrientation(body)
    dynamics = bullet.getDynamicsInfo(body, -1)
    com_to_link = bullet.invertTransform(dynamics[3], dynamics[4])
    position, orientation = bullet.multiplyTransforms(*world_com, *com_to_link)
    return {
        "position": list(position),
        "orientation_xyzw": list(orientation),
        "pose_frame": "link",
        "base_com_position": list(world_com[0]),
        "base_com_orientation_xyzw": list(world_com[1]),
        "local_inertial_position": list(dynamics[3]),
        "local_inertial_orientation_xyzw": list(dynamics[4]),
    }


def require_root_link_measurement(body, parent_link, measurement):
    """An unlabelled root COM must never silently become a scene placement."""
    child_names = {joint.child for joint in body.joints}
    if parent_link not in child_names and measurement.get("pose_frame") != "link":
        raise ValueError("reference root pose is ambiguous: regenerate with measured link-frame recording")


def reset_base_link_pose(bullet, body, position, orientation, **kwargs):
    """Bullet reset consumes a COM pose, unlike URDF loading's root-link pose."""
    dynamics = bullet.getDynamicsInfo(body, -1, **kwargs)
    world_com = bullet.multiplyTransforms(position, orientation, dynamics[3], dynamics[4])
    bullet.resetBasePositionAndOrientation(body, *world_com, **kwargs)
