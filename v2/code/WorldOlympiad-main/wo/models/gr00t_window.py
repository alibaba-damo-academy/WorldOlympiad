"""Native GR00T DROID window contract, independent of Isaac imports.

Use wo.models.gr00t_client.N16Client for N1.6 (its .npy wire protocol is
incompatible with RoboLab's N1.7 msgpack_numpy client). N1.6 takes
joint/gripper state; N1.7
additionally requires the real EE pose. No generated video is claimed:
CAE must use the native simulator's executed state trace.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

DROID_EEF_ROTATION_CORRECT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)


def eef_9d(position, quat_wxyz):
    p = np.asarray(position, dtype=float)
    q = np.asarray(quat_wxyz, dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('EEF position/quaternion must be finite 3D/4D vectors')
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError('EEF quaternion must have nonzero norm')
    w, x, y, z = q / norm
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]) @ DROID_EEF_ROTATION_CORRECT
    return np.concatenate([p, rotation[:2].reshape(6)]).astype(np.float32)


def _image(image):
    import cv2
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError('DROID camera must be HWC RGB uint8')
    # Matches RoboLab GR00T resize_no_pad, not a Cosmos tiled canvas.
    return cv2.resize(a, (320, 180), interpolation=cv2.INTER_AREA)[None, None]


def build_request(*, exterior_left, wrist, joints, gripper, prompt,
                  version='n1.6', ee_position=None, ee_quat_wxyz=None):
    j = np.asarray(joints, dtype=np.float32)
    g = np.asarray(gripper, dtype=np.float32).reshape(-1)
    if j.shape != (7,) or g.shape != (1,) or not np.isfinite(j).all() or not np.isfinite(g).all():
        raise ValueError('DROID requires seven finite arm joints and one gripper value')
    if not 0 <= g[0] <= 1:
        raise ValueError('DROID gripper must be normalized to [0,1]')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('instruction is required')
    if version not in {'n1.6', 'n1.7'}:
        raise ValueError(f'unsupported GR00T version: {version}')
    request = {
        'video.exterior_image_1_left': _image(exterior_left),
        'video.wrist_image_left': _image(wrist),
        'state.joint_position': j[None, None],
        'state.gripper_position': g[None, None],
        'annotation.language.language_instruction': [prompt],
    }
    if version == 'n1.7':
        request['state.eef_9d'] = eef_9d(ee_position, ee_quat_wxyz)[None, None]
    return request


def unpack_actions(response, horizon=8):
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError('execution horizon must be a positive integer')
    action_dict = response[0] if isinstance(response, tuple) else response
    def component(name, width):
        value = action_dict.get('action.' + name, action_dict.get(name))
        if value is None:
            raise ValueError(f'missing GR00T action {name}')
        a = np.asarray(value, dtype=np.float32)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if width == 1 and a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2 or a.shape[1] != width or not np.isfinite(a).all():
            raise ValueError(f'invalid GR00T {name} shape or nonfinite values')
        return a
    j = component('joint_position', 7)
    g = component('gripper_position', 1)
    if len(j) != len(g) or len(j) < horizon:
        raise ValueError('GR00T returned unequal or insufficient action chunks')
    result = np.concatenate([j[:horizon], g[:horizon]], axis=1)
    result[:, -1] = (result[:, -1] > 0.5).astype(np.float32)
    return result


def run_native_window(*, window_root, observation_root, frame_index, joints, gripper,
                      prompt, host, port, client_factory=None):
    """Adapt two untiled native RGB frames to the runner's initial+future timeline."""
    from PIL import Image
    from wo.models.gr00t_client import N16Client
    factory = client_factory or N16Client
    client = factory(host=host, port=port)
    try:
        images = {}
        for view, short in [('exterior_left', 'left'), ('wrist', 'wrist')]:
            path = Path(observation_root) / f'step_{frame_index:06d}_{short}_image.png'
            images[view] = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
        future, metadata = run_window(client=client, window_root=Path(window_root),
                                     joints=joints, gripper=gripper, prompt=prompt, **images)
    finally:
        client.close()
    initial = np.concatenate([np.asarray(joints, dtype=np.float32), [gripper]])[None]
    timeline = np.concatenate([initial, future], axis=0).astype(np.float32)
    return timeline, timeline.copy(), metadata


def run_window(*, client, window_root: Path, horizon=8, **observation):
    request = build_request(**observation)
    actions = unpack_actions(client.get_action(request), horizon=horizon)
    window_root.mkdir(parents=True, exist_ok=True)
    np.save(window_root / 'policy_actions.npy', actions, allow_pickle=False)
    metadata = {'model': 'gr00t_n1.6_droid', 'action_source': 'native_policy',
                'action_space': 'absolute_joint_position', 'policy_horizon': len(actions),
                'camera_resize': 'no_pad_stretch_area', 'trajectory_source': 'native_simulator'}
    (window_root / 'model_metadata.json').write_text(json.dumps(metadata, indent=2))
    return actions, metadata
