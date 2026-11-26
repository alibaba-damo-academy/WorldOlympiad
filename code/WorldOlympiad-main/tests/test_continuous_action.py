from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_action.adapters import (
    cosmos_rot6d_actions_to_trajectory,
    load_cosmos_absolute_poses,
    load_cosmos_joint_positions,
    load_dream2flow_plan,
    load_dream2flow_video_object_flow,
    load_lingbot_actions,
)
from continuous_action.chunks import ChunkSpec, validate_worldolympiad_chunks
from continuous_action.evaluator import evaluate_manifest, write_evaluation
from continuous_action.metrics import score_action_trajectories, FORMAL_INDEPENDENT_METRIC_VERSION
from continuous_action.quality import TrackingQualityConfig, assess_tracking_quality
from continuous_action.schema import ActionTrajectory
from continuous_action.workspaces import robot_workspace_contract_for_model


IDENTITY_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class ContinuousActionMetricTest(unittest.TestCase):
    def trajectory(self, xyz: list[list[float]], timestamps: list[float]) -> ActionTrajectory:
        return ActionTrajectory(np.asarray(xyz), np.asarray(timestamps))


    def test_evaluation_results_are_immutable_but_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "continuous_action_metric_v13_result.json"
            result = {
                "metric_version": FORMAL_INDEPENDENT_METRIC_VERSION,
                "manifest_sha256": "a" * 64,
            }
            self.assertEqual(write_evaluation(result, output), output)
            self.assertEqual(write_evaluation(dict(result), output), output)
            changed = dict(result)
            changed["manifest_sha256"] = "b" * 64
            with self.assertRaisesRegex(FileExistsError, "different inputs"):
                write_evaluation(changed, output)



























    def test_cosmos_rot6d_is_integrated_in_initial_ee_frame(self) -> None:
        actions = np.asarray(
            [
                [1.0, 0.0, 0.0, *IDENTITY_ROT6D, 1.0],
                [1.0, 0.0, 0.0, *IDENTITY_ROT6D, 1.0],
            ]
        )
        trajectory = cosmos_rot6d_actions_to_trajectory(actions, duration=2.0)
        np.testing.assert_allclose(
            trajectory.positions,
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            atol=1e-8,
        )

    def test_cosmos_absolute_policy_pose_is_moved_to_initial_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cosmos_abs.npy"
            # The initial EE is translated in world coordinates; this offset
            # must disappear while the physical one-metre path remains.
            np.save(
                path,
                np.asarray(
                    [
                        [5, 1, 0, 0, 0, 0, 1, 0],
                        [6, 1, 0, 0, 0, 0, 1, 0],
                    ],
                    dtype=np.float64,
                ),
            )
            trajectory = load_cosmos_absolute_poses(path, duration=1.0)
            np.testing.assert_allclose(trajectory.positions, [[0, 0, 0], [1, 0, 0]])

    def test_cosmos_joint_policy_requires_matching_fk_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actions_path = root / "actions.npy"
            actions = np.zeros((2, 8), dtype=np.float32)
            np.save(actions_path, actions)
            with self.assertRaisesRegex(ValueError, "fk_ee_trajectory_path"):
                load_cosmos_joint_positions(actions_path, fk_ee_trajectory_path=None, duration=1.0)

            fk_path = root / "actions_fk.npz"
            np.savez(
                fk_path,
                ee_trajectory=np.asarray(
                    [[5, 0, 0, 0, 0, 0, 1], [6, 0, 0, 0, 0, 0, 1]],
                    dtype=np.float64,
                ),
            )
            fk_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "source_actions_sha256": hashlib.sha256(actions_path.read_bytes()).hexdigest(),
                        "urdf_path": "panda.urdf",
                        "target_link_name": "panda_hand_tcp",
                        "joint_order": [f"panda_joint{i}" for i in range(1, 8)],
                    }
                )
            )
            trajectory = load_cosmos_joint_positions(
                actions_path,
                fk_ee_trajectory_path=fk_path,
                duration=1.0,
            )
            np.testing.assert_allclose(trajectory.positions, [[0, 0, 0], [1, 0, 0]])

    def test_lingbot_requires_units_and_frame_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lingbot.npy"
            np.save(path, np.asarray([[1, 0, 0], [1, 0, 0]], dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "translation_scale"):
                load_lingbot_actions(path, duration=1.0)
            unmapped = load_lingbot_actions(path, duration=1.0, translation_scale=20.0)
            reference = self.trajectory([[0, 0, 0], [0.05, 0, 0], [0.1, 0, 0]], [0, 0.5, 1])
            with self.assertRaisesRegex(ValueError, "same registered coordinate frame"):
                score_action_trajectories(reference, unmapped)
            with self.assertRaisesRegex(ValueError, "proper orthonormal rotation"):
                load_lingbot_actions(
                    path,
                    duration=1.0,
                    translation_scale=20.0,
                    rotation_to_initial_ee=np.diag([2.0, 1.0, 1.0]),
                )

    def test_dream2flow_reference_rejects_intermediate_object_flow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object_flow_result.npz"
            np.savez(path, object_positions=np.zeros((3, 10, 3), dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "ee_trajectory"):
                load_dream2flow_plan(path, duration=1.0)

    def test_dream2flow_plan_uses_embedded_source_video_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory_optimization_plan.npz"
            ee = np.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [2, 0, 0, 0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            np.savez(
                path,
                ee_trajectory=ee,
                timestamps=np.asarray([0.0, 0.5, 2.0]),
                source_frame_indices=np.asarray([0, 1, 4]),
                source_video_fps=np.asarray(2.0),
                observed_frame_mask=np.asarray([True, True, False]),
            )
            trajectory = load_dream2flow_plan(path, duration=99.0)
            np.testing.assert_allclose(trajectory.timestamps, [0.0, 0.5, 2.0])
            self.assertEqual(trajectory.metadata["source_frame_indices"], [0, 1, 4])
            self.assertEqual(
                trajectory.metadata["observed_frame_mask"], [True, True, False]
            )
            self.assertEqual(
                trajectory.metadata["timing_source"],
                "embedded_source_video_timestamps",
            )

    def test_retired_median_adapter_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "median object-flow adaptation retired"):
            load_dream2flow_video_object_flow('unused.npz')


class ChunkContractTest(unittest.TestCase):
    def test_contiguous_at_most_six(self) -> None:
        validate_worldolympiad_chunks(
            [ChunkSpec(0, 1.0, 2.0), ChunkSpec(1, 2.0, 3.0)]
        )
        with self.assertRaisesRegex(ValueError, "gap"):
            validate_worldolympiad_chunks(
                [ChunkSpec(0, 1.0, 2.0), ChunkSpec(1, 2.5, 3.0)]
            )
        with self.assertRaisesRegex(ValueError, "at most six"):
            validate_worldolympiad_chunks(
                [ChunkSpec(index, float(index), float(index + 1)) for index in range(7)]
            )



def promote_test_flows_to_plan_fixtures(path):
    """Synthetic adapter fixtures only. Integration tests run the real optimizer separately."""
    from continuous_action.official_plan import sha256, TRAJECTORY_SOURCE, UPSTREAM_COMMIT
    manifest = json.loads(path.read_text())
    root = path.parent
    registration = root / 'test_registration.json'
    registration.write_text('{"synthetic_unit_test":true}')
    urdf = root / 'test_robot.urdf'
    urdf.write_text('<robot name="synthetic_unit_test"/>')
    lock = json.loads((REPO_ROOT / 'scripts/official_dream2flow_source_lock.json').read_text())
    if 'rollout' not in manifest: return path
    roll = manifest['rollout']
    for spec in [roll['reference'], *roll['predictions'].values()]:
        if 'path' not in spec: continue
        source = root / spec['path']
        if spec.get('type') != 'dream2flow_video_plan' or source.suffix != '.npz' or not source.exists(): continue
        with np.load(source, allow_pickle=False) as data:
            if 'position' not in data: continue
            position, valid = data['position'], data['valid_mask']
        flow = source.with_name(source.stem + '_raw_fixture.npz')
        source.rename(flow)
        fps = spec.get('sample_rate', (len(position)-1) / (roll['end_sec']-roll.get('start_sec',0)))
        indices = np.arange(0, len(position), max(1,len(position)//100))
        xyz = position[indices,0] - position[0,0]
        ee = np.column_stack([xyz, np.tile([0,0,0,1], (len(xyz),1))])
        observed = valid.any(axis=1)[indices]
        np.savez(source,ee_trajectory=ee,timestamps=indices/fps,source_frame_indices=indices,observed_frame_mask=observed)
        meta = dict(evaluation_protocol='video_imagination_dream2flow_official_action_trajectory_v2', initialization_kind='frame_zero_kinematic_grasp', initial_pose_source='registered_grasp_target', optimizer_config=lock['optimizer_numerical_defaults'], trajectory_source=TRAJECTORY_SOURCE,upstream_commit=UPSTREAM_COMMIT,source_sha256=lock['source_sha256'],video_role=spec['video_role'],source_video_sha256=sha256(root/spec['video_path']),plan_sha256=sha256(source),coordinate_frame='registered_world',object_to_robot_ik_used=True,action_head_used=False,object_name=spec['object_name'],source_video_frame_count=len(position),source_video_fps=fps)
        for key, artifact in [('flow',flow),('registration',registration),('registered_urdf',urdf)]:
            meta[key+'_path']=str(artifact);meta[key+'_sha256']=sha256(artifact)
        source.with_suffix('.json').write_text(json.dumps(meta))
    return path



if __name__ == "__main__":
    unittest.main()
