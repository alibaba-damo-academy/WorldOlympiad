import contextlib
import json
import multiprocessing
import os
import queue
import socket
import tempfile
import time
import unittest

from pathlib import Path

import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scenecode.agent_utils.blender.request_lock import (
    DEFAULT_LOCK_PATH,
    LOCK_ENV_VAR,
    acquire_blender_request_lock,
    get_blender_request_lock_path,
    get_blender_request_lock_owner_path,
)
from scenecode.agent_utils.blender.render_dataclasses import LinkMeshInfo
from scenecode.agent_utils.blender.server_manager import BlenderServer
from scenecode.agent_utils.rendering import (
    _build_scene_object_metadata,
    render_per_wall_ortho_views,
    save_directive_as_blend,
    save_scene_as_blend,
)
from pydrake.all import RigidTransform

from scenecode.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenecode.robot_eval.tools.vision_tools import _render_validation_scene


def _lock_worker(lock_path: str, hold_seconds: float, result_queue) -> None:
    os.environ[LOCK_ENV_VAR] = lock_path
    start = time.monotonic()
    with acquire_blender_request_lock("integration"):
        acquired = time.monotonic()
        result_queue.put(acquired - start)
        time.sleep(hold_seconds)


class _FakePort:
    def __init__(self, on_eval):
        self._on_eval = on_eval

    def Eval(self, context):
        self._on_eval()
        return "ok"


class _FakeSensor:
    def color_image_output_port(self):
        return object()


class _FakeBuilder:
    def __init__(self, on_eval):
        self._on_eval = on_eval

    def GetSubsystemByName(self, name):
        return _FakeSensor()

    def ExportOutput(self, port, name):
        return None

    def Build(self):
        return MagicMock(
            CreateDefaultContext=MagicMock(return_value="context"),
            GetOutputPort=MagicMock(return_value=_FakePort(self._on_eval)),
        )


class TestBlenderRequestLock(unittest.TestCase):
    def test_get_blender_request_lock_path_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(LOCK_ENV_VAR, None)
            self.assertEqual(get_blender_request_lock_path(), DEFAULT_LOCK_PATH)

    def test_get_blender_request_lock_path_respects_env_override(self):
        override = "/tmp/custom/blender.lock"
        with patch.dict(os.environ, {LOCK_ENV_VAR: override}, clear=False):
            self.assertEqual(get_blender_request_lock_path(), Path(override))

    def test_acquire_blender_request_lock_releases_in_finally(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = str(Path(temp_dir) / "blender.lock")
            with patch.dict(os.environ, {LOCK_ENV_VAR: lock_path}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    with acquire_blender_request_lock("first"):
                        raise RuntimeError("boom")

                with acquire_blender_request_lock("second"):
                    self.assertTrue(True)

    def test_acquire_blender_request_lock_writes_owner_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "blender.lock"
            with patch.dict(os.environ, {LOCK_ENV_VAR: str(lock_path)}, clear=False):
                owner_path = get_blender_request_lock_owner_path(lock_path)
                with acquire_blender_request_lock("metadata"):
                    owner_data = json.loads(owner_path.read_text(encoding="utf-8"))
                    stat_result = lock_path.stat()
                    self.assertEqual(owner_data["pid"], os.getpid())
                    self.assertEqual(owner_data["hostname"], socket.gethostname())
                    self.assertEqual(owner_data["purpose"], "metadata")
                    self.assertEqual(owner_data["lock_path"], str(lock_path))
                    self.assertEqual(owner_data["st_dev"], stat_result.st_dev)
                    self.assertEqual(owner_data["st_ino"], stat_result.st_ino)
                    self.assertIn("acquired_at", owner_data)
                    self.assertIn("heartbeat_at", owner_data)

                self.assertFalse(owner_path.exists())

    def test_acquire_blender_request_lock_preserves_overwritten_owner_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "blender.lock"
            with patch.dict(os.environ, {LOCK_ENV_VAR: str(lock_path)}, clear=False):
                owner_path = get_blender_request_lock_owner_path(lock_path)
                with acquire_blender_request_lock("metadata"):
                    owner_path.write_text(
                        json.dumps({"owner_token": "other"}),
                        encoding="utf-8",
                    )

                owner_data = json.loads(owner_path.read_text(encoding="utf-8"))
                self.assertEqual(owner_data["owner_token"], "other")

    def test_acquire_blender_request_lock_blocks_other_processes(self):
        ctx = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = str(Path(temp_dir) / "blender.lock")
            result_queue = ctx.Queue()
            first = ctx.Process(
                target=_lock_worker,
                args=(lock_path, 0.5, result_queue),
            )
            second = ctx.Process(
                target=_lock_worker,
                args=(lock_path, 0.0, result_queue),
            )

            first.start()
            time.sleep(0.1)
            second.start()

            first_wait = result_queue.get(timeout=2.0)
            self.assertLess(first_wait, 0.2)

            with self.assertRaises(queue.Empty):
                result_queue.get(timeout=0.2)

            second_wait = result_queue.get(timeout=2.0)
            self.assertGreater(second_wait, 0.3)

            first.join(timeout=2.0)
            second.join(timeout=2.0)
            self.assertEqual(first.exitcode, 0)
            self.assertEqual(second.exitcode, 0)

    def test_canonicalize_mesh_locks_only_request(self):
        lock_events = []
        request_lock_state = {"held": False}

        @contextlib.contextmanager
        def fake_lock(purpose):
            lock_events.append(("enter", purpose))
            request_lock_state["held"] = True
            try:
                yield Path("/tmp/test.lock")
            finally:
                request_lock_state["held"] = False
                lock_events.append(("exit", purpose))

        server = BlenderServer(port=8000)
        server._running = True

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "scenecode.agent_utils.blender.server_manager.acquire_blender_request_lock",
            side_effect=fake_lock,
        ):
            input_path = Path(temp_dir) / "input.gltf"
            output_path = Path(temp_dir) / "output.gltf"
            input_path.write_text("mesh")

            def fake_request(**kwargs):
                self.assertTrue(request_lock_state["held"])
                self.assertEqual(kwargs["endpoint"], "/canonicalize")
                return {}

            server._make_request_with_retry = MagicMock(side_effect=fake_request)

            result = server.canonicalize_mesh(
                input_path=input_path,
                output_path=output_path,
                up_axis="+Z",
                front_axis="+Y",
            )

        self.assertEqual(result, output_path)
        self.assertEqual(
            lock_events,
            [("enter", "canonicalize_mesh"), ("exit", "canonicalize_mesh")],
        )

    def test_convert_glb_to_gltf_locks_only_request(self):
        lock_events = []
        request_lock_state = {"held": False}

        @contextlib.contextmanager
        def fake_lock(purpose):
            lock_events.append(("enter", purpose))
            request_lock_state["held"] = True
            try:
                yield Path("/tmp/test.lock")
            finally:
                request_lock_state["held"] = False
                lock_events.append(("exit", purpose))

        server = BlenderServer(port=8000)
        server._running = True

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "scenecode.agent_utils.blender.server_manager.acquire_blender_request_lock",
            side_effect=fake_lock,
        ):
            input_path = Path(temp_dir) / "input.glb"
            output_path = Path(temp_dir) / "output.gltf"
            input_path.write_text("mesh")

            def fake_request(**kwargs):
                self.assertTrue(request_lock_state["held"])
                self.assertEqual(kwargs["endpoint"], "/convert_glb_to_gltf")
                return output_path

            server._make_request_with_retry = MagicMock(side_effect=fake_request)

            result = server.convert_glb_to_gltf(
                input_path=input_path,
                output_path=output_path,
                export_yup=True,
            )

        self.assertEqual(result, output_path)
        self.assertEqual(
            lock_events,
            [
                ("enter", "convert_glb_to_gltf"),
                ("exit", "convert_glb_to_gltf"),
            ],
        )

    @patch("scenecode.agent_utils.blender.server_manager.convert_gltf_to_glb")
    def test_render_multiview_for_analysis_locks_only_request(self, mock_convert):
        lock_events = []
        request_lock_state = {"held": False}

        @contextlib.contextmanager
        def fake_lock(purpose):
            lock_events.append(("enter", purpose))
            request_lock_state["held"] = True
            try:
                yield Path("/tmp/test.lock")
            finally:
                request_lock_state["held"] = False
                lock_events.append(("exit", purpose))

        server = BlenderServer(port=8000)
        server._running = True

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "scenecode.agent_utils.blender.server_manager.acquire_blender_request_lock",
            side_effect=fake_lock,
        ):
            mesh_path = Path(temp_dir) / "mesh.glb"
            mesh_path.write_bytes(b"mesh")
            output_dir = Path(temp_dir) / "renders"

            def fake_request(**kwargs):
                self.assertTrue(request_lock_state["held"])
                self.assertEqual(kwargs["endpoint"], "/render_multiview")
                return {"image_paths": [str(output_dir / "view.png")]}

            server._make_multipart_request_with_retry = MagicMock(side_effect=fake_request)

            image_paths = server.render_multiview_for_analysis(
                mesh_path=mesh_path,
                output_dir=output_dir,
                elevation_degrees=30.0,
            )

        self.assertEqual(image_paths, [output_dir / "view.png"])
        self.assertEqual(
            lock_events,
            [("enter", "render_multiview"), ("exit", "render_multiview")],
        )
        mock_convert.assert_not_called()

    def test_render_multiview_articulated_locks_only_request(self):
        lock_events = []
        request_lock_state = {"held": False}

        @contextlib.contextmanager
        def fake_lock(purpose):
            lock_events.append(("enter", purpose))
            request_lock_state["held"] = True
            try:
                yield Path("/tmp/test.lock")
            finally:
                request_lock_state["held"] = False
                lock_events.append(("exit", purpose))

        server = BlenderServer(port=8000)
        server._running = True

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "scenecode.agent_utils.blender.server_manager.acquire_blender_request_lock",
            side_effect=fake_lock,
        ):
            mesh_path = Path(temp_dir) / "drawer.obj"
            mesh_path.write_text("mesh")
            output_dir = Path(temp_dir) / "renders"

            def fake_request(**kwargs):
                self.assertTrue(request_lock_state["held"])
                self.assertEqual(kwargs["endpoint"], "/render_multiview_articulated")
                self.assertEqual(kwargs["json"]["output_dir"], str(output_dir))
                self.assertEqual(
                    kwargs["json"]["link_meshes"][0]["mesh_paths"],
                    [str(mesh_path)],
                )
                return {
                    "combined_image_paths": [str(output_dir / "combined.png")],
                    "link_image_paths": {
                        "drawer": [str(output_dir / "drawer_0.png")]
                    },
                    "link_dimensions": {"drawer": [1.0, 0.5, 0.25]},
                    "combined_dimensions": [1.0, 0.5, 0.25],
                }

            server._make_request_with_retry = MagicMock(side_effect=fake_request)

            render_result = server.render_multiview_articulated(
                link_meshes=[
                    LinkMeshInfo(
                        link_name="drawer",
                        mesh_paths=[mesh_path],
                        origins=[(0.0, 0.0, 0.0)],
                    )
                ],
                output_dir=output_dir,
            )

        self.assertEqual(
            render_result.combined_image_paths,
            [output_dir / "combined.png"],
        )
        self.assertEqual(
            render_result.link_image_paths,
            {"drawer": [output_dir / "drawer_0.png"]},
        )
        self.assertEqual(
            lock_events,
            [
                ("enter", "render_multiview.articulated"),
                ("exit", "render_multiview.articulated"),
            ],
        )

    @patch("scenecode.agent_utils.rendering.ApplyCameraConfig")
    @patch("scenecode.agent_utils.rendering.DiagramBuilder")
    @patch("scenecode.agent_utils.rendering.create_drake_plant_and_scene_graph_from_scene")
    @patch("scenecode.agent_utils.rendering.requests.post")
    def test_render_per_wall_ortho_views_locks_overlay_request(
        self,
        mock_post,
        mock_create_scene,
        mock_builder_cls,
        mock_apply_camera_config,
    ):
        del mock_apply_camera_config
        lock_events = []
        request_lock_state = {"held": False}

        @contextlib.contextmanager
        def fake_lock(purpose):
            lock_events.append(("enter", purpose))
            request_lock_state["held"] = True
            try:
                yield Path("/tmp/test.lock")
            finally:
                request_lock_state["held"] = False
                lock_events.append(("exit", purpose))

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "scenecode.agent_utils.rendering.acquire_blender_request_lock",
            side_effect=fake_lock,
        ):
            output_dir = Path(temp_dir)
            server = MagicMock()
            server.get_url.return_value = "http://127.0.0.1:8000"
            scene = MagicMock()
            scene.objects = {}
            cfg = SimpleNamespace(background_color=[1.0, 1.0, 1.0])

            def fake_post(*args, **kwargs):
                self.assertTrue(request_lock_state["held"])
                return MagicMock(status_code=200, text="ok")

            def on_eval():
                self.assertTrue(request_lock_state["held"])
                (output_dir / "raw.png").write_text("png")

            mock_post.side_effect = fake_post
            mock_create_scene.return_value = (MagicMock(), MagicMock())
            mock_builder_cls.return_value = _FakeBuilder(on_eval)

            image_paths = render_per_wall_ortho_views(
                scene=scene,
                server=server,
                wall_surfaces=[{"surface_id": "surface_1", "wall_id": "wall_1"}],
                wall_furniture_map={},
                base_config_payload={},
                output_dir=output_dir,
                cfg=cfg,
            )

        self.assertEqual(image_paths, [output_dir / "wall_wall_1_ortho.png"])
        self.assertEqual(
            lock_events,
            [
                ("enter", "render_overlay.wall:wall_1"),
                ("exit", "render_overlay.wall:wall_1"),
            ],
        )
        self.assertEqual(mock_post.call_count, 1)

    @patch("scenecode.robot_eval.tools.vision_tools.ApplyCameraConfig")
    @patch("scenecode.robot_eval.tools.vision_tools.create_plant_from_dmd")
    @patch("scenecode.robot_eval.tools.vision_tools.requests.post")
    def test_render_validation_scene_locks_overlay_request(
        self,
        mock_post,
        mock_create_plant,
        mock_apply_camera_config,
    ):
        del mock_apply_camera_config
        lock_events = []
        request_lock_state = {"held": False}

        @contextlib.contextmanager
        def fake_lock(purpose):
            lock_events.append(("enter", purpose))
            request_lock_state["held"] = True
            try:
                yield Path("/tmp/test.lock")
            finally:
                request_lock_state["held"] = False
                lock_events.append(("exit", purpose))

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "scenecode.robot_eval.tools.vision_tools.acquire_blender_request_lock",
            side_effect=fake_lock,
        ):
            output_dir = Path(temp_dir) / "renders"
            blender_server = MagicMock()
            blender_server.get_url.return_value = "http://127.0.0.1:8000"
            scene = MagicMock()
            scene.dmd_path = Path(temp_dir) / "scene.yaml"
            scene.scene_dir = Path(temp_dir)
            scene.scene_state = {"_walls": []}

            def fake_post(*args, **kwargs):
                self.assertTrue(request_lock_state["held"])
                return MagicMock(status_code=200, text="ok")

            def on_eval():
                self.assertTrue(request_lock_state["held"])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "top.png").write_text("png")

            mock_post.side_effect = fake_post
            mock_create_plant.return_value = (_FakeBuilder(on_eval), MagicMock(), MagicMock())

            image_paths = _render_validation_scene(
                scene=scene,
                blender_server=blender_server,
                output_dir=output_dir,
            )

        self.assertEqual(image_paths, [output_dir / "top.png"])
        self.assertEqual(
            lock_events,
            [
                ("enter", "render_overlay.robot_eval"),
                ("exit", "render_overlay.robot_eval"),
            ],
        )
        self.assertEqual(mock_post.call_count, 1)

    def test_floor_plan_render_config_and_drake_eval_share_lock(self):
        source_path = (
            Path(__file__).resolve().parents[2]
            / "scenecode"
            / "floor_plan_agents"
            / "tools"
            / "vision_tools.py"
        )
        source = source_path.read_text(encoding="utf-8")

        lock_index = source.index('with acquire_blender_request_lock("render_floor_plan"):')
        post_index = source.index("config_response = requests.post", lock_index)
        eval_index = source.index('diagram.GetOutputPort("rgba_image").Eval(context)', lock_index)
        label_index = source.index("self._add_room_labels(output_path)", eval_index)
        material_index = source.index(
            "material_usage = self._compute_material_usage_from_layout()",
            eval_index,
        )

        self.assertLess(lock_index, post_index)
        self.assertLess(post_index, eval_index)
        self.assertLess(eval_index, label_index)
        self.assertLess(eval_index, material_index)

    def test_build_scene_object_metadata_uses_effective_bbox_center(self):
        obj = SceneObject(
            object_id=UniqueID("wardrobe_0"),
            object_type=ObjectType.FURNITURE,
            name="Wardrobe",
            description="Test articulated wardrobe",
            transform=RigidTransform(p=[1.0, 2.0, 0.0]),
            internal_model_pose=RigidTransform(p=[0.0, -0.5, 0.0]),
            bbox_min=np.array([0.0, 0.0, 0.0]),
            bbox_max=np.array([2.0, 2.0, 2.0]),
        )

        metadata = _build_scene_object_metadata(obj)

        self.assertEqual(metadata["position"], [1.0, 2.0, 0.0])
        np.testing.assert_array_almost_equal(
            metadata["bounding_box"]["center"],
            [2.0, 2.5, 1.0],
        )

    @patch("scenecode.agent_utils.rendering.BlenderServer")
    @patch("scenecode.agent_utils.rendering.ApplyCameraConfig")
    @patch("scenecode.agent_utils.rendering.DiagramBuilder")
    @patch("scenecode.agent_utils.rendering.create_drake_plant_and_scene_graph_from_scene")
    @patch("scenecode.agent_utils.rendering.requests.post")
    def test_save_scene_as_blend_locks_per_export_attempt(
        self,
        mock_post,
        mock_create_scene,
        mock_builder_cls,
        mock_apply_camera_config,
        mock_server_cls,
    ):
        del mock_apply_camera_config
        lock_events = []
        request_lock_state = {"held": False}

        @contextlib.contextmanager
        def fake_lock(purpose):
            lock_events.append(("enter", purpose))
            request_lock_state["held"] = True
            try:
                yield Path("/tmp/test.lock")
            finally:
                request_lock_state["held"] = False
                lock_events.append(("exit", purpose))

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "scenecode.agent_utils.rendering.acquire_blender_request_lock",
            side_effect=fake_lock,
        ):
            output_path = Path(temp_dir) / "scene.blend"
            mock_server = MagicMock()
            mock_server.get_url.return_value = "http://127.0.0.1:8000"
            mock_server.is_running.return_value = True
            mock_server_cls.return_value = mock_server

            def fake_post(*args, **kwargs):
                self.assertTrue(request_lock_state["held"])
                return MagicMock(status_code=200, text="ok")

            def on_eval():
                self.assertTrue(request_lock_state["held"])
                output_path.write_text("blend")

            mock_post.side_effect = fake_post
            mock_create_scene.return_value = (MagicMock(), MagicMock())
            mock_builder_cls.return_value = _FakeBuilder(on_eval)

            result = save_scene_as_blend(scene=MagicMock(), output_path=output_path)

        self.assertEqual(result, output_path)
        self.assertEqual(
            lock_events,
            [("enter", "save_blend.scene"), ("exit", "save_blend.scene")],
        )
        self.assertEqual(mock_post.call_count, 1)
        mock_server.start.assert_called_once()
        mock_server.stop.assert_called_once()

    @patch("scenecode.agent_utils.rendering.BlenderServer")
    @patch("scenecode.agent_utils.rendering.ApplyCameraConfig")
    @patch("scenecode.agent_utils.rendering.create_plant_from_dmd")
    @patch("scenecode.agent_utils.rendering.requests.post")
    def test_save_directive_as_blend_locks_each_retry(
        self,
        mock_post,
        mock_create_plant,
        mock_apply_camera_config,
        mock_server_cls,
    ):
        del mock_apply_camera_config
        lock_events = []
        request_lock_state = {"held": False}

        @contextlib.contextmanager
        def fake_lock(purpose):
            lock_events.append(("enter", purpose))
            request_lock_state["held"] = True
            try:
                yield Path("/tmp/test.lock")
            finally:
                request_lock_state["held"] = False
                lock_events.append(("exit", purpose))

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "scenecode.agent_utils.rendering.acquire_blender_request_lock",
            side_effect=fake_lock,
        ):
            output_path = Path(temp_dir) / "scene.blend"
            directive_path = Path(temp_dir) / "scene.yaml"
            directive_path.write_text("directives: []")
            first_builder = _FakeBuilder(lambda: None)
            second_builder = _FakeBuilder(lambda: output_path.write_text("blend"))

            first_server = MagicMock()
            first_server.get_url.return_value = "http://127.0.0.1:8000"
            first_server.is_running.return_value = True
            second_server = MagicMock()
            second_server.get_url.return_value = "http://127.0.0.1:8001"
            second_server.is_running.return_value = True
            mock_server_cls.side_effect = [first_server, second_server]

            def fake_post(*args, **kwargs):
                self.assertTrue(request_lock_state["held"])
                return MagicMock(status_code=200, text="ok")

            mock_post.side_effect = fake_post
            mock_create_plant.side_effect = [
                (first_builder, MagicMock(), MagicMock()),
                (second_builder, MagicMock(), MagicMock()),
            ]

            result = save_directive_as_blend(
                directive_path=directive_path,
                output_path=output_path,
                max_retries=2,
            )

        self.assertEqual(result, output_path)
        self.assertEqual(
            lock_events,
            [
                ("enter", "save_blend.directive.attempt_1"),
                ("exit", "save_blend.directive.attempt_1"),
                ("enter", "save_blend.directive.attempt_2"),
                ("exit", "save_blend.directive.attempt_2"),
            ],
        )
        self.assertEqual(mock_post.call_count, 2)
        first_server.stop.assert_called_once()
        second_server.stop.assert_called_once()
