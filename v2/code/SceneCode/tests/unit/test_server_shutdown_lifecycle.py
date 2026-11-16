import shutil
import sys
import tempfile
import threading
import types
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

sys.modules.setdefault("bpy", types.ModuleType("bpy"))

from omegaconf import OmegaConf

from scenecode.agent_utils.articulated_retrieval_server.server_app import (
    ArticulatedRetrievalApp,
)
from scenecode.agent_utils.articulated_retrieval_server.server_manager import (
    ArticulatedRetrievalServer,
)
from scenecode.agent_utils.geometry_generation_server.server_app import (
    GeometryGenerationApp,
)
from scenecode.agent_utils.geometry_generation_server.server_manager import (
    GeometryGenerationServer,
)
from scenecode.agent_utils.hssd_retrieval_server.server_app import HssdRetrievalApp
from scenecode.agent_utils.hssd_retrieval_server.server_manager import (
    HssdRetrievalServer,
)
from scenecode.agent_utils.materials_retrieval_server.server_app import (
    MaterialsRetrievalApp,
)
from scenecode.agent_utils.materials_retrieval_server.server_manager import (
    MaterialsRetrievalServer,
)
from scenecode.agent_utils.objaverse_retrieval_server.server_app import (
    ObjaverseRetrievalApp,
)
from scenecode.agent_utils.objaverse_retrieval_server.server_manager import (
    ObjaverseRetrievalServer,
)
from scenecode.experiments.indoor_scene_generation import (
    IndoorSceneGenerationExperiment,
)


class TestManagerShutdownLifecycle(unittest.TestCase):
    @patch(
        "scenecode.agent_utils.materials_retrieval_server.server_manager.is_port_available",
        return_value=True,
    )
    def test_materials_stop_uses_owned_server_lifecycle(self, _mock_port_available):
        server = MaterialsRetrievalServer()
        server._running = True
        server._app = MagicMock()
        server._wsgi_server = MagicMock()
        server._server_thread = MagicMock()

        call_order = []
        server._app.stop_processing.side_effect = lambda: call_order.append("processing")

        with patch(
            "scenecode.agent_utils.materials_retrieval_server.server_manager.stop_threaded_wsgi_server",
            side_effect=lambda *args, **kwargs: call_order.append("wsgi"),
        ), patch(
            "scenecode.agent_utils.materials_retrieval_server.server_manager.requests.post"
        ) as mock_post:
            server.stop()

        self.assertEqual(call_order, ["processing", "wsgi"])
        mock_post.assert_not_called()
        self.assertFalse(server._running)
        self.assertIsNone(server._app)
        self.assertIsNone(server._wsgi_server)
        self.assertIsNone(server._server_thread)

    @patch(
        "scenecode.agent_utils.materials_retrieval_server.server_manager.is_port_available",
        return_value=True,
    )
    def test_materials_stop_preserves_state_on_shutdown_failure(
        self, _mock_port_available
    ):
        server = MaterialsRetrievalServer()
        server._running = True
        server._app = MagicMock()
        server._wsgi_server = MagicMock()
        server._server_thread = MagicMock()

        with patch(
            "scenecode.agent_utils.materials_retrieval_server.server_manager.stop_threaded_wsgi_server",
            side_effect=RuntimeError("stuck"),
        ):
            with self.assertRaisesRegex(RuntimeError, "stuck"):
                server.stop()

        server._app.stop_processing.assert_called_once()
        self.assertTrue(server._running)
        self.assertIsNotNone(server._app)
        self.assertIsNotNone(server._wsgi_server)
        self.assertIsNotNone(server._server_thread)

    @patch(
        "scenecode.agent_utils.geometry_generation_server.server_manager.is_port_available",
        return_value=True,
    )
    def test_geometry_stop_uses_owned_server_lifecycle(self, _mock_port_available):
        server = GeometryGenerationServer()
        server._running = True
        server._app = MagicMock()
        server._wsgi_server = MagicMock()
        server._server_thread = MagicMock()

        call_order = []
        server._app.stop_processing.side_effect = lambda: call_order.append("processing")

        with patch(
            "scenecode.agent_utils.geometry_generation_server.server_manager.stop_threaded_wsgi_server",
            side_effect=lambda *args, **kwargs: call_order.append("wsgi"),
        ), patch(
            "scenecode.agent_utils.geometry_generation_server.server_manager.requests.post"
        ) as mock_post:
            server.stop()

        self.assertEqual(call_order, ["processing", "wsgi"])
        mock_post.assert_not_called()
        self.assertFalse(server._running)
        self.assertIsNone(server._app)
        self.assertIsNone(server._wsgi_server)
        self.assertIsNone(server._server_thread)

    @patch(
        "scenecode.agent_utils.geometry_generation_server.server_manager.is_port_available",
        return_value=True,
    )
    def test_geometry_stop_preserves_state_on_shutdown_failure(
        self, _mock_port_available
    ):
        server = GeometryGenerationServer()
        server._running = True
        server._app = MagicMock()
        server._wsgi_server = MagicMock()
        server._server_thread = MagicMock()

        with patch(
            "scenecode.agent_utils.geometry_generation_server.server_manager.stop_threaded_wsgi_server",
            side_effect=RuntimeError("stuck"),
        ):
            with self.assertRaisesRegex(RuntimeError, "stuck"):
                server.stop()

        server._app.stop_processing.assert_called_once()
        self.assertTrue(server._running)
        self.assertIsNotNone(server._app)
        self.assertIsNotNone(server._wsgi_server)
        self.assertIsNotNone(server._server_thread)

    @patch(
        "scenecode.agent_utils.hssd_retrieval_server.server_manager.is_port_available",
        return_value=True,
    )
    def test_hssd_stop_uses_owned_server_lifecycle(self, _mock_port_available):
        server = HssdRetrievalServer()
        server._running = True
        server._app = MagicMock()
        server._wsgi_server = MagicMock()
        server._server_thread = MagicMock()

        call_order = []
        server._app.stop_processing.side_effect = lambda: call_order.append("processing")

        with patch(
            "scenecode.agent_utils.hssd_retrieval_server.server_manager.stop_threaded_wsgi_server",
            side_effect=lambda *args, **kwargs: call_order.append("wsgi"),
        ), patch(
            "scenecode.agent_utils.hssd_retrieval_server.server_manager.requests.post"
        ) as mock_post:
            server.stop()

        self.assertEqual(call_order, ["processing", "wsgi"])
        mock_post.assert_not_called()
        self.assertFalse(server._running)
        self.assertIsNone(server._app)
        self.assertIsNone(server._wsgi_server)
        self.assertIsNone(server._server_thread)

    @patch(
        "scenecode.agent_utils.objaverse_retrieval_server.server_manager.is_port_available",
        return_value=True,
    )
    def test_objaverse_stop_uses_owned_server_lifecycle(self, _mock_port_available):
        server = ObjaverseRetrievalServer()
        server._running = True
        server._app = MagicMock()
        server._wsgi_server = MagicMock()
        server._server_thread = MagicMock()

        call_order = []
        server._app.stop_processing.side_effect = lambda: call_order.append("processing")

        with patch(
            "scenecode.agent_utils.objaverse_retrieval_server.server_manager.stop_threaded_wsgi_server",
            side_effect=lambda *args, **kwargs: call_order.append("wsgi"),
        ), patch(
            "scenecode.agent_utils.objaverse_retrieval_server.server_manager.requests.post"
        ) as mock_post:
            server.stop()

        self.assertEqual(call_order, ["processing", "wsgi"])
        mock_post.assert_not_called()
        self.assertFalse(server._running)
        self.assertIsNone(server._app)
        self.assertIsNone(server._wsgi_server)
        self.assertIsNone(server._server_thread)

    @patch(
        "scenecode.agent_utils.articulated_retrieval_server.server_manager.is_port_available",
        return_value=True,
    )
    def test_articulated_stop_uses_owned_server_lifecycle(
        self, _mock_port_available
    ):
        server = ArticulatedRetrievalServer()
        server._running = True
        server._app = MagicMock()
        server._wsgi_server = MagicMock()
        server._server_thread = MagicMock()

        call_order = []
        server._app.stop_processing.side_effect = lambda: call_order.append("processing")

        with patch(
            "scenecode.agent_utils.articulated_retrieval_server.server_manager.stop_threaded_wsgi_server",
            side_effect=lambda *args, **kwargs: call_order.append("wsgi"),
        ), patch(
            "scenecode.agent_utils.articulated_retrieval_server.server_manager.requests.post"
        ) as mock_post:
            server.stop()

        self.assertEqual(call_order, ["processing", "wsgi"])
        mock_post.assert_not_called()
        self.assertFalse(server._running)
        self.assertIsNone(server._app)
        self.assertIsNone(server._wsgi_server)
        self.assertIsNone(server._server_thread)


class _FakeManagedServer:
    def __init__(self, name: str, call_log: list[str]):
        self._name = name
        self._call_log = call_log
        self._running = False

    def start(self) -> None:
        self._call_log.append(f"{self._name}:start")
        self._running = True

    def wait_until_ready(self, timeout_s: float = 30) -> None:
        del timeout_s
        if not self._running:
            raise RuntimeError(f"{self._name} not running")
        self._call_log.append(f"{self._name}:ready")

    def stop(self) -> None:
        self._call_log.append(f"{self._name}:stop")
        self._running = False

    def is_running(self) -> bool:
        return self._running


class TestIndoorSceneGenerationLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="pipeline_lifecycle_test_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _build_minimal_cfg(self):
        shared_asset_manager = {
            "backend": "hunyuan3d",
            "general_asset_source": "hssd",
            "router": {
                "enabled": True,
                "strategies": {
                    "generated": {"enabled": True},
                    "articulated": {"enabled": True},
                },
            },
            "hssd": {
                "data_path": "data/hssd-models",
                "preprocessed_path": "data/preprocessed",
                "use_top_k": 5,
            },
            "objaverse": {
                "data_path": "data/objathor-assets",
                "preprocessed_path": "data/objathor-assets/preprocessed",
                "use_top_k": 5,
            },
            "articulated": {
                "embedding_path": "data/articulated/embedding.pt",
            },
        }

        return OmegaConf.create(
            {
                "name": "test_pipeline_startup_shutdown_only",
                "experiment": {
                    "_name": "indoor_scene_generation",
                    "name": "test_pipeline_startup_shutdown_only",
                    "output_dir": str(self.temp_dir),
                    "prompts": ["startup shutdown only"],
                    "csv_path": None,
                    "num_workers": 1,
                    "pipeline": {
                        "start_stage": "floor_plan",
                        "stop_stage": "floor_plan",
                        "parallel_rooms": False,
                        "max_parallel_rooms": 1,
                        "resume_from_path": None,
                    },
                    "geometry_generation_server": {
                        "host": "127.0.0.1",
                        "port": 7000,
                    },
                    "hssd_retrieval_server": {
                        "host": "127.0.0.1",
                        "port": 7001,
                    },
                    "objaverse_retrieval_server": {
                        "host": "127.0.0.1",
                        "port": 7007,
                    },
                    "articulated_retrieval_server": {
                        "host": "127.0.0.1",
                        "port": 7002,
                    },
                    "materials_retrieval_server": {
                        "host": "127.0.0.1",
                        "port": 7008,
                    },
                },
                "furniture_agent": {
                    "_name": "stateful_furniture_agent",
                    "asset_manager": shared_asset_manager,
                },
                "manipuland_agent": {
                    "_name": "stateful_manipuland_agent",
                    "asset_manager": {
                        **shared_asset_manager,
                        "general_asset_source": "hssd",
                    },
                },
                "wall_agent": {
                    "_name": "stateful_wall_agent",
                    "asset_manager": {
                        **shared_asset_manager,
                        "general_asset_source": "objaverse",
                    },
                },
                "ceiling_agent": {
                    "_name": "stateful_ceiling_agent",
                    "asset_manager": shared_asset_manager,
                },
                "floor_plan_agent": {
                    "_name": "stateful_floor_plan_agent",
                },
            }
        )

    def test_generate_scenes_only_exercises_server_lifecycle(self):
        cfg = self._build_minimal_cfg()
        call_log: list[str] = []

        def make_server(name: str):
            return lambda *args, **kwargs: _FakeManagedServer(name, call_log)

        experiment = IndoorSceneGenerationExperiment(cfg=cfg)

        with patch(
            "scenecode.experiments.indoor_scene_generation.GeometryGenerationServer",
            side_effect=make_server("geometry"),
        ), patch(
            "scenecode.experiments.indoor_scene_generation.HssdRetrievalServer",
            side_effect=make_server("hssd"),
        ), patch(
            "scenecode.experiments.indoor_scene_generation.ObjaverseRetrievalServer",
            side_effect=make_server("objaverse"),
        ), patch(
            "scenecode.experiments.indoor_scene_generation.ArticulatedRetrievalServer",
            side_effect=make_server("articulated"),
        ), patch.object(
            IndoorSceneGenerationExperiment,
            "_run_serial_generation",
            autospec=True,
            side_effect=lambda *args, **kwargs: call_log.append("generation"),
        ) as mock_run_serial:
            experiment.generate_scenes()

        self.assertEqual(mock_run_serial.call_count, 1)
        self.assertEqual(
            call_log,
            [
                "geometry:start",
                "geometry:ready",
                "hssd:start",
                "hssd:ready",
                "objaverse:start",
                "objaverse:ready",
                "articulated:start",
                "articulated:ready",
                "generation",
                "articulated:stop",
                "objaverse:stop",
                "hssd:stop",
                "geometry:stop",
            ],
        )
        self.assertIsNone(experiment.geometry_server)
        self.assertIsNone(experiment.hssd_server)
        self.assertIsNone(experiment.objaverse_server)
        self.assertIsNone(experiment.articulated_server)


class TestShutdownEndpoints(unittest.TestCase):
    def _assert_shutdown_endpoint(self, app_factory):
        app = app_factory()
        callback_called = threading.Event()
        app.register_shutdown_callback(callback_called.set)
        response = app.test_client().post("/shutdown")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "shutting down")
        self.assertTrue(callback_called.wait(timeout=1.0))

    def _assert_shutdown_endpoint_requires_callback(self, app_factory):
        app = app_factory()
        response = app.test_client().post("/shutdown")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["message"], "shutdown unavailable")

    def test_shutdown_endpoint_invokes_callback_for_supported_apps(self):
        factories = [
            ("materials", lambda: MaterialsRetrievalApp(preload_retriever=False)),
            ("articulated", lambda: ArticulatedRetrievalApp(preload_retriever=False)),
            ("hssd", lambda: HssdRetrievalApp(preload_retriever=False)),
            ("objaverse", lambda: ObjaverseRetrievalApp(preload_retriever=False)),
        ]

        for name, factory in factories:
            with self.subTest(name=name):
                self._assert_shutdown_endpoint(factory)

        with patch(
            "scenecode.agent_utils.geometry_generation_server.server_app.GPUWorkerPool"
        ):
            self._assert_shutdown_endpoint(
                lambda: GeometryGenerationApp(preload_pipeline=False)
            )

    def test_shutdown_endpoint_requires_registered_callback(self):
        factories = [
            ("materials", lambda: MaterialsRetrievalApp(preload_retriever=False)),
            ("articulated", lambda: ArticulatedRetrievalApp(preload_retriever=False)),
            ("hssd", lambda: HssdRetrievalApp(preload_retriever=False)),
            ("objaverse", lambda: ObjaverseRetrievalApp(preload_retriever=False)),
        ]

        for name, factory in factories:
            with self.subTest(name=name):
                self._assert_shutdown_endpoint_requires_callback(factory)

        with patch(
            "scenecode.agent_utils.geometry_generation_server.server_app.GPUWorkerPool"
        ):
            self._assert_shutdown_endpoint_requires_callback(
                lambda: GeometryGenerationApp(preload_pipeline=False)
            )
