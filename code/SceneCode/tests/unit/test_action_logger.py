"""Unit tests for action logging functionality."""

import json
import tempfile
import unittest

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from pydrake.all import Quaternion, RigidTransform, RotationMatrix

from scenecode.agent_utils.action_logger import (
    ActionLogEntry,
    _serialize_value,
    load_action_log,
    log_scene_action,
)
from scenecode.agent_utils.room import UniqueID


class TestSerializeValue(unittest.TestCase):
    """Tests for _serialize_value function."""

    def test_serialize_none(self):
        """Test serialization of None."""
        assert _serialize_value(None) is None

    def test_serialize_primitives(self):
        """Test serialization of primitive types."""
        assert _serialize_value("hello") == "hello"
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value(True) is True
        assert _serialize_value(False) is False

    def test_serialize_path(self):
        """Test serialization of Path objects."""
        path = Path("/home/user/test.txt")
        assert _serialize_value(path) == "/home/user/test.txt"

    def test_serialize_custom_class_with_str(self):
        """Test serialization of custom classes with __str__ method."""
        uid = UniqueID("object_123")
        assert _serialize_value(uid) == "object_123"

    def test_serialize_rigid_transform(self):
        """Test serialization of RigidTransform objects."""
        # Create a simple transform.
        translation = np.array([1.0, 2.0, 3.0])
        quaternion = Quaternion(wxyz=[1.0, 0.0, 0.0, 0.0])  # Identity rotation.
        rotation = RotationMatrix(quaternion)
        transform = RigidTransform(rotation, translation)

        result = _serialize_value(transform)

        assert "position" in result
        assert "quaternion" in result
        assert result["position"] == [1.0, 2.0, 3.0]
        assert result["quaternion"] == [1.0, 0.0, 0.0, 0.0]

    def test_serialize_dataclass(self):
        """Test serialization of dataclass instances."""

        @dataclass
        class TestData:
            name: str
            value: int

        data = TestData(name="test", value=42)
        result = _serialize_value(data)

        assert result == {"name": "test", "value": 42}

    def test_serialize_nested_dataclass_with_complex_types(self):
        """Test serialization of nested dataclass with Path and RigidTransform."""

        @dataclass
        class ComplexData:
            path: Path
            transform: RigidTransform
            name: str

        data = ComplexData(
            path=Path("/test/file.txt"),
            transform=RigidTransform(p=[1.0, 2.0, 3.0]),
            name="test_object",
        )
        result = _serialize_value(data)

        assert isinstance(result, dict)
        assert result["path"] == "/test/file.txt"
        assert isinstance(result["transform"], dict)
        assert "position" in result["transform"]
        assert "quaternion" in result["transform"]
        assert result["transform"]["position"] == [1.0, 2.0, 3.0]
        assert result["name"] == "test_object"

    def test_serialize_list(self):
        """Test serialization of lists."""
        data = [1, 2, "three", Path("/test")]
        result = _serialize_value(data)

        assert result == [1, 2, "three", "/test"]

    def test_serialize_tuple(self):
        """Test serialization of tuples."""
        data = (1, 2, "three")
        result = _serialize_value(data)

        assert result == [1, 2, "three"]

    def test_serialize_dict(self):
        """Test serialization of dictionaries."""
        data = {
            "name": "test",
            "path": Path("/test"),
            "id": UniqueID("obj_1"),
        }
        result = _serialize_value(data)

        assert result == {
            "name": "test",
            "path": "/test",
            "id": "obj_1",
        }

    def test_serialize_nested_structures(self):
        """Test serialization of nested data structures."""
        data = {
            "items": [
                {"id": UniqueID("a"), "value": 1},
                {"id": UniqueID("b"), "value": 2},
            ],
            "config": {
                "path": Path("/config"),
                "enabled": True,
            },
        }
        result = _serialize_value(data)

        assert result == {
            "items": [
                {"id": "a", "value": 1},
                {"id": "b", "value": 2},
            ],
            "config": {
                "path": "/config",
                "enabled": True,
            },
        }

    def test_serialize_unsupported_type_raises_error(self):
        """Test that unsupported types raise TypeError."""

        class UnsupportedType:
            # Override __str__ to raise error, making it unsuitable for serialization.
            def __str__(self):
                raise ValueError("Cannot convert to string")

        with self.assertRaises(ValueError) as ctx:
            _serialize_value(UnsupportedType())

        self.assertIn("Cannot convert to string", str(ctx.exception))

    def test_serialize_numpy_array(self):
        """Test serialization of numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _serialize_value(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_serialize_numpy_array_with_nan_raises(self):
        """Test that numpy arrays with NaN values raise ValueError."""
        arr = np.array([1.0, np.nan, 3.0])
        with self.assertRaises(ValueError) as ctx:
            _serialize_value(arr)
        self.assertIn("NaN or Inf", str(ctx.exception))

    def test_serialize_numpy_array_with_inf_raises(self):
        """Test that numpy arrays with Inf values raise ValueError."""
        arr = np.array([1.0, np.inf, 3.0])
        with self.assertRaises(ValueError) as ctx:
            _serialize_value(arr)
        self.assertIn("NaN or Inf", str(ctx.exception))

    def test_serialize_numpy_integer_preserves_type(self):
        """Test that numpy integers are preserved as int, not converted to float."""
        value = np.int64(42)
        result = _serialize_value(value)
        assert result == 42
        assert isinstance(result, int)

    def test_serialize_numpy_float(self):
        """Test serialization of numpy float types."""
        value = np.float64(3.14)
        result = _serialize_value(value)
        assert result == 3.14
        assert isinstance(result, float)

    def test_serialize_numpy_float_with_nan_raises(self):
        """Test that numpy float scalars with NaN raise ValueError."""
        value = np.float64(np.nan)
        with self.assertRaises(ValueError) as ctx:
            _serialize_value(value)
        self.assertIn("NaN or Inf", str(ctx.exception))

    def test_serialize_numpy_bool(self):
        """Test serialization of numpy bool types."""
        value_true = np.bool_(True)
        value_false = np.bool_(False)
        assert _serialize_value(value_true) is True
        assert _serialize_value(value_false) is False

    def test_serialize_enum(self):
        """Test serialization of Enum types."""

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        result = _serialize_value(Color.RED)
        assert result == "red"


class TestLogSceneActionDecorator(unittest.TestCase):
    """Tests for @log_scene_action decorator."""

    def test_decorator_with_scene_first_arg(self):
        """Test decorator when RoomScene is first argument."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "action_log.json"

            # Create a mock scene with action_log_path.
            mock_scene = MagicMock()
            mock_scene.action_log_path = log_path

            @log_scene_action
            def test_function(scene, x: float, y: float, name: str = "default"):
                return f"Called with {x}, {y}, {name}"

            # Call the function.
            result = test_function(mock_scene, x=1.0, y=2.0, name="test")

            self.assertEqual(result, "Called with 1.0, 2.0, test")

            # Check that action log was created.
            self.assertTrue(log_path.exists())

            # Load and verify log contents.
            with open(log_path, encoding="utf-8") as f:
                log_entries = json.load(f)

            self.assertEqual(len(log_entries), 1)
            entry = log_entries[0]
            self.assertEqual(entry["step_number"], 1)
            self.assertEqual(entry["tool_name"], "test_function")
            self.assertEqual(entry["arguments"], {"x": 1.0, "y": 2.0, "name": "test"})

    def test_decorator_with_self_having_scene(self):
        """Test decorator when first arg is self with .scene attribute."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "action_log.json"

            # Create a simple class to hold the scene.
            class MockTools:
                def __init__(self, scene):
                    self.scene = scene

            class MockScene:
                def __init__(self, log_path):
                    self.action_log_path = log_path

            mock_scene = MockScene(log_path=log_path)
            mock_self = MockTools(scene=mock_scene)

            @log_scene_action
            def test_method(self, object_id: str, value: int):
                return f"Method called with {object_id}, {value}"

            # Call the method.
            result = test_method(mock_self, object_id="obj_1", value=42)

            self.assertEqual(result, "Method called with obj_1, 42")

            # Verify log was created.
            self.assertTrue(log_path.exists())

            with open(log_path, encoding="utf-8") as f:
                log_entries = json.load(f)

            self.assertEqual(len(log_entries), 1)
            self.assertEqual(
                log_entries[0]["arguments"], {"object_id": "obj_1", "value": 42}
            )

    def test_decorator_skips_when_no_action_log_path(self):
        """Test decorator skips logging when action_log_path is None."""
        mock_scene = MagicMock()
        mock_scene.action_log_path = None

        @log_scene_action
        def test_function(scene, x: float):
            return f"Called with {x}"

        result = test_function(mock_scene, x=1.0)

        self.assertEqual(result, "Called with 1.0")
        # No log file should be created.

    def test_decorator_handles_defaults(self):
        """Test decorator includes default argument values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "action_log.json"

            mock_scene = MagicMock()
            mock_scene.action_log_path = log_path

            @log_scene_action
            def test_function(
                scene,
                x: float,
                y: float = 0.0,
                z: float = 0.0,
            ):
                return "ok"

            # Call with only required args.
            test_function(mock_scene, x=1.0)

            with open(log_path, encoding="utf-8") as f:
                log_entries = json.load(f)

            # Default values should be included.
            self.assertEqual(
                log_entries[0]["arguments"], {"x": 1.0, "y": 0.0, "z": 0.0}
            )

    def test_decorator_multiple_calls_append(self):
        """Test that multiple calls append to log."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "action_log.json"

            mock_scene = MagicMock()
            mock_scene.action_log_path = log_path

            @log_scene_action
            def test_function(scene, value: int):
                return "ok"

            # Make multiple calls.
            test_function(mock_scene, value=1)
            test_function(mock_scene, value=2)
            test_function(mock_scene, value=3)

            with open(log_path, encoding="utf-8") as f:
                log_entries = json.load(f)

            self.assertEqual(len(log_entries), 3)
            self.assertEqual(log_entries[0]["step_number"], 1)
            self.assertEqual(log_entries[1]["step_number"], 2)
            self.assertEqual(log_entries[2]["step_number"], 3)
            self.assertEqual(log_entries[0]["arguments"]["value"], 1)
            self.assertEqual(log_entries[1]["arguments"]["value"], 2)
            self.assertEqual(log_entries[2]["arguments"]["value"], 3)


class TestLoadActionLog(unittest.TestCase):
    """Tests for load_action_log function."""

    def test_load_valid_log(self):
        """Test loading a valid action log."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "action_log.json"

            # Create a valid log file.
            log_data = [
                {
                    "step_number": 1,
                    "timestamp": "2025-11-07T10:00:00",
                    "tool_name": "test_tool",
                    "arguments": {"x": 1.0, "y": 2.0},
                },
                {
                    "step_number": 2,
                    "timestamp": "2025-11-07T10:00:01",
                    "tool_name": "another_tool",
                    "arguments": {"id": "obj_1"},
                },
            ]

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f)

            # Load the log.
            entries = load_action_log(log_path=log_path)

            self.assertEqual(len(entries), 2)
            self.assertIsInstance(entries[0], ActionLogEntry)
            self.assertEqual(entries[0].step_number, 1)
            self.assertEqual(entries[0].tool_name, "test_tool")
            self.assertEqual(entries[0].arguments, {"x": 1.0, "y": 2.0})

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "nonexistent.json"

            with self.assertRaises(FileNotFoundError) as ctx:
                load_action_log(log_path=log_path)

            self.assertIn("Action log file not found", str(ctx.exception))
            self.assertIn("re-run the experiment", str(ctx.exception))

    def test_load_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "invalid.json"

            with open(log_path, "w", encoding="utf-8") as f:
                f.write("not valid json{")

            with self.assertRaises(ValueError) as ctx:
                load_action_log(log_path=log_path)

            self.assertIn("Failed to parse action log JSON", str(ctx.exception))

    def test_load_non_list_raises_error(self):
        """Test that non-list JSON raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "not_list.json"

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({"not": "a list"}, f)

            with self.assertRaises(ValueError) as ctx:
                load_action_log(log_path=log_path)

            self.assertIn("is not a list", str(ctx.exception))

    def test_load_invalid_entry_raises_error(self):
        """Test that invalid entry structure raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "invalid_entry.json"

            # Missing required fields.
            log_data = [
                {
                    "step_number": 1,
                    # Missing timestamp, tool_name, arguments.
                }
            ]

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f)

            with self.assertRaises(ValueError) as ctx:
                load_action_log(log_path=log_path)

            self.assertIn("Invalid action log entry", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
