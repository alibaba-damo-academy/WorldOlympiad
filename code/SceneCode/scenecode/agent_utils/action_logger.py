"""Action logging infrastructure for scene replication/replay.

This module provides functionality to log scene-modifying tool calls
to enable deterministic replay of scene generation processes.
"""

import inspect
import json
import logging

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pydrake.math import RigidTransform

console_logger = logging.getLogger(__name__)


@dataclass
class ActionLogEntry:
    """Represents a single scene-modifying action in the log."""

    step_number: int
    """Sequential step number for ordering."""

    timestamp: str
    """ISO format timestamp when action was logged."""

    tool_name: str
    """Name of the tool function that was called."""

    arguments: dict[str, Any]
    """Dictionary of serialized function arguments."""

    metadata: dict[str, Any] | None = None
    """Optional metadata providing execution context (e.g., furniture_id for
    manipuland placement)."""


def _serialize_value(value: Any) -> Any:
    """Serialize a value to a JSON-compatible format.

    Args:
        value: The value to serialize.

    Returns:
        JSON-serializable representation of the value.

    Raises:
        TypeError: If the value cannot be serialized.
    """
    # Handle None.
    if value is None:
        return None

    # Handle numpy types before primitives (numpy types are subclasses of Python
    # primitives).
    if isinstance(value, np.ndarray):
        # Validate for NaN/Inf values (fail-fast for research codebase).
        if np.any(np.isnan(value)) or np.any(np.isinf(value)):
            raise ValueError(
                f"Cannot serialize numpy array containing NaN or Inf values: {value}"
            )
        return value.tolist()

    # Handle numpy scalar types.
    if isinstance(value, (np.integer, np.floating)):
        if isinstance(value, np.integer):
            # Preserve integer type.
            return int(value)
        else:
            # Validate for NaN/Inf values (fail-fast for research codebase).
            float_value = float(value)
            if np.isnan(float_value) or np.isinf(float_value):
                raise ValueError(
                    f"Cannot serialize numpy scalar with NaN or Inf value: {value}"
                )
            return float_value

    # Handle numpy bool.
    if isinstance(value, np.bool_):
        return bool(value)

    # Handle primitives (after numpy types since numpy types are subclasses).
    if isinstance(value, (str, int, float, bool)):
        return value

    # Handle Path.
    if isinstance(value, Path):
        return str(value)

    # Handle Enum types.
    if isinstance(value, Enum):
        return value.value

    # Handle RigidTransform.
    if isinstance(value, RigidTransform):
        position = value.translation()
        quaternion = value.rotation().ToQuaternion()
        return {
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "quaternion": [
                float(quaternion.w()),
                float(quaternion.x()),
                float(quaternion.y()),
                float(quaternion.z()),
            ],
        }

    # Handle dataclasses (recursively serialize nested fields).
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _serialize_value(v) for k, v in asdict(value).items()}

    # Handle lists.
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    # Handle tuples.
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]

    # Handle dicts.
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}

    # Handle objects with __str__ method (e.g., UniqueID).
    # This avoids circular import with scene.py.
    if hasattr(value, "__str__") and not isinstance(value, type):
        # Check if this is a custom class (not a built-in type).
        if type(value).__module__ not in ("builtins", "__main__"):
            return str(value)

    # If we can't serialize it, raise an error.
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}: {value}")


def _append_action_to_log(
    log_path: Path,
    tool_name: str,
    arguments: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append an action entry to the action log file.

    Args:
        log_path: Path to the action_log.json file.
        tool_name: Name of the tool function.
        arguments: Dictionary of function arguments.
        metadata: Optional execution context metadata.
    """
    # Serialize arguments.
    try:
        serialized_args = _serialize_value(arguments)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize arguments for {tool_name}: {e}") from e

    # Serialize metadata if present.
    serialized_metadata = None
    if metadata is not None:
        try:
            serialized_metadata = _serialize_value(metadata)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to serialize metadata for {tool_name}: {e}"
            ) from e

    # Load existing log or create empty list.
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                log_entries = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            console_logger.warning(
                f"Failed to load existing action log at {log_path}: {e}. Starting "
                "new log."
            )
            log_entries = []
    else:
        log_entries = []

    # Create new entry.
    step_number = len(log_entries) + 1
    timestamp = datetime.now().isoformat()
    entry = ActionLogEntry(
        step_number=step_number,
        timestamp=timestamp,
        tool_name=tool_name,
        arguments=serialized_args,
        metadata=serialized_metadata,
    )

    # Append and save.
    log_entries.append(asdict(entry))
    try:
        # Ensure parent directory exists.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=2)
    except IOError as e:
        raise IOError(f"Failed to write action log to {log_path}: {e}") from e


def log_scene_action(func: Callable) -> Callable:
    """Decorator to log scene-modifying tool calls.

    This decorator captures the function name and arguments, serializes them,
    and appends them to the action log file associated with the Scene.

    The decorator supports two patterns:
    1. Scene object as first argument: func(scene, ...)
    2. Object with scene attribute as first argument: func(self, ...)
       where self.scene exists

    Args:
        func: The tool function to decorate.

    Returns:
        Wrapped function that logs actions before execution.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract scene object.
        if len(args) == 0:
            console_logger.warning(
                f"@log_scene_action decorator on {func.__name__} called with no "
                "arguments. Expected Scene as first argument or object with .scene "
                "attribute. Skipping action logging."
            )
            return func(*args, **kwargs)

        first_arg = args[0]

        # Check if first arg is Scene directly or has .scene attribute.
        scene = None
        if hasattr(first_arg, "action_log_path"):
            # First arg is the Scene object directly.
            scene = first_arg
        elif hasattr(first_arg, "scene"):
            # First arg is an object (like self) with a scene attribute.
            scene = first_arg.scene
        else:
            console_logger.debug(
                f"First argument to {func.__name__} does not have action_log_path "
                "or .scene attribute. Skipping action logging."
            )
            return func(*args, **kwargs)

        # Check if scene has action_log_path attribute.
        if not hasattr(scene, "action_log_path"):
            console_logger.debug(
                f"Scene object for {func.__name__} does not have action_log_path "
                "attribute. Skipping action logging."
            )
            return func(*args, **kwargs)

        action_log_path = scene.action_log_path
        if action_log_path is None:
            console_logger.debug(
                f"Scene.action_log_path is None for {func.__name__}. Skipping "
                "action logging."
            )
            return func(*args, **kwargs)

        # Build arguments dict and get function signature to include defaults.
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Remove first parameter (scene or self).
        arguments_dict = dict(bound_args.arguments)
        param_names = list(sig.parameters.keys())
        if len(param_names) > 0:
            first_param = param_names[0]
            if first_param in arguments_dict:
                del arguments_dict[first_param]

        # Extract metadata if present (special keyword argument).
        metadata = arguments_dict.pop("_action_metadata", None)

        # Also check inside kwargs if it wasn't at top level (for functions with
        # **kwargs).
        if metadata is None and "kwargs" in arguments_dict:
            kwargs_dict = arguments_dict.get("kwargs", {})
            if isinstance(kwargs_dict, dict) and "_action_metadata" in kwargs_dict:
                metadata = kwargs_dict.pop("_action_metadata")
                # Remove empty kwargs dict to keep arguments clean.
                if not kwargs_dict:
                    arguments_dict.pop("kwargs")

        # Log the action.
        _append_action_to_log(
            log_path=action_log_path,
            tool_name=func.__name__,
            arguments=arguments_dict,
            metadata=metadata,
        )

        # Execute the actual function.
        return func(*args, **kwargs)

    return wrapper


def load_action_log(log_path: Path) -> list[ActionLogEntry]:
    """Load action log from file.

    Args:
        log_path: Path to the action_log.json file.

    Returns:
        List of ActionLogEntry objects.

    Raises:
        FileNotFoundError: If the log file does not exist.
        ValueError: If the log file is not valid JSON or has invalid structure.
    """
    if not log_path.exists():
        raise FileNotFoundError(
            f"Action log file not found: {log_path}. This scene was likely "
            "generated before action logging was added. Please re-run the experiment "
            "to generate a new scene with action logging enabled."
        )

    try:
        with open(log_path, encoding="utf-8") as f:
            log_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse action log JSON at {log_path}: {e}")

    if not isinstance(log_data, list):
        raise ValueError(
            f"Action log at {log_path} is not a list. Got type: {type(log_data)}"
        )

    # Parse entries.
    entries = []
    for i, entry_dict in enumerate(log_data):
        try:
            entry = ActionLogEntry(**entry_dict)
            entries.append(entry)
        except TypeError as e:
            raise ValueError(
                f"Invalid action log entry at index {i} in {log_path}: {e}"
            )

    return entries
