import json
import logging
import pickle
import tempfile
import time

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import lxml.etree as ET
import numpy as np

from PIL import Image

from scenecode.agent_utils.room import RoomScene

console_logger = logging.getLogger(__name__)


class RoomContext:
    """Context manager for room-specific logging paths and log files.

    Temporarily changes the logger's output directory to a room-specific
    subdirectory and creates a dedicated room.log file for per-room logging.
    All logging operations within the context will be routed to the room's
    directory, and logs will be captured to room.log.
    """

    def __init__(self, logger: "ConsoleLogger", room_id: str):
        """Initialize room context.

        Args:
            logger: The ConsoleLogger instance to modify.
            room_id: Room identifier for directory naming.
        """
        self.logger = logger
        self.room_id = room_id
        self.original_output_dir: Path | None = None
        self.room_dir: Path | None = None
        self.file_handler: logging.FileHandler | None = None

    def __enter__(self) -> Path:
        """Enter room context and return room directory path.

        Creates the room directory, sets up room.log file handler, and
        redirects logger output.

        Returns:
            Path to the room-specific directory.
        """
        self.original_output_dir = self.logger.output_dir
        self.room_dir = self.original_output_dir / f"room_{self.room_id}"
        self.room_dir.mkdir(parents=True, exist_ok=True)
        self.logger.output_dir = self.room_dir

        # Create file handler for room-specific logging.
        room_log_path = self.room_dir / "room.log"
        self.file_handler = logging.FileHandler(room_log_path)
        self.file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # Add handler to root logger to capture all logs.
        root_logger = logging.getLogger()
        root_logger.addHandler(self.file_handler)

        return self.room_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit room context, restore output directory, and clean up file handler."""
        if self.original_output_dir is not None:
            self.logger.output_dir = self.original_output_dir

        # Remove and close the room file handler.
        if self.file_handler:
            root_logger = logging.getLogger()
            if self.file_handler in root_logger.handlers:
                root_logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self.file_handler = None


class BaseLogger(ABC):
    """Abstract base class defining the logger API for experiment tracking."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)

    @abstractmethod
    def log(self, data: dict[str, Any]) -> None:
        """Log metrics and data."""

    @abstractmethod
    def log_hyperparams(self, data: dict[str, Any]) -> None:
        """Log hyperparameters."""

    @abstractmethod
    def log_pickle(self, name: str, obj: Any, use_temp_file: bool = True) -> None:
        """
        Log a pickle file.

        Args:
            name: The name of the pickle file.
            obj: The object to be pickled.
            use_temp_file: Whether to use a temporary file. Otherwise, `name` is
                saved relative to `output_dir`.
        """

    @abstractmethod
    def log_sdf(
        self, name: str, sdf_tree: ET.ElementTree, output_dir: Path | None = None
    ) -> Path:
        """
        Log an SDF file.

        Args:
            name: The name of the SDF file.
            sdf_tree: The SDF ElementTree object to be saved.
            output_dir: Optional output directory. If provided, saves to this directory.
                Otherwise, saves to logger's default output directory.

        Returns:
            Path to the saved SDF file.
        """

    @abstractmethod
    def log_images_to_dir(self, images: list[np.ndarray], dir: Path | str) -> None:
        """
        Log images to a directory.

        Args:
            images: The images to be logged.
            dir: The directory to save the images to. Will be created if it does not
                exist.
        """

    @abstractmethod
    def log_scene(
        self, scene: RoomScene, name: str | None = None, output_dir: Path | None = None
    ) -> Path:
        """
        Log scene state including objects metadata and Drake directive.

        Args:
            scene: The Scene object to log.
            name: Name for the scene state snapshot. Creates subdirectory in
                scene_states/. Required if output_dir not provided.
            output_dir: Custom directory to save scene state. If provided,
                takes precedence over name.

        Returns:
            Path to the directory containing scene files.

        Raises:
            ValueError: If neither name nor output_dir is provided.
        """

    @abstractmethod
    def room_context(self, room_id: str) -> "RoomContext":
        """Create a context manager for room-specific logging.

        All logging operations within the context will be routed to
        `output_dir/room_{room_id}/`.

        Args:
            room_id: Room identifier for directory naming.

        Returns:
            Context manager that temporarily modifies output paths.
        """


class ConsoleLogger(BaseLogger):
    """Logger implementation that logs to console and saves files locally."""

    def __init__(self, output_dir: Path | str):
        super().__init__(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._step_counter = 0
        """Counter for the number of steps logged."""

    def room_context(self, room_id: str) -> RoomContext:
        """Create a context manager for room-specific logging.

        All logging operations within the context will be routed to
        `output_dir/room_{room_id}/`. A dedicated `room.log` file will be
        created to capture all logs within the context.

        Args:
            room_id: Room identifier for directory naming.

        Returns:
            RoomContext that temporarily modifies output paths and creates
            a per-room log file.

        Example:
            with logger.room_context("living_room") as room_dir:
                logger.log_scene(scene)  # Saved to room_living_room/
                # Logs captured to room_living_room/room.log
        """
        return RoomContext(self, room_id)

    def log(self, data: dict[str, Any]) -> None:
        """Log metrics to console."""
        console_logger.info(f"Step {self._step_counter}: {data}")
        self._step_counter += 1

    def log_hyperparams(self, data: dict[str, Any]) -> None:
        """Log hyperparameters to console."""
        console_logger.info(f"Hyperparameters: {data}")

    def log_pickle(self, name: str, obj: Any, use_temp_file: bool = True) -> None:
        """
        Log a pickle file to local filesystem.

        Args:
            name: The name of the pickle file.
            obj: The object to be pickled.
            use_temp_file: Whether to use a temporary file. Otherwise, `name` is
                saved relative to `output_dir`.
        """
        if use_temp_file:
            with tempfile.NamedTemporaryFile(
                "wb",
                prefix=f"{name}_{self._step_counter}__",
                suffix=".pkl",
                delete=False,
            ) as temp_file:
                pickle.dump(obj, temp_file)
                file_path = temp_file.name
        else:
            if not name.endswith(".pkl"):
                console_logger.warning(
                    f"Name {name} does not end with '.pkl'. Appending '.pkl'."
                )
                name += ".pkl"
            file_path = str(self.output_dir / name)
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)

        console_logger.info(f"Saved pickle file: {file_path}")

    def log_sdf(
        self, name: str, sdf_tree: ET.ElementTree, output_dir: Path | None = None
    ) -> Path:
        """
        Log an SDF file to local filesystem.

        Args:
            name: The name of the SDF file.
            sdf_tree: The SDF ElementTree object to be saved.
            output_dir: Optional output directory. If provided, saves to this directory.
                Otherwise, saves to logger's default output directory.

        Returns:
            Path to the saved SDF file.
        """
        # Determine output directory.
        save_dir = output_dir if output_dir is not None else self.output_dir

        # Ensure .sdf extension.
        if not name.endswith(".sdf"):
            name += ".sdf"

        # Create directory and file path.
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / name

        # Write SDF content.
        with open(file_path, "w") as f:
            sdf_content = ET.tostring(
                sdf_tree.getroot(), encoding="unicode", pretty_print=True
            )
            f.write(sdf_content)

        console_logger.info(f"Saved SDF file: {file_path}")
        return file_path

    def log_images_to_dir(self, images: list[np.ndarray], dir: Path | str) -> None:
        """
        Log images to a directory.

        Args:
            images: The images to be logged as numpy arrays with shape (H, W, C)
                   or (H, W) for grayscale. Values should be in range [0, 255] as uint8.
            dir: The directory to save the images to. Will be created if it does not
                exist.
        """
        dir_path = Path(dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images):
            # Convert numpy array to PIL Image.
            if image.dtype != np.uint8:
                console_logger.warning(
                    f"Image {i} is not uint8. Converting from {image.dtype}."
                )
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)
            image_path = dir_path / f"image_{self._step_counter}_{i:04d}.png"
            pil_image.save(image_path)

            console_logger.info(f"Saved image: {image_path}")

    def log_scene(
        self, scene: RoomScene, name: str | None = None, output_dir: Path | None = None
    ) -> Path:
        """
        Log scene state including objects metadata and Drake directive.

        Args:
            scene: The Scene object to log.
            name: Name for the scene state snapshot. Creates subdirectory in
                scene_states/. Required if output_dir not provided.
            output_dir: Custom directory to save scene state. If provided,
                takes precedence over name.

        Returns:
            Path to the directory containing scene files.

        Raises:
            ValueError: If neither name nor output_dir is provided.
        """
        if name is None and output_dir is None:
            raise ValueError("Must provide either 'name' or 'output_dir'")

        # Determine output directory.
        save_dir = output_dir if output_dir is not None else self.output_dir
        scene_dir = save_dir / "scene_states" / name if name is not None else save_dir

        scene_dir.mkdir(parents=True, exist_ok=True)

        # Get scene state and add timestamp.
        state_data = scene.to_state_dict()
        state_data["timestamp"] = time.time()

        with open(scene_dir / "scene_state.json", "w") as f:
            json.dump(state_data, f, indent=2)

        # Save Drake directive with absolute paths for local debugging.
        # For portable scenes, use HouseScene.assemble() which creates
        # package://scene/ URIs with package.xml at scene root.
        directive_content = scene.to_drake_directive()
        with open(scene_dir / "scene.dmd.yaml", "w") as f:
            f.write(directive_content)

        console_logger.info(f"Saved scene state: {scene_dir}")
        return scene_dir


class FileLoggingContext:
    """Context manager to redirect all loggers to scene-specific log files.

    This class captures ALL logging that occurs within its context.
    """

    def __init__(self, log_file_path: Path, suppress_stdout: bool = False):
        """
        Args:
            log_file_path: Path to the scene-specific log file
            suppress_stdout: If True, prevents logs from also going to stdout
        """
        self.log_file_path = log_file_path
        self.suppress_stdout = suppress_stdout
        self.file_handler = None
        self.original_handlers = []

    def __enter__(self):
        """Set up file handler and redirect all loggers."""
        # Create file handler with consistent formatting.
        self.file_handler = logging.FileHandler(self.log_file_path)
        self.file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # Get root logger to capture everything.
        root_logger = logging.getLogger()

        # Add our file handler.
        root_logger.addHandler(self.file_handler)

        if self.suppress_stdout:
            # Save original handlers and remove console handlers temporarily.
            self.original_handlers = root_logger.handlers[:]
            for handler in self.original_handlers:
                if handler != self.file_handler:
                    root_logger.removeHandler(handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up handlers and restore original state."""
        root_logger = logging.getLogger()

        # Remove our file handler.
        if self.file_handler in root_logger.handlers:
            root_logger.removeHandler(self.file_handler)

        if self.suppress_stdout and self.original_handlers:
            # Restore original handlers.
            for handler in self.original_handlers:
                if handler not in root_logger.handlers:
                    root_logger.addHandler(handler)

        # Close the file handler.
        if self.file_handler:
            self.file_handler.close()
