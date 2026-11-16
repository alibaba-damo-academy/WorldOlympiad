"""Mock utilities for unit tests."""

from pathlib import Path
from unittest.mock import Mock

from scenecode.utils.logging import BaseLogger


def create_mock_logger(output_dir: Path) -> Mock:
    """
    Create a properly configured mock logger for unit tests.

    Args:
        output_dir: The output directory for the logger.

    Returns:
        Mock logger with spec=BaseLogger and output_dir set.
    """
    mock_logger = Mock(spec=BaseLogger)
    mock_logger.output_dir = output_dir
    return mock_logger
