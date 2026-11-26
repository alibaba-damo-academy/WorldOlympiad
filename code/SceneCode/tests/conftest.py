"""Pytest configuration and fixtures for scenecode tests.

This module provides pytest hooks and fixtures that apply across all tests.
"""

# isort: off
# Import bpy first to avoid OpenGL context conflicts with Drake rendering.
# When bpy is imported after Drake initializes its rendering context, there's a
# segfault due to conflicting OpenGL contexts. Importing bpy first ensures bpy
# initializes its context before Drake, avoiding the conflict.
import bpy  # noqa: F401

# isort: on

import gc
import logging

import pytest

console_logger = logging.getLogger(__name__)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """Force garbage collection after each test to clean up Drake C++ objects."""
    gc.collect()
    console_logger.debug(f"Garbage collection completed after test: {item.nodeid}")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Force final garbage collection after all tests complete.

    This ensures Drake C++ objects are cleaned up after all tests complete but
    before pytest exits, preventing hangs during Drake's leak detector cleanup.
    """
    del session, exitstatus  # Unused but required by hookspec.
    gc.collect()
    console_logger.debug("Final garbage collection completed after test session")
