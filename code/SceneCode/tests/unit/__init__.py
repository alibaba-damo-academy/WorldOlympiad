# Import bpy first to avoid potential symbol loading issues.
import sys

try:
    import bpy  # noqa: F401
except ImportError as e:
    # If real bpy fails to import, mock it.
    from unittest.mock import MagicMock

    sys.modules["bpy"] = MagicMock()

    print(f"Mocking bpy due to import error: {e}")
