#!/usr/bin/env python3
"""Unified WorldOlympiad 2.0 entrypoint.

This keeps the historical score_video_physical_3d.py command working while
exposing Physical, Geometry, Interaction, and optional Continuous Action under
a name that no longer implies only three tracks.
"""

from score_video_physical_3d import main


if __name__ == "__main__":
    main()
