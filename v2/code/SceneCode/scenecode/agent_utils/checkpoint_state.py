"""Checkpoint state management for scene rollback and iteration tracking.

This module provides data structures and initialization utilities for managing
checkpoint state across scene design iterations. Maintains N-1 and N checkpoints
to enable resetting to previous states when design changes degrade quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scenecode.agent_utils.scoring import CritiqueWithScores


@dataclass
class CheckpointState:
    """Checkpoint state structure for placement agents.

    Maintains two checkpoints (N-1 and N) to enable resetting to the state
    before degradation occurred.
    """

    # Scene state checkpoints.
    previous_scene_checkpoint: dict[str, Any] | None = None
    scene_checkpoint: dict[str, Any] | None = None

    # Score tracking for checkpoints.
    previous_checkpoint_scores: CritiqueWithScores | None = None
    checkpoint_scores: CritiqueWithScores | None = None
    previous_scores: CritiqueWithScores | None = None

    # Render directory tracking for checkpoints.
    previous_checkpoint_render_dir: Path | None = None
    checkpoint_render_dir: Path | None = None

    # Placement style (default: natural).
    placement_style: str = "natural"


def initialize_checkpoint_attributes(target: Any) -> None:
    """Initialize checkpoint state attributes on a target object.

    This function initializes all checkpoint-related attributes directly on the
    target object (typically self in an __init__ method). This approach maintains
    backward compatibility with existing code that accesses checkpoint state as
    direct attributes.

    Args:
        target: Object to initialize checkpoint attributes on (typically self).
    """
    # Scene checkpoint state for rollback functionality.
    # We maintain two checkpoints (N-1 and N) to enable resetting to
    # the state before degradation occurred.
    target.previous_scene_checkpoint: dict[str, Any] | None = None
    target.scene_checkpoint: dict[str, Any] | None = None
    target.previous_checkpoint_scores: CritiqueWithScores | None = None
    target.checkpoint_scores: CritiqueWithScores | None = None
    target.previous_scores: CritiqueWithScores | None = None

    # Track render directories for checkpoints.
    target.previous_checkpoint_render_dir: Path | None = None
    target.checkpoint_render_dir: Path | None = None

    # Track the final render directory separately from checkpoint tracking.
    # This is needed because final critique uses update_checkpoint=False to preserve
    # N-1 for reset comparison, but we still need to copy from the actual final render.
    target.final_render_dir: Path | None = None

    # Hash of scene state at last checkpoint for detecting redundant final critiques.
    # If scene is unchanged since last critique, skip final critique to save cost.
    target.checkpoint_scene_hash: int | None = None

    # Placement style tracking (default: natural).
    target.placement_style: str = "natural"
