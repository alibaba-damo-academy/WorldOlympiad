"""Predicate resolver for policy interface.

Uses the unified PolicyInterfaceAgent to resolve tasks to object bindings,
then computes exact poses for robot execution.
"""

import asyncio
import logging

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from omegaconf import DictConfig
from pydantic import BaseModel, Field
from pydrake.all import Quaternion
from pydrake.math import RigidTransform

from scenecode.robot_eval.dmd_scene import DMDScene
from scenecode.robot_eval.policy_interface.policy_agent import (
    PolicyInterfaceAgent,
    PolicyInterfaceOutput,
)

if TYPE_CHECKING:
    from scenecode.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


class ExactPosePredicate(BaseModel):
    """Exact pose predicate for robot execution.

    Contains both a sampled target pose and a valid placement region (AABB).
    The placement region is shrunk by the target object's half-extents,
    so any point sampled from it is guaranteed to be valid.
    """

    model_config = {"arbitrary_types_allowed": True}

    action: str = Field(description="Action type: 'pick_and_place'")
    drake_model_name: str = Field(description="Drake model instance name to manipulate")
    target_position: list[float] = Field(description="Target [x, y, z] in world frame")
    target_rotation_wxyz: list[float] = Field(
        default=[1.0, 0.0, 0.0, 0.0],
        description="Target orientation as quaternion [w, x, y, z]",
    )

    # Valid placement region (shrunk by target half-extents).
    placement_bounds_min: list[float] | None = Field(default=None)
    placement_bounds_max: list[float] | None = Field(default=None)

    # Traceability.
    source_predicate: str = Field(default="", description="e.g., 'on', 'inside'")
    reference_id: str = Field(default="", description="Reference object ID")

    # Agent ranking metadata.
    rank: int = Field(default=1, description="Agent ranking (1 = best)")
    confidence: float = Field(default=1.0, description="Agent confidence (0.0-1.0)")
    reasoning: str = Field(default="", description="Agent reasoning for this binding")

    def to_rigid_transform(self) -> RigidTransform:
        """Convert to Drake RigidTransform."""
        return RigidTransform(
            Quaternion(*self.target_rotation_wxyz), self.target_position
        )


class ResolverResult(BaseModel):
    """Result of predicate resolution with ALL valid candidates."""

    poses: list[ExactPosePredicate] = Field(
        description="All resolved poses, ranked by agent preference"
    )
    agent_output: PolicyInterfaceOutput | None = Field(
        default=None, description="Full agent output"
    )
    success: bool = Field(description="Whether at least one pose was resolved")
    notes: list[str] = Field(default_factory=list, description="Resolution notes")


@dataclass
class PredicateResolver:
    """Resolves tasks to exact poses for robot execution.

    Uses the unified PolicyInterfaceAgent to:
    1. Parse task → goal predicates + preconditions
    2. Find objects matching categories in scene
    3. Verify preconditions using state/vision tools
    4. Return ranked valid (target, reference) bindings

    Then computes poses for each valid binding.
    """

    scene: DMDScene
    cfg: DictConfig
    robot_position: np.ndarray | None = None
    blender_server: "BlenderServer | None" = None

    _agent: PolicyInterfaceAgent | None = field(default=None, init=False)

    @property
    def agent(self) -> PolicyInterfaceAgent:
        """Lazily create policy interface agent."""
        if self._agent is None:
            self._agent = PolicyInterfaceAgent(
                scene=self.scene,
                cfg=self.cfg,
                blender_server=self.blender_server,
            )
        return self._agent

    async def resolve_async(self, task_description: str) -> ResolverResult:
        """Resolve task to exact poses using unified agent.

        Args:
            task_description: Natural language task (e.g., "Pick a cup from the floor
                and put it in the sink").

        Returns:
            ResolverResult with all valid poses ranked by agent preference.
        """
        poses: list[ExactPosePredicate] = []
        notes: list[str] = []

        # Run unified policy interface agent.
        agent_output = await self.agent.resolve(task_description=task_description)

        if not agent_output.overall_success:
            notes.append("Agent found no valid candidates")
            notes.extend(agent_output.notes)
            return ResolverResult(
                poses=[], agent_output=agent_output, success=False, notes=notes
            )

        # Compute poses for all valid bindings.
        for binding in agent_output.valid_bindings:
            pose = self._compute_pose(
                predicate_type=agent_output.goal_predicate,
                target_id=binding.target_id,
                ref_id=binding.reference_id,
            )
            if pose:
                pose.rank = binding.rank
                pose.confidence = binding.confidence
                pose.reasoning = binding.reasoning
                poses.append(pose)
            else:
                notes.append(
                    f"Failed to compute pose for {binding.target_id} -> "
                    f"{binding.reference_id}"
                )

        poses.sort(key=lambda p: p.rank)

        return ResolverResult(
            poses=poses, agent_output=agent_output, success=len(poses) > 0, notes=notes
        )

    def resolve(self, task_description: str) -> ResolverResult:
        """Resolve task to exact poses (sync wrapper)."""
        return asyncio.run(self.resolve_async(task_description=task_description))

    def _get_object_bbox(
        self, obj_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Get object position and bounding box.

        Returns:
            Tuple of (position, bbox_min, bbox_max) or None if object not found.
        """
        obj = self.scene.scene_state["objects"].get(obj_id)
        if obj is None:
            return None
        pos = np.array(obj["transform"]["translation"])
        bbox_min = np.array(obj.get("bbox_min", [0, 0, 0]))
        bbox_max = np.array(obj.get("bbox_max", [0, 0, 0]))
        return pos, bbox_min, bbox_max

    def _resolve_drake_model_name(self, obj_id: str) -> str:
        """Resolve object ID to drake model name.

        For regular objects, the object_id IS the drake model name.
        For composites (stack/pile/filled_container), returns the appropriate
        member's drake_model_name (topmost for stacks, first fill for containers).

        Args:
            obj_id: Object ID from scene_state.

        Returns:
            Drake model name that can be used with plant.GetModelInstanceByName().
        """
        obj_data = self.scene.scene_state["objects"].get(obj_id)
        if obj_data is None:
            return obj_id

        metadata = obj_data.get("metadata", {})
        composite_type = metadata.get("composite_type")

        if composite_type not in ("stack", "pile", "filled_container"):
            return obj_id

        member_names = metadata.get("member_model_names", [])
        if not member_names:
            console_logger.warning(
                f"Composite {obj_id} has no member_model_names, using obj_id as fallback"
            )
            return obj_id

        # For stacks/piles: return topmost item (last in list) for pick operations.
        # For filled_container: return first non-container member.
        if composite_type in ("stack", "pile"):
            drake_name = member_names[-1]
        else:
            drake_name = next(
                (name for name in member_names if not name.endswith("_c")),
                member_names[0],
            )

        console_logger.info(
            f"Resolved composite {obj_id} ({composite_type}) -> {drake_name}"
        )
        return drake_name

    def _compute_pose(
        self, predicate_type: str, target_id: str, ref_id: str
    ) -> ExactPosePredicate | None:
        """Compute exact pose for predicate."""
        if predicate_type == "on":
            return self._compute_on_pose(target_id=target_id, ref_id=ref_id)
        elif predicate_type == "inside":
            return self._compute_inside_pose(target_id=target_id, ref_id=ref_id)
        elif predicate_type == "near":
            return self._compute_near_pose(target_id=target_id, ref_id=ref_id)
        else:
            console_logger.warning(f"Unknown predicate type: {predicate_type}")
            return None

    def _compute_on_pose(
        self, target_id: str, ref_id: str, z_margin: float = 0.05
    ) -> ExactPosePredicate | None:
        """Compute pose for 'on' predicate (place target on reference surface).

        Args:
            target_id: Object to place.
            ref_id: Surface to place on.
            z_margin: Extra height margin above max_target_extent (default 5cm).
        """
        ref_bbox = self._get_object_bbox(ref_id)
        target_bbox = self._get_object_bbox(target_id)
        if ref_bbox is None or target_bbox is None:
            return None

        ref_pos, ref_min, ref_max = ref_bbox
        _, target_min, target_max = target_bbox
        target_half = (target_max - target_min) / 2.0

        surface_z = ref_pos[2] + ref_max[2]
        target_z = surface_z - target_min[2]
        max_target_extent = float(np.max(target_max - target_min))

        # Contained bounds (shrunk by target half-extents).
        raw_min = ref_pos + ref_min
        raw_max = ref_pos + ref_max
        contained_min = [
            raw_min[0] + target_half[0],
            raw_min[1] + target_half[1],
            surface_z,
        ]
        contained_max = [
            raw_max[0] - target_half[0],
            raw_max[1] - target_half[1],
            surface_z + max_target_extent + z_margin,
        ]

        bounds_min, bounds_max = None, None
        if (
            contained_min[0] <= contained_max[0]
            and contained_min[1] <= contained_max[1]
        ):
            bounds_min, bounds_max = contained_min, contained_max

        return ExactPosePredicate(
            action="pick_and_place",
            drake_model_name=self._resolve_drake_model_name(target_id),
            target_position=[ref_pos[0], ref_pos[1], target_z],
            placement_bounds_min=bounds_min,
            placement_bounds_max=bounds_max,
            source_predicate="on",
            reference_id=ref_id,
        )

    def _compute_inside_pose(
        self, target_id: str, ref_id: str
    ) -> ExactPosePredicate | None:
        """Compute pose for 'inside' predicate (place target inside container)."""
        ref_bbox = self._get_object_bbox(ref_id)
        target_bbox = self._get_object_bbox(target_id)
        if ref_bbox is None or target_bbox is None:
            return None

        ref_pos, ref_min, ref_max = ref_bbox
        _, target_min, target_max = target_bbox
        target_half = (target_max - target_min) / 2.0

        container_min = ref_pos + ref_min
        container_max = ref_pos + ref_max
        target_z = container_min[2] - target_min[2]

        contained_min = [
            container_min[0] + target_half[0],
            container_min[1] + target_half[1],
            container_min[2],
        ]
        contained_max = [
            container_max[0] - target_half[0],
            container_max[1] - target_half[1],
            container_max[2],
        ]

        bounds_min, bounds_max = None, None
        if all(contained_min[i] <= contained_max[i] for i in range(3)):
            bounds_min, bounds_max = contained_min, contained_max

        return ExactPosePredicate(
            action="pick_and_place",
            drake_model_name=self._resolve_drake_model_name(target_id),
            target_position=[ref_pos[0], ref_pos[1], target_z],
            placement_bounds_min=bounds_min,
            placement_bounds_max=bounds_max,
            source_predicate="inside",
            reference_id=ref_id,
        )

    def _compute_near_pose(
        self, target_id: str, ref_id: str, distance: float = 0.3
    ) -> ExactPosePredicate | None:
        """Compute pose for 'near' predicate (place target near reference)."""
        ref_bbox = self._get_object_bbox(ref_id)
        target_bbox = self._get_object_bbox(target_id)
        if ref_bbox is None or target_bbox is None:
            return None

        ref_pos, ref_min, ref_max = ref_bbox
        _, target_min, target_max = target_bbox
        target_half = (target_max - target_min) / 2.0
        max_target_extent = float(np.max(target_max - target_min))

        ref_world_min = ref_pos + ref_min
        ref_world_max = ref_pos + ref_max

        contained_min = [
            ref_world_min[0] - distance - target_half[0],
            ref_world_min[1] - distance - target_half[1],
            0.0,
        ]
        contained_max = [
            ref_world_max[0] + distance + target_half[0],
            ref_world_max[1] + distance + target_half[1],
            ref_pos[2] + max_target_extent,
        ]

        target_position = [
            (contained_min[0] + contained_max[0]) / 2,
            (contained_min[1] + contained_max[1]) / 2,
            ref_pos[2],
        ]

        bounds_min, bounds_max = None, None
        if (
            contained_min[0] <= contained_max[0]
            and contained_min[1] <= contained_max[1]
        ):
            bounds_min, bounds_max = contained_min, contained_max

        return ExactPosePredicate(
            action="pick_and_place",
            drake_model_name=self._resolve_drake_model_name(target_id),
            target_position=target_position,
            placement_bounds_min=bounds_min,
            placement_bounds_max=bounds_max,
            source_predicate="near",
            reference_id=ref_id,
        )
