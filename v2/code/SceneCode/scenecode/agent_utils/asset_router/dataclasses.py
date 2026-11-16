"""Dataclasses for the asset router module."""

from dataclasses import dataclass, field
from pathlib import Path

from scenecode.agent_utils.room import ObjectType


@dataclass
class AssetItem:
    """Single asset to generate (from LLM analysis)."""

    description: str
    """The asset description to use for generation."""

    short_name: str
    """Short name for the asset (lowercase_with_underscores)."""

    dimensions: list[float]
    """Dimensions [width, depth, height] in meters."""

    object_type: ObjectType
    """Type: FURNITURE, MANIPULAND, or EITHER."""

    strategies: list[str]
    """Strategy chain to try, e.g. ["articulated", "code_generated"]."""

    thin_covering_type: str | None = None
    """Type of thin covering texture: "tileable" or "single_image".
    Only set when strategies includes "thin_covering".
    - "tileable": Pattern repeats across surface (rugs, carpets).
    - "single_image": One image spans entire surface (posters, paintings)."""

    code_object_profile: str | None = None
    """Optional Code_Object specialization profile for code-generated assets.

    Examples: "wall_art" for framed prints, posters, murals, and other artwork-like
    wall assets that need dedicated prompts and material preprocessing;
    "SimpleObject" and "manipuland" for other Code_Object prompt specializations.
    """


@dataclass
class AnalysisResult:
    """LLM analysis output from request analysis."""

    items: list[AssetItem]
    """List of assets to generate."""

    original_description: str | None
    """Set if the request was modified (split or filtered)."""

    discarded_manipulands: list[str] | None
    """Manipulands filtered out by furniture agent."""

    error: str | None = None
    """Set if the request was rejected."""

    @property
    def was_modified(self) -> bool:
        """True if original request was changed (split or filtered)."""
        return (
            self.original_description is not None
            or self.discarded_manipulands is not None
        )


@dataclass
class ModificationInfo:
    """Feedback to designer when request was modified."""

    original_description: str
    """What was originally requested."""

    resulting_descriptions: list[str]
    """What was actually generated."""

    discarded_manipulands: list[str] | None = None
    """What manipulands were filtered out (furniture agent only)."""


@dataclass
class ValidationResult:
    """Output from VLM validation of generated asset."""

    is_acceptable: bool
    """Whether the asset passes validation."""

    reason: str
    """Explanation for the decision (logged for debugging)."""

    suggestions: list[str] = field(default_factory=list)
    """Suggestions if rejected (what to try differently)."""


@dataclass
class GeneratedGeometry:
    """Result of validated geometry generation or retrieval."""

    geometry_path: Path
    """Path to the validated geometry file (GLB/GLTF)."""

    item: AssetItem
    """The item that was generated/retrieved."""

    asset_source: str
    """Source strategy that produced this geometry (e.g., 'code_generated',
    'generated', 'hssd', 'articulated', 'thin_covering'). Used to track provenance
    in asset metadata."""

    image_path: Path | None = None
    """Path to the reference image (for generated assets, None for HSSD)."""

    hssd_id: str | None = None
    """HSSD object ID (for HSSD assets, None for others)."""

    objaverse_uid: str | None = None
    """Objaverse/ObjectThor unique identifier (for objaverse assets, None for others)."""

    code_object_output_dir: Path | None = None
    """Raw Code_Object output directory for code_generated assets."""

    object_plan_path: Path | None = None
    """Code_Object ObjectPlan path for code_generated assets."""

    code_dir: Path | None = None
    """Code_Object generated code directory for code_generated assets."""

    pipeline_result_path: Path | None = None
    """Code_Object pipeline_result.json path for code_generated assets."""

    full_object_render_path: Path | None = None
    """Code_Object full object render path for code_generated assets."""


@dataclass
class CodeArticulatedGeometry:
    """Generated articulated asset before URDF->SDF runtime conversion."""

    urdf_path: Path
    """Path to the generated URDF file."""

    item: AssetItem
    """The item that was generated."""

    image_path: Path | None = None
    """Path to the reference image used for Code_Object generation."""

    geometry_path: Path | None = None
    """Path to the rigid mesh export from the base Code_Object pipeline."""

    code_object_output_dir: Path | None = None
    """Raw Code_Object output directory for this generated articulated asset."""

    object_plan_path: Path | None = None
    """Code_Object ObjectPlan path."""

    code_dir: Path | None = None
    """Code_Object generated code directory."""

    pipeline_result_path: Path | None = None
    """Code_Object pipeline_result.json path."""

    full_object_render_path: Path | None = None
    """Code_Object full object render path."""

    sdf_path: Path | None = None
    """Packaged SDF path when router precomputes articulated conversion."""

    analysis_path: Path | None = None
    """Analysis JSON path when router precomputes articulated conversion."""

    validation_mesh_path: Path | None = None
    """Packaged mesh path used for articulated validation."""


@dataclass
class ArticulatedGeometry:
    """Result of articulated object retrieval or generated articulated conversion.

    Unlike GeneratedGeometry which contains a single mesh, articulated objects
    have multi-link SDF files with joints (doors, drawers, etc.).
    """

    sdf_path: Path
    """Path to the articulated SDF file."""

    item: AssetItem
    """The item that was retrieved or generated."""

    source: str
    """Underlying articulated source, e.g. dataset name or 'code_articulated'."""

    object_id: str
    """Object ID or generated asset ID for provenance tracking."""

    bounding_box_min: list[float]
    """Bounding box minimum [x, y, z] at default pose (joints=0)."""

    bounding_box_max: list[float]
    """Bounding box maximum [x, y, z] at default pose (joints=0)."""

    asset_source: str = "articulated"
    """Top-level asset source used in scene metadata."""

    image_path: Path | None = None
    """Reference image path for generated articulated assets."""

    code_object_output_dir: Path | None = None
    """Raw Code_Object output directory for generated articulated assets."""

    object_plan_path: Path | None = None
    """Code_Object ObjectPlan path for generated articulated assets."""

    code_dir: Path | None = None
    """Code_Object generated code directory for generated articulated assets."""

    pipeline_result_path: Path | None = None
    """Code_Object pipeline_result.json path for generated articulated assets."""

    full_object_render_path: Path | None = None
    """Code_Object full object render path for generated articulated assets."""

    urdf_path: Path | None = None
    """Generated URDF path for generated articulated assets."""

    analysis_path: Path | None = None
    """Analysis JSON path for generated articulated assets."""
