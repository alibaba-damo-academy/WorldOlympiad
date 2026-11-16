"""Main Objaverse retrieval logic with two-stage process: CLIP -> size ranking."""

import logging

from dataclasses import dataclass

import numpy as np
import trimesh

from scenecode.agent_utils.objaverse_retrieval.clip_similarity import (
    get_top_k_similar_meshes,
)
from scenecode.agent_utils.objaverse_retrieval.config import ObjaverseConfig
from scenecode.agent_utils.objaverse_retrieval.data_loader import (
    ObjaverseMeshMetadata,
    construct_objaverse_mesh_path,
    load_preprocessed_data,
)

console_logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """Candidate mesh for retrieval."""

    uid: str
    """Objaverse mesh UID."""

    mesh: trimesh.Trimesh
    """Loaded mesh."""

    metadata: ObjaverseMeshMetadata
    """Objaverse metadata."""

    clip_score: float
    """CLIP similarity score."""

    bbox_score: float
    """Bounding box size difference score (L1 distance)."""


class ObjaverseRetriever:
    """Objaverse (ObjectThor) asset retrieval system.

    Implements two-stage retrieval:
    1. CLIP semantic filtering (select top-K candidates)
    2. Size-based ranking (rank by dimension match)

    Unlike HSSD, Objaverse does not have pre-computed orientation metadata,
    so meshes are loaded directly without alignment transforms. Downstream
    VLM physics analysis will determine orientation during canonicalization.
    """

    def __init__(self, config: ObjaverseConfig, clip_device: str | None = None) -> None:
        """Initialize Objaverse retriever.

        Args:
            config: Objaverse configuration.
            clip_device: Target device for CLIP model (e.g., "cuda:0"). If None,
                uses default.
        """
        self.config = config
        self.clip_device = clip_device
        self.preprocessed_data = load_preprocessed_data(config.preprocessed_path)
        console_logger.info(
            f"Objaverse retriever initialized (clip_device={clip_device})"
        )

    def _calculate_bbox_score(
        self, target_dimensions: np.ndarray, mesh_extents: np.ndarray
    ) -> float:
        """Calculate orientation-invariant bounding box score.

        Since Objaverse meshes are not pre-canonicalized (orientation is determined
        by VLM after retrieval), we sort dimensions before comparing. This ensures
        a mesh stored as (0.5, 0.9, 0.5) matches a target of (0.9, 0.5, 0.5).

        This matches Holodeck's approach.

        Args:
            target_dimensions: Desired dimensions (3,).
            mesh_extents: Actual mesh extents (3,).

        Returns:
            L1 distance score (lower is better).
        """
        sorted_target = np.sort(target_dimensions)
        sorted_extents = np.sort(mesh_extents)
        return float(np.sum(np.abs(sorted_target - sorted_extents)))

    def _load_mesh(self, uid: str) -> trimesh.Trimesh:
        """Load mesh from Objaverse data directory.

        Unlike HSSD, Objaverse meshes are loaded directly without alignment
        transforms. VLM physics analysis handles orientation downstream.

        Args:
            uid: Objaverse mesh UID.

        Returns:
            Loaded mesh (Y-up GLB format).
        """
        mesh_path = construct_objaverse_mesh_path(
            data_path=self.config.data_path, uid=uid
        )

        mesh = trimesh.load(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded mesh is not a Trimesh: {type(mesh)}")

        return mesh

    def retrieve(
        self,
        description: str,
        object_type: str,
        desired_dimensions: np.ndarray | None = None,
    ) -> tuple[trimesh.Trimesh, str, float, ObjaverseMeshMetadata]:
        """Retrieve best matching Objaverse mesh for description.

        Two-stage process:
        1. CLIP semantic filtering -> top-K candidates
        2. Size-based ranking -> best dimension match

        Args:
            description: Object description text.
            object_type: Object type (e.g., "FURNITURE", "MANIPULAND").
            desired_dimensions: Optional desired dimensions (width, height, depth).

        Returns:
            Tuple of (mesh, uid, clip_score, metadata) where:
            - mesh: Best matching mesh in Y-up coordinates
            - uid: Objaverse UID identifying the mesh
            - clip_score: CLIP similarity score (0.0 to 1.0)
            - metadata: Objaverse mesh metadata

        Raises:
            ValueError: If no suitable mesh is found.
        """
        candidates = self.retrieve_multiple(
            description=description,
            object_type=object_type,
            desired_dimensions=desired_dimensions,
            max_candidates=1,
        )

        if not candidates:
            raise ValueError(
                f"No suitable mesh found for '{description}' (type={object_type})"
            )

        best = candidates[0]
        return best.mesh, best.uid, best.clip_score, best.metadata

    def retrieve_multiple(
        self,
        description: str,
        object_type: str,
        desired_dimensions: np.ndarray | None = None,
        max_candidates: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Retrieve multiple matching Objaverse meshes for description.

        Same two-stage process as retrieve(), but returns all candidates
        sorted by bbox_score instead of just the best one.

        Args:
            description: Object description text.
            object_type: Object type (e.g., "FURNITURE", "MANIPULAND").
            desired_dimensions: Optional desired dimensions (width, depth, height).
            max_candidates: Maximum candidates to return. If None, returns all
                available (up to use_top_k CLIP candidates).

        Returns:
            List of RetrievalCandidate sorted by bbox_score (best first).
            Empty list if no suitable meshes found.
        """
        console_logger.info(
            f"Retrieving multiple Objaverse meshes: description='{description}', "
            f"type={object_type}, dimensions={desired_dimensions}"
        )

        category = self.config.object_type_mapping.get(object_type.upper())
        if category is None:
            console_logger.warning(
                f"Unknown object type: {object_type}. "
                f"Available: {list(self.config.object_type_mapping.keys())}"
            )
            return []

        top_k_meshes = get_top_k_similar_meshes(
            text_description=description,
            preprocessed_data=self.preprocessed_data,
            category=category,
            top_k=self.config.use_top_k,
            device=self.clip_device,
        )

        if not top_k_meshes:
            console_logger.warning(f"No meshes found for category: {category}")
            return []

        console_logger.info(f"Processing {len(top_k_meshes)} CLIP-filtered candidates")

        candidates: list[RetrievalCandidate] = []

        for uid, clip_score in top_k_meshes:
            metadata = self.preprocessed_data.get_metadata(uid)
            if metadata is None:
                console_logger.warning(f"Metadata not found for mesh {uid}")
                continue

            try:
                mesh = self._load_mesh(uid)
            except Exception as e:
                console_logger.warning(f"Failed to load mesh {uid}: {e}", exc_info=True)
                continue

            if desired_dimensions is not None:
                mesh_extents = mesh.extents
                bbox_score = self._calculate_bbox_score(
                    target_dimensions=desired_dimensions, mesh_extents=mesh_extents
                )
            else:
                bbox_score = 0.0

            candidate = RetrievalCandidate(
                uid=uid,
                mesh=mesh,
                metadata=metadata,
                clip_score=clip_score,
                bbox_score=bbox_score,
            )
            candidates.append(candidate)

            console_logger.debug(
                f"Candidate {uid[:8]}: CLIP={clip_score:.3f}, "
                f"bbox={bbox_score:.3f}, extents={mesh.extents}"
            )

        if not candidates:
            console_logger.warning("No valid candidates found after mesh loading")
            return []

        # Sort by bbox_score (lower is better).
        candidates.sort(key=lambda c: c.bbox_score)

        # Limit results if requested.
        if max_candidates is not None and len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        console_logger.info(
            f"Returning {len(candidates)} candidates (sorted by bbox_score)"
        )

        return candidates
