"""Dataclasses for convex decomposition server request/response types.

This module defines typed dataclasses for the convex decomposition server API.
Currently these are not actively used (the server uses raw JSON dicts), but
they serve as documentation for the API schema and could be used for type-safe
client implementations in the future.
"""

from dataclasses import dataclass, field


@dataclass
class ConvexDecompositionRequest:
    """Request parameters for convex decomposition collision geometry generation.

    The server supports two decomposition methods:
    - "coacd": CoACD (faster, simpler geometry, good for small objects)
    - "vhacd": V-HACD (better shape preservation, good for large objects)
    """

    mesh_path: str
    method: str = "coacd"

    # CoACD parameters (used when method="coacd").
    threshold: float = 0.05
    max_convex_hull: int = -1
    preprocess_mode: str = "auto"
    preprocess_resolution: int = 50
    resolution: int = 2000
    mcts_nodes: int = 20
    mcts_iterations: int = 150
    mcts_max_depth: int = 3
    pca: bool = False
    merge: bool = True
    decimate: bool = False
    max_ch_vertex: int = 256
    extrude: bool = False
    extrude_margin: float = 0.01
    apx_mode: str = "ch"
    seed: int = 0

    # V-HACD parameters (used when method="vhacd").
    max_convex_hulls: int = 64
    vhacd_resolution: int = 400000
    max_recursion_depth: int = 10
    max_num_vertices_per_ch: int = 64
    min_volume_percent_error: float = 1.0
    shrink_wrap: bool = True
    fill_mode: str = "flood"
    min_edge_length: int = 2
    find_best_plane: bool = False


@dataclass
class ConvexPiece:
    """A single convex piece from the decomposition."""

    vertices: list[list[float]]
    faces: list[list[int]]


@dataclass
class ConvexDecompositionResponse:
    """Response from convex decomposition collision geometry generation."""

    success: bool
    collision_pieces: list[ConvexPiece] = field(default_factory=list)
    error_message: str | None = None
    processing_time_s: float = 0.0
