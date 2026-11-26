"""HTTP client for convex decomposition server."""

import logging

from pathlib import Path

import requests
import trimesh

console_logger = logging.getLogger(__name__)


class ConvexDecompositionClient:
    """HTTP client for communicating with convex decomposition server.

    This client sends collision geometry generation requests to the server
    and converts the response to trimesh objects. Supports both CoACD and V-HACD
    decomposition methods.
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 7100, timeout: float = 300.0
    ) -> None:
        """Initialize the convex decomposition client.

        Args:
            host: Host address of the server.
            port: Port number of the server.
            timeout: HTTP request timeout in seconds. Decomposition can take a while
                for complex meshes, so default is 5 minutes.
        """
        self._host = host
        self._port = port
        self._timeout = timeout
        self._base_url = f"http://{host}:{port}"

    def generate_collision_geometry(
        self,
        mesh_path: Path,
        method: str = "coacd",
        # CoACD parameters.
        threshold: float = 0.05,
        max_convex_hull: int = -1,
        preprocess_mode: str = "auto",
        preprocess_resolution: int = 50,
        resolution: int = 2000,
        mcts_nodes: int = 20,
        mcts_iterations: int = 150,
        mcts_max_depth: int = 3,
        pca: bool = False,
        merge: bool = True,
        decimate: bool = False,
        max_ch_vertex: int = 256,
        extrude: bool = False,
        extrude_margin: float = 0.01,
        apx_mode: str = "ch",
        seed: int = 0,
        # V-HACD parameters.
        max_convex_hulls: int = 64,
        vhacd_resolution: int = 400000,
        max_recursion_depth: int = 10,
        max_num_vertices_per_ch: int = 64,
        min_volume_percent_error: float = 1.0,
        shrink_wrap: bool = True,
        fill_mode: str = "flood",
        min_edge_length: int = 2,
        find_best_plane: bool = False,
    ) -> list[trimesh.Trimesh]:
        """Generate collision geometry via HTTP request to decomposition server.

        Args:
            mesh_path: Path to the mesh file (GLTF/GLB/OBJ).
            method: Decomposition method ("coacd" or "vhacd").

            CoACD parameters (used when method="coacd"):
                threshold: Approximation threshold (0.01-0.1 typical range).
                    Lower = more pieces, higher fidelity.
                    Higher = fewer pieces, simpler.
                max_convex_hull: Maximum number of convex hulls (-1 for unlimited).
                preprocess_mode: Preprocessing mode ("auto", "on", "off").
                preprocess_resolution: Resolution for preprocessing.
                resolution: Voxel resolution for decomposition.
                mcts_nodes: MCTS nodes for optimization.
                mcts_iterations: MCTS iterations for optimization.
                mcts_max_depth: MCTS max depth for optimization.
                pca: Enable PCA preprocessing.
                merge: Merge small hulls.
                decimate: Decimate mesh before decomposition.
                max_ch_vertex: Max vertices per convex hull.
                extrude: Extrude thin parts.
                extrude_margin: Extrusion margin.
                apx_mode: Approximation mode ("ch" or "box").
                seed: Random seed for reproducibility.

            V-HACD parameters (used when method="vhacd"):
                max_convex_hulls: Maximum number of convex hulls.
                vhacd_resolution: Voxel resolution (higher = more accurate).
                max_recursion_depth: Max recursion depth for decomposition.
                max_num_vertices_per_ch: Max vertices per convex hull.
                min_volume_percent_error: Min volume error percent to stop.
                shrink_wrap: Enable shrink wrap.
                fill_mode: Fill mode ("flood", "surface", "raycast").
                min_edge_length: Minimum edge length.
                find_best_plane: Find best plane.

        Returns:
            List of convex trimesh objects from the decomposition.

        Raises:
            RuntimeError: If the server request fails or returns an error.
            requests.RequestException: If there's a network/connection error.
        """
        url = f"{self._base_url}/generate_collision"

        payload = {"mesh_path": str(mesh_path), "method": method}

        # Add method-specific parameters.
        if method == "coacd":
            payload.update(
                {
                    "threshold": threshold,
                    "max_convex_hull": max_convex_hull,
                    "preprocess_mode": preprocess_mode,
                    "preprocess_resolution": preprocess_resolution,
                    "resolution": resolution,
                    "mcts_nodes": mcts_nodes,
                    "mcts_iterations": mcts_iterations,
                    "mcts_max_depth": mcts_max_depth,
                    "pca": pca,
                    "merge": merge,
                    "decimate": decimate,
                    "max_ch_vertex": max_ch_vertex,
                    "extrude": extrude,
                    "extrude_margin": extrude_margin,
                    "apx_mode": apx_mode,
                    "seed": seed,
                }
            )
        else:  # vhacd
            payload.update(
                {
                    "max_convex_hulls": max_convex_hulls,
                    "vhacd_resolution": vhacd_resolution,
                    "max_recursion_depth": max_recursion_depth,
                    "max_num_vertices_per_ch": max_num_vertices_per_ch,
                    "min_volume_percent_error": min_volume_percent_error,
                    "shrink_wrap": shrink_wrap,
                    "fill_mode": fill_mode,
                    "min_edge_length": min_edge_length,
                    "find_best_plane": find_best_plane,
                }
            )

        console_logger.debug(
            f"Requesting collision geometry for {mesh_path.name} (method={method})"
        )

        try:
            response = requests.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()

            try:
                data = response.json()
            except ValueError as e:
                console_logger.error(f"Invalid JSON response from server: {e}")
                raise RuntimeError(f"Failed to parse server response: {e}") from e

            if not data.get("success", False):
                error_msg = data.get("error_message", "Unknown error")
                raise RuntimeError(f"Convex decomposition server error: {error_msg}")

            # Convert response pieces to trimesh objects.
            collision_pieces = data.get("collision_pieces", [])
            if not collision_pieces:
                console_logger.warning(
                    f"Server returned no collision pieces for {mesh_path.name}"
                )

            convex_meshes = []
            for piece in collision_pieces:
                if not isinstance(piece, dict):
                    raise RuntimeError(f"Invalid collision piece format: {type(piece)}")
                if "vertices" not in piece or "faces" not in piece:
                    raise RuntimeError("Collision piece missing vertices or faces")
                mesh = trimesh.Trimesh(
                    vertices=piece["vertices"],
                    faces=piece["faces"],
                )
                convex_meshes.append(mesh)

            processing_time = data.get("processing_time_s", 0)
            console_logger.info(
                f"{method.upper()} decomposition: {len(convex_meshes)} pieces "
                f"in {processing_time:.2f}s"
            )

            return convex_meshes

        except requests.RequestException as e:
            console_logger.error(f"Server request failed: {e}")
            raise RuntimeError(f"Failed to connect to server: {e}") from e

    def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = requests.get(f"{self._base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
