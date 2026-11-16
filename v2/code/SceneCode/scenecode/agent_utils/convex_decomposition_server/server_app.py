"""Flask application for convex decomposition collision geometry generation server."""

import logging
import time

from pathlib import Path

import coacd
import flask
import numpy as np
import trimesh

console_logger = logging.getLogger(__name__)


class ConvexDecompositionServerApp(flask.Flask):
    """Flask server for convex decomposition (CoACD and V-HACD).

    This server isolates convex decomposition operations in a separate process
    to avoid deadlocks when used with ThreadPoolExecutor in the main worker.
    """

    def __init__(self) -> None:
        """Initialize the Flask app."""
        super().__init__("convex_decomposition_server")

        self.add_url_rule("/", view_func=self._root_endpoint)
        self.add_url_rule(
            "/health",
            endpoint="health",
            methods=["GET"],
            view_func=self._health_endpoint,
        )
        self.add_url_rule(
            "/generate_collision",
            endpoint="generate_collision",
            methods=["POST"],
            view_func=self._generate_collision_endpoint,
        )

        # Set CoACD log level once at startup.
        coacd.set_log_level("error")
        console_logger.info("Convex decomposition server initialized")

    def _root_endpoint(self) -> str:
        """Display a banner page at the server root."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Convex Decomposition Server</title>
        </head>
        <body>
            <h1>Convex Decomposition Collision Geometry Server</h1>
            <p>This server provides convex decomposition services using CoACD or V-HACD.</p>
            <p>POST to /generate_collision with mesh_path and method to generate collision geometry.</p>
        </body>
        </html>
        """

    def _health_endpoint(self) -> flask.Response:
        """Health check endpoint."""
        return flask.jsonify({"status": "healthy"})

    def _generate_collision_endpoint(self) -> flask.Response:
        """Generate collision geometry from a mesh file.

        Expects JSON body with:
            mesh_path: Path to the mesh file (GLTF/GLB/OBJ).
            method: Decomposition method ("coacd" or "vhacd", default: "coacd").

            CoACD parameters (used when method="coacd"):
                threshold: Approximation threshold (default: 0.05).
                max_convex_hull: Max convex hulls (default: -1 for unlimited).
                preprocess_mode: Preprocessing mode (default: "auto").
                preprocess_resolution: Resolution for preprocessing (default: 50).
                resolution: Voxel resolution (default: 2000).
                mcts_nodes: MCTS nodes (default: 20).
                mcts_iterations: MCTS iterations (default: 150).
                mcts_max_depth: MCTS max depth (default: 3).
                pca: PCA preprocessing (default: false).
                merge: Merge small hulls (default: true).
                decimate: Decimate mesh before decomposition (default: false).
                max_ch_vertex: Max vertices per convex hull (default: 256).
                extrude: Extrude thin parts (default: false).
                extrude_margin: Extrusion margin (default: 0.01).
                apx_mode: Approximation mode (default: "ch").
                seed: Random seed (default: 0).

            V-HACD parameters (used when method="vhacd"):
                max_convex_hulls: Max number of convex hulls (default: 64).
                resolution: Voxel resolution (default: 400000).
                max_recursion_depth: Max recursion depth (default: 10).
                max_num_vertices_per_ch: Max vertices per convex hull (default: 64).
                min_volume_percent_error: Min volume error percent (default: 1.0).
                shrink_wrap: Enable shrink wrap (default: true).
                fill_mode: Fill mode (default: "flood").
                min_edge_length: Min edge length (default: 2).
                find_best_plane: Find best plane (default: false).

        Returns:
            JSON response with collision_pieces list.
        """
        request_start = time.time()

        try:
            data = flask.request.get_json()
            if not data:
                flask.abort(400, description="Missing JSON body")

            mesh_path_str = data.get("mesh_path")
            if not mesh_path_str:
                flask.abort(400, description="Missing mesh_path parameter")

            mesh_path = Path(mesh_path_str)
            if not mesh_path.exists():
                flask.abort(400, description=f"Mesh file not found: {mesh_path}")

            # Get decomposition method.
            method = data.get("method", "coacd")
            if method not in ("coacd", "vhacd"):
                flask.abort(400, description=f"Invalid method: {method}")

            console_logger.info(f"Processing mesh: {mesh_path.name} (method={method})")

            # Load mesh.
            mesh = trimesh.load(mesh_path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())

            # Dispatch to appropriate decomposition method.
            if method == "vhacd":
                pieces = self._run_vhacd(mesh, data)
            else:
                pieces = self._run_coacd(mesh, data)

            processing_time = time.time() - request_start
            console_logger.info(
                f"{method.upper()} decomposition complete: {len(pieces)} pieces "
                f"from {len(mesh.vertices)} vertices in {processing_time:.2f}s"
            )

            return flask.jsonify(
                {
                    "success": True,
                    "collision_pieces": pieces,
                    "processing_time_s": processing_time,
                }
            )

        except Exception as e:
            # Don't catch HTTPException from flask.abort() - let Flask handle it.
            from werkzeug.exceptions import HTTPException

            if isinstance(e, HTTPException):
                raise
            console_logger.error(f"Convex decomposition failed: {e}")
            return (
                flask.jsonify(
                    {
                        "success": False,
                        "error_message": str(e),
                        "collision_pieces": [],
                        "processing_time_s": time.time() - request_start,
                    }
                ),
                500,
            )

    def _run_coacd(
        self, mesh: trimesh.Trimesh, data: dict
    ) -> list[dict[str, list[list[float]]]]:
        """Run CoACD convex decomposition.

        Args:
            mesh: Input mesh to decompose.
            data: Request data containing CoACD parameters.

        Returns:
            List of convex pieces as dicts with vertices and faces.
        """
        # Extract CoACD parameters.
        threshold = data.get("threshold", 0.05)
        max_convex_hull = data.get("max_convex_hull", -1)
        preprocess_mode = data.get("preprocess_mode", "auto")
        preprocess_resolution = data.get("preprocess_resolution", 50)
        resolution = data.get("resolution", 2000)
        mcts_nodes = data.get("mcts_nodes", 20)
        mcts_iterations = data.get("mcts_iterations", 150)
        mcts_max_depth = data.get("mcts_max_depth", 3)
        pca = data.get("pca", False)
        merge = data.get("merge", True)
        decimate = data.get("decimate", False)
        max_ch_vertex = data.get("max_ch_vertex", 256)
        extrude = data.get("extrude", False)
        extrude_margin = data.get("extrude_margin", 0.01)
        apx_mode = data.get("apx_mode", "ch")
        seed = data.get("seed", 0)

        console_logger.debug(f"CoACD params: threshold={threshold}")

        # Run CoACD convex decomposition.
        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        coacd_result = coacd.run_coacd(
            coacd_mesh,
            threshold=threshold,
            max_convex_hull=max_convex_hull,
            preprocess_mode=preprocess_mode,
            preprocess_resolution=preprocess_resolution,
            resolution=resolution,
            mcts_nodes=mcts_nodes,
            mcts_iterations=mcts_iterations,
            mcts_max_depth=mcts_max_depth,
            pca=pca,
            merge=merge,
            decimate=decimate,
            max_ch_vertex=max_ch_vertex,
            extrude=extrude,
            extrude_margin=extrude_margin,
            apx_mode=apx_mode,
            seed=seed,
        )

        # Convert to response format.
        pieces = []
        for vertices, faces in coacd_result:
            if isinstance(vertices, np.ndarray):
                vertices = vertices.tolist()
            if isinstance(faces, np.ndarray):
                faces = faces.tolist()
            pieces.append({"vertices": vertices, "faces": faces})

        return pieces

    def _run_vhacd(
        self, mesh: trimesh.Trimesh, data: dict
    ) -> list[dict[str, list[list[float]]]]:
        """Run V-HACD convex decomposition.

        Args:
            mesh: Input mesh to decompose.
            data: Request data containing V-HACD parameters.

        Returns:
            List of convex pieces as dicts with vertices and faces.
        """
        # Extract V-HACD parameters.
        max_convex_hulls = data.get("max_convex_hulls", 64)
        resolution = data.get("vhacd_resolution", 400000)
        max_recursion_depth = data.get("max_recursion_depth", 10)
        max_num_vertices_per_ch = data.get("max_num_vertices_per_ch", 64)
        min_volume_percent_error = data.get("min_volume_percent_error", 1.0)
        shrink_wrap = data.get("shrink_wrap", True)
        fill_mode = data.get("fill_mode", "flood")
        min_edge_length = data.get("min_edge_length", 2)
        find_best_plane = data.get("find_best_plane", False)

        console_logger.debug(f"V-HACD params: max_convex_hulls={max_convex_hulls}")

        # Run V-HACD via trimesh.
        vhacd_result = mesh.convex_decomposition(
            maxConvexHulls=max_convex_hulls,
            resolution=resolution,
            maxRecursionDepth=max_recursion_depth,
            maxNumVerticesPerCH=max_num_vertices_per_ch,
            minimumVolumePercentErrorAllowed=min_volume_percent_error,
            shrinkWrap=shrink_wrap,
            fillMode=fill_mode,
            minEdgeLength=min_edge_length,
            findBestPlane=find_best_plane,
        )

        # Handle single mesh result (convex_decomposition returns list of Trimesh).
        if not isinstance(vhacd_result, list):
            vhacd_result = [vhacd_result]

        # Convert to response format.
        # convex_decomposition() returns a list of trimesh.Trimesh objects.
        pieces = []
        for piece_mesh in vhacd_result:
            vertices = piece_mesh.vertices
            faces = piece_mesh.faces
            if isinstance(vertices, np.ndarray):
                vertices = vertices.tolist()
            if isinstance(faces, np.ndarray):
                faces = faces.tolist()
            pieces.append({"vertices": vertices, "faces": faces})

        return pieces
