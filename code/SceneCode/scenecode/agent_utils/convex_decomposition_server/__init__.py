"""Convex decomposition collision geometry server package.

This package provides a server-client architecture for convex decomposition
(CoACD and V-HACD) that isolates OpenMP operations from the main worker process
to prevent deadlocks when using ThreadPoolExecutor.

Usage:
    from scenecode.agent_utils.convex_decomposition_server import (
        ConvexDecompositionServer,
        ConvexDecompositionClient,
    )

    # Start server (with optional log file for persistent logging)
    server = ConvexDecompositionServer(
        port_range=(7100, 7150), omp_threads=4, log_file=Path("server.log")
    )
    server.start()
    server.wait_until_ready()

    # Get client from server
    client = server.get_client()

    # Generate collision geometry (default: CoACD)
    pieces = client.generate_collision_geometry(mesh_path, threshold=0.05)

    # Or use V-HACD for more accurate collision geometry
    pieces = client.generate_collision_geometry(
        mesh_path, method="vhacd", max_convex_hulls=64
    )

    # Cleanup
    server.stop()
"""

from scenecode.agent_utils.convex_decomposition_server.client import (
    ConvexDecompositionClient,
)
from scenecode.agent_utils.convex_decomposition_server.server_manager import (
    ConvexDecompositionServer,
)

__all__ = ["ConvexDecompositionServer", "ConvexDecompositionClient"]
