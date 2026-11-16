"""Articulated retrieval server for semantic search over articulated object datasets.

This package provides a Flask-based server for CLIP-based semantic retrieval
over articulated object datasets (PartNet-Mobility, ArtVIP). The server pattern
avoids CUDA multiprocessing issues by keeping the CLIP model in a single process.

Usage:
    # Programmatic usage (recommended for experiments)
    from scenecode.agent_utils.articulated_retrieval_server import (
        ArticulatedRetrievalServer,
        ArticulatedRetrievalClient,
        ArticulatedRetrievalServerRequest,
    )

    server = ArticulatedRetrievalServer(host="127.0.0.1", port=7002)
    server.start()
    server.wait_until_ready()

    client = ArticulatedRetrievalClient(host="127.0.0.1", port=7002)
    requests = [ArticulatedRetrievalServerRequest(
        object_description="Modern wooden cabinet",
        object_type="FURNITURE",
        output_dir="/tmp/output",
    )]
    for index, response in client.retrieve_objects(requests):
        print(f"Retrieved: {response.results[0].mesh_path}")

    server.stop()

    # Standalone usage (for testing/debugging/microservice deployment)
    # python -m scenecode.agent_utils.articulated_retrieval_server.standalone_server
"""

from .client import ArticulatedRetrievalClient
from .dataclasses import (
    ArticulatedRetrievalResult,
    ArticulatedRetrievalServerRequest,
    ArticulatedRetrievalServerResponse,
)
from .server_manager import ArticulatedRetrievalServer

__all__ = [
    "ArticulatedRetrievalClient",
    "ArticulatedRetrievalResult",
    "ArticulatedRetrievalServer",
    "ArticulatedRetrievalServerRequest",
    "ArticulatedRetrievalServerResponse",
]
