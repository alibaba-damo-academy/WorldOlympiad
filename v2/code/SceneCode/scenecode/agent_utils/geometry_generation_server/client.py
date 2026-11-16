import json
import logging
import time

from typing import Iterator

import requests

from .dataclasses import (
    GeometryGenerationError,
    GeometryGenerationServerRequest,
    GeometryGenerationServerResponse,
    StreamedResult,
)

console_logger = logging.getLogger(__name__)


class GeometryGenerationClient:
    """Client for making requests to the geometry generation server.

    Provides a high-level interface for generating 3D geometry from images
    using the geometry generation server. Handles HTTP communication, retries,
    error handling, and response parsing.

    The client maintains a persistent HTTP session for connection pooling
    and includes automatic retry logic with exponential backoff for
    transient failures.

    Example:
        >>> client = GeometryGenerationClient()
        >>> requests = [GeometryGenerationServerRequest(
        ...     image_path="/path/to/image.png",
        ...     output_dir="/path/to/output",
        ...     prompt="Modern wooden chair"
        ... )]
        >>> for index, response in client.generate_geometries(requests):
        ...     print(f"Generated geometry: {response.geometry_path}")
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7000):
        """Initialize geometry generation client.

        Args:
            host: Server hostname or IP address. Should be accessible from
                the current network context. Defaults to localhost.
            port: Server port number. Must match the port where the geometry
                generation server is listening. Defaults to 7000.
        """
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        console_logger.debug(
            f"Geometry generation client initialized for {self.base_url}"
        )

    def generate_geometries(
        self,
        geometry_requests: list[GeometryGenerationServerRequest],
        max_retries: int = 3,
        timeout_s: int = 3600,
    ) -> Iterator[
        tuple[int, GeometryGenerationServerResponse | GeometryGenerationError]
    ]:
        """Send batch geometry generation requests and yield results as they complete.

        Submits a batch of geometry generation requests to the server and yields
        results as they stream back. This enables pipelining where the client can
        start processing earlier results while the server continues working on
        later requests.

        Individual request failures are yielded as GeometryGenerationError objects
        rather than raising exceptions, allowing the batch to continue processing.

        Args:
            geometry_requests: List of geometry generation requests to process as a batch.
            max_retries: Maximum number of retries for transient failures.
            timeout_s: Timeout in seconds for the entire batch. Should scale with
                batch size and expected server queue depth.

        Yields:
            Tuple of (index, result) where index corresponds to the request's
            position in the input list and result is either:
            - GeometryGenerationServerResponse: Contains the generated geometry path
            - GeometryGenerationError: Contains error details for failed requests

        Raises:
            ConnectionError: If unable to connect to server after max retries.
            RuntimeError: If server returns invalid data (e.g., malformed JSON).
            TimeoutError: If request exceeds timeout limit.
            ValueError: If the requests list is empty.
        """
        if not geometry_requests:
            raise ValueError("Requests list cannot be empty")

        for attempt in range(max_retries):
            try:
                console_logger.debug(
                    f"Sending batch request (attempt {attempt + 1}) with "
                    f"{len(geometry_requests)} requests"
                )

                # Prepare request payload.
                request_data = [req.to_dict() for req in geometry_requests]

                # Send streaming request.
                http_response = self.session.post(
                    f"{self.base_url}/generate_geometries",
                    json=request_data,
                    stream=True,
                    timeout=(10, timeout_s),  # 10s connect, timeout_s read
                )
                http_response.raise_for_status()

                # Parse streaming NDJSON response.
                for line in http_response.iter_lines():
                    if line:
                        try:
                            result_data = json.loads(line.decode("utf-8"))
                            streamed_result = StreamedResult(**result_data)

                            if streamed_result.status == "error":
                                # Yield error and continue processing remaining results.
                                console_logger.warning(
                                    f"Geometry generation failed for request "
                                    f"{streamed_result.index}: {streamed_result.error}"
                                )
                                yield streamed_result.index, GeometryGenerationError(
                                    index=streamed_result.index,
                                    error_message=streamed_result.error
                                    or "Unknown error",
                                )
                                continue

                            # Convert to response object.
                            response = GeometryGenerationServerResponse(
                                **streamed_result.data
                            )
                            yield streamed_result.index, response

                        except json.JSONDecodeError as e:
                            raise RuntimeError(
                                f"Invalid JSON in streaming response: {e}"
                            ) from e

                console_logger.debug("Batch request completed successfully")
                return  # Success, exit retry loop

            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    console_logger.warning(
                        f"Connection failed, retrying... ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(min(2**attempt, 60))  # Exponential backoff with max 60s
                else:
                    console_logger.error("Asset server connection failed after retries")
                    raise ConnectionError(
                        f"Failed to connect to asset server at {self.base_url}"
                    ) from e

            except requests.exceptions.HTTPError as e:
                if e.response.status_code >= 500:
                    # Server error, might be temporary.
                    if attempt < max_retries - 1:
                        console_logger.warning(
                            f"Server error, retrying... ({attempt + 1}/{max_retries})"
                        )
                        time.sleep(2**attempt)
                        continue

                # Client error or persistent server error.
                try:
                    error_detail = e.response.json()["error"]
                except (KeyError, ValueError):
                    error_detail = str(e)
                console_logger.error(f"HTTP error from asset server: {error_detail}")
                raise RuntimeError(f"Asset server error: {error_detail}") from e

            except requests.exceptions.Timeout as e:
                console_logger.error("Batch asset generation request timed out")
                raise TimeoutError("Batch asset generation request timed out") from e

    def health_check(self) -> bool:
        """Check if the geometry generation server is healthy and responsive.

        Returns:
            True if server responds successfully to health check within
            5 seconds, False if server is unreachable, returns an error,
            or times out.
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            console_logger.warning(f"Health check failed: {e}")
            return False
