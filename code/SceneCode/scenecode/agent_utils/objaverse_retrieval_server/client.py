import json
import logging
import time

from typing import Iterator

import requests

from .dataclasses import (
    ObjaverseRetrievalServerRequest,
    ObjaverseRetrievalServerResponse,
    StreamedResult,
)

console_logger = logging.getLogger(__name__)


class ObjaverseRetrievalClient:
    """Client for making requests to the Objaverse retrieval server.

    Provides a high-level interface for retrieving 3D objects from the
    ObjectThor library using semantic search with CLIP. Handles HTTP
    communication, retries, error handling, and response parsing.

    The client maintains a persistent HTTP session for connection pooling
    and includes automatic retry logic with exponential backoff for
    transient failures.

    Example:
        >>> client = ObjaverseRetrievalClient()
        >>> requests = [ObjaverseRetrievalServerRequest(
        ...     object_description="Modern wooden chair",
        ...     object_type="FURNITURE",
        ...     desired_dimensions=(0.6, 0.6, 1.0)
        ... )]
        >>> for index, response in client.retrieve_objects(requests):
        ...     print(f"Retrieved: {response.results[0].mesh_path}")
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7007):
        """Initialize Objaverse retrieval client.

        Args:
            host: Server hostname or IP address. Should be accessible from
                the current network context. Defaults to localhost.
            port: Server port number. Must match the port where the Objaverse
                retrieval server is listening. Defaults to 7007.
        """
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        console_logger.debug(
            f"Objaverse retrieval client initialized for {self.base_url}"
        )

    def retrieve_objects(
        self,
        retrieval_requests: list[ObjaverseRetrievalServerRequest],
        max_retries: int = 3,
        timeout_s: int = 3600,
    ) -> Iterator[tuple[int, ObjaverseRetrievalServerResponse]]:
        """Send batch Objaverse retrieval requests and yield results as they complete.

        Submits a batch of Objaverse retrieval requests to the server and yields
        results as they stream back. This enables pipelining where the client can
        start processing earlier results while the server continues working on
        later requests.

        Args:
            retrieval_requests: List of Objaverse retrieval requests to process as a
                batch.
            max_retries: Maximum number of retries for transient failures.
            timeout_s: Timeout in seconds for the entire batch. Should scale with
                batch size and expected server queue depth.

        Yields:
            Tuple of (index, response) where index corresponds to the request's
            position in the input list and response contains the retrieved object
            data.

        Raises:
            ConnectionError: If unable to connect to server after max retries.
            RuntimeError: If server returns an error response or invalid data.
            TimeoutError: If request exceeds timeout limit.
            ValueError: If the requests list is empty.
        """
        if not retrieval_requests:
            raise ValueError("Requests list cannot be empty")

        for attempt in range(max_retries):
            try:
                console_logger.debug(
                    f"Sending batch request (attempt {attempt + 1}) with "
                    f"{len(retrieval_requests)} requests"
                )

                # Prepare request payload.
                request_data = [req.to_dict() for req in retrieval_requests]

                # Send streaming request.
                http_response = self.session.post(
                    f"{self.base_url}/retrieve_objects",
                    json=request_data,
                    stream=True,
                    timeout=(10, timeout_s),  # 10s connect, timeout_s read.
                )
                http_response.raise_for_status()

                # Parse streaming NDJSON response.
                for line in http_response.iter_lines():
                    if line:
                        try:
                            result_data = json.loads(line.decode("utf-8"))
                            streamed_result = StreamedResult(**result_data)

                            if streamed_result.status == "error":
                                raise RuntimeError(
                                    f"Objaverse retrieval failed for request "
                                    f"{streamed_result.index}: "
                                    f"{streamed_result.error}"
                                )

                            # Convert to response object using from_dict for proper
                            # nested deserialization.
                            if streamed_result.data is None:
                                raise RuntimeError(
                                    f"Server returned success status but no data for "
                                    f"request {streamed_result.index}"
                                )
                            response = ObjaverseRetrievalServerResponse.from_dict(
                                streamed_result.data
                            )
                            yield streamed_result.index, response

                        except json.JSONDecodeError as e:
                            raise RuntimeError(
                                f"Invalid JSON in streaming response: {e}"
                            ) from e

                console_logger.debug("Batch request completed successfully")
                return  # Success, exit retry loop.

            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    console_logger.warning(
                        f"Connection failed, retrying... "
                        f"({attempt + 1}/{max_retries})"
                    )
                    # Exponential backoff with max 60s.
                    time.sleep(min(2**attempt, 60))
                else:
                    console_logger.error(
                        "Objaverse retrieval server connection failed after retries"
                    )
                    raise ConnectionError(
                        f"Failed to connect to Objaverse retrieval server at "
                        f"{self.base_url}"
                    ) from e

            except requests.exceptions.HTTPError as e:
                if e.response.status_code >= 500:
                    # Server error, might be temporary.
                    if attempt < max_retries - 1:
                        console_logger.warning(
                            f"Server error, retrying... "
                            f"({attempt + 1}/{max_retries})"
                        )
                        time.sleep(2**attempt)
                        continue

                # Client error or persistent server error.
                try:
                    error_detail = e.response.json()["error"]
                except (KeyError, ValueError):
                    error_detail = str(e)
                console_logger.error(
                    f"HTTP error from Objaverse retrieval server: {error_detail}"
                )
                raise RuntimeError(
                    f"Objaverse retrieval server error: {error_detail}"
                ) from e

            except requests.exceptions.Timeout as e:
                console_logger.error("Batch Objaverse retrieval request timed out")
                raise TimeoutError("Batch Objaverse retrieval request timed out") from e

    def health_check(self) -> bool:
        """Check if the Objaverse retrieval server is healthy and responsive.

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
