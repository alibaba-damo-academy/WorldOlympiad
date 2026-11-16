"""Fair round-robin scheduler for server request processing.

This module provides a generic scheduler that can be used by any Flask-based
server that needs fair batch scheduling across multiple clients.
"""

import threading

from collections import deque
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

# Generic request type for type-safe scheduling.
RequestT = TypeVar("RequestT")


@dataclass
class QueuedRequest(Generic[RequestT]):
    """A request waiting to be processed in the scheduler queue.

    Attributes:
        client_id: Unique identifier for the client that submitted this request.
        request_index: Index of this request within the client's batch.
        request: The actual request object to be processed.
        callback: Function to call with results when processing completes.
        received_timestamp: Time when request was received by server (time.time()).
    """

    client_id: str
    request_index: int
    request: RequestT
    callback: Callable[[int, tuple[str, dict]], None]
    received_timestamp: float


class StrictRoundRobinScheduler(Generic[RequestT]):
    """Fair round-robin scheduler for server requests.

    This scheduler ensures that all active clients receive equal processing time
    by cycling through clients in a strict round-robin fashion. Each client gets
    exactly one request processed per round, ensuring fairness regardless of
    batch sizes or arrival times.

    Fairness Properties:
    - New clients join at the END of the rotation order
    - No queue interleaving - maintains strict round-robin sequence
    - Each active client gets exactly 1/N of processing time (where N = clients)
    - Clients who arrived earlier do not lose share to later arrivals

    Example:
        Initial: Client A (5 requests), Client B (3 requests)
        Processing: A1 → B1 → A2 → B2 → A3 → B3 → A4 → A5

        Client C arrives with 4 requests:
        Processing continues: ...A4 → A5 → C1 → C2 → C3 → C4
        (C joins rotation after A and B complete)

        If C arrived earlier:
        Processing: A1 → B1 → C1 → A2 → B2 → C2 → A3 → B3 → C3 → A4 → C4 → A5
        Each gets exactly 1/3 of processing time.

    This design maximizes parallel CPU utilization across all clients since
    they receive steady streams of completed results for downstream processing.

    Type Parameters:
        RequestT: The type of request objects this scheduler handles.
    """

    def __init__(self) -> None:
        """Initialize the round-robin scheduler."""
        self._lock = threading.Lock()
        """Lock for thread-safe access to scheduler state."""

        self.client_queues: dict[str, deque[QueuedRequest[RequestT]]] = {}
        """Per-client queues of requests waiting to be processed."""

        self.client_order: list[str] = []
        """Fixed rotation order based on client arrival time."""

        self.next_client_index = 0
        """Index of the next client to serve in the rotation."""

    def add_batch(
        self,
        client_id: str,
        requests: list[RequestT],
        callback: Callable[[int, tuple[str, dict]], None],
        received_timestamp: float,
    ) -> None:
        """Add a batch of requests to the scheduler.

        New clients are added to the END of the rotation order, ensuring they
        don't disrupt the processing time allocation of existing clients.

        Args:
            client_id: Unique identifier for the client (typically batch ID).
            requests: List of requests from this client.
            callback: Function to call with results for this client.
            received_timestamp: Time when request was received by server.
        """
        with self._lock:
            if client_id not in self.client_queues:
                self.client_queues[client_id] = deque()
                self.client_order.append(client_id)

            for request_index, request in enumerate(requests):
                queued_request: QueuedRequest[RequestT] = QueuedRequest(
                    client_id=client_id,
                    request_index=request_index,
                    request=request,
                    callback=callback,
                    received_timestamp=received_timestamp,
                )
                self.client_queues[client_id].append(queued_request)

    def get_next_request(self) -> QueuedRequest[RequestT] | None:
        """Get the next request to process using fair round-robin.

        Returns:
            The next request to process, or None if no requests are pending.
        """
        with self._lock:
            if not self.client_order:
                return None

            attempts = 0
            while attempts < len(self.client_order):
                client_id = self.client_order[self.next_client_index]
                queue = self.client_queues.get(client_id)

                if queue:
                    request = queue.popleft()

                    if not queue:
                        del self.client_queues[client_id]
                        self.client_order.remove(client_id)
                        if self.client_order:
                            self.next_client_index %= len(self.client_order)
                        else:
                            self.next_client_index = 0
                    else:
                        self.next_client_index = (self.next_client_index + 1) % len(
                            self.client_order
                        )

                    return request

                self.next_client_index = (self.next_client_index + 1) % len(
                    self.client_order
                )
                attempts += 1

            return None

    def get_queue_size(self) -> int:
        """Get total number of pending requests across all clients."""
        with self._lock:
            return sum(len(queue) for queue in self.client_queues.values())

    def get_client_count(self) -> int:
        """Get number of active clients with pending requests."""
        with self._lock:
            return len(self.client_order)
