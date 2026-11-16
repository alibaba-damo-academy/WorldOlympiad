"""Loop detection utility for preventing infinite repeated function calls.

This module provides a decorator-based loop detector that tracks recent function
calls and blocks repeated identical attempts. Configurable with maximum attempts,
tracking window size, and custom error response factories.
"""

import json
import logging

from collections import deque
from functools import wraps
from typing import Any, Callable

console_logger = logging.getLogger(__name__)


class LoopDetector:
    """Detects and prevents repeated identical tool calls.

    Can be applied to any tool method to prevent infinite loops.
    Tracks recent calls and blocks repeated identical attempts.

    Usage:
        # Initialize with config and optional error factory.
        detector = LoopDetector(
            max_attempts=3,
            window_size=20,
            default_error_factory=my_error_factory,
        )

        # Apply to methods in __init__.
        self.method = detector(self.method)

        # Or with method-specific error handling.
        @detector.protected(error_response_factory=custom_error_factory)
        def another_method(self, ...):
            ...
    """

    def __init__(
        self,
        max_attempts: int = 3,
        window_size: int = 20,
        enabled: bool = True,
        default_error_factory: Callable[[str, int, tuple, dict], str] | None = None,
    ) -> None:
        """Initialize loop detector.

        Args:
            max_attempts: Max identical calls before blocking.
            window_size: Size of recent call tracking window.
            enabled: Whether loop detection is active.
            default_error_factory: Optional factory to create default error responses.
                Gets (method_name, attempt_count, args, kwargs) and returns JSON string.
        """
        self.max_attempts = max_attempts
        self.window_size = window_size
        self.enabled = enabled
        self.recent_calls: deque[str] = deque(maxlen=window_size)
        self.default_error_factory = default_error_factory

    def __call__(self, func: Callable) -> Callable:
        """Allow using as @loop_detector decorator directly.

        This enables: @self.loop_detector instead of @self.loop_detector.protected()
        """
        return self.protected()(func)

    def protected(
        self, error_response_factory: Callable[[str, int], str] | None = None
    ) -> Callable:
        """Decorator to protect a method from infinite loops.

        Args:
            error_response_factory: Optional factory to create error response.
                Gets (method_name, attempt_count) and returns JSON string response.
                If None, uses default_error_factory from __init__ or returns simple error string.

        Returns:
            Decorator function that adds loop protection.
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self.enabled:
                    return func(*args, **kwargs)

                # Create signature for this call (skip 'self' arg).
                sig_args = args[1:] if args else ()
                signature = {
                    "method": func.__name__,
                    "args": sig_args,
                    "kwargs": kwargs,
                }
                sig_str = json.dumps(signature, sort_keys=True, default=str)

                # Count recent identical calls.
                identical_count = sum(1 for s in self.recent_calls if s == sig_str)
                console_logger.debug(f"Identical count: {identical_count}")

                if identical_count >= self.max_attempts:
                    console_logger.warning(
                        f"Loop detected: {identical_count} identical calls to "
                        f"{func.__name__}"
                    )

                    if error_response_factory:
                        return error_response_factory(func.__name__, identical_count)
                    else:
                        # Default error response for furniture tools.
                        return self._default_error_response(
                            func.__name__, identical_count, args, kwargs
                        )

                # Track this call.
                self.recent_calls.append(sig_str)

                # Execute the actual function.
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def _default_error_response(
        self, method_name: str, attempt_count: int, args: tuple, kwargs: dict[str, Any]
    ) -> str:
        """Create default error response.

        Args:
            method_name: Name of the method that was called.
            attempt_count: Number of identical attempts.
            args: Positional arguments to the method.
            kwargs: Keyword arguments to the method.

        Returns:
            JSON error response.
        """
        if self.default_error_factory:
            return self.default_error_factory(method_name, attempt_count, args, kwargs)
        else:
            # Simple fallback error message.
            return (
                f"Loop detected: {attempt_count} identical calls to {method_name}. "
                "Try a different approach."
            )

    def reset(self) -> None:
        """Clear tracking history."""
        self.recent_calls.clear()
