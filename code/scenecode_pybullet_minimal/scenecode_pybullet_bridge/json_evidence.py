"""JSON evidence serialization that stays parseable by strict readers.

Contact-margin measurements that no geometry could produce are recorded as
``-inf`` in the in-memory result records, and ``Infinity``/``NaN`` are not JSON.
Python's own reader accepts them, so the defect stayed invisible until a strict
read-only inspector parsed the evidence and refused the file.  Every bridge
artifact therefore goes through this writer, which reports a non-finite number
as ``null``: the decision semantics are untouched, because a missing margin
still fails every finite-margin comparison the readers make, and the record now
states honestly that no margin was measurable.
"""

from __future__ import annotations

import json
import math
from typing import Any


def json_safe(value: Any) -> Any:
    """Return ``value`` with every non-finite float replaced by ``None``."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def dumps(value: Any, **kwargs: Any) -> str:
    """Serialize evidence exactly like ``json.dumps`` but without NaN/Infinity."""

    return json.dumps(json_safe(value), **kwargs)
