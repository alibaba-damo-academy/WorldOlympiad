"""CPU-side native FastWAM HTTP transport for the RoboTwin simulator.

No EE conversion, action normalization, or simulator execution happens here.
The caller executes the returned prefix with action_type='qpos'. Transport and
protocol failures are infrastructure errors, never task failures.
"""
from __future__ import annotations

import json
from urllib.request import Request, urlopen

import numpy as np

from wo.models.fastwam_server import encode_observation


class FastWAMClient:
    def __init__(self, port: int = 29058, timeout: float = 300):
        self.url = f"http://127.0.0.1:{int(port)}/infer"
        self.timeout = timeout

    def infer(self, observation: dict, instruction: str) -> np.ndarray:
        payload = encode_observation(observation, instruction)
        request = Request(self.url, data=json.dumps(payload, allow_nan=False).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        return executable_prefix(result)


def executable_prefix(result: dict) -> np.ndarray:
    """Validate the wire contract and keep only the upstream replanning prefix."""
    if result.get("action_type") != "qpos":
        raise RuntimeError("FastWAM response must contain absolute qpos actions")
    actions = np.asarray(result.get("actions"), dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
        raise RuntimeError("FastWAM response has invalid native joint actions")
    count = result.get("replan_steps")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(actions):
        raise RuntimeError("FastWAM response has invalid replanning prefix")
    return actions[:count].copy()
