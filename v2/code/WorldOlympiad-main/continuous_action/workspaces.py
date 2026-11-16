"""Frozen robot-workspace provenance used by formal CAE.

The workspace diameter is embodiment evidence, not a score-tunable metric
parameter.  Each formal adapter resolves one reviewed contract here and the
contract hash travels with the evaluation case.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping


WORKSPACE_CONTRACT_SCHEMA_VERSION = "worldolympiad-robot-workspace-v1"

_CONTRACTS: dict[str, dict[str, Any]] = {
    "cosmos3_nano_droid": {
        "schema_version": WORKSPACE_CONTRACT_SCHEMA_VERSION,
        "source_model": "cosmos3_nano_droid",
        "embodiment_id": "franka_arm",
        "endpoint_link": "manufacturer_reach_reference_point",
        "diameter_m": 1.71,
        "diameter_semantics": "twice_manufacturer_max_reach",
        "algorithm_version": "manufacturer_max_reach_diameter_v1",
        "source": (
            "https://download.franka.de/documents/"
            "Product%20Manual%20Franka%20Production%203_R02110_1.1_EN.pdf"
        ),
        "source_sha256": (
            "4a36212e3485607f7e21e5c2b1ec2125c4f08e703be8f9b94cb0059e6e311120"
        ),
        "source_measurement": "maximum_reach_m=0.855",
    },
    "lingbot_va_robotwin": {
        "schema_version": WORKSPACE_CONTRACT_SCHEMA_VERSION,
        "source_model": "lingbot_va_robotwin",
        "embodiment_id": "aloha_agilex_arx5_arm",
        "endpoint_link": "fr_link6",
        "diameter_m": 1.824341085722686,
        "diameter_semantics": "twice_urdf_serial_chain_upper_bound",
        "algorithm_version": "urdf_serial_chain_norm_upper_bound_v1",
        "source": (
            "RoboTwin/assets/embodiments/aloha-agilex/urdf/"
            "arx5_description_isaac.urdf"
        ),
        "source_sha256": (
            "097c59fb19a7b482249c6097df8319586ea7cfd268c015516f103b289a7e761a"
        ),
        "source_measurement": "serial_chain_upper_bound_m=0.912170542861343",
    },
    # The historical base generator uses the same reviewed ALOHA/ARX5 task
    # embodiment.  It remains explicit so a case can never silently fall back
    # to a video-derived scale.
    "lingbot_va_base": {
        "schema_version": WORKSPACE_CONTRACT_SCHEMA_VERSION,
        "source_model": "lingbot_va_base",
        "embodiment_id": "aloha_agilex_arx5_arm",
        "endpoint_link": "fr_link6",
        "diameter_m": 1.824341085722686,
        "diameter_semantics": "twice_urdf_serial_chain_upper_bound",
        "algorithm_version": "urdf_serial_chain_norm_upper_bound_v1",
        "source": (
            "RoboTwin/assets/embodiments/aloha-agilex/urdf/"
            "arx5_description_isaac.urdf"
        ),
        "source_sha256": (
            "097c59fb19a7b482249c6097df8319586ea7cfd268c015516f103b289a7e761a"
        ),
        "source_measurement": "serial_chain_upper_bound_m=0.912170542861343",
    },
}


def _sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def robot_workspace_contract_for_model(source_model: str) -> dict[str, Any]:
    """Return the one frozen workspace contract for an adapter model."""

    try:
        payload = deepcopy(_CONTRACTS[source_model])
    except KeyError as exc:
        raise ValueError(
            f"no reviewed robot-workspace contract for source_model={source_model!r}"
        ) from exc
    payload["contract_sha256"] = _sha256(payload)
    return payload


def validate_robot_workspace_contract(
    raw: Mapping[str, Any], *, source_model: str | None = None
) -> dict[str, Any]:
    """Validate digest, value types, and (when known) the reviewed registry row."""

    payload = dict(raw)
    digest = payload.pop("contract_sha256", None)
    if not isinstance(digest, str) or digest != _sha256(payload):
        raise ValueError("robot workspace contract_sha256 is missing or invalid")
    if payload.get("schema_version") != WORKSPACE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported robot workspace contract schema")
    diameter = payload.get("diameter_m")
    if (
        isinstance(diameter, bool)
        or not isinstance(diameter, (int, float))
        or not math.isfinite(float(diameter))
        or float(diameter) <= 0.0
    ):
        raise ValueError("robot workspace diameter_m must be finite and positive")
    for name in (
        "source_model",
        "embodiment_id",
        "endpoint_link",
        "diameter_semantics",
        "algorithm_version",
        "source",
        "source_sha256",
        "source_measurement",
    ):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise ValueError(f"robot workspace contract requires {name}")
    if len(payload["source_sha256"]) != 64:
        raise ValueError("robot workspace source_sha256 must be a SHA-256 digest")
    resolved_model = source_model or payload["source_model"]
    expected = robot_workspace_contract_for_model(resolved_model)
    normalized = {**payload, "contract_sha256": digest}
    if normalized != expected:
        raise ValueError(
            f"robot workspace contract differs from reviewed {resolved_model!r} contract"
        )
    return normalized
