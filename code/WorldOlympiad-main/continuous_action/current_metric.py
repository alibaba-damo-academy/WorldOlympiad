"""The single active CAE definition: complete path shape and physical speed."""
import hashlib
import json

CURRENT_METRIC_VERSION = "full_trajectory_shape_speed"
CURRENT_METRIC_PROFILE = "full_trajectory_shape_speed"
CURRENT_SCHEMA_VERSION = "worldolympiad-continuous-action-shape-speed-v1"
CURRENT_MANIFEST_NAME = "continuous_action_manifest.json"
CURRENT_PAIR_METADATA_NAME = "pair_metadata_cae.json"


def current_metric_config():
    return {
        "metric_version": CURRENT_METRIC_VERSION,
        "metric_profile": CURRENT_METRIC_PROFILE,
        "path_score": "1 - rms(r(s)-p(s))/(rms(r(s))+rms(p(s)))",
        "velocity_score": "2*integral(v_ref(u)*v_pred(u))/integral(v_ref(u)^2+v_pred(u)^2)",
        "aggregation": "sqrt(path_score * velocity_score)",
        "scored_components": ["trajectory_shape", "speed_curve"],
        "path_parameter": "normalized cumulative arclength; relative positions in metres; no fitted scale or rotation",
        "speed_parameter": "normalized full native duration; speed remains metres per second",
        "integration": "exact piecewise integration on the union of all source knots",
        "timeline": "all supplied trajectory samples and all leading, internal and trailing pauses",
        "alignment": "subtract each initial position only; preserve shared registered axes and metric extent; no DTW",
        "missing_observations": "score the complete supplied optimizer plan; visibility is diagnostic only and never a multiplier",
        "observation_support": "adjacent source intervals with both endpoints observed; unobserved optimizer poses remain reconstructed estimates",
        "zero_motion": "both stationary=1; exactly one stationary=0",
        "orientation_evaluated": False,
        "smoothing": "none",
    }


def current_metric_config_sha256():
    return hashlib.sha256(json.dumps(current_metric_config(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
