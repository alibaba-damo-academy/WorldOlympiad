from pathlib import Path

import numpy as np
import pytest

from scripts.build_cup_to_tray_worldir import (
    _bounded_inner_proxy_evidence,
    _geometry_bounds,
)


def test_axis_audit_accepts_pose_roundoff_but_rejects_real_rotation() -> None:
    geometry = {
        "kind": "box", "size": [1.0, 2.0, 3.0],
        "pose": {"xyz": [0.0, 0.0, 0.0],
                 "quat_xyzw": [-1.339744823415e-8, 0.0, 0.0, 1.0]},
    }
    low, high = _geometry_bounds(Path("/tmp"), geometry)
    assert np.allclose(low, [-0.5, -1.0, -1.5])
    assert np.allclose(high, [0.5, 1.0, 1.5])
    geometry["pose"]["quat_xyzw"] = [0.1, 0.0, 0.0, 0.9949874371]
    with pytest.raises(ValueError, match="axis-aligned"):
        _geometry_bounds(Path("/tmp"), geometry)


def test_explicit_inner_collision_proxy_needs_geometry_enclosure() -> None:
    visual = (np.array([-.045, -.045, 0.0]),
              np.array([.045, .045, .254]))
    collision = (np.array([-.0302, -.0302, 0.0]),
                 np.array([.0302, .0302, .2406]))
    declaration = {
        "policy": "explicit_backend_collision_proxy",
        "compiled_proxy_count": 1,
        "source_worldir_collision_count": 41,
        "compiled_proxies": [{"kind": "cylinder", "radius": .0302,
                              "length": .2406}],
    }
    assert _bounded_inner_proxy_evidence(visual, collision, declaration)["passed"]
    shifted = (collision[0] + np.array([.04, 0.0, 0.0]),
               collision[1] + np.array([.04, 0.0, 0.0]))
    assert not _bounded_inner_proxy_evidence(visual, shifted, declaration)["passed"]
    wider = dict(declaration)
    wider["compiled_proxies"] = [{"kind": "cylinder", "radius": .02,
                                  "length": .2406}]
    assert not _bounded_inner_proxy_evidence(visual, collision, wider)["passed"]
