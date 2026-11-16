from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import os
import pytest
pytest.importorskip("pxr")
pytest.importorskip("pxr")
pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from wo.models.reference_environment import add_reference_ground


def test_native_environment_matches_actual_reference_plane_geometry_and_material():
    path = os.environ.get("WO_REFERENCE_PLANE_URDF")
    if path is None:
        pybullet_data = pytest.importorskip("pybullet_data")
        path = str(Path(pybullet_data.getDataPath()) / "plane.urdf")
    source = ET.parse(path).getroot()
    collision = source.find("link/collision")
    size = np.array([float(v) for v in collision.find("geometry/box").get("size").split()])
    position = np.array([float(v) for v in collision.find("origin").get("xyz").split()])
    friction = float(source.find("link/contact/lateral_friction").get("value"))
    stage = Usd.Stage.CreateInMemory()
    evidence = add_reference_ground(stage)
    prim = stage.GetPrimAtPath(evidence["prim_path"])
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], False, True)
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    assert np.asarray(bounds.GetMin()) == pytest.approx(position-size/2)
    assert np.asarray(bounds.GetMax()) == pytest.approx(position+size/2)
    assert prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
    assert UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get() == friction
