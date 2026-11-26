"""Environment collider shared by WO references and native Isaac policy runs."""


def add_reference_ground(stage):
    # World._build loads pybullet_data/plane.urdf: fixed box 200x200x10,
    # origin (0,0,-5), lateral friction 1. This is environment, not a task entity.
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    path = "/World/ReferenceGround"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -5))
    xform.AddScaleOp().Set(Gf.Vec3d(200, 200, 10))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    # Keep physics geometry hidden, as in the WorldIR collision layer.
    UsdGeom.Imageable(cube).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    material = UsdShade.Material.Define(stage, path + "Material")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.0)
    physics.CreateDynamicFrictionAttr(1.0)
    physics.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose="physics")
    return {"source": "wo.sim.world.World._build -> pybullet_data/plane.urdf",
            "prim_path": path, "size_m": [200, 200, 10], "position_m": [0, 0, -5],
            "top_z_m": 0.0, "lateral_friction": 1.0, "task_entity": False}
