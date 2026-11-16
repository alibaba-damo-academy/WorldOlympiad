"""Author fixed-base reduced-coordinate mechanisms from a compiled body tree."""
from __future__ import annotations


def mechanism_tree_root(body: dict) -> str:
    """Validate a reduced-coordinate tree without imposing a fixed base.

    Fixed subtrees must already have been merged by the lossless WorldIR
    collapse pass. No movable joint may be discarded to enter this path.
    """
    names = {link['name'] for link in body['links']}
    if len(names) != len(body['links']):
        raise ValueError('duplicate articulation link names')
    children = set()
    edges = {name: [] for name in names}
    if not body.get('joints'):
        raise ValueError('articulation requires joints')
    for joint in body['joints']:
        parent, child = joint['parent'], joint['child']
        if joint['type'] not in {'revolute', 'prismatic', 'continuous'}:
            raise ValueError('this fixed articulation path requires movable tree joints')
        if parent not in names or child not in names or child in children:
            raise ValueError('invalid articulation tree topology')
        children.add(child)
        edges[parent].append(child)
    roots = names - children
    if len(roots) != 1:
        raise ValueError('fixed articulation needs exactly one root link')
    root = next(iter(roots))
    visited = set()
    def visit(name):
        if name in visited:
            raise ValueError('articulation contains a cycle')
        visited.add(name)
        for child in edges[name]:
            visit(child)
    visit(root)
    if visited != names:
        raise ValueError('articulation is disconnected')
    return root


def fixed_tree_root(body: dict) -> str:
    if not body.get('static') or not body.get('joints'):
        raise ValueError('fixed articulation requires a static host with joints')
    return mechanism_tree_root(body)


def floating_tree_root(body: dict) -> str:
    if body.get('static') or not body.get('joints'):
        raise ValueError('floating articulation requires a dynamic host with joints')
    return mechanism_tree_root(body)


def author_floating_articulation(stage, body: dict) -> dict:
    """Author a freely moving base, never a world anchor or parent rigid body.

    Called only after the compiler authors each link with its measured inertia
    and every movable joint. Parent transforms must NOT be rigid: nesting root
    and child rigid bodies underneath another rigid body is invalid in PhysX.
    This function deliberately refuses that layout instead of repairing it by
    deleting physics APIs (which could silently discard measurements).
    """
    from pxr import UsdPhysics
    root_name = floating_tree_root(body)
    body_path = f"/World/Scene/{body['prim_name']}"
    root_link = next(link for link in body['links'] if link['name'] == root_name)
    root_path = f"{body_path}/{root_link['prim_name']}"
    parent = stage.GetPrimAtPath(body_path)
    if not parent.IsValid() or parent.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('floating articulation parent must be a non-rigid transform')
    for link in body['links']:
        prim = stage.GetPrimAtPath(f"{body_path}/{link['prim_name']}")
        if (not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or not prim.HasAPI(UsdPhysics.MassAPI)):
            raise ValueError('floating articulation requires measured rigid links')
    prim = stage.GetPrimAtPath(root_path)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    return {'body': body['name'], 'root_link': root_path,
            'root_joint': None, 'representation': 'floating_base_reduced_coordinates'}


def validate_floating_articulation(stage, body: dict) -> None:
    """Check reopened layout without mutating or repairing the persisted USD."""
    from pxr import UsdPhysics
    root_name = floating_tree_root(body)
    parent_path = f"/World/Scene/{body['prim_name']}"
    parent = stage.GetPrimAtPath(parent_path)
    if (not parent.IsValid() or parent.HasAPI(UsdPhysics.RigidBodyAPI)
            or parent.HasAPI(UsdPhysics.ArticulationRootAPI)):
        raise ValueError('floating parent must remain non-rigid and non-root')
    for link in body['links']:
        prim = stage.GetPrimAtPath(f"{parent_path}/{link['prim_name']}")
        if (not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or not prim.HasAPI(UsdPhysics.MassAPI)):
            raise ValueError('floating link lost measured rigid physics')
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) != (link['name'] == root_name):
            raise ValueError('floating articulation root identity changed')
    anchor = stage.GetPrimAtPath(parent_path + '/Joints/RootToWorld')
    if anchor.IsValid():
        raise ValueError('floating articulation must not have a world anchor')


def author_fixed_articulation(stage, body: dict) -> dict:
    from pxr import UsdPhysics
    root_name = fixed_tree_root(body)
    root_link = next(link for link in body['links'] if link['name'] == root_name)
    body_path = f"/World/Scene/{body['prim_name']}"
    root_path = f"{body_path}/{root_link['prim_name']}"
    prim = stage.GetPrimAtPath(root_path)
    if not prim.IsValid():
        raise ValueError('articulation root link was not authored')
    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    # A fixed-base articulation uses the root link's authored pose. Per NVIDIA
    # documentation, joint frames on its world anchor are ignored.
    joint_path = body_path + '/Joints/RootToWorld'
    anchor = UsdPhysics.FixedJoint.Define(stage, joint_path)
    anchor.CreateBody0Rel().SetTargets([root_path])
    UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
    return {'body': body['name'], 'root_link': root_path,
            'root_joint': joint_path, 'representation': 'fixed_base_reduced_coordinates'}
