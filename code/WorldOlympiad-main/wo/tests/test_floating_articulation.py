"""CPU contract tests; these do not claim native PhysX execution evidence."""
import sys
import types

import pytest

from autoresearch.isaac_articulation import (
    author_floating_articulation, fixed_tree_root, floating_tree_root,
)


def body():
    return {'name': 'bottle', 'prim_name': 'bottle', 'static': False,
            'links': [{'name': n, 'prim_name': n} for n in ('cap', 'base')],
            'joints': [{'parent': 'base', 'child': 'cap', 'type': 'prismatic'}]}


def test_root_is_topological_not_first_link_and_never_fixed():
    b = body()
    assert floating_tree_root(b) == 'base'
    with pytest.raises(ValueError, match='static host'):
        fixed_tree_root(b)
    b['static'] = True
    assert fixed_tree_root(b) == 'base'
    with pytest.raises(ValueError, match='dynamic host'):
        floating_tree_root(b)


@pytest.mark.parametrize('variant', ['fixed', 'cycle', 'disconnected', 'duplicate', 'unknown'])
def test_invalid_or_uncollapsed_tree_fails_closed(variant):
    b = body()
    if variant == 'fixed':
        b['joints'][0]['type'] = 'fixed'
    elif variant == 'cycle':
        b['joints'].append({'parent': 'cap', 'child': 'base', 'type': 'revolute'})
    elif variant == 'disconnected':
        b['links'].append({'name': 'loose', 'prim_name': 'loose'})
    elif variant == 'duplicate':
        b['links'].append(b['links'][0])
    else:
        b['joints'][0]['parent'] = 'missing'
    with pytest.raises(ValueError):
        floating_tree_root(b)


class Prim:
    def __init__(self, *apis, valid=True):
        self.apis = set(apis)
        self.valid = valid

    def IsValid(self):
        return self.valid

    def HasAPI(self, api):
        return api in self.apis


class RootAPI:
    @staticmethod
    def Apply(prim):
        prim.apis.add(RootAPI)


def fake_stage(monkeypatch):
    rigid, mass = object(), object()
    physics = types.SimpleNamespace(RigidBodyAPI=rigid, MassAPI=mass,
                                    ArticulationRootAPI=RootAPI)
    monkeypatch.setitem(sys.modules, 'pxr', types.SimpleNamespace(UsdPhysics=physics))
    prims = {'/World/Scene/bottle': Prim(),
             '/World/Scene/bottle/base': Prim(rigid, mass),
             '/World/Scene/bottle/cap': Prim(rigid, mass)}
    stage = types.SimpleNamespace(GetPrimAtPath=lambda p: prims.get(p, Prim(valid=False)))
    return stage, prims, rigid, mass


def test_reopened_floating_validation_does_not_mutate(monkeypatch):
    from autoresearch.isaac_articulation import validate_floating_articulation
    stage, prims, _, _ = fake_stage(monkeypatch)
    author_floating_articulation(stage, body())
    before = {p: set(prim.apis) for p, prim in prims.items()}
    validate_floating_articulation(stage, body())
    assert before == {p: set(prim.apis) for p, prim in prims.items()}
    prims['/World/Scene/bottle/cap'].apis.add(RootAPI)
    with pytest.raises(ValueError, match='root identity'):
        validate_floating_articulation(stage, body())
    prims['/World/Scene/bottle/cap'].apis.remove(RootAPI)
    prims['/World/Scene/bottle/Joints/RootToWorld'] = Prim()
    with pytest.raises(ValueError, match='world anchor'):
        validate_floating_articulation(stage, body())


def test_floating_authors_only_root_no_world_anchor(monkeypatch):
    stage, prims, _, _ = fake_stage(monkeypatch)
    evidence = author_floating_articulation(stage, body())
    assert evidence['root_joint'] is None
    assert evidence['root_link'] == '/World/Scene/bottle/base'
    assert prims[evidence['root_link']].HasAPI(RootAPI)
    assert not prims['/World/Scene/bottle'].HasAPI(RootAPI)
    assert not prims['/World/Scene/bottle/cap'].HasAPI(RootAPI)
    assert len(prims) == 3


@pytest.mark.parametrize('failure', ['parent_rigid', 'missing_mass', 'missing_link'])
def test_floating_rejects_nested_or_unmeasured_rigid_links(monkeypatch, failure):
    stage, prims, rigid, mass = fake_stage(monkeypatch)
    if failure == 'parent_rigid':
        prims['/World/Scene/bottle'].apis.add(rigid)
    elif failure == 'missing_mass':
        prims['/World/Scene/bottle/base'].apis.remove(mass)
    else:
        del prims['/World/Scene/bottle/cap']
    with pytest.raises(ValueError):
        author_floating_articulation(stage, body())
    assert not prims['/World/Scene/bottle/base'].HasAPI(RootAPI)
