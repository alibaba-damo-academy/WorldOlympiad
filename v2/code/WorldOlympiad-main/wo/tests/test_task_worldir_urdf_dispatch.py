import pytest
from autoresearch.models import Pose
from scripts.build_task_graph_worldir import bind_task_physics, import_task_urdf
from wo.tests.test_cosmos_urdf_archive import fixture


def test_urdf_only_binds_without_sdf_fallback(tmp_path):
    _, _, manifest, digests = fixture(tmp_path)
    paths = bind_task_physics(manifest, {'b': {'physics_urdf': 'b/physics.urdf',
        'source_sdf': 'does_not_exist.sdf'}}, tmp_path / 'runtime', digests,
        allow_subject_joint_lock=False)
    assert paths['b'].suffix == '.urdf'
    assert (tmp_path / 'runtime/urdf/urdf_asset_binding_receipt.json').exists()
    assert not (tmp_path / 'runtime/sdf').exists()


def test_ambiguous_authority_refused(tmp_path):
    with pytest.raises(ValueError, match='ambiguous'):
        bind_task_physics(tmp_path / 'manifest', {'b': {'physics_urdf': 'a',
            'physics_sdf': 'b'}}, tmp_path / 'r', {}, allow_subject_joint_lock=False)


def test_dynamic_measured_tree_import_preserves_movable_joint(tmp_path):
    inertia = '<inertial><mass value=".2"/><inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial>'
    geom = '<visual><geometry><box size=".03 .04 .05"/></geometry></visual><collision><geometry><box size=".03 .04 .05"/></geometry></collision>'
    path = tmp_path / 'body.urdf'
    path.write_text('<robot name="b">' + ''.join('<link name="'+n+'">'+inertia+geom+'</link>' for n in ['base','fixed','cap']) +
        '<joint name="f" type="fixed"><parent link="base"/><child link="fixed"/><origin xyz=".1 0 0"/></joint>' +
        '<joint name="slide" type="prismatic"><parent link="fixed"/><child link="cap"/><origin xyz="0 0 .1"/><axis xyz="0 0 1"/><limit lower="0" upper=".02" effort="2" velocity="1"/></joint></robot>')
    body = import_task_urdf(path, body_name='b', semantic_category='bottle', pose=Pose(), static=False,
        target=True, support='table', metadata={'task_object_id': 'b'})
    assert body.static is False and body.target and body.support == 'table'
    assert [j.name for j in body.joints] == ['slide']
    assert body.joints[0].parent == 'base'
    assert body.joints[0].upper == .02
    assert body.metadata['fixed_joint_collapse']['passed']
    assert body.metadata['task_object_id'] == 'b'
    assert sum(l.inertial.mass for l in body.links) == pytest.approx(.6)
    assert sum(len(l.visuals) for l in body.links) == 3
