import xml.etree.ElementTree as ET
import pytest
from wo.models.cosmos import _archive_expert_physics
from autoresearch.robotics_runtime_sdf_rebind import bind_runtime_physics_sdfs


def test_publishes_actual_expert_urdf_not_source_physics(tmp_path):
    sdf = tmp_path / 'source.sdf'
    sdf.write_text('<sdf><model><link name="different_source_physics"/></model></sdf>')
    expert = tmp_path / 'expert.urdf'
    expert.write_text('<robot name="b"><link name="base"><inertial><mass value=".4"/>'
                      '</inertial></link><link name="cap"/>'
                      '<joint name="slide" type="prismatic"><parent link="base"/>'
                      '<child link="cap"/><axis xyz="0 0 1"/>'
                      '<limit lower="0" upper=".02" effort="2" velocity="1"/>'
                      '</joint></robot>')
    destination = tmp_path / 'new_bundle' / 'assets' / 'b'
    record = _archive_expert_physics(sdf, expert.with_suffix('.sdf'), destination, 'b')
    assert record == {'source_sdf': 'assets/b/source.sdf',
                      'physics_urdf': 'assets/b/physics_source.urdf'}
    result = ET.parse(destination / 'physics_source.urdf')
    assert ET.tostring(result.find('.//joint')) == ET.tostring(ET.parse(expert).find('.//joint'))
    assert result.find('.//inertial/mass').get('value') == '.4'
    assert 'physics_sdf' not in record


def test_existing_sdf_expert_stays_authoritative(tmp_path):
    sdf = tmp_path / 'source.sdf'
    sdf.write_text('<sdf><model><link name="base"/></model></sdf>')
    phys = tmp_path / 'expert.sdf'
    phys.write_text('<sdf><model><link name="expert_base"/></model></sdf>')
    destination = tmp_path / 'out'
    destination.mkdir()
    record = _archive_expert_physics(sdf, phys, destination, 'b')
    assert record['physics_sdf'] == 'assets/b/physics_source.sdf'
    assert 'physics_urdf' not in record
    assert ET.parse(destination / 'physics_source.sdf').find('.//link').get('name') == 'expert_base'


def test_missing_expert_physics_never_uses_source(tmp_path):
    sdf = tmp_path / 'source.sdf'
    sdf.write_text('<sdf><model><link/></model></sdf>')
    with pytest.raises(FileNotFoundError, match='missing expert physics'):
        _archive_expert_physics(sdf, tmp_path / 'missing.sdf', tmp_path / 'out', 'b')


def test_urdf_record_cannot_silently_rebind_source_sdf_physics(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='substitution is forbidden'):
        bind_runtime_physics_sdfs(manifest, {'b': {'source_sdf': 'source.sdf',
                                  'physics_urdf': 'expert.urdf'}},
                                  tmp_path.parent / (tmp_path.name + '_runtime'), {})
