import json
from pathlib import Path
import pytest
from autoresearch.io import sha256_file
from autoresearch.articulated_reference_replay_binding import physical_signature, validate_reference_binding


def prepare(tmp_path):
    old=tmp_path/'simulation_assets/source_sdf/host';old.mkdir(parents=True)
    mesh=old/'collision.obj';mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    original=old/'host.sdf'
    original.write_text('<sdf version="1.7"><model name="host"><link name="door"><inertial><mass>2</mass></inertial><collision name="c"><geometry><mesh><uri>collision.obj</uri></mesh></geometry></collision><visual name="v"><material>old-color</material></visual></link><joint name="hinge" type="revolute"><axis><xyz>1 0 0</xyz><limit><upper>1.57</upper></limit></axis></joint></model></sdf>')
    native=tmp_path/'native.sdf'
    native.write_text(original.read_text().replace('collision.obj',str(mesh)).replace('old-color','new-color'))
    cert=tmp_path/'alignment.json';manifest=tmp_path/'ref.json'
    def bind():
        cert.write_text(json.dumps({'passed':True,'visual_sdf_sha256':sha256_file(native)}))
        manifest.write_text(json.dumps({'visual_physics_alignment_certificate':cert.name,'visual_physics_alignment_certificate_sha256':sha256_file(cert)}))
        return {'source_sdf':str(native),'source_sdf_sha256':sha256_file(native),'reference_manifest':str(manifest),
                'reference_manifest_sha256':sha256_file(manifest),'runtime_alignment_certificate_sha256':sha256_file(cert)}
    return original,native,bind,{'binding':{'host_sdf_sha256':sha256_file(original)}}


def test_visual_revision_keeps_collision_inertia_and_joint_identity(tmp_path):
    original,native,bind,ref=prepare(tmp_path)
    assert physical_signature(original)==physical_signature(native)
    assert validate_reference_binding(ref,bind())['passed'] is True


@pytest.mark.parametrize('old,new',[('<mass>2</mass>','<mass>3</mass>'),('<upper>1.57</upper>','<upper>1.0</upper>')])
def test_visual_certificate_cannot_authorize_a_physics_change(tmp_path,old,new):
    original,native,bind,ref=prepare(tmp_path)
    native.write_text(native.read_text().replace(old,new))
    with pytest.raises(ValueError,match='physics differs'):validate_reference_binding(ref,bind())


def test_different_collision_geometry_is_rejected(tmp_path):
    original,native,bind,ref=prepare(tmp_path)
    other=tmp_path/'other.obj';other.write_text('different geometry')
    native.write_text(native.read_text().replace(str(original.parent/'collision.obj'),str(other)))
    with pytest.raises(ValueError,match='physics differs'):validate_reference_binding(ref,bind())
