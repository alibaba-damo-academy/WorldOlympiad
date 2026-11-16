"""Bind world-baked CodeObject part meshes to their declared URDF link frames.

Only image/code assets with matching authoritative per-part glTF exports are
accepted. Joint topology, origins, axes, limits and mesh bytes stay unchanged.
"""
from __future__ import annotations
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .io import sha256_file
from .scenecode_mesh_frame_conversion import verified_mesh_basis, Y_UP_TO_Z_UP


def urdf_closed_link_frames(root):
    names={node.get('name') for node in root.findall('link')}
    joints=root.findall('joint')
    children={j.find('child').get('link') for j in joints}
    roots=names-children
    if len(roots)!=1:raise ValueError('CodeObject URDF requires one connected root')
    frames={next(iter(roots)):np.eye(4)}
    remaining=list(joints)
    while remaining:
        progress=False
        for joint in list(remaining):
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent not in frames:continue
            joint_type=joint.get('type')
            if joint_type not in {'fixed','revolute','prismatic','continuous'}:
                raise ValueError(f"unsupported CodeObject joint type {joint_type!r}: {joint.get('name')}")
            limit=joint.find('limit')
            if joint_type != 'continuous' and limit is not None and not float(limit.get('lower'))<=0<=float(limit.get('upper')):
                raise ValueError('CodeObject assembled mesh must represent zero-coordinate closed pose')
            origin=joint.find('origin');transform=np.eye(4)
            if origin is not None:
                transform[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                transform[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            frames[child]=frames[parent]@transform;remaining.remove(joint);progress=True
        if not progress:raise ValueError('CodeObject URDF link graph is cyclic or disconnected')
    if set(frames)!=names:raise ValueError('CodeObject URDF has disconnected link geometry')
    return frames


def align_world_baked_mesh_frames(urdf_path, part_mesh_dir, output_path):
    source=Path(urdf_path).resolve();parts=Path(part_mesh_dir).resolve();output=Path(output_path).resolve()
    if output.exists():raise FileExistsError('URDF frame derivative is immutable')
    tree=ET.parse(source);root=tree.getroot();frames=urdf_closed_link_frames(root)
    joints_before=[ET.tostring(j) for j in root.findall('joint')];records=[]
    for link in root.findall('link'):
        link_name=link.get('name')
        for kind in ['visual','collision']:
            for node in link.findall(kind):
                mesh=node.find('geometry/mesh')
                if mesh is None:raise ValueError('CodeObject frame repair requires per-part mesh evidence')
                declared=(source.parent/mesh.get('filename')).resolve()
                reference=parts/(declared.stem+'.gltf')
                if not reference.is_file():raise ValueError('authoritative CodeObject part export missing: '+declared.stem)
                if not np.allclose(np.fromstring(mesh.get('scale','1 1 1'),sep=' '),1):
                    raise ValueError('CodeObject mesh-frame repair may not change non-unit mesh scale')
                basis=verified_mesh_basis(reference,declared)
                destination_basis=Y_UP_TO_Z_UP if declared.suffix.lower() in {'.gltf','.glb'} else np.eye(3)
                rotation=np.asarray(basis['basis_matrix'])@destination_basis
                desired=np.eye(4);desired[:3,:3]=rotation
                local=np.linalg.inv(frames[link_name])@desired
                origin=node.find('origin')
                previous=dict(origin.attrib) if origin is not None else None
                if origin is None:origin=ET.SubElement(node,'origin')
                origin.set('xyz',' '.join(format(v,'.17g') for v in local[:3,3]))
                rpy=Rotation.from_matrix(local[:3,:3]).as_euler('xyz')
                origin.set('rpy',' '.join(format(v,'.17g') for v in rpy))
                records.append({'link':link_name,'kind':kind,'before':previous,'after':dict(origin.attrib),
                    'closed_link_frame_world':frames[link_name].tolist(),'mesh_basis':basis})
    if joints_before!=[ET.tostring(j) for j in root.findall('joint')]:
        raise RuntimeError('mesh-frame alignment changed declared mechanism')
    tree.write(output,encoding='utf-8',xml_declaration=True)
    receipt={'schema_version':'code_object_urdf_mesh_frames.v1','source_urdf':str(source),
        'source_urdf_sha256':sha256_file(source),'output_urdf':str(output),'output_urdf_sha256':sha256_file(output),
        'joint_topology_origins_axes_limits_unchanged':True,'mesh_bytes_unchanged':True,
        'authority':'checked CodeObject per-part world-baked glTF at assembled zero-coordinate pose',
        'mesh_origin_bindings':records,'physical_reference_success':False}
    output.with_suffix('.frame_alignment.json').write_text(json.dumps(receipt,indent=2))
    return receipt
