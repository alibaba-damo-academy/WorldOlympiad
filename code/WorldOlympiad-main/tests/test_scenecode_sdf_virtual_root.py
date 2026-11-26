import xml.etree.ElementTree as ET
import pytest
from autoresearch.scenecode_sdf_virtual_root import normalize_virtual_roots, read_sdf_preserving_namespaces


def model(joint_type="fixed"):
    return ET.fromstring(f'<model name="object" canonical_link="assembly"><link name="assembly"><inertial><mass>0</mass></inertial></link><link name="body"><pose relative_to="assembly">0 0 .4 0 0 0</pose><inertial><mass>.06</mass></inertial></link><joint name="attachment" type="{joint_type}"><parent>assembly</parent><child>body</child></joint></model>')


def test_empty_root_becomes_a_frame_and_child_pose_stays_identical():
    root = model()
    pose = ET.tostring(root.find('link[@name="body"]/pose'))
    result = normalize_virtual_roots(root)
    assert result[0]['root'] == 'assembly'
    assert root.find('link[@name="assembly"]') is None
    assert root.find('frame[@name="assembly"]').get('attached_to') == 'body'
    assert ET.tostring(root.find('link[@name="body"]/pose')) == pose
    assert root.get('canonical_link') == 'body'
    assert normalize_virtual_roots(root) == []


@pytest.mark.parametrize('kind', ['revolute', 'physical'])
def test_physical_or_articulated_zero_mass_root_is_never_silently_discarded(kind):
    root = model('revolute' if kind == 'revolute' else 'fixed')
    if kind == 'physical':
        ET.SubElement(root.find('link[@name="assembly"]'), 'collision')
    with pytest.raises(ValueError, match='zero-mass'):
        normalize_virtual_roots(root)


def test_drake_collision_filter_namespace_and_real_members_are_preserved(tmp_path):
    path = tmp_path / 'asset.sdf'
    root = model()
    group = ET.SubElement(root, '{drake.mit.edu}collision_filter_group', name='self')
    for name in ['assembly', 'body']:
        ET.SubElement(group, '{drake.mit.edu}member').text = name
    sdf = ET.Element('sdf', version='1.7');sdf.append(root)
    ET.register_namespace('drake', 'drake.mit.edu')
    ET.ElementTree(sdf).write(path)
    tree = read_sdf_preserving_namespaces(path)
    normalize_virtual_roots(tree.getroot().find('model'))
    tree.write(path)
    assert b'<drake:collision_filter_group' in path.read_bytes()
    members = tree.getroot().findall('.//{drake.mit.edu}member')
    assert [item.text for item in members] == ['body']


def test_fixed_siblings_keep_geometry_and_articulated_descendant():
    root = model()
    ET.SubElement(root, 'link', name='panel').extend([
        ET.fromstring('<pose relative_to="assembly">.2 -.3 .1 .1 .2 .3</pose>'),
        ET.fromstring('<inertial><mass>.02</mass></inertial>')])
    ET.SubElement(root, 'link', name='door').extend([
        ET.fromstring('<pose relative_to="body">.1 0 0 0 0 0</pose>'),
        ET.fromstring('<inertial><mass>.01</mass></inertial>')])
    root.append(ET.fromstring('<joint name="panel_weld" type="fixed"><parent>assembly</parent><child>panel</child></joint>'))
    root.append(ET.fromstring('<joint name="hinge" type="revolute"><parent>body</parent><child>door</child><axis><xyz>0 0 1</xyz></axis></joint>'))
    links = {node.get('name'): ET.tostring(node) for node in root.findall('link') if node.get('name') != 'assembly'}
    hinge = ET.tostring(root.find('joint[@name="hinge"]'))
    result = normalize_virtual_roots(root)
    assert result[0]['reparented_fixed_joints'] == ['panel_weld']
    assert root.findtext('joint[@name="panel_weld"]/parent') == 'body'
    assert {node.get('name'): ET.tostring(node) for node in root.findall('link')} == links
    assert ET.tostring(root.find('joint[@name="hinge"]')) == hinge
    assert normalize_virtual_roots(root) == []

@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
def test_fixed_anchor_retains_sibling_motion_and_all_declared_frames(joint_type):
    root = model()
    root.append(ET.fromstring('<link name="door"><pose relative_to="assembly">.2 .1 .3 .1 .2 .3</pose><inertial><mass>.02</mass></inertial></link>'))
    moving = ET.fromstring(f'<joint name="motion" type="{joint_type}"><parent>assembly</parent><child>door</child><pose relative_to="assembly">.2 .1 .3 0 0 0</pose><axis><xyz expressed_in="assembly">0 1 0</xyz><limit><lower>0</lower><upper>.6</upper></limit></axis></joint>')
    root.insert(0, moving)  # The anchor need not be the first outgoing joint.
    links = {n.get("name"): ET.tostring(n) for n in root.findall("link") if n.get("name") != "assembly"}
    mechanism = {n.tag: ET.tostring(n) for n in moving if n.tag != "parent"}
    result = normalize_virtual_roots(root)
    assert result[0]["physical_child"] == "body"
    assert result[0]["removed_fixed_joint"] == "attachment"
    assert result[0]["reparented_moving_joints"] == ["motion"]
    assert moving.get("type") == joint_type
    assert moving.findtext("parent") == "body"
    assert {n.tag: ET.tostring(n) for n in moving if n.tag != "parent"} == mechanism
    assert {n.get("name"): ET.tostring(n) for n in root.findall("link")} == links
    assert normalize_virtual_roots(root) == []
