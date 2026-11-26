"""Represent an explicitly massless, empty assembly root as an SDF frame."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET


def read_sdf_preserving_namespaces(path):
    for _event, (prefix, uri) in ET.iterparse(Path(path), events=("start-ns",)):
        # Drake's importer requires this conventional prefix as well as its URI.
        if uri == "drake.mit.edu":
            prefix = "drake"
        ET.register_namespace(prefix, uri)
    return ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def normalize_virtual_roots(model):
    """Keep frame/pose semantics without importing a spurious default 1 kg body.

    An empty root welded to a positive-mass child has a physical anchor.
    Preserve the root as a frame on that child and reparent its other joints
    without changing their types, poses, axes, limits or child link poses.
    """
    removed = []
    for link in list(model.findall("link")):
        mass = link.findtext("inertial/mass")
        if mass is None or float(mass) != 0.0:
            continue
        name = link.get("name")
        joints = model.findall("joint")
        incoming = [joint for joint in joints if joint.findtext("child") == name]
        outgoing = [joint for joint in joints if joint.findtext("parent") == name]
        if incoming or link.find("visual") is not None or link.find("collision") is not None:
            raise ValueError("zero-mass physical or intermediate link needs explicit model repair")
        anchor = next((joint for joint in outgoing if joint.get("type") == "fixed"), None)
        if (anchor is None or any(joint.get("type") not in {"fixed", "revolute", "prismatic"} for joint in outgoing)
                or any(item.tag not in {"inertial", "pose"} for item in link)
                or any(item.tag not in {"parent", "child", "pose", "axis"} for joint in outgoing for item in joint)):
            raise ValueError("ambiguous zero-mass root cannot be replaced by a frame")
        child = anchor.findtext("child")
        for joint in outgoing:
            body = model.find(f'link[@name="{joint.findtext("child")}"]')
            if body is None or float(body.findtext("inertial/mass", "0")) <= 0:
                raise ValueError("virtual root has no positive-mass child body")
        old_pose = link.find("pose")
        if old_pose is not None and old_pose.get("relative_to") not in (None, "", "__model__"):
            raise ValueError("virtual root pose must be expressed in the model frame")
        frame = ET.Element("frame", name=name, attached_to=child)
        pose = copy.deepcopy(old_pose) if old_pose is not None else ET.Element("pose")
        pose.set("relative_to", "__model__")
        pose.text = pose.text or "0 0 0 0 0 0"
        frame.append(pose)
        index = list(model).index(link)
        model.remove(link)
        model.remove(anchor)
        reparented = []
        moving = []
        for joint in outgoing:
            if joint is anchor:
                continue
            # The joint pose remains in its original frame (default: child),
            # and every child link still resolves through the retained frame.
            # Root and anchor are welded, so the original motion is retained.
            joint.find("parent").text = child
            (reparented if joint.get("type") == "fixed" else moving).append(joint.get("name"))
        model.insert(index, frame)
        if model.get("canonical_link") == name:
            model.set("canonical_link", child)
        for parent in model.iter():
            for item in list(parent):
                if isinstance(item.tag, str) and item.tag.rsplit("}", 1)[-1] == "member" and (item.text or "").strip() == name:
                    parent.remove(item)
        removed.append({"root": name, "physical_child": child, "declared_root_mass": 0.0,
                        "representation": "frame", "removed_fixed_joint": anchor.get("name"),
                        "reparented_fixed_joints": reparented})
        if moving:
            removed[-1]["reparented_moving_joints"] = moving
    return removed
