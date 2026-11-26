"""Utilities for creating ROS-style package.xml files for scene portability."""

from pathlib import Path


def create_package_xml(scene_dir: Path) -> None:
    """Create a package.xml file in the scene directory for Drake portability.

    This makes the scene directory a ROS-style package that can be discovered
    via ROS_PACKAGE_PATH. All SDF paths in directive files use package://scene/
    URIs, which Drake resolves using the package map.

    Usage with model_visualizer:
        export ROS_PACKAGE_PATH=/path/to/scene_dir:$ROS_PACKAGE_PATH
        python3 -m pydrake.visualization.model_visualizer house.dmd.yaml

    Usage programmatically:
        parser.package_map().Add("scene", str(scene_dir))

    Args:
        scene_dir: Path to the scene directory where package.xml will be created.
    """
    package_xml = """<?xml version="1.0"?>
<package format="2">
  <name>scene</name>
  <version>1.0.0</version>
  <description>Scene-agent generated scene</description>
  <maintainer email="noreply@example.com">scenecode</maintainer>
  <license>MIT</license>
</package>
"""
    (scene_dir / "package.xml").write_text(package_xml)
