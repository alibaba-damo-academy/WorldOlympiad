import pytest

from autoresearch.robotics_scene_objects import articulated_component_links


def test_articulated_semantic_label_includes_only_fixed_descendants(tmp_path):
    sdf = tmp_path / "cabinet.sdf"
    sdf.write_text("""<sdf version="1.9"><model name="cabinet">
      <link name="body"/><link name="door"/><link name="handle"/>
      <link name="loose_panel"/>
      <joint name="hinge" type="revolute"><parent>body</parent><child>door</child></joint>
      <joint name="grip" type="fixed"><parent>door</parent><child>handle</child></joint>
      <joint name="other" type="fixed"><parent>body</parent><child>loose_panel</child></joint>
    </model></sdf>""")
    assert articulated_component_links(sdf, "door") == ["door", "handle"]
    with pytest.raises(ValueError, match="absent"):
        articulated_component_links(sdf, "missing")
