from types import SimpleNamespace
from autoresearch.robotics_scenecode_case_generator import _physical_category_tokens


def item(category, description, oid='prop'):
    return SimpleNamespace(category=category, visual_description=description, object_id=oid)


def test_role_labels_require_physical_nouns_not_color():
    assert _physical_category_tokens(item('distractor', 'purple cup 6 cm diameter')) == {'cup'}
    assert _physical_category_tokens(item('distractor', 'white cube 5 cm wide')) == {'block'}
    assert _physical_category_tokens(item('container', 'blue cup 8 cm tall')) == {'cup'}
    assert _physical_category_tokens(item('contents', 'three red beads')) == {'beads'}


def test_concrete_category_gate_and_unknown_are_preserved():
    assert _physical_category_tokens(item('bin', 'blue cube beside bin')) == {'bin'}
    assert _physical_category_tokens(item('container', 'blue cube')) == {'container'}
    assert _physical_category_tokens(item('distractor', 'purple 5 cm')) == {'distractor'}


def test_composite_members_use_shared_physical_category_gate():
    from autoresearch.robotics_scenecode_case_generator import _match_composite_asset_to_task_object
    declared = [item('container', 'blue cup 12 cm diameter', 'blue_cup'),
                item('container', 'green bowl 18 cm diameter', 'green_bowl'),
                item('contents', 'three red beads 3 cm diameter', 'red_beads')]
    match = _match_composite_asset_to_task_object
    assert match({'asset_id': 'cup_0', 'name': 'cup'}, declared) == 'blue_cup'
    assert match({'asset_id': 'red_bead_0', 'name': 'red_bead'}, declared) == 'red_beads'
    assert match({'asset_id': 'bowl_0', 'name': 'bowl'}, declared) == 'green_bowl'
    assert match({'name': 'blue cube'}, declared) is None


def test_composite_member_ambiguity_stays_unresolved():
    from autoresearch.robotics_scenecode_case_generator import _match_composite_asset_to_task_object
    declared = [item('container', 'blue cup', 'blue_cup'),
                item('container', 'green cup', 'green_cup')]
    assert _match_composite_asset_to_task_object({'name': 'cup'}, declared) is None
