import pytest

from autoresearch.physical_apparatus import apparatus


def test_dominoes_are_six_separate_dynamic_bodies(tmp_path):
    bodies,layout,trigger,_=apparatus('domino_chain',tmp_path)
    assert len({b.name for b in bodies})==6
    assert all(not b.static and not b.joints for b in bodies)
    assert all(layout[bodies[i+1].name][0]>layout[bodies[i].name][0] for i in range(5))
    assert trigger['body']==bodies[0].name
    assert trigger['end_s']>trigger['start_s']


def test_containment_has_a_bottom_four_walls_and_one_free_ball(tmp_path):
    bodies,layout,_,_=apparatus('shallow_containment',tmp_path)
    ball=next(b for b in bodies if b.name=='subject')
    assert len([b for b in bodies if b.static])==5
    assert ball.links[0].collisions[0].radius==pytest.approx(.04)
    bottom=next(b for b in bodies if b.name=='tray_bottom').links[0].collisions[0]
    wall=next(b for b in bodies if b.name=='tray_front').links[0].collisions[0]
    assert 2*ball.links[0].collisions[0].radius < bottom.size[1]-2*wall.size[1]
    assert layout['subject'][2]>.04


def test_collision_chain_has_one_external_velocity_trigger(tmp_path):
    bodies,layout,trigger,_=apparatus('collision_chain',tmp_path)
    assert len(bodies)==3
    assert trigger['body']=='subject'
    assert trigger['velocity'][0]>0
    assert layout['subject'][0]<layout['ball_blue'][0]<layout['ball_yellow'][0]
