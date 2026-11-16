import pytest
from autoresearch.isaac_articulation import fixed_tree_root


def test_branched_fixed_mechanism_has_one_root():
    body={'static':True,'links':[{'name':n} for n in ['base','door','drawer']],
          'joints':[{'parent':'base','child':'door','type':'revolute'},
                    {'parent':'base','child':'drawer','type':'prismatic'}]}
    assert fixed_tree_root(body)=='base'
    body['joints'].append({'parent':'door','child':'drawer','type':'revolute'})
    with pytest.raises(ValueError,match='topology'):
        fixed_tree_root(body)


def test_disconnected_cycle_is_not_a_valid_articulation():
    body={'static':True,'links':[{'name':n} for n in ['base','a','b']],
          'joints':[{'parent':'a','child':'b','type':'revolute'},
                    {'parent':'b','child':'a','type':'revolute'}]}
    with pytest.raises(ValueError,match='disconnected'):
        fixed_tree_root(body)
