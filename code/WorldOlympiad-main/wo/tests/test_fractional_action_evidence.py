"""Action evidence must reproduce fractional gripper commands exactly."""
from types import SimpleNamespace
import numpy as np
import pytest
from wo.sim.world import World, SUBSTEPS


@pytest.mark.parametrize('gripper,expected', [(0., 0.), (.25, .25), (2/3, 2/3), (1., 1.), (-1., 0.), (2., 1.)])
def test_act_records_bounded_fraction_not_binary(gripper, expected):
    commands, recorded, physics = [], [], []
    world = SimpleNamespace(lower=np.full(7, -1.), upper=np.full(7, 1.),
                            command=lambda q, g: commands.append((q, g)),
                            step_physics=physics.append,
                            record=lambda **kw: recorded.append(kw['commanded']))
    World.act(world, [2.] * 7 + [gripper])
    assert recorded == [[1.] * 7 + [expected]]
    assert physics == [SUBSTEPS]
    assert commands[0][1] == gripper


def test_record_false_does_not_emit_action_evidence():
    world = SimpleNamespace(lower=np.full(7, -1.), upper=np.full(7, 1.),
                            command=lambda q, g: None, step_physics=lambda n: None,
                            record=lambda **kw: pytest.fail('unexpected recording'))
    World.act(world, [0.] * 7 + [.25], record=False)
