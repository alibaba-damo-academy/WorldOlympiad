"""Native articulation predicates survive legacy scene compilation."""
import pytest

from wo.spec import Articulation, Predicate, Step, TaskObject, TaskSpec
from wo.tasks.scene import legacy_task


@pytest.mark.parametrize("joint_type,goal,operation,kind", [
    ("revolute", "joint_open", "pull", "door_open"),
    ("prismatic", "joint_open", "pull", "drawer_open"),
    ("revolute", "joint_closed", "close", "closed"),
    ("prismatic", "pressed", "press", "pressed"),
])
def test_articulation_mapping(joint_type, goal, operation, kind):
    spec = TaskSpec(
        case_id="articulation_mapping", family="articulation", instruction="Move the component.",
        objects=(TaskObject("table", "table", "support", "80 x 60 cm table"),
                 TaskObject("host", "cabinet", "articulated_host", "28 x 20 x 20 cm cabinet")),
        articulations=(Articulation("component", "host", joint_type),),
        steps=(Step("articulate", "component", Predicate(goal, "component")),),
    )
    task = legacy_task(spec)
    assert task["success_predicates"] == [{"kind": kind, "subject": "component", "object": "host"}]
    assert task["action_program"][0]["operation"] == operation
    assert task["action_program"][0]["target"] == "host"
