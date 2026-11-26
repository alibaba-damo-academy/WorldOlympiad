"""Resolve the actual native contact target without inventing a receptacle."""


def contact_target_body(plan, graph):
    bodies = plan['bodies']
    trays = [body for body in bodies if body['name'] == 'tray']
    if trays:
        return trays[0]
    # Unary orientation tasks still have an explicit physical support in their
    # compiled on-region goal. Use that SAME body for the legacy contact slot.
    goals = graph.get('success_program') or []
    targets = {goal.get('object') for goal in goals}
    if not goals or len(targets) != 1 or None in targets:
        raise ValueError('native contact target is not uniquely declared')
    target = next(iter(targets))
    matches = [body for body in bodies if
               (body.get('metadata') or {}).get('task_object_id', body['name']) == target]
    if len(matches) != 1 or not matches[0].get('static'):
        raise ValueError('tray-free native contact target must be one actual static body')
    if any(goal.get('predicate') not in {'on', 'on_region', 'stacked'} for goal in goals):
        raise ValueError('tray-free contact target requires a support goal')
    return matches[0]
