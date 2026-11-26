import json

import pytest

from wo.tasks.scene import archive_transient_incomplete_recoveries


@pytest.mark.parametrize('missing,outage,stage,archived', [
    (True, True, 'task_asset_inventory', True),
    (False, True, 'task_asset_inventory', False),
    (True, False, 'task_asset_inventory', False),
    (True, True, 'other', False),
])
def test_only_missing_inventory_with_failed_service_conversion_is_archived(
        tmp_path, missing, outage, stage, archived):
    failure = tmp_path / 'robotics_scenecode_failures/b/c.json'
    failure.parent.mkdir(parents=True)
    failure.write_text(json.dumps({'failed_stage': stage, 'matches': [
        {'status': 'missing' if missing else 'matched'}]}))
    checkpoints = []
    for folder in ('robotics_scenecode_recoveries', 'robotics_scenecode_partial_recoveries'):
        checkpoint = tmp_path / folder / 'b/c'
        log = checkpoint / 'scene_run/scene_000/room_workspace/room.log'
        log.parent.mkdir(parents=True)
        log.write_text('VLM analysis failed: Error code: 503' if outage else
                       'unrelated HTTP 503; asset geometry invalid')
        checkpoints.append(checkpoint)
    archive_transient_incomplete_recoveries(tmp_path, 'b', 'c')
    for checkpoint in checkpoints:
        assert checkpoint.exists() is (not archived)
        archives = list(checkpoint.parent.glob('c.service_incomplete_*'))
        assert bool(archives) is archived
        if archived:
            assert list(archives[0].rglob('room.log'))[0].read_text().endswith('503')
    assert failure.exists()
