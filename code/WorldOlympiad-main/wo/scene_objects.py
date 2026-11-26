"""Normalize SceneCode physical occurrences without changing their world poses.

Composite member transforms are already world-frame transforms. Keep the raw
transform and internal_model_pose separate, just as for ordinary scene objects;
the canonical importer composes these once to obtain the base-link world pose,
matching SceneCode's deserialize_composite_member_effective_transform.
"""
from copy import deepcopy
from typing import Any


def physical_objects(objects: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Expand stack/pile/filled-container records into physical occurrences.

    Asset IDs identify source assets, not occurrences: a composite can use an
    asset also placed independently, or use it repeatedly. Qualified occurrence
    IDs are shared by inventory and importer. Unknown composites remain intact.
    No parent's pose/bounding box/description is inherited by its members.
    """
    result: dict[str, dict[str, Any]] = {}
    reserved = {str(key) for key in objects}
    for key, record in objects.items():
        if not isinstance(record, dict):
            continue
        parent = str(key)
        metadata = record.get('metadata') or {}
        kind = metadata.get('composite_type')
        if kind in ('stack', 'pile'):
            members = list(metadata.get('member_assets') or [])
        elif kind == 'filled_container':
            container = metadata.get('container_asset')
            members = ([container] if container else []) + list(metadata.get('fill_assets') or [])
        else:
            result[parent] = deepcopy(record)
            continue
        for index, member in enumerate(members):
            if not isinstance(member, dict):
                raise ValueError(f'Invalid composite member {parent}[{index}]')
            transform = member.get('transform') or {}
            if 'translation' not in transform or 'rotation_wxyz' not in transform:
                raise ValueError(f'Composite member {parent}[{index}] lacks world transform')
            source_id = str(member.get('asset_id', f'member_{index}'))
            occurrence = f'{parent}::member_{index}::{source_id}'
            if occurrence in reserved or occurrence in result:
                raise ValueError(f'Composite occurrence ID collision: {occurrence}')
            item = deepcopy(member)
            item['object_id'] = occurrence
            item.setdefault('object_type', record.get('object_type', 'manipuland'))
            item['metadata'] = {
                **(item.get('metadata') or {}),
                'composite_parent_id': parent,
                'composite_member_index': index,
                'source_asset_id': source_id,
            }
            result[occurrence] = item
    return result
