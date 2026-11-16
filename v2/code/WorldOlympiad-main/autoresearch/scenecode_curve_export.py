"""Preserve evaluated curve surfaces at SceneCode's mesh-only export boundary."""
from __future__ import annotations


def prepare_curve_surfaces_for_export():
    import bpy
    import json
    import numpy as np
    graph=bpy.context.evaluated_depsgraph_get()
    candidates=[obj for obj in bpy.context.scene.objects if obj.type in {'CURVE','SURFACE','FONT'}]
    records=[]
    for obj in candidates:
        name=obj.name
        evaluated=obj.evaluated_get(graph)
        mesh=evaluated.to_mesh()
        try:
            if mesh is None or not mesh.polygons:
                raise ValueError('curve export has no evaluated surface: '+name)
            before=np.asarray([tuple(evaluated.matrix_world@v.co) for v in mesh.vertices])
            faces=len(mesh.polygons)
            materials=[m.name if m else None for m in mesh.materials]
        finally:
            evaluated.to_mesh_clear()
        transform=obj.matrix_world.copy()
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True);bpy.context.view_layer.objects.active=obj
        bpy.ops.object.convert(target='MESH')
        converted=bpy.context.view_layer.objects.active
        converted.matrix_world=transform
        bpy.context.view_layer.update()
        after=np.asarray([tuple(converted.matrix_world@v.co) for v in converted.data.vertices])
        after_materials=[m.name if m else None for m in converted.data.materials]
        if (converted.name!=name or before.shape!=after.shape or len(converted.data.polygons)!=faces
                or materials!=after_materials or not np.allclose(before,after,rtol=0,atol=1e-6)):
            raise ValueError('curve surface conversion changed geometry/materials: '+name)
        records.append({'name':name,'vertices':len(after),'faces':faces,'material_names':materials,
                        'max_world_vertex_error_m':float(np.abs(before-after).max())})
    if records:
        print('CURVE_EXPORT_EVIDENCE:'+json.dumps({'schema_version':'scenecode_curve_surface_export.v1','objects':records},sort_keys=True))
    return records


def curve_surface_export_code():
    """Embed the small adapter; isolated Blender has no project imports."""
    import inspect
    return inspect.getsource(prepare_curve_surfaces_for_export)+"\nprepare_curve_surfaces_for_export()\n"
