def _wo_assign_blender_weight(value, owner):
    import bpy
    import math
    if isinstance(owner, bpy.types.WeightedNormalModifier) and isinstance(value, float):
        prop = owner.bl_rna.properties['weight']
        if not math.isfinite(value) or not value.is_integer() or (not prop.hard_min <= value <= prop.hard_max):
            raise TypeError('WeightedNormalModifier.weight requires an integer in [1, 100]; refusing lossy conversion: ' + repr(value))
        value = int(value)
        print('WorldOlympiad Blender API repair: exact integer conversion for WeightedNormalModifier.weight')
    setattr(owner, 'weight', value)
import bpy
import math

def _remove_object_if_present(name):
    """Remove an existing object with the requested name."""
    existing = bpy.data.objects.get(name)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)

def _rounded_rectangle_ring(width, depth, radius, z, segments_per_corner=8):
    """
    Generate a centered rounded-rectangle perimeter ring.

    The returned points are ordered around the perimeter and do not duplicate
    the corner transition points.
    """
    radius = min(radius, width * 0.5, depth * 0.5)
    points = []
    corners = [(width * 0.5 - radius, depth * 0.5 - radius, 0.0), (-width * 0.5 + radius, depth * 0.5 - radius, math.pi * 0.5), (-width * 0.5 + radius, -depth * 0.5 + radius, math.pi), (width * 0.5 - radius, -depth * 0.5 + radius, math.pi * 1.5)]
    for cx, cy, start_angle in corners:
        for i in range(segments_per_corner):
            angle = start_angle + math.pi * 0.5 * i / segments_per_corner
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle), z))
    return points

def _set_socket_value(socket, value):
    """Assign a shader socket value while respecting its dimensionality."""
    if socket is None:
        return
    try:
        socket.default_value = value
    except (TypeError, ValueError):
        if isinstance(value, tuple):
            try:
                socket.default_value = float(value[0])
            except (TypeError, ValueError):
                pass
        else:
            try:
                socket.default_value = (value, value, value, 1.0)
            except (TypeError, ValueError):
                pass

def _create_clear_plastic_material():
    """Create or update the reusable clear plastic material."""
    material_name = 'food_storage_box_body_material'
    mat = bpy.data.materials.get(material_name)
    if mat is None:
        mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    bsdf = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    output = next((node for node in nodes if node.type == 'OUTPUT_MATERIAL'), None)
    if output is None:
        output = nodes.new(type='ShaderNodeOutputMaterial')
    for link in list(links):
        links.remove(link)
    surface_socket = output.inputs.get('Surface')
    shader_socket = bsdf.outputs.get('BSDF')
    if surface_socket is not None and shader_socket is not None:
        links.new(shader_socket, surface_socket)
    _set_socket_value(bsdf.inputs.get('Base Color'), (0.93, 0.97, 1.0, 1.0))
    _set_socket_value(bsdf.inputs.get('Roughness'), 0.14)
    _set_socket_value(bsdf.inputs.get('Metallic'), 0.0)
    transmission_socket = bsdf.inputs.get('Transmission Weight')
    if transmission_socket is None:
        transmission_socket = bsdf.inputs.get('Transmission')
    _set_socket_value(transmission_socket, 0.82)
    _set_socket_value(bsdf.inputs.get('IOR'), 1.46)
    _set_socket_value(bsdf.inputs.get('Alpha'), 0.28)
    try:
        mat.surface_render_method = 'DITHERED'
    except (AttributeError, TypeError):
        pass
    try:
        mat.diffuse_color = (0.93, 0.97, 1.0, 0.28)
    except (AttributeError, TypeError):
        pass
    return mat

def create_food_storage_box_body():
    """
    Create the larger rigid clear food-storage box with a closed, solid top.
    This destination is intentionally not an open container: the top is a
    continuous collision-backed support surface for stacking.

    Model-space dimensions:
        Width  = 1.0
        Depth  = 0.75
        Height = 0.4

    World-space position is the geometric bounding-box center:
        (0.0, 0.0, 0.2)

    The resulting world-space extents are:
        X: -0.5 to 0.5
        Y: -0.375 to 0.375
        Z:  0.0 to 0.4
    """
    object_name = 'food_storage_box_body'
    _remove_object_if_present(object_name)
    width = 1.0
    depth = 0.75
    height = 0.4
    wall_thickness = 0.05
    base_thickness = 0.02
    rim_width = 0.05
    outer_corner_radius = 0.08
    inner_corner_radius = 0.03
    ring_segments = 8
    local_bottom_z = -height * 0.5
    local_top_z = height * 0.5
    local_floor_z = local_bottom_z + base_thickness
    inner_width = width - 2.0 * wall_thickness
    inner_depth = depth - 2.0 * wall_thickness
    if inner_width <= 0.0 or inner_depth <= 0.0:
        raise ValueError('Wall thickness is too large for the requested container.')
    opening_width = width - 2.0 * rim_width
    opening_depth = depth - 2.0 * rim_width
    inner_width = min(inner_width, opening_width)
    inner_depth = min(inner_depth, opening_depth)
    outer_bottom = _rounded_rectangle_ring(width, depth, outer_corner_radius, local_bottom_z, ring_segments)
    outer_top = _rounded_rectangle_ring(width, depth, outer_corner_radius, local_top_z, ring_segments)
    # Physical v3 repair: this destination is an ordinary closed storage box,
    # not an open vessel. Build one continuous solid shell: side wall plus
    # independent bottom and top fans. The previous fan reused the rim edge
    # while the inner wall also terminated there, giving three incident faces.
    ring_count = len(outer_bottom)
    if len(outer_top) != ring_count:
        raise RuntimeError('Container perimeter rings have inconsistent topology.')
    vertices = outer_bottom + outer_top
    outer_bottom_start = 0
    outer_top_start = ring_count
    faces = []
    for i in range(ring_count):
        j = (i + 1) % ring_count
        faces.append((outer_bottom_start + i, outer_bottom_start + j, outer_top_start + j, outer_top_start + i))
    bottom_center_index = len(vertices)
    vertices.append((0.0, 0.0, local_bottom_z))
    top_center_index = len(vertices)
    vertices.append((0.0, 0.0, local_top_z))
    for i in range(ring_count):
        j = (i + 1) % ring_count
        faces.append((bottom_center_index, outer_bottom_start + j, outer_bottom_start + i))
        faces.append((top_center_index, outer_top_start + i, outer_top_start + j))
    mesh = bpy.data.meshes.new('food_storage_box_body_mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = (0.0, 0.0, 0.2)
    obj.rotation_euler = (math.radians(0.0), math.radians(0.0), math.radians(0.0))
    bevel = obj.modifiers.new(name='Bevel', type='BEVEL')
    bevel.width = 0.006
    bevel.segments = 3
    bevel.limit_method = 'ANGLE'
    bevel.angle_limit = math.radians(30.0)
    bevel.harden_normals = True
    try:
        weighted_normal = obj.modifiers.new(name='Weighted Normals', type='WEIGHTED_NORMAL')
        weighted_normal.keep_sharp = True
        _wo_assign_blender_weight(50, weighted_normal)
    except (AttributeError, TypeError, RuntimeError):
        pass
    material = _create_clear_plastic_material()
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    obj.parent = None
    expected_dimensions = (width, depth, height)
    actual_dimensions = tuple((round(value, 6) for value in obj.dimensions))
    expected_rounded = tuple((round(value, 6) for value in expected_dimensions))
    if actual_dimensions != expected_rounded:
        raise RuntimeError(f'Container mesh dimensions do not match PartPlan: expected {expected_rounded}, got {actual_dimensions}')
    return obj
if __name__ == '__main__':
    created_object = create_food_storage_box_body()
    print(f'Created {created_object.name}')

# WORLDOLYMPIAD_PART_BOUNDS_CONTRACT
if __name__ == '__main__':
    import math
    def _wob_check_part_bounds(spec: dict, low: list[float], high: list[float]) -> dict:
        """Validate bounds expressed in planned local axes about the planned center."""
        dimensions = spec["shape"]["dimensions"]
        expected = [float(dimensions[key]) for key in ("width", "depth", "height")]
        values = [*expected, *low, *high]
        if len(low) != 3 or len(high) != 3 or any(not math.isfinite(x) for x in values) or min(expected) <= 0:
            raise ValueError("part bounds require finite coordinates and positive dimensions")
        center = [(a + b) / 2 for a, b in zip(low, high)]
        measured = [b - a for a, b in zip(low, high)]
        # Small bevels and approximate custom contours are allowed. A half-height
        # displacement from confusing a base origin with a center is not.
        tolerance = [max(0.005, 0.05 * size) for size in expected]
        passed = all(abs(offset) <= tol for offset, tol in zip(center, tolerance)) and all(
            abs(actual - planned) <= 2 * tol for actual, planned, tol in zip(measured, expected, tolerance)
        )
        return {"schema_version": "scenecode_part_bounds.v1", "part_name": spec["name"],
                "passed": passed, "planned_dimensions": expected, "measured_dimensions": measured,
                "center_offset_in_planned_axes": center, "tolerance": tolerance,
                "low_in_planned_axes": low, "high_in_planned_axes": high}

    import bpy as _wob_bpy
    import math
    import json
    from mathutils import Vector as _wob_Vector, Euler as _wob_Euler
    _wob_spec = json.loads('{"name": "food_storage_box_body", "part_type": "closed_storage_box_body", "shape": {"base_shape": "custom", "dimensions": {"width": 1.0, "depth": 0.75, "height": 0.4}, "modifiers": ["bevel"], "description": "Single-piece closed rectangular storage box mesh with an outer footprint of 1.0 by 0.75 model units and total height of 0.4 model units. The outer corners are mildly rounded rather than sharp. The bottom and top are flat, closed, and collision-backed; the top has positive full-box support thickness. Four side walls connect the two faces as one watertight shell. This is a closed destination box for stable centered stacking, not an open cavity."}, "position": {"x": 0.0, "y": 0.0, "z": 0.2}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}, "is_symmetric": false, "symmetric_axis": null, "is_movable": false, "must_be_independent": false, "description": "The complete rigid closed storage-box body, including flat bottom and top support faces, four side walls, and rounded corners. This one-piece destination is centered at the origin in X and Y, with bottom z=0 and top z=0.4. Its physical dimensions are exactly 0.20 m by 0.15 m by 0.08 m. The bottom exterior contacts the counter or support, while the full top face provides stable centered stacking.", "priority": 0, "material": {"type": "plastic", "base_color": [0.93, 0.97, 1.0], "roughness": 0.14, "metallic": 0.0, "texture": "none"}}')
    _wob_instances = _wob_spec.get("instances") or [{"name": _wob_spec["name"], "position": _wob_spec.get("position", {}), "rotation": _wob_spec.get("rotation", {})}]


    def _wob_plan_rotation(rotation):
        return _wob_Euler(tuple(math.radians(float(rotation.get(axis, 0))) for axis in "xyz"), "XYZ").to_matrix()


    def _wob_measure(instance):
        _wob_name = instance["name"]
        _wob_object = _wob_bpy.data.objects.get(_wob_name)
        if _wob_object is None:
            raise RuntimeError("PART_BOUNDS_CONTRACT: declared object missing: " + _wob_name)
        _wob_position = instance.get("position", _wob_spec.get("position", {}))
        _wob_rotation = instance.get("rotation", _wob_spec.get("rotation", {}))
        _wob_center = _wob_Vector(tuple(float(_wob_position.get(axis, 0)) for axis in "xyz"))
        _wob_inverse = _wob_plan_rotation(_wob_rotation).transposed()
        _wob_bpy.context.view_layer.update()
        _wob_graph = _wob_bpy.context.evaluated_depsgraph_get()
        _wob_points = []
        for _wob_child in [_wob_object, *_wob_object.children_recursive]:
            # Curves with bevels/extrusion are real rendered surfaces. Measuring
            # only mesh objects incorrectly rejects native Code_Object handles,
            # woven strands and other valid curve-built parts.
            if _wob_child.type not in {"MESH", "CURVE", "SURFACE", "FONT"}:
                continue
            _wob_evaluated = _wob_child.evaluated_get(_wob_graph)
            _wob_mesh = _wob_evaluated.to_mesh()
            try:
                if _wob_mesh is None or not _wob_mesh.polygons:
                    continue
                _wob_points.extend(_wob_inverse @ (_wob_evaluated.matrix_world @ vertex.co - _wob_center) for vertex in _wob_mesh.vertices)
            finally:
                _wob_evaluated.to_mesh_clear()
        if not _wob_points:
            raise RuntimeError("PART_BOUNDS_CONTRACT: no mesh geometry: " + _wob_name)
        _wob_low = [min(point[i] for point in _wob_points) for i in range(3)]
        _wob_high = [max(point[i] for point in _wob_points) for i in range(3)]
        return _wob_object, _wob_check_part_bounds({**_wob_spec, "name": _wob_name}, _wob_low, _wob_high)


    for _wob_instance in _wob_instances:
        _wob_object, _wob_measurement = _wob_measure(_wob_instance)
        print("PART_BOUNDS_EVIDENCE:" + json.dumps(_wob_measurement, sort_keys=True))
        if _wob_measurement["passed"]:
            continue
        # Repair instead of rejecting.  A constructor that confused a base origin
        # with a bounding-box center is off by a translation, so the part is moved
        # onto its declared center and re-measured.  A remaining dimension
        # deviation is recorded as a warning: size and shape are judged downstream
        # (native-view framing, physical usability, the certified expert rollout),
        # where a bad asset is visible as such instead of merely malformed.
        _wob_rotation = _wob_instance.get("rotation", _wob_spec.get("rotation", {}))
        _wob_shift = _wob_plan_rotation(_wob_rotation) @ _wob_Vector(tuple(_wob_measurement["center_offset_in_planned_axes"]))
        _wob_object.location = _wob_object.location - _wob_shift
        _wob_bpy.context.view_layer.update()
        _, _wob_recheck = _wob_measure(_wob_instance)
        _wob_correction = {"schema_version": "scenecode_part_bounds_correction.v1",
                           "recentered": True, "shift_applied": list(_wob_shift),
                           "before": _wob_measurement, "after": _wob_recheck}
        print("PART_BOUNDS_CORRECTION:" + json.dumps(_wob_correction, sort_keys=True))
        if not _wob_recheck["passed"]:
            print("PART_BOUNDS_WARNING:" + json.dumps(_wob_recheck, sort_keys=True))
