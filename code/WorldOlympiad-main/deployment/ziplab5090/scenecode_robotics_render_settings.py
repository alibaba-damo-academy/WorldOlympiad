"""SceneCode Blender settings for trajectory replay at model-native sizes."""

import bpy


scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
# Blender 4.5 defaults to 64 EEVEE samples. Sixteen retains anti-aliased
# robotics edges at the registered 320/640-pixel planes while avoiding four
# redundant samples for every imported trajectory state.
scene.eevee.taa_render_samples = 16


def _enable_transmission_refraction(
    *_args, _done: set = set(), _enabled: list = [False]
) -> None:
    """Resolve authored glass through raytracing, but only when glass exists.

    EEVEE only resolves KHR_materials_transmission through screen-space
    raytracing.  Without it every authored bottle, jar or liquid surface falls
    back to a frosted white shell and hides its contents.  Raytracing costs
    roughly three times a plain EEVEE frame, so the scene only turns it on once
    an imported material actually asks for transmission; all-opaque scenes keep
    the cheap path.  The SceneCode Blender server executes this file with `bpy`
    as the only global, so the memo and the flag live in the default arguments
    instead of module-level names the call could not see.
    """

    for material in bpy.data.materials:
        if material.name in _done:
            continue
        tree = material.node_tree if material.use_nodes else None
        if tree is None:
            continue
        transmitted = False
        for node in tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            for socket in node.inputs:
                if socket.name not in {"Transmission Weight", "Transmission"}:
                    continue
                try:
                    if float(socket.default_value) > 0.0:
                        transmitted = True
                except (TypeError, ValueError):
                    pass
        if not transmitted:
            continue
        for attribute in ("use_raytrace_refraction", "use_screen_refraction"):
            if hasattr(material, attribute):
                try:
                    setattr(material, attribute, True)
                except (AttributeError, TypeError):
                    pass
        _done.add(material.name)
        if not _enabled[0]:
            try:
                scene = bpy.context.scene
                scene.eevee.use_raytracing = True
                scene.eevee.ray_tracing_options.use_denoise = True
                scene.eevee.ray_tracing_options.trace_max_roughness = 0.7
                scene.eevee.ray_tracing_options.screen_trace_max_roughness = 0.7
            except AttributeError:
                pass
            _enabled[0] = True


_enable_transmission_refraction()
bpy.app.handlers.depsgraph_update_post.append(_enable_transmission_refraction)
