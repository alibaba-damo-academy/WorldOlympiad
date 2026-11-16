import bpy
import os
from typing import Dict, Any

def bake_materials_for_export(resolution: int = 1024, use_selection: bool = False) -> Dict[str, list]:
    """
    烘焙材质，并返回物体原本的材质映射表，以便后续还原。
    """
    print(f"开始烘焙材质 (分辨率: {resolution}x{resolution})...")
    
    original_engine = bpy.context.scene.render.engine
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 16
    
    if use_selection:
        objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    else:
        objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        
    # 用于记录物体原本的材质
    original_materials_map = {}
        
    for obj in objs:
        # 记录当前物体的所有材质
        original_materials_map[obj.name] = [slot.material for slot in obj.material_slots]
        
        if not obj.data.uv_layers:
            print(f"为 {obj.name} 生成 UV 贴图...")
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project()
            bpy.ops.object.mode_set(mode='OBJECT')
            
        for slot in obj.material_slots:
            if not slot.material or not slot.material.use_nodes:
                continue
                
            # 复制材质，避免破坏原材质数据
            slot.material = slot.material.copy()
            mat = slot.material
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            out_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if not bsdf or not out_node:
                continue
                
            print(f"正在烘焙材质: {mat.name}")
            
            img_color = bpy.data.images.new(name=f"{mat.name}_Color", width=resolution, height=resolution)
            img_metallic = bpy.data.images.new(name=f"{mat.name}_Metallic", width=resolution, height=resolution)
            img_roughness = bpy.data.images.new(name=f"{mat.name}_Roughness", width=resolution, height=resolution)
            img_normal = bpy.data.images.new(name=f"{mat.name}_Normal", width=resolution, height=resolution)
            
            node_color = nodes.new('ShaderNodeTexImage')
            node_color.image = img_color
            node_color.name = "Bake_Color"
            
            node_metallic = nodes.new('ShaderNodeTexImage')
            node_metallic.image = img_metallic
            node_metallic.image.colorspace_settings.name = 'Non-Color'
            node_metallic.name = "Bake_Metallic"
            
            node_roughness = nodes.new('ShaderNodeTexImage')
            node_roughness.image = img_roughness
            node_roughness.image.colorspace_settings.name = 'Non-Color'
            node_roughness.name = "Bake_Roughness"
            
            node_normal = nodes.new('ShaderNodeTexImage')
            node_normal.image = img_normal
            node_normal.image.colorspace_settings.name = 'Non-Color'
            node_normal.name = "Bake_Normal"
            
            emit_node = nodes.new('ShaderNodeEmission')
            
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            
            def bake_pass_via_emit(input_name, target_node, img):
                in_socket = bsdf.inputs[input_name]
                if in_socket.is_linked:
                    source_socket = in_socket.links[0].from_socket
                    links.new(source_socket, emit_node.inputs['Color'])
                else:
                    val = in_socket.default_value
                    if isinstance(val, float) or isinstance(val, int):
                        emit_node.inputs['Color'].default_value = (val, val, val, 1.0)
                    else:
                        emit_node.inputs['Color'].default_value = val
                
                links.new(emit_node.outputs['Emission'], out_node.inputs['Surface'])
                
                for n in nodes: n.select = False
                target_node.select = True
                nodes.active = target_node
                
                bpy.ops.object.bake(type='EMIT')
                img.pack()

            print(f"  -> 烘焙 Base Color...")
            bake_pass_via_emit('Base Color', node_color, img_color)
            
            print(f"  -> 烘焙 Metallic...")
            bake_pass_via_emit('Metallic', node_metallic, img_metallic)
            
            print(f"  -> 烘焙 Roughness...")
            bake_pass_via_emit('Roughness', node_roughness, img_roughness)
            
            print(f"  -> 烘焙 Normal...")
            links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
            for n in nodes: n.select = False
            node_normal.select = True
            nodes.active = node_normal
            bpy.ops.object.bake(type='NORMAL')
            img_normal.pack()
            
            for n in list(nodes):
                if n not in [bsdf, out_node, node_color, node_metallic, node_roughness, node_normal]:
                    nodes.remove(n)
                    
            links.new(node_color.outputs['Color'], bsdf.inputs['Base Color'])
            links.new(node_metallic.outputs['Color'], bsdf.inputs['Metallic'])
            links.new(node_roughness.outputs['Color'], bsdf.inputs['Roughness'])
            
            normal_map = nodes.new('ShaderNodeNormalMap')
            links.new(node_normal.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
            
            links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
            
    bpy.context.scene.render.engine = original_engine
    print("烘焙完成！")
    
    # 返回原本的材质映射表
    return original_materials_map


def export_scene_to_gltf(
    output_path: str,
    export_format: str = "GLTF_SEPARATE",
    use_selection: bool = False,
    export_apply: bool = False,
    bake_textures: bool = True,
    bake_resolution: int = 1024,
) -> Dict[str, Any]:
    
    out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    normalized_format = str(export_format).upper()
    ext = os.path.splitext(out_path)[1].lower()
    expected_ext = ".glb" if normalized_format == "GLB" else ".gltf"
    if ext != expected_ext:
        raise ValueError(f"output_path 必须以 {expected_ext} 结尾")

    if hasattr(bpy.ops, "object") and bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")

    original_materials_map = {}
    
    try:
        # 1. 烘焙并获取原材质记录
        if bake_textures:
            original_materials_map = bake_materials_for_export(
                resolution=bake_resolution, 
                use_selection=use_selection
            )

        # 2. 导出 GLB
        bpy.ops.export_scene.gltf(
            filepath=out_path,
            export_format=export_format,
            use_selection=bool(use_selection),
            export_apply=bool(export_apply),
            export_cameras=False,
            export_lights=False,
        )
        
    finally:
        # 3. 【核心修复】无论导出成功还是失败，都把材质还原回去！
        if bake_textures and original_materials_map:
            print("正在还原场景原始材质...")
            for obj_name, original_mats in original_materials_map.items():
                obj = bpy.data.objects.get(obj_name)
                if obj:
                    for i, mat in enumerate(original_mats):
                        if i < len(obj.material_slots):
                            obj.material_slots[i].material = mat
            print("材质还原完毕！场景已恢复原状。")

    return {
        "success": True,
        "output_path": out_path,
        "export_format": export_format,
        "use_selection": bool(use_selection),
        "export_apply": bool(export_apply),
        "baked_textures": bake_textures,
    }

if __name__ == "__main__":
    export_dir = "/tmp"
    test_output_path = os.path.join(export_dir, "test_export.glb")
    
    print(f"准备导出到: {test_output_path}")
    
    try:
        result = export_scene_to_gltf(
            output_path=test_output_path,
            export_format="GLB", 
            use_selection=False,           
            export_apply=True,
            bake_textures=True,            
            bake_resolution=2048           # 【建议】如果觉得细节不够，可以把这里改成 2048 甚至 4096
        )
        print("导出成功!")
        print(result)
    except Exception as e:
        print(f"导出失败: {e}")