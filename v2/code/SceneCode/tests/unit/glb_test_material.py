import bpy
import os
from typing import Dict, Any

def bake_materials_for_export(resolution: int = 1024, use_selection: bool = False):
    """
    将程序化材质烘焙为图像纹理，以便于 glTF 导出。
    """
    print("开始烘焙材质...")
    
    # 1. 切换到 Cycles 渲染器（只有 Cycles 支持烘焙）
    original_engine = bpy.context.scene.render.engine
    bpy.context.scene.render.engine = 'CYCLES'
    
    # 降低采样率以加快烘焙速度（程序化纹理不需要高采样）
    bpy.context.scene.cycles.samples = 16
    
    # 获取需要烘焙的物体
    if use_selection:
        objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    else:
        objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        
    for obj in objs:
        # 确保物体有 UV 贴图
        if not obj.data.uv_layers:
            print(f"为 {obj.name} 生成 UV 贴图...")
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project()
            bpy.ops.object.mode_set(mode='OBJECT')
            
        # 遍历物体的材质槽
        for slot in obj.material_slots:
            if not slot.material or not slot.material.use_nodes:
                continue
                
            # 复制材质，使其成为单用户（避免烘焙时互相干扰）
            slot.material = slot.material.copy()
            mat = slot.material
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # 查找 Principled BSDF 节点
            bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if not bsdf:
                continue
                
            print(f"正在烘焙材质: {mat.name}")
            
            # 创建用于烘焙的图像
            img_color = bpy.data.images.new(name=f"{mat.name}_Color", width=resolution, height=resolution)
            img_normal = bpy.data.images.new(name=f"{mat.name}_Normal", width=resolution, height=resolution)
            img_roughness = bpy.data.images.new(name=f"{mat.name}_Roughness", width=resolution, height=resolution)
            
            # 创建图像纹理节点
            node_color = nodes.new('ShaderNodeTexImage')
            node_color.image = img_color
            node_color.name = "Bake_Color"
            
            node_normal = nodes.new('ShaderNodeTexImage')
            node_normal.image = img_normal
            node_normal.image.colorspace_settings.name = 'Non-Color'
            node_normal.name = "Bake_Normal"
            
            node_roughness = nodes.new('ShaderNodeTexImage')
            node_roughness.image = img_roughness
            node_roughness.image.colorspace_settings.name = 'Non-Color'
            node_roughness.name = "Bake_Roughness"
            
            # 设置当前活动物体
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            
            # --- 烘焙 Base Color ---
            for n in nodes: n.select = False
            node_color.select = True
            nodes.active = node_color
            print(f"  -> 烘焙 Base Color...")
            bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})
            img_color.pack() # 打包到 blend 文件中，方便导出
            
            # --- 烘焙 Normal ---
            for n in nodes: n.select = False
            node_normal.select = True
            nodes.active = node_normal
            print(f"  -> 烘焙 Normal...")
            bpy.ops.object.bake(type='NORMAL')
            img_normal.pack()
            
            # --- 烘焙 Roughness ---
            for n in nodes: n.select = False
            node_roughness.select = True
            nodes.active = node_roughness
            print(f"  -> 烘焙 Roughness...")
            bpy.ops.object.bake(type='ROUGHNESS')
            img_roughness.pack()
            
            # --- 重新连接节点 ---
            # 找到输出节点
            out_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            
            # 删除原有的程序化节点
            for n in list(nodes):
                if n not in [bsdf, out_node, node_color, node_normal, node_roughness]:
                    nodes.remove(n)
                    
            # 连接烘焙好的图像到 BSDF
            # 注意：Blender 4.0+ 中 Base Color 依然叫 'Base Color'，Roughness 叫 'Roughness'
            links.new(node_color.outputs['Color'], bsdf.inputs['Base Color'])
            links.new(node_roughness.outputs['Color'], bsdf.inputs['Roughness'])
            
            # 添加法线贴图节点
            normal_map = nodes.new('ShaderNodeNormalMap')
            links.new(node_normal.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
            
    # 恢复原来的渲染器
    bpy.context.scene.render.engine = original_engine
    print("烘焙完成！")


def export_scene_to_gltf(
    output_path: str,
    export_format: str = "GLTF_SEPARATE",
    use_selection: bool = False,
    export_apply: bool = False,
    bake_textures: bool = True,  # 新增参数，控制是否烘焙
    bake_resolution: int = 1024, # 新增参数，控制烘焙分辨率
) -> Dict[str, Any]:
    # 将路径转换为绝对路径
    out_path = os.path.abspath(output_path)
    
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 确保当前处于物体模式 (OBJECT mode)
    if hasattr(bpy.ops, "object") and bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")

    # 如果需要，在导出前执行烘焙
    if bake_textures:
        bake_materials_for_export(resolution=bake_resolution, use_selection=use_selection)

    # 调用 Blender 的 glTF 导出操作
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format=export_format,
        use_selection=bool(use_selection),
        export_apply=bool(export_apply),
        export_cameras=False,
        export_lights=False,
    )

    return {
        "success": True,
        "output_path": out_path,
        "export_format": export_format,
        "use_selection": bool(use_selection),
        "export_apply": bool(export_apply),
        "baked_textures": bake_textures,
    }

# ==========================================
# 测试运行代码 (点击 Blender 文本编辑器上的 "运行脚本" 按钮)
# ==========================================
if __name__ == "__main__":
    # 设置导出的路径
    export_dir = "/tmp"
        
    test_output_path = os.path.join(export_dir, "test_export.glb")
    
    print(f"准备导出到: {test_output_path}")
    
    try:
        result = export_scene_to_gltf(
            output_path=test_output_path,
            export_format="GLB", 
            use_selection=False,           
            export_apply=True,
            bake_textures=True,            # 开启烘焙
            bake_resolution=1024           # 烘焙贴图分辨率 (1024x1024)
        )
        print("导出成功!")
        print(result)
    except Exception as e:
        print(f"导出失败: {e}")
