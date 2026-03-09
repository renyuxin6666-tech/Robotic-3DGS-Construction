# -*- coding: utf-8 -*-
"""
材质管理模块

该模块提供用于多通道渲染的材质创建和应用功能。
支持以下通道：
1. Silhouette (Mask): 黑色发射材质，用于生成二值轮廓
2. Depth: 归一化深度材质（0-1），需要配合合成器节点使用
3. Normal: 世界空间法线材质
4. Clay (White): 白模光照材质，用于大模型识别
"""

import bpy

def clear_materials(obj):
    """清空物体的所有材质槽"""
    obj.data.materials.clear()

def apply_material(obj, mat):
    """应用材质到物体"""
    clear_materials(obj)
    obj.data.materials.append(mat)

def get_silhouette_material():
    """获取或创建黑色轮廓材质"""
    mat_name = "mat_silhouette"
    mat = bpy.data.materials.get(mat_name)
    if mat: return mat
    
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (0, 0, 0, 1)
    emit.inputs["Strength"].default_value = 1.0
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    
    return mat

def get_normal_material():
    """获取或创建法线材质 (World Space Normal Map)"""
    mat_name = "mat_normal"
    mat = bpy.data.materials.get(mat_name)
    if mat: return mat

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Geometry -> Vector Transform (Object to World) -> Emission -> Output
    # 注意：Geometry Node 的 Normal 输出通常是 World Space，但为了保险起见，
    # 或者如果物体有旋转，我们直接用 Geometry 的 Normal 即可（Cycles/Eevee 默认是 World Space）
    # 但为了可视化更好，通常将范围 [-1, 1] 映射到 [0, 1]
    
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    
    geom = nodes.new("ShaderNodeNewGeometry")
    
    # Normal is [-1, 1], map to [0, 1] for color: (N + 1) / 2
    # Add 1
    vec_add = nodes.new("ShaderNodeVectorMath")
    vec_add.operation = 'ADD'
    vec_add.inputs[1].default_value = (1, 1, 1)
    
    # Divide by 2
    vec_div = nodes.new("ShaderNodeVectorMath")
    vec_div.operation = 'DIVIDE'
    vec_div.inputs[1].default_value = (2, 2, 2)
    
    links.new(geom.outputs["Normal"], vec_add.inputs[0])
    links.new(vec_add.outputs["Vector"], vec_div.inputs[0])
    links.new(vec_div.outputs["Vector"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])

    return mat

def get_clay_material():
    """获取或创建白模材质 (Principled BSDF)"""
    mat_name = "mat_clay"
    mat = bpy.data.materials.get(mat_name)
    if mat: return mat

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    
    # 设置为浅灰色，粗糙度适中
    bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1)
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    
    return mat

def setup_lighting():
    """设置标准三点光照（用于白模渲染）"""
    # 清除现有灯光
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()

    # 主光 (Key Light)
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    key = bpy.context.active_object
    key.data.energy = 3.0
    key.rotation_euler = (0.785, 0, 0.785) # 45 deg

    # 补光 (Fill Light)
    bpy.ops.object.light_add(type='SUN', location=(-5, -5, 5))
    fill = bpy.context.active_object
    fill.data.energy = 1.5
    fill.rotation_euler = (0.785, 0, -0.785)

    # 背光 (Back Light)
    bpy.ops.object.light_add(type='SUN', location=(0, 5, 5))
    back = bpy.context.active_object
    back.data.energy = 2.0
    back.rotation_euler = (-0.785, 0, 3.14)
