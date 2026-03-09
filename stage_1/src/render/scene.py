import bpy
import mathutils

def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

def setup_world_white():
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 1.0

def import_obj(path):
    bpy.ops.wm.obj_import(filepath=str(path))
    objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active

def cleanup_and_seal_mesh(obj):
    """
    清理网格并封口：
    1. Merge by Distance (去除重叠点)
    2. Fill Holes (封住开放边缘)
    3. Recalculate Normals (修复法线朝向)
    """
    # 确保物体被选中且为激活状态
    bpy.context.view_layer.objects.active = obj
    
    # 进入编辑模式
    bpy.ops.object.mode_set(mode='EDIT')
    
    # 全选
    bpy.ops.mesh.select_all(action='SELECT')
    
    # 1. 合并重叠顶点 (阈值设小一点，避免误合)
    bpy.ops.mesh.remove_doubles(threshold=0.001)
    
    # 2. 填充所有孔洞 (sides=0 表示不限制边数)
    bpy.ops.mesh.fill_holes(sides=0)
    
    # 3. 重算法线 (确保一致向外)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    
    # 回到物体模式
    bpy.ops.object.mode_set(mode='OBJECT')

def normalize_scene(obj):
    """
    将物体归一化：
    1. 将几何中心移动到原点
    2. 计算包围球半径
    """
    # 1. 设置原点为几何中心
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    # 2. 移动物体到世界原点
    obj.location = (0, 0, 0)
    
    # 3. 更新场景以确保变换应用
    bpy.context.view_layer.update()
    
    # 4. 计算最大半径 (包围球)
    # 遍历所有顶点，找到距离原点最远的点
    max_dist = 0.0
    # 注意：obj.bound_box 是局部坐标，因为我们已经apply location到0,0,0，
    # 且origin在中心，所以可以直接用 bound_box 的角点估算，或者遍历顶点。
    # 用 bound_box 角点更高效。
    for corner in obj.bound_box:
        # corner 是 Vector((x, y, z))
        dist = (obj.matrix_world @ mathutils.Vector(corner)).length
        if dist > max_dist:
            max_dist = dist
            
    return max_dist