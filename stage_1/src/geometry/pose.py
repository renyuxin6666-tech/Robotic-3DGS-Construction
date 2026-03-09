import bpy  # 导入 Blender 的 Python API，用于访问场景中的物体（如相机）
import mathutils  # 导入 Blender 的数学工具库，用于处理矩阵（Matrix）、向量（Vector）等

def camera_world_to_camera_matrix(camera):
    """
    计算相机的【外参矩阵】(Extrinsic Matrix)。
    
    在计算机视觉中，我们需要把世界坐标系下的点转换到相机坐标系下，
    这个变换矩阵就叫做“世界到相机矩阵” (World-to-Camera Matrix)。
    
    参数:
        camera (bpy.types.Object): Blender场景中的相机对象
        
    返回:
        mathutils.Matrix: 一个 4x4 的变换矩阵
    """
    
    # 1. camera.matrix_world: 
    #    这是 Blender 存储的“相机模型矩阵”。
    #    它表示：把相机从原点(0,0,0)移动到当前位置的变换。
    #    也就是：【相机坐标系 -> 世界坐标系】 (Camera-to-World)
    
    # 2. .inverted():
    #    我们需要的是反过来的变换：把世界中的物体转换到相机眼里。
    #    也就是：【世界坐标系 -> 相机坐标系】 (World-to-Camera)
    #    所以我们需要对矩阵求逆。
    return camera.matrix_world.inverted()

def matrix_to_list(mat):
    """
    将 Blender 的矩阵对象转换为标准的 Python 列表。
    
    原因：
        Blender 的 mathutils.Matrix 对象不能直接保存到 JSON 文件中。
        我们需要把它转换成普通的嵌套列表（List of Lists）。
    
    参数:
        mat (mathutils.Matrix): 4x4 的矩阵对象
        
    返回:
        list: 一个 4x4 的二维列表，例如 [[1,0,0,0], [0,1,0,0], ...]
    """
    # 使用双重循环遍历 4x4 矩阵的每一个元素
    # i 代表行 (row), j 代表列 (column)
    # range(4) 会生成 0, 1, 2, 3
    return [[mat[i][j] for j in range(4)] for i in range(4)]
