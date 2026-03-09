import argparse
import sys
import yaml
import math
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage

# Add project root to path
# 将项目根目录添加到系统路径，以便导入项目中的其他模块
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# Try to import pyrender, trimesh, and pymeshlab
# 尝试导入渲染库 pyrender, 网格处理库 trimesh, 和 MeshLab 接口 pymeshlab
try:
    import pyrender
    import trimesh
    import pymeshlab
except ImportError as e:
    print(f"Error: Required library not found ({e}). Please install them:")
    print("pip install pyrender trimesh pymeshlab")
    sys.exit(1)

def load_config(config_path):
    """
    加载 YAML 配置文件。
    Load YAML configuration file.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def fibonacci_sphere(samples=1000):
    """
    使用斐波那契格点（Fibonacci Lattice）算法在球面上生成均匀分布的点。
    Generate points on a sphere using Fibonacci Lattice.
    
    原理 (Principle):
    利用黄金角度（Golden Angle）来使得点在球面上分布得尽可能均匀，避免极点聚集现象。
    
    Args:
        samples (int): 需要生成的采样点数量 (Number of points to generate).
        
    Returns:
        np.array: (samples, 3) 的数组，包含每个点的 [x, y, z] 坐标，均为单位向量。
    """
    points = []
    # 黄金角度 (Golden Angle) ≈ 137.508度
    # phi = pi * (3 - sqrt(5))
    phi = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        # 1. 计算 y 坐标 (Vertical coordinate)
        # y 从 1 线性变化到 -1 (y goes from 1 to -1)
        # 这种线性映射保证了球冠面积的均匀切分
        y = 1 - (i / float(samples - 1)) * 2

        # 2. 计算当前高度的半径 (Radius at height y)
        # r = sqrt(1 - y^2) 基于圆的方程 x^2 + z^2 = r^2 = 1 - y^2
        radius = math.sqrt(1 - y * y)

        # 3. 计算经度角度 (Longitude angle theta)
        # 每次增加一个黄金角度
        theta = phi * i

        # 4. 转换为笛卡尔坐标 (Convert to Cartesian coordinates)
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def get_look_at_matrix(eye, target, up):
    """
    计算 LookAt 矩阵，并转换为相机到世界（Camera-to-World）的变换矩阵（Pose Matrix）。
    Calculate LookAt matrix and convert to Camera-to-World Pose matrix.
    
    Args:
        eye (np.array): 相机位置 (Camera position).
        target (np.array): 相机看向的目标点 (Target point).
        up (np.array): 世界坐标系中的上向量 (Up vector in World space).
        
    Returns:
        np.array: 4x4 的相机位姿矩阵 (Camera Pose Matrix).
    """
    # 1. 计算相机坐标系的 Z 轴 (Forward axis)
    # 在常规图形学（如 OpenGL）中，相机看向 -Z 方向，因此 Z 轴指向相机后方（从目标指向眼睛）
    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # 2. 计算相机坐标系的 X 轴 (Right axis)
    # 通过上向量和 Z 轴的叉积得到右向量
    x_axis = np.cross(up, z_axis)
    
    # 处理奇异点：如果 up 向量与 Z 轴平行（如从正上方俯视），叉积为 0
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0]) # 此时假设 X 轴为世界坐标系的 X 轴
        
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 3. 计算相机坐标系的 Y 轴 (True Up axis)
    # 通过 Z 轴和 X 轴的叉积重新计算正交的上向量
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 4. 构建旋转矩阵 R (Rotation Matrix)
    # 这是一个从世界坐标系到相机坐标系的旋转矩阵的逆矩阵（即相机坐标系的基向量在世界坐标系中的表示）
    # R = [x_axis, y_axis, z_axis]
    
    # 5. 构建位姿矩阵 (Pose Matrix: Camera -> World)
    # pyrender 使用的是 Camera-to-World 矩阵
    # 矩阵的左上 3x3 是旋转矩阵（列向量为相机的坐标轴），第四列是相机位置
    pose = np.eye(4)
    pose[:3, 0] = x_axis  # Column 0: Camera Right (X)
    pose[:3, 1] = y_axis  # Column 1: Camera Up (Y)
    pose[:3, 2] = z_axis  # Column 2: Camera Back (Z)
    pose[:3, 3] = eye     # Column 3: Camera Position
    
    return pose

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Fibonacci Lattice Rendering (斐波那契球面采样渲染)")
    parser.add_argument("--config", default="stage_1/configs/render_fibonacci.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # 检查配置文件
    config_path = project_root / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return
    config = load_config(config_path)
    
    # 路径设置 (Paths)
    input_dir = project_root / config['paths']['input_dir']
    output_dir = project_root / config['paths']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 渲染设置 (Settings)
    resolution = config['render']['resolution']
    num_views = config['camera']['num_views']
    dist_scale = config['camera']['distance_scale'] # 距离缩放因子，控制相机离物体的远近
    yfov = np.radians(config['camera']['fov'])      # 垂直视场角 (Field of View)
    
    # 1. 生成相机位置 (Generate Camera Positions)
    print(f"Generating {num_views} Fibonacci sampling points...")
    cam_positions = fibonacci_sphere(num_views)
    
    # 2. 初始化离屏渲染器 (Setup Renderer)
    # 注意：在无头服务器上可能需要配置 EGL (On headless servers, EGL might be required)
    r = pyrender.OffscreenRenderer(resolution, resolution)
    
    # 获取所有 OBJ 模型文件
    obj_files = list(input_dir.glob("*.obj"))
    print(f"Found {len(obj_files)} models in {input_dir}")
    
    # 3. 遍历模型进行渲染 (Iterate over models)
    for obj_path in tqdm(obj_files, desc="Rendering Models"):
        branch_id = obj_path.stem # e.g. "model_1"
        
        # 加载网格 (Load mesh)
        try:
            # 使用 Trimesh 加载
            raw_mesh = trimesh.load(obj_path, force='mesh')
            
            # ---------------------------------------------------------
            # 强力修复：体素化重建 (Robust Repair: Voxelization & Remeshing)
            # ---------------------------------------------------------
            # 这种方法通过将网格转化为体素再重建(Marching Cubes)，能保证生成 100% 封闭(Watertight)的网格。
            # 这对于生成 Mask 尤其有效，因为它能填补所有孔洞。
            
            # 1. 计算合适的体素大小 (Pitch)
            # 分辨率设为 128 (即最长边被切分为 128 份)，足以保持树枝形状
            target_resolution = 128
            pitch = raw_mesh.extents.max() / target_resolution
            
            # 2. 体素化
            # 使用 fill_method='base' 填充内部
            voxelized = raw_mesh.voxelized(pitch=pitch)
            
            # 3. 强力填充内部 (Slice-wise Fill)
            # 普通的 voxelized.fill() 无法填充两端开口的管子(如树枝)
            # 我们通过沿 X/Y/Z 三个方向分别对切片进行 2D 孔洞填充，来强制实心化
            matrix = voxelized.matrix.copy()
            
            # Fill along X axis slices
            for i in range(matrix.shape[0]):
                matrix[i, :, :] = ndimage.binary_fill_holes(matrix[i, :, :])
                
            # Fill along Y axis slices
            for i in range(matrix.shape[1]):
                matrix[:, i, :] = ndimage.binary_fill_holes(matrix[:, i, :])
            
            # Fill along Z axis slices
            for i in range(matrix.shape[2]):
                matrix[:, :, i] = ndimage.binary_fill_holes(matrix[:, :, i])
                
            # Re-create VoxelGrid with filled matrix
            new_voxel = trimesh.voxel.VoxelGrid(
                trimesh.voxel.encoding.DenseEncoding(matrix), 
                transform=voxelized.transform
            )
            
            # 4. Marching Cubes 重建
            trimesh_mesh = new_voxel.marching_cubes
            
            # 5. 平滑一下 (可选，减少体素块状感)
            trimesh.smoothing.filter_laplacian(trimesh_mesh, iterations=2)
            
            # 确保法线正确
            trimesh_mesh.fix_normals()
            
            print(f"  - Repaired {obj_path.name}: Watertight={trimesh_mesh.is_watertight}, Faces={len(trimesh_mesh.faces)}")
            
        except Exception as e:
            print(f"Failed to load/repair {obj_path} with Voxelization: {e}")
            # Fallback to simple load if voxelization fails
            try:
                trimesh_mesh = trimesh.load(obj_path, force='mesh')
            except Exception as e2:
                 print(f"Failed to load {obj_path} with trimesh: {e2}")
                 continue

        # ---------------------------------------------------------
        # 几何中心对齐 (Center Alignment)
        # ---------------------------------------------------------
        # 预处理：归一化网格 (Normalize mesh)
        # 将网格中心移动到原点，确保旋转时围绕物体中心
        center = trimesh_mesh.centroid
        trimesh_mesh.vertices -= center
        
        # ---------------------------------------------------------
        # 自适应距离计算 (Adaptive Camera Distance)
        # ---------------------------------------------------------
        # 计算相机距离 (Calculate camera distance)
        # 计算包围球半径，确保物体完全在视锥体内
        bounding_sphere_radius = np.max(np.linalg.norm(trimesh_mesh.vertices, axis=1))
        
        # 根据 FOV 和包围球半径计算合适的拍摄距离
        # dist = radius / sin(fov/2)
        # 增加额外的 10% 留白 (Padding)，确保物体不贴边
        padding_factor = 1.1 
        cam_dist = (bounding_sphere_radius * dist_scale * padding_factor) / math.sin(yfov / 2.0)
        
        # 创建 Pyrender 场景 (Create Pyrender Scene)
        # 使用黑色材质渲染剪影 (Force black material for silhouette)
        # baseColorFactor=[R, G, B, A] -> 全黑
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.0, 0.0, 0.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, material=material)
        
        # 准备输出目录
        branch_out_dir = output_dir / branch_id
        branch_out_dir.mkdir(exist_ok=True)
        
        # 元数据列表 (Metadata list)
        meta_records = []
        
        # 4. 遍历每个视角进行渲染 (Render each view)
        for i, pos_norm in enumerate(cam_positions):
            # 创建场景，背景设为白色
            scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0]) 
            scene.add(mesh)
            
            # 计算相机位姿 (Camera Pose)
            eye = pos_norm * cam_dist       # 相机位置
            target = np.array([0, 0, 0])    # 目标点（物体中心）
            up = np.array([0, 1, 0])        # 初始上向量
            
            # 如果视线方向与上向量平行，调整上向量以避免奇异性
            if np.abs(np.dot(pos_norm, up)) > 0.99:
                up = np.array([0, 0, 1])
                
            camera_pose = get_look_at_matrix(eye, target, up)
            
            # 添加相机
            camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
            scene.add(camera, pose=camera_pose)
            
            # 添加灯光 (可选，渲染剪影不需要灯光，或使用环境光)
            # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            # scene.add(light, pose=camera_pose)
            
            # 执行渲染 (Render)
            # color: (H, W, 3) RGB image
            color, _ = r.render(scene)
            
            # 保存图像 (Save Image)
            img_filename = f"{i:04d}.png"
            img_path = branch_out_dir / img_filename
            
            # 使用 PIL 保存
            from PIL import Image
            im = Image.fromarray(color)
            im.save(img_path)
            
            # 记录元数据 (Record Metadata)
            meta_records.append({
                "view_id": i,
                "camera_pose": camera_pose.tolist(), # 4x4 矩阵
                "camera_pos_sphere": pos_norm.tolist(), # 单位球面上的位置向量
                "camera_dist": float(cam_dist),
                "image_path": str(img_path.relative_to(project_root))
            })
            
        # 保存元数据 (Save Metadata)
        with open(branch_out_dir / "meta.json", 'w') as f:
            json.dump(meta_records, f, indent=2)
            
    print("Rendering complete.")

if __name__ == "__main__":
    main()
