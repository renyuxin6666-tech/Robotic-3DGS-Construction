# -*- coding: utf-8 -*-
"""
渲染数据集脚本

该脚本用于从3D模型生成多视角的渲染图像数据集。
主要功能包括：
- 加载3D模型文件(.obj格式)
- 设置渲染环境（相机、光照、背景）
- 从不同角度（方位角和仰角）渲染模型
- 保存渲染图像和对应的相机姿态信息

用途：用于计算机视觉和3D重建任务的数据集生成
"""

import sys
from pathlib import Path

# 添加 stage_1 目录到 Python 路径，以便导入 src 模块
# 脚本位于: stage_1/scripts/render_dataset.py
# parents[0] = scripts
# parents[1] = stage_1
sys.path.append(str(Path(__file__).resolve().parents[1]))

# 导入自定义工具模块
from src.utils.io import load_yaml, write_jsonl  # YAML配置加载和JSONL文件写入
from src.utils.paths import mesh_dir, output_dir  # 路径管理工具

import math

# 导入渲染场景相关模块
from src.render.scene import clear_scene, setup_world_white, import_obj, normalize_scene, cleanup_and_seal_mesh  # 场景设置和模型导入
# from src.render.silhouette import apply_black_emission  # 废弃：使用 materials 模块
from src.render.materials import (
    apply_material, 
    get_silhouette_material, 
    get_normal_material, 
    get_clay_material,
    setup_lighting
)

# 导入相机控制模块
from src.render.camera import setup_camera, set_camera_pose  # 相机设置和姿态控制

# 导入渲染输出模块
from src.render.export import setup_render, render_image  # 渲染设置和图像输出

# 导入几何变换模块
from src.geometry.pose import camera_world_to_camera_matrix, matrix_to_list  # 相机姿态矩阵转换

def main():
    """
    主函数：执行数据集渲染流程
    
    流程：
    1. 加载配置文件
    2. 设置渲染参数
    3. 遍历所有3D模型文件
    4. 为每个模型生成多视角渲染图像
    5. 保存图像和对应的相机姿态信息
    """
    
    # 加载渲染配置文件
    # 使用相对于脚本的路径，确保在任何 CWD 下都能找到
    script_dir = Path(__file__).resolve().parent
    stage1_dir = script_dir.parent
    config_path = stage1_dir / "configs" / "render.yaml"
    cfg = load_yaml(config_path)
    
    # 获取所有.obj格式的3D模型文件（递归查找）
    # 适配结构：assets/branches_3d/log_xx/model.obj
    meshes = list(mesh_dir(cfg).glob("**/*.obj"))
    
    # 设置渲染分辨率（宽度和高度）
    setup_render(cfg["render"]["width"], cfg["render"]["height"])
    
    # 遍历每个3D模型文件
    for mesh in meshes:
        # 确定唯一ID：如果文件名是model.obj，使用父文件夹名（如log_1）
        branch_id = mesh.parent.name if mesh.name == "model.obj" else mesh.stem
        print(f"正在处理模型: {branch_id} ({mesh.name})")
        
        # 清空当前场景，准备新的渲染
        clear_scene()
        
        # 设置白色背景的世界环境
        setup_world_white()
        
        # 导入3D模型对象
        obj = import_obj(mesh)
        
        # 修复模型：封口、清理重叠点、重算法线
        # 这对于 Normal Map 和 Clay 渲染至关重要
        cleanup_and_seal_mesh(obj)
        print(f"  - 模型已封口与清理")
        
        # 归一化物体（居中并获取包围半径）
        bounding_radius = normalize_scene(obj)
        print(f"  - 归一化完成: 几何中心已移至原点, 包围半径 = {bounding_radius:.4f}")
        
        # 初始化光照（为白模渲染做准备）
        setup_lighting()
        
        # 设置相机，配置视场角（FOV）
        fov_deg = cfg["camera"]["fov_deg"]
        cam = setup_camera(fov_deg)
        
        # 计算自适应相机距离
        # 公式: dist = radius / sin(fov/2)
        # 增加 1.1 倍安全边距，防止边缘相切
        fov_rad = math.radians(fov_deg)
        cam_dist = (bounding_radius / math.sin(fov_rad / 2)) * 1.1
        # 如果计算出的距离小于配置的最小半径（可选），使用配置值，或者完全信任计算值
        # 这里我们完全信任计算值，但为了避免极其微小的物体导致相机过近，可以设置一个下限
        cam_dist = max(cam_dist, cfg["camera"].get("min_radius", 0.5))
        print(f"  - 自适应相机距离: {cam_dist:.4f}")
        
        # 初始化记录列表，用于存储每张图像的元数据
        records = []
        
        # 创建输出目录结构
        # data/rendered/branch_id/
        #   ├── mask/    (轮廓)
        #   ├── normal/  (法线)
        #   ├── clay/    (白模)
        #   └── meta.jsonl
        base_out_dir = output_dir(cfg) / branch_id
        dirs = {
            "mask": base_out_dir / "mask",
            "normal": base_out_dir / "normal",
            "clay": base_out_dir / "clay",
            # "depth": base_out_dir / "depth" # 深度图通常需要 OpenEXR 或特殊节点，暂缓
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        # 图像索引计数器
        idx = 0
        
        # 遍历所有仰角（elevation）设置
        for el in cfg["camera"]["elevation_deg"]:
            # 从0度开始，按方位角步长递增，直到360度
            az = 0
            while az < 360:
                # 1. 设置相机姿态 (对于所有通道都是一样的)
                set_camera_pose(
                    cam,
                    cam_dist,                 # 使用计算出的自适应距离
                    az,                       # 方位角（水平旋转角度）
                    el                        # 仰角（垂直角度）
                )
                
                img_name = f"{idx:06d}.png"
                
                # --- 多通道渲染循环 ---
                
                # Pass 1: Mask (Silhouette)
                # 黑色物体，白色背景
                setup_world_white()
                apply_material(obj, get_silhouette_material())
                render_image(dirs["mask"] / img_name)
                
                # Pass 2: Normal Map
                # 法线材质，白色背景
                apply_material(obj, get_normal_material())
                render_image(dirs["normal"] / img_name)
                
                # Pass 3: Clay (White Model)
                # 白模材质，需要光照
                apply_material(obj, get_clay_material())
                # 注意：Clay 渲染最好用灰色背景，避免高光与背景混淆，但为了统一这里先用白色或根据需求调整
                # 这里保持白色背景，或者可以 setup_world_grey()
                render_image(dirs["clay"] / img_name)
                
                # ---------------------

                # 获取当前相机的姿态矩阵（世界坐标系到相机坐标系的变换）
                pose = camera_world_to_camera_matrix(cam)
                
                # 记录当前视角的元数据
                # 注意：image_path 现在记录 mask 的路径作为主路径，其他路径可以推导
                records.append({
                    "branch_id": branch_id,           # 模型标识（文件名或父目录名）
                    "view_id": idx,                   # 视角ID（唯一标识）
                    "azimuth_deg": az,                # 方位角度数
                    "elevation_deg": el,              # 仰角度数
                    "camera_pose": matrix_to_list(pose),  # 相机姿态矩阵（4x4）
                    "camera_dist": cam_dist,          # 记录自适应距离
                    "image_paths": {                  # 记录多通道路径
                        "mask": str((dirs["mask"] / img_name).relative_to(output_dir(cfg))),
                        "normal": str((dirs["normal"] / img_name).relative_to(output_dir(cfg))),
                        "clay": str((dirs["clay"] / img_name).relative_to(output_dir(cfg)))
                    }
                })
                
                # 增加方位角，步长为配置文件中设定的值
                az += cfg["camera"]["azimuth_step_deg"]
                
                # 递增图像索引
                idx += 1
        
        # 将当前模型的所有视角元数据保存到JSONL文件
        write_jsonl(base_out_dir / "meta.jsonl", records)
        print(f"完成模型 {branch_id} 的渲染，共生成 {idx} 组图像 (Mask/Normal/Clay)")

if __name__ == "__main__":
    """
    脚本入口点
    当直接运行此脚本时执行main函数
    """
    main()