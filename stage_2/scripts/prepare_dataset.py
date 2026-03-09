import sys
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# 添加 stage_2 目录到 Python 路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_yaml, write_jsonl
from src.utils.paths import get_stage1_dir, get_output_dir
from src.readers.stage01_reader import Stage1Reader
from src.local_views.crop import RandomCropper

def main():
    # 1. 加载配置
    script_dir = Path(__file__).resolve().parent
    stage2_dir = script_dir.parent
    config_path = stage2_dir / "config" / "prepare.yaml"
    cfg = load_yaml(config_path)
    
    # 设置随机种子
    random.seed(cfg["processing"]["split"]["seed"])
    
    # 2. 准备路径
    stage1_root = get_stage1_dir(cfg)
    output_root = get_output_dir(cfg)
    print(f"输入目录: {stage1_root}")
    print(f"输出目录: {output_root}")
    
    # 3. 初始化 Reader 和 Cropper
    reader = Stage1Reader(stage1_root)
    cropper = RandomCropper(
        scale_range=cfg["processing"]["crop"]["scale_range"],
        min_foreground_ratio=cfg["processing"]["crop"]["min_foreground_ratio"]
    )
    
    # 4. 获取所有树枝并划分数据集 (Train/Val/Test)
    branches = reader.get_all_branches()
    random.shuffle(branches)
    
    n_total = len(branches)
    n_train = int(n_total * cfg["processing"]["split"]["train"])
    n_val = int(n_total * cfg["processing"]["split"]["val"])
    
    splits = {
        "train": branches[:n_train],
        "val": branches[n_train:n_train+n_val],
        "test": branches[n_train+n_val:]
    }
    
    print(f"数据集划分: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    # 5. 开始处理
    for split_name, split_branches in splits.items():
        print(f"正在处理 {split_name} 集...")
        split_dir = output_root / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        index_records = []
        
        for branch_id in tqdm(split_branches):
            # 读取该树枝的所有全局视角数据
            meta_records = reader.read_branch_meta(branch_id)
            
            for meta in meta_records:
                # 原始图像路径 (Mask 通道)
                # 注意：cfg["dataset"]["channels"] 是列表，这里我们默认取 mask
                if "mask" not in meta["image_paths"]:
                    continue
                    
                src_path = meta["image_paths"]["mask"]
                view_id = meta["view_id"]
                
                # --- A. 处理全局样本 ---
                # 将原始图片复制到新目录 (重命名以保证唯一性)
                # 命名格式: {branch_id}_view_{view_id}_full.png
                full_img_name = f"{branch_id}_view_{view_id:06d}_full.png"
                dst_full_path = images_dir / full_img_name
                shutil.copy(src_path, dst_full_path)
                
                # 记录全局样本索引
                index_records.append({
                    "sample_id": f"{branch_id}_{view_id}_full",
                    "image_path": str(dst_full_path.relative_to(output_root)), # 存相对路径
                    "branch_id": branch_id,
                    "view_id": view_id,
                    "is_local": False,
                    "visible_ratio": 1.0,
                    "pose": meta["camera_pose"], # 继承位姿
                    "azimuth_deg": meta.get("azimuth_deg"),
                    "elevation_deg": meta.get("elevation_deg")
                })
                
                # --- B. 生成局部样本 ---
                if cfg["processing"]["crop"]["enabled"]:
                    crops = cropper.crop(
                        src_path, 
                        num_crops=cfg["processing"]["crop"]["num_crops_per_view"]
                    )
                    
                    for i, crop_data in enumerate(crops):
                        # 保存裁剪后的图片
                        # 命名格式: {branch_id}_view_{view_id}_local_{i}.png
                        local_img_name = f"{branch_id}_view_{view_id:06d}_local_{i}.png"
                        dst_local_path = images_dir / local_img_name
                        crop_data["image"].save(dst_local_path)
                        
                        # 记录局部样本索引
                        index_records.append({
                            "sample_id": f"{branch_id}_{view_id}_local_{i}",
                            "image_path": str(dst_local_path.relative_to(output_root)),
                            "branch_id": branch_id,
                            "view_id": view_id,
                            "is_local": True,
                            "crop_box": crop_data["box"],   # [x, y, w, h]
                            "crop_scale": crop_data["scale"],
                            "pose": meta["camera_pose"],    # 继承位姿 (关键!)
                            "azimuth_deg": meta.get("azimuth_deg"),
                            "elevation_deg": meta.get("elevation_deg")
                        })
        
        # 保存该 Split 的索引文件
        write_jsonl(split_dir / "index.jsonl", index_records)
        print(f"完成 {split_name} 集: 生成了 {len(index_records)} 个样本")

if __name__ == "__main__":
    main()
