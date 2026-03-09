import sys
import yaml
import torch
from pathlib import Path

# 添加 stage_3 目录到 Python 路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset.dataset import BranchDataset
from src.models.model import BranchEmbeddingModel
from src.train.validate import Validator

def main():
    # 1. 加载配置
    script_dir = Path(__file__).resolve().parent
    stage3_dir = script_dir.parent
    config_path = stage3_dir / "config" / "train.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    # 2. 初始化环境
    # 强制使用 CPU (RTX 5060 兼容性问题)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 3. 加载模型
    output_dir = stage3_dir / cfg["training"]["output_dir"]
    ckpt_path = output_dir / "best.ckpt"
    
    print(f"Loading model from {ckpt_path}...")
    model = BranchEmbeddingModel(
        backbone_name=cfg["model"]["backbone"],
        pretrained=False, # 验证时不需要下载预训练权重，直接加载 checkpoint
        embedding_dim=cfg["model"]["embedding_dim"]
    ).to(device)
    
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 4. 准备数据集 (验证集和测试集)
    data_root = (stage3_dir / cfg["data"]["root_dir"]).resolve()
    
    for split in ["val", "test"]:
        print(f"\nEvaluating on {split} set...")
        dataset = BranchDataset(
            root_dir=data_root,
            split=split,
            image_size=cfg["data"]["image_size"]
        )
        
        if len(dataset) == 0:
            print(f"Skipping empty split: {split}")
            continue
            
        # 5. 执行验证
        validator = Validator(model, dataset, device=device)
        
        # 5.1 检索性能
        retrieval_res = validator.evaluate_retrieval(top_k=(1, 5))
        print(f"Retrieval Metrics: {retrieval_res}")
        
        # 5.2 局部-全局一致性
        l2g_res = validator.evaluate_local_to_global()
        print(f"Local-to-Global Metrics: {l2g_res}")

if __name__ == "__main__":
    main()
