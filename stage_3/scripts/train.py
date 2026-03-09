import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# 添加 stage_3 目录到 Python 路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset.dataset import BranchDataset
from src.models.model import BranchEmbeddingModel
from src.losses.contrastive import SupervisedContrastiveLoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # 1. 加载配置
    script_dir = Path(__file__).resolve().parent
    stage3_dir = script_dir.parent
    config_path = stage3_dir / "config" / "train.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    # 2. 初始化环境
    set_seed(cfg["training"]["seed"])
    
    # 自动选择设备 (优先使用 GPU)
    # 注意：RTX 5060 Laptop (Compute Capability 12.0/sm_120) 目前 PyTorch 官方预编译包尚未完全支持
    # 会导致 RuntimeError: CUDA error: no kernel image is available for execution on the device
    # 因此暂时强制回退到 CPU，直到 PyTorch 发布支持 sm_120 的版本
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     # 启用 cudnn benchmark 以加速
    #     torch.backends.cudnn.benchmark = True
    #     print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    # else:
    device = torch.device("cpu")
    print(f"Using device: {device} (Forced CPU due to RTX 5060/sm_120 compatibility issue)")
    
    output_dir = stage3_dir / cfg["training"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 准备数据
    # 注意：stage_3 的 root_dir 是相对于 stage_3 目录的
    # config 中是 "../stage_2/data/prepared"
    data_root = (stage3_dir / cfg["data"]["root_dir"]).resolve()
    
    train_dataset = BranchDataset(
        root_dir=data_root,
        split="train",
        image_size=cfg["data"]["image_size"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True, # 必须打乱，否则 contrastive loss 无法采样负样本
        num_workers=0, # Windows下多进程可能报错，改为0
        pin_memory=True,
        drop_last=True # 丢弃最后一个不完整的 batch，避免 batchNorm 报错
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    
    # 4. 准备模型
    model = BranchEmbeddingModel(
        backbone_name=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        embedding_dim=cfg["model"]["embedding_dim"]
    ).to(device)
    
    # 5. 准备优化器和损失函数
    criterion = SupervisedContrastiveLoss(
        temperature=cfg["loss"]["temperature"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"]
    )
    
    # 6. 训练循环
    best_loss = float("inf")
    start_epoch = 0
    
    # 尝试断点续训
    last_ckpt_path = output_dir / "last.ckpt"
    if last_ckpt_path.exists():
        print(f"Found checkpoint at {last_ckpt_path}, resuming...")
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        
        # 兼容旧版本 (只保存了 state_dict)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            print(f"Resuming from epoch {start_epoch}")
        else:
            # 旧版本 checkpoint，只能加载权重
            model.load_state_dict(checkpoint)
            print("Warning: Legacy checkpoint detected. Resuming weights only (optimizer reset, epoch unknown).")
            
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward
            embeddings = model(images)
            
            # Loss
            loss = criterion(embeddings, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # 保存状态字典 (包含 optimizer 和 epoch)
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss
        }
        
        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 更新 best_loss 并保存
            checkpoint_state["best_loss"] = best_loss
            torch.save(checkpoint_state, output_dir / "best.ckpt")
            print(f"Saved best model with loss {best_loss:.4f}")
            
        # 保存最后模型
        torch.save(checkpoint_state, output_dir / "last.ckpt")

if __name__ == "__main__":
    main()
