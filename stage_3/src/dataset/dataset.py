import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class BranchDataset(Dataset):
    """
    树枝数据集
    从 stage_2 的 index.jsonl 读取数据
    """
    def __init__(self, root_dir, split="train", image_size=224):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        
        # 读取索引文件
        self.index_path = self.split_dir / "index.jsonl"
        self.samples = self._load_index()
        
        # 建立 branch_id 到 数字 label 的映射
        self.branch_to_label = self._build_label_map()
        
        # 图像预处理
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            # 归一化 (使用 ImageNet 均值方差，因为用的是预训练 ResNet)
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_index(self):
        samples = []
        with open(self.index_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def _build_label_map(self):
        # 收集所有唯一的 branch_id
        branch_ids = sorted(list(set(s["branch_id"] for s in self.samples)))
        return {bid: i for i, bid in enumerate(branch_ids)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. 读取图像
        # image_path 是相对路径 (相对于 stage_2/data/prepared)
        # 所以我们需要拼接 stage_2 的 root
        # 注意：这里的 root_dir 是传入的参数，指向 stage_2/data/prepared
        img_path = self.root_dir / item["image_path"]
        
        # 确保转为 RGB (即使是黑白图，模型也通常接受 3 通道)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # 2. 获取标签
        label = self.branch_to_label[item["branch_id"]]
        
        pose = item.get("pose", [])
        if pose:
            pose = torch.tensor(pose, dtype=torch.float32)
        else:
            pose = torch.eye(4, dtype=torch.float32) # Default identity if missing

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "branch_id": item["branch_id"],
            "is_local": item["is_local"],
            "pose": pose,
            "view_id": item.get("view_id", -1)
        }
