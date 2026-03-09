import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import sys

# 导入同目录下的模型定义
from model import BranchEmbeddingModel

def load_model(ckpt_path, device="cpu"):
    """
    加载训练好的模型
    """
    # 1. 初始化模型结构 (参数需与训练配置一致)
    model = BranchEmbeddingModel(
        backbone_name="resnet18",
        pretrained=False, # 推理时不需要下载预训练权重，因为我们要加载 ckpt
        embedding_dim=128
    )
    
    # 2. 加载权重
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 检查 checkpoint 结构
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
             # 格式 1: 云端新训练的格式 (包含 optimizer 等)
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            # 格式 2: PyTorch Lightning 或其他封装格式
            state_dict = checkpoint["state_dict"]
        else:
            # 格式 3: 可能直接就是 state_dict
            state_dict = checkpoint
    else:
        # 格式 4: 如果直接保存了 state_dict 对象
        state_dict = checkpoint

    # 处理可能的键名前缀 (移除 "model." 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."): 
            new_state_dict[k.replace("model.", "")] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def get_transform():
    """
    获取预处理转换 (需与训练时保持一致)
    """
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # ImageNet 标准归一化
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_embedding(model, image_path, device="cpu"):
    """
    提取单张图片的特征向量
    """
    transform = get_transform()
    
    # 读取并处理图片
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) # [1, 3, 224, 224]
    
    # 推理
    with torch.no_grad():
        embedding = model(image_tensor)
        
    return embedding.cpu().numpy()[0]

if __name__ == "__main__":
    # 配置
    current_dir = Path(__file__).parent
    ckpt_path = current_dir / "best.ckpt"
    
    # 自动选择设备 (优先 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 1. 加载模型
        model = load_model(ckpt_path, device)
        print("Model loaded successfully!")
        
        # 2. 示例：创建一个伪造的图片进行测试 (或者替换为真实图片路径)
        # dummy_image_path = "path/to/your/image.jpg" 
        # 这里为了演示直接创建一个随机噪声图
        print("Running test inference on dummy data...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            embedding = model(dummy_input)
            
        print(f"Output Embedding Shape: {embedding.shape}")
        print(f"Output Vector (First 10 dims): {embedding[0, :10]}")
        print("\nSUCCESS: Model is ready for use.")
        
    except FileNotFoundError:
        print("Error: best.ckpt not found. Please run the setup script first.")
    except Exception as e:
        print(f"An error occurred: {e}")