# Stage 3 Branch Embedding Model

这是一个用于提取树枝几何语义特征的轻量级 Embedding 模型。该模型基于 ResNet18，将输入图像映射到 128 维的单位超球面上。

## 目录结构
- `best.ckpt`: 训练好的模型权重 (PyTorch Lightning Checkpoint)。
- `model.py`: 模型网络结构定义。
- `inference.py`: 推理演示脚本。

## 环境依赖
运行此模型需要以下 Python 库：
- torch
- torchvision
- pillow (PIL)

## 快速开始

### 1. 加载模型
你可以直接使用 `inference.py` 中的 `load_model` 函数：

```python
from inference import load_model

model = load_model("best.ckpt")
```

### 2. 数据预处理
模型输入要求如下：
- 尺寸: resize 到 **(224, 224)**
- 格式: RGB
- 归一化: 使用 ImageNet 均值和方差
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

### 3. 输出
模型输出为 **128维** 的特征向量，并且已经经过 L2 归一化（模长为 1）。
因此，两个向量之间的相似度可以直接通过点积（Cosine Similarity）计算。

## 模型参数
- **Backbone**: ResNet18 (fc layer removed)
- **Input Size**: 224x224
- **Output Dim**: 128
- **Metric**: Euclidean / Cosine (L2 Normalized)