import torch
import torch.nn as nn
import torchvision.models as models

class BranchEmbeddingModel(nn.Module):
    """
    基础 Embedding 模型
    结构：Backbone (ResNet) -> Flatten -> MLP -> L2 Norm
    """
    def __init__(self, backbone_name="resnet18", pretrained=True, embedding_dim=128):
        super().__init__()
        
        # 1. 加载 Backbone
        if backbone_name == "resnet18":
            # 使用官方 torchvision 实现
            # weights="DEFAULT" 等同于 pretrained=True
            weights = "DEFAULT" if pretrained else None
            resnet = models.resnet18(weights=weights)
            
            # 去掉最后的全连接层 (fc)
            # ResNet18 的 fc 输入维度是 512
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = 512
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not supported yet.")
            
        # 2. Embedding Head (MLP)
        # 将特征映射到目标维度的单位球面上
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x):
        # x shape: [B, 3, H, W]
        features = self.backbone(x) # [B, 512, 1, 1]
        embeddings = self.head(features) # [B, embedding_dim]
        
        # 3. L2 归一化
        # 这一步非常重要，保证所有向量都在单位超球面上
        # 这样 Euclidean Distance 就等价于 Cosine Similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
