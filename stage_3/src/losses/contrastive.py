import torch
import torch.nn as nn

class SupervisedContrastiveLoss(nn.Module):
    """
    有监督对比损失 (SupConLoss)
    
    原理：
    让同一类别的样本（同树枝）在特征空间中靠近，
    让不同类别的样本（不同树枝）在特征空间中远离。
    
    参考: "Supervised Contrastive Learning", NeurIPS 2020
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        features: [batch_size, dim] - 已经过 L2 归一化的特征
        labels: [batch_size] - 类别标签 (branch_id 对应的数字)
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 1. 计算相似度矩阵 (Cosine Similarity)
        # sim_matrix[i, j] = features[i] dot features[j]
        sim_matrix = torch.matmul(features, features.T)
        
        # 2. 构建 mask
        # labels: [B] -> [B, 1]
        labels = labels.contiguous().view(-1, 1)
        # mask[i, j] = 1 if label[i] == label[j]
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 3. 移除对角线 (自己与自己的相似度不计入)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 4. 计算 Logits
        # 除以温度系数
        logits = sim_matrix / self.temperature
        
        # 为了数值稳定性，减去每行的最大值
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # 5. 计算分母 (所有负样本 + 其他正样本)
        # exp_logits: [B, B]
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # 6. 计算均值
        # 只考虑 mask 为 1 的部分 (即正样本对)
        # sum(log_prob * mask) / sum(mask)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss
