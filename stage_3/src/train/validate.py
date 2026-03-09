import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

class Validator:
    """
    模型验证器
    功能：
    1. 计算 Embedding 检索准确率 (Top-1, Top-5)
    2. 验证局部-全局一致性
    """
    def __init__(self, model, dataset, device="cpu", batch_size=32):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        
        self.loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
    def extract_features(self):
        """提取整个数据集的特征"""
        self.model.eval()
        embeddings_list = []
        labels_list = []
        is_local_list = []
        
        with torch.no_grad():
            for batch in tqdm(self.loader, desc="Extracting features"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                is_local = batch["is_local"]
                
                # Forward
                embs = self.model(images)
                
                embeddings_list.append(embs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
                is_local_list.extend(is_local)
                
        return (
            np.concatenate(embeddings_list, axis=0),
            np.concatenate(labels_list, axis=0),
            np.array(is_local_list)
        )
        
    def evaluate_retrieval(self, top_k=(1, 5)):
        """
        评估检索性能
        对于每个样本，在剩余样本中查找最近邻，看是否属于同一树枝
        """
        embeddings, labels, is_local = self.extract_features()
        n_samples = len(labels)
        
        # 计算相似度矩阵
        # [N, N]
        sim_matrix = np.dot(embeddings, embeddings.T)
        
        # 将对角线（自己与自己）设为极小值，避免检索到自己
        np.fill_diagonal(sim_matrix, -1.0)
        
        # 统计指标
        accuracies = {k: 0 for k in top_k}
        
        # 遍历每个样本
        for i in range(n_samples):
            # 获取相似度排序索引 (从大到小)
            # argsort 返回从小到大的索引，所以取最后 k 个并倒序
            sorted_indices = np.argsort(sim_matrix[i])[::-1]
            
            true_label = labels[i]
            
            for k in top_k:
                # 取前 k 个最近邻
                top_indices = sorted_indices[:k]
                pred_labels = labels[top_indices]
                
                # 如果前 k 个中只要有一个是同类，就算命中 (Recall@K)
                if true_label in pred_labels:
                    accuracies[k] += 1
                    
        # 计算平均准确率
        results = {f"Recall@{k}": acc / n_samples for k, acc in accuracies.items()}
        return results

    def evaluate_local_to_global(self):
        """
        评估局部-全局一致性
        仅使用局部样本作为 Query，全局样本作为 Gallery
        """
        embeddings, labels, is_local = self.extract_features()
        
        # 分离局部和全局
        local_mask = is_local
        global_mask = ~is_local
        
        if not np.any(local_mask) or not np.any(global_mask):
            return {"Local2Global": 0.0}
            
        local_embs = embeddings[local_mask]
        local_labels = labels[local_mask]
        
        global_embs = embeddings[global_mask]
        global_labels = labels[global_mask]
        
        # 计算相似度: [N_local, N_global]
        sim_matrix = np.dot(local_embs, global_embs.T)
        
        # 统计 Top-1 准确率
        correct = 0
        n_local = len(local_labels)
        
        for i in range(n_local):
            # 找到最相似的全局样本索引
            best_idx = np.argmax(sim_matrix[i])
            pred_label = global_labels[best_idx]
            
            if pred_label == local_labels[i]:
                correct += 1
                
        return {"Local2Global_Acc": correct / n_local}
