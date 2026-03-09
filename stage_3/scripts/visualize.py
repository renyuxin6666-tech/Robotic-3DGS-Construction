import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.manifold import TSNE

# 添加 stage_3 目录到 Python 路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset.dataset import BranchDataset
from src.models.model import BranchEmbeddingModel
from src.train.validate import Validator

def denormalize(tensor):
    """将 Tensor 反归一化为 0-1 之间的图像，用于显示"""
    # Mean/Std from ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.cpu().numpy().transpose(1, 2, 0) # CHW -> HWC
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def visualize_retrieval(model, dataset, device, num_queries=5, output_path="retrieval_viz.png"):
    """
    可视化检索结果
    随机抽取 num_queries 个样本作为 Query
    显示它们在库中找到的 Top-5 最相似图片
    """
    validator = Validator(model, dataset, device=device)
    embeddings, labels, is_local = validator.extract_features()
    
    # 随机选择 Query
    indices = np.random.choice(len(dataset), num_queries, replace=False)
    
    # 计算相似度矩阵
    sim_matrix = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(sim_matrix, -1.0) # 排除自己
    
    # 绘图
    # 行 = Query, 列 = Query图 + Top-5 Retrieved
    fig, axes = plt.subplots(num_queries, 6, figsize=(15, 3 * num_queries))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    for i, query_idx in enumerate(indices):
        # 1. 显示 Query
        query_img = dataset[query_idx]["image"]
        query_label = labels[query_idx]
        is_loc = is_local[query_idx]
        
        ax = axes[i, 0]
        ax.imshow(denormalize(query_img))
        ax.set_title(f"Query\nLabel: {query_label}\n{'Local' if is_loc else 'Global'}", color="blue", fontsize=10)
        ax.axis("off")
        
        # 2. 找到 Top-5
        sorted_indices = np.argsort(sim_matrix[query_idx])[::-1]
        top5_indices = sorted_indices[:5]
        
        for k, match_idx in enumerate(top5_indices):
            match_img = dataset[match_idx]["image"]
            match_label = labels[match_idx]
            match_score = sim_matrix[query_idx][match_idx]
            match_is_local = is_local[match_idx]
            
            ax = axes[i, k+1]
            ax.imshow(denormalize(match_img))
            
            # 如果 Label 一致，标题为绿色；否则为红色
            color = "green" if match_label == query_label else "red"
            title = f"Top-{k+1}\nLabel: {match_label}\nSim: {match_score:.2f}\n{'Local' if match_is_local else 'Global'}"
            ax.set_title(title, color=color, fontsize=10)
            ax.axis("off")
            
    plt.suptitle("Retrieval Visualization (Green=Correct, Red=Wrong)", fontsize=16)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved retrieval visualization to {output_path}")

def visualize_tsne(model, dataset, device, output_path="tsne_viz.png"):
    """
    可视化 t-SNE 降维结果
    展示 Embedding 在 2D 空间中的分布，不同颜色代表不同树枝
    """
    validator = Validator(model, dataset, device=device)
    embeddings, labels, is_local = validator.extract_features()
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    # 获取唯一类别
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        # 筛选当前类别的点
        mask = labels == label
        
        # 区分 Global 和 Local
        global_mask = mask & (~is_local)
        local_mask = mask & is_local
        
        # 绘制 Global (圆形)
        plt.scatter(
            embeddings_2d[global_mask, 0], 
            embeddings_2d[global_mask, 1],
            color=colors[i],
            marker='o',
            s=50,
            label=f"Branch {label}" if i < 10 else None, # 图例只显示前10个
            alpha=0.8,
            edgecolors='w'
        )
        
        # 绘制 Local (三角形)
        plt.scatter(
            embeddings_2d[local_mask, 0], 
            embeddings_2d[local_mask, 1],
            color=colors[i],
            marker='^',
            s=30,
            alpha=0.6
        )
        
    plt.title("t-SNE Visualization of Branch Embeddings (Circle=Global, Triangle=Local)", fontsize=14)
    if len(unique_labels) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved t-SNE visualization to {output_path}")

def main():
    # 1. 加载配置
    script_dir = Path(__file__).resolve().parent
    stage3_dir = script_dir.parent
    config_path = stage3_dir / "config" / "train.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cpu")
    
    # 2. 加载模型
    output_dir = stage3_dir / cfg["training"]["output_dir"]
    ckpt_path = output_dir / "best.ckpt"
    
    model = BranchEmbeddingModel(
        backbone_name=cfg["model"]["backbone"],
        pretrained=False,
        embedding_dim=cfg["model"]["embedding_dim"]
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    # 3. 加载测试集
    data_root = (stage3_dir / cfg["data"]["root_dir"]).resolve()
    test_dataset = BranchDataset(
        root_dir=data_root,
        split="test",
        image_size=cfg["data"]["image_size"]
    )
    
    # 4. 执行可视化
    viz_retrieval_path = stage3_dir / "outputs" / "retrieval_viz.png"
    visualize_retrieval(model, test_dataset, device, num_queries=5, output_path=str(viz_retrieval_path))
    
    viz_tsne_path = stage3_dir / "outputs" / "tsne_viz.png"
    visualize_tsne(model, test_dataset, device, output_path=str(viz_tsne_path))

if __name__ == "__main__":
    main()
