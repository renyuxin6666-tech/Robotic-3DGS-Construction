import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# ==============================================================================
# 0. 环境与路径设置 (Setup Paths)
# ==============================================================================
# 获取当前脚本所在的目录 (stage_4/scripts)
current_dir = Path(__file__).resolve().parent
# 获取 stage_4 的根目录
stage_4_root = current_dir.parent
# 获取 stage_3 的根目录 (因为我们需要用到 stage_3 里定义的模型和数据集类)
stage_3_root = stage_4_root.parent / "stage_3"

# 将这些目录加入 Python 的搜索路径，这样我们才能 import 那些文件
sys.path.append(str(stage_3_root))
sys.path.append(str(stage_4_root))

# 导入我们可以复用的代码模块
from src.models.model import BranchEmbeddingModel  # 我们的 AI 模型 (负责看图)
from src.dataset.dataset import BranchDataset      # 数据读取器 (负责读图)
from src.indexer.faiss_engine import FaissEngine   # 向量搜索引擎 (负责找图)

def load_config(config_path):
    """
    读取 YAML 配置文件
    就像读取一个说明书，告诉程序各种参数在哪里
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def query_index():
    """
    主函数：模拟真实的检索过程
    场景：机器人看到一个局部树枝 (Local)，去数据库 (Index) 里找它是哪棵树 (Global)
    """
    
    # ==============================================================================
    # 1. 加载配置 (Load Config)
    # ==============================================================================
    config_path = stage_4_root / "config" / "index.yaml"
    config = load_config(config_path)
    
    # 检测电脑是否有显卡 (GPU)，虽然下面为了兼容性我们会强制用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"检测到设备: {device}")
    
    # ==============================================================================
    # 2. 加载向量索引库 (Load Index / Database)
    # ==============================================================================
    # 这里加载的是我们之前用 build_index.py 建立好的“底库”
    # 里面存了所有完整树枝 (Global) 的特征向量
    index_dir = (stage_4_root / config['paths']['index_dir']).resolve()
    index_path = index_dir / config['paths']['index_file']
    meta_path = index_dir / config['paths']['meta_file']
    
    # 初始化搜索引擎
    indexer = FaissEngine(
        dim=config['index']['dim'],       # 向量维度 (128维)
        index_type=config['index']['type'], # 索引类型 (Flat: 精确搜索)
        metric=config['index']['metric']    # 距离度量 (L2: 欧氏距离)
    )
    print(f"正在从 {index_path} 加载索引库...")
    indexer.load(index_path, meta_path)
    print(f"✅ 索引库加载完毕! 库里共有 {len(indexer)} 个树枝样本。")
    
    # ==============================================================================
    # 3. 加载 AI 模型 (Load Model)
    # ==============================================================================
    # 这个模型就是我们的“眼睛”，它负责把图片变成 128维的向量
    checkpoint_path = (stage_4_root / config['paths']['model_checkpoint']).resolve()
    print(f"正在加载模型权重: {checkpoint_path}")
    
    # 强制使用 CPU 运行 (为了解决 RTX 5060 显卡的兼容性问题)
    device = torch.device("cpu")
    print(f"⚠️ 使用设备: {device} (已强制切换为 CPU 以保证稳定运行)")

    # 初始化模型结构
    model = BranchEmbeddingModel(
        backbone_name=config['model']['backbone'],
        embedding_dim=config['model']['embedding_dim'],
        pretrained=False # 不需要下载预训练权重，因为我们会加载自己训练好的
    )
    
    # 读取训练好的参数
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 把模型放到 CPU 上，并开启评估模式 (eval)
    # eval 模式会关闭 Dropout 等训练时才用的功能，保证输出稳定
    model.to(device)
    model.eval()
    
    # ==============================================================================
    # 4. 准备查询数据 (Prepare Query Data)
    # ==============================================================================
    # 我们用测试集 (Test Set) 中的数据来模拟查询
    # 注意：我们稍后会只筛选出 "Local" (局部) 图片作为查询输入
    data_root = (stage_4_root / config['paths']['data_root']).resolve()
    dataset = BranchDataset(
        root_dir=data_root,
        split="test", # 使用测试集
        image_size=config['model']['image_size']
    )
    
    # 数据加载器，负责一批一批地把数据喂给程序
    dataloader = DataLoader(
        dataset,
        batch_size=32, # 一次处理 32 张图
        shuffle=False, # 顺序读取，不需要打乱
        num_workers=4  # 4个进程并行读取
    )
    
    # ==============================================================================
    # 5. 开始检索 (Run Queries)
    # ==============================================================================
    print("🚀 开始模拟检索 (Local -> Global)...")
    
    correct_top1 = 0   # 记录 Top-1 猜对的次数
    correct_top5 = 0   # 记录 Top-5 猜对的次数
    total_queries = 0  # 总共查询了多少次
    
    # with torch.no_grad(): 告诉 PyTorch 不需要计算梯度，这样更省内存、更快
    with torch.no_grad():
        # tqdm 是进度条工具，让我们看到处理进度
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            branch_ids = batch['branch_id'] # 图片对应的真实树枝 ID
            is_locals = batch['is_local']   # 标记：这张图是不是局部图
            
            # --- 关键步骤：筛选 ---
            # 我们只关心 Local (局部) 图片作为查询
            # is_locals 是一个布尔列表，True 表示是局部图
            query_mask = is_locals
            
            # 如果这一批里没有局部图，就跳过
            if not query_mask.any():
                continue
                
            # 挑出局部图和它们对应的 ID
            query_images = images[query_mask]
            query_branch_ids = np.array(branch_ids)[query_mask.cpu().numpy()]
            
            # --- 关键步骤：特征提取 ---
            # 让模型看图，算出特征向量
            embeddings = model(query_images)
            embeddings = embeddings.cpu().numpy() # 转成 numpy 数组给 FAISS 用
            
            # --- 关键步骤：库内搜索 ---
            # indexer.search 会返回：
            # D: 距离 (Distances)，越小越相似
            # results: 搜索结果的元数据 (Metadata)，包含了找到的树枝 ID
            # k=5: 我们要求返回最像的前 5 个
            D, results = indexer.search(embeddings, k=5)
            
            # --- 关键步骤：评估对不对 ---
            for i in range(len(embeddings)):
                gt_branch_id = query_branch_ids[i] # 真实答案 (Ground Truth)
                retrieved_metas = results[i]       # 检索回来的 5 个结果
                
                # 检查 Top-1 (第一名是不是对的？)
                # retrieved_metas[0] 就是最像的那张
                if retrieved_metas[0]['branch_id'] == gt_branch_id:
                    correct_top1 += 1
                    
                # 检查 Top-5 (前五名里有没有对的？)
                found = False
                for meta in retrieved_metas:
                    if meta['branch_id'] == gt_branch_id:
                        found = True
                        break
                if found:
                    correct_top5 += 1
                    
                total_queries += 1
                
                # 为了直观，打印前 5 个查询的详细结果给用户看
                if total_queries <= 5:
                    print(f"\n📝 查询示例 {total_queries}:")
                    print(f"   真实身份 (GT): {gt_branch_id}")
                    print(f"   系统猜测 (Top-5):")
                    for j, meta in enumerate(retrieved_metas):
                        # 打印排名、预测ID、以及距离分数
                        print(f"     第 {j+1} 名: {meta['branch_id']} (距离: {D[i][j]:.4f})")
                        
    # ==============================================================================
    # 6. 生成报告 (Report)
    # ==============================================================================
    # 计算百分比
    acc_top1 = correct_top1 / total_queries * 100 if total_queries > 0 else 0
    acc_top5 = correct_top5 / total_queries * 100 if total_queries > 0 else 0
    
    print("\n" + "="*50)
    print("📊 仿真结果报告 (局部查询 -> 全局底库)")
    print("="*50)
    print(f"总查询次数: {total_queries}")
    print(f"Top-1 准确率 (一击即中): {acc_top1:.2f}%")
    print(f"Top-5 准确率 (前五命中): {acc_top5:.2f}%")
    print("="*50)

if __name__ == "__main__":
    query_index()
