# 阶段 4：向量检索与离线建库

## 1. 目标
将 Stage 3 训练好的 Embedding 模型转化为实际可用的检索服务。
模拟真实场景：
1. **离线建库 (Offline Indexing)**：把所有树枝的“标准照”（全局图）转换成向量，存入 FAISS。
2. **在线检索 (Online Query)**：给一张新的局部图，快速找出它是哪根树枝、哪个姿态。

## 2. 文件夹结构
```
stage_4/
├── config/
│   └── index.yaml       # 索引配置（FAISS 参数、库路径）
├── data/
│   └── index/           # 存放生成的 .index 文件和元数据
│       ├── vector.index # FAISS 索引文件
│       └── meta.pkl     # 对应的元数据（branch_id, pose 等）
├── scripts/
│   ├── build_index.py   # 建库脚本 (只用 Global 样本)
│   └── query_index.py   # 检索脚本 (模拟真实调用)
└── src/
    └── indexer/
        └── faiss_engine.py # FAISS 封装类
```

## 3. 核心逻辑
*   **Gallery (库)**: 只包含 `is_local=False` 的全局样本。这是我们的“底库”。
*   **Query (查)**: 使用 `is_local=True` 的局部样本进行测试。
*   **Pose Estimation**: 检索不仅返回 branch_id，还返回 Top-1 的 `camera_pose`。这是一种非参数化的姿态估计方法（Nearest Neighbor Pose Estimation）。
