# Spatial Model for Branch Recognition & Pose Estimation
## 树枝识别与位姿估计空间模型工程

本工程旨在实现基于双目视觉和机械臂环境下的树枝识别、姿态估计及抓取辅助。项目分为五个阶段，从数据生成到模型部署，形成了一套完整的技术闭环。

---

## 📂 工程目录结构概览

```
d:\2026科研工作\1月\07_spatial_model\
├── README.md               # [本文档] 工程总说明
├── models/                 # [新建] 存放预训练模型权重 (如 mobile_sam.pt)
├── Onsite_picture/         # 现场采集的真实图片及测试结果
├── stage_1/                # [数据生成] 3D 渲染阶段
├── stage_2/                # [数据准备] 预处理与数据集划分
├── stage_3/                # [模型训练] Embedding 模型训练
├── stage_4/                # [向量建库] 构建 FAISS 检索库
└── stage_5/                # [推理部署] 现场评估与姿态估计
```

---

## 🛠️ 环境依赖 (Prerequisites)

请确保安装以下核心库：
```bash
pip install torch torchvision
pip install ultralytics  # YOLO & SAM
pip install faiss-cpu    # 向量检索
pip install transformers # CLIP 模型
pip install opencv-python pillow pyyaml tqdm
```

---

## 🧭 新工程师快速上手 (Onboarding)

### 你需要先理解的 3 件事
1.  **Stage 1-4 是离线资产**：生成/加工/训练/建库，属于“一次性或偶尔更新”。  
2.  **Stage 5 是在线推理**：对现场图像做分割/对齐/检索/（可选）姿态精修，属于“每次抓取都要跑”。  
3.  **检索负责给出“候选姿态”**：最终给机械臂的空间位姿，需要结合双目深度、手眼标定等几何链路才能落地。

### 常用入口脚本
- 生成合成多视角数据：`stage_1/scripts/render_dataset.py` 或 `stage_1/scripts/render_fibonacci.py`
- 制作训练集：`stage_2/scripts/prepare_dataset.py`
- 训练 Embedding：`stage_3/scripts/train.py`
- 构建向量库：`stage_4/scripts/build_index.py`
- 在线推理：`stage_5/scripts/infer_pose.py`
- 现场自动提取树枝 Mask（SAM + CLIP）：`stage_5/scripts/eval_yolo_onsite.py`

### 关键说明文档（建议按顺序阅读）
- [README.md](file:///d:/2026科研工作/1月/07_spatial_model/README.md)：项目地图与入口
- [RISK_ASSESSMENT.md](file:///d:/2026科研工作/1月/07_spatial_model/RISK_ASSESSMENT.md)：从识别到抓取的主要风险点
- [POSE_TRANSFORMATION.md](file:///d:/2026科研工作/1月/07_spatial_model/POSE_TRANSFORMATION.md)：检索结果如何转为机械臂可用的位姿链路
- [CENTER_ALIGNMENT.md](file:///d:/2026科研工作/1月/07_spatial_model/CENTER_ALIGNMENT.md)：相机不对准中心时如何“检测-裁剪-对齐”
- [IN_PLANE_ROTATION.md](file:///d:/2026科研工作/1月/07_spatial_model/IN_PLANE_ROTATION.md)：Roll（平面内旋转）问题的工程解法
- [SAMPLING_STRATEGY.md](file:///d:/2026科研工作/1月/07_spatial_model/SAMPLING_STRATEGY.md)：斐波那契采样精度（N=1000）与 PnP 工作流

---

## 🚀 阶段详解 (Pipeline Stages)

### Stage 1: 数据生成 (Data Generation)
**目标**: 将 3D 树枝模型 (`.obj`) 渲染为多视角的 2D 轮廓图。
*   **输入**: `stage_1/assets/branches_3d/*.obj`
*   **输出**: `stage_1/data/rendered/log_*/mask/*.png`
*   **核心脚本**:
    ```bash
    python stage_1/scripts/render_dataset.py
    ```
*   **配置文件**: `stage_1/configs/render.yaml`
*   **可选（均匀球面采样）**: `stage_1/scripts/render_fibonacci.py` + `stage_1/configs/render_fibonacci.yaml`

### Stage 2: 数据准备 (Data Preparation)
**目标**: 对渲染图像进行预处理（如局部裁剪、遮挡模拟），并划分为 Train/Val/Test 集。
*   **输入**: Stage 1 的渲染数据
*   **输出**: `stage_2/data/prepared/`
*   **核心脚本**:
    ```bash
    python stage_2/scripts/prepare_dataset.py
    ```
*   **配置文件**: `stage_2/config/prepare.yaml`

### Stage 3: 模型训练 (Model Training)
**目标**: 训练 ResNet-based Embedding Model，使其能提取具备几何语义的特征向量。
*   **输入**: Stage 2 的 Prepared 数据
*   **输出**: `stage_3/outputs/exp_minimal/best.ckpt` (最佳模型权重)
*   **核心脚本**:
    ```bash
    python stage_3/scripts/train.py
    ```
*   **配置文件**: `stage_3/config/train.yaml`

### Stage 4: 向量建库 (Index Building)
**目标**: 使用训练好的模型，为所有树枝的全局视角建立 FAISS 向量索引库。
*   **输入**: Stage 1 的全局数据 + Stage 3 的模型权重
*   **输出**: 
    *   `stage_4/data/index/vector.index` (FAISS 索引)
    *   `stage_4/data/index/meta.pkl` (元数据)
*   **核心脚本**:
    ```bash
    python stage_4/scripts/build_index.py
    ```
*   **配置文件**: `stage_4/config/index.yaml`

### Stage 5: 推理与部署 (Inference & Deployment)
**目标**: 在真实场景中进行树枝识别、遮挡测试和自动标注。

#### 1. 现场数据自动标注 (Auto-Labeling)
利用 **SAM (Segment Anything)** 和 **CLIP** 模型，自动从现场照片中提取树枝 Mask，无需人工标注。
*   **脚本**: `stage_5/scripts/eval_yolo_onsite.py`
*   **运行**:
    ```bash
    python stage_5/scripts/eval_yolo_onsite.py --input_dir Onsite_picture
    ```
*   **结果**: 生成的 Mask 保存在 `Onsite_picture/results_sam_clip/`，可用于微调 YOLO。

#### 2. 遮挡鲁棒性测试 (Occlusion Test)
测试模型在不同遮挡比例下的识别准确率。
*   **脚本**: `stage_5/scripts/test_occlusion.py`
*   **运行**:
    ```bash
    python stage_5/scripts/test_occlusion.py "path/to/image.png"
    ```

#### 3. 标准姿态估计流水线 (Pose Estimation Pipeline)
完整的 "预处理 -> Embedding -> 检索 -> 姿态解算" 流程。
*   **脚本**: `stage_5/scripts/infer_pose.py`
*   **配置文件**: `stage_5/configs/infer.yaml`

---

## 📝 复盘与维护建议 (Maintenance)

1.  **数据流向**: 始终遵循 `Stage 1 -> 2 -> 3 -> 4 -> 5` 的数据流向。修改上游数据后，需重新运行下游脚本。
2.  **配置管理**: 每个阶段都有独立的 `config(s)/` 文件夹，修改参数（如图像尺寸、Batch Size）请去对应 yaml 文件。
3.  **现场图片**: 建议将现场采集的图片按日期归档在 `Onsite_picture` 下，并定期使用 `eval_yolo_onsite.py` 清洗数据。
4.  **模型权重**: 推荐将下载的第三方权重（如 `mobile_sam.pt`, `yolov8n-seg.pt`）统一移动到 `models/` 目录，并在脚本中更新路径（*注：本次整理已为您创建 models 目录*）。
5.  **显卡兼容性**: 如果遇到 CUDA 架构不兼容（如 RTX 5060 / sm_120）导致的推理报错，优先在推理脚本中强制使用 CPU，保证流程可跑通。

---

*文档生成日期: 2026-01-17*
