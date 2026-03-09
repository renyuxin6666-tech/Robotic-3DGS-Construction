# Fibonacci Sampling & PnP Workflow Strategy
## 斐波那契球面采样与 PnP 姿态解算策略

本文档详细解释了为什么选择 **Fibonacci Lattice (N=1000)** 作为采样策略，以及它如何为后续的 PnP 精准抓取提供支持。

---

## 1. 采样策略核心参数 (Sampling Precision)

### 1.1 推荐参数
*   **采样算法**: Fibonacci Lattice (斐波那契球面均匀采样)
*   **采样点数 ($N$)**: **1000**
*   **覆盖范围**: 完整球面 ($4\pi$ steradians)

### 1.2 为什么是 N=1000？ (The "Sweet Spot")

#### A. 角度分辨率 (Angular Resolution)
在球面上均匀分布 $N$ 个点，相邻两点之间的平均角距离 $\theta$ 近似为：
$$ \theta \approx \sqrt{\frac{4\pi}{N}} \text{ (radians)} $$

*   当 $N=100$: $\theta \approx 20.3^\circ$ (太稀疏，检索误差大)
*   当 $N=500$: $\theta \approx 9.1^\circ$ (接近之前的经纬度网格)
*   **当 $N=1000$: $\theta \approx 6.4^\circ$**
    *   这意味着：真实世界中任意视角的树枝，在库中一定能找到一个 **偏差 < 3.2°** 的模板。
    *   3.2° 的初始误差对于 PnP 算法（如 Levenberg-Marquardt）来说是**极佳的初始值**，几乎保证收敛。

#### B. 数据规模 (Data Scale)
*   您的 `log_model` 目录下有约 100 根树枝。
*   总图片量 = $100 \text{ branches} \times 1000 \text{ views} = 100,000$ 张。
*   这是一个非常健康的训练集规模（ImageNet 是 100 万张），既能训练出鲁棒的 Embedding 模型，又不会导致训练时间过长（单卡几小时即可）。

---

## 2. 从检索到 PnP 的工作流 (Workflow)

我们不是用“大模型”直接算出 6D 姿态，而是采用 **"Coarse-to-Fine" (由粗到精)** 的策略。

### Step 1: 粗定位 (Coarse Retrieval)
*   **输入**: 机械臂拍摄的树枝轮廓图（经过检测+裁剪）。
*   **动作**: 在 N=1000 的库中进行 Embedding 检索。
*   **输出**: Top-1 匹配结果。
    *   **Branch ID**: 确定是哪根树枝（例如 `model_42`）。
    *   **Coarse Pose ($R_{init}$)**: 确定大概转了多少度（误差 < 3.2°）。

### Step 2: 特征匹配 (Feature Matching)
*   **已知**:
    *   3D 模型: `model_42.obj`
    *   初始姿态: $R_{init}$
*   **动作**:
    1.  将 3D 模型按 $R_{init}$ 投影到 2D 平面上，得到“虚拟轮廓”。
    2.  将“虚拟轮廓”与“真实轮廓”进行 ICP (Iterative Closest Point) 匹配或边缘特征点匹配。
*   **输出**: 一组 2D-3D 对应点对 $(u_i, v_i) \leftrightarrow (X_i, Y_i, Z_i)$。

### Step 3: PnP 精解算 (Fine PnP)
*   **输入**:
    *   2D-3D 点对
    *   相机内参 $K$
    *   初始猜测 $R_{init}$
*   **动作**: 调用 `cv2.solvePnP` (使用 `SOLVEPNP_ITERATIVE` 方法)。
*   **输出**: 精确的位姿 $R_{final}, t_{final}$。
*   **精度**: 此时的旋转误差通常可以压到 < 1°，满足精密抓取需求。

---

## 3. 斐波那契采样算法原理 (Implementation Details)

我们将在 `render_fibonacci.py` 中使用以下公式生成相机坐标：

对于 $i = 0, 1, ..., N-1$:
1.  $z = (2i - 1) / N - 1$  (Z 轴均匀分布)
2.  $r = \sqrt{1 - z^2}$    (XY 平面半径)
3.  $\phi = 2\pi \cdot i \cdot (1 - \frac{1}{\sqrt{5}})$ (黄金角增量)
4.  $x = r \cdot \cos(\phi)$
5.  $y = r \cdot \sin(\phi)$

生成的 $(x, y, z)$ 即为单位球上的相机位置向量。

---

## 4. 结论

**N=1000 的斐波那契采样是连接“深度学习识别”与“传统几何控制”的最佳桥梁。**
它以极小的存储代价，换取了 PnP 算法所需的高质量初始值。

*文档生成日期: 2026-01-17*
