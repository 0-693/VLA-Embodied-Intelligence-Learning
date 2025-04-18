

## SpatialVLA 的 Action Head 设计详解

在 SpatialVLA 模型中，**Action Head（动作头）** 采用了 **自适应空间动作网格(Adaptive Spatial Action Grids)** 的设计，用于将机器人连续动作离散化为可学习的 token，并作为模型输出。该机制不仅提升了动作建模的表达能力，还提高了推理效率与跨机器人适配能力。

---

### 一、核心思想

SpatialVLA 将机器人的每一步动作拆分为以下三部分：

| 类型       | 维度                     | 含义                            |
|------------|--------------------------|---------------------------------|
| 平移动作 ΔT | \(x, y, z\) → \(\phi, \theta, r\) | 用球坐标建模位置移动方向与距离 |
| 旋转动作 ΔR | roll, pitch, yaw         | 表示姿态变化                    |
| 抓取动作 G  | gripper (0/1)            | 表示夹爪开/合                   |

这些连续动作维度均被 **归一化到 [-1, 1] 区间**，并通过统计整体数据集分布进行 **高斯建模**，进而划分出**离散动作网格（grids）**。

---

### 二、Adaptive Action Grids 构造过程

1. **拟合分布**：对每个动作维度进行高斯分布拟合 \( \mathcal{N}(\mu_a, \Sigma_a) \)  
2. **等概率离散化**：将每个维度划分为 \(M\) 个区间，使每个区间的概率密度积分为 \( \frac{1}{M} \)
3. **生成 token**：每个区间对应一个离散 token，最终得到：
   - \(M_{\phi} \times M_{\theta} \times M_r\) 个平移 token
   - \(M_{\text{roll}} \times M_{\text{pitch}} \times M_{\text{yaw}}\) 个旋转 token
   - 2 个 gripper token（开/合）

最终形成**总计 8194 个 token**，并嵌入为可训练向量表示（Embedding）。

---

### 三、Action Head 的输入输出流程

#### 编码（训练时）：
- 将动作 \((\Delta T, \Delta R, G)\) → 转换为 token id → 输入作为训练目标
- 损失函数为标准的交叉熵：  
  \[
  \mathcal{L} = \mathbb{E}_{p(A_t|o_t)} [\text{CrossEntropy}(a_t, \tilde{a}_t)]
  \]

#### 解码（推理时）：
- 由模型输出 3 个 token：translation token、rotation token、gripper token
- 使用网格索引反推得到连续动作向量 \((x, y, z, \text{roll, pitch, yaw}, g)\)

---

### 四、与其他模型对比

| 模型       | Action Head类型           | 动作维度 | 每步Token数 | 是否空间建模 | 是否自适应分布 |
|------------|----------------------------|-----------|--------------|----------------|------------------|
| RT-1       | 固定256-bin离散化          | 11        | 7+            | 否             | 否               |
| RT-2       | 固定256-bin离散化 + LLM输出 | 8         | 7             | 否             | 否               |
| OpenVLA    | 固定Bin + LLM              | 7         | 7             | 否             | 否               |
| **SpatialVLA** | **Adaptive Action Grids + LLM** | 7         | **3**         | **是**          | **是**            |

---

### 五、设计优势总结

- **空间感知增强**：引入 Ego3D 坐标编码，对动作与观察均建立 3D 结构关联；
- **推理效率高**：只需生成 3 个 token（相比传统 7 个 token）→ 推理速度提升至 20Hz；
- **跨机器人迁移强**：通过后训练阶段的 re-discretization 支持低成本适配新机器人；
- **鲁棒性强**：在各类视角、光照、背景变化任务中泛化能力显著优于其他模型。

