

## π0 的 Action Head 设计详解

π0 是一个融合了预训练视觉语言模型（VLM）与高频控制能力的机器人基础模型。其核心组件是一个 **Flow Matching Action Head（流匹配动作头）**，该模块通过学习连续控制动作分布，实现对高精度、长时序、多模态动作的高效建模。

---

### 一、核心思想

π0 的动作头不再使用离散 token 输出或标准回归，而是通过 **条件流匹配（Conditional Flow Matching）** 模型在连续动作空间中进行建模，每次预测一段完整的动作序列（称为 Action Chunk）。

| 控制维度      | 含义                                             |
|---------------|--------------------------------------------------|
| 动作 Chunk At | 50 步（H=50）的连续控制序列                      |
| 每步动作 at   | 通常为 7–18 维（包含单臂/双臂/移动底盘等配置）     |
| 观察 ot       | 图像 + 文本 + 机器人状态                         |

---

### 二、Action Head 结构与机制

#### 动作流建模流程

- 使用 PaliGemma (3B) 作为主干 VLM；
- 添加一个独立的 **Action Expert 模块（300M 参数）**，负责动作 token 的建模；
- 对每一个动作 at，采用如下流程：

  - 对动作加入噪声 \( A_t^\tau = \tau A_t + (1 - \tau) \epsilon \)
  - 输入 noisy 动作与观察 \( o_t \) 进入 transformer；
  - 输出 denoising 向量 \( v_\theta(A^\tau_t, o_t) \)；
  - 监督信号为 \( u(A^\tau_t | A_t) = \epsilon - A_t \)

- 损失函数为 Flow Matching 损失：
  \[
  \mathcal{L}_\tau = \mathbb{E} \left[ \|v_\theta(A^\tau_t, o_t) - u(A^\tau_t | A_t)\|^2 \right]
  \]

#### 时间步采样：

- 从 Beta 分布中采样 flow matching 时间步 τ，强调低噪声（高还原）区域；
- 推理阶段用欧拉积分逐步去噪，10 步恢复完整动作。

---

### 三、训练与推理流程

#### 训练阶段：

- 对每个 action chunk 加入高斯噪声；
- 用 transformer + action expert 预测去噪向量；
- 优化目标是逼近真实 denoising vector；
- 使用 blockwise attention（图像+语言、状态、动作分区）。

#### 推理阶段：

- 从高斯噪声采样 \( A_0 \sim \mathcal{N}(0, I) \)；
- 使用 learned vector field 进行 10 步积分：
  \[
  A_{\tau+\delta} = A_\tau + \delta \cdot v_\theta(A_\tau, o)
  \]
- 输出完整的动作 chunk（如 50 步）用于控制器执行。

---

### 四、与其他模型对比（Action Head）

| 模型       | Action Head 类型           | 输出形式        | 是否自回归 | 支持高频控制 | 多模态表达能力 |
|------------|-----------------------------|------------------|--------------|----------------|------------------|
| RT-2       | token 离散化 + 自回归        | 逐 token 输出     | 是          | 否             | 中               |
| Octo       | diffusion + MLP 生成动作块    | 动作 chunk（如 L=4） | 否       | 是             | 是               |
| TinyVLA    | 单步扩散策略头               | 单帧动作          | 否           | 是             | 中               |
| **π0**     | **Flow Matching + Action Expert** | **动作 chunk（H=50）** | 否       | 是（50Hz）  | 是（精确表达） |

---

### 五、设计优势总结

- **高频控制能力**：动作 chunk 输出支持 50Hz 控制；
- **强多模态建模**：结合图像、语言与状态进行连续动作建模；
- **高精度动作生成**：比离散化/回归方法更具拟合能力；
- **模块解耦结构**：图文由 VLM 处理，动作由专用 expert 处理；
- **统一训练流程**：支持预训练/微调分离，可适应不同任务精度需求；
- **兼容多种机器人配置**：单臂、双臂、移动平台等通过 padding 统一建模。

