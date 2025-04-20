

## CogACT 的 Action Head 设计详解

CogACT 是一个融合推理模块的视觉-语言-动作（VLA）模型，其 **Action Head** 基于**条件扩散模型（Conditional Diffusion Model）** 实现，结合语言模型生成的高层次语义 token 和多模态观察，生成高精度的连续动作向量，支持复杂操作任务和高时序控制需求。

---

### 一、核心思想

CogACT 将动作建模过程分为两阶段：

| 阶段             | 说明                                                           |
|------------------|----------------------------------------------------------------|
| 自回归阶段        | 使用 LLM 推理模块生成语言 token 作为上下文条件                     |
| 动作生成阶段（Action Head） | 使用**Latent Diffusion Policy Head**，将上下文编码解码为连续动作向量 |

该动作头采用标准的 **DDPM（Denoising Diffusion Probabilistic Model）** 框架，并通过推理模块的中间表示进行条件控制，支持多种机器人控制类型。

---

### 二、动作头结构与机制

#### 动作建模方式

- 使用一个**Latent Diffusion Decoder**，将 Transformer 编码后的语义表示解码为动作；
- 动作维度包括机器人末端执行器的位置/姿态/夹爪状态等；
- 每次生成一个完整的 joint-space 控制向量（适用于多个 robot embodiment）；

#### 网络结构：

- **Transformer Encoder**（处理视觉、语言、多模态输入）；
- **Diffusion Policy Decoder**（动作生成部分）：
  - 标准 DDPM 结构；
  - 条件输入为 LLM reasoning tokens；
  - 动作解码输出维度可变（根据机器手臂自由度）；
  - 最后一层为 **MLP → Joint Space 动作向量**。

#### 推理模块注入：

- 将 reasoning 模块输出的语义 token 注入到 Transformer 编码器；
- 使用 FiLM（Feature-wise Linear Modulation）方式实现模块调控；
- 避免将 reasoning 输出作为直接 token，提升推理效率与稳定性。

---

### 三、训练与推理流程

#### 训练阶段：

- 动作向量加噪形成 latent；
- 使用 diffusion decoder 预测噪声或 denoising vector；
- 优化 DDPM 损失：
  $$
  \begin{align*}
  \mathcal{L}_{\text{diff}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, c)\|^2]
  \end{align*}
  $$

#### 推理阶段：

- 给定当前视觉观测 + 任务文本；
- LLM reasoning → Transformer token embedding；
- DDPM 多步采样动作 latent → 预测 robot joint 控制动作。

---

### 四、与其他模型对比（Action Head）


| 模型         | Action Head 类型                     | 是否自回归 | 是否语言联合 | 控制粒度 | 模块结构                     |
|--------------|----------------------------------------|--------------|----------------|------------|------------------------------|
| RT-2         | token-based autoregressive             | 是           | 是             | 中         | 统一结构                     |
| OpenVLA      | token → De-Tokenizer                   | 是           | 是            | 中         | 共享输出空间                 |
| Octo         | MLP + Diffusion Chunk Decoder          | 否          | 部分          | 高         | 解耦结构（动作 readout）     |
| π0           | Flow Matching + Action Expert          | 否          | 是             | 高         | **VLM + 专用动作模块解耦**    |
| **CogACT**   | **LLM reasoning + Diffusion Head**     | 否          | 是（推理增强） | **高**     | **LLM + Diffusion 解耦结构** |


---


### 五、设计优势总结

- **推理-动作解耦架构**：语言 token 不再参与动作输出，提升鲁棒性与效率；
- **高精度扩散控制**：相比回归/token方法更平滑、更能建模动作多样性；
- **可解释 reasoning 机制**：LLM reasoning 结果可视化，支持错误诊断与调试；
- **多 embodiment 适配**：通过切换 MLP 输出层适配不同机器人结构；
- **灵活微调策略**：可单独微调 reasoning 模块或动作生成模块。

