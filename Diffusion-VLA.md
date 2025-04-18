

## Diffusion-VLA 的 Action Head 设计详解

**Diffusion-VLA (DiVLA)** 提出了一种统一的推理与动作生成架构，将 **自回归语言模型（用于推理）** 与 **扩散模型（用于动作生成）** 相结合。其 **Action Head** 采用 **Latent Diffusion Policy Head**，基于高维视觉-语言-语义上下文，在连续动作空间中生成高质量的动作向量序列，支持高频率机器人控制和多任务泛化。

---

### 一、核心思想

| 组件               | 说明                                                           |
|--------------------|----------------------------------------------------------------|
| 推理模块            | 使用 LLM（如 Qwen2-VL）生成文本推理信息                     |
| 动作解码器（Action Head） | 基于 **Latent Diffusion 模型**，输出机器人动作（joint space） |

动作不再离散为 token，也不使用自回归结构，而是由一个条件扩散模型直接从高斯噪声中采样，逐步去噪获得完整动作向量。

---

### 二、Action Head 的结构与机制

#### 输入条件：
- 多模态编码（图像、文本）来自 VLM backbone；
- 推理结果（Reasoning Token）通过 **FiLM（Feature-wise Linear Modulation）** 注入扩散模型，作为条件向量；
- 控制目标为机器人的 joint-space 动作（如 7DOF 末端 + gripper）。

#### 扩散过程：
- 使用标准 DDPM 扩散过程，在 latent 空间进行动作建模；
- 每步动作从高斯噪声逐步去噪生成；
- 控制频率高达 82Hz（DiVLA-2B），可部署于真实机器人。

#### 输出层：
- 扩散过程最后连接一个 MLP 解码器，用于将去噪 latent 映射为具体动作；
- 若换 robot embodiment，仅需替换 MLP 输出层（无需重训模型）。

---

### 三、训练与推理流程

#### 训练阶段：
- 损失函数组合：
  \[
  \mathcal{L} = \mathcal{L}_{\text{diffusion}} + \alpha \cdot \mathcal{L}_{\text{NTP}}
  \]
  其中：
  - \( \mathcal{L}_{\text{diffusion}} \)：用于训练动作解码器；
  - \( \mathcal{L}_{\text{NTP}} \)：用于训练推理模块的 next-token prediction；
  - \( \alpha = 1 \)，保证推理和控制学习平衡。

#### 推理阶段：
- 语言输入生成 reasoning phrase；
- 将推理 embedding 注入扩散模型（FiLM 注入）；
- 从高斯噪声采样动作 latent，10 步反向生成控制动作。

---

### 四、与其他模型对比（Action Head）

| 模型         | Action Head 类型                      | 是否自回归 | 是否语言联动 | 控制频率 | 特点                     |
|--------------|----------------------------------------|--------------|----------------|------------|--------------------------|
| RT-2         | Token autoregression + de-tokenizer    | 是           | 是             | 慢         | 统一 token 空间          |
| OpenVLA      | token → de-tokenizer                   | 是           | 是             | 慢         | 分离 action tokenizer     |
| π0           | Flow Matching Expert                   | 否         | 是             | 高（50Hz） | 精确建模动作流           |
| CogACT       | Reasoning + Latent Diffusion Head      | 否         | 是（LLM推理）   | 中高        | 结构解耦，可解释推理注入 |
| **Diffusion-VLA** | **LLM推理 + FiLM注入 + Latent Diffusion** | 否           | 是             | 是（82Hz）| **推理融合 + 高效控制**     |

---

### 五、设计优势总结

- **双模融合（语言 + 控制）**：将 LLM 推理与扩散动作生成完美结合；
- **结构解耦，高效适配**：VLM 与 Diffusion Head 分离，可快速适配新机器人；
- **快速推理部署**：DiVLA-2B 可达 82Hz 控制速率，远高于 OpenVLA（5Hz）；
- **可扩展性强**：模型可从 2B → 7B → 72B，无需改动结构；
- **无需动作 chunking**：完整生成高质量动作序列，泛化能力优于 π0 类结构；
- **推理可注入可诊断**：Reasoning phrase 可查看，有助于策略可解释性与调试。

