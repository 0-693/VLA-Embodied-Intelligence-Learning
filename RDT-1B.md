## RDT-1B 的 Action Head 设计详解

**RDT-1B**（论文正式标题为 *Robotics Diffusion Transformer*）是专为双臂精细操作设计的扩散式机器人基础模型。其 Action Head 采用 **扩散建模（Diffusion Modeling）+ MLP解码器** 的方式，在一个统一的物理可解释动作空间（Unified Action Space）中生成连续、高精度、可泛化的控制信号，特别适用于双臂任务的多模态控制建模。

---

### 一、核心思想

| 输入类型              | 说明                                                         |
|-----------------------|--------------------------------------------------------------|
| 图像输入              | 三路图像：正面、右腕、左腕                                   |
| 文本输入              | 自然语言指令，使用 T5-XXL 编码                                 |
| 低维输入              | proprioception（身体状态）+ 控制频率 + 扩散时间步              |
| 动作输出（Action Head） | 使用 Diffusion Transformer + MLP 生成动作 chunk（序列）         |

动作生成目标是建模 \( p(a_{t:t+T_a} | \ell, o_t) \)，即在当前观测和指令下输出未来多个时刻的连续控制序列。

---

### 二、Action Head 架构与机制

#### 动作形式（连续控制）：

- 目标输出为 **动作块（action chunk）**，默认长度 \( T_a = 64 \)；
- 每帧动作包含左右手 joint positions、EEF pose、gripper 状态、base control 等；
- 所有动作都嵌入到一个 **统一的 128 维动作空间** 中（详见论文附录C）。

#### 扩散建模流程：

1. **前向过程**：添加高斯噪声：
   \[
   \tilde{a}_t = \sqrt{\bar{\alpha}_k} a_t + \sqrt{1 - \bar{\alpha}_k} \epsilon
   \]
2. **反向去噪（训练目标）**：
   \[
   \mathcal{L}(\theta) = \mathbb{E}\left[ \left\| a_t - f_\theta(\ell, o_t, \tilde{a}_t, k) \right\|^2 \right]
   \]

3. **推理阶段**：从纯噪声开始，使用 denoising steps (如 5 步) 恢复动作 chunk。

#### 模型结构（fθ）：

- 主干：28 层 Transformer（DiT架构），支持多模态 cross-attention；
- 三大关键改进：
  - **MLP Decoder**：非线性映射，提高对复杂物理控制的拟合能力；
  - **QKNorm + RMSNorm**：解决高频物理量的数值稳定性问题；
  - **Alternating Condition Injection (ACI)**：交替注入图像与语言 tokens，防止注意力偏置。

---

### 三、训练与推理流程

#### 训练：

- 使用 DDPM 训练动作生成模块；
- 使用 MSE loss 监督 denoising network；
- 所有低维量使用带 Fourier 特征的 MLP 编码；
- 文本由 T5-XXL 编码，图像由 SigLIP 编码（均为冻结）。

#### 推理：

- 输入图像 + 指令 → 编码后条件；
- 从高斯噪声采样 latent 动作 → 5 步反向扩散生成完整动作 chunk；
- 控制频率最高可达 **381Hz**（chunk 频率为 6Hz）。

---

### 四、与其他模型对比（Action Head）

| 模型         | Action Head 类型                        | 是否自回归 | 控制频率 | 是否 chunk 输出 | 多模态注入机制 |
|--------------|------------------------------------------|--------------|------------|------------------|----------------|
| RT-2         | Token autoregression + de-tokenizer      | 是           | 中低       | 否               | Prompt文本      |
| π0           | Flow Matching Expert                     | 否           | 是（50Hz）| 是              | FiLM 注入       |
| CogACT       | Reasoning + Latent Diffusion Head        | 否           | 中高       |是               | FiLM + Alternating |
| FAST         | DCT + BPE + Token 生成                   | 是           | 是       | 是              | Token 序列      |
| **TRACEVLA** | **DiT + Alternating Condition + MLP Decoder** | 否    | 是（381Hz） | 是               | **Cross-Attn + ACI** |

---

### 五、设计优势总结

- **精准建模双臂动作多模态分布**，适应性强于离散方法；
- **超高推理效率**，仅需 5 步扩散可达 381Hz 控制频率；
- **统一动作空间设计**，可兼容单臂、双臂、带底盘机器人；
- **模块解耦训练策略**，视觉/语言冻结，动作头可独立优化；
- **增强时序一致性**，chunk 预测减小累积误差；
- **大模型扩展性**，当前为最大扩散式双臂模型（1.2B参数）；
- **跨机器人泛化**：预训练覆盖 46 套系统、1M+ 轨迹。

