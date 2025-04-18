

## Octo 的 Action Head 设计详解

Octo 是一个面向多机器人、多任务的通用策略模型。其 **Action Head** 采用 **条件扩散策略解码器（conditional diffusion decoding head）** 来直接生成连续动作，支持生成高质量、多模态、精细控制动作序列。这种设计摆脱了传统 MSE 回归或离散 token 分类的限制，在性能和可表达性上均取得显著提升。

---

### 一、核心思想

Octo 的 Action Head 不再将动作离散为 token 或直接用 MSE 回归单步动作，而是使用**扩散模型（Diffusion Model）**在连续动作空间中进行建模，输出完整的连续动作 chunk。

| 控制维度       | 表达内容                                    |
|----------------|---------------------------------------------|
| ΔPosition (3D) | 末端执行器位置变化（x, y, z）                |
| ΔRotation (3D) | 姿态变化（roll, pitch, yaw）                 |
| Gripper (1D)   | 抓取器控制（开/合）                          |
| **总计**        | **7维连续控制动作，或更多（如双臂14维）**     |

此外，在多任务场景下，Octo 支持动作 chunking（一次预测多步），适用于长时间、稳定控制。

---

### 二、Action Head 结构与机制

#### 动作头结构：

- Action Head 是一个**3层 MLP**（隐藏维度256，带 LayerNorm 和残差）；
- 接收来自 transformer 的 **Readout Token** embedding；
- 输出一个长度为 L（如 L=4）的动作 chunk，每个 chunk 是一组连续控制值。

#### 扩散过程：

- 使用标准 DDPM 机制进行训练和生成，包含 20 步反向去噪：
  \[
  x_{k-1} = \alpha(x_k - \gamma \epsilon_\theta(x_k, e, k)) + \mathcal{N}(0, \sigma^2 I)
  \]
  其中 \(x_k\) 是当前噪声状态，\(e\) 为 transformer 提供的上下文表示，\(\epsilon_\theta\) 是预测网络。

- 噪声调度采用 cosine schedule。

#### 动作 chunk 表达：

- 输出为一个或多个连续动作（如：64帧，分批执行）；
- 每个 chunk 可用于 receding horizon control（RHC）或直接执行。

---

### 三、训练与推理流程

#### 训练阶段：
- 将 ground-truth 动作加入高斯噪声 → 得到 \(x_k\)；
- 模型学习从 \(x_k\) 预测去噪向量 \(\epsilon\)，还原原始动作；
- 损失函数为 DDPM 损失：
  \[
  \mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_k, e, k)\|^2]
  \]

#### 推理阶段：
- 采样初始高斯噪声 \(x_K\)；
- 使用扩散模型迭代生成动作序列；
- 最终输出 L 步连续动作，用于控制器执行。

---

### 四、与其他模型对比（动作头）

| 模型         | Action Head类型         | Token/步 | 是否自回归 | 是否支持 chunk | 表达能力 |
|--------------|--------------------------|----------|--------------|------------------|-----------|
| RT-1         | 分类器（离散 token）      | 11       | 否           | 否               | 中         |
| RT-2         | 自回归 token + de-token  | 7–8      | 是           | 否               | 中         |
| OpenVLA      | Token + De-tokenizer     | 7        | 是           | 否               | 中         |
| TinyVLA      | Diffusion（单步）        | 1次生成  | 否           | 否               | 高         |
| **Octo**     | **Diffusion Chunk Decoder** | **1次生成多个** | 否        | 是（支持64步）     | 是（极高） |

---

### 五、设计优势总结

- **多步预测动作（chunking）**：支持长时间稳定控制，减少误差积累；
- **精准建模多模态分布**：相比 MSE 能建模动作的多样性，提升控制表现；
- **非自回归生成，推理高效**：不依赖 token-by-token 输出，一次生成整个动作块；
- **与 transformer 解耦可扩展结构**：只需对 readout token 添加扩散头；
- **适配多动作空间**：单臂（7D）、双臂（14D）、关节控制均可处理；
- **与语言/图像输入自然结合**：可接受 goal image 或语言指令条件生成。

