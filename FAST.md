

## FAST 的 Action Head 设计详解

**FAST（Frequency-space Action Sequence Tokenization）** 提出了一种**高效动作 token 化机制**，用于支持高频率、精细操作任务下的 Vision-Language-Action (VLA) 模型训练。FAST 的核心是将机器人动作轨迹进行频域压缩（使用 **离散余弦变换 DCT**），再使用 **字节对编码（BPE）** 将压缩结果离散为高效 token，从而显著提高自回归 Transformer 对连续动作序列的建模能力。

---

### 一、核心思想

传统 VLA（如 RT-2、OpenVLA）采用逐维 binning 离散化，每秒控制频率为 50Hz 时将产生数百 token，导致训练困难、收敛缓慢。

FAST 则采用如下压缩流程作为其 Action Head 的设计方式：

| 步骤            | 操作                                                   |
|------------------|--------------------------------------------------------|
| 1. 归一化         | 将每个维度动作值归一化到 [?1, 1] 区间（基于1%、99%分位） |
| 2. DCT转换        | 对每个维度执行离散余弦变换，转为频域系数               |
| 3. 量化压缩       | 保留显著系数，舍弃高频噪声项，压缩冗余                 |
| 4. Flatten         | 将所有维度系数展开成一维序列，低频项优先               |
| 5. BPE压缩        | 使用 Byte Pair Encoding 压缩为 token 序列               |
| 6. 解码           | 所有过程可逆，推理阶段可恢复出连续动作序列               |

最终，每个 **1秒动作 chunk** 压缩为约 **30~60 个离散 token**，可输入自回归语言模型进行动作预测。

---

### 二、训练与推理流程

#### 训练阶段：

- 任务形式为标准 **next-token prediction**；
- 将 token 序列与语言 token 拼接输入 VLA 模型；
- 目标是预测下一个 token，优化交叉熵损失。

#### 推理阶段：

- 语言模型输出离散 token；
- 通过 BPE 解码 → 频域恢复 → 反 DCT → 连续动作 chunk；
- 最终动作控制频率达 20Hz~50Hz（或更高）。

---

### 三、与其他模型对比（Action Head）

| 模型         | 动作头类型                     | 是否自回归 | 控制频率 | 是否压缩 | 解码效率 | 可扩展性 |
|--------------|--------------------------------|--------------|------------|------------|-------------|-------------|
| RT-2         | 离散 binning token              | 是           | 中低       | 否         | 高           | 是（共享token） |
| OpenVLA      | token → de-tokenizer            | 是           | 中         | 否         | 中           | 是             |
| π0           | Flow Matching (Diffusion)       | 否           | 是（50Hz）| 否         | 是         | 是（expert独立）|
| CogACT       | Reasoning + Latent Diffusion    | 否           | 中高       | 否         | 是           | 是（推理注入）  |
| **FAST**     | **DCT + BPE压缩 + token解码**   | 是           | 是（50Hz+）| 是     | 是           | 是（通用tokenizer）|

---

### 四、设计优势总结

- **超强压缩性能**：相较 RT-2/OpenVLA，平均压缩比高达 5~13x；
- **高频控制支持**：首次可用于训练 autoregressive VLA 于 50Hz+ 精细操作任务；
- **无需结构改动**：可直接替换 token 化组件，适配任意 Transformer backbone；
- **训练速度提升**：训练收敛速度比 π0 diffusion 提升约 5 倍；
- **通用 tokenizer**：训练了 FAST+ 通用动作 tokenizer，适用于任意机器人动作流；
- **解码效率高**：与向量量化（VQ）相比，DCT+BPE 无需复杂神经网络模块，推理高效可逆；
- **已验证多种环境泛化能力**：如 DROID 数据集、T-shirt folding 等真实任务。
