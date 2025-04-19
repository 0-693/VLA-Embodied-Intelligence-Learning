# VLA Embodied Intelligence Learning Hub

本仓库是我个人学习入门 Vision-Language-Action (VLA) 具身智能领域的个人笔记，仅供参考。通过阅读主流论文、拆解模型架构并深入调研各类 Action Head 的实现方式，快速理解和掌握该领域的核心技术。

---

## ? 项目内容概览

本仓库围绕具身智能中的 VLA 模型，从以下两个方面进行系统整理：

### 1. **模型调研与分类**

- 共调研了当前具代表性的 13 个 VLA 模型，包括：
  - RT-1 / RT-2 / RT-Trajectory / OpenVLA / TinyVLA / TraceVLA
  - Octo / π0 / CogACT / Diffusion-VLA
  - SpatialVLA / FAST / RDT-1B
- 每个模型都根据其架构特点归类为：
  - 多分类器模型
  - 自回归 token 模型
  - 扩散式控制模型

### 2. **Action Head 设计深度剖析**

详见 [`report.md`](./report.md)，主要内容包括：

- Action Head 的定义与分类
- 主流设计方式（多分类、自回归、扩散、token压缩等）对比分析
- 每种设计的核心机制、代表模型、优缺点总结
- 表格式汇总与未来趋势展望
- 对 SpatialVLA 和 FAST 等代表性结构的逐模块解构说明

---

## ? 主要文件说明

| 文件/目录        | 内容描述                                               |
|------------------|--------------------------------------------------------|
| `report.md`      | Action Head 对比调研总文档，包含 13 个模型总结与分类 |
| `README.md`      | 当前项目简介                                            |


---

## ? 参考资料

感谢大佬的[仓库](https://github.com/TianxingChen/Embodied-AI-Guide)

