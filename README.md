# 🌬️ GDE-IF: 基于门控双专家与反转融合的超短期风电功率预测

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目提出了一种**基于门控双专家与反转融合 (Gated Dual-Expert with Inversion Fusion, GDE-IF)** 的超短期风电功率预测框架。该模型专门针对复杂天气工况（如快速爬坡、阵风扰动）下的风电序列非平稳特性设计，有效解决了传统单一深度学习模型在“平稳”与“波动”状态下顾此失彼的问题。

## 💡 核心创新点 (Key Innovations)

1. **物理规则驱动的动态状态感知**：利用多尺度分解（Wavelet）结合风电物理指标（爬坡幅值与波动能量），为门控网络提供动态伪标签监督，解决冷启动与门控塌陷问题。
2. **混合专家架构 (MoE)**：自适应调度“稳定专家”与“波动专家”，实现分状态精细化建模。
3. **双向视角反转融合机制**：
   - ⏱️ **时间反转 (Time Inversion)**：针对平稳状态，利用时间对称性消除随机测量噪声。
   - 🔀 **特征反转 (Feature Inversion)**：针对波动状态，强迫模型学习多尺度特征间的非线性耦合，增强突变响应能力。
4. **复合损失函数优化**：结合波动段峰值加权损失、动态退火门控损失以及专家一致性约束（KL散度），全面提升模型鲁棒性。

## 🏗️ 模型架构 (Architecture)

<img width="752" height="722" alt="初稿 drawio (1)" src="https://github.com/user-attachments/assets/4d090abd-fd90-48ed-90e2-1880f26c439d" />

![GDE-IF Architecture](images/architecture.png)

## 📊 实验结果 (Results)

基于某内陆风电场 2019 年实际运行数据（采样率15min，预测未来1.5小时即6个步长），GDE-IF 模型在各项误差指标上均达到最优，特别是在波动工况下（Fluct_MAE）表现出极强的适应性：

| 模型 (Model) | MAE (kW) | RMSE (kW) | Fluct_MAE (kW) |
| :--- | :--- | :--- | :--- |
| Baseline-LSTM | 239.07 | 385.54 | 294.20 |
| Baseline-GRU | 244.17 | 389.84 | 300.67 |
| GDE (No Inv) | 245.10 | 390.23 | 301.02 |
| **GDE-IF (Ours)** | **235.08** | **371.05** | **287.53** |

## 🚀 快速开始 (Quick Start)

### 环境依赖
```bash
pip install -r requirements.txt
```

### 数据预处理与训练
*(根据你的实际代码文件名称修改)*
```bash
# 1. 运行多尺度分解与数据标准化
python preprocess.py

# 2. 启动超参数寻优与模型训练
python train_optuna.py

# 3. 生成预测结果与高级对比图表
python c9_plot_advanced_fix.py
```

## 📝 引用 (Citation)
如果您在研究中使用了本项目的代码或思路，请考虑引用我们的工作：
```bibtex
@article{YourName2024GDEIF,
  title={基于门控双专家与反转融合的超短期风电功率预测},
  author={zjxxx},
  year={2026}
}
```
