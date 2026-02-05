# wind-power-forecast
基于门控双专家与反转融合的超短期风电功率预测
# 超短期风电功率预测（门控双专家 + 反转融合）

本项目实现一种基于**状态判别 + 门控双专家 + 反转融合**的超短期风电功率预测方法。  
通过多尺度分解获取 raw/low/high 三通道特征，采用共享 LSTM 编码与门控机制动态融合稳定/波动专家预测，并引入动态伪标签和门控一致性约束提升训练稳定性与波动段表现。

---

## ✅ 目录结构
```
wind-power-forecast/
├── README.md
├── requirements.txt
├── main.py                 # 主模型训练脚本
├── c4_compare.py           # 对比实验脚本
└── c3_optuna_tuning.py     # Optuna 调参脚本
```

---

## ✅ 环境依赖
- Python 3.8+
- PyTorch 1.10+（建议 2.x）
- numpy / pandas / matplotlib / pywavelets / optuna

---

## ✅ 安装依赖
```bash
pip install -r requirements.txt
```

---

## ✅ 运行步骤
1. 修改脚本中的 `DATA_PATH` 为你的本地 CSV 路径  
2. 运行主训练脚本：

```bash
python main.py
```

---

## ✅ 数据格式
CSV 文件至少包含一列：
```
power
```

---

## ✅ 输出结果
训练完成后会输出：
- 预测指标（MAE / RMSE / MAPE / R²）
- 残差图、散点图、峰值对齐图（保存为 png）

---

## ✅ 说明
若缺少 `pywt`，多尺度分解会自动退化为滑动平均分解。  
可通过 `c4_compare.py` 进行消融对比，  
通过 `c3_optuna_tuning.py` 进行超参搜索。
