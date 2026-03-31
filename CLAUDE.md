# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

多模态股价预测研究项目，核心目标：证明 **FinBERT 情感 + LSTM 融合模型** 优于纯 LSTM 模型。

- **股票标的**：AAPL、TSLA、AMZN
- **数据范围**：2020-01-01 ~ 2025-03-31
- **三组对比实验**：纯 LSTM 基线 / 纯情感 MLP / FinBERT+LSTM 融合模型
- **评估指标**：RMSE、MAE、方向准确率、R²

## 技术栈

- Python 3.10，Conda 虚拟环境 `stock`
- PyTorch（CUDA 12.1）+ Transformers（FinBERT: `ProsusAI/finbert`）
- yfinance（股价数据）、scikit-learn、seaborn/matplotlib
- Jupyter Notebook 为主要开发方式

## 常用命令

```bash
# 激活环境
conda activate stock

# 启动 Jupyter
jupyter notebook

# 安装依赖
pip install -r requirements.txt
```

## 架构设计

项目按 `resources/科研日作战计划_3周版.md` 中的文件结构组织：

- `data/raw/` — 原始股价和新闻数据
- `data/processed/` — 清洗后的特征数据（stock_features、sentiment_scores、merged_features）
- `models/` — 三个模型定义（lstm_baseline.py、sentiment_only.py、fusion_model.py）
- `notebooks/` — 按流程编号的 Jupyter notebooks（01-06）
- `utils/` — 数据加载、预处理、评估指标、可视化工具函数
- `results/` — 图表（300dpi PNG）、结果 CSV、模型权重

## 关键约定

- **数据切分必须按时间顺序**（70/15/15），不能随机打乱
- 滑动窗口：过去 30 天 → 预测第 31 天收盘价
- 归一化使用 MinMaxScaler，需保存 scaler 用于反归一化
- 无新闻日期填充中性情感（pos=0.33, neg=0.33, neu=0.33）
- FinBERT 批量推理使用 GPU 加速，batch_size=64
- 训练使用 MSE loss + Adam + 早停法（patience=10）
