"""
Step 6: 可视化 — 生成所有海报/PPT用图表
输出目录：results/figures/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

os.makedirs("results/figures", exist_ok=True)
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# ==================== 全局字体设置（海报级别，4英尺可读）====================
TITLE_FS   = 18   # 主标题
SUBTITLE_FS = 14  # 子图标题
LABEL_FS   = 13   # 轴标签
TICK_FS    = 11   # 刻度
ANNOT_FS   = 11   # 柱顶数字
LEGEND_FS  = 11   # 图例

# ==================== 颜色方案 ====================
COLORS = {
    "纯LSTM":         "#4C72B0",
    "FinBERT融合":    "#DD8452",
    "预计算情感融合": "#55A868",
    "纯情感MLP":      "#C44E52",
}
LABEL_EN = {
    "纯LSTM":         "Pure LSTM",
    "FinBERT融合":    "FinBERT Fusion",
    "预计算情感融合": "Pre-computed Fusion",
    "纯情感MLP":      "Sentiment-only MLP",
}
TICKER_LABELS = {"AAPL": "Apple (AAPL)", "TSLA": "Tesla (TSLA)", "AMZN": "Amazon (AMZN)"}

results = pd.read_csv("results/tables/all_results.csv")


# ==================== 图1：RMSE 对比柱状图 ====================
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("RMSE Comparison Across Models and Stocks", fontsize=TITLE_FS, fontweight="bold")

for ax, ticker in zip(axes, ["AAPL", "TSLA", "AMZN"]):
    sub = results[results["ticker"] == ticker]
    labels_en = [LABEL_EN.get(m, m) for m in sub["model"]]
    bars = ax.bar(labels_en, sub["RMSE"],
                  color=[COLORS.get(m, "#888") for m in sub["model"]],
                  edgecolor="white", linewidth=1.0)
    ax.set_title(TICKER_LABELS[ticker], fontsize=SUBTITLE_FS, fontweight="bold")
    ax.set_ylabel("RMSE (lower is better)", fontsize=LABEL_FS)
    ax.set_ylim(0, 0.27)
    ax.tick_params(axis="x", rotation=20, labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    for bar, val in zip(bars, sub["RMSE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=ANNOT_FS, fontweight="bold")

plt.tight_layout()
plt.savefig("results/figures/fig1_rmse_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("图1 保存完成")


# ==================== 图2：R² 对比柱状图（排除MLP，避免尺度崩溃）====================
# 纯情感MLP 的 R² 最低达 -4.72，会压缩其他柱子使对比不可见
# 在图下方用文字注明 MLP 结果
LSTM_MODELS = ["纯LSTM", "FinBERT融合", "预计算情感融合"]

fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.suptitle("R² Score Comparison — LSTM-based Models (higher is better)",
             fontsize=TITLE_FS, fontweight="bold")

mlp_notes = []  # 收集MLP注释，统一放到图底部

for ax, ticker in zip(axes, ["AAPL", "TSLA", "AMZN"]):
    sub = results[(results["ticker"] == ticker) & (results["model"].isin(LSTM_MODELS))]
    labels_en = [LABEL_EN.get(m, m) for m in sub["model"]]
    bars = ax.bar(labels_en, sub["R2"],
                  color=[COLORS.get(m, "#888") for m in sub["model"]],
                  edgecolor="white", linewidth=1.0)
    ax.set_title(TICKER_LABELS[ticker], fontsize=SUBTITLE_FS, fontweight="bold")
    ax.set_ylabel("R²", fontsize=LABEL_FS)
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="x", rotation=30, labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_xticklabels(labels_en, rotation=30, ha="right", fontsize=TICK_FS)
    for bar, val in zip(bars, sub["R2"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=ANNOT_FS, fontweight="bold")

    mlp_r2 = results[(results["ticker"] == ticker) & (results["model"] == "纯情感MLP")]["R2"]
    if len(mlp_r2):
        mlp_notes.append(f"{ticker}: R²={mlp_r2.values[0]:.2f}")

# MLP注释统一放在图底部，不与x轴标签重叠
note_text = "† Sentiment-only MLP excluded to preserve scale — " + ",  ".join(mlp_notes)
fig.text(0.5, 0.01, note_text, ha="center", fontsize=9, color="#C44E52", style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("results/figures/fig2_r2_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("图2 保存完成")


# ==================== 图3：预测 vs 实际价格 ====================
fig, axes = plt.subplots(3, 2, figsize=(16, 13))
fig.suptitle("Predicted vs Actual Close Price (Test Set, Normalized)",
             fontsize=TITLE_FS, fontweight="bold")

plot_pairs = [
    ("AAPL", "纯LSTM"),    ("AAPL", "FinBERT融合"),
    ("TSLA", "纯LSTM"),    ("TSLA", "FinBERT融合"),
    ("AMZN", "纯LSTM"),    ("AMZN", "FinBERT融合"),
]

for ax, (ticker, model) in zip(axes.flat, plot_pairs):
    fname = f"results/tables/{ticker}_{model.replace(' ','_')}_preds.csv"
    if not os.path.exists(fname):
        ax.set_visible(False)
        continue
    df = pd.read_csv(fname)
    ax.plot(df["actual"].values, label="Actual", color="#333333", linewidth=2.0)
    ax.plot(df["pred"].values, label="Predicted",
            color=COLORS.get(model, "#888"), linewidth=2.0, alpha=0.85)
    rmse = results[(results["ticker"] == ticker) & (results["model"] == model)]["RMSE"].values
    rmse_str = f"RMSE={rmse[0]:.4f}" if len(rmse) else ""
    ax.set_title(f"{TICKER_LABELS[ticker]} — {LABEL_EN.get(model, model)}  ({rmse_str})",
                 fontsize=SUBTITLE_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS)
    ax.set_xlabel("Test Days", fontsize=LABEL_FS)
    ax.set_ylabel("Normalized Price", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)

plt.tight_layout()
plt.savefig("results/figures/fig3_pred_vs_actual.png", dpi=300, bbox_inches="tight")
plt.close()
print("图3 保存完成")


# ==================== 图4：AAPL 专项 — 三模型对比（左全图 + 右放大后40天）====================
fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]})
fig.suptitle("AAPL: FinBERT vs Pre-computed Sentiment vs Pure LSTM (Test Set)",
             fontsize=TITLE_FS, fontweight="bold")

aapl_models = ["纯LSTM", "FinBERT融合", "预计算情感融合"]
line_styles = ["-", "-", "--"]
line_widths = [2.0, 2.5, 2.0]

data_cache = {}
for model, ls, lw in zip(aapl_models, line_styles, line_widths):
    fname = f"results/tables/AAPL_{model.replace(' ','_')}_preds.csv"
    if not os.path.exists(fname):
        continue
    df = pd.read_csv(fname)
    data_cache[model] = df

    for ax in axes:
        if model == "纯LSTM" and "actual_plotted" not in ax.__dict__:
            ax.plot(df["actual"].values, label="Actual", color="#333333",
                    linewidth=2.5, zorder=5)
            ax.__dict__["actual_plotted"] = True
        ax.plot(df["pred"].values, label=LABEL_EN[model], color=COLORS[model],
                linewidth=lw, linestyle=ls, alpha=0.9)

# 左图：全测试集
axes[0].set_title("Full Test Period", fontsize=SUBTITLE_FS, fontweight="bold")
axes[0].set_xlabel("Test Days", fontsize=LABEL_FS)
axes[0].set_ylabel("Normalized Price", fontsize=LABEL_FS)
axes[0].tick_params(labelsize=TICK_FS)
axes[0].legend(fontsize=LEGEND_FS)

# 右图：放大最后40天，差异最显著区域
zoom_start = -40
axes[1].set_title("Last 40 Days (Zoomed)", fontsize=SUBTITLE_FS, fontweight="bold")
axes[1].set_xlabel("Test Days", fontsize=LABEL_FS)
axes[1].tick_params(labelsize=TICK_FS)
# 重绘右图只取后40天
axes[1].cla()
ref_df = list(data_cache.values())[0]
total = len(ref_df)
x_zoom = np.arange(total + zoom_start, total)
axes[1].plot(ref_df["actual"].values[zoom_start:], label="Actual", color="#333333",
             linewidth=2.5, zorder=5)
for model, ls, lw in zip(aapl_models, line_styles, line_widths):
    if model in data_cache:
        axes[1].plot(data_cache[model]["pred"].values[zoom_start:],
                     label=LABEL_EN[model], color=COLORS[model],
                     linewidth=lw, linestyle=ls, alpha=0.9)
        rmse = results[(results["ticker"] == "AAPL") & (results["model"] == model)]["RMSE"].values
        if len(rmse):
            axes[1].set_title("Last 40 Days (Zoomed)", fontsize=SUBTITLE_FS, fontweight="bold")

# 在右图加RMSE标注
y_top = axes[1].get_ylim()[1]
for i, model in enumerate(aapl_models):
    rmse = results[(results["ticker"] == "AAPL") & (results["model"] == model)]["RMSE"].values
    if len(rmse):
        axes[1].text(0.03, 0.97 - i * 0.1, f"{LABEL_EN[model]}: RMSE={rmse[0]:.4f}",
                     transform=axes[1].transAxes, fontsize=9.5, color=COLORS[model],
                     fontweight="bold", va="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

axes[1].set_xlabel("Test Days (last 40)", fontsize=LABEL_FS)
axes[1].tick_params(labelsize=TICK_FS)

plt.tight_layout()
plt.savefig("results/figures/fig4_aapl_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("图4 保存完成")


# ==================== 图5：方向准确率对比（含显著性说明）====================
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_title("Directional Accuracy by Model and Stock", fontsize=TITLE_FS, fontweight="bold")

tickers = ["AAPL", "TSLA", "AMZN"]
# 测试集天数（15% of 总数据），用于判断差异是否显著
test_n = {"AAPL": 189, "TSLA": 75, "AMZN": 75}

x = np.arange(len(tickers))
models_to_plot = ["纯LSTM", "FinBERT融合"]
width = 0.3

for i, model in enumerate(models_to_plot):
    vals = []
    for t in tickers:
        v = results[(results["ticker"] == t) & (results["model"] == model)]["DirAcc"].values
        vals.append(v[0] if len(v) else 0)
    bars = ax.bar(x + i * width, vals, width, label=LABEL_EN[model],
                  color=COLORS[model], edgecolor="white", linewidth=1.0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", fontsize=ANNOT_FS, fontweight="bold")

# 标注统计不显著的对比（z < 1.96）
for j, ticker in enumerate(tickers):
    v_lstm = results[(results["ticker"] == ticker) & (results["model"] == "纯LSTM")]["DirAcc"].values
    v_finb = results[(results["ticker"] == ticker) & (results["model"] == "FinBERT融合")]["DirAcc"].values
    if len(v_lstm) and len(v_finb):
        diff = abs(v_finb[0] - v_lstm[0])
        n = test_n[ticker]
        p_pooled = (v_lstm[0] + v_finb[0]) / 2
        se = np.sqrt(p_pooled * (1 - p_pooled) / n * 2)
        z = diff / se if se > 0 else 0
        if z < 1.96:   # 差异不显著
            mid_x = x[j] + width / 2
            y_top = max(v_lstm[0], v_finb[0]) + 0.018
            ax.text(mid_x, y_top, "n.s.", ha="center", fontsize=10,
                    color="#666666", style="italic")

ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Random baseline (0.5)")
ax.set_xticks(x + width / 2)
ax.set_xticklabels([TICKER_LABELS[t] for t in tickers], fontsize=TICK_FS + 1)
ax.set_ylabel("Directional Accuracy", fontsize=LABEL_FS)
ax.set_ylim(0.45, 0.63)
ax.tick_params(axis="y", labelsize=TICK_FS)
ax.legend(fontsize=LEGEND_FS)

# 说明注释
ax.text(0.99, 0.04,
        "n.s. = not statistically significant (z < 1.96)\n"
        "Test set: AAPL n≈189 days, TSLA/AMZN n≈75 days",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#666666", style="italic")

plt.tight_layout()
plt.savefig("results/figures/fig5_directional_accuracy.png", dpi=300, bbox_inches="tight")
plt.close()
print("图5 保存完成")


print("\n所有图表已保存至 results/figures/")
