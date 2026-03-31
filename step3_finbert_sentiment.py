"""
Step 3: 用 FinBERT 对所有新闻标题做情感分析
输入：data/raw/news_headlines.csv
输出：data/processed/finbert_sentiment.csv
格式：date | ticker | sentiment_neg | sentiment_neu | sentiment_pos
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
import os

os.makedirs("data/processed", exist_ok=True)

# ==================== 配置 ====================
BATCH_SIZE = 64
MAX_LENGTH = 128
MODEL_NAME = "ProsusAI/finbert"
# ==============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 加载模型
print("加载 FinBERT 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
# FinBERT 标签顺序：positive=0, negative=1, neutral=2
LABEL_ORDER = ["sentiment_pos", "sentiment_neg", "sentiment_neu"]

def batch_predict(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    return probs  # shape: (batch, 3) — [pos, neg, neu]

# 读取新闻
df = pd.read_csv("data/raw/news_headlines.csv")
df = df.dropna(subset=["title"])
df["title"] = df["title"].astype(str).str.strip()
print(f"共 {len(df)} 条新闻待分析")

# 批量推理
all_probs = []
titles = df["title"].tolist()

for i in tqdm(range(0, len(titles), BATCH_SIZE), desc="FinBERT推理"):
    batch = titles[i : i + BATCH_SIZE]
    probs = batch_predict(batch)
    all_probs.extend(probs.tolist())

# 写入结果
df["sentiment_pos"] = [p[0] for p in all_probs]
df["sentiment_neg"] = [p[1] for p in all_probs]
df["sentiment_neu"] = [p[2] for p in all_probs]

# 按 date + ticker 聚合（每天取均值）
daily = (
    df.groupby(["date", "ticker"])[["sentiment_neg", "sentiment_neu", "sentiment_pos"]]
    .mean()
    .reset_index()
)

# 为每只股票补全缺失日期，填充中性值
tickers = daily["ticker"].unique()
date_ranges = {
    "AAPL": pd.date_range("2020-01-01", "2024-12-31", freq="D"),
    "TSLA": pd.date_range("2020-01-01", "2022-12-31", freq="D"),
    "AMZN": pd.date_range("2020-01-01", "2022-12-31", freq="D"),
}

filled_frames = []
for ticker in tickers:
    sub = daily[daily["ticker"] == ticker].set_index("date")
    all_dates = date_ranges[ticker].strftime("%Y-%m-%d")
    sub = sub.reindex(all_dates, fill_value=1/3)
    sub.index.name = "date"
    sub = sub.reset_index()
    sub["ticker"] = ticker
    filled_frames.append(sub)

result = pd.concat(filled_frames, ignore_index=True)
result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

result.to_csv("data/processed/finbert_sentiment.csv", index=False)
print(f"\n保存完成：{len(result)} 条（含补全的无新闻日期）")
print(result.groupby("ticker").size())
