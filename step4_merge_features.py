"""
Step 4: 合并股价特征 + FinBERT情感 → 最终训练数据
输出：
  data/processed/merged_{TICKER}_finbert.csv   — 三只股票通用
  data/processed/merged_AAPL_precomputed.csv   — AAPL专项对比用
"""
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

DATE_RANGES = {
    "AAPL": ("2020-01-01", "2024-12-31"),
    "TSLA": ("2020-01-01", "2022-12-31"),
    "AMZN": ("2020-01-01", "2022-12-31"),
}

# 读取 FinBERT 情感
sentiment = pd.read_csv("data/processed/finbert_sentiment.csv")

for ticker, (start, end) in DATE_RANGES.items():
    # 读取股价特征（已含技术指标，未归一化版本）
    price = pd.read_csv(f"data/processed/{ticker}_features_scaled.csv")
    price = price.rename(columns={"Date": "date", "index": "date"})

    # 统一 date 列名（yfinance 输出的列名是索引）
    if "Date" in price.columns:
        price = price.rename(columns={"Date": "date"})
    else:
        price = price.rename(columns={price.columns[0]: "date"})

    price["date"] = pd.to_datetime(price["date"]).dt.strftime("%Y-%m-%d")
    price = price[(price["date"] >= start) & (price["date"] <= end)]

    # 取该股票的 FinBERT 情感
    sent = sentiment[sentiment["ticker"] == ticker][
        ["date", "sentiment_neg", "sentiment_neu", "sentiment_pos"]
    ]

    # 合并（左连接，以股价日期为准）
    merged = price.merge(sent, on="date", how="left")

    # 无情感数据的日期填充中性值
    for col in ["sentiment_neg", "sentiment_neu", "sentiment_pos"]:
        merged[col] = merged[col].fillna(1/3)

    out_path = f"data/processed/merged_{ticker}_finbert.csv"
    merged.to_csv(out_path, index=False)
    print(f"{ticker}: {len(merged)} 条 → {out_path}")
    print(f"  列：{merged.columns.tolist()}\n")

# AAPL 专项：用预计算情感替换 FinBERT 情感
print("生成 AAPL 预计算情感版本...")
price_aapl = pd.read_csv("data/processed/AAPL_features_scaled.csv")
if "Date" in price_aapl.columns:
    price_aapl = price_aapl.rename(columns={"Date": "date"})
else:
    price_aapl = price_aapl.rename(columns={price_aapl.columns[0]: "date"})
price_aapl["date"] = pd.to_datetime(price_aapl["date"]).dt.strftime("%Y-%m-%d")
price_aapl = price_aapl[
    (price_aapl["date"] >= "2020-01-01") & (price_aapl["date"] <= "2024-12-31")
]

precomp = pd.read_csv("data/processed/AAPL_precomputed_sentiment.csv")
merged_pre = price_aapl.merge(precomp, on="date", how="left")
for col in ["sentiment_neg", "sentiment_neu", "sentiment_pos"]:
    merged_pre[col] = merged_pre[col].fillna(1/3)

merged_pre.to_csv("data/processed/merged_AAPL_precomputed.csv", index=False)
print(f"AAPL预计算版: {len(merged_pre)} 条 → data/processed/merged_AAPL_precomputed.csv")

print("\n全部完成！")
