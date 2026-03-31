"""
Step 2: 提取 AAPL 预计算情感分数（apple_news_data.csv 自带）
输出：data/processed/AAPL_precomputed_sentiment.csv
格式：date | sentiment_neg | sentiment_neu | sentiment_pos
"""
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# 读取原始数据，只取需要的列
df = pd.read_csv(
    "data/raw/apple_news_data.csv",
    usecols=["date", "sentiment_neg", "sentiment_neu", "sentiment_pos"]
)

# 处理日期（带时区）
df["date"] = pd.to_datetime(df["date"], utc=True).dt.strftime("%Y-%m-%d")

# 过滤到 2020-2024
df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2024-12-31")]

# 按日期聚合（每天取均值）
daily = df.groupby("date")[["sentiment_neg", "sentiment_neu", "sentiment_pos"]].mean()

# 补全所有交易日（无新闻日填充中性值）
all_dates = pd.date_range("2020-01-01", "2024-12-31", freq="D").strftime("%Y-%m-%d")
daily = daily.reindex(all_dates, fill_value=1/3)

daily.index.name = "date"
daily = daily.reset_index()

daily.to_csv("data/processed/AAPL_precomputed_sentiment.csv", index=False)
print(f"保存完成：{len(daily)} 天")
print(daily.describe().round(3))
