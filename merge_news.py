import pandas as pd

# --- AAPL：2020-2025 全年 ---
aapl = pd.read_csv("data/raw/apple_news_data.csv", usecols=["date", "title"])
aapl["ticker"] = "AAPL"
aapl["date"] = pd.to_datetime(aapl["date"], utc=True).dt.strftime("%Y-%m-%d")
aapl = aapl[(aapl["date"] >= "2020-01-01") & (aapl["date"] <= "2024-12-31")]

# --- TSLA：2020-2022 ---
tsla = pd.read_csv("data/raw/tesla_news_2020_2022.csv")
tsla = tsla.rename(columns={"company": "ticker"})[["date", "title", "ticker"]]
tsla["date"] = pd.to_datetime(tsla["date"], errors="coerce").dt.strftime("%Y-%m-%d")
tsla = tsla[(tsla["date"] >= "2020-01-01") & (tsla["date"] <= "2022-12-31")]

# --- AMZN：2020-2022 ---
amzn = pd.read_csv("data/raw/amazon_news_full_2019_2022.csv")
amzn["ticker"] = "AMZN"
amzn["date"] = pd.to_datetime(amzn["date"], errors="coerce").dt.strftime("%Y-%m-%d")
amzn = amzn[(amzn["date"] >= "2020-01-01") & (amzn["date"] <= "2022-12-31")]

# --- 合并 ---
all_news = pd.concat([aapl, tsla, amzn], ignore_index=True)
all_news = all_news.dropna(subset=["date", "title"])
all_news = all_news.drop_duplicates(subset=["date", "ticker", "title"])
all_news = all_news.sort_values(["ticker", "date"]).reset_index(drop=True)

all_news.to_csv("data/raw/news_headlines.csv", index=False)

# 验证
print(all_news.groupby("ticker").size())
print(f"\n总计：{len(all_news)} 条")
print("\n每只股票每年新闻数量：")
all_news["year"] = all_news["date"].str[:4]
print(all_news.groupby(["ticker", "year"]).size().unstack(fill_value=0))
