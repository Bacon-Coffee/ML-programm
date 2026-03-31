import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# ==================== 配置 ====================
TICKERS = ['AAPL', 'TSLA', 'AMZN']
START_DATE = '2020-01-01'
END_DATE = '2026-03-31'
OUTPUT_DIR = 'data/processed'
RAW_DIR = 'data/raw'
# ==============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd


def process_ticker(ticker):
    print(f"\n正在处理 {ticker}...")

    # 1. 下载数据
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    if df.empty:
        print(f"  警告: {ticker} 数据为空，跳过")
        return

    # 展平多级列名（yfinance 新版本会产生多级列）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"  下载完成: {len(df)} 条记录 ({df.index[0].date()} ~ {df.index[-1].date()})")

    # 2. 检查缺失值
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  发现 {missing} 个缺失值，进行前向填充")
        df.ffill(inplace=True)

    # 3. 保存原始数据
    raw_path = os.path.join(RAW_DIR, f'{ticker}_raw.csv')
    df.to_csv(raw_path)
    print(f"  原始数据已保存: {raw_path}")

    # 4. 计算技术指标
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])

    # 删除因滑动窗口产生的 NaN 行（前26行）
    df.dropna(inplace=True)
    print(f"  计算技术指标后剩余: {len(df)} 条记录")

    # 5. 归一化
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD']
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 保存 scaler（反归一化时需要）
    scaler_path = os.path.join(OUTPUT_DIR, f'{ticker}_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler 已保存: {scaler_path}")

    # 6. 保存处理后的数据（同时保存原始值版本和归一化版本）
    processed_path = os.path.join(OUTPUT_DIR, f'{ticker}_features.csv')
    df.to_csv(processed_path)

    scaled_path = os.path.join(OUTPUT_DIR, f'{ticker}_features_scaled.csv')
    df_scaled.to_csv(scaled_path)

    print(f"  特征数据已保存: {processed_path}")
    print(f"  归一化数据已保存: {scaled_path}")

    # 7. 打印数据概览
    print(f"\n  --- {ticker} 数据概览 ---")
    print(f"  特征列: {feature_cols}")
    print(f"  收盘价范围: ${df['Close'].min():.2f} ~ ${df['Close'].max():.2f}")
    print(f"  RSI 范围: {df['RSI'].min():.1f} ~ {df['RSI'].max():.1f}")


# ==================== 主程序 ====================
if __name__ == '__main__':
    print("=" * 50)
    print("股价数据下载 & 特征工程")
    print(f"股票: {TICKERS}")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print("=" * 50)

    for ticker in TICKERS:
        process_ticker(ticker)

    print("\n全部完成！")
    print(f"原始数据目录: {RAW_DIR}/")
    print(f"处理后数据目录: {OUTPUT_DIR}/")
