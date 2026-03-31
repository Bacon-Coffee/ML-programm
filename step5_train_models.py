"""
Step 5: 模型训练 + 评估
包含三类模型：
  1. 纯 LSTM（只用股价特征，9维）
  2. FinBERT 融合 LSTM（股价 + FinBERT情感，12维）
  3. 预计算情感融合 LSTM（AAPL专项，12维）
  4. 纯情感 MLP（只用情感特征，3维）
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, json

os.makedirs("results/models", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")

# ==================== 超参数 ====================
WINDOW      = 30      # 用过去30天预测第31天
BATCH_SIZE  = 32
LR          = 0.001
EPOCHS      = 100
PATIENCE    = 10
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
# =================================================

PRICE_COLS     = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA20", "RSI", "MACD"]
SENTIMENT_COLS = ["sentiment_neg", "sentiment_neu", "sentiment_pos"]
TARGET_COL     = "Close"


# ==================== 数据集 ====================
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_windows(df, feature_cols):
    X, y = [], []
    vals = df[feature_cols].values
    target = df[TARGET_COL].values
    for i in range(WINDOW, len(df)):
        X.append(vals[i - WINDOW:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def split_data(X, y):
    n = len(y)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)
    return (X[:t1], y[:t1]), (X[t1:t2], y[t1:t2]), (X[t2:], y[t2:])


# ==================== 模型 ====================
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


class MLPModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, window, features) → 取最后一天的情感
        return self.net(x[:, -1, :]).squeeze(1)


# ==================== 训练 ====================
def train(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val, no_improve = float("inf"), 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(DEVICE)), y_b.to(DEVICE)).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  早停于 epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model, test_loader, y_test_raw):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            preds.extend(model(X_b.to(DEVICE)).cpu().numpy())
    preds = np.array(preds)

    rmse = np.sqrt(mean_squared_error(y_test_raw, preds))
    mae  = mean_absolute_error(y_test_raw, preds)
    r2   = r2_score(y_test_raw, preds)
    # 方向准确率
    actual_dir = np.diff(y_test_raw)
    pred_dir   = np.diff(preds)
    dir_acc    = np.mean(np.sign(actual_dir) == np.sign(pred_dir))

    return {"RMSE": round(rmse, 6), "MAE": round(mae, 6),
            "R2": round(r2, 4), "DirAcc": round(dir_acc, 4)}, preds


# ==================== 主流程 ====================
EXPERIMENTS = [
    # (ticker, data_file,              feature_cols,                      model_type, label)
    ("AAPL", "merged_AAPL_finbert",     PRICE_COLS,                        "lstm",  "纯LSTM"),
    ("AAPL", "merged_AAPL_finbert",     PRICE_COLS + SENTIMENT_COLS,       "lstm",  "FinBERT融合"),
    ("AAPL", "merged_AAPL_precomputed", PRICE_COLS + SENTIMENT_COLS,       "lstm",  "预计算情感融合"),
    ("AAPL", "merged_AAPL_finbert",     SENTIMENT_COLS,                    "mlp",   "纯情感MLP"),
    ("TSLA", "merged_TSLA_finbert",     PRICE_COLS,                        "lstm",  "纯LSTM"),
    ("TSLA", "merged_TSLA_finbert",     PRICE_COLS + SENTIMENT_COLS,       "lstm",  "FinBERT融合"),
    ("TSLA", "merged_TSLA_finbert",     SENTIMENT_COLS,                    "mlp",   "纯情感MLP"),
    ("AMZN", "merged_AMZN_finbert",     PRICE_COLS,                        "lstm",  "纯LSTM"),
    ("AMZN", "merged_AMZN_finbert",     PRICE_COLS + SENTIMENT_COLS,       "lstm",  "FinBERT融合"),
    ("AMZN", "merged_AMZN_finbert",     SENTIMENT_COLS,                    "mlp",   "纯情感MLP"),
]

all_results = []

for ticker, data_file, feat_cols, model_type, label in EXPERIMENTS:
    print(f"\n{'='*50}")
    print(f"  {ticker} — {label}  ({len(feat_cols)}维输入)")
    print(f"{'='*50}")

    df = pd.read_csv(f"data/processed/{data_file}.csv")
    X, y = build_windows(df, feat_cols)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_data(X, y)

    train_loader = DataLoader(StockDataset(X_tr, y_tr), BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(StockDataset(X_va, y_va), BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(StockDataset(X_te, y_te), BATCH_SIZE, shuffle=False)

    model = (LSTMModel(len(feat_cols)) if model_type == "lstm"
             else MLPModel(len(feat_cols))).to(DEVICE)

    model = train(model, train_loader, val_loader)
    metrics, preds = evaluate(model, test_loader, y_te)

    print(f"  RMSE={metrics['RMSE']}  MAE={metrics['MAE']}  "
          f"R2={metrics['R2']}  DirAcc={metrics['DirAcc']}")

    # 保存模型
    torch.save(model.state_dict(),
               f"results/models/{ticker}_{label.replace(' ','_')}.pt")

    # 保存预测值（画图用）
    pd.DataFrame({"actual": y_te, "pred": preds}).to_csv(
        f"results/tables/{ticker}_{label.replace(' ','_')}_preds.csv", index=False)

    all_results.append({"ticker": ticker, "model": label, **metrics})

# 汇总表格
results_df = pd.DataFrame(all_results)
results_df.to_csv("results/tables/all_results.csv", index=False)
print("\n\n========== 全部实验结果 ==========")
print(results_df.to_string(index=False))
print(f"\n结果已保存至 results/tables/all_results.csv")
