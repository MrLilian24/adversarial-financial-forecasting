import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch as th
from typing import Tuple

def sliding_window(X: np.ndarray, lookback: int = 60) -> Tuple[th.Tensor, th.Tensor]:
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(X[i+lookback, 0])
    X_seq = th.tensor(np.array(X_seq), dtype=th.float32)
    y_seq = th.tensor(np.array(y_seq), dtype=th.float32)
    return X_seq, y_seq

def prepare_data(ticker: str = "AAPL", period: str = "5y", lookback: int = 60):
    data = yf.download(tickers=[ticker], period=period, interval="1d", auto_adjust=True)
    data.columns = data.columns.droplevel(1)
    data = data[["Close"]].dropna()

    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)

    X_train = data[:train_size].values
    X_val = data[train_size:train_size+val_size].values
    X_test = data[train_size+val_size:].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_seq, y_train_seq = sliding_window(X_train_scaled, lookback)
    X_val_seq, y_val_seq = sliding_window(X_val_scaled, lookback)
    X_test_seq, y_test_seq = sliding_window(X_test_scaled, lookback)

    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, scaler
