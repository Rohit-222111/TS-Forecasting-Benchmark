#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import torch
import torch.nn as nn
import warnings
from pmdarima import auto_arima
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

warnings.filterwarnings("ignore")
DATA_DIR = Path("data")

try:
    with open(DATA_DIR / "series_list.pkl", "rb") as f:
        series_list = pickle.load(f)
    with open(DATA_DIR / "test_sub.pkl", "rb") as f:
        actuals = pickle.load(f)
    print(f"Loaded {len(series_list)} series.")
except FileNotFoundError as e:
    print(f"Data file missing: {e}")
except Exception as e:
    print(f"Load error: {e}")

FORECAST_HORIZON = 18
N_SERIES         = len(series_list)


# In[2]:


arima_median = np.zeros((N_SERIES, FORECAST_HORIZON))
arima_lower  = np.zeros((N_SERIES, FORECAST_HORIZON))
arima_upper  = np.zeros((N_SERIES, FORECAST_HORIZON))

consecutive_failures = 0

for i, s in enumerate(series_list):
    clean_s = s[~np.isnan(s) & ~np.isinf(s)]

    if len(clean_s) == 0:
        arima_median[i] = 0
        arima_lower[i]  = 0
        arima_upper[i]  = 0
        continue

    if len(clean_s) < FORECAST_HORIZON + 2:
        arima_median[i] = clean_s[-1]
        arima_lower[i]  = clean_s[-1] * 0.9
        arima_upper[i]  = clean_s[-1] * 1.1
        continue

    if len(clean_s) != len(s):
        print(f"Series {i}: removed {len(s) - len(clean_s)} "
              f"NaN/Inf values before fitting.")

    if len(clean_s) < 36:
        seasonal = False
    else:
        seasonal = True

    try:
        model  = auto_arima(clean_s, seasonal=seasonal, m=12,
                            suppress_warnings=True,
                            error_action="ignore",
                            stepwise=True, maxiter=50,
                            with_ocsb=False,
                            with_ch=False)
        fc, ci = model.predict(n_periods=FORECAST_HORIZON,
                               return_conf_int=True, alpha=0.05)

        if len(fc) != FORECAST_HORIZON:
            raise ValueError(
                f"ARIMA returned {len(fc)} steps, "
                f"expected {FORECAST_HORIZON}.")
        if np.any(np.isnan(fc)) or np.any(np.isinf(fc)):
            raise ValueError(
                "ARIMA forecast contains NaN or Inf.")
        if np.any(np.isnan(ci)) or np.any(np.isinf(ci)):
            raise ValueError(
                "ARIMA confidence intervals contain NaN or Inf.")

        arima_median[i]      = fc
        arima_lower[i]       = ci[:, 0]
        arima_upper[i]       = ci[:, 1]
        consecutive_failures = 0

    except ValueError as e:
        print(f"ARIMA ValueError on series {i}: {e}")
        arima_median[i] = clean_s[-1]
        arima_lower[i]  = clean_s[-1] * 0.9
        arima_upper[i]  = clean_s[-1] * 1.1
        consecutive_failures += 1
        if consecutive_failures >= 20:
            print("20 consecutive ARIMA failures — stopping early.")
            break
        continue
    except RuntimeError as e:
        print(f"ARIMA RuntimeError on series {i}: {e}")
        arima_median[i] = clean_s[-1]
        arima_lower[i]  = clean_s[-1] * 0.9
        arima_upper[i]  = clean_s[-1] * 1.1
        consecutive_failures += 1
        if consecutive_failures >= 20:
            print("20 consecutive ARIMA failures — stopping early.")
            break
        continue
    except Exception as e:
        print(f"ARIMA unexpected error on series {i}: {e}")
        arima_median[i] = clean_s[-1]
        arima_lower[i]  = clean_s[-1] * 0.9
        arima_upper[i]  = clean_s[-1] * 1.1
        consecutive_failures += 1
        if consecutive_failures >= 20:
            print("20 consecutive ARIMA failures — stopping early.")
            break
        continue

    if (i + 1) % 50 == 0:
        print(f"{i+1}/{N_SERIES}")

try:
    with open(DATA_DIR / "arima_results.pkl", "wb") as f:
        pickle.dump({"median": arima_median, "lower": arima_lower,
                     "upper": arima_upper}, f)
    print("ARIMA results saved.")
except (OSError, IOError) as e:
    print(f"File write error saving ARIMA results: {e}")
except Exception as e:
    print(f"Could not save ARIMA results: {e}")


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=2, output_size=18, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# In[4]:


SEQ_LEN    = 36
BATCH_SIZE = 64
EPOCHS     = 80

def prepare_lstm_data(series_list, seq_len):
    X_all, y_all, scalers = [], [], []
    for s in series_list:
        if len(s) < seq_len + FORECAST_HORIZON:
            scalers.append(None)
            continue
        try:
            scaler = StandardScaler()
            s_sc   = scaler.fit_transform(s.reshape(-1, 1)).flatten()
            scalers.append(scaler)
            for t in range(seq_len, len(s_sc) - FORECAST_HORIZON + 1):
                X_all.append(s_sc[t - seq_len:t])
                y_all.append(s_sc[t:t + FORECAST_HORIZON])
        except Exception as e:
            print(f"LSTM data prep error: {e}")
            scalers.append(None)
            continue
    return np.array(X_all), np.array(y_all), scalers

try:
    X, y, scalers = prepare_lstm_data(series_list, SEQ_LEN)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training samples: {len(X_t)}")
except Exception as e:
    print(f"Failed to prepare LSTM data: {e}")


# In[5]:


try:
    model     = LSTMForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            try:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Batch error at epoch {epoch}: {e}")
                continue
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}  loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), DATA_DIR / "lstm_checkpoint.pt")
    print("Checkpoint saved.")
except Exception as e:
    print(f"LSTM training failed: {e}")


# In[6]:


lstm_median = np.zeros((N_SERIES, FORECAST_HORIZON))

try:
    model.eval()
    with torch.no_grad():
        for i, s in enumerate(series_list):
            try:
                scaler = scalers[i]
                if scaler is None or len(s) < SEQ_LEN:
                    lstm_median[i] = s[-1]
                    continue
                s_sc  = scaler.transform(s.reshape(-1, 1)).flatten()
                x_in  = torch.tensor(s_sc[-SEQ_LEN:],
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                pred  = model(x_in).cpu().numpy().flatten()
                lstm_median[i] = scaler.inverse_transform(
                    pred.reshape(-1, 1)).flatten()
                if np.any(np.isnan(lstm_median[i])) or \
                   np.any(np.isinf(lstm_median[i])):
                    raise ValueError(
                        f"LSTM produced NaN/Inf for series {i}.")
            except Exception as e:
                print(f"LSTM inference failed on series {i}: {e}")
                lstm_median[i] = s[-1]
                continue
except Exception as e:
    print(f"LSTM inference loop failed: {e}")

lstm_lower = lstm_median * 0.92
lstm_upper = lstm_median * 1.08

try:
    with open(DATA_DIR / "lstm_results.pkl", "wb") as f:
        pickle.dump({"median": lstm_median, "lower": lstm_lower,
                     "upper": lstm_upper}, f)
    print("LSTM results saved.")
    print("LSTM median shape:", lstm_median.shape)
except Exception as e:
    print(f"Could not save LSTM results: {e}")

