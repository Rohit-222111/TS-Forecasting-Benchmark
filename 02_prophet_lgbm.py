#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
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


prophet_median = np.zeros((N_SERIES, FORECAST_HORIZON))
prophet_lower  = np.zeros((N_SERIES, FORECAST_HORIZON))
prophet_upper  = np.zeros((N_SERIES, FORECAST_HORIZON))

for i, s in enumerate(series_list):
    try:
        if len(s) < FORECAST_HORIZON + 2:
            print(f"Series {i} too short for Prophet ({len(s)} points). Skipping.")
            continue
        if np.any(np.isnan(s)) or np.any(np.isinf(s)):
            raise ValueError(f"Series {i} contains NaN or Inf values.")
        df = pd.DataFrame({
            "ds": pd.date_range("2000-01", periods=len(s), freq="MS"),
            "y":  s
        })
        m      = Prophet(interval_width=0.95, yearly_seasonality=True,
                         weekly_seasonality=False, daily_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=FORECAST_HORIZON, freq="MS")
        fc     = m.predict(future).tail(FORECAST_HORIZON)
        if len(fc) != FORECAST_HORIZON:
            raise ValueError(
                f"Prophet returned {len(fc)} steps, expected {FORECAST_HORIZON}.")
        prophet_median[i] = fc["yhat"].values
        prophet_lower[i]  = fc["yhat_lower"].values
        prophet_upper[i]  = fc["yhat_upper"].values
    except ValueError as e:
        print(f"Prophet ValueError on series {i}: {e}")
        prophet_median[i] = s[-1]
        prophet_lower[i]  = s[-1] * 0.9
        prophet_upper[i]  = s[-1] * 1.1
        continue
    except RuntimeError as e:
        print(f"Prophet RuntimeError on series {i}: {e}")
        prophet_median[i] = s[-1]
        prophet_lower[i]  = s[-1] * 0.9
        prophet_upper[i]  = s[-1] * 1.1
        continue
    except Exception as e:
        print(f"Prophet unexpected error on series {i}: {e}")
        prophet_median[i] = s[-1]
        prophet_lower[i]  = s[-1] * 0.9
        prophet_upper[i]  = s[-1] * 1.1
        continue
    if (i + 1) % 100 == 0:
        print(f"{i+1}/{N_SERIES}")

try:
    with open(DATA_DIR / "prophet_results.pkl", "wb") as f:
        pickle.dump({"median": prophet_median, "lower": prophet_lower,
                     "upper": prophet_upper}, f)
    print("Prophet results saved.")
except (OSError, IOError) as e:
    print(f"File write error saving Prophet results: {e}")
except Exception as e:
    print(f"Could not save Prophet results: {e}")


# In[3]:


def make_lag_features(series, n_lags=24):
    '''
    Convert a time series into supervised learning format using
    a sliding window of lag features.

    Args:
        series (np.ndarray): 1D time series array
        n_lags (int): number of past values to use as features

    Returns:
        tuple: (X, y) as np.ndarray pairs

    Raises:
        ValueError: if series is too short to create any samples
        TypeError: if series is not a numpy array
    '''
    if not isinstance(series, np.ndarray):
        raise TypeError(
            f"make_lag_features expects np.ndarray, got {type(series)}.")
    if len(series) <= n_lags:
        raise ValueError(
            f"Series length {len(series)} must be greater than "
            f"n_lags {n_lags}.")
    X, y = [], []
    for t in range(n_lags, len(series)):
        X.append(series[t - n_lags:t])
        y.append(series[t])
    return np.array(X), np.array(y)

N_LAGS    = 24
QUANTILES = [0.025, 0.5, 0.975]

lgbm_median = np.zeros((N_SERIES, FORECAST_HORIZON))
lgbm_lower  = np.zeros((N_SERIES, FORECAST_HORIZON))
lgbm_upper  = np.zeros((N_SERIES, FORECAST_HORIZON))

for i, s in enumerate(series_list):
    if len(s) < N_LAGS + 5:
        continue
    if np.any(np.isnan(s)) or np.any(np.isinf(s)):
        print(f"Series {i} has NaN/Inf — skipping LightGBM.")
        continue
    try:
        scaler   = StandardScaler()
        s_scaled = scaler.fit_transform(s.reshape(-1, 1)).flatten()
        X, y     = make_lag_features(s_scaled, N_LAGS)
        preds    = {}
        for q in QUANTILES:
            try:
                model = lgb.LGBMRegressor(
                    objective="quantile", alpha=q,
                    n_estimators=200, num_leaves=31, verbose=-1)
                model.fit(X, y)
                history  = list(s_scaled[-N_LAGS:])
                forecast = []
                for _ in range(FORECAST_HORIZON):
                    x_in = np.array(
                        history[-N_LAGS:]).reshape(1, -1)
                    pred = model.predict(x_in)[0]
                    if np.isnan(pred) or np.isinf(pred):
                        raise ValueError(
                            f"LightGBM produced NaN/Inf prediction "
                            f"at quantile {q}.")
                    forecast.append(pred)
                    history.append(pred)
                preds[q] = scaler.inverse_transform(
                    np.array(forecast).reshape(-1, 1)).flatten()
            except ValueError as e:
                print(f"LightGBM ValueError at quantile {q}, "
                      f"series {i}: {e}")
                preds[q] = np.full(FORECAST_HORIZON, s[-1])
                continue
            except Exception as e:
                print(f"LightGBM error at quantile {q}, "
                      f"series {i}: {e}")
                preds[q] = np.full(FORECAST_HORIZON, s[-1])
                continue

        if len(preds) != 3:
            raise RuntimeError(
                f"Only {len(preds)}/3 quantiles fitted for "
                f"series {i}.")

        lgbm_lower[i]  = preds[0.025]
        lgbm_median[i] = preds[0.5]
        lgbm_upper[i]  = preds[0.975]

    except (TypeError, ValueError) as e:
        print(f"LightGBM setup error on series {i}: {e}")
        lgbm_median[i] = s[-1]
        lgbm_lower[i]  = s[-1] * 0.9
        lgbm_upper[i]  = s[-1] * 1.1
        continue
    except RuntimeError as e:
        print(f"LightGBM RuntimeError on series {i}: {e}")
        lgbm_median[i] = s[-1]
        lgbm_lower[i]  = s[-1] * 0.9
        lgbm_upper[i]  = s[-1] * 1.1
        continue
    except Exception as e:
        print(f"LightGBM unexpected error on series {i}: {e}")
        lgbm_median[i] = s[-1]
        lgbm_lower[i]  = s[-1] * 0.9
        lgbm_upper[i]  = s[-1] * 1.1
        continue

    if (i + 1) % 100 == 0:
        print(f"{i+1}/{N_SERIES}")

try:
    with open(DATA_DIR / "lgbm_results.pkl", "wb") as f:
        pickle.dump({"median": lgbm_median, "lower": lgbm_lower,
                     "upper": lgbm_upper}, f)
    print("LightGBM results saved.")
except (OSError, IOError) as e:
    print(f"File write error saving LightGBM results: {e}")
except Exception as e:
    print(f"Could not save LightGBM results: {e}")


# In[5]:


with open(DATA_DIR / "lgbm_results.pkl", "rb") as f:
    lgbm_fix = pickle.load(f)

lgbm_lower  = lgbm_fix["lower"]
lgbm_median = lgbm_fix["median"]
lgbm_upper  = lgbm_fix["upper"]

fixed_count = 0
for i in range(len(lgbm_lower)):
    try:
        crossed = lgbm_lower[i] > lgbm_upper[i]
        if np.any(crossed):
            mid           = (lgbm_lower[i] + lgbm_upper[i]) / 2
            spread        = np.abs(lgbm_upper[i] - lgbm_lower[i])
            lgbm_lower[i] = np.where(crossed, mid - spread/2,
                                     lgbm_lower[i])
            lgbm_upper[i] = np.where(crossed, mid + spread/2,
                                     lgbm_upper[i])
            fixed_count  += 1
    except Exception as e:
        print(f"Could not fix crossing for series {i}: {e}")
        continue

print(f"Quantile crossing fixed for {fixed_count} series.")

try:
    with open(DATA_DIR / "lgbm_results.pkl", "wb") as f:
        pickle.dump({"median": lgbm_median, "lower": lgbm_lower,
                     "upper": lgbm_upper}, f)
    print("LightGBM results resaved with crossing fix.")
except (OSError, IOError) as e:
    print(f"File write error saving fixed LightGBM results: {e}")
except Exception as e:
    print(f"Could not save fixed LightGBM results: {e}")


# In[6]:


try:
    with open(DATA_DIR / "prophet_results.pkl", "rb") as f:
        pr = pickle.load(f)
    with open(DATA_DIR / "lgbm_results.pkl", "rb") as f:
        lg = pickle.load(f)
    print("Prophet median shape:", pr["median"].shape)
    print("LightGBM median shape:", lg["median"].shape)
    print("Prophet sample (series 0):", pr["median"][0].round(1))
    print("LightGBM sample (series 0):", lg["median"][0].round(1))
except FileNotFoundError as e:
    print(f"Results file missing: {e}")
except Exception as e:
    print(f"Verification error: {e}")

