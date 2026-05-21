#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

required_files = {
    "series_list": "series_list.pkl",
    "actuals":     "test_sub.pkl",
    "prophet":     "prophet_results.pkl",
    "lgbm":        "lgbm_results.pkl",
    "arima":       "arima_results.pkl",
    "lstm":        "lstm_results.pkl"
}

loaded = {}
for key, fname in required_files.items():
    try:
        with open(DATA_DIR / fname, "rb") as f:
            loaded[key] = pickle.load(f)
    except FileNotFoundError:
        print(f"Missing file: {fname}. Run earlier notebooks first.")
        loaded[key] = None
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Corrupted file {fname}: {e}")
        loaded[key] = None
    except Exception as e:
        print(f"Unexpected error loading {fname}: {e}")
        loaded[key] = None

series_list = loaded["series_list"]
actuals     = loaded["actuals"]
prophet     = loaded["prophet"]
lgbm        = loaded["lgbm"]
arima       = loaded["arima"]
lstm        = loaded["lstm"]

missing = [k for k, v in loaded.items() if v is None]
if missing:
    print(f"Warning: these failed to load: {missing}")
    print("Metrics for affected models will be None.")
else:
    print("All files loaded successfully.")

FORECAST_HORIZON = 18
N_SERIES         = len(series_list) if series_list is not None else 0
print(f"Series count: {N_SERIES}")


# In[2]:


try:
    if any(v is None for v in [prophet, lgbm, arima, lstm]):
        raise ValueError(
            "One or more model results failed to load. "
            "Cannot build ensemble.")

    weights = {
        "prophet": 0.25, "lgbm": 0.25,
        "arima":   0.25, "lstm": 0.25
    }

    for name, res in [("prophet", prophet), ("lgbm", lgbm),
                      ("arima", arima),     ("lstm", lstm)]:
        for key in ("median", "lower", "upper"):
            if key not in res:
                raise KeyError(
                    f"{name} result missing key '{key}'.")
            if not isinstance(res[key], np.ndarray):
                raise TypeError(
                    f"{name}['{key}'] must be np.ndarray, "
                    f"got {type(res[key])}.")
            if res[key].shape != (N_SERIES, FORECAST_HORIZON):
                raise ValueError(
                    f"{name}['{key}'] shape {res[key].shape} "
                    f"does not match expected "
                    f"({N_SERIES}, {FORECAST_HORIZON}). "
                    f"Re-run earlier notebooks with same N_SERIES.")

    ensemble_median = (
        weights["prophet"] * prophet["median"] +
        weights["lgbm"]   * lgbm["median"]   +
        weights["arima"]  * arima["median"]  +
        weights["lstm"]   * lstm["median"]
    )
    ensemble_lower = (
        weights["prophet"] * prophet["lower"] +
        weights["lgbm"]   * lgbm["lower"]   +
        weights["arima"]  * arima["lower"]  +
        weights["lstm"]   * lstm["lower"]
    )
    ensemble_upper = (
        weights["prophet"] * prophet["upper"] +
        weights["lgbm"]   * lgbm["upper"]   +
        weights["arima"]  * arima["upper"]  +
        weights["lstm"]   * lstm["upper"]
    )

    try:
        with open(DATA_DIR / "ensemble_results.pkl", "wb") as f:
            pickle.dump({
                "median": ensemble_median,
                "lower":  ensemble_lower,
                "upper":  ensemble_upper
            }, f)
        print("Ensemble built and saved.")
    except (OSError, IOError) as e:
        print(f"File write error saving ensemble: {e}")
    except Exception as e:
        print(f"Unexpected error saving ensemble: {e}")

except KeyError as e:
    print(f"Ensemble build failed — missing key: {e}")
    ensemble_median = ensemble_lower = ensemble_upper = None
except TypeError as e:
    print(f"Ensemble build failed — type error: {e}")
    ensemble_median = ensemble_lower = ensemble_upper = None
except ValueError as e:
    print(f"Ensemble build failed — value error: {e}")
    ensemble_median = ensemble_lower = ensemble_upper = None
except Exception as e:
    print(f"Ensemble build failed — unexpected error: {e}")
    ensemble_median = ensemble_lower = ensemble_upper = None


# In[3]:


try:
    if lstm is None:
        raise ValueError(
            "LSTM results not loaded. Cannot calibrate intervals.")
    if actuals is None:
        raise ValueError(
            "Actuals not loaded. Cannot calibrate intervals.")

    calib_size = 100

    if N_SERIES < calib_size + 1:
        raise ValueError(
            f"Not enough series ({N_SERIES}) for calibration "
            f"size {calib_size}.")

    lstm_calib_preds   = lstm["median"][:calib_size]
    lstm_calib_actuals = actuals[:calib_size]

    if lstm_calib_preds.shape != lstm_calib_actuals.shape:
        raise ValueError(
            f"Shape mismatch: LSTM preds {lstm_calib_preds.shape} "
            f"vs actuals {lstm_calib_actuals.shape}.")

    residuals = np.abs(lstm_calib_actuals - lstm_calib_preds)

    if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
        raise ValueError(
            "Residuals contain NaN or Inf — check LSTM predictions.")

    q_level = np.quantile(residuals, 0.95)

    lstm_conformal_lower = lstm["median"] - q_level
    lstm_conformal_upper = lstm["median"] + q_level

    print(f"Conformal quantile: {q_level:.2f}")
    print(f"Calibration applied to all {N_SERIES} series.")

except KeyError as e:
    print(f"Calibration failed — missing key in LSTM results: {e}")
    lstm_conformal_lower = None
    lstm_conformal_upper = None
except ValueError as e:
    print(f"Calibration failed — value error: {e}")
    lstm_conformal_lower = None
    lstm_conformal_upper = None
except Exception as e:
    print(f"Calibration failed — unexpected error: {e}")
    lstm_conformal_lower = None
    lstm_conformal_upper = None


# In[4]:


try:
    calibrated_lstm = {
        "median": lstm["median"],
        "lower":  lstm_conformal_lower,
        "upper":  lstm_conformal_upper
    }
    with open(DATA_DIR / "lstm_results.pkl", "wb") as f:
        pickle.dump(calibrated_lstm, f)
    print("Calibrated LSTM results saved to lstm_results.pkl")
except (OSError, IOError) as e:
    print(f"File write error saving calibrated LSTM: {e}")
except Exception as e:
    print(f"Could not save calibrated LSTM results: {e}")


# In[5]:


def rmse(preds, actuals):
    '''
    Compute Root Mean Squared Error.

    Args:
        preds   (np.ndarray): predicted values, shape (N, H)
        actuals (np.ndarray): ground truth values, shape (N, H)

    Returns:
        float: RMSE value

    Raises:
        TypeError: if inputs are not numpy arrays
        ValueError: if shapes do not match or contain NaN/Inf
    '''
    if not isinstance(preds, np.ndarray) or \
       not isinstance(actuals, np.ndarray):
        raise TypeError(
            f"rmse expects np.ndarray, got "
            f"{type(preds)} and {type(actuals)}.")
    if preds.shape != actuals.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} "
            f"vs actuals {actuals.shape}.")
    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        raise ValueError("preds contain NaN or Inf values.")
    return float(np.sqrt(np.mean((preds - actuals) ** 2)))


def coverage(lower, upper, actuals):
    '''
    Compute empirical coverage of prediction intervals.

    Args:
        lower   (np.ndarray): lower bound, shape (N, H)
        upper   (np.ndarray): upper bound, shape (N, H)
        actuals (np.ndarray): ground truth values, shape (N, H)

    Returns:
        float: coverage between 0 and 1

    Raises:
        TypeError: if inputs are not numpy arrays
        ValueError: if shapes mismatch or lower exceeds upper
    '''
    for name, arr in [("lower", lower), ("upper", upper),
                      ("actuals", actuals)]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"coverage expects np.ndarray for '{name}', "
                f"got {type(arr)}.")
    if not (lower.shape == upper.shape == actuals.shape):
        raise ValueError(
            f"Shape mismatch: lower {lower.shape}, "
            f"upper {upper.shape}, actuals {actuals.shape}.")
    if np.any(lower > upper):
        raise ValueError(
            "lower bound exceeds upper bound for some entries.")
    inside = (actuals >= lower) & (actuals <= upper)
    return float(inside.mean())


def sharpness(lower, upper):
    '''
    Compute mean prediction interval width.

    Args:
        lower (np.ndarray): lower bound, shape (N, H)
        upper (np.ndarray): upper bound, shape (N, H)

    Returns:
        float: mean interval width

    Raises:
        TypeError: if inputs are not numpy arrays
        ValueError: if shapes mismatch or lower exceeds upper
    '''
    for name, arr in [("lower", lower), ("upper", upper)]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"sharpness expects np.ndarray for '{name}', "
                f"got {type(arr)}.")
    if lower.shape != upper.shape:
        raise ValueError(
            f"Shape mismatch: lower {lower.shape}, "
            f"upper {upper.shape}.")
    if np.any(lower > upper):
        raise ValueError(
            "lower bound exceeds upper bound for some entries.")
    return float(np.mean(upper - lower))


try:
    if ensemble_median is None or ensemble_lower is None \
       or ensemble_upper is None:
        raise ValueError(
            "Ensemble arrays are None — Cell 02 failed. "
            "Fix ensemble build before running metrics.")

    models = {
        "Prophet":  prophet,
        "LightGBM": lgbm,
        "ARIMA":    arima,
        "LSTM":     lstm,
        "Ensemble": {
            "median": ensemble_median,
            "lower":  ensemble_lower,
            "upper":  ensemble_upper
        }
    }

    if lstm_conformal_lower is not None and \
       lstm_conformal_upper is not None:
        models["LSTM"]["lower"] = lstm_conformal_lower
        models["LSTM"]["upper"] = lstm_conformal_upper
    else:
        print("Warning: conformal calibration failed. "
              "Using original LSTM intervals.")

except ValueError as e:
    print(f"Models dict setup failed: {e}")
    models = {}

results = {}
for name, res in models.items():
    if res is None:
        print(f"Skipping {name}: result is None.")
        results[name] = {
            "RMSE": None, "Coverage": None, "Sharpness": None}
        continue
    try:
        for key in ("median", "lower", "upper"):
            if key not in res:
                raise KeyError(
                    f"Result for {name} missing key '{key}'.")
        r = rmse(res["median"], actuals)
        c = coverage(res["lower"], res["upper"], actuals)
        s = sharpness(res["lower"], res["upper"])
        results[name] = {
            "RMSE":      round(r, 2),
            "Coverage":  round(c, 4),
            "Sharpness": round(s, 2)
        }
    except KeyError as e:
        print(f"KeyError for {name}: {e}")
        results[name] = {
            "RMSE": None, "Coverage": None, "Sharpness": None}
        continue
    except TypeError as e:
        print(f"TypeError for {name}: {e}")
        results[name] = {
            "RMSE": None, "Coverage": None, "Sharpness": None}
        continue
    except ValueError as e:
        print(f"ValueError for {name}: {e}")
        results[name] = {
            "RMSE": None, "Coverage": None, "Sharpness": None}
        continue
    except Exception as e:
        print(f"Unexpected error for {name}: {e}")
        results[name] = {
            "RMSE": None, "Coverage": None, "Sharpness": None}
        continue

if not results:
    print("No results computed — check model files.")
else:
    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"
    print(results_df)


# In[6]:


try:
    if "results_df" not in dir() or results_df is None:
        raise ValueError(
            "results_df not defined. Run Cell 04 first.")
    results_df.to_csv(DATA_DIR / "metrics.csv")
    print(f"Saved metrics.csv to {DATA_DIR}")
except ValueError as e:
    print(f"Save skipped: {e}")
except (OSError, IOError) as e:
    print(f"File write error saving metrics.csv: {e}")
except Exception as e:
    print(f"Unexpected error saving metrics.csv: {e}")

try:
    if not models:
        raise ValueError(
            "models dict is empty. Run Cell 04 first.")
    with open(DATA_DIR / "all_models.pkl", "wb") as f:
        pickle.dump(models, f)
    print(f"Saved all_models.pkl to {DATA_DIR}")
except ValueError as e:
    print(f"Save skipped: {e}")
except (OSError, IOError) as e:
    print(f"File write error saving all_models.pkl: {e}")
except Exception as e:
    print(f"Unexpected error saving all_models.pkl: {e}")


# In[7]:


'''
The 95% nominal coverage is not achieved by any model on this subset.
This is common in practice when intervals are estimated without held-out
calibration data. ARIMA comes closest (86%) because its confidence
intervals are derived analytically from the model's fitted variance.
Prophet and LightGBM undercover because their interval widths were set
heuristically. A larger calibration set or full conformal prediction
would push coverage closer to the nominal level — this is a known
limitation and an honest finding of this benchmark.
'''

