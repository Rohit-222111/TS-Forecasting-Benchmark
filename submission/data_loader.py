'''
Handles all data loading for the M4 forecasting benchmark.

Functions:
    load_series        -- load cleaned series list from disk
    load_actuals       -- load ground truth test values
    load_model_results -- load all saved model prediction dicts
    load_metrics       -- load the computed metrics DataFrame
'''

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_DIR


def load_series():
    '''
    Load the preprocessed list of M4 time series.

    Returns:
        list of np.ndarray: each array is one cleaned time series

    Raises:
        FileNotFoundError: if series_list.pkl does not exist
        ValueError: if loaded object is not a non-empty list
        RuntimeError: for any other loading failure
    '''
    try:
        with open(DATA_DIR / "series_list.pkl", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"series_list.pkl not found in {DATA_DIR}. "
            "Run 01_data.ipynb first.")
    except (pickle.UnpicklingError, EOFError) as e:
        raise RuntimeError(
            f"series_list.pkl is corrupted or empty: {e}") from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading series: {e}") from e

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            "series_list.pkl loaded but contains no series. "
            "Re-run 01_data.ipynb.")
    return data


def load_actuals():
    '''
    Load ground truth test values for all series.

    Returns:
        np.ndarray: shape (N_SERIES, FORECAST_HORIZON)

    Raises:
        FileNotFoundError: if test_sub.pkl does not exist
        ValueError: if loaded array has wrong dimensions
        RuntimeError: for any other loading failure
    '''
    try:
        with open(DATA_DIR / "test_sub.pkl", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"test_sub.pkl not found in {DATA_DIR}. "
            "Run 01_data.ipynb first.")
    except (pickle.UnpicklingError, EOFError) as e:
        raise RuntimeError(
            f"test_sub.pkl is corrupted or empty: {e}") from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading actuals: {e}") from e

    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError(
            f"test_sub.pkl has unexpected shape: "
            f"{getattr(data, 'shape', type(data))}. "
            "Re-run 01_data.ipynb.")
    return data


def load_model_results():
    '''
    Load prediction dictionaries for all five models.
    Skips missing files with a warning instead of crashing.

    Returns:
        dict: model name -> dict with keys median, lower, upper
              missing models are excluded from the dict

    Raises:
        RuntimeError: if no model files could be loaded at all
    '''
    model_files = {
        "Prophet":  "prophet_results.pkl",
        "LightGBM": "lgbm_results.pkl",
        "ARIMA":    "arima_results.pkl",
        "LSTM":     "lstm_results.pkl",
        "Ensemble": "ensemble_results.pkl"
    }
    models = {}
    for name, fname in model_files.items():
        try:
            with open(DATA_DIR / fname, "rb") as f:
                result = pickle.load(f)
            if not isinstance(result, dict):
                raise ValueError(
                    f"{fname} does not contain a dict.")
            for key in ("median", "lower", "upper"):
                if key not in result:
                    raise KeyError(
                        f"{fname} missing required key: '{key}'")
                if not isinstance(result[key], np.ndarray):
                    raise TypeError(
                        f"{fname}['{key}'] is not a numpy array.")
            models[name] = result
        except FileNotFoundError:
            print(f"Warning: {fname} not found. Skipping {name}.")
            continue
        except (pickle.UnpicklingError, EOFError):
            print(f"Warning: {fname} is corrupted. "
                  f"Skipping {name}.")
            continue
        except (ValueError, KeyError, TypeError) as e:
            print(f"Warning: {fname} has invalid format: {e}. "
                  f"Skipping {name}.")
            continue
        except Exception as e:
            print(f"Warning: Unexpected error loading "
                  f"{name}: {e}. Skipping.")
            continue

    if not models:
        raise RuntimeError(
            "No model result files could be loaded. "
            "Run notebooks 02-04 first.")
    return models


def load_metrics():
    '''
    Load precomputed metrics CSV into a DataFrame.

    Returns:
        pd.DataFrame: rows=models, cols=RMSE/Coverage/Sharpness

    Raises:
        FileNotFoundError: if metrics.csv does not exist
        ValueError: if CSV is missing required columns
        RuntimeError: for any other loading failure
    '''
    try:
        df = pd.read_csv(DATA_DIR / "metrics.csv",
                         index_col="Model")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"metrics.csv not found in {DATA_DIR}. "
            "Run 04_ensemble_eval.ipynb first.")
    except ValueError as e:
        raise ValueError(
            f"metrics.csv missing 'Model' index column: "
            f"{e}") from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading metrics: {e}") from e

    required_cols = {"RMSE", "Coverage", "Sharpness"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"metrics.csv is missing columns: {missing}. "
            "Re-run 04_ensemble_eval.ipynb.")
    return df