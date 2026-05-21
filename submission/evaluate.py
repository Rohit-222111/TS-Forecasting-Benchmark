'''
Computes forecasting evaluation metrics for the benchmark.

Metrics:
    RMSE       -- Root Mean Squared Error
    Coverage   -- Fraction of actuals inside prediction interval
    Sharpness  -- Average prediction interval width

Functions:
    rmse, coverage, sharpness, evaluate_all_models
'''

import numpy as np
import pandas as pd


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
        ValueError: if shapes mismatch or contain NaN/Inf
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
        raise ValueError(
            "preds contain NaN or Inf values.")
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


def evaluate_all_models(models, actuals):
    '''
    Evaluate all models and return a summary DataFrame.
    Uses continue to skip models with missing data.
    Uses break when no remaining models have valid data.

    Args:
        models  (dict): model name -> dict with median/lower/upper
        actuals (np.ndarray): ground truth, shape (N, H)

    Returns:
        pd.DataFrame: rows=models, cols=RMSE/Coverage/Sharpness

    Raises:
        TypeError: if models is not a dict or actuals not ndarray
        ValueError: if models dict is empty
        RuntimeError: if all models fail evaluation
    '''
    if not isinstance(models, dict):
        raise TypeError(
            f"evaluate_all_models expects a dict, "
            f"got {type(models)}.")
    if not isinstance(actuals, np.ndarray):
        raise TypeError(
            f"actuals must be np.ndarray, got {type(actuals)}.")
    if len(models) == 0:
        raise ValueError(
            "models dict is empty — nothing to evaluate.")

    results = {}
    names   = list(models.keys())

    for i, name in enumerate(names):
        res = models.get(name)

        if res is None:
            print(f"Skipping {name}: result is None.")
            continue

        remaining = [models[n] for n in names[i:]
                     if models.get(n) is not None]
        if not remaining:
            print("No remaining models with data — stopping.")
            break

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
                "RMSE": None, "Coverage": None,
                "Sharpness": None}
            continue
        except TypeError as e:
            print(f"TypeError for {name}: {e}")
            results[name] = {
                "RMSE": None, "Coverage": None,
                "Sharpness": None}
            continue
        except ValueError as e:
            print(f"ValueError for {name}: {e}")
            results[name] = {
                "RMSE": None, "Coverage": None,
                "Sharpness": None}
            continue
        except Exception as e:
            print(f"Unexpected error for {name}: {e}")
            results[name] = {
                "RMSE": None, "Coverage": None,
                "Sharpness": None}
            continue

    if not results:
        raise RuntimeError(
            "All models failed evaluation. "
            "Check your result files.")

    df            = pd.DataFrame(results).T
    df.index.name = "Model"
    return df