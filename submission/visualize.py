'''
Generates all dashboard visualizations for the forecasting benchmark.
Saves interactive HTML charts to the charts/ directory.

Functions:
    plot_fan_chart   -- interval fan chart for a single series
    plot_metrics     -- bar charts for RMSE, Coverage, Sharpness
    plot_calibration -- coverage calibration curves per model
    save_all_charts  -- runs all plots and saves HTML files
'''

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from config import CHARTS_DIR, FORECAST_HORIZON

COLORS = {
    "Prophet":  ("rgba(99,110,250,0.15)",  "rgb(99,110,250)"),
    "LightGBM": ("rgba(239,85,59,0.15)",   "rgb(239,85,59)"),
    "ARIMA":    ("rgba(0,204,150,0.15)",    "rgb(0,204,150)"),
    "LSTM":     ("rgba(255,161,90,0.15)",   "rgb(255,161,90)"),
    "Ensemble": ("rgba(171,99,250,0.15)",   "rgb(171,99,250)")
}


def _validate_model_result(name, res):
    '''
    Validate a single model result dict has correct structure.

    Args:
        name (str): model name for error messages
        res  (dict): result dict to validate

    Raises:
        TypeError: if res is not a dict
        KeyError: if required keys are missing
        ValueError: if arrays have wrong dimensions
    '''
    if not isinstance(res, dict):
        raise TypeError(
            f"{name} result must be a dict, got {type(res)}.")
    for key in ("median", "lower", "upper"):
        if key not in res:
            raise KeyError(
                f"{name} result missing required key: '{key}'.")
        if not isinstance(res[key], np.ndarray):
            raise TypeError(
                f"{name}['{key}'] must be np.ndarray, "
                f"got {type(res[key])}.")
        if res[key].ndim != 2:
            raise ValueError(
                f"{name}['{key}'] must be 2D, "
                f"got shape {res[key].shape}.")


def plot_fan_chart(series_idx, series_list, actuals,
                   models, save=True):
    '''
    Plot prediction interval fan chart for one series.

    Args:
        series_idx  (int):  index of series to plot
        series_list (list): list of np.ndarray time series
        actuals     (np.ndarray): ground truth test values
        models      (dict): model predictions
        save        (bool): whether to save HTML to disk

    Returns:
        plotly.graph_objects.Figure or empty Figure on failure

    Raises:
        TypeError: if series_list or actuals have wrong types
        IndexError: if series_idx is out of range
    '''
    if not isinstance(series_list, list) or \
       len(series_list) == 0:
        raise TypeError(
            "series_list must be a non-empty list.")
    if not isinstance(actuals, np.ndarray):
        raise TypeError(
            f"actuals must be np.ndarray, got {type(actuals)}.")
    if not isinstance(series_idx, int) or \
       series_idx < 0 or series_idx >= len(series_list):
        raise IndexError(
            f"series_idx {series_idx} out of range "
            f"for list of length {len(series_list)}.")
    if not isinstance(models, dict) or len(models) == 0:
        raise TypeError("models must be a non-empty dict.")

    try:
        history = series_list[series_idx][-24:]
        actual  = actuals[series_idx]
        h_steps = list(range(-len(history), 0))
        f_steps = list(range(FORECAST_HORIZON))
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract data for series "
            f"{series_idx}: {e}") from e

    fig = go.Figure()

    try:
        fig.add_trace(go.Scatter(
            x=h_steps, y=history.tolist(), mode="lines",
            line=dict(color="black", width=2), name="History"))
        fig.add_trace(go.Scatter(
            x=f_steps, y=actual.tolist(), mode="lines",
            line=dict(color="black", width=2, dash="dot"),
            name="Actual"))
    except Exception as e:
        raise RuntimeError(
            f"Failed to add base traces: {e}") from e

    for name, res in models.items():
        if res is None:
            continue
        try:
            _validate_model_result(name, res)
            fill_col, line_col = COLORS.get(
                name,
                ("rgba(128,128,128,0.15)", "rgb(128,128,128)"))
            upper  = res["upper"][series_idx]
            lower  = res["lower"][series_idx]
            median = res["median"][series_idx]
            if upper.shape != (FORECAST_HORIZON,) or \
               lower.shape != (FORECAST_HORIZON,):
                raise ValueError(
                    f"{name} forecast arrays have wrong length.")
            fig.add_trace(go.Scatter(
                x=f_steps, y=upper.tolist(),
                line=dict(width=0), showlegend=False,
                fillcolor=fill_col, name=name))
            fig.add_trace(go.Scatter(
                x=f_steps, y=lower.tolist(),
                line=dict(width=0), fill="tonexty",
                fillcolor=fill_col, showlegend=False,
                name=name))
            fig.add_trace(go.Scatter(
                x=f_steps, y=median.tolist(),
                line=dict(color=line_col, width=1.5),
                name=name))
        except (TypeError, KeyError, ValueError,
                IndexError) as e:
            print(f"Skipping {name} in fan chart: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error plotting {name}: {e}")
            continue

    fig.update_layout(
        title=f"Forecast fan chart — series {series_idx}",
        xaxis_title="Steps", yaxis_title="Value",
        template="plotly_white", height=450)

    if save:
        try:
            CHARTS_DIR.mkdir(exist_ok=True)
            pio.write_html(
                fig,
                CHARTS_DIR / f"fan_chart_{series_idx}.html")
        except (OSError, IOError) as e:
            print(f"Could not save fan chart: {e}")
        except Exception as e:
            print(f"Unexpected error saving fan chart: {e}")
    return fig


def plot_metrics(results_df, save=True):
    '''
    Plot side-by-side bar charts for RMSE, Coverage, Sharpness.

    Args:
        results_df (pd.DataFrame): output of evaluate_all_models()
        save       (bool): whether to save HTML to disk

    Returns:
        plotly.graph_objects.Figure

    Raises:
        TypeError: if results_df is not a DataFrame
        ValueError: if required metric columns are missing
    '''
    import pandas as pd
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError(
            f"results_df must be a DataFrame, "
            f"got {type(results_df)}.")
    required = {"RMSE", "Coverage", "Sharpness"}
    missing  = required - set(results_df.columns)
    if missing:
        raise ValueError(
            f"results_df missing columns: {missing}.")
    if len(results_df) == 0:
        raise ValueError(
            "results_df is empty — no models to plot.")

    metrics    = ["RMSE", "Coverage", "Sharpness"]
    bar_colors = [c[1] for c in COLORS.values()]
    fig        = make_subplots(rows=1, cols=3,
                               subplot_titles=metrics)

    for col_idx, metric in enumerate(metrics):
        try:
            values = results_df[metric].tolist()
            if all(v is None for v in values):
                print(f"All values None for {metric} — skipping.")
                continue
            fig.add_trace(
                go.Bar(x=results_df.index.tolist(),
                       y=values,
                       marker_color=bar_colors,
                       showlegend=False),
                row=1, col=col_idx + 1)
        except KeyError as e:
            print(f"Column {metric} not found: {e}")
            continue
        except Exception as e:
            print(f"Could not plot metric {metric}: {e}")
            continue

    fig.update_layout(
        title="Model comparison — all metrics",
        template="plotly_white", height=400)

    if save:
        try:
            CHARTS_DIR.mkdir(exist_ok=True)
            pio.write_html(
                fig,
                CHARTS_DIR / "metrics_comparison.html")
        except (OSError, IOError) as e:
            print(f"Could not save metrics chart: {e}")
        except Exception as e:
            print(f"Unexpected error saving metrics chart: {e}")
    return fig


def plot_calibration(models, actuals, save=True):
    '''
    Plot coverage calibration curves for all models.
    Models on or near the diagonal are well-calibrated.

    Args:
        models  (dict): model predictions
        actuals (np.ndarray): ground truth values
        save    (bool): whether to save HTML to disk

    Returns:
        plotly.graph_objects.Figure

    Raises:
        TypeError: if models is not a dict or actuals not ndarray
    '''
    if not isinstance(models, dict):
        raise TypeError(
            f"models must be a dict, got {type(models)}.")
    if not isinstance(actuals, np.ndarray):
        raise TypeError(
            f"actuals must be np.ndarray, got {type(actuals)}.")

    target_coverages = np.linspace(0.5, 1.0, 10)
    fig = go.Figure()

    try:
        fig.add_trace(go.Scatter(
            x=[0.5, 1.0], y=[0.5, 1.0], mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Perfect calibration"))
    except Exception as e:
        raise RuntimeError(
            f"Failed to add diagonal to calibration "
            f"plot: {e}") from e

    for name, res in models.items():
        if res is None:
            continue
        try:
            _validate_model_result(name, res)
            actual_coverages = []
            for target in target_coverages:
                try:
                    width  = res["upper"] - res["lower"]
                    half   = (width / 2) * (target / 0.95)
                    mid    = res["median"]
                    inside = ((actuals >= mid - half) &
                              (actuals <= mid + half))
                    actual_coverages.append(
                        float(inside.mean()))
                except Exception as e:
                    print(f"Coverage calc error for {name} "
                          f"at target {target:.2f}: {e}")
                    actual_coverages.append(float("nan"))
                    continue

            fig.add_trace(go.Scatter(
                x=target_coverages.tolist(),
                y=actual_coverages,
                mode="lines+markers",
                line=dict(color=COLORS.get(
                    name,
                    (None, "rgb(128,128,128)"))[1]),
                name=name))
        except (TypeError, KeyError, ValueError) as e:
            print(f"Skipping calibration for {name}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error in calibration "
                  f"for {name}: {e}")
            continue

    fig.update_layout(
        title="Coverage calibration plot",
        xaxis_title="Expected coverage",
        yaxis_title="Actual coverage",
        template="plotly_white", height=450)

    if save:
        try:
            CHARTS_DIR.mkdir(exist_ok=True)
            pio.write_html(fig, CHARTS_DIR / "calibration.html")
        except (OSError, IOError) as e:
            print(f"Could not save calibration chart: {e}")
        except Exception as e:
            print(
                f"Unexpected error saving calibration chart: {e}")
    return fig


def save_all_charts(series_list, actuals, models, results_df):
    '''
    Generate and save all chart types to the charts/ directory.

    Args:
        series_list (list): list of np.ndarray time series
        actuals     (np.ndarray): ground truth values
        models      (dict): model predictions
        results_df  (pd.DataFrame): metrics table
    '''
    if not isinstance(series_list, list) or \
       len(series_list) == 0:
        raise TypeError(
            "series_list must be a non-empty list.")
    if not isinstance(actuals, np.ndarray):
        raise TypeError(
            f"actuals must be np.ndarray, got {type(actuals)}.")

    for idx in range(3):
        try:
            plot_fan_chart(
                idx, series_list, actuals, models, save=True)
        except (IndexError, TypeError, RuntimeError) as e:
            print(f"Fan chart {idx} failed: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error in fan chart {idx}: {e}")
            continue

    try:
        plot_metrics(results_df, save=True)
    except (TypeError, ValueError) as e:
        print(f"Metrics chart failed: {e}")
    except Exception as e:
        print(f"Unexpected error in metrics chart: {e}")

    try:
        plot_calibration(models, actuals, save=True)
    except (TypeError, ValueError) as e:
        print(f"Calibration chart failed: {e}")
    except Exception as e:
        print(f"Unexpected error in calibration chart: {e}")

    print("All charts saved to charts")