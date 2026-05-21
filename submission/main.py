'''
Multi-Model Time-Series Forecasting Benchmark
==============================================
Benchmarks five forecasting models on the M4 monthly dataset:
    Prophet, LightGBM, ARIMA, LSTM, Ensemble

Evaluates using:
    RMSE       -- point forecast accuracy
    Coverage   -- calibrated prediction interval reliability
    Sharpness  -- prediction interval precision

Usage:
    python main.py

Run notebooks 01-04 first to generate all model result files.
'''

from data_loader import (load_series, load_actuals,
                          load_model_results)
from evaluate import evaluate_all_models
from visualize import save_all_charts
from config import DATA_DIR


def main():
    '''
    Entry point for the forecasting benchmark pipeline.
    Loads data, evaluates models, prints results, saves charts.
    '''
    print("Loading data and model results...")

    try:
        series_list = load_series()
    except FileNotFoundError as e:
        print(f"Data missing: {e}")
        return
    except (ValueError, RuntimeError) as e:
        print(f"Data load error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error loading series: {e}")
        return

    try:
        actuals = load_actuals()
    except FileNotFoundError as e:
        print(f"Actuals missing: {e}")
        return
    except (ValueError, RuntimeError) as e:
        print(f"Actuals load error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error loading actuals: {e}")
        return

    try:
        models = load_model_results()
    except RuntimeError as e:
        print(f"Model results error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error loading models: {e}")
        return

    print(f"Loaded {len(models)} models: "
          f"{list(models.keys())}")

    print("\nEvaluating models...")
    try:
        results_df = evaluate_all_models(models, actuals)
    except (TypeError, ValueError, RuntimeError) as e:
        print(f"Evaluation failed: {e}")
        return
    except Exception as e:
        print(f"Unexpected evaluation error: {e}")
        return

    print("\n=== Benchmark Results ===")
    print(results_df.to_string())

    try:
        results_df.to_csv(DATA_DIR / "metrics_final.csv")
        print(f"\nMetrics saved to "
              f"{DATA_DIR / 'metrics_final.csv'}")
    except (OSError, IOError) as e:
        print(f"Could not save metrics CSV: {e}")
    except Exception as e:
        print(f"Unexpected error saving metrics: {e}")

    print("\nGenerating dashboard charts...")
    try:
        save_all_charts(series_list, actuals, models, results_df)
    except (TypeError, ValueError) as e:
        print(f"Chart generation failed: {e}")
    except Exception as e:
        print(f"Unexpected error in chart generation: {e}")

    print("\nDone. Open charts to view the dashboard.")


if __name__ == "__main__":
    main()