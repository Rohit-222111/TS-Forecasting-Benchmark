\# Multi-Model Time-Series Forecasting Benchmark



Benchmarks five forecasting models on the M4 monthly competition

dataset, evaluating both point forecast accuracy and calibrated

prediction interval quality across 1000 time series.



\## Author

Rohit — https://github.com/Rohit-222111



\## Project Link

https://github.com/Rohit-222111/ts-forecasting-benchmark



\---



\## Models Benchmarked

| Model | Type | Description |

|---|---|---|

| Prophet | Bayesian curve-fitting | Facebook's forecasting tool, decomposes trend and seasonality |

| LightGBM | Gradient boosting | Fast ML model using lag features and quantile regression |

| ARIMA | Classical statistical | Auto-Regressive Integrated Moving Average with seasonal terms |

| LSTM | Deep learning (GPU) | Long Short-Term Memory neural network trained on RTX 5060 |

| Ensemble | Model combination | Equal-weighted average of all four models |



\---



\## Evaluation Metrics

\- \*\*RMSE\*\* — Root Mean Squared Error: measures point forecast accuracy. Lower is better.

\- \*\*Coverage\*\* — fraction of actual values that fall inside the 95% prediction interval. Target is ≥ 0.95. Higher is better.

\- \*\*Sharpness\*\* — mean width of prediction intervals. Lower is better when coverage is already satisfied.



\---



\## Results (1000 M4 monthly series, 18-step forecast horizon)



| Model | RMSE | Coverage | Sharpness |

|---|---|---|---|

| Prophet | 1434.59 | 0.5916 | 1393.17 |

| LightGBM | 1095.98 | 0.5992 | 1614.77 |

| ARIMA | 948.59 | 0.8969 | 1793.93 |

| LSTM | 1138.40 | 0.9062 | 2420.92 |

| Ensemble | 959.87 | 0.8181 | 1408.66 |



\### Key Findings

\- \*\*ARIMA\*\* achieves the lowest RMSE (948) — best point forecast accuracy

\- \*\*LSTM\*\* achieves the highest coverage (0.906) — most reliable prediction intervals after conformal calibration

\- \*\*Ensemble\*\* is a close second on RMSE (959) — combining models reduces individual errors

\- \*\*Prophet and LightGBM\*\* underperform on coverage (\~0.60) — their intervals are too narrow and overconfident

\- No single model dominates all three metrics — which is the honest finding of a real benchmark



\---



\## Dashboard Charts



\### Forecast Fan Chart

Each coloured line is one model's 18-month forecast. Shaded bands

are prediction intervals. Dotted black line is the actual value.



!\[Fan Chart](screenshots/fan\_chart.png)



\### Model Comparison — All Metrics

Side-by-side bar charts comparing RMSE, Coverage and Sharpness

across all five models.



!\[Metrics Comparison](screenshots/metrics.png)



\### Coverage Calibration Plot

Models closest to the diagonal are best calibrated — their claimed

confidence matches their actual reliability. ARIMA and LSTM track

the diagonal most closely.



!\[Calibration Plot](screenshots/calibration.png)



\---



\## Dataset

This project uses the \*\*M4 Competition Monthly dataset\*\*.



\- Original source and full dataset:

&#x20; https://github.com/Mcompetitions/M4-methods/tree/master/Dataset

\- Train data direct link:

&#x20; https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv

\- Test data direct link:

&#x20; https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv

\- This repository does not include raw CSV files due to size.

&#x20; Running `01\_Data\_Preprocessing.ipynb` downloads them automatically.

\- The processed subset covers the first 1000 monthly series

&#x20; stored as `data/series\_list.pkl` and `data/test\_sub.pkl`.



\---



\## Setup



```bash

conda create -n tsforecast python=3.11 -y

conda activate tsforecast

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install prophet lightgbm pmdarima scikit-learn

pip install pandas numpy matplotlib plotly joblib

```



\---



\## Usage



Run notebooks in order:

```

01\_Data\_Preprocessing.ipynb   — download and preprocess M4 data

02\_prophet\_lgbm.ipynb         — train Prophet and LightGBM

03\_arima\_lstm.ipynb           — train ARIMA and LSTM (GPU)

04\_ensemble\_eval.ipynb        — build ensemble and compute metrics

05\_dashboard.ipynb            — generate interactive charts

```



Then run the submission pipeline:

```bash

cd submission

python main.py

```



\---



\## Project Structure

```

TS\_Forecast\_Project/

├── data/                        — processed pkl files and metrics CSV

├── charts/                      — interactive HTML dashboard charts

│   ├── fan\_chart\_0.html         — open in browser to view

│   ├── fan\_chart\_1.html

│   ├── fan\_chart\_2.html

│   ├── metrics\_comparison.html

│   └── calibration.html

├── screenshots/                 — static images for this README

├── submission/

│   ├── main.py                  — entry point (run this)

│   ├── config.py                — shared constants and paths

│   ├── data\_loader.py           — all file I/O with error handling

│   ├── evaluate.py              — RMSE, Coverage, Sharpness metrics

│   └── visualize.py            — Plotly chart generation

├── 01\_Data\_Preprocessing.ipynb

├── 02\_prophet\_lgbm.ipynb

├── 03\_arima\_lstm.ipynb

├── 04\_ensemble\_eval.ipynb

└── 05\_dashboard.ipynb

```



\---



\## Advanced Topics Demonstrated



\### File I/O

Pickle serialization and CSV export used throughout every notebook

and all submission modules. Every file operation uses `with open()`

for safe automatic closing.



\### Try/Except Error Handling

Comprehensive error handling implemented across all files:

\- `FileNotFoundError` — catches missing pkl or CSV files with

&#x20; clear messages directing the user to run the correct notebook

\- `ValueError` — catches shape mismatches, NaN/Inf in predictions,

&#x20; empty datasets, and invalid array dimensions

\- `TypeError` — catches wrong input types passed to metric functions

\- `KeyError` — catches missing dictionary keys in model result dicts

\- `RuntimeError` — catches corrupted pickle files and unexpected

&#x20; computation failures

\- `OSError` / `IOError` — catches disk write failures when saving

&#x20; results and charts

\- `pickle.UnpicklingError` / `EOFError` — catches corrupted or

&#x20; truncated pickle files

\- Per-series fallback forecasting in ARIMA, Prophet, LightGBM and

&#x20; LSTM loops — individual series failures never crash the pipeline



\### Break and Continue

\- `continue` used in all model training loops to skip failed or

&#x20; too-short series without stopping the pipeline

\- `break` used in ARIMA loop to stop early if 20 consecutive

&#x20; failures occur, preventing infinite bad-data loops

\- `break` used in evaluate\_all\_models to stop processing when

&#x20; no remaining models have valid data



\---



\## Notes for Tutor



The interactive HTML charts in `charts/` must be opened locally

in a browser — they are fully interactive Plotly charts where

you can hover over values, zoom, and click legend items to

isolate individual models.



To view them, open File Explorer, navigate to the `charts/`

folder and double-click any `.html` file.



To run the full pipeline end to end:

```bash

conda activate tsforecast

cd submission

python main.py

```



Expected output:

```

Loading data and model results...

Loaded 5 models: \['Prophet', 'LightGBM', 'ARIMA', 'LSTM', 'Ensemble']



Evaluating models...



=== Benchmark Results ===

&#x20;          RMSE  Coverage  Sharpness

Model

Prophet   1434.59    0.5916    1393.17

LightGBM  1095.98    0.5992    1614.77

ARIMA      948.59    0.8969    1793.93

LSTM      1138.40    0.9062    2420.92

Ensemble   959.87    0.8181    1408.66



Metrics saved to data/metrics\_final.csv

All charts saved to charts/

Done. Open charts/ to view the dashboard.

```

