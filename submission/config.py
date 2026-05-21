'''
Configuration constants shared across all modules.
Centralizes hyperparameters and paths so changing one
value updates the entire pipeline.
'''

from pathlib import Path

BASE_DIR         = Path(__file__).parent.parent
DATA_DIR         = BASE_DIR / "data"
CHARTS_DIR       = BASE_DIR / "charts"
FORECAST_HORIZON = 18
N_SERIES         = 1000
N_LAGS           = 24
SEQ_LEN          = 36
BATCH_SIZE       = 64
EPOCHS           = 80
CALIB_SIZE       = 100