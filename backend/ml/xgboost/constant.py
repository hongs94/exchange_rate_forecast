from pathlib import Path


LOOK_BACK = 60
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
SCALER_DIR = BASE_DIR / "scalers"
PRED_TRUE_DIR = BASE_DIR / "pred_true"
PRED_TRUE_CSV = "pred_true.csv"
MODEL_FILE_TEMPLATE = "_xgboost.pkl"
CURRENCIES = ["usd", "cny", "jpy", "eur"]
RESULTS_PATH = BASE_DIR / "train_results.json"
