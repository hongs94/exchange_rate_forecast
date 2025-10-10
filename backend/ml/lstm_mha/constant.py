from pathlib import Path

LOOK_BACK = 60
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
SCALER_DIR = BASE_DIR / "scalers"
PRED_TRUE_DIR = BASE_DIR / "pred_true"
PRED_TRUE_CSV = "pred_true.csv"
ACCUMULATED_PRED_CSV = "accumulated_pred.csv"
KERAS_FILE_TEMPLATE = "_lstm.keras"
