from pathlib import Path

LOOK_BACK = 60
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
SCALER_DIR = BASE_DIR / "scalers"
PRED_TRUE_DIR = BASE_DIR / "pred_true"

RESULTS_FILE_TEMPLATE = "train_results_{model_name}.json" 
PRED_TRUE_CSV_TEMPLATE = "{target}_{model_name}_pred_true.csv"
ACCUMULATED_PRED_CSV_TEMPLATE = "accumulated_pred_{model_name}.csv"

KERAS_FILE_TEMPLATE_AL = "_attention_lstm.keras"
KERAS_FILE_TEMPLATE_L = "_lstm.keras"
MODEL_FILE_TEMPLATE = "_xgboost.pkl"