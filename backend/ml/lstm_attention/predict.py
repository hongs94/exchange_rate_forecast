import torch
import json
from .data_processor import DataProcessor
from .constant import LOOK_BACK, MODEL_DIR, TRAIN_RESULT, MODEL_FILE_TEMPLATE
from .model import AttentionLSTM


def load_model(target: str, num_features: int, params: dict):
    model_path = MODEL_DIR / f"{target}{MODEL_FILE_TEMPLATE}"

    model = AttentionLSTM(
        num_features=num_features,
        lstm_units=params["lstm_units"],
        dropout_rate=params["dropout_rate"],
    )
    # Load model with CPU mapping for compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def predict_next_day():
    data_processor = DataProcessor()

    with open(TRAIN_RESULT, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    targets = data_processor.targets
    pred_values = {}

    for target in targets:
        data = data_processor.get_proceed_data(target=target)
        data.ffill(inplace=True)
        data.dropna(inplace=True)

        X = data.drop(columns=[target])
        num_features = X.shape[1]

        params = all_results[target]["best_params"]

        model = load_model(target, num_features, params)

        X_last = X.iloc[-LOOK_BACK:]

        feature_scaler = data_processor.get_feature_scaler(target=target)
        X_input_scaled = feature_scaler.transform(X_last)

        X_tensor = torch.FloatTensor(X_input_scaled).unsqueeze(0)

        with torch.no_grad():
            y_pred_scaled = model(X_tensor)

        y_pred_scaled_np = y_pred_scaled.cpu().numpy()

        target_scaler = data_processor.get_target_scaler(target)
        y_pred = target_scaler.inverse_transform(y_pred_scaled_np).flatten()[0]
        pred_values[target] = y_pred

    return pred_values
