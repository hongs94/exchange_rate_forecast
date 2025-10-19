import joblib
from .data_processor import DataProcessor
from .constant import LOOK_BACK, MODEL_DIR, MODEL_FILE_TEMPLATE


def load_model(target: str):
    model_path = MODEL_DIR / f"{target}{MODEL_FILE_TEMPLATE}"
    return joblib.load(model_path)


def predict_next_day():
    data_processor = DataProcessor()
    targets = data_processor.targets
    pred_values = {}

    for target in targets:
        data = data_processor.get_proceed_data(target=target)

        model = load_model(target)

        X = data.drop(columns=[target])
        X_last = X.iloc[-LOOK_BACK:]

        feature_scaler = data_processor.get_feature_scaler(target=target)
        X_input_scaled = feature_scaler.transform(X_last).reshape(1, -1)

        y_pred_scaled = model.predict(X_input_scaled)

        target_scaler = data_processor.get_target_scaler(target=target)
        y_pred = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()[0]

        pred_values[target] = y_pred

    return pred_values


if __name__ == "__main__":
    print(predict_next_day())
