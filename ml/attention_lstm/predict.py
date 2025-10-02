import numpy as np

from tensorflow import keras
from .model import Attention
from .data_processor import DataProcessor
from .constant import LOOK_BACK, MODEL_DIR, KERAS_FILE_TEMPLATE

def load_model(target: str):
    model_path = MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE}"
    return keras.models.load_model(
        model_path, compile=False, custom_objects={"Attention": Attention}
    )

def predict_next_day():
    data_processor = DataProcessor()
    targets = data_processor.targets
    pred_values = {}

    for target in targets:
        # 데이터 전처리
        data = data_processor.get_proceed_data(target=target)
        try:
            model = load_model(target)
        except Exception as e:
            print(f"Error loading model for {target}: {e}")
            continue

        # feature 컬럼만 추출
        X = data.drop(columns=[target])
        X_last = X.iloc[-LOOK_BACK:]
        
        # 단일 특징 스케일러 로드
        feature_scaler = data_processor.get_feature_scaler(target=target)
        
        if feature_scaler is None:
            print(f"Error: 피처 스케일러 찾지 못함 {target}")
            continue

        X_input_scaled_2d = feature_scaler.transform(X_last)
        X_input_scaled = X_input_scaled_2d.reshape(1, LOOK_BACK, -1)
        
        # 예측 수행
        y_pred_scaled = model.predict(X_input_scaled, verbose=0)

        # 역변환
        target_scaler = data_processor.get_target_scaler(target=target)
        y_pred = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()[0]

        pred_values[target] = y_pred
    return pred_values

if __name__ == "__main__":
    print(predict_next_day())
