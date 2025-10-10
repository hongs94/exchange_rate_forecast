import os
import pandas as pd
from tensorflow import keras
from .data_processor import DataProcessor
from .constant import LOOK_BACK, MODEL_DIR, KERAS_FILE_TEMPLATE, PRED_TRUE_DIR, ACCUMULATED_PRED_CSV

def load_model(target: str):
    model_path = MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE}"
    return keras.models.load_model(
        model_path, compile=False
    )

def predict_next_day():
    data_processor = DataProcessor()
    targets = data_processor.targets
    pred_values = {}
    
    # 1. 데이터셋의 가장 마지막 날짜를 가져옴
    last_date = data_processor.origin_data.index.max()
    
    # 2. 예측 날짜를 마지막 날짜의 다음 날로 설정
    pred_date = last_date + pd.Timedelta(days=1)
    
    prediction_df = pd.DataFrame(index=[pred_date])

    for target in targets:
        data = data_processor.get_proceed_data(target=target)

        data.ffill(inplace=True)
        last_index = data.index[-1]
        
        data.dropna(inplace=True)
        model = load_model(target)

        X = data.drop(columns=[target])
        X_last = X.iloc[-LOOK_BACK:]

        feature_scaler = data_processor.get_feature_scaler(target=target)
        X_input_scaled = feature_scaler.transform(X_last).reshape(1, LOOK_BACK, -1)

        y_pred_scaled = model.predict(X_input_scaled)

        target_scaler = data_processor.get_target_scaler(target=target)
        y_pred = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()[0]

        pred_values[target] = y_pred
        
        prediction_df[target] = y_pred
    
    os.makedirs(PRED_TRUE_DIR, exist_ok=True)
    accumulated_path = PRED_TRUE_DIR / ACCUMULATED_PRED_CSV
    
    if accumulated_path.exists():
        accumulated_df = pd.read_csv(accumulated_path, index_col=0, parse_dates=True)
        prediction_df.index.name = accumulated_df.index.name
        combined_df = pd.concat([accumulated_df, prediction_df]).drop_duplicates(keep='last')
        combined_df.to_csv(accumulated_path)
    else:
        prediction_df.index.name = 'date'
        prediction_df.to_csv(accumulated_path)
    print(f"예측 결과가 {ACCUMULATED_PRED_CSV}에 누적 저장됨. 예측 날짜: {pred_date.strftime('%Y-%m-%d')}")
    return pred_values

if __name__ == "__main__":
    print(predict_next_day())