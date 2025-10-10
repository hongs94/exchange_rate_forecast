import os
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from .data_processor import DataProcessor
from .constant import LOOK_BACK, MODEL_DIR, MODEL_FILE_TEMPLATE, PRED_TRUE_DIR, ACCUMULATED_PRED_CSV

def load_model(target: str):
    model_path = MODEL_DIR / f"{target}{MODEL_FILE_TEMPLATE}"
    return joblib.load(model_path)

def predict_next_day():
    data_processor = DataProcessor()
    targets = data_processor.targets
    pred_values = {}
    
    # 예측 날짜를 마지막 학습 데이터의 다음 날로 설정
    last_date = data_processor.origin_data.index.max()
    pred_date = last_date + pd.Timedelta(days=1)
    prediction_df = pd.DataFrame(index=[pred_date])

    for target in targets:
        data = data_processor.get_proceed_data(target=target)

        data.ffill(inplace=True)
        data.dropna(inplace=True)
        model = load_model(target)

        X = data.drop(columns=[target])
        X_last = X.iloc[-LOOK_BACK:] # LOOK_BACK 기간의 피처 데이터

        feature_scaler = data_processor.get_feature_scaler(target=target)
        
        # XGBoost를 위해 X_last를 1D 벡터로 변환 후 2D로 reshape
        X_input_scaled = feature_scaler.transform(X_last).flatten().reshape(1, -1)
        
        y_pred_scaled = model.predict(X_input_scaled)

        target_scaler = data_processor.get_target_scaler(target=target)
        
        # 예측값 역변환
        y_pred = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1) # XGBoost 예측값은 1D array이므로 reshape(-1, 1) 필요
        ).flatten()[0]

        pred_values[target] = y_pred
        prediction_df[target] = y_pred
    
    os.makedirs(PRED_TRUE_DIR, exist_ok=True)
    accumulated_path = PRED_TRUE_DIR / ACCUMULATED_PRED_CSV
    
    if accumulated_path.exists():
        accumulated_df = pd.read_csv(accumulated_path, index_col=0, parse_dates=True)
        
        # 예측 날짜 인덱스 이름 설정
        prediction_df.index.name = accumulated_df.index.name
        
        # 기존 accumulated_df와 새로운 예측값을 합치고 중복 제거 (keep='last'로 최신 예측값 유지)
        combined_df = pd.concat([accumulated_df, prediction_df]).drop_duplicates(keep='last')
        combined_df.to_csv(accumulated_path)
    else:
        # 파일이 없을 경우
        prediction_df.index.name = 'date'
        prediction_df.to_csv(accumulated_path)
        
    print(f"예측 결과가 {ACCUMULATED_PRED_CSV}에 누적 저장됨. 예측 날짜: {pred_date.strftime('%Y-%m-%d')}")
    return pred_values

if __name__ == "__main__":
    print(predict_next_day())