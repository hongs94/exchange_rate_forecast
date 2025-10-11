import os
import json 
import numpy as np
import pandas as pd

from pathlib import Path
from .model import AttentionLayer
from keras.models import load_model
from .data_processor import DataProcessor
from ..constant import (LOOK_BACK, MODEL_DIR, KERAS_FILE_TEMPLATE_AL, PRED_TRUE_DIR, ACCUMULATED_PRED_CSV_TEMPLATE) 

MODEL_NAME = "attention_lstm"

def load_model_keras(target: str):
    model_path = MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE_AL}"
    model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    return model

def predict_next_day():
    data_processor = DataProcessor()
    targets = data_processor.targets
    pred_values = {}

    last_date = data_processor.origin_data.index.max()

    pred_date = last_date + pd.Timedelta(days=1)
    prediction_df = pd.DataFrame(index=[pred_date])
    prediction_df.index.name = 'date' 

    print(f"[{MODEL_NAME.upper()}] 예측 날짜: {pred_date.strftime('%Y-%m-%d')}")

    for target in targets:
        data = data_processor.get_proceed_data(target=target)

        # 마지막 데이터가 NaN일 경우 이전 값으로 채움 (예측에 필요한 데이터는 존재해야 함)
        data.ffill(inplace=True)
        data.dropna(inplace=True)

        # 모델 로드
        try:
            model = load_model_keras(target)
        except Exception as e:
            print(f"⚠️ {target} 모델 로드 실패: {e}")
            continue

        # 예측에 사용할 피처 선택 및 시퀀스 추출
        X = data.drop(columns=[target])
        # LOOK_BACK 만큼의 마지막 시퀀스를 가져옴
        X_last = X.iloc[-LOOK_BACK:]

        # 스케일러 로드 및 데이터 스케일링
        feature_scaler = data_processor.get_feature_scaler(target=target)
        # 3D 형태 (1, LOOK_BACK, num_features)로 변환
        X_input_scaled = feature_scaler.transform(X_last).reshape(1, LOOK_BACK, -1)

        # 예측 수행 (스케일된 값)
        y_pred_scaled_np = model.predict(X_input_scaled, verbose=0)

        # 역변환하여 실제 값으로 변환
        target_scaler = data_processor.get_target_scaler(target)
        y_pred = target_scaler.inverse_transform(y_pred_scaled_np.reshape(-1, 1)).flatten()[0]

        pred_values[target] = y_pred
        prediction_df[target] = y_pred
        print(f"  - {target.upper()} 예측값: {y_pred:.4f}")

    # 예측값 누적 파일 저장
    os.makedirs(PRED_TRUE_DIR, exist_ok=True)
    accumulated_path = PRED_TRUE_DIR / ACCUMULATED_PRED_CSV_TEMPLATE.format(model_name=MODEL_NAME)

    if accumulated_path.exists():
        # index_col=0, parse_dates=True를 사용하여 인덱스를 날짜로 파싱
        accumulated_df = pd.read_csv(accumulated_path, index_col=0, parse_dates=True)
        
        # 새로운 예측값을 기존 데이터프레임에 추가 (인덱스가 중복되지 않는 경우에만)
        if pred_date not in accumulated_df.index:
            combined_df = pd.concat([accumulated_df, prediction_df])
            # 중복 날짜는 제거하고 날짜 순으로 정렬
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
            combined_df.to_csv(accumulated_path)
            print(f"✅ {accumulated_path.name}에 새로운 예측값 추가 및 저장.")
        else:
            print(f"⚠️ {pred_date.strftime('%Y-%m-%d')} 날짜의 예측값은 이미 {accumulated_path.name}에 존재합니다. 저장하지 않습니다.")
    else:
        # 파일이 없으면 새로 생성
        prediction_df.to_csv(accumulated_path)
        print(f"✅ {accumulated_path.name} 파일 새로 생성 및 저장.")

    # 예측값 딕셔너리 반환
    return pred_values

if __name__ == "__main__":
    predict_next_day()