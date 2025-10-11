import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import product
from .model import build_model
from .data_processor import DataProcessor
from .predict import predict_next_day
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from ..constant import (BASE_DIR, LOOK_BACK, MODEL_DIR, PRED_TRUE_DIR, KERAS_FILE_TEMPLATE_L)

MODEL_NAME = "lstm"

# 성능지표 계산
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }

# 롤링 윈도우 방식의 인덱스 생성
def rolling_split_index(total_len, train_size=1200, test_size=300):
    for start in range(0, total_len - train_size, test_size):
        end = min(start + train_size + test_size, total_len)
        yield (
            np.arange(start, start + train_size),
            np.arange(start + train_size, end),
        )

# 최적의 하이퍼파라미터 탐색
def find_best_hyperparams(X_train, y_train, X_val, y_val, num_features):
    hyperparams_candidates = {
        "lstm_units": [100, 150],
        "dropout_rate": [0.2, 0.3],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [32, 64]
    }

    best_val_loss = float("inf")
    best_params = None

    for lstm_units, dropout_rate, learning_rate, batch_size in product(
        *hyperparams_candidates.values()
    ):
        model = build_model(
            LOOK_BACK, num_features, lstm_units, dropout_rate, learning_rate
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=batch_size,
            callbacks=[early_stopping],
            shuffle=False,
            verbose=0,
        )

        val_loss = min(history.history["val_loss"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (lstm_units, dropout_rate, learning_rate, batch_size)

    return best_params

def train():
    data_processor = DataProcessor()
    targets = data_processor.targets
    results = {}

    results_file = BASE_DIR / "train_results.json"
    existing_results = {}
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)

    for target in targets:
        model_path = MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE_L}"
        
        # 모델 파일 존재하면 학습 건너뛰기
        if model_path.exists():
            print(f"✅ {target.upper()} 모델 파일이 이미 존재해 학습 생략.")
            if target in existing_results:
                results[target] = existing_results[target]
            continue

        # 시퀀스 데이터 로드
        X_seq, y_seq, y_idxs = data_processor.get_sequence_data(target=target)

        y_preds_all, y_true_all, y_idxs_all = [], [], []
        best_params = None

        total_len = len(X_seq)
        print(f"✅ {target.upper()} 모델 학습 (총 {total_len}개 시퀀스)")

        # 롤링 윈도우 방식으로 학습 및 예측 수행
        for i, (train_idx, test_idx) in enumerate(rolling_split_index(total_len)):
            X_train, y_train = X_seq[train_idx], y_seq[train_idx]
            X_test, y_test = X_seq[test_idx], y_seq[test_idx]

            # 첫 번째 윈도우에서만 최적 하이퍼파라미터 탐색
            if i == 0:
                print("최적 하이퍼파라미터 탐색 중")
                best_params = find_best_hyperparams(
                    X_train, y_train, X_test, y_test, X_seq.shape[2]
                )
                print(
                    f"최적 하이퍼파라미터: LSTM Units={best_params[0]}, Dropout={best_params[1]}, LR={best_params[2]}"
                )

            print(f"롤링 윈도우 {i + 1} 학습 및 예측 수행")
            model = build_model(LOOK_BACK, X_seq.shape[2], *best_params)
            
            # 롤링 윈도우 학습
            model.fit(X_train, y_train, epochs=50, batch_size=32, shuffle=False, verbose=0)

            # 예측
            y_pred = model.predict(X_test, verbose=0)

            y_preds_all.append(y_pred)
            y_true_all.append(y_test)
            y_idxs_all.append(y_idxs[test_idx])

        # 모든 롤링 윈도우의 결과 취합
        y_pred_concat = np.concatenate(y_preds_all)
        y_true_concat = np.concatenate(y_true_all)
        y_idxs_concat = np.concatenate(y_idxs_all)

        # 역변환 및 성능 지표 계산
        target_scaler = data_processor.get_target_scaler(target)
        y_true_inv = target_scaler.inverse_transform(y_true_concat)
        y_pred_inv = target_scaler.inverse_transform(y_pred_concat)
        metrics = evaluate_predictions(y_true_inv, y_pred_inv)

        # CSV 저장 로직
        pred_df = pd.DataFrame(
            {"true": y_true_inv.flatten(), "pred": y_pred_inv.flatten()},
            index=y_idxs_concat,
        )
        os.makedirs(PRED_TRUE_DIR, exist_ok=True)
        pred_df.to_csv(PRED_TRUE_DIR / f"{target}_{MODEL_NAME}_pred_true.csv")
        print(f"저장: {target}_{MODEL_NAME}_pred_true.csv")

        # 전체 데이터로 최종 모델 학습 및 저장
        print("전체 데이터로 최종 모델 학습 및 저장")
        final_model = build_model(LOOK_BACK, X_seq.shape[2], *best_params)
        final_model.fit(
            X_seq,
            y_seq,
            epochs=50,
            batch_size=32,
            shuffle=False,
            verbose=0,
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        final_model.save(MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE_L}")

        # 학습 결과 저장
        best_params_dict = {
            "lstm_units": best_params[0],
            "dropout_rate": best_params[1],
            "learning_rate": best_params[2],
        }
        results[target] = {"best_params": best_params_dict, "metrics": metrics}

        # # 디버깅 메세지
        # print(f"데이터 시퀀스 최대 날짜: {y_idxs.max()}")
        # print(f"롤링 윈도우 마지막 인덱스: {train_idx[-1]}, {test_idx[-1]}")
        # print(f"최종 저장된 데이터 최대 날짜: {y_idxs_concat.max()}")

    # 학습 완료 후 1일 예측값 생성 및 누적 저장
    print("1일 후 예측값 생성 중")
    predict_next_day()

    # 최종 결과 저장
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("모든 학습/결과 처리 완료.")

if __name__ == "__main__":
    train()