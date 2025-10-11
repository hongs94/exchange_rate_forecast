import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from itertools import product
from .model import build_model
from .predict import predict_next_day
from .data_processor import DataProcessor
from .predict import predict_next_day
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from .constant import (BASE_DIR, LOOK_BACK, MODEL_DIR, PRED_TRUE_DIR, MODEL_FILE_TEMPLATE)

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
def find_best_hyperparams(X_train, y_train, X_val, y_val):
    hyperparams_candidates = {
        "n_estimators": [100, 200],
        "max_depth": [6, 8],
        "learning_rate": [0.1, 0.05],
    }

    best_val_loss = float("inf")
    best_params = None

    for n_estimators, max_depth, learning_rate in product(
        *hyperparams_candidates.values()
    ):
        model = build_model(n_estimators, max_depth, learning_rate)

        eval_set = [(X_val, y_val.flatten())]
        model.fit(
            X_train,
            y_train.flatten(),
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=10,
        )

        # 예측 및 MSE 계산
        y_val_pred = model.predict(X_val)
        val_loss = mean_squared_error(y_val.flatten(), y_val_pred)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (n_estimators, max_depth, learning_rate)

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
        model_path = MODEL_DIR / f"{target}{MODEL_FILE_TEMPLATE}"
        
        if model_path.exists():
            print(f"✅ {target.upper()} 모델 파일이 이미 존재해 학습 생략.")
            if target in existing_results:
                results[target] = existing_results[target]
            continue

        # X_seq는 (Samples, LOOK_BACK * Features) 모양의 2D 데이터
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
                # X_seq.shape[2] 인자 제거
                best_params = find_best_hyperparams(X_train, y_train, X_test, y_test)
                print(
                    f"최적 하이퍼파라미터: n_estimators={best_params[0]}, max_depth={best_params[1]}, LR={best_params[2]}"
                )

            print(f"롤링 윈도우 {i + 1} 학습 및 예측 수행")
            model = build_model(*best_params)
            
            # 롤링 윈도우 학습 및 예측
            model.fit(X_train, y_train.flatten(), verbose=False)
            y_pred = model.predict(X_test)

            y_preds_all.append(y_pred.reshape(-1, 1))
            y_true_all.append(y_test)
            y_idxs_all.append(y_idxs[test_idx])

        # 모든 롤링 윈도우의 결과 취합
        y_pred_concat = np.concatenate(y_preds_all)
        y_true_concat = np.concatenate(y_true_all)
        y_idxs_concat = np.concatenate(y_idxs_all)

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
        pred_df.to_csv(PRED_TRUE_DIR / f"{target}_pred_true.csv")
        print(f"저장: {target}_pred_true.csv")

        # 전체 데이터로 최종 모델 학습 및 저장
        print("전체 데이터로 최종 모델 학습 및 저장")
        final_model = build_model(*best_params)
        final_model.fit(X_seq, y_seq.flatten(), verbose=False)

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(final_model, MODEL_DIR / f"{target}{MODEL_FILE_TEMPLATE}")

        # 학습 결과 저장
        best_params_dict = {
            "n_estimators": best_params[0],
            "max_depth": best_params[1],
            "learning_rate": best_params[2],
        }
        results[target] = {"best_params": best_params_dict, "metrics": metrics}

        # # 디버깅 메세지
        # print(f"데이터 시퀀스 최대 날짜: {y_idxs.max()}")
        # print(f"롤링 윈도우 마지막 인덱스: {train_idx[-1]}, {test_idx[-1]}")
        # print(f"최종 저장된 데이터 최대 날짜: {y_idxs_concat.max()}")

    # 학습 완료 후 1일 예측값 생성 및 누적 저장
    print("1일 후 예측값 생성")
    predict_next_day()

    # 최종 결과 저장
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("모든 학습/결과 처리 완료.")

if __name__ == "__main__":
    train()