import json
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import tensorflow as tf

from .model import build_model
from .constant import (
    RESULTS_PATH,
    KERAS_FILE_TEMPLATE,
    LOOK_BACK,
    MODEL_DIR,
    PRED_TRUE_DIR,
    PRED_TRUE_CSV,
)
from .data_processor import DataProcessor


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """역변환된 실제값과 예측값으로 성능 지표를 계산합니다."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def rolling_split_index(total_len, train_size=1200, test_size=300):
    """롤링 윈도우 방식으로 train/test 인덱스 배열을 생성합니다."""
    for start in range(0, total_len - train_size - test_size + 1, test_size):
        yield (
            np.arange(start, start + train_size),
            np.arange(start + train_size, start + train_size + test_size),
        )


def find_best_hyperparams(X_train, y_train, X_val, y_val, num_features):
    """주어진 데이터셋에서 최적의 하이퍼파라미터를 탐색합니다."""
    hyperparams_candidates = {
        "lstm_units": [100, 150],
        "dropout_rate": [0.2, 0.3],
        "learning_rate": [0.001, 0.0005],
    }

    best_val_loss = float("inf")
    best_params = None

    for lstm_units, dropout_rate, learning_rate in product(
        *hyperparams_candidates.values()
    ):
        model = build_model(
            LOOK_BACK, num_features, lstm_units, dropout_rate, learning_rate
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            callbacks=callbacks,
            shuffle=False,
            verbose=0,
        )

        val_loss = min(history.history["val_loss"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (lstm_units, dropout_rate, learning_rate)

    return best_params


def train():
    data_processor = DataProcessor()
    targets = data_processor.targets
    results = {}

    existing_results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing_results = json.load(f)

    for target in targets:
        model_path = MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE}"
        if model_path.exists():
            print(f"✅ {target.upper()} 모델 파일이 이미 존재하므로 학습을 건너뜁니다.")
            if target in existing_results:
                results[target] = existing_results[target]
            continue

        X_seq, y_seq, y_idxs = data_processor.get_sequence_data(target=target)

        y_preds_all, y_true_all, y_idxs_all = [], [], []
        best_params = None

        total_len = len(X_seq)
        print(f"✅ {target.upper()} 모델 신규 학습 시작 (총 {total_len}개 시퀀스)")

        for i, (train_idx, test_idx) in enumerate(rolling_split_index(total_len)):
            X_train, y_train = X_seq[train_idx], y_seq[train_idx]
            X_test, y_test = X_seq[test_idx], y_seq[test_idx]

            if i == 0:
                print("   - 최적 하이퍼파라미터 탐색 중...")
                best_params = find_best_hyperparams(
                    X_train, y_train, X_test, y_test, X_seq.shape[2]
                )
                print(
                    f"   - 최적 하이퍼파라미터: LSTM Units={best_params[0]}, Dropout={best_params[1]}, LR={best_params[2]}"
                )

            print(f"   - 롤링 윈도우 {i + 1} 학습 및 예측 수행...")
            model = build_model(LOOK_BACK, X_seq.shape[2], *best_params)
            model.fit(
                X_train, y_train, epochs=50, batch_size=32, shuffle=False, verbose=0
            )

            y_pred = model.predict(X_test, verbose=0)

            y_preds_all.append(y_pred)
            y_true_all.append(y_test)
            y_idxs_all.append(y_idxs[test_idx])

        y_pred_concat = np.concatenate(y_preds_all)
        y_true_concat = np.concatenate(y_true_all)
        y_idxs_concat = np.concatenate(y_idxs_all)

        target_scaler = data_processor.get_target_scaler(target)
        y_true_inv = target_scaler.inverse_transform(y_true_concat)
        y_pred_inv = target_scaler.inverse_transform(y_pred_concat)
        metrics = evaluate_predictions(y_true_inv, y_pred_inv)

        pred_df = pd.DataFrame(
            {"true": y_true_inv.flatten(), "pred": y_pred_inv.flatten()},
            index=y_idxs_concat,
        )
        os.makedirs(PRED_TRUE_DIR, exist_ok=True)
        pred_df.to_csv(PRED_TRUE_DIR / f"{target}_{PRED_TRUE_CSV}")
        print(f"   - Saved: {target}_{PRED_TRUE_CSV}")

        print("   - 전체 데이터로 최종 모델 학습 및 저장 중...")
        final_model = build_model(LOOK_BACK, X_seq.shape[2], *best_params)
        final_model.fit(
            X_seq, y_seq, epochs=50, batch_size=32, shuffle=False, verbose=0
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        final_model.save(MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE}")

        best_params_dict = {
            "lstm_units": best_params[0],
            "dropout_rate": best_params[1],
            "learning_rate": best_params[2],
        }
        results[target] = {"best_params": best_params_dict, "metrics": metrics}

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("✨ 모든 학습/결과 처리가 완료되었습니다.")


if __name__ == "__main__":
    train()
