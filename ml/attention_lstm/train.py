import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import product
from .model import build_model
from .data_processor import DataProcessor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from .constant import (BASE_DIR, KERAS_FILE_TEMPLATE, LOOK_BACK, MODEL_DIR, PRED_TRUE_DIR)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(y_true_valid) == 0:
        return {"rmse": np.nan, "r2": np.nan, "mape": np.nan}

    return {
        "rmse": np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        "r2": r2_score(y_true_valid, y_pred_valid),
        "mape": mean_absolute_percentage_error(y_true_valid, y_pred_valid) * 100,
    }

def rolling_split_index(total_len, train_size=1200, test_size=300):
    for start in range(0, total_len - train_size - test_size + 1, 100):
        yield (
            np.arange(start, start + train_size),
            np.arange(start + train_size, start + train_size + test_size),
        )

def find_best_hyperparams(X_train_seq, y_train_seq, X_val_seq, y_val_seq, num_features):
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
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ]

        history = model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=100,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=False,
            verbose=1,
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
    train_size = 1200
    test_size = 300

    results_file = BASE_DIR / "train_results.json"
    existing_results = {}
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)

    for target in targets:
        model_path = MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE}"
        if model_path.exists():
            print(f"{target.upper()} 모델 파일 이미 존재. 학습 생략.")
            if target in existing_results:
                results[target] = existing_results[target]
            continue
            
        X_seq_full, y_seq_full, y_idxs_full = data_processor.get_sequence_data(target=target)
        
        total_len_seq = len(X_seq_full)
        num_features = X_seq_full.shape[2]

        seq_splits = list(rolling_split_index(total_len_seq, train_size, test_size))

        if not seq_splits:
             print(f"데이터 부족 {target} 학습 생략")
             results[target] = None
             continue
        
        y_preds_all, y_true_all, y_idxs_all = [], [], []
        best_params = None
        
        print(f"{target.upper()} 모델 학습 (총 {len(seq_splits)}개 롤링 윈도우)")

        # 2. 롤링 윈도우 교차 검증
        for i, (train_seq_idx, test_seq_idx) in enumerate(seq_splits):
            
            X_train_seq = X_seq_full[train_seq_idx]
            y_train_seq = y_seq_full[train_seq_idx]
            X_test_seq = X_seq_full[test_seq_idx]
            y_test_seq = y_seq_full[test_seq_idx]
            y_test_idxs_seq = y_idxs_full[test_seq_idx]
            
            if i == 0:
                print("최적 하이퍼파라미터 탐색")
                best_params = find_best_hyperparams(
                    X_train_seq, y_train_seq, X_test_seq, y_test_seq, num_features
                )
                if best_params is None:
                    print("하이퍼파라미터 탐색 실패, 기본값 (150, 0.3, 0.001, 32) 사용.")
                    best_params = (150, 0.3, 0.001, 32) 
                    
                print(
                    f"최적 하이퍼파라미터: LSTM Units={best_params[0]}, Dropout={best_params[1]}, LR={best_params[2]}, Batch Size={best_params[3]}"
                )
            
            lstm_units, dropout_rate, learning_rate, batch_size = best_params
            print(f"롤링 윈도우 {i + 1} 학습 및 예측")
            
            model = build_model(LOOK_BACK, num_features, lstm_units, dropout_rate, learning_rate)
            model.fit(
                X_train_seq, y_train_seq, epochs=10, batch_size=batch_size, shuffle=False, verbose=0
            )

            y_pred = model.predict(X_test_seq, verbose=0)

            y_preds_all.append(y_pred)
            y_true_all.append(y_test_seq)
            y_idxs_all.append(y_test_idxs_seq)

        y_pred_concat = np.concatenate(y_preds_all)
        y_true_concat = np.concatenate(y_true_all)
        y_idxs_concat = np.concatenate(y_idxs_all)
        
        # 3. 역변환 및 평가
        target_scaler = data_processor.get_target_scaler(target)
        
        y_true_inv = target_scaler.inverse_transform(y_true_concat.reshape(-1, 1))
        y_pred_inv = target_scaler.inverse_transform(y_pred_concat.reshape(-1, 1))
        
        metrics = evaluate_predictions(y_true_inv.flatten(), y_pred_inv.flatten())

        print(f"\n{target.upper()} 최종 성능 평가 → RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}, MAPE={metrics['mape']:.2f}%")

        # 4. 예측 결과 저장
        pred_df = pd.DataFrame(
            {"true": y_true_inv.flatten(), "pred": y_pred_inv.flatten()},
            index=y_idxs_concat,
        )
        os.makedirs(PRED_TRUE_DIR, exist_ok=True)
        pred_df.to_csv(PRED_TRUE_DIR / f"{target}_pred_true.csv")
        print(f"저장: {target}_pred_true.csv")

        # 5. 전체 데이터로 최종 모델 학습 및 저장
        print("전체 데이터로 최종 모델 학습 및 저장")
        
        final_model = build_model(LOOK_BACK, num_features, lstm_units, dropout_rate, learning_rate)
        final_model.fit(
            X_seq_full, y_seq_full, epochs=50, batch_size=batch_size, shuffle=False, verbose=1
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        final_model.save(MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE}")
        print(f"모델 저장: {target}{KERAS_FILE_TEMPLATE}")

        best_params_dict = {
            "lstm_units": best_params[0],
            "dropout_rate": best_params[1],
            "learning_rate": best_params[2],
            "batch_size": best_params[3],
        }
        results[target] = {"best_params": best_params_dict, "metrics": metrics}

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("모든 학습/결과 처리 완료.")

if __name__ == "__main__":
    train()