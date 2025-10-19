import json
import os
import numpy as np
import joblib
from itertools import product
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from model import build_model
from constant import MODEL_FILE_TEMPLATE, LOOK_BACK, MODEL_DIR, RESULTS_PATH
from .data_processor import DataProcessor


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def rolling_split_index(total_len, train_size=1200, test_size=300):
    """
    전체 데이터 길이에서 rolling window 방식으로
    train/test 인덱스 배열을 생성합니다.
    """
    for start in range(0, total_len - train_size - test_size + 1, test_size):
        yield (
            np.arange(start, start + train_size),
            np.arange(start + train_size, start + train_size + test_size),
        )


def get_best_fold(X_train, y_train, X_test, y_test):
    hyperparams_candidates = {
        "n_estimators": [100, 200],
        "max_depth": [6, 8],
        "learning_rate": [0.1, 0.05],
    }

    best_val_loss = float("inf")
    best_params = None

    for n_estimators, max_depth, learning_rate in product(
        hyperparams_candidates["n_estimators"],
        hyperparams_candidates["max_depth"],
        hyperparams_candidates["learning_rate"],
    ):
        model = build_model(n_estimators, max_depth, learning_rate)

        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        val_loss = mean_squared_error(y_test, y_pred)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (n_estimators, max_depth, learning_rate)

    return best_params, best_val_loss


def train():
    data_processor = DataProcessor()
    targets = data_processor.targets

    results = {}

    for target in targets:
        X_seq, y_seq, y_idxs = data_processor.get_sequence_data(target=target)

        # XGBoost는 3D 입력을 받지 않으므로 2D로 변환
        X_seq_2d = X_seq.reshape(X_seq.shape[0], -1)

        best_params = tuple()
        best_val_loss = float("inf")
        last_test_idx = None

        total_len = len(X_seq) + LOOK_BACK
        print(f"{target.upper()} 모델 학습 시작: 데이터갯수: {total_len}")
        for train_idx, test_idx in rolling_split_index(total_len):
            X_train, y_train = X_seq_2d[train_idx], y_seq[train_idx].flatten()
            X_test, y_test = X_seq_2d[test_idx], y_seq[test_idx].flatten()
            y_test_idx = y_idxs[test_idx]
            last_test_idx = (X_test, y_test, y_test_idx)

            fold_best_params, fold_best_val_loss = get_best_fold(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )

            if fold_best_val_loss < best_val_loss:
                best_params = fold_best_params

        print(f"{target.upper()} 최적 하이퍼파라미터: {best_params}")

        model = build_model(*best_params)
        model.fit(X_seq_2d, y_seq.flatten(), verbose=False)

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_name = f"{target}{MODEL_FILE_TEMPLATE}"
        joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}"))

        X_test, y_test, y_test_idx = last_test_idx
        y_pred = model.predict(X_test)
        data_processor.save_predictions_csv(
            y_test, y_pred, target=target, index=y_test_idx
        )

        target_scaler = data_processor.get_target_scaler(target)
        y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        metrics = evaluate_predictions(y_test_inv, y_pred_inv)

        best_params_dict = {
            "n_estimators": best_params[0],
            "max_depth": best_params[1],
            "learning_rate": best_params[2],
        }
        results[target] = {"best_params": best_params_dict, "metrics": metrics}

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    train()
