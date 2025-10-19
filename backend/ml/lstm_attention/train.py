import json
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import AttentionLSTM
from .constant import (
    TRAIN_RESULT,
    MODEL_DIR,
    PRED_TRUE_DIR,
    PRED_TRUE_CSV,
    MODEL_FILE_TEMPLATE,
)
from .data_processor import DataProcessor


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def rolling_split_index(total_len, train_size=1500, test_size=300):
    for start in range(0, total_len - train_size - test_size + 1, test_size):
        yield (
            np.arange(start, start + train_size),
            np.arange(start + train_size, start + train_size + test_size),
        )


def find_best_hyperparams(train_loader, val_loader, num_features):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        model = AttentionLSTM(num_features, lstm_units, dropout_rate).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        patience = 3
        epochs_no_improve = 0
        min_val_loss = float("inf")

        for epoch in range(30):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()

            val_loss /= len(val_loader)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                break

        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_params = (lstm_units, dropout_rate, learning_rate, 32)

    return best_params


def train():
    data_processor = DataProcessor()
    targets = data_processor.targets
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    existing_results = {}
    if TRAIN_RESULT.exists():
        with open(TRAIN_RESULT, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                print("Warning: result file is corrupted or empty. Starting fresh.")

    for target in targets:
        model_path = MODEL_DIR / f"{target}{MODEL_FILE_TEMPLATE}"
        if model_path.exists():
            print(f"✅ {target.upper()} 모델 파일이 이미 존재하므로 학습을 건너뜁니다.")
            if target in existing_results:
                results[target] = existing_results[target]
            continue

        X_seq, y_seq, y_idxs = data_processor.get_sequence_data(target=target)

        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq)

        y_preds_all, y_true_all, y_idxs_all = [], [], []
        best_params = None

        total_len = len(X_tensor)
        print(f"✅ {target.upper()} 모델 신규 학습 시작 (총 {total_len}개 시퀀스)")

        for i, (train_idx, test_idx) in enumerate(rolling_split_index(total_len)):
            X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
            X_test, y_test = X_tensor[test_idx], y_tensor[test_idx]

            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            if i == 0:
                print("   - 최적 하이퍼파라미터 탐색 중...")
                best_params = find_best_hyperparams(
                    train_loader, val_loader, X_tensor.shape[2]
                )
                if best_params is None:
                    best_params = (150, 0.3, 0.001, 32)
                print(
                    f"   - 최적 하이퍼파라미터: LSTM Units={best_params[0]}, Dropout={best_params[1]}, LR={best_params[2]}"
                )

            lstm_units, dropout_rate, learning_rate, batch_size = best_params
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False
            )

            print(f"   - 롤링 윈도우 {i + 1} 학습 및 예측 수행...")
            model = AttentionLSTM(X_tensor.shape[2], lstm_units, dropout_rate).to(
                device
            )
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            for epoch in range(50):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test.to(device))

            y_preds_all.append(y_pred.cpu().numpy())
            y_true_all.append(y_test.cpu().numpy())
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
        full_dataset = TensorDataset(X_tensor, y_tensor)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        final_model = AttentionLSTM(X_tensor.shape[2], lstm_units, dropout_rate).to(
            device
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)

        for epoch in range(50):
            for inputs, labels in full_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = final_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(final_model.state_dict(), model_path)
        print(f"모델 저장: {model_path.name}")

        best_params_dict = {
            "lstm_units": best_params[0],
            "dropout_rate": best_params[1],
            "learning_rate": best_params[2],
            "batch_size": best_params[3],
        }
        results[target] = {"best_params": best_params_dict, "metrics": metrics}

    with open(TRAIN_RESULT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("✨ 모든 학습/결과 처리가 완료되었습니다.")


if __name__ == "__main__":
    train()
