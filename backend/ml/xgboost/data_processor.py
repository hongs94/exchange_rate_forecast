import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ..data_merge import create_merged_dataset
from .constant import (
    PRED_TRUE_DIR,
    LOOK_BACK,
    PRED_TRUE_CSV,
    SCALER_DIR,
)

class DataProcessor:
    def __init__(self):
        self.origin_data = create_merged_dataset()
        
        self.origin_data["kr_us_diff"] = (self.origin_data["kr_rate"] - self.origin_data["us_rate"])
        self.origin_data["us_jp_diff"] = (self.origin_data["dgs10"] - self.origin_data["jpy10"])
        self.origin_data["us_eu_diff"] = (self.origin_data["dgs10"] - self.origin_data["eur10"])
        
        self.targets = ["usd", "cny", "jpy", "eur"]

        self.feature_map = {
            "usd": ["dgs10", "vix", "dxy", "kr_us_diff", "kr_rate", "us_rate"],
            "cny": ["cny_fx_reserves", "cny_trade_bal", "wti", "vix"],
            "jpy": ["jpy10", "dgs10", "us_jp_diff", "vix"],
            "eur": ["eur10", "dxy", "us_eu_diff", "vix"],
        }
    
    def add_indicators(self, data: pd.DataFrame, target: str):
        periods = [5, 20, 60, 120]
        for p in periods:
            data[f"MA_{p}"] = data[target].rolling(p).mean()
            data[f"EMA_{p}"] = data[target].ewm(span=p, adjust=False).mean()
        data.dropna(inplace=True)

    def get_target_scaler(self, target: str):
        scaler_path = SCALER_DIR / f"{target}_target_scaler.pkl"
        return joblib.load(scaler_path)

    def get_feature_scaler(self, target: str):
        scaler_path = SCALER_DIR / f"{target}_feature_scaler.pkl"
        return joblib.load(scaler_path)

    # XGBoost를 위해 시퀀스를 2D (LOOK_BACK * Features)로 평탄화
    def create_sequences(
        self, X: np.ndarray, y: np.ndarray, seq_len: int, y_index=None
    ):
        Xs, ys, idxs = [], [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len].flatten())
            ys.append(y[i + seq_len])
            if y_index is not None:
                idxs.append(y_index[i + seq_len])
        
        if y_index is not None:
            return np.array(Xs), np.array(ys), np.array(idxs)
        return np.array(Xs), np.array(ys)
    
    def get_proceed_data(self, target) -> pd.DataFrame:
        data = self.origin_data.copy()

        features_for_target = self.feature_map.get(target)
        if not features_for_target:
            raise ValueError(f"'{target}'에 대한 Feature 목록이 정의되지 않았음.")
            
        keep_cols = [target] + features_for_target
        data = data[keep_cols]
        
        self.add_indicators(data=data, target=target)

        return data

    def get_sequence_data(self, target: str):
        data = self.get_proceed_data(target=target)

        X = data.drop(columns=[target])
        y = data[target]

        target_scaler = MinMaxScaler()
        feature_scaler = MinMaxScaler()

        target_scaled = target_scaler.fit_transform(y.to_frame())
        features_scaled = feature_scaler.fit_transform(X)

        os.makedirs(SCALER_DIR, exist_ok=True)
        joblib.dump(
            target_scaler, os.path.join(SCALER_DIR, f"{target}_target_scaler.pkl")
        )
        joblib.dump(
            feature_scaler, os.path.join(SCALER_DIR, f"{target}_feature_scaler.pkl")
        )

        X_seq, y_seq, y_idxs = self.create_sequences(
            X=features_scaled, y=target_scaled, seq_len=LOOK_BACK, y_index=y.index
        )

        return X_seq, y_seq, y_idxs