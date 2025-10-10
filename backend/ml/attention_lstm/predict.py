import os
import json 
import numpy as np
import pandas as pd

from .model import AttentionLayer
from .data_processor import DataProcessor
from tensorflow.keras.models import load_model
from .constant import LOOK_BACK, MODEL_DIR, BASE_DIR, KERAS_FILE_TEMPLATE, PRED_TRUE_DIR, ACCUMULATED_PRED_CSV 

def load_model_keras(target: str):
    model_path = MODEL_DIR / f"{target}{KERAS_FILE_TEMPLATE}"
    model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    return model

def predict_next_day():
    data_processor = DataProcessor()

    results_file = BASE_DIR / "train_results.json"
    with open(results_file, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    targets = data_processor.targets
    pred_values = {}

    for target in targets:
        data = data_processor.get_proceed_data(target=target)
        data.ffill(inplace=True)
        data.dropna(inplace=True)

        X = data.drop(columns=[target])
        model = load_model_keras(target)

        X_last = X.iloc[-LOOK_BACK:]

        feature_scaler = data_processor.get_feature_scaler(target=target)
        X_input_scaled = feature_scaler.transform(X_last)
        
        X_input_keras = np.expand_dims(X_input_scaled, axis=0)

        y_pred_scaled_np = model.predict(X_input_keras, verbose=0)
        
        target_scaler = data_processor.get_target_scaler(target)
        y_pred = target_scaler.inverse_transform(y_pred_scaled_np).flatten()[0]
        pred_values[target] = y_pred

    return pred_values