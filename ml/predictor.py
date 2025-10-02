import pandas as pd
import ml.lstm.predict as lstm_predict
import ml.xgboost_new.predict as XGBoost_predict
import ml.attention_lstm.predict as Attention_lstm_predict

from datetime import datetime
from .database import MongoDB

def insert_predicted_price():
    # 각 모델별 예측 결과에 index(모델명) 추가
    results = []
    today = datetime.now().strftime("%Y-%m-%d")

    results.append({"date": today, "model": "LSTM", **lstm_predict.predict_next_day()})
    
    results.append({"date": today, "model": "Attention-LSTM", **Attention_lstm_predict.predict_next_day()})

    results.append({"date": today, "model": "XGBoost", **XGBoost_predict.predict_next_day()})

    df = pd.DataFrame(results)
    df.set_index("date", inplace=True)

    print(df)
    
    MongoDB.connect()
    db = MongoDB.get_database()
    db["predicted_price"].insert_many(df.reset_index().to_dict("records"))
    MongoDB.close()

if __name__ == "__main__":
    insert_predicted_price()
