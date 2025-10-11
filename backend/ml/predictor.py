import pandas as pd

from datetime import datetime
from .database import MongoDB
from .lstm_mha import predict as lstm_predict
from .xgboost import predict as XGBoost_predict
from .attention_lstm import predict as attention_lstm_predict

def insert_predicted_price():

    # 각 모델별 예측 결과에 index(모델명) 추가
    results = []
    today = datetime.now().strftime("%Y-%m-%d")

    results.append(
        {"date": today, "model": "LSTM-Rolling", **lstm_predict.predict_next_day()}
    )
    results.append(
        {"date": today, "model": "Attention_LSTM-Rolling", **attention_lstm_predict.predict_next_day()}
    )
    results.append(
        {"date": today, "model": "XGBoost-Rolling", **XGBoost_predict.predict_next_day()}
    )

    df = pd.DataFrame(results)
    df.set_index("date", inplace=True)

    print(df)
    
    MongoDB.connect()
    db = MongoDB.get_database()
    db["predicted_price"].insert_many(df.reset_index().to_dict("records"))
    MongoDB.close()

if __name__ == "__main__":
    insert_predicted_price()
