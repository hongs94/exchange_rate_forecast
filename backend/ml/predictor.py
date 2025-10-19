from datetime import datetime
import pandas as pd
import ml.lstm.predict as lstm_predict
import ml.lstm_attention.predict as attention_lstm_rolling_predict
import ml.xgboost.predict as XGBoost_predict
from .database import MongoDB


def insert_predicted_price():
    next_day = (datetime.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    results = [
        {"date": next_day, "model": "LSTM", **lstm_predict.predict_next_day()},
        {
            "date": next_day,
            "model": "LSTM_Attention",
            **attention_lstm_rolling_predict.predict_next_day(),
        },
        {"date": next_day, "model": "XGBoost", **XGBoost_predict.predict_next_day()},
    ]

    df = pd.DataFrame(results)
    df.set_index("date", inplace=True)

    print(df)

    MongoDB.connect()
    db = MongoDB.get_database()
    db["predicted_price"].insert_many(df.reset_index().to_dict("records"))
    MongoDB.close()


if __name__ == "__main__":
    insert_predicted_price()
