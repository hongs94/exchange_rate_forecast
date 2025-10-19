import xgboost as xgb


def build_model(n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    XGBoost 회귀 모델을 생성합니다.
    """
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
    )
    return model
