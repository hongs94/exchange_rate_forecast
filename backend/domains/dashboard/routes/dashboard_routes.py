from fastapi import APIRouter, Query
from modules.mongodb import MongoDB
import pandas as pd
from pathlib import Path

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

@router.get("/")
async def dashboard(
    currency: str = Query("usd", description="조회할 통화 코드 (예: usd, eur, jpy)")
):
    db = MongoDB.get_database()

    base_dir = Path(__file__).resolve().parent

    # CSV 읽기 (날짜를 인덱스로)
    merged_data = pd.read_csv(base_dir / "dataset.csv", index_col=0, parse_dates=True)
    merged_data = merged_data[[currency]]  # 원하는 통화 열만 선택
    merged_data.index = merged_data.index.strftime("%Y-%m-%d")

    # MongoDB 조회
    cursor = db["predicted_price"].find(
        {}, {"_id": 0, "model": 1, "date": 1, currency: 1}
    )
    results = await cursor.to_list(length=None)
    df = pd.DataFrame(results)

    return {
        "data": merged_data.reset_index().to_dict(orient="records"),  # 날짜 포함
        "predicted_price": df.to_dict(orient="records")                   # date 포함
    }
