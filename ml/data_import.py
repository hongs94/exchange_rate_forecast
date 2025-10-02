import pandas as pd
from pymongo.collection import Collection

# MongoDB에서 모든 문서 조회
def import_from_db(collection: Collection) -> pd.DataFrame:
    cursor = collection.find({}, {"_id": 0})  # _id 제외
    data = list(cursor)

    if not data:
        print(f"{collection} 컬렉션에 데이터가 없음.")
        return

    df = pd.DataFrame(data)
    df.sort_values("date", inplace=True)  # 날짜순
    return df