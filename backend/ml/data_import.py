import pandas as pd
from pymongo.collection import Collection


def import_from_db(collection: Collection) -> pd.DataFrame:
    # MongoDB에서 모든 문서 조회
    cursor = collection.find({}, {"_id": 0})  # _id 제외
    data = list(cursor)

    if not data:
        print(f"{collection} 컬렉션에 데이터가 없습니다.")
        return

    df = pd.DataFrame(data)
    df.sort_values("date", inplace=True)  # 날짜 순으로 정렬
    return df
