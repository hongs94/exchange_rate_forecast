from .database import MongoDB
from .config import config
from .data_ingest import (
    insert_with_datareader,
    insert_with_yfinance,
    insert_with_ecos,
)

if __name__ == "__main__":
    MongoDB.connect()
    db = MongoDB.get_database()
    eco_api_key = config.ECOS_API_KEY

    # DGS10: % 단위 (공휴일/휴장일 결측치)
    insert_with_datareader(
        db=db, coll_name="dgs10", reader_name="DGS10", data_source="fred"
    )

    # 중국 외환보유액 (China FX Reserves): 백만 달러 단위
    insert_with_datareader(
        db=db,
        coll_name="cny_fx_reserves",
        reader_name="TRESEGCNM052N",
        data_source="fred",
    )

    # 중국 무역수지 (월별): 위안 단위
    insert_with_datareader(
        db=db,
        coll_name="cny_trade_bal",
        reader_name="XTNTVA01CNM664S",
        data_source="fred",
    )

    # eur 10년 만기 국채 수익률: % 단위
    insert_with_datareader(
        db=db, coll_name="eur10", reader_name="IRLTLT01EZM156N", data_source="fred"
    )

    # 일본 10년물 국채 수익률: % 단위
    insert_with_datareader(
        db=db, coll_name="jpy10", reader_name="IRLTLT01JPM156N", data_source="fred"
    )

    # 단위: 지수 (1973.03 [고정환율 -> 변동환율 변환 시점] = 100 기준)
    insert_with_yfinance(db=db, coll_name="dxy", ticker="DX-Y.NYB")

    # 한국 기준금리 (월별), 단위: 연%
    insert_with_ecos(
        db=db,
        coll_name="kr_rate",
        api_key=eco_api_key,
        stat_code="722Y001",
        interval="M",
        code="0101000",
    )

    # 미국 기준금리 (월별), 단위: 퍼센트
    insert_with_datareader(
        db=db, coll_name="us_rate", reader_name="FEDFUNDS", data_source="fred"
    )

    # VIX: 지수 단위 (S&P 500 옵션 가격으로 향후 30일간 예상 변동성 추정한 값)
    insert_with_yfinance(db=db, coll_name="vix", ticker="^VIX")

    # WTI: USD/배럴 단위
    insert_with_yfinance(db=db, coll_name="wti", ticker="CL=F")

    # 통화 4종: USD, JPY, CNY, EUR
    insert_with_ecos(
        db=db,
        coll_name="usd",
        api_key=eco_api_key,
        stat_code="731Y001",
        interval="D",
        code="0000001",
    )

    insert_with_ecos(
        db=db,
        coll_name="jpy(100)",
        api_key=eco_api_key,
        stat_code="731Y001",
        interval="D",
        code="0000002",
    )

    insert_with_ecos(
        db=db,
        coll_name="eur",
        api_key=eco_api_key,
        stat_code="731Y001",
        interval="D",
        code="0000003",
    )

    insert_with_ecos(
        db=db,
        coll_name="cny",
        api_key=eco_api_key,
        stat_code="731Y001",
        start_date="20160101",
        interval="D",
        code="0000053",
    )

    # CNY/USD (15.09.01 ~ 15.12.31)
    insert_with_ecos(
        db=db,
        coll_name="usd_cny",
        api_key=eco_api_key,
        stat_code="731Y002",
        start_date="20150901",
        end_date="20151231",
        interval="D",
        code="0000027",
    )

    MongoDB.close()
