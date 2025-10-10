from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from .config import config
from .logger import log

class MongoDB:
    _client: AsyncIOMotorClient | None = None
    _db: AsyncIOMotorDatabase | None = None

    @classmethod
    def connect(cls):
        """MongoDB 클라이언트 초기화"""
        if cls._client is None:
            cls._client = AsyncIOMotorClient(config.MONGO_URI)
            cls._db = cls._client[config.MONGO_DB_NAME]
            log.info("✅ MongoDB 연결")

    @classmethod
    def close(cls):
        """MongoDB 연결 종료"""
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            log.info("🛑 MongoDB 연결 종료")

    @classmethod
    def get_database(cls) -> AsyncIOMotorDatabase:
        """DB 핸들 가져오기"""
        if cls._db is None:
            raise RuntimeError(
                "❌ MongoDB 연결 실패. MongoDB.connect()를 호출 하셨습니까?"
            )
        return cls._db
