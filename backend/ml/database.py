from pymongo import MongoClient
from pymongo.database import Database
from .config import config

# MongoDB 연결 관리 유틸리티 클래스
class MongoDB:
    _client: MongoClient | None = None
    _db: Database | None = None

    # MongoDB 클라이언트 초기화
    @classmethod
    def connect(cls):
        if cls._client is None:
            cls._client = MongoClient(config.MONGO_URI)
            cls._db = cls._client[config.MONGO_DB_NAME]
            print("✅ MongoDB connected")

    # MongoDB 연결 종료
    @classmethod
    def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            print("🛑 MongoDB connection closed")

    # 데이터베이스 핸들 반환
    @classmethod
    def get_database(cls) -> Database:
        if cls._db is None:
            raise RuntimeError("❌ MongoDB not connected. Did you call MongoDB.connect()?")
        return cls._db

if __name__ == "__main__":
    print("MONGO DB 테스트:")
    MongoDB.connect()
    db = MongoDB.get_database()
    print("컬렉션 목록:", db.list_collection_names())
    MongoDB.close()
