from .config import config
from .logger import log
import redis.asyncio as aioredis

class Redis:
    _client: aioredis.Redis | None = None

    @classmethod
    async def connect(cls):
        """Redis 클라이언트 초기화"""
        if cls._client is None:
            cls._client = aioredis.from_url(
                config.REDIS_URI,
                decode_responses=True,  # 문자열 반환을 위해
            )
            # 연결 테스트
            try:
                await cls._client.ping()
                log.info("✅ Redis 연결")
            except Exception as e:
                log.error(f"❌ Redis 연결 실패: {e}")
                cls._client = None
                raise

    @classmethod
    async def close(cls):
        """Redis 연결 종료"""
        if cls._client:
            await cls._client.close()
            cls._client = None
            log.info("🛑 Redis 연결 종료")

    @classmethod
    def get_client(cls) -> aioredis.Redis:
        """Redis 클라이언트 가져오기"""
        if cls._client is None:
            raise RuntimeError(
                "❌ Redis 연결 실패. Redis.connect()를 호출 하셨습니까?"
            )
        return cls._client