import re
import json
from uuid import uuid4
from typing import Union
from modules.logger import log
from modules.redis import Redis
from fastapi import Request, WebSocket
from passlib.context import CryptContext
from domains.users.schemes.user_info import UserInfo

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SESSION_TTL = 3600  # 1 hour
SESSION_ID_PATTERN = re.compile(r"session_id=([^;]+)")


def hash_password(password: str) -> str | None:
    """
    비밀번호를 해시합니다.

    Args:
        password (str): 평문 비밀번호

    Returns:
        str: 해시된 비밀번호
    """
    try:
        return pwd_context.hash(password)
    except Exception as e:
        log.error(e)
        return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    비밀번호 일치 여부를 확인합니다.

    Args:
        plain_password (str): 평문 비밀번호
        hashed_password (str): 해시된 비밀번호

    Returns:
        bool: 일치 여부
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        log.error(e)
        return False


async def create_session(user_info: UserInfo) -> str | None:
    """
    사용자 세션을 생성하고 Redis에 저장합니다.

    Args:
        user_info (UserInfo): 사용자 ID

    Returns:
        str: 세션 ID
    """
    
    try:
        session_id = str(uuid4())
        redis = Redis.get_client()

        await redis.set(f"session:{session_id}", json.dumps(user_info.model_dump()), ex=SESSION_TTL)
        return session_id
    except Exception as e:
        log.error(e)
        return None


async def get_user_info(session_id: str) -> UserInfo | None:
    """
    클라이언트 연결(WebSocket 또는 HTTP Request)에서 세션 ID를 추출하고,
    Redis에서 세션 정보를 조회하여 로그인한 사용자의 정보를 반환합니다.

    Args:
        conn (WebSocket | Request): 클라이언트 연결 객체

    Returns:
        UserInfo | None: 로그인한 사용자의 정보,
                         세션이 없거나 만료되었거나 사용자가 존재하지 않으면 None
    """
    try:
        if not session_id:
            return None
        
        redis = Redis.get_client()
        
        user_info_json = await redis.get(f"session:{session_id}")  # Redis에서 세션 조회
        if not user_info_json:
            return None
        
        data = json.loads(user_info_json)
        
        return UserInfo(**data)
    except Exception as e:
        log.error(e)
        return None
    
async def get_session_id(conn: Union[WebSocket, Request]) -> str | None:
    try:
        
        headers = conn.headers  # WebSocket도 request와 동일하게 headers 속성 있음
        cookie_header = headers.get("cookie", "")
        match = SESSION_ID_PATTERN.search(cookie_header)
            
        if not match:
            return None
        
        session_id = match.group(1)
        return session_id
    
    except Exception as e:
        log.error(e)
        return None
    
async def refresh_session(session_id: str):
    redis = Redis.get_client()
    exists = await redis.exists(f"session:{session_id}")
    if exists:
        await redis.expire(f"session:{session_id}", SESSION_TTL)