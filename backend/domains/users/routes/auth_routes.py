from fastapi import APIRouter, Cookie, HTTPException, Response
from domains.users.schemes.user_info import UserInfo
from modules.mongodb import MongoDB
from modules.redis import Redis
from domains.users.schemes.login_request import LoginRequest
from domains.users.services.auth_service import (
    SESSION_TTL,
    create_session,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
async def login_user(login_request: LoginRequest, response: Response) -> dict:
    """
    사용자 로그인 처리 함수

    Args:
        login_request (LoginRequest): 로그인 요청 데이터
        response (Response): FastAPI Response 객체

    Returns:
        dict: 로그인 성공 메시지
    """
    user = await MongoDB.get_database().users.find_one(
        {"username": login_request.username}
    )

    if not user or not verify_password(login_request.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    user_info = UserInfo(uid=str(user["_id"]), username=user["username"])
    session_id = await create_session(user_info=user_info)

    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=SESSION_TTL,
        path="/",
        domain=None,
    )

    return {"message": "Login successful"}


@router.post("/logout")
async def logout_user(response: Response, session_id: str = Cookie(None)) -> dict:
    """
    사용자 로그아웃 처리 함수

    Args:
        response (Response): FastAPI Response 객체
        session_id (str, optional): 쿠키에서 전달된 세션 ID

    Returns:
        dict: 로그아웃 성공 메시지
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="No session provided")

    redis = Redis.get_client()
    deleted = await redis.delete(f"session:{session_id}")

    if deleted == 0:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    response.delete_cookie(
        key="session_id",
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
        domain=None,
    )

    return {"message": "Logout successful"}
