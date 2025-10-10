from modules.mongodb import MongoDB
from domains.users.models.user import User
from fastapi import APIRouter, HTTPException, Request
from domains.users.schemes.register_request import RegisterRequest
from domains.users.services.auth_service import (
    get_session_id,
    get_user_info,
    hash_password,
)

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/profile")
async def get_profile(request: Request):
    """
    현재 로그인한 사용자의 프로필 정보를 반환합니다.

    Args:
        request (Request): FastAPI 요청 객체

    Returns:
        dict: 사용자 정보
    """
    session_id = await get_session_id(request)
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user_info = await get_user_info(session_id)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid session or user not found")

    return user_info.model_dump()


@router.post("/register")
async def register_user(register_data: RegisterRequest) -> dict[str, str]:
    """
    새로운 사용자를 등록합니다.

    Args:
        register_data (RegisterRequest): 사용자 등록 요청 데이터

    Returns:
        dict[str, str]: 등록 성공 메시지
    """
    existing_user = await MongoDB.get_database().users.find_one(
        {"username": register_data.username}
    )
    if existing_user:
        raise HTTPException(status_code=409, detail="User already exists")

    user = User(**register_data.model_dump())
    user.password = hash_password(user.password)
    await MongoDB.get_database().users.insert_one(user.model_dump())

    return {"message": "Register successful"}
