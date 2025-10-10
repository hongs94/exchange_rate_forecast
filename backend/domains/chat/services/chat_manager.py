# ChatRoomManager
import json
from typing import Dict
from fastapi import WebSocket
from datetime import datetime
from modules.redis import Redis

MAX_MSG = 100  # Redis 캐시 최대 메시지 수
CHAT_REDIS_KEY = "chat"

class ChatRoomManager:
    sockets: Dict[str, WebSocket] = {}

    @classmethod
    async def connect(cls, ws: WebSocket, uid: str):
        await ws.accept()

        old_ws = cls.sockets.get(uid, None)

        if old_ws:
            old_ws.close()

        cls.sockets[uid] = ws

        # Redis에서 최근 메시지 불러오기
        redis = Redis.get_client()
        recent_msgs = await redis.lrange(CHAT_REDIS_KEY, 0, -1)
        for msg in reversed(recent_msgs):
            try:
                msg_data = json.loads(msg)
                is_me = msg_data.get("uid") == uid
                msg_data["isMe"] = is_me
                await ws.send_text(json.dumps(msg_data))
            except Exception:
                # 혹시 json decode 에러 등 있을 때는 무시
                continue

    @classmethod
    async def disconnect(cls, ws: WebSocket, uid: str):
        ws = cls.sockets.pop(uid, None)
        try:
            await ws.close()
        except RuntimeError:
            pass

    @classmethod
    async def broadcast(cls, uid: str, username: str, message: str):
        redis = Redis.get_client()
        now = datetime.now().isoformat()
        # 기본 메시지 객체 생성
        msg_obj = {
            "username": username,
            "message": message,
            "time": now,
            "uid": uid,
        }
        # Redis에 저장
        await redis.lpush(CHAT_REDIS_KEY, json.dumps(msg_obj))
        await redis.ltrim(CHAT_REDIS_KEY, 0, MAX_MSG - 1)

        # WebSocket으로 메시지 전송
        for uid_key, ws in cls.sockets.items():
            try:
                # isMe 필드만 추가해서 전송
                msg_obj_with_isme = {**msg_obj, "isMe": uid == uid_key}
                await ws.send_text(json.dumps(msg_obj_with_isme))
            except:
                await cls.disconnect(ws, uid_key)
