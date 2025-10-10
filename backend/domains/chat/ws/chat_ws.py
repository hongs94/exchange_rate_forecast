# WebSocket Endpoint
from modules.kafka import send_kafka_chat
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from domains.chat.services.chat_manager import ChatRoomManager
from domains.users.services.auth_service import (
    get_session_id,
    get_user_info,
    refresh_session,
)

def register_chat_ws(app: FastAPI):
    @app.websocket("/ws/chat")
    async def chat(ws: WebSocket):
        session_id = await get_session_id(ws)
        user_info = await get_user_info(session_id)

        if not user_info:
            return
        uid = user_info.uid
        await ChatRoomManager.connect(ws, uid)

        try:
            while True:
                data = await ws.receive_text()
                user_info = await get_user_info(session_id)

                await refresh_session(session_id)
                await send_kafka_chat(user_info.uid, user_info.username, data)

        except WebSocketDisconnect:
            await ChatRoomManager.disconnect(ws, uid)
