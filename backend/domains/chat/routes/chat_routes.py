from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/chat", tags=["chat"])


# -------------------------------
# 테스트용 HTML 페이지
# -------------------------------
@router.get("")
async def test_chat_page():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chat Test</title>
    </head>
    <body>
        <h1>WebSocket Chat Test</h1>
        <div>
            <input id="messageInput" type="text" placeholder="메시지를 입력하세요" />
            <button onclick="sendMessage()">보내기</button>
        </div>
        <ul id="messages"></ul>

        <script>
            // 현재 접속한 도메인/포트 기반으로 WebSocket 연결
            const ws = new WebSocket(`ws://${location.host}/ws/chat`);

            ws.onopen = () => {
                console.log("✅ WebSocket 연결됨");
            };

            ws.onmessage = (event) => {
                const messages = document.getElementById("messages");
                const li = document.createElement("li");
                li.textContent = event.data;
                messages.appendChild(li);
            };

            ws.onclose = (event) => {
                console.log("❌ WebSocket 닫힘", event);
            };

            function sendMessage() {
                const input = document.getElementById("messageInput");
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(input.value);
                    input.value = "";
                } else {
                    alert("⚠️ WebSocket 연결이 닫혀서 메시지를 보낼 수 없습니다.");
                }
            }
        </script>
    </body>
    </html>
    """)
