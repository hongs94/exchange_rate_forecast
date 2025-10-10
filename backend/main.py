import time
import asyncio
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Response, HTTPException, Request

from modules.redis import Redis
from modules.mongodb import MongoDB

from domains.chat.ws.chat_ws import register_chat_ws
from domains.users.services.auth_service import (
    get_session_id,
    get_user_info,
    refresh_session,
)
from modules.kafka import (
    send_kafka_log,
    start_kafka,
    kafka_producer,
    kafka_chat_consumer,
    kafka_log_consumer,
)

from domains.chat.routes.chat_routes import router as chat_router
from domains.users.routes.auth_routes import router as auth_router
from domains.users.routes.user_routes import router as users_router
from domains.dashboard.routes.dashboard_routes import router as dashboard_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await Redis.connect()
    MongoDB.connect()

    loop = asyncio.get_running_loop()

    await start_kafka(loop)

    yield

    await Redis.close()
    MongoDB.close()

    await kafka_producer.stop()
    await kafka_chat_consumer.stop()
    await kafka_log_consumer.stop()


app = FastAPI(lifespan=lifespan)

app.include_router(router=auth_router)
app.include_router(router=users_router)
app.include_router(router=chat_router)
app.include_router(router=dashboard_router)
register_chat_ws(app=app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트엔드 도메인
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code, content={"status": "error", "message": exc.detail}
    )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> Response:
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    session_id = await get_session_id(request)
    user_info = await get_user_info(session_id)

    if user_info:
        await refresh_session(session_id)

        asyncio.create_task(
            send_kafka_log(
                uid=user_info.uid,
                type="http",
                username=user_info.username,
                path=request.url.path,
                method=request.method,
                status_code=response.status_code,
                process_time=round(process_time, 3),
            )
        )
    else:
        asyncio.create_task(
            send_kafka_log(
                uid="",
                type="http",
                username="Guest",
                path=request.url.path,
                method=request.method,
                status_code=response.status_code,
                process_time=round(process_time, 3),
            )
        )
    return response
