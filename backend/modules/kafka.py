import time
import json
import asyncio

from modules.config import config
from modules.mongodb import MongoDB
from datetime import datetime, timezone
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from domains.chat.services.chat_manager import ChatRoomManager

# Kafka Producer/Consumer
kafka_producer: AIOKafkaProducer = None
kafka_log_consumer: AIOKafkaConsumer = None
kafka_chat_consumer: AIOKafkaConsumer = None

async def start_kafka(loop):
    global kafka_producer, kafka_log_consumer, kafka_chat_consumer

    kafka_producer = AIOKafkaProducer(bootstrap_servers=config.KAFKA_BROKER, loop=loop)
    await kafka_producer.start()

    kafka_log_consumer = AIOKafkaConsumer(
        "log-topic",
        bootstrap_servers=config.KAFKA_BROKER,
        group_id="log-service",
        auto_offset_reset="earliest",
        loop=loop,
    )
    await kafka_log_consumer.start()
    asyncio.create_task(log_consumer_task())
    
    kafka_chat_consumer = AIOKafkaConsumer(
        "chat-topic",
        bootstrap_servers=config.KAFKA_BROKER,
        group_id="chat-service",
        auto_offset_reset="earliest",
        loop=loop,
    )
    await kafka_chat_consumer.start()
    asyncio.create_task(chat_consumer_task())
    
async def log_consumer_task():
    buffer = []
    last_flush = time.time()
    mongo_db = MongoDB.get_database()

    async for msg in kafka_log_consumer:
        data = json.loads(msg.value.decode())
        buffer.append({
            "type": data["type"],
            "uid": data["uid"],
            "username": data["username"],
            "path": data["path"],
            "method": data["method"],
            "status_code": data["status_code"],
            "process_time": data["process_time"],
            "timestamp": datetime.now(timezone.utc)
        })

        # Flush 조건
        if time.time() - last_flush > config.KAFKA_FLUSH_INTERVAL:
            if buffer:
                await mongo_db["system_logs"].insert_many(buffer)
                buffer.clear()
            last_flush = time.time()
            
            
async def send_kafka_log(
    uid: str,
    type :str,
    username: str,
    path: str,
    method: str,
    status_code: str,
    process_time: float,
):
    while kafka_producer is None:
        await asyncio.sleep(0.1)
        
    data = {
        "type": type,
        "uid": uid,
        "username": username,
        "path": path,
        "method": method,
        "status_code": status_code,
        "process_time": process_time,
    }
    await kafka_producer.send_and_wait("log-topic", json.dumps(data).encode())
    
    
async def chat_consumer_task():
    buffer = []
    last_flush = time.time()
    mongo_db = MongoDB.get_database()
    
    async for msg in kafka_chat_consumer:
        data = json.loads(msg.value.decode())
        buffer.append({
            "uid": data["uid"],
            "username": data["username"],
            "message": data["msg"],
            "timestamp": datetime.now(timezone.utc)
        })
        
        await ChatRoomManager.broadcast(
            uid=data["uid"], username=data["username"], message=data["msg"]
        )
        
        # Flush 조건
        if time.time() - last_flush > config.KAFKA_FLUSH_INTERVAL:
            if buffer:
                await mongo_db["chat_logs"].insert_many(buffer)
                buffer.clear()
            last_flush = time.time()
            
            
async def send_kafka_chat(uid: str, username: str, message: str):
    while kafka_producer is None:
        await asyncio.sleep(0.1)
        
    data = {"uid": uid, "username": username, "msg": message}
    await kafka_producer.send_and_wait("chat-topic", json.dumps(data).encode())