import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
    
    REDIS_URI = os.getenv("REDIS_URI")
    
    KAFKA_BROKER = os.getenv("KAFKA_BROKER")
    
    KAFKA_FLUSH_INTERVAL = 10  # seconds
    
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")


config = Config()
