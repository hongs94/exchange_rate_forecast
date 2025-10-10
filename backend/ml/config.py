import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
    ECOS_API_KEY = os.getenv("ECOS_API_KEY")

config = Config()
