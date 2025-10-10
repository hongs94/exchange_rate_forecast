from pydantic import BaseModel


class UserInfo(BaseModel):
    uid: str
    username: str
