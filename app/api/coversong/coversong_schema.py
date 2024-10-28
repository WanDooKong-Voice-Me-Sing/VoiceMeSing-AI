from pydantic import BaseModel, Field, validator
from api.model.model_schema import Model
from api.user.user_schema import User
from typing import Optional

class CoverSongCreate(BaseModel):
    model_id: int
    audio_path: str
    title: str
    user_id: int
    @validator('title')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

class CoverSong(BaseModel):
    coversong_id: int
    coversong_path: str
    title: str
    user: Optional[User]
    model_id: int


class CoverSongDelete(BaseModel):
    coversong_id: int
