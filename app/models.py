from sqlmodel import Field, Relationship, SQLModel
from pydantic import BaseModel, Field
from typing import Union
from fastapi import UploadFile
from typing import Optional

class Model(BaseModel):
    model_id: Optional[str] = Field(default=None, primary_key=True)
    user_id: str = Field()
    trained_model_path: str


class ModelRequest(BaseModel):
    user_id: str
    origin_voice: bytes = Field(..., description="Audio file of user's origin voice")

class ModelResponse(BaseModel):
    model_id: str
    trained_model_path: str

class CoverSong(BaseModel):
    id: int = Field(default=None, primary_key=True)
    model_id: str = Field(foreign_key="model.model_id", index=True)
    user_id: str
    audio_data: bytes
    title: str

class CoverSongRequest(BaseModel):
    user_id: str
    model_id: str
    audio_data: bytes

class CoverSongResponse(BaseModel):
    user_id : str
    model_id : str
    cover_song_data: bytes