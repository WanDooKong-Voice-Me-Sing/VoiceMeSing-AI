from pydantic import BaseModel, validator
from api.user.user_schema import User
from typing import Optional


class Model(BaseModel):
    model_id: int
    model_path: str
    user_id: Optional[User]


class ModelCreate(BaseModel):
    voiceId: str


class ModelResponse(BaseModel):
    status: str
    message: str
    data: str
