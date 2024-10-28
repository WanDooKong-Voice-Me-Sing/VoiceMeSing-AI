from pydantic import BaseModel, validator
from api.user.user_schema import User
from typing import Optional

class Model(BaseModel):
    model_id: int
    model_path: str
    user_id: Optional[User]



class ModelCreate(BaseModel):
    user_id: str
    voice_id: str
    voice_model_name: str
    @validator('voice_id')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v



class ModelDelete(BaseModel):
    model_id: str