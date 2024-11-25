from pydantic_settings import BaseSettings
from typing import List, Union

#This is example
class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Application"
    API_V1_STR: str = "/api/v1"
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///./test.db"
    BACKEND_CORS_ORIGINS: List[Union[str, None]] = ["*"]
    ENVIRONMENT: str = "local"


settings = Settings()
