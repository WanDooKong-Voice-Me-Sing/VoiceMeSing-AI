
from fastapi import APIRouter, Depends, UploadFile, BackgroundTasks
from sqlalchemy.orm import Session
from starlette import status

from core.db import get_db
from model import model_schema
from model.model_router import ModelResponse, ModelRequest

router = APIRouter(
    prefix="/api/model"
)

