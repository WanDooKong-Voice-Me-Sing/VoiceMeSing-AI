from fastapi import APIRouter, Depends, UploadFile, BackgroundTasks
from sqlalchemy.orm import Session
from starlette import status
from core.db import get_db
from api.model import model_schema, model_crud 
from models import Model,Voice_Temp
import httpx
import asyncio
import uuid
from gpu_task import gpu_task_queue, process_gpu_queue
from infer_start import voice_extraction, train


router = APIRouter(
    prefix="/api/model"
)
import subprocess
    


@router.post("/create_model", status_code=status.HTTP_202_ACCEPTED)
async def model_create(
    request: model_schema.ModelCreate,
    background_tasks: BackgroundTasks,    
    db: Session = Depends(get_db)

):
    task_id = str(uuid.uuid4())
    await gpu_task_queue.put((task_id, "model_creation", request, db))
    background_tasks.add_task(process_gpu_queue)
    return {"task_id": task_id}