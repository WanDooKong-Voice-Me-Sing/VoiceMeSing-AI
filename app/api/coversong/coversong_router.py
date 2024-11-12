from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from core.db import get_db
from starlette import status
from api.model import model_schema, model_crud
from api.coversong import coversong_schema, coversong_crud
from api.user import user_schema, user_crud
from models import Model, CoverSong
import uuid
from gpu_task import gpu_task_queue,process_gpu_queue


router = APIRouter(
    prefix="/api/coversong"
)


# cover_song_router.py 예시
@router.post("/create_cover_song", status_code=status.HTTP_202_ACCEPTED)
async def create_cover_song(
    request: coversong_schema.CoverSongCreate, 
    background_tasks: BackgroundTasks,    
    db: Session = Depends(get_db)

):
    task_id = str(uuid.uuid4())
    await gpu_task_queue.put((task_id, "cover_song_creation", request, db))
    background_tasks.add_task(process_gpu_queue)
    return {"task_id": task_id}

