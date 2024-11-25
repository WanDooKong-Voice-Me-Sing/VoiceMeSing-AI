from fastapi import APIRouter, HTTPException
from starlette import status
from api.coversong import coversong_schema
from api.worker.celery_worker import coversong_creation

router = APIRouter(
    prefix="/api/coversong"
)

@router.post("/create", status_code=status.HTTP_200_OK)
async def create_cover_song(request: coversong_schema.CoverSongCreate):

    # user_id = request.user_id
    # song_id = request.coverSongId

    if not request.coverSongId:
        return {"status": "error", "message": "coverSongId is required."}     
    
    task = coversong_creation.apply_async(args=[request.coverSongId])
    return {"status": "success", "message": f"Task queued for coversong ID: {request.coverSongId}"}
