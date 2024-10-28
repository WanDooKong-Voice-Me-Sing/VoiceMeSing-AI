from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from core.db import get_db
from starlette import status
from api.model import model_schema, model_crud
from api.coversong import coversong_schema, coversong_crud
from api.user import user_schema, user_crud
from models import Model, CoverSong
import httpx

from infer_start import coversong_train, mixing

router = APIRouter(
    prefix="/api/coversong"
)

@router.get("/")
def get_coversongs():
    return {"message": "Coversongs endpoint"}

@router.post("/create",  status_code=status.HTTP_204_NO_CONTENT) #반환할 데이터의 형식
def create_cover_song(
    request: coversong_schema.CoverSongCreate,
    db: Session = Depends(get_db)):

    # title = request.title
    # origin_song = request.audio_path
    # user_id = request.user_id
    # model_id = request.model_id

    # model = model = db.query(Model).filter(
    # Model.model_id == model_id,
    # Model.user_id == user_id
    # ).first()
    # model_path = model.model_path
    # if not model:
    #      raise HTTPException(status_code=404, detail="Model not found")
    model_name = "user_2"
    origin_song = "/app/source/song/LiMYY"
    print("커버송 제작시작")
    coversong_train(sid0=f"{model_name}.pth", input_audio_path=f"{origin_song}.mp3", index_path="")
    output_path=""

    #mixing()
    # Save the cover song to database
    # cover_song = CoverSong(model_id=model_id, user_id=user_id, audio_path=output_path, title=title) #If you need a new title
    # db.add(cover_song)
    # db.commit()
    # db.refresh(cover_song)

# 훈련 완료 후 알림을 보내는 부분
    # async with httpx.AsyncClient() as client:
    #     response = await client.post("http://backend-server-url/notify", json={"status": "completed", "cover_song_id": cover_song.coversong_id})

    #     if response.status_code != 200:
    #         print("알림 전송 실패:", response.text)
            
    # return cover_song.cover_song_data
    return 0

