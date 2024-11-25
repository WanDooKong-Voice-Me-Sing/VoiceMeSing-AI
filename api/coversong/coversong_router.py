from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from api.core.db import get_db
from starlette import status
from api.coversong import coversong_schema
from models import Model, CoverSong, Song_Temp
import httpx
import redis
import json
import api.Redis as red
from pydub import AudioSegment


from infer_start import coversong_train, mixing, save_cover_music

router = APIRouter(
    prefix="/api/coversong"
)

@router.post("/create", status_code=status.HTTP_200_OK)
def create_cover_song(request: coversong_schema.CoverSongCreate):
    if not request.coverSongId:
        return {"status": "error", "message": "voiceId is required."}
    
    red.r.rpush("task_queue", f"create_cover_song:{request.coverSongId}")
    
    return {"status": "success", "message": f"Task queued for voice ID: {request.coverSongId}"}


@router.post("/create", status_code=status.HTTP_200_OK)
def create_cover_song(
    request: coversong_schema.CoverSongCreate,
    backgroundTasks: BackgroundTasks,
    db: Session = Depends(get_db)):

    # user_id = request.user_id
    song_id = request.coverSongId

    if not song_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="voiceId is required.")         
    
    backgroundTasks.add_task(creation, song_id, db)

    
    return {"status": "success", "message": f"{song_id} coversong created start."}


# async def creation(song_id, db: Session = Depends(get_db), ):
#     song_id = int(song_id)
#     coversong = db.query(Song_Temp).filter(Song_Temp.song_id == song_id).first()


#     model_id = coversong.voice_model_id
#     song_name = coversong.result_song_name
#     origin_cover = coversong.cover_song_file
#     user_id = coversong.user_id

#     origin_song = save_cover_music(user_id, origin_cover)


#     model = db.query(Model).filter(
#     Model.voice_model_id == model_id,
#     Model.user_id == user_id
#     ).first()
    

#     model_name = model.voice_model_name


#     if not model:
#         raise HTTPException(status_code=404, detail="Model not found")
    
    
#     await coversong_train(sid0=f"{model_name}.pth", user_id=user_id, model_id=model_id, input_audio_path=f"{origin_song}/{user_id}_{song_name}.mp3", index_path="")
#     #output_path=""


#     result = await mixing(f"result/{user_id}/{model_id}_output_audio.wav",f"source/{user_id}/inst/instrument_{user_id}_{model_name}.mp3_10.wav",f"result/{user_id}/Cover_{model_name}.wav")


#     # Save the cover song to database
#     cover_song = CoverSong(cover_song_file=result.getvalue(), result_song_name=song_name,is_public=True, user_id=user_id) #If you need a new title
#     db.add(cover_song)
#     db.commit()
#     db.refresh(cover_song)



