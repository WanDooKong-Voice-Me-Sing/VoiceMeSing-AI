from fastapi import APIRouter, Depends, BackgroundTasks, Request, HTTPException
from sqlalchemy.orm import Session
from starlette import status
from api.core.db import get_db
from api.model import model_schema
from models import Model,Voice_Temp
import httpx
import redis
import json
import api.Redis as red

from infer_start import voice_extraction, train, save_origin_music

from sqlalchemy import inspect


router = APIRouter(
    prefix="/api/model"
)


@router.post("/create", status_code=status.HTTP_200_OK)
async def model_create(request: model_schema.ModelCreate):
    if not request.voiceId:
        return {"status": "error", "message": "voiceId is required."}
    task = model_creation.apply_async(args=[request.voice_id])   
    return {"status": "success", "message": f"Task queued for voice ID: {request.voiceId}"}

    



# @router.post("/create", status_code=status.HTTP_200_OK)
# async def model_create(request: model_schema.ModelCreate, backgroundTasks: BackgroundTasks, db: Session = Depends(get_db)):
    
#     voice_id = request.voiceId
#     if not voice_id:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="voiceId is required.")         
    
#     backgroundTasks.add_task(creation, voice_id, db)


#     return {"status": "success", "message": f"{voice_id} Model created start."}


# async def creation(voice_id,  db: Session = Depends(get_db)):

#     Voice = db.query(Voice_Temp).filter(Voice_Temp.voice_id == int(voice_id)).first()

#     model_name =  Voice.voice_model_name
#     user_id = Voice.user_id
#     origin_voice = Voice.voice_file # 유튜브 링크 받기...or 클라우드 스토리지이용....or 내꺼랑 연결해서 서버에 저장 ....or 그냥 파일로 전송 받기....
    
#     path = save_origin_music(user_id, origin_voice, model_name)

#     voice_extraction(input=f"{path}/origin",save_vocal=f"{path}/vocal",save_ins=f"{path}/inst") 
    
#     Temp = await train(exp_dir1=model_name, trainset_dir4=f"{path}/vocal") #피치 조절 필요할수도 있음!
    
    
#     model = Model(voice_model_name=model_name, voice_model_file=model_name, user_id=user_id)
#     db.add(model)
#     db.commit()
#     db.refresh(model)